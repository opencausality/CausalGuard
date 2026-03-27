"""Mitigation extractor.

Uses the LLM to extract and map mitigations to the failure modes they address.
"""

from __future__ import annotations

import json
import logging

from causalguard.config import Settings, get_settings
from causalguard.data.schema import FailureMode, Mitigation
from causalguard.exceptions import ExtractionError
from causalguard.llm.adapter import LLMAdapter
from causalguard.llm.parsers import parse_mitigations
from causalguard.llm.prompts import (
    MITIGATION_EXTRACTION_PROMPT,
    STRICT_JSON_RETRY_SUFFIX,
    SYSTEM_PROMPT,
)

logger = logging.getLogger("causalguard.extraction.mitigations")


class MitigationExtractor:
    """Extract and map mitigations to failure modes using the LLM.

    Parameters
    ----------
    adapter:
        LLM adapter to use. Creates a new one from settings if omitted.
    settings:
        Application settings. Uses the global singleton if omitted.
    """

    def __init__(
        self,
        adapter: LLMAdapter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._adapter = adapter or LLMAdapter(settings=self._settings)

    def extract(
        self,
        failure_modes: list[FailureMode],
        mitigations_text: str,
    ) -> list[Mitigation]:
        """Use the LLM to extract mitigations and map them to failure modes.

        Parameters
        ----------
        failure_modes:
            List of already-extracted failure modes.  These are serialised to
            JSON and included in the prompt so the LLM can use their names.
        mitigations_text:
            Plain-text descriptions of the mitigations (one per paragraph, or
            a numbered list).

        Returns
        -------
        list[Mitigation]
            Extracted and validated mitigations with failure-mode mappings.

        Raises
        ------
        ExtractionError
            If both LLM calls fail to produce parseable output, or if
            ``mitigations_text`` is empty.
        """
        if not mitigations_text.strip():
            raise ExtractionError("mitigations_text cannot be empty.")
        if not failure_modes:
            raise ExtractionError(
                "failure_modes list is empty — extract failure modes before mitigations."
            )

        # Serialise failure modes for the prompt (name + description only for brevity)
        failure_modes_json = json.dumps(
            [
                {
                    "id": fm.id,
                    "name": fm.name,
                    "description": fm.description,
                    "node_type": fm.node_type,
                    "severity": fm.severity,
                }
                for fm in failure_modes
            ],
            indent=2,
        )

        prompt = MITIGATION_EXTRACTION_PROMPT.format(
            failure_modes_json=failure_modes_json,
            mitigations_text=mitigations_text.strip(),
        )

        logger.info(
            "Extracting mitigations from %d-char description for %d failure modes.",
            len(mitigations_text),
            len(failure_modes),
        )

        # Primary attempt
        raw = self._adapter.complete(prompt, system=SYSTEM_PROMPT)
        try:
            mitigations = parse_mitigations(raw)
            # Post-process: ensure all referenced failure mode names are valid
            valid_names = {fm.name for fm in failure_modes}
            mitigations = self._filter_invalid_references(mitigations, valid_names)
            logger.info("Extracted %d mitigations on primary attempt.", len(mitigations))
            return mitigations
        except ExtractionError as primary_error:
            logger.warning(
                "Primary mitigation parse failed: %s. Retrying with stricter prompt.",
                primary_error,
            )

        # Retry with stricter JSON prompt
        strict_prompt = prompt + STRICT_JSON_RETRY_SUFFIX
        raw_retry = self._adapter.complete(strict_prompt, system=SYSTEM_PROMPT)
        try:
            mitigations = parse_mitigations(raw_retry)
            valid_names = {fm.name for fm in failure_modes}
            mitigations = self._filter_invalid_references(mitigations, valid_names)
            logger.info("Extracted %d mitigations on retry attempt.", len(mitigations))
            return mitigations
        except ExtractionError as retry_error:
            raise ExtractionError(
                f"Failed to extract mitigations after 2 attempts. "
                f"Primary error: {primary_error}. "
                f"Retry error: {retry_error}."
            ) from retry_error

    @staticmethod
    def _filter_invalid_references(
        mitigations: list[Mitigation],
        valid_names: set[str],
    ) -> list[Mitigation]:
        """Remove invalid failure mode references from mitigation mappings.

        If the LLM hallucinates a failure mode name that does not exist in the
        extracted set, we silently drop that reference to avoid downstream errors.
        Valid references are preserved.  If a mitigation loses ALL its references
        we keep it but log a warning.
        """
        cleaned: list[Mitigation] = []
        for m in mitigations:
            original_refs = m.blocks_failure_modes
            valid_refs = [name for name in original_refs if name in valid_names]
            invalid_refs = [name for name in original_refs if name not in valid_names]
            if invalid_refs:
                logger.warning(
                    "Mitigation '%s' references unknown failure modes (removed): %s",
                    m.name,
                    invalid_refs,
                )
            # Rebuild with filtered references
            cleaned.append(
                m.model_copy(update={"blocks_failure_modes": valid_refs})
            )
        return cleaned
