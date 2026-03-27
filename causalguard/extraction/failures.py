"""Failure mode extractor.

Uses the LLM to extract structured failure modes from a system description
and incident logs.  Implements retry logic with a stricter prompt on parse
failure.
"""

from __future__ import annotations

import logging

from causalguard.config import Settings, get_settings
from causalguard.data.schema import FailureMode
from causalguard.exceptions import ExtractionError
from causalguard.llm.adapter import LLMAdapter
from causalguard.llm.parsers import parse_failure_modes
from causalguard.llm.prompts import (
    FAILURE_EXTRACTION_PROMPT,
    STRICT_JSON_RETRY_SUFFIX,
    SYSTEM_PROMPT,
)

logger = logging.getLogger("causalguard.extraction.failures")


class FailureExtractor:
    """Extract structured failure modes from system description and incident logs.

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
        system_description: str,
        incident_logs: str,
    ) -> list[FailureMode]:
        """Use the LLM to extract structured failure modes.

        Makes up to two LLM calls:
        1. Primary call with the standard extraction prompt.
        2. If parsing fails, retries with a stricter JSON-only suffix appended.

        Parameters
        ----------
        system_description:
            Text describing the AI system (architecture, deployment context,
            inputs, outputs, thresholds).
        incident_logs:
            Text describing observed incidents or potential failure scenarios.

        Returns
        -------
        list[FailureMode]
            Extracted and validated failure modes.

        Raises
        ------
        ExtractionError
            If both LLM calls fail to produce parseable output.
        """
        if not system_description.strip():
            raise ExtractionError("system_description cannot be empty.")
        if not incident_logs.strip():
            raise ExtractionError("incident_logs cannot be empty.")

        prompt = FAILURE_EXTRACTION_PROMPT.format(
            system_description=system_description.strip(),
            incident_logs=incident_logs.strip(),
        )

        logger.info("Extracting failure modes from system description (%d chars) and incident logs (%d chars).",
                    len(system_description), len(incident_logs))

        # Primary attempt
        raw = self._adapter.complete(prompt, system=SYSTEM_PROMPT)
        try:
            failure_modes = parse_failure_modes(raw)
            logger.info("Extracted %d failure modes on primary attempt.", len(failure_modes))
            return failure_modes
        except ExtractionError as primary_error:
            logger.warning(
                "Primary extraction parse failed: %s. Retrying with stricter prompt.",
                primary_error,
            )

        # Retry with stricter JSON prompt
        strict_prompt = prompt + STRICT_JSON_RETRY_SUFFIX
        raw_retry = self._adapter.complete(strict_prompt, system=SYSTEM_PROMPT)
        try:
            failure_modes = parse_failure_modes(raw_retry)
            logger.info(
                "Extracted %d failure modes on retry attempt.", len(failure_modes)
            )
            return failure_modes
        except ExtractionError as retry_error:
            raise ExtractionError(
                f"Failed to extract failure modes after 2 attempts. "
                f"Primary error: {primary_error}. "
                f"Retry error: {retry_error}."
            ) from retry_error
