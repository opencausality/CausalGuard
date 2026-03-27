"""Output parsers for CausalGuard LLM responses.

Parsing strategy:
1. Attempt ``json.loads()`` on the raw LLM output.
2. If that fails, strip markdown fences and retry.
3. Validate the parsed dict against the Pydantic schema.
4. If validation fails, raise ``ExtractionError`` (caller may retry with stricter prompt).

All raw LLM output is logged at DEBUG level for troubleshooting.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from causalguard.data.schema import FailureMode, Mitigation
from causalguard.exceptions import ExtractionError

logger = logging.getLogger("causalguard.llm.parsers")

# ── JSON extraction helpers ───────────────────────────────────────────────────


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences and leading/trailing prose from LLM output.

    Handles patterns like:
    - ```json\\n{...}\\n```
    - ```\\n{...}\\n```
    - Text before/after the JSON object
    """
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

    # Try to find the outermost JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)

    return text.strip()


def _parse_json_safe(raw: str) -> dict[str, Any]:
    """Attempt to parse raw LLM output as JSON, with fallback stripping.

    Parameters
    ----------
    raw:
        Raw text from the LLM.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ExtractionError
        If parsing fails even after stripping.
    """
    logger.debug("Raw LLM output (first 500 chars): %s", raw[:500])

    # First attempt: parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Second attempt: strip markdown fences and extract JSON object
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Could not parse LLM output as JSON even after stripping markdown. "
            f"JSON error: {exc}. "
            f"Raw output (first 300 chars): {raw[:300]!r}"
        ) from exc


# ── Failure mode parser ───────────────────────────────────────────────────────


def parse_failure_modes(output: str) -> list[FailureMode]:
    """Parse LLM output into a list of :class:`FailureMode` objects.

    Parameters
    ----------
    output:
        Raw LLM response expected to contain a JSON object with a
        ``failure_modes`` list.

    Returns
    -------
    list[FailureMode]
        Validated failure modes.

    Raises
    ------
    ExtractionError
        If the output cannot be parsed or fails schema validation.
    """
    data = _parse_json_safe(output)

    if "failure_modes" not in data:
        raise ExtractionError(
            f"LLM response missing 'failure_modes' key. "
            f"Got keys: {list(data.keys())}. "
            f"Raw output (first 300 chars): {output[:300]!r}"
        )

    raw_modes = data["failure_modes"]
    if not isinstance(raw_modes, list):
        raise ExtractionError(
            f"'failure_modes' must be a list, got {type(raw_modes).__name__}."
        )

    failure_modes: list[FailureMode] = []
    errors: list[str] = []

    for i, item in enumerate(raw_modes):
        try:
            fm = FailureMode.model_validate(item)
            failure_modes.append(fm)
        except Exception as exc:
            errors.append(f"Item {i} ({item.get('id', '?')}): {exc}")

    if errors:
        logger.warning(
            "Some failure modes failed validation and were skipped: %s",
            "; ".join(errors),
        )

    if not failure_modes:
        raise ExtractionError(
            f"No valid failure modes could be parsed from LLM output. "
            f"Errors: {errors}. Raw output (first 500 chars): {output[:500]!r}"
        )

    logger.debug("Parsed %d failure modes successfully.", len(failure_modes))
    return failure_modes


# ── Mitigation parser ─────────────────────────────────────────────────────────


def parse_mitigations(output: str) -> list[Mitigation]:
    """Parse LLM output into a list of :class:`Mitigation` objects.

    Parameters
    ----------
    output:
        Raw LLM response expected to contain a JSON object with a
        ``mitigations`` list.

    Returns
    -------
    list[Mitigation]
        Validated mitigations.

    Raises
    ------
    ExtractionError
        If the output cannot be parsed or fails schema validation.
    """
    data = _parse_json_safe(output)

    if "mitigations" not in data:
        raise ExtractionError(
            f"LLM response missing 'mitigations' key. "
            f"Got keys: {list(data.keys())}. "
            f"Raw output (first 300 chars): {output[:300]!r}"
        )

    raw_mitigations = data["mitigations"]
    if not isinstance(raw_mitigations, list):
        raise ExtractionError(
            f"'mitigations' must be a list, got {type(raw_mitigations).__name__}."
        )

    mitigations: list[Mitigation] = []
    errors: list[str] = []

    for i, item in enumerate(raw_mitigations):
        try:
            m = Mitigation.model_validate(item)
            mitigations.append(m)
        except Exception as exc:
            errors.append(f"Item {i} ({item.get('id', '?')}): {exc}")

    if errors:
        logger.warning(
            "Some mitigations failed validation and were skipped: %s",
            "; ".join(errors),
        )

    if not mitigations:
        raise ExtractionError(
            f"No valid mitigations could be parsed from LLM output. "
            f"Errors: {errors}. Raw output (first 500 chars): {output[:500]!r}"
        )

    logger.debug("Parsed %d mitigations successfully.", len(mitigations))
    return mitigations
