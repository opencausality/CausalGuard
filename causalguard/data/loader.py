"""Data loading utilities for CausalGuard.

Provides helpers to load raw text inputs and serialised SafetyCase JSON files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from causalguard.data.schema import SafetyCase
from causalguard.exceptions import CausalGuardError

logger = logging.getLogger("causalguard.data.loader")


def load_text(path: Path) -> str:
    """Read a text file and return its contents as a string.

    Parameters
    ----------
    path:
        Path to the text file. Accepts any plain-text format (UTF-8).

    Returns
    -------
    str
        File contents.

    Raises
    ------
    CausalGuardError
        If the file cannot be read or does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise CausalGuardError(f"File not found: {path}")
    if not path.is_file():
        raise CausalGuardError(f"Path is not a file: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CausalGuardError(f"Cannot read file {path}: {exc}") from exc

    if not content.strip():
        logger.warning("File %s is empty or contains only whitespace.", path)

    logger.debug("Loaded %d characters from %s", len(content), path)
    return content


def load_safety_case(path: Path) -> SafetyCase:
    """Load and validate a SafetyCase from a JSON file.

    Parameters
    ----------
    path:
        Path to a JSON file previously saved by :func:`causalguard.cases.exporter.save_safety_case`.

    Returns
    -------
    SafetyCase
        The deserialised and validated safety case.

    Raises
    ------
    CausalGuardError
        If the file cannot be read, is not valid JSON, or fails schema validation.
    """
    path = Path(path)
    raw = load_text(path)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CausalGuardError(f"File {path} is not valid JSON: {exc}") from exc

    try:
        case = SafetyCase.model_validate(data)
    except Exception as exc:
        raise CausalGuardError(
            f"File {path} does not conform to the SafetyCase schema: {exc}"
        ) from exc

    logger.debug(
        "Loaded safety case for system=%r with %d failure modes, %d mitigations.",
        case.system_name,
        len(case.failure_modes),
        len(case.mitigations),
    )
    return case
