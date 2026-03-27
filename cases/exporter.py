"""Safety case export and import utilities.

Handles serialisation of :class:`~causalguard.data.schema.SafetyCase` to and
from JSON, and produces the human-readable text report shown by ``causalguard
verify``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from causalguard.data.schema import SafetyCase
from causalguard.exceptions import CausalGuardError

logger = logging.getLogger("causalguard.cases.exporter")

_DIVIDER = "══════════════════════════════════"


def save_safety_case(case: SafetyCase, path: Path) -> None:
    """Serialise a :class:`SafetyCase` to JSON.

    Parameters
    ----------
    case:
        The safety case to save.
    path:
        Destination file path.  Parent directories are created if needed.

    Raises
    ------
    CausalGuardError
        If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(case.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Safety case saved to %s", path)
    except OSError as exc:
        raise CausalGuardError(f"Failed to save safety case to {path}: {exc}") from exc


def load_safety_case(path: Path) -> SafetyCase:
    """Load a :class:`SafetyCase` from a JSON file.

    Parameters
    ----------
    path:
        Path to a previously saved SafetyCase JSON.

    Returns
    -------
    SafetyCase

    Raises
    ------
    CausalGuardError
        If the file is missing or contains invalid JSON.
    """
    if not path.exists():
        raise CausalGuardError(f"Safety case file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SafetyCase.model_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise CausalGuardError(f"Failed to parse safety case from {path}: {exc}") from exc


def format_text_report(case: SafetyCase) -> str:
    """Produce a human-readable safety case report.

    Matches the format shown in the README::

        CausalGuard — Safety Case Report
        ══════════════════════════════════

        System: Pneumonia Detection AI v2.1
        Failure modes extracted: 6
        Mitigations mapped: 4
        Paths analyzed: 8

        Coverage Results:

        Path 1: distribution_shift → high_confidence_wrong → harm_to_patient
          ✅ COVERED by: "Human radiologist review required for all recommendations"

        ...

        ══════════════════════════════════
        Safety verdict: PARTIAL (6/8 paths covered = 75%)

    Parameters
    ----------
    case:
        The safety case to format.

    Returns
    -------
    str
        Multi-line formatted report.
    """
    lines: list[str] = [
        "CausalGuard — Safety Case Report",
        _DIVIDER,
        "",
        f"System: {case.system_name}",
        f"Failure modes extracted: {len(case.failure_modes)}",
        f"Mitigations mapped: {len(case.mitigations)}",
        f"Paths analyzed: {len(case.coverage_results)}",
        "",
        "Coverage Results:",
        "",
    ]

    total = len(case.coverage_results)
    covered_count = sum(1 for r in case.coverage_results if r.is_covered)

    for i, result in enumerate(case.coverage_results, 1):
        lines.append(f"Path {i}: {result.path_string}")
        if result.is_covered:
            mits = ", ".join(f'"{m}"' for m in result.blocking_mitigations)
            lines.append(f"  ✅ COVERED by: {mits}")
        else:
            lines.append("  ❌ UNCOVERED — no mitigation blocks this path")
            if result.gap_at_edge:
                lines.append(f"  Gap at: {result.gap_at_edge}")
        lines.append("")

    lines.append(_DIVIDER)

    verdict_colour = {"SAFE": "✅", "PARTIAL": "⚠️", "UNSAFE": "❌"}.get(
        case.safety_verdict, "?"
    )
    lines.append(
        f"Safety verdict: {case.safety_verdict} "
        f"({covered_count}/{total} paths covered = {case.coverage_percentage:.0f}%)"
    )
    lines.append("Target: 100% coverage")

    if case.uncovered_paths:
        lines.append("")
        lines.append(f"Critical gaps: {len(case.uncovered_paths)} uncovered path(s) to harm")
        lines.append("Recommendations:")
        for j, rec in enumerate(case.recommendations, 1):
            lines.append(f"  {j}. {rec}")

    return "\n".join(lines)
