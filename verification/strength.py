"""Mitigation strength and defense-in-depth analysis.

Computes per-path mitigation depth and an aggregate defense-in-depth score
indicating how many paths benefit from multiple independent mitigations.
"""

from __future__ import annotations

import logging

from causalguard.data.schema import SafetyCase

logger = logging.getLogger("causalguard.verification.strength")


def compute_mitigation_depth(case: SafetyCase) -> dict[str, int]:
    """Return the number of mitigations covering each path.

    A higher depth indicates defense-in-depth — the path is protected by
    multiple independent controls.

    Parameters
    ----------
    case:
        A populated :class:`~causalguard.data.schema.SafetyCase`.

    Returns
    -------
    dict[str, int]
        Mapping of ``path_string`` → number of blocking mitigations on that path.
        Uncovered paths map to ``0``.
    """
    depth: dict[str, int] = {}
    for result in case.coverage_results:
        depth[result.path_string] = len(result.blocking_mitigations)
    return depth


def compute_defense_in_depth_score(case: SafetyCase) -> float:
    """Fraction of covered paths that have two or more independent mitigations.

    A score of 1.0 means every covered path has defense-in-depth.
    A score of 0.0 means every covered path relies on exactly one mitigation.

    Parameters
    ----------
    case:
        A populated :class:`~causalguard.data.schema.SafetyCase`.

    Returns
    -------
    float
        Value in [0.0, 1.0].  Returns 0.0 if no paths are covered.
    """
    covered = [r for r in case.coverage_results if r.is_covered]
    if not covered:
        logger.debug("No covered paths — defense-in-depth score is 0.0")
        return 0.0

    multi_covered = sum(1 for r in covered if len(r.blocking_mitigations) >= 2)
    score = multi_covered / len(covered)

    logger.debug(
        "Defense-in-depth: %d/%d covered paths have ≥2 mitigations (score=%.2f)",
        multi_covered,
        len(covered),
        score,
    )
    return round(score, 4)


def strongest_path(case: SafetyCase) -> str | None:
    """Return the path_string of the most heavily mitigated path.

    Parameters
    ----------
    case:
        A populated :class:`~causalguard.data.schema.SafetyCase`.

    Returns
    -------
    str | None
        The path string with the most blocking mitigations, or ``None`` if there
        are no covered paths.
    """
    covered = [r for r in case.coverage_results if r.is_covered]
    if not covered:
        return None
    return max(covered, key=lambda r: len(r.blocking_mitigations)).path_string
