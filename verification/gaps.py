"""Coverage gap analysis for CausalGuard.

Identifies uncovered hazard→harm paths and generates actionable recommendations
for closing each safety gap.
"""

from __future__ import annotations

import logging

from causalguard.data.schema import CoverageResult, FailureMode

logger = logging.getLogger("causalguard.verification.gaps")


def find_gaps(coverage_results: list[CoverageResult]) -> list[CoverageResult]:
    """Return only the uncovered paths (where coverage_gap is True).

    Parameters
    ----------
    coverage_results:
        Full list of per-path coverage results from :func:`~causalguard.verification.coverage.verify_all_paths`.

    Returns
    -------
    list[CoverageResult]
        Subset of ``coverage_results`` where ``coverage_gap`` is True,
        sorted by path length (longest/most complex gaps first).
    """
    gaps = [r for r in coverage_results if r.coverage_gap]
    # Sort by path length descending — longer unmitigated paths are higher risk
    gaps.sort(key=lambda r: len(r.path), reverse=True)
    logger.info("Found %d coverage gap(s) out of %d paths.", len(gaps), len(coverage_results))
    return gaps


def generate_recommendations(
    gaps: list[CoverageResult],
    failure_modes: list[FailureMode],
) -> list[str]:
    """Generate specific actionable recommendations for closing safety gaps.

    For each uncovered path, the recommendation identifies:
    1. The path that is unmitigated.
    2. The specific nodes (failure modes) in the path that need mitigation.
    3. A suggested mitigation type based on the node type and severity.

    Parameters
    ----------
    gaps:
        List of uncovered paths (output of :func:`find_gaps`).
    failure_modes:
        All failure modes (used to look up severity and type for each node).

    Returns
    -------
    list[str]
        Deduplicated list of recommendation strings, sorted by severity.
    """
    if not gaps:
        return ["All hazard-to-harm paths are covered. No additional mitigations required."]

    # Build a lookup by name
    fm_by_name: dict[str, FailureMode] = {fm.name: fm for fm in failure_modes}

    recommendations: list[str] = []
    seen_nodes: set[str] = set()

    for gap in gaps:
        path_str = gap.path_string

        # Find the most critical unmitigated node in this path
        most_critical_node: str | None = None
        most_critical_severity = "LOW"
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

        for node in gap.path:
            fm = fm_by_name.get(node)
            if fm:
                if severity_order.get(fm.severity, 0) > severity_order.get(most_critical_severity, 0):
                    most_critical_node = node
                    most_critical_severity = fm.severity

        # Generate a recommendation for each unmitigated node in the path
        for node in gap.path:
            if node in seen_nodes:
                continue
            seen_nodes.add(node)

            fm = fm_by_name.get(node)
            if not fm:
                continue

            severity_label = fm.severity
            node_type = fm.node_type

            if node_type == "HAZARD":
                rec = (
                    f"[{severity_label}] Add a PREVENTIVE mitigation for '{node}': "
                    f"This is a root hazard that initiates the unmitigated path: {path_str}. "
                    f"Consider input validation, pre-condition checks, or deployment constraints."
                )
            elif node_type == "HARM":
                rec = (
                    f"[{severity_label}] Add a CORRECTIVE mitigation for '{node}': "
                    f"This harm has no safety net in the path: {path_str}. "
                    f"Consider incident response procedures, fallback mechanisms, or human escalation."
                )
            else:
                rec = (
                    f"[{severity_label}] Add a DETECTIVE or PREVENTIVE mitigation for '{node}': "
                    f"This intermediate failure mode is unmitigated on the path: {path_str}. "
                    f"Consider monitoring, anomaly detection, or automated circuit-breakers."
                )

            recommendations.append(rec)

    # Deduplicate while preserving order
    seen_recs: set[str] = set()
    unique_recommendations: list[str] = []
    for rec in recommendations:
        if rec not in seen_recs:
            seen_recs.add(rec)
            unique_recommendations.append(rec)

    # Sort by severity: CRITICAL first
    def _severity_sort_key(rec: str) -> int:
        if rec.startswith("[CRITICAL]"):
            return 0
        if rec.startswith("[HIGH]"):
            return 1
        if rec.startswith("[MEDIUM]"):
            return 2
        return 3

    unique_recommendations.sort(key=_severity_sort_key)
    return unique_recommendations
