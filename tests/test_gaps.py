"""Tests for gap detection and safety verdict logic."""

from __future__ import annotations

import pytest

from causalguard.data.schema import (
    CoverageResult,
    FailureMode,
    Mitigation,
    SafetyCase,
    SafetyEdge,
)
from causalguard.verification.gaps import find_gaps, generate_recommendations
from causalguard.verification.strength import (
    compute_defense_in_depth_score,
    compute_mitigation_depth,
)


def _make_case(covered: int, total: int) -> SafetyCase:
    """Build a SafetyCase with *covered* covered paths and *total* total paths."""
    results = []
    for i in range(total):
        is_covered = i < covered
        results.append(
            CoverageResult(
                path=[f"h{i}", f"harm{i}"],
                path_string=f"h{i} → harm{i}",
                is_covered=is_covered,
                blocking_mitigations=["M1"] if is_covered else [],
                coverage_gap=not is_covered,
            )
        )

    uncovered = [r for r in results if not r.is_covered]
    pct = (covered / total * 100.0) if total > 0 else 0.0
    verdict = "SAFE" if pct >= 100 else ("PARTIAL" if pct >= 50 else "UNSAFE")

    return SafetyCase(
        system_name="Test",
        failure_modes=[],
        mitigations=[],
        safety_edges=[],
        coverage_results=results,
        hazard_nodes=[f"h{i}" for i in range(total)],
        harm_nodes=[f"harm{i}" for i in range(total)],
        uncovered_paths=uncovered,
        safety_verdict=verdict,
        coverage_percentage=round(pct, 2),
        recommendations=[],
        created_at="2026-03-24T00:00:00+00:00",
    )


class TestFindGaps:
    def test_no_gaps_when_all_covered(self) -> None:
        """find_gaps returns empty list when all paths are covered."""
        case = _make_case(covered=3, total=3)
        gaps = find_gaps(case.coverage_results)
        assert gaps == []

    def test_finds_uncovered_paths(self) -> None:
        """find_gaps returns only uncovered paths."""
        case = _make_case(covered=2, total=4)
        gaps = find_gaps(case.coverage_results)
        assert len(gaps) == 2
        assert all(g.coverage_gap for g in gaps)


class TestGenerateRecommendations:
    def test_recommendations_for_each_gap(self) -> None:
        """generate_recommendations produces at least one recommendation per gap."""
        case = _make_case(covered=0, total=2)
        failure_modes: list[FailureMode] = []
        recs = generate_recommendations(case.uncovered_paths, failure_modes)
        assert len(recs) >= len(case.uncovered_paths)

    def test_no_recs_when_no_gaps(self) -> None:
        case = _make_case(covered=2, total=2)
        recs = generate_recommendations([], [])
        assert recs == []


class TestSafetyVerdicts:
    def test_safe_verdict(self) -> None:
        case = _make_case(covered=4, total=4)
        assert case.safety_verdict == "SAFE"
        assert case.coverage_percentage == 100.0

    def test_partial_verdict(self) -> None:
        case = _make_case(covered=3, total=4)
        assert case.safety_verdict == "PARTIAL"

    def test_unsafe_verdict(self) -> None:
        case = _make_case(covered=1, total=4)
        assert case.safety_verdict == "UNSAFE"


class TestDefenseInDepth:
    def test_depth_mapping(self) -> None:
        """compute_mitigation_depth returns correct counts per path."""
        result_a = CoverageResult(
            path=["h", "f", "harm"],
            path_string="h → f → harm",
            is_covered=True,
            blocking_mitigations=["M1", "M2"],
            coverage_gap=False,
        )
        result_b = CoverageResult(
            path=["h2", "harm"],
            path_string="h2 → harm",
            is_covered=True,
            blocking_mitigations=["M1"],
            coverage_gap=False,
        )
        case = SafetyCase(
            system_name="T",
            failure_modes=[],
            mitigations=[],
            safety_edges=[],
            coverage_results=[result_a, result_b],
            hazard_nodes=["h", "h2"],
            harm_nodes=["harm"],
            uncovered_paths=[],
            safety_verdict="SAFE",
            coverage_percentage=100.0,
            recommendations=[],
            created_at="2026-03-24T00:00:00+00:00",
        )

        depth = compute_mitigation_depth(case)
        assert depth["h → f → harm"] == 2
        assert depth["h2 → harm"] == 1

    def test_defense_in_depth_score(self) -> None:
        """Score is 0.5 when half of covered paths have ≥2 mitigations."""
        result_a = CoverageResult(
            path=["h", "harm"],
            path_string="h → harm",
            is_covered=True,
            blocking_mitigations=["M1", "M2"],
            coverage_gap=False,
        )
        result_b = CoverageResult(
            path=["h2", "harm"],
            path_string="h2 → harm",
            is_covered=True,
            blocking_mitigations=["M1"],
            coverage_gap=False,
        )
        case = SafetyCase(
            system_name="T",
            failure_modes=[],
            mitigations=[],
            safety_edges=[],
            coverage_results=[result_a, result_b],
            hazard_nodes=["h", "h2"],
            harm_nodes=["harm"],
            uncovered_paths=[],
            safety_verdict="SAFE",
            coverage_percentage=100.0,
            recommendations=[],
            created_at="2026-03-24T00:00:00+00:00",
        )

        score = compute_defense_in_depth_score(case)
        assert score == 0.5
