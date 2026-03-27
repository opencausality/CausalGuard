"""Tests for coverage verification logic."""

from __future__ import annotations

import networkx as nx
import pytest

from causalguard.data.schema import FailureMode, Mitigation
from causalguard.verification.coverage import verify_all_paths


def _build_linear_graph() -> tuple[nx.DiGraph, list[str], list[str]]:
    """Build a simple 3-node DAG: H → F → R (Hazard → Failure → Harm)."""
    g = nx.DiGraph()
    g.add_edge("distribution_shift", "high_confidence_wrong")
    g.add_edge("high_confidence_wrong", "harm_to_patient")
    return g, ["distribution_shift"], ["harm_to_patient"]


def _make_failure_modes() -> list[FailureMode]:
    return [
        FailureMode(
            id="H1",
            name="distribution_shift",
            description="Distribution shift",
            severity="HIGH",
            node_type="HAZARD",
        ),
        FailureMode(
            id="F1",
            name="high_confidence_wrong",
            description="Wrong confident prediction",
            severity="CRITICAL",
            node_type="FAILURE",
        ),
        FailureMode(
            id="HR1",
            name="harm_to_patient",
            description="Patient harm",
            severity="CRITICAL",
            node_type="HARM",
        ),
    ]


class TestVerifyAllPaths:
    def test_covered_path(self) -> None:
        """A mitigation blocking a node in the path marks it as covered."""
        graph, hazards, harms = _build_linear_graph()
        failure_modes = _make_failure_modes()
        mitigations = [
            Mitigation(
                id="M1",
                name="Human review",
                description="Radiologist reviews all predictions",
                blocks_failure_modes=["high_confidence_wrong"],
            )
        ]

        results = verify_all_paths(
            graph=graph,
            hazards=hazards,
            harms=harms,
            mitigations=mitigations,
            failure_modes=failure_modes,
        )

        assert len(results) == 1
        assert results[0].is_covered is True
        assert results[0].coverage_gap is False
        assert "Human review" in results[0].blocking_mitigations

    def test_uncovered_path(self) -> None:
        """No mitigations on path → coverage_gap=True."""
        graph, hazards, harms = _build_linear_graph()
        failure_modes = _make_failure_modes()
        mitigations: list[Mitigation] = []

        results = verify_all_paths(
            graph=graph,
            hazards=hazards,
            harms=harms,
            mitigations=mitigations,
            failure_modes=failure_modes,
        )

        assert len(results) == 1
        assert results[0].is_covered is False
        assert results[0].coverage_gap is True
        assert results[0].blocking_mitigations == []

    def test_coverage_percentage_calculation(self) -> None:
        """Two paths, one covered → 50% coverage."""
        # Add a second path: H → F2 → R
        graph, hazards, harms = _build_linear_graph()
        graph.add_edge("image_artifact", "harm_to_patient")
        hazards.append("image_artifact")
        failure_modes = _make_failure_modes() + [
            FailureMode(
                id="H2",
                name="image_artifact",
                description="Image artifact",
                severity="HIGH",
                node_type="HAZARD",
            )
        ]
        mitigations = [
            Mitigation(
                id="M1",
                name="Human review",
                description="Review",
                blocks_failure_modes=["high_confidence_wrong"],
            )
        ]

        results = verify_all_paths(
            graph=graph,
            hazards=hazards,
            harms=harms,
            mitigations=mitigations,
            failure_modes=failure_modes,
        )

        covered = sum(1 for r in results if r.is_covered)
        assert covered == 1
        assert len(results) == 2

    def test_path_string_format(self) -> None:
        """path_string uses ' → ' as separator."""
        graph, hazards, harms = _build_linear_graph()
        failure_modes = _make_failure_modes()
        mitigations: list[Mitigation] = []

        results = verify_all_paths(
            graph=graph,
            hazards=hazards,
            harms=harms,
            mitigations=mitigations,
            failure_modes=failure_modes,
        )

        assert " → " in results[0].path_string
