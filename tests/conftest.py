"""Shared pytest fixtures for CausalGuard tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from causalguard.data.schema import (
    CoverageResult,
    FailureMode,
    Mitigation,
    SafetyCase,
    SafetyEdge,
)


@pytest.fixture
def sample_failure_modes() -> list[FailureMode]:
    """A minimal set of failure modes: 1 hazard, 1 failure, 1 harm."""
    return [
        FailureMode(
            id="H1",
            name="distribution_shift",
            description="Model encounters data from a different distribution than training.",
            severity="HIGH",
            node_type="HAZARD",
            trigger_conditions=["New hospital site", "Different scanner model"],
            downstream_effects=["high_confidence_wrong"],
        ),
        FailureMode(
            id="F1",
            name="high_confidence_wrong",
            description="Model outputs a confident but incorrect prediction.",
            severity="CRITICAL",
            node_type="FAILURE",
            trigger_conditions=["distribution_shift"],
            downstream_effects=["harm_to_patient"],
        ),
        FailureMode(
            id="HR1",
            name="harm_to_patient",
            description="Patient receives incorrect diagnosis, leading to harm.",
            severity="CRITICAL",
            node_type="HARM",
            trigger_conditions=["high_confidence_wrong"],
            downstream_effects=[],
        ),
    ]


@pytest.fixture
def sample_mitigations() -> list[Mitigation]:
    """A mitigation that covers the FAILURE node in the test graph."""
    return [
        Mitigation(
            id="M1",
            name="Human radiologist review",
            description="All AI recommendations reviewed by a qualified radiologist.",
            blocks_failure_modes=["high_confidence_wrong"],
            coverage_confidence=0.9,
            mitigation_type="DETECTIVE",
        ),
    ]


@pytest.fixture
def uncovered_mitigations() -> list[Mitigation]:
    """A mitigation that does NOT cover the test failure path."""
    return [
        Mitigation(
            id="M2",
            name="Unrelated audit",
            description="Quarterly software audit unrelated to inference correctness.",
            blocks_failure_modes=["software_bug"],
            coverage_confidence=0.5,
            mitigation_type="DETECTIVE",
        ),
    ]


@pytest.fixture
def sample_safety_case(
    sample_failure_modes: list[FailureMode],
    sample_mitigations: list[Mitigation],
) -> SafetyCase:
    """A SafetyCase with one covered path."""
    covered_result = CoverageResult(
        path=["distribution_shift", "high_confidence_wrong", "harm_to_patient"],
        path_string="distribution_shift → high_confidence_wrong → harm_to_patient",
        is_covered=True,
        blocking_mitigations=["Human radiologist review"],
        coverage_gap=False,
    )
    return SafetyCase(
        system_name="Test AI System",
        failure_modes=sample_failure_modes,
        mitigations=sample_mitigations,
        safety_edges=[
            SafetyEdge(
                cause="distribution_shift",
                effect="high_confidence_wrong",
                mechanism="Distribution shift causes incorrect predictions",
                severity="HIGH",
            ),
            SafetyEdge(
                cause="high_confidence_wrong",
                effect="harm_to_patient",
                mechanism="Wrong prediction leads to harmful treatment decisions",
                severity="CRITICAL",
            ),
        ],
        coverage_results=[covered_result],
        hazard_nodes=["distribution_shift"],
        harm_nodes=["harm_to_patient"],
        uncovered_paths=[],
        safety_verdict="SAFE",
        coverage_percentage=100.0,
        recommendations=[],
        created_at="2026-03-24T00:00:00+00:00",
        model_used="ollama/llama3.1",
    )


@pytest.fixture
def mock_llm_adapter() -> MagicMock:
    """A mock LLM adapter that returns deterministic JSON responses."""
    mock = MagicMock()
    mock.complete.return_value = (
        '{"failure_modes": [], "edges": []}'
    )
    return mock
