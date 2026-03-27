"""Tests for LLM-based failure mode and mitigation extraction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from causalguard.data.schema import FailureMode, Mitigation


class TestFailureModeExtraction:
    """Tests for FailureModeExtractor with a mocked LLM."""

    def _make_mock_llm(self, response_json: str) -> MagicMock:
        mock = MagicMock()
        mock.complete.return_value = response_json
        return mock

    def test_extraction_returns_failure_modes(self) -> None:
        """Extractor returns a list of FailureMode objects from valid LLM JSON."""
        from causalguard.config import Settings
        from causalguard.extraction.failures import FailureModeExtractor

        payload = {
            "failure_modes": [
                {
                    "id": "H1",
                    "name": "distribution_shift",
                    "description": "Training-serving skew",
                    "severity": "HIGH",
                    "node_type": "HAZARD",
                    "trigger_conditions": ["New hospital"],
                    "downstream_effects": ["wrong_prediction"],
                },
            ],
            "edges": [],
        }
        mock_llm = self._make_mock_llm(json.dumps(payload))
        settings = Settings()

        extractor = FailureModeExtractor(llm=mock_llm, settings=settings)
        result = extractor.extract(
            system_description="A medical AI system",
            incident_logs="Incident: wrong classification",
        )

        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(fm, FailureMode) for fm in result)

    def test_extraction_handles_empty_response(self) -> None:
        """Extractor falls back gracefully when LLM returns no failure modes."""
        from causalguard.config import Settings
        from causalguard.extraction.failures import FailureModeExtractor

        mock_llm = self._make_mock_llm('{"failure_modes": [], "edges": []}')
        settings = Settings()

        extractor = FailureModeExtractor(llm=mock_llm, settings=settings)
        result = extractor.extract(
            system_description="Simple system",
            incident_logs="No incidents",
        )

        assert isinstance(result, list)


class TestMitigationExtraction:
    """Tests for MitigationExtractor with a mocked LLM."""

    def _make_failure_modes(self) -> list[FailureMode]:
        return [
            FailureMode(
                id="F1",
                name="wrong_prediction",
                description="Model predicts incorrectly",
                severity="HIGH",
                node_type="FAILURE",
            )
        ]

    def _make_mock_llm(self, response_json: str) -> MagicMock:
        mock = MagicMock()
        mock.complete.return_value = response_json
        return mock

    def test_extraction_returns_mitigations(self) -> None:
        """Extractor returns Mitigation objects with blocks_failure_modes set."""
        from causalguard.config import Settings
        from causalguard.extraction.mitigations import MitigationExtractor

        payload = {
            "mitigations": [
                {
                    "id": "M1",
                    "name": "Human review",
                    "description": "Radiologist reviews all predictions",
                    "blocks_failure_modes": ["wrong_prediction"],
                    "coverage_confidence": 0.9,
                    "mitigation_type": "DETECTIVE",
                }
            ]
        }
        mock_llm = self._make_mock_llm(json.dumps(payload))
        settings = Settings()
        failure_modes = self._make_failure_modes()

        extractor = MitigationExtractor(llm=mock_llm, settings=settings)
        result = extractor.extract(
            mitigations_text="Human radiologist reviews all AI recommendations.",
            failure_modes=failure_modes,
        )

        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(m, Mitigation) for m in result)
