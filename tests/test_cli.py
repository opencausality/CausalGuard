"""CLI tests for CausalGuard using typer's test runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from causalguard.cli import app
from causalguard.data.schema import (
    CoverageResult,
    SafetyCase,
    SafetyEdge,
)

runner = CliRunner()


def _write_safety_case(tmp_path: Path, case: SafetyCase) -> Path:
    p = tmp_path / "safety_case.json"
    p.write_text(case.model_dump_json(indent=2), encoding="utf-8")
    return p


def _make_simple_case() -> SafetyCase:
    result = CoverageResult(
        path=["distribution_shift", "harm_to_patient"],
        path_string="distribution_shift → harm_to_patient",
        is_covered=True,
        blocking_mitigations=["Human review"],
        coverage_gap=False,
    )
    return SafetyCase(
        system_name="Test AI",
        failure_modes=[],
        mitigations=[],
        safety_edges=[
            SafetyEdge(
                cause="distribution_shift",
                effect="harm_to_patient",
                mechanism="Direct",
                severity="HIGH",
            )
        ],
        coverage_results=[result],
        hazard_nodes=["distribution_shift"],
        harm_nodes=["harm_to_patient"],
        uncovered_paths=[],
        safety_verdict="SAFE",
        coverage_percentage=100.0,
        recommendations=[],
        created_at="2026-03-24T00:00:00+00:00",
    )


class TestVerifyCommand:
    def test_verify_shows_report(self, tmp_path: Path) -> None:
        case = _make_simple_case()
        case_path = _write_safety_case(tmp_path, case)

        result = runner.invoke(app, ["verify", "--case", str(case_path)])

        assert result.exit_code == 0
        assert "Safety Case Report" in result.output or "Safety verdict" in result.output

    def test_verify_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        result = runner.invoke(app, ["verify", "--case", str(missing)])
        assert result.exit_code != 0


class TestGapsCommand:
    def test_gaps_no_gaps_message(self, tmp_path: Path) -> None:
        """When all paths are covered, gaps command reports no gaps."""
        case = _make_simple_case()
        case_path = _write_safety_case(tmp_path, case)

        result = runner.invoke(app, ["gaps", "--case", str(case_path)])

        assert result.exit_code == 0
        assert "No coverage gaps" in result.output or "covered" in result.output.lower()

    def test_gaps_shows_uncovered_paths(self, tmp_path: Path) -> None:
        """When gaps exist, gaps command lists them."""
        uncovered = CoverageResult(
            path=["ood_input", "harm"],
            path_string="ood_input → harm",
            is_covered=False,
            blocking_mitigations=[],
            coverage_gap=True,
        )
        case = _make_simple_case()
        case.coverage_results.append(uncovered)
        case.uncovered_paths.append(uncovered)
        object.__setattr__(case, "safety_verdict", "PARTIAL")
        object.__setattr__(case, "coverage_percentage", 50.0)
        case_path = _write_safety_case(tmp_path, case)

        result = runner.invoke(app, ["gaps", "--case", str(case_path)])

        assert result.exit_code == 0
        assert "ood_input" in result.output or "Gap" in result.output


class TestVersionFlag:
    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "CausalGuard" in result.output
