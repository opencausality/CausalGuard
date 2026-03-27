"""REST API routes for CausalGuard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from causalguard.data.schema import SafetyCase

logger = logging.getLogger("causalguard.api.routes")

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────


class BuildRequest(BaseModel):
    """Request body for the /build endpoint."""

    system_description: str
    incident_logs: str
    mitigations_text: str
    system_name: str = "AI System"


class VerifyResponse(BaseModel):
    """Slim coverage summary returned by /verify."""

    verdict: str
    coverage_percentage: float
    total_paths: int
    covered_paths: int
    gap_paths: int
    recommendations: list[str]


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/build", response_model=SafetyCase)
async def build_safety_case(request: BuildRequest) -> SafetyCase:
    """Build a safety case from system documents.

    Runs LLM extraction, graph construction, and coverage verification.
    Returns the full :class:`~causalguard.data.schema.SafetyCase`.
    """
    from causalguard.cases.builder import SafetyCaseBuilder
    from causalguard.config import Settings
    from causalguard.exceptions import CausalGuardError
    from causalguard.extraction.failures import FailureModeExtractor
    from causalguard.extraction.mitigations import MitigationExtractor
    from causalguard.llm.adapter import LLMAdapter

    try:
        settings = Settings()
        adapter = LLMAdapter(settings=settings)

        failure_extractor = FailureModeExtractor(llm=adapter, settings=settings)
        failure_modes = failure_extractor.extract(
            system_description=request.system_description,
            incident_logs=request.incident_logs,
        )

        mitigation_extractor = MitigationExtractor(llm=adapter, settings=settings)
        mitigations = mitigation_extractor.extract(
            mitigations_text=request.mitigations_text,
            failure_modes=failure_modes,
        )

        builder = SafetyCaseBuilder(settings=settings)
        case = builder.build(
            system_name=request.system_name,
            failure_modes=failure_modes,
            mitigations=mitigations,
            model_used=settings.litellm_model,
            source_system_description=request.system_description,
            source_incident_logs=request.incident_logs,
            source_mitigations_text=request.mitigations_text,
        )
        return case

    except CausalGuardError as exc:
        logger.error("Build failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/verify", response_model=VerifyResponse)
async def verify_safety_case(case: SafetyCase) -> VerifyResponse:
    """Verify coverage of an existing safety case.

    Accepts a :class:`~causalguard.data.schema.SafetyCase` and returns
    a slim coverage summary.
    """
    covered = sum(1 for r in case.coverage_results if r.is_covered)
    gaps = len(case.uncovered_paths)

    return VerifyResponse(
        verdict=case.safety_verdict,
        coverage_percentage=case.coverage_percentage,
        total_paths=len(case.coverage_results),
        covered_paths=covered,
        gap_paths=gaps,
        recommendations=case.recommendations,
    )
