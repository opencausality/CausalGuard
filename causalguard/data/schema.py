"""Core data models for CausalGuard.

These Pydantic models define every structured artefact produced and consumed
by the CausalGuard pipeline: failure modes, mitigations, safety edges,
coverage results, and the final safety case.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FailureMode(BaseModel):
    """A single failure mode in the causal failure graph.

    Failure modes are typed as HAZARD (root trigger conditions), FAILURE
    (intermediate propagation states), or HARM (final impacts to users or
    society).  The LLM assigns node_type based on the system description.
    """

    id: str = Field(description="Short identifier, e.g. F1, F2, H1.")
    name: str = Field(description="Concise human-readable name.")
    description: str = Field(description="What goes wrong and why it matters.")
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        description="Severity level if this failure mode is realised."
    )
    trigger_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that trigger or enable this failure mode.",
    )
    downstream_effects: list[str] = Field(
        default_factory=list,
        description="Names of failure modes or harms that this failure leads to.",
    )
    node_type: Literal["HAZARD", "FAILURE", "HARM"] = Field(
        default="FAILURE",
        description=(
            "HAZARD = root input condition (in-degree 0 in the graph), "
            "FAILURE = intermediate failure state, "
            "HARM = final harm to users or society (out-degree 0 or terminal)."
        ),
    )


class Mitigation(BaseModel):
    """A control measure that blocks or detects one or more failure modes."""

    id: str = Field(description="Short identifier, e.g. M1, M2.")
    name: str = Field(description="Concise name of the mitigation.")
    description: str = Field(description="What the mitigation does and how.")
    blocks_failure_modes: list[str] = Field(
        default_factory=list,
        description="Names of failure modes this mitigation prevents or detects.",
    )
    coverage_confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Confidence (0.0–1.0) that this mitigation is effective.",
    )
    mitigation_type: Literal["PREVENTIVE", "DETECTIVE", "CORRECTIVE"] = Field(
        default="PREVENTIVE",
        description=(
            "PREVENTIVE = prevents the failure from occurring, "
            "DETECTIVE = detects the failure after it occurs, "
            "CORRECTIVE = corrects or recovers from the failure."
        ),
    )


class SafetyEdge(BaseModel):
    """A directed causal edge in the failure propagation graph."""

    cause: str = Field(description="Name of the causing failure mode.")
    effect: str = Field(description="Name of the resulting failure mode or harm.")
    mechanism: str = Field(description="Explanation of how the cause leads to the effect.")
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Severity of this causal link.",
    )


class CoverageResult(BaseModel):
    """Coverage assessment for a single hazard→harm path."""

    path: list[str] = Field(description="Ordered list of node names from hazard to harm.")
    path_string: str = Field(description="Human-readable path, e.g. 'Hazard → F1 → Harm'.")
    is_covered: bool = Field(
        description="True if at least one mitigation covers at least one node on this path."
    )
    blocking_mitigations: list[str] = Field(
        default_factory=list,
        description="Names of mitigations that block this path.",
    )
    coverage_gap: bool = Field(
        description="True if NO mitigation covers this path (inverse of is_covered)."
    )
    gap_at_edge: Optional[str] = Field(
        default=None,
        description="The first edge (cause → effect) on this path that has no mitigation.",
    )


class SafetyCase(BaseModel):
    """Complete safety case for an AI system.

    Contains all extracted failure modes, mitigations, the causal graph edges,
    path-level coverage results, the safety verdict, and recommendations for
    closing any coverage gaps.
    """

    system_name: str = Field(description="Name of the AI system being assessed.")
    failure_modes: list[FailureMode] = Field(
        default_factory=list,
        description="All extracted failure modes (HAZARD, FAILURE, HARM nodes).",
    )
    mitigations: list[Mitigation] = Field(
        default_factory=list,
        description="All extracted and mapped mitigations.",
    )
    safety_edges: list[SafetyEdge] = Field(
        default_factory=list,
        description="Directed edges of the causal failure graph.",
    )
    coverage_results: list[CoverageResult] = Field(
        default_factory=list,
        description="Per-path coverage assessments.",
    )
    hazard_nodes: list[str] = Field(
        default_factory=list,
        description="Names of HAZARD nodes (root causes).",
    )
    harm_nodes: list[str] = Field(
        default_factory=list,
        description="Names of HARM nodes (terminal impacts).",
    )
    uncovered_paths: list[CoverageResult] = Field(
        default_factory=list,
        description="Subset of coverage_results where coverage_gap is True.",
    )
    safety_verdict: Literal["SAFE", "UNSAFE", "PARTIAL"] = Field(
        description=(
            "SAFE = all paths are covered at or above threshold, "
            "UNSAFE = coverage below 50%, "
            "PARTIAL = some paths covered but below threshold."
        )
    )
    coverage_percentage: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of hazard→harm paths covered by at least one mitigation.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations for closing coverage gaps.",
    )
    created_at: str = Field(description="ISO-8601 timestamp of when this safety case was built.")
    model_used: str = Field(
        default="",
        description="LLM model used for extraction.",
    )
    source_system_description: str = Field(
        default="",
        description="Original system description text used as input.",
    )
    source_incident_logs: str = Field(
        default="",
        description="Original incident logs text used as input.",
    )
    source_mitigations_text: str = Field(
        default="",
        description="Original mitigations text used as input.",
    )
