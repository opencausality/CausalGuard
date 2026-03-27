"""Safety case builder for CausalGuard.

Orchestrates the full pipeline: failure graph construction, path coverage
verification, gap detection, and final SafetyCase assembly.

The ``SafetyCaseBuilder`` is intentionally free of LLM calls — it receives
already-extracted failure modes and mitigations, and applies pure graph and
verification logic to produce a fully populated :class:`SafetyCase`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from causalguard.config import Settings, get_settings
from causalguard.data.schema import FailureMode, Mitigation, SafetyCase
from causalguard.graph.builder import build_failure_graph, identify_hazards, identify_harms
from causalguard.graph.validator import validate_graph
from causalguard.verification.coverage import verify_all_paths
from causalguard.verification.gaps import find_gaps, generate_recommendations

logger = logging.getLogger("causalguard.cases.builder")

# ── Verdict thresholds ────────────────────────────────────────────────────────

_UNSAFE_THRESHOLD = 0.50  # below this fraction → UNSAFE


class SafetyCaseBuilder:
    """Build a complete :class:`~causalguard.data.schema.SafetyCase` from
    extracted failure modes and mitigations.

    This class is the main integration point between graph construction,
    coverage verification, gap detection, and final report assembly.

    Parameters
    ----------
    settings:
        Application settings used for the coverage threshold and model name.
        Falls back to the global singleton if omitted.

    Examples
    --------
    >>> builder = SafetyCaseBuilder()
    >>> case = builder.build("My AI System", failure_modes, mitigations)
    >>> print(case.safety_verdict)
    'PARTIAL'
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        system_name: str,
        failure_modes: list[FailureMode],
        mitigations: list[Mitigation],
        *,
        model_used: str = "",
        source_system_description: str = "",
        source_incident_logs: str = "",
        source_mitigations_text: str = "",
    ) -> SafetyCase:
        """Build a :class:`SafetyCase` from failure modes and mitigations.

        Pipeline:
        1. Construct the causal failure graph (NetworkX DAG).
        2. Identify hazard nodes (root triggers) and harm nodes (terminal impacts).
        3. Validate the graph is structurally sound.
        4. Enumerate all hazard→harm paths and verify coverage.
        5. Detect coverage gaps and generate recommendations.
        6. Compute coverage percentage and determine the safety verdict.
        7. Assemble and return the :class:`SafetyCase`.

        Parameters
        ----------
        system_name:
            Human-readable name of the AI system being assessed.
        failure_modes:
            Extracted failure modes (HAZARD / FAILURE / HARM nodes).
        mitigations:
            Extracted mitigations with their ``blocks_failure_modes`` mappings.
        model_used:
            Optional: the LLM model used for extraction (recorded for audit).
        source_system_description:
            Original system description text (stored for traceability).
        source_incident_logs:
            Original incident logs text (stored for traceability).
        source_mitigations_text:
            Original mitigations text (stored for traceability).

        Returns
        -------
        SafetyCase
            Fully populated safety case with verdict, coverage results,
            gap paths, and recommendations.

        Raises
        ------
        GraphBuildError
            If no failure modes are provided, the graph contains unresolvable
            cycles, or no hazard/harm nodes can be found.
        VerificationError
            If path coverage verification fails due to structural issues.
        """
        logger.info(
            "Building safety case for '%s' with %d failure modes and %d mitigations.",
            system_name,
            len(failure_modes),
            len(mitigations),
        )

        # ── Step 1: Build causal failure graph ────────────────────────────
        graph, safety_edges = build_failure_graph(failure_modes)

        # ── Step 2: Identify hazards and harms ────────────────────────────
        hazard_nodes = identify_hazards(graph, failure_modes)
        harm_nodes = identify_harms(graph, failure_modes)

        logger.info(
            "Graph: %d nodes, %d edges. Hazards: %s. Harms: %s.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            hazard_nodes,
            harm_nodes,
        )

        # ── Step 3: Validate graph structure ──────────────────────────────
        validate_graph(graph, hazard_nodes, harm_nodes)

        # ── Step 4: Verify all hazard→harm path coverage ──────────────────
        coverage_results = verify_all_paths(
            graph=graph,
            hazards=hazard_nodes,
            harms=harm_nodes,
            mitigations=mitigations,
            failure_modes=failure_modes,
        )

        # ── Step 5: Detect gaps and generate recommendations ──────────────
        uncovered_paths = find_gaps(coverage_results)
        recommendations = generate_recommendations(uncovered_paths, failure_modes)

        # ── Step 6: Compute coverage percentage and verdict ───────────────
        total = len(coverage_results)
        covered = sum(1 for r in coverage_results if r.is_covered)
        coverage_percentage = (covered / total * 100.0) if total > 0 else 0.0

        safety_verdict = self._compute_verdict(coverage_percentage)

        logger.info(
            "Safety case built: coverage=%.1f%%, verdict=%s, gaps=%d/%d paths.",
            coverage_percentage,
            safety_verdict,
            len(uncovered_paths),
            total,
        )

        # ── Step 7: Assemble the SafetyCase ───────────────────────────────
        created_at = datetime.now(timezone.utc).isoformat()

        return SafetyCase(
            system_name=system_name,
            failure_modes=failure_modes,
            mitigations=mitigations,
            safety_edges=safety_edges,
            coverage_results=coverage_results,
            hazard_nodes=hazard_nodes,
            harm_nodes=harm_nodes,
            uncovered_paths=uncovered_paths,
            safety_verdict=safety_verdict,
            coverage_percentage=round(coverage_percentage, 2),
            recommendations=recommendations,
            created_at=created_at,
            model_used=model_used,
            source_system_description=source_system_description,
            source_incident_logs=source_incident_logs,
            source_mitigations_text=source_mitigations_text,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_verdict(self, coverage_percentage: float) -> str:
        """Derive the safety verdict from coverage percentage.

        Thresholds (using the configured ``coverage_threshold``):

        - ``SAFE``    — coverage_percentage >= threshold * 100
        - ``PARTIAL`` — 50% <= coverage_percentage < threshold * 100
        - ``UNSAFE``  — coverage_percentage < 50%

        Parameters
        ----------
        coverage_percentage:
            Percentage of hazard→harm paths that are covered (0–100).

        Returns
        -------
        str
            One of ``"SAFE"``, ``"PARTIAL"``, or ``"UNSAFE"``.
        """
        threshold_pct = self._settings.coverage_threshold * 100.0

        if coverage_percentage >= threshold_pct:
            return "SAFE"
        if coverage_percentage >= _UNSAFE_THRESHOLD * 100.0:
            return "PARTIAL"
        return "UNSAFE"
