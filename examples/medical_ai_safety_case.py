"""Example: build a safety case for the medical AI pneumonia detection system.

This script demonstrates using CausalGuard programmatically to:
1. Load system description, incident logs, and mitigations from text files
2. Extract failure modes and mitigations via LLM
3. Build and verify the safety case
4. Print the human-readable report

Run:
    python examples/medical_ai_safety_case.py

Requirements:
    - Ollama running locally with llama3.1 pulled, OR
    - CAUSALGUARD_LLM_PROVIDER=anthropic with ANTHROPIC_API_KEY set
"""

from __future__ import annotations

import logging
from pathlib import Path

from causalguard.cases.builder import SafetyCaseBuilder
from causalguard.cases.exporter import format_text_report, save_safety_case
from causalguard.config import Settings, configure_logging
from causalguard.extraction.failures import FailureModeExtractor
from causalguard.extraction.mitigations import MitigationExtractor
from causalguard.llm.adapter import LLMAdapter

# ── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
SYSTEM_DESC = FIXTURES / "system_description.txt"
INCIDENT_LOGS = FIXTURES / "incident_logs.txt"
MITIGATIONS = FIXTURES / "mitigations.txt"
OUTPUT = Path("medical_ai_safety_case.json")


def main() -> None:
    """Run the full CausalGuard pipeline on the medical AI example."""
    settings = Settings()
    configure_logging(settings)

    logger.info("Loading input documents…")
    system_text = SYSTEM_DESC.read_text(encoding="utf-8")
    incidents_text = INCIDENT_LOGS.read_text(encoding="utf-8")
    mitigations_text = MITIGATIONS.read_text(encoding="utf-8")

    logger.info("Initialising LLM adapter: %s", settings.litellm_model)
    adapter = LLMAdapter(settings=settings)

    # ── Step 1: Extract failure modes ─────────────────────────────────────────
    logger.info("Extracting failure modes…")
    failure_extractor = FailureModeExtractor(llm=adapter, settings=settings)
    failure_modes = failure_extractor.extract(
        system_description=system_text,
        incident_logs=incidents_text,
    )
    logger.info("  %d failure modes extracted", len(failure_modes))
    for fm in failure_modes:
        logger.info("  [%s] %s (%s)", fm.node_type, fm.name, fm.severity)

    # ── Step 2: Extract and map mitigations ───────────────────────────────────
    logger.info("Extracting and mapping mitigations…")
    mitigation_extractor = MitigationExtractor(llm=adapter, settings=settings)
    mitigations = mitigation_extractor.extract(
        mitigations_text=mitigations_text,
        failure_modes=failure_modes,
    )
    logger.info("  %d mitigations mapped", len(mitigations))
    for m in mitigations:
        logger.info("  [%s] %s → blocks: %s", m.mitigation_type, m.name, m.blocks_failure_modes)

    # ── Step 3: Build the safety case ─────────────────────────────────────────
    logger.info("Building safety case…")
    builder = SafetyCaseBuilder(settings=settings)
    case = builder.build(
        system_name="Pneumonia Detection AI v2.1",
        failure_modes=failure_modes,
        mitigations=mitigations,
        model_used=settings.litellm_model,
        source_system_description=system_text,
        source_incident_logs=incidents_text,
        source_mitigations_text=mitigations_text,
    )

    # ── Step 4: Print the report ──────────────────────────────────────────────
    print("\n" + format_text_report(case))

    # ── Step 5: Save to disk ──────────────────────────────────────────────────
    save_safety_case(case, OUTPUT)
    logger.info("Safety case saved to %s", OUTPUT)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nVerdict: {case.safety_verdict}")
    print(f"Coverage: {case.coverage_percentage:.1f}% ({len(case.coverage_results)} paths)")
    if case.uncovered_paths:
        print(f"Gaps: {len(case.uncovered_paths)} uncovered path(s)")
        for rec in case.recommendations:
            print(f"  → {rec}")
    else:
        print("All paths covered. Safety case: COMPLETE.")


if __name__ == "__main__":
    main()
