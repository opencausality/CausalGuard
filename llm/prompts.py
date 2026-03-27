"""LLM prompt templates for CausalGuard.

All prompts:
- Request ONLY valid JSON output — no markdown fences, no prose.
- Include schema descriptions and few-shot examples for consistent output.
- Distinguish between correlation and causation where applicable.
- Handle ambiguous cases conservatively.
"""

from __future__ import annotations

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert AI safety engineer specialising in causal failure analysis. "
    "You extract structured safety information from system descriptions and incident logs. "
    "You ALWAYS return valid JSON exactly matching the requested schema — never markdown, "
    "never prose, never code fences. When uncertain, include the item with lower confidence "
    "rather than omitting it."
)

# ── Failure mode extraction ───────────────────────────────────────────────────

FAILURE_EXTRACTION_PROMPT = """You are an AI safety engineer. Analyze this AI system description and incident logs to extract ALL failure modes — including hazards (root triggers), intermediate failures, and terminal harms.

System description:
{system_description}

Incident logs:
{incident_logs}

For each failure mode, extract:
- id: short identifier (F1, F2, H1, etc.)
- name: concise name (3-6 words)
- description: what goes wrong and why it matters (1-2 sentences)
- severity: one of CRITICAL, HIGH, MEDIUM, LOW
- trigger_conditions: list of conditions that trigger or enable this failure
- downstream_effects: list of NAMES (not IDs) of other failure modes or harms this directly leads to
- node_type: one of:
    - HAZARD: root trigger condition (has no upstream causes in this system — e.g. distribution shift, corrupted input)
    - FAILURE: intermediate failure that propagates through the system
    - HARM: terminal harm to users, patients, or society (no downstream effects within this system)

Rules:
1. Extract ALL failure modes visible in the incidents, not just obvious ones.
2. Every HAZARD must have at least one downstream_effect.
3. Every HARM must have empty downstream_effects [].
4. FAILURE nodes connect HAZARDs to HARMs.
5. Use consistent names — the same failure mode must have the same name everywhere.
6. Distinguish actual causal mechanisms from mere correlations.

Few-shot example (medical AI):
{{
  "failure_modes": [
    {{
      "id": "H1",
      "name": "Distribution Shift",
      "description": "Model is tested on data from a different population than it was trained on.",
      "severity": "HIGH",
      "trigger_conditions": ["patient demographics differ from training data", "equipment differences"],
      "downstream_effects": ["Degraded Model Accuracy"],
      "node_type": "HAZARD"
    }},
    {{
      "id": "F1",
      "name": "Degraded Model Accuracy",
      "description": "Model predictions are systematically less accurate for the affected population.",
      "severity": "HIGH",
      "trigger_conditions": ["Distribution Shift", "Data Quality Issues"],
      "downstream_effects": ["Missed Diagnosis"],
      "node_type": "FAILURE"
    }},
    {{
      "id": "HARM1",
      "name": "Missed Diagnosis",
      "description": "A patient with the condition is not diagnosed, delaying or preventing treatment.",
      "severity": "CRITICAL",
      "trigger_conditions": ["Degraded Model Accuracy"],
      "downstream_effects": [],
      "node_type": "HARM"
    }}
  ]
}}

Now extract all failure modes from the provided system description and incident logs.
Return ONLY valid JSON (no markdown, no prose):
{{"failure_modes": [...]}}"""

# ── Mitigation extraction and mapping ─────────────────────────────────────────

MITIGATION_EXTRACTION_PROMPT = """You are an AI safety engineer. Given these failure modes and mitigation descriptions, map each mitigation to the failure modes it addresses.

Failure modes (JSON):
{failure_modes_json}

Mitigation descriptions:
{mitigations_text}

For each mitigation, extract:
- id: short identifier (M1, M2, etc.)
- name: concise name (3-6 words)
- description: what the mitigation does and how it works (1-2 sentences)
- blocks_failure_modes: list of failure mode NAMES (not IDs) that this mitigation prevents, detects, or corrects.
  Be thorough — include both direct blocks and any downstream effects that are indirectly prevented.
  Use the EXACT names from the failure_modes list above.
- coverage_confidence: float 0.0–1.0 indicating how confident you are this mitigation is effective.
  - 0.9+ : strong, well-specified control with clear mechanism
  - 0.7-0.9: reasonable control but may have limitations
  - 0.5-0.7: partial control, likely to miss some cases
  - <0.5: weak or uncertain control
- mitigation_type: one of:
    - PREVENTIVE: prevents the failure mode from occurring in the first place
    - DETECTIVE: detects the failure after it has occurred (enabling human intervention)
    - CORRECTIVE: fixes or recovers from the failure after detection

Rules:
1. Map mitigations to ALL failure modes they address, not just the most obvious one.
2. A human review process is typically DETECTIVE (detects wrong outputs) not PREVENTIVE.
3. Input validation is typically PREVENTIVE (prevents bad inputs reaching the model).
4. Monitoring is typically DETECTIVE.
5. Use EXACT failure mode names from the list — spelling must match exactly.

Few-shot example:
{{
  "mitigations": [
    {{
      "id": "M1",
      "name": "Human Radiologist Review",
      "description": "All AI recommendations must be reviewed and confirmed by a radiologist before clinical action.",
      "blocks_failure_modes": ["Missed Diagnosis", "False Positive Diagnosis", "Degraded Model Accuracy"],
      "coverage_confidence": 0.9,
      "mitigation_type": "DETECTIVE"
    }},
    {{
      "id": "M2",
      "name": "Input Image Quality Check",
      "description": "Validates DICOM image quality before inference; rejects images below quality threshold.",
      "blocks_failure_modes": ["Data Quality Issues", "Preprocessing Error"],
      "coverage_confidence": 0.85,
      "mitigation_type": "PREVENTIVE"
    }}
  ]
}}

Now extract and map all mitigations from the provided descriptions.
Return ONLY valid JSON (no markdown, no prose):
{{"mitigations": [...]}}"""

# ── Stricter retry prompt (used when first attempt fails to parse) ─────────────

STRICT_JSON_RETRY_SUFFIX = """

IMPORTANT: Your previous response could not be parsed as JSON.
Return ONLY a raw JSON object — no markdown code fences (```), no "json" prefix,
no explanatory text before or after. Start your response with {{ and end with }}.
"""
