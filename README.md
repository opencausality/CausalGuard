<div align="center">

# CausalGuard

**Causal safety cases for AI systems.**

*"Every causal path from known hazard to harm is blocked by at least one verified mitigation."*
Now you can prove it.

[![CI](https://github.com/opencausality/causalguard/actions/workflows/causalguard-ci.yml/badge.svg)](https://github.com/opencausality/causalguard/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is CausalGuard?

CausalGuard builds **causal safety cases** for AI systems. Given incident logs, a system
description, and a list of mitigations, it:

1. Extracts failure modes and their causal relationships (using LLM)
2. Builds a causal DAG: hazards → failure modes → harms
3. Formally verifies that every path from hazard to harm is **blocked by at least one mitigation**
4. Identifies gaps — uncovered paths — and recommends new mitigations

```
System description + incidents + mitigations
              │
              ▼ LLM extraction
        Failure modes + Mitigations
              │
              ▼ graph construction
  Failure cascade DAG (hazard → failure → harm)
              │
              ▼ path enumeration
  All hazard → harm paths
              │
              ▼ coverage verification
  ✅ Covered paths (mitigation blocks at least one node)
  ❌ Gap paths (no mitigation on any node in path)
              │
              ▼
  Safety verdict: SAFE / UNSAFE / PARTIAL
```

### Key Features

- 🔍 **LLM extraction** — extracts structured failure modes and mitigations from text
- 🕸️ **Failure cascade graph** — causal DAG connecting hazards, failures, and harms
- ✅ **Coverage verification** — formal check: every hazard→harm path has a blocking mitigation
- ❌ **Gap detection** — identifies uncovered paths with specific recommendations
- 📋 **Safety case export** — structured JSON + human-readable report
- 🏠 **Local-first** — Ollama default, no API key required

---

## Why Behavioral Testing Isn't Enough

Standard AI safety evaluation runs test cases and checks outputs:
*"We tested 10,000 inputs and the model passed 97% of them."*

**The problem**: behavioral testing shows *that* the system works in tested scenarios.
It cannot show *why* it's safe, which failure modes exist, or whether every causal
pathway to harm has been addressed.

| | Behavioral Testing | Red-Teaming | CausalGuard |
|---|---|---|---|
| **What it proves** | Passes test cases | Finds adversarial examples | Every failure path is mitigated |
| **Systematic coverage** | Sample-dependent | Adversary-dependent | Formal path enumeration |
| **Failure mechanism** | Unknown | Partially explored | Explicitly modeled |
| **Mitigation verification** | Not checked | Not checked | ✅ Verified per path |
| **Residual risk** | Unknown | Partially known | Explicitly quantified |
| **Regulatory evidence** | Test results | Red team report | Causal safety case |

### Concrete Example

**System**: Medical AI for pneumonia detection from chest X-rays.

**CausalGuard output:**
```
CausalGuard — Safety Case Report
══════════════════════════════════

System: Pneumonia Detection AI v2.1
Failure modes extracted: 6
Mitigations mapped: 4
Paths analyzed: 8

Coverage Results:

Path 1: distribution_shift → high_confidence_wrong → harm_to_patient
  ✅ COVERED by: "Human radiologist review required for all recommendations"

Path 2: image_artifact → wrong_preprocessing → harm_to_patient
  ✅ COVERED by: "Input validation: image quality score check"

Path 3: out_of_distribution_input → high_confidence_wrong → harm_to_patient
  ❌ UNCOVERED — no mitigation blocks this path
  Gap at: out_of_distribution_input
  Recommendation: Add OOD detection before inference

Path 4: software_update → silent_performance_degradation → harm_to_patient
  ❌ UNCOVERED — no mitigation detects silent degradation
  Gap at: silent_performance_degradation
  Recommendation: Add automated performance regression tests on every deployment

Path 5: corrupted_metadata → wrong_preprocessing → harm_to_patient
  ✅ COVERED by: "Input validation: image quality score check"

... (3 more paths)

══════════════════════════════════
Safety verdict: PARTIAL (6/8 paths covered = 75%)
Target: 100% coverage

Critical gaps: 2 uncovered paths to patient harm
Recommendations:
  1. Add OOD detection before inference (covers 1 gap path)
  2. Add automated regression testing on deployment (covers 1 gap path)
  3. Add monitoring for model confidence calibration drift
```

The safety case is not *"we think it's safe"* — it's *"here are all known hazard→harm paths,
and here is the evidence that each is covered (or not)."*

---

## Alignment with AI Safety Frameworks

**International AI Safety Report 2026** (Bengio et al., 100+ experts):
> "AI safety arguments should link observed outcomes to identifiable mechanisms through
> causal models. Behavioral testing alone is insufficient to establish safety guarantees."

**EU AI Act — High-Risk AI Systems**:
The Act requires high-risk AI (medical, hiring, credit, law enforcement) to demonstrate:
- Systematic risk assessment and mitigation
- Ongoing monitoring and incident reporting
- Auditability of safety-relevant decisions

CausalGuard produces structured safety case artifacts that address these requirements.

**NIST AI Risk Management Framework (AI RMF)**:
CausalGuard's failure mode extraction and coverage verification map directly to the
*Govern*, *Map*, and *Measure* functions of the NIST AI RMF.

---

## Installation

```bash
pip install causalguard
# or
uv add causalguard
```

**Requirements**: Python 3.10+. Local Ollama (recommended) for LLM extraction.

---

## Quick Start

### 1. Build a safety case

```bash
causalguard build \
  --system system_description.txt \
  --incidents incident_logs.txt \
  --mitigations mitigations.txt \
  --output safety_case.json
```

### 2. Verify coverage

```bash
causalguard verify --case safety_case.json
```

### 3. Find uncovered paths (gaps)

```bash
causalguard gaps --case safety_case.json
```

### 4. Visualize the safety case graph

```bash
causalguard show --case safety_case.json
```

Opens an interactive browser graph. Hazard nodes in orange, harm nodes in red, covered
edges in green, uncovered edges in bold red.

---

## CLI Reference

```bash
# Build safety case from documents
causalguard build \
  --system system.txt \
  --incidents incidents.txt \
  --mitigations mitigations.txt \
  --output safety_case.json \
  [--show]

# Verify coverage of existing safety case
causalguard verify --case safety_case.json

# Show only uncovered (gap) paths
causalguard gaps --case safety_case.json

# Visualize safety case graph
causalguard show --case safety_case.json

# LLM provider status
causalguard providers

# REST API server
causalguard serve --port 8000
```

---

## Architecture

```
system.txt + incidents.txt + mitigations.txt
               │
               ▼
┌─────────────────────────┐
│  Failure Mode Extractor │  ← LLM extracts:
│  (extraction/failures)  │    - HAZARD nodes (root input conditions)
│                         │    - FAILURE nodes (intermediate failures)
│                         │    - HARM nodes (final patient/user harm)
└─────────────────────────┘
               │
               ▼
┌─────────────────────────┐
│  Mitigation Extractor   │  ← LLM maps mitigations to failure modes
│  (extraction/mitigations)    they block
└─────────────────────────┘
               │
               ▼
┌─────────────────────────┐
│  Failure Graph Builder  │  ← NetworkX DAG:
│  (graph/builder)        │    HAZARD → FAILURE → HARM
└─────────────────────────┘
               │
               ▼
┌─────────────────────────┐
│  Coverage Verification  │  ← For each hazard→harm path:
│  (verification/coverage)│    is at least one node covered?
└─────────────────────────┘
               │
               ▼
┌─────────────────────────┐
│  Gap Detector           │  ← Uncovered paths
│  + Recommendations      │    Per-gap recommendations
└─────────────────────────┘
               │
               ▼ SafetyCase JSON
┌─────────────────────────┐
│  Exporter               │  ← Structured JSON + rich text report
└─────────────────────────┘
```

---

## Input Documents

### System description (`system.txt`)

Any plain text describing your AI system. Include:
- What the system does
- Inputs it accepts
- Outputs it produces
- Deployment context

### Incident logs (`incidents.txt`)

Real or hypothetical incident descriptions. Each incident should describe:
- What went wrong
- What the system did incorrectly
- What harm resulted or could have resulted

### Mitigations (`mitigations.txt`)

List of mitigations you have or plan to implement:
- Human review requirements
- Input validation checks
- Output confidence thresholds
- Performance monitoring
- Access controls

---

## Coverage Model

A **path** from hazard to harm is *covered* if at least one node in that path is addressed
by at least one mitigation. Specifically: a node N is covered if any mitigation lists N's
name (or a close match) in its `blocks_failure_modes` list.

**Coverage percentage** = (covered paths / total paths) × 100%

**Safety verdicts:**
- `SAFE`: coverage_percentage ≥ coverage_threshold (default 100%)
- `PARTIAL`: coverage_percentage between 50% and threshold
- `UNSAFE`: coverage_percentage < 50%

---

## Configuration

```env
# LLM for extraction
CAUSALGUARD_LLM_PROVIDER=ollama
CAUSALGUARD_LLM_MODEL=ollama/llama3.1

# Cloud providers
CAUSALGUARD_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Safety settings
CAUSALGUARD_COVERAGE_THRESHOLD=1.0  # 1.0 = 100% path coverage required

CAUSALGUARD_LOG_LEVEL=INFO
```

---

## Data Model

```python
@dataclass
class FailureMode:
    id: str
    name: str
    description: str
    severity: str           # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    node_type: str          # "HAZARD", "FAILURE", "HARM"
    trigger_conditions: list[str]
    downstream_effects: list[str]

@dataclass
class CoverageResult:
    path: list[str]
    path_string: str        # "hazard → failure1 → harm"
    is_covered: bool
    blocking_mitigations: list[str]
    coverage_gap: bool

@dataclass
class SafetyCase:
    system_name: str
    failure_modes: list[FailureMode]
    mitigations: list[Mitigation]
    coverage_results: list[CoverageResult]
    safety_verdict: str     # "SAFE", "UNSAFE", "PARTIAL"
    coverage_percentage: float
    recommendations: list[str]
```

---

## Philosophy

CausalGuard is built on the principle that **AI safety requires causal argumentation**.

- 🏠 **Local-first**: Ollama default — your system documents never leave your machine
- 🔓 **Open source**: All safety case logic is MIT licensed
- 🚫 **No telemetry**: Zero data collection
- 🧠 **Causal, not behavioral**: Formal path coverage, not test case coverage

---

## Contributing

CausalGuard is free for research, safety engineering, and educational use.
If you're building commercial AI safety tooling on top of CausalGuard, consider contributing
domain-specific failure mode templates (medical AI, autonomous systems, financial AI).

*"Safety is not the absence of failures. It is the presence of defenses on every failure path."*
