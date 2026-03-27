"""CausalGuard exception hierarchy.

All CausalGuard-specific errors inherit from ``CausalGuardError`` so callers
can catch the entire family with a single ``except CausalGuardError`` clause.
"""

from __future__ import annotations


class CausalGuardError(Exception):
    """Base class for all CausalGuard exceptions."""


class ProviderError(CausalGuardError):
    """Raised when no LLM provider is reachable or a call fails after retries.

    Examples
    --------
    - Ollama is not running locally
    - API key is missing or invalid
    - All retry attempts exhausted
    """


class ExtractionError(CausalGuardError):
    """Raised when the LLM output cannot be parsed into structured failure modes or mitigations.

    Examples
    --------
    - LLM returned prose instead of JSON
    - JSON is structurally invalid
    - Required fields are missing or have wrong types
    """


class GraphBuildError(CausalGuardError):
    """Raised when the causal failure graph cannot be constructed.

    Examples
    --------
    - No failure modes extracted
    - Graph contains cycles (violates DAG requirement)
    - No HAZARD nodes found
    - No HARM nodes found
    """


class VerificationError(CausalGuardError):
    """Raised when the safety case verification process fails.

    Examples
    --------
    - Graph is disconnected (hazards not reachable from harms)
    - Coverage computation fails due to malformed input
    """
