"""
app/routing/result.py
──────────────────────
Phase 2 Part 5 — RouterResult: structured output of the IntelligentRouter.

RouterResult is the single object returned by IntelligentRouter.route().
It carries the full routing pipeline context alongside (or instead of) the
actual LLM response, enabling callers to inspect every step of the decision.

Design rules:
    - All fields are Optional so partial results can be returned on any
      failure path without raising exceptions to the caller.
    - The `status` field is the primary discriminator:
          "success"          – model selected and provider returned a response.
          "routing_failure"  – pipeline could not produce a valid routing
                               decision (no candidates, unknown model/provider).
          "provider_failure" – routing succeeded but the provider call failed.
    - RouterResult is a Pydantic BaseModel (read-only after construction).
    - The router preserves the original request_id throughout.
    - Factor-level scoring breakdown is surfaced in factor_scores so callers
      can inspect WHY a model was chosen (cost, latency, quality, task, complexity).
    - llm_response carries the raw LLMResponse for callers that only need output.

Phase 2 Part 5 scope:
    - RouterResult Pydantic model.
    - RouterStatus literal type.
    - Convenience properties: is_success, is_routing_failure, is_provider_failure.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.providers.base import LLMResponse
from app.routing.analyzer import RequestAnalysis


# Literal type for the three terminal states the router can reach.
RouterStatus = Literal["success", "routing_failure", "provider_failure"]


class RouterResult(BaseModel):
    """
    Structured output of IntelligentRouter.route().

    All fields are Optional to accommodate partial results on failure paths.
    Use `status` to determine whether the pipeline completed successfully.

    Fields
    ──────
    Identity
        request_id          Original LLMRequest.request_id (always present).
        status              Pipeline terminal state.
        error               Human-readable error description (failure paths only).

    Analysis pipeline
        request_analysis    Full RequestAnalysis from Part 1.
        complexity          Predicted complexity label value ("simple"/"moderate"/"complex").
        classifier_confidence  Classifier confidence score for the predicted class [0,1].
        classifier_model_id    Which classifier produced the prediction.

    Candidate selection
        candidate_model_ids  Model IDs that passed ALL filters in Part 3.

    Routing decision
        selected_model_id   Model ID chosen by Part 4 (highest routing score).
        selected_provider   Provider name for the selected model.
        routing_score       Final weighted routing score for the selected model [0,1].
        factor_scores       Per-factor score breakdown {"cost":…, "latency":…, …}.

    Provider execution
        llm_response        Raw LLMResponse from the provider (success only).
        output              Generated text (convenience accessor).
        input_tokens        Prompt token count from provider response.
        output_tokens       Generated token count from provider response.
        total_tokens        Total token count.
        estimated_cost      Estimated USD cost from provider response.
        latency_ms          End-to-end provider latency in milliseconds.
        finish_reason       Provider-reported stop reason.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id: str = Field(
        description="Mirrors LLMRequest.request_id — present on all paths."
    )
    status: RouterStatus = Field(
        description="Terminal pipeline state: success | routing_failure | provider_failure."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error description on failure paths; None on success.",
    )

    # ── Analysis pipeline ─────────────────────────────────────────────────────
    request_analysis: Optional[RequestAnalysis] = Field(
        default=None,
        description="Structured feature set from Part 1 RequestAnalyzer.",
    )
    complexity: Optional[str] = Field(
        default=None,
        description="Predicted complexity label value (simple/moderate/complex).",
    )
    classifier_confidence: Optional[float] = Field(
        default=None,
        description="Classifier confidence score for the predicted class [0,1].",
    )
    classifier_model_id: Optional[str] = Field(
        default=None,
        description="Identifier of the classifier that produced the prediction.",
    )

    # ── Candidate selection ───────────────────────────────────────────────────
    candidate_model_ids: Optional[list[str]] = Field(
        default=None,
        description="Model IDs that passed all Part 3 capability filters.",
    )

    # ── Routing decision ──────────────────────────────────────────────────────
    selected_model_id: Optional[str] = Field(
        default=None,
        description="Model ID selected by Part 4 RoutingScorer (rank-1).",
    )
    selected_provider: Optional[str] = Field(
        default=None,
        description="Provider name for the selected model.",
    )
    routing_score: Optional[float] = Field(
        default=None,
        description="Final weighted routing score for the selected model [0,1].",
    )
    factor_scores: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Per-factor routing score breakdown: "
            "{cost, latency, quality, task, complexity}."
        ),
    )

    # ── Provider execution ────────────────────────────────────────────────────
    llm_response: Optional[LLMResponse] = Field(
        default=None,
        description="Raw LLMResponse from the provider (present on success).",
    )
    output: Optional[str] = Field(
        default=None,
        description="Generated text output (convenience copy from llm_response.output).",
    )
    input_tokens: Optional[int] = Field(
        default=None,
        description="Prompt token count from the provider response.",
    )
    output_tokens: Optional[int] = Field(
        default=None,
        description="Generated token count from the provider response.",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total token count from the provider response.",
    )
    estimated_cost: Optional[float] = Field(
        default=None,
        description="Estimated USD cost from the provider response.",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="End-to-end provider latency in milliseconds.",
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Provider-reported stop reason (stop, length, error, etc.).",
    )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_success(self) -> bool:
        """True when the full pipeline completed and a response was generated."""
        return self.status == "success"

    @property
    def is_routing_failure(self) -> bool:
        """True when the routing decision could not be made (no candidates, etc.)."""
        return self.status == "routing_failure"

    @property
    def is_provider_failure(self) -> bool:
        """True when routing succeeded but the provider inference call failed."""
        return self.status == "provider_failure"

    def __repr__(self) -> str:
        return (
            f"RouterResult("
            f"request_id={self.request_id!r}, "
            f"status={self.status!r}, "
            f"selected={self.selected_model_id!r}, "
            f"error={self.error!r})"
        )
