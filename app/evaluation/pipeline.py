"""
app/evaluation/pipeline.py
──────────────────────────
Phase 3 Part 5 — Evaluation Pipeline Integration.

Orchestrates the full evaluation flow:

    LLMRequest + LLMResponse
            ↓
        Evaluator.evaluate()
            ↓
        EvaluationResult
            ↓
        QualityDecisionEngine.decide()
            ↓
        QualityDecisionResult

Design rules:
  - Stateless: no global mutable state, no caches, no persistence.
  - Evaluator is injected at construction time (DI); the pipeline itself
    never instantiates or selects a concrete evaluator.
  - Unexpected evaluator exceptions are caught and converted into a
    structured EvaluationResult(status=FAILURE) before being passed
    to the decision engine — the pipeline never propagates raw exceptions.
  - The distinction between Evaluation and Decision is preserved:
      Evaluation  → "How good is this response?"
      Decision    → "What does that mean operationally?"
      Pipeline    → "How do we run both together?"
  - ESCALATE in the decision result means "report ESCALATE". The pipeline
    does NOT select another model, does NOT retry, does NOT route anywhere.
    That belongs to a future architectural stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.decision_engine import QualityDecisionEngine, ThresholdConfig
from app.evaluation.models import (
    EvaluationResult,
    EvaluationSource,
    EvaluationStatus,
    QualityDecision,
)
from app.providers.base import LLMRequest, LLMResponse


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class EvaluationPipelineResult:
    """
    Structured output of the EvaluationPipeline.

    Carries the key context needed by downstream components without
    duplicating the full nested model trees.

    Attributes:
        request_id:         Mirrors LLMRequest.request_id.
        model_id:           Model that produced the evaluated response.
        provider:           Provider that served the response.
        evaluation_result:  Full EvaluationResult from the evaluator.
        decision:           Final QualityDecision (PASS/FAIL/ESCALATE).
        quality_score:      overall_quality_score, or None if unavailable.
        evaluation_status:  Status from the EvaluationResult.
        evaluation_source:  Source/method that performed the evaluation.
        reason:             Human-readable explanation of the decision.
    """
    request_id: str
    model_id: str
    provider: str
    evaluation_result: EvaluationResult
    decision: QualityDecision
    quality_score: Optional[float]
    evaluation_status: EvaluationStatus
    evaluation_source: EvaluationSource
    reason: str


# ── Pipeline ──────────────────────────────────────────────────────────────────

class EvaluationPipeline:
    """
    Stateless evaluation orchestrator.

    Wires a BaseEvaluator and a QualityDecisionEngine together into a single
    reusable async call.

    Usage::

        evaluator = DeterministicEvaluator()
        pipeline = EvaluationPipeline(evaluator=evaluator)
        pipeline_result = await pipeline.run(request, response)

        if pipeline_result.decision == QualityDecision.PASS:
            ...
        elif pipeline_result.decision == QualityDecision.ESCALATE:
            ...
        else:
            ...

    To use a custom decision threshold::

        pipeline = EvaluationPipeline(
            evaluator=evaluator,
            threshold_config=ThresholdConfig(pass_threshold=0.8, escalate_threshold=0.5),
        )
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        threshold_config: Optional[ThresholdConfig] = None,
    ) -> None:
        if not isinstance(evaluator, BaseEvaluator):
            raise TypeError(
                f"evaluator must implement BaseEvaluator, got {type(evaluator).__name__}."
            )
        self._evaluator = evaluator
        self._decision_engine = QualityDecisionEngine(threshold_config)

    async def run(self, request: LLMRequest, response: LLMResponse) -> EvaluationPipelineResult:
        """
        Execute the full evaluation → decision pipeline.

        Args:
            request:  The original LLMRequest.
            response: The LLMResponse to evaluate.

        Returns:
            EvaluationPipelineResult with evaluation and decision context.
        """
        # ── Step 1: Evaluate ──────────────────────────────────────────────────
        try:
            evaluation_result = await self._evaluator.evaluate(request, response)
        except Exception as exc:
            # Defensive: convert unexpected evaluator exceptions into a failure
            # result rather than crashing the orchestration layer.
            req_id = getattr(request, "request_id", "unknown")
            mod_id = getattr(response, "model_id", "unknown")
            prov = getattr(response, "provider", "unknown")
            source = getattr(self._evaluator, "_source", EvaluationSource.DETERMINISTIC)

            evaluation_result = EvaluationResult(
                request_id=req_id,
                model_id=mod_id,
                provider=prov,
                status=EvaluationStatus.FAILURE,
                source=source,
                error=(
                    f"Evaluator raised an unexpected exception: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        # ── Step 2: Decide ────────────────────────────────────────────────────
        decision_result = self._decision_engine.decide(evaluation_result)

        # ── Step 3: Assemble pipeline result ──────────────────────────────────
        return EvaluationPipelineResult(
            request_id=evaluation_result.request_id,
            model_id=evaluation_result.model_id,
            provider=evaluation_result.provider,
            evaluation_result=evaluation_result,
            decision=decision_result.decision,
            quality_score=decision_result.quality_score,
            evaluation_status=evaluation_result.status,
            evaluation_source=evaluation_result.source,
            reason=decision_result.reason,
        )
