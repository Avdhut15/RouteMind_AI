"""
app/evaluation/decision_engine.py
──────────────────────────────────
Phase 3 Part 4 — Quality Threshold & Decision Engine.

Consumes an EvaluationResult and produces a structured QualityDecisionResult
(PASS / FAIL / ESCALATE) based on configurable thresholds.

Design rules:
  - Stateless and deterministic: same inputs always produce the same output.
  - No LLM calls, no network access, no provider dependencies.
  - Explicit precedence rules for evaluation status and score:
      1. EvaluationStatus.FAILURE → DecisionOutcome.FAIL  (cannot trust the score)
      2. EvaluationStatus.SKIPPED → DecisionOutcome.ESCALATE  (unknown quality)
      3. overall_quality_score is None → DecisionOutcome.ESCALATE
         (conservatively escalate when no score is available)
      4. score >= pass_threshold  → DecisionOutcome.PASS
      5. score >= escalate_threshold → DecisionOutcome.ESCALATE
      6. score <  escalate_threshold → DecisionOutcome.FAIL

Threshold defaults:
  - pass_threshold:     0.7  (response must score ≥0.7 to be accepted)
  - escalate_threshold: 0.4  (below 0.4 the response is rejected outright)

These defaults are conservative but reasonable for a production quality gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from app.evaluation.models import (
    EvaluationResult,
    EvaluationSource,
    EvaluationStatus,
    QualityDecision,
)


# ── Default threshold values ───────────────────────────────────────────────────

DEFAULT_PASS_THRESHOLD: float = 0.7
DEFAULT_ESCALATE_THRESHOLD: float = 0.4


# ── Threshold configuration ───────────────────────────────────────────────────

@dataclass
class ThresholdConfig:
    """
    Configurable thresholds for the quality decision engine.

    Attributes:
        pass_threshold:     Minimum overall score required for PASS.
                            Must be in (0.0, 1.0].
        escalate_threshold: Minimum score for ESCALATE (below this → FAIL).
                            Must be in [0.0, pass_threshold).
    """
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
    escalate_threshold: float = DEFAULT_ESCALATE_THRESHOLD

    def __post_init__(self) -> None:
        _validate_threshold("pass_threshold", self.pass_threshold)
        _validate_threshold("escalate_threshold", self.escalate_threshold)
        if self.escalate_threshold >= self.pass_threshold:
            raise ValueError(
                f"escalate_threshold ({self.escalate_threshold}) must be strictly "
                f"less than pass_threshold ({self.pass_threshold})."
            )


def _validate_threshold(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value).__name__}.")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be a finite number, got {value}.")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}.")


# ── Decision result ───────────────────────────────────────────────────────────

@dataclass
class QualityDecisionResult:
    """
    Structured output of the quality decision engine.

    Carries enough context for Part 5 to act on the decision without
    needing to re-inspect the underlying EvaluationResult.

    Attributes:
        request_id:         Mirrors the originating LLMRequest.request_id.
        model_id:           Model that produced the evaluated response.
        provider:           Provider that served the response.
        decision:           Final quality decision (PASS/FAIL/ESCALATE).
        quality_score:      overall_quality_score from the EvaluationResult,
                            or None when unavailable.
        pass_threshold:     Threshold that was configured for PASS.
        escalate_threshold: Threshold that was configured for ESCALATE.
        reason:             Human-readable explanation of the decision.
        evaluation_status:  Status from the source EvaluationResult.
        evaluation_source:  Source/method from the EvaluationResult.
    """
    request_id: str
    model_id: str
    provider: str
    decision: QualityDecision
    quality_score: Optional[float]
    pass_threshold: float
    escalate_threshold: float
    reason: str
    evaluation_status: EvaluationStatus
    evaluation_source: EvaluationSource


# ── Decision engine ───────────────────────────────────────────────────────────

class QualityDecisionEngine:
    """
    Stateless decision engine that converts an EvaluationResult into a
    QualityDecisionResult.

    The engine applies the following precedence rules:

    Rule 1 — Evaluation failure takes precedence over all scores:
        EvaluationStatus.FAILURE → FAIL
        (A failed evaluation cannot establish acceptable quality.)

    Rule 2 — Skipped evaluations escalate conservatively:
        EvaluationStatus.SKIPPED → ESCALATE
        (Unknown quality should be escalated rather than accepted.)

    Rule 3 — Missing score escalates conservatively:
        overall_quality_score is None → ESCALATE
        (Treating a missing score as zero would unfairly bias toward FAIL;
        escalating preserves the option to retry with a stronger evaluator.)

    Rule 4–6 — Score-based thresholds (applied only when score is available):
        score >= pass_threshold      → PASS
        score >= escalate_threshold  → ESCALATE
        score <  escalate_threshold  → FAIL
    """

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self._config = config or ThresholdConfig()

    @property
    def config(self) -> ThresholdConfig:
        return self._config

    def decide(self, evaluation: EvaluationResult) -> QualityDecisionResult:
        """
        Produce a QualityDecisionResult from an EvaluationResult.

        Args:
            evaluation: The EvaluationResult to inspect.

        Returns:
            A fully constructed QualityDecisionResult.
        """
        pt = self._config.pass_threshold
        et = self._config.escalate_threshold
        score = evaluation.overall_quality_score

        # ── Rule 1: Evaluation itself failed ─────────────────────────────────
        if evaluation.status == EvaluationStatus.FAILURE:
            return QualityDecisionResult(
                request_id=evaluation.request_id,
                model_id=evaluation.model_id,
                provider=evaluation.provider,
                decision=QualityDecision.FAIL,
                quality_score=None,
                pass_threshold=pt,
                escalate_threshold=et,
                reason=(
                    "Evaluation failed; quality cannot be established. "
                    f"Error: {evaluation.error or 'unknown'}"
                ),
                evaluation_status=evaluation.status,
                evaluation_source=evaluation.source,
            )

        # ── Rule 2: Evaluation was skipped ────────────────────────────────────
        if evaluation.status == EvaluationStatus.SKIPPED:
            return QualityDecisionResult(
                request_id=evaluation.request_id,
                model_id=evaluation.model_id,
                provider=evaluation.provider,
                decision=QualityDecision.ESCALATE,
                quality_score=None,
                pass_threshold=pt,
                escalate_threshold=et,
                reason="Evaluation was skipped; escalating to ensure quality review.",
                evaluation_status=evaluation.status,
                evaluation_source=evaluation.source,
            )

        # ── Rule 3: Score is missing ──────────────────────────────────────────
        if score is None:
            return QualityDecisionResult(
                request_id=evaluation.request_id,
                model_id=evaluation.model_id,
                provider=evaluation.provider,
                decision=QualityDecision.ESCALATE,
                quality_score=None,
                pass_threshold=pt,
                escalate_threshold=et,
                reason=(
                    "No quality score is available; escalating conservatively "
                    "rather than assuming acceptability."
                ),
                evaluation_status=evaluation.status,
                evaluation_source=evaluation.source,
            )

        # ── Rules 4–6: Score-based threshold comparison ───────────────────────
        if score >= pt:
            decision = QualityDecision.PASS
            reason = (
                f"Score {score:.4f} meets or exceeds pass threshold {pt:.4f}."
            )
        elif score >= et:
            decision = QualityDecision.ESCALATE
            reason = (
                f"Score {score:.4f} is below pass threshold {pt:.4f} "
                f"but meets escalate threshold {et:.4f}."
            )
        else:
            decision = QualityDecision.FAIL
            reason = (
                f"Score {score:.4f} is below escalate threshold {et:.4f}."
            )

        return QualityDecisionResult(
            request_id=evaluation.request_id,
            model_id=evaluation.model_id,
            provider=evaluation.provider,
            decision=decision,
            quality_score=score,
            pass_threshold=pt,
            escalate_threshold=et,
            reason=reason,
            evaluation_status=evaluation.status,
            evaluation_source=evaluation.source,
        )
