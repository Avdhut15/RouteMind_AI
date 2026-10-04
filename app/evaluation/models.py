"""
app/evaluation/models.py
────────────────────────
Phase 3 Part 1 — Evaluation Result Schema.

Domain models and enums for representing response quality evaluations.
"""

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class EvaluationStatus(str, Enum):
    """Execution status of the evaluation itself."""
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class EvaluationSource(str, Enum):
    """The source or method used for evaluation."""
    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"
    DEEPEVAL = "deepeval"
    HYBRID = "hybrid"


class QualityDecision(str, Enum):
    """The outcome decision based on the quality evaluation."""
    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"


class EvaluationResult(BaseModel):
    """
    Structured result of a quality evaluation.

    Encapsulates the decision, the scores, and any explanatory reasoning.
    Designed to safely represent evaluation failures without propagating exceptions.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id: str = Field(..., description="Matches the original LLMRequest and LLMResponse.")
    model_id: str = Field(..., description="Model that generated the response being evaluated.")
    provider: str = Field(..., description="Provider that served the response.")

    # ── Evaluation Context ────────────────────────────────────────────────────
    status: EvaluationStatus = Field(..., description="Evaluation execution status.")
    source: EvaluationSource = Field(..., description="Method or evaluator used.")
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Scores (Only on SUCCESS/SKIPPED) ──────────────────────────────────────
    overall_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized aggregate score [0.0, 1.0].",
    )
    individual_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of specific metric scores (e.g., 'coherence': 0.9).",
    )

    # ── Outcomes ──────────────────────────────────────────────────────────────
    decision: Optional[QualityDecision] = Field(
        default=None,
        description="Final pass/fail/escalate decision.",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable or LLM-generated reasoning for the evaluation.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if evaluation status is FAILURE.",
    )

    @model_validator(mode="after")
    def validate_evaluation_state(self) -> "EvaluationResult":
        """
        Enforce logical consistency constraints on the evaluation result.
        """
        # Validate overall score
        if self.overall_quality_score is not None:
            if math.isnan(self.overall_quality_score) or math.isinf(self.overall_quality_score):
                raise ValueError("overall_quality_score cannot be NaN or Infinity.")

        # Validate individual scores
        for k, v in self.individual_scores.items():
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"Score for '{k}' cannot be NaN or Infinity.")
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Score for '{k}' must be between 0.0 and 1.0.")

        # Validate FAILURE state constraints
        if self.status == EvaluationStatus.FAILURE:
            if self.overall_quality_score is not None:
                raise ValueError("A failed evaluation cannot have an overall_quality_score.")
            if self.decision is not None:
                raise ValueError("A failed evaluation cannot yield a QualityDecision.")
            if not self.error:
                raise ValueError("A failed evaluation must provide an error message.")

        # Validate SUCCESS state constraints
        if self.status == EvaluationStatus.SUCCESS:
            if self.error is not None:
                raise ValueError("A successful evaluation should not contain an error message.")

        return self
