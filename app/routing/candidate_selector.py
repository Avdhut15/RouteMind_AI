"""
app/routing/candidate_selector.py
───────────────────────────────────
Phase 2 Part 3 — Model Capability & Candidate Selection.

Takes the analyzed request (RequestAnalysis) and predicted complexity
(ComplexityPrediction) and returns the set of models that are ELIGIBLE
to handle the request.

What this component does:
    - Filters models by enabled status.
    - Applies capability-requirement rules deterministically.
    - Returns a structured CandidateSelectionResult for downstream use.

What this component does NOT do:
    - Does NOT rank or score candidates.
    - Does NOT select the final model.
    - Does NOT call any LLM provider.
    - Does NOT require API keys or network access.
    - Does NOT implement routing, escalation, or quality evaluation.

==================================================
FILTERING RULES (deterministic, in order applied)
==================================================

1. ENABLED FILTER
   Only models with enabled=True are considered.

2. COMPLEXITY / QUALITY-TIER FILTER
   Maps predicted complexity → minimum acceptable QualityTier:
     simple   → tier_1, tier_2, tier_3  (all tiers)
     moderate → tier_2, tier_3
     complex  → tier_3 only

3. REASONING REQUIREMENT FILTER
   If requires_reasoning=True in RequestAnalysis:
     Keep only models where reasoning_score >= REASONING_THRESHOLD (0.60).

4. CODE REQUIREMENT FILTER
   If requires_code=True in RequestAnalysis:
     Keep only models where coding_score >= CODING_THRESHOLD (0.55).

5. STRUCTURED OUTPUT REQUIREMENT FILTER
   If requires_structured_output=True in RequestAnalysis:
     Keep only models where supports_structured_output=True.

6. CONTEXT WINDOW FILTER
   Always applied: keep only models where
     model.context_window >= request.approx_token_count * CONTEXT_SAFETY_FACTOR (2.0)
   (Ensures models have enough headroom for input + expected output.)

The result is the set of models that pass ALL applicable filters.
Models are returned in a deterministic order (sorted by model_id).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.classification.complexity_classifier import ComplexityLabel, ComplexityPrediction
from app.models.config import ModelConfig, QualityTier
from app.models.registry import ModelRegistry
from app.routing.analyzer import RequestAnalysis


# ── Filtering thresholds ──────────────────────────────────────────────────────
# Kept as module-level constants so downstream components can reference them
# and so they can be overridden in tests via monkeypatching if needed.

REASONING_SCORE_THRESHOLD: float = 0.60
CODING_SCORE_THRESHOLD: float = 0.55
CONTEXT_SAFETY_FACTOR: float = 2.0   # context_window must be ≥ token_count × factor


# ── Complexity → minimum tier mapping ────────────────────────────────────────

_COMPLEXITY_MIN_TIERS: dict[ComplexityLabel, set[QualityTier]] = {
    ComplexityLabel.SIMPLE: {
        QualityTier.TIER_1,
        QualityTier.TIER_2,
        QualityTier.TIER_3,
    },
    ComplexityLabel.MODERATE: {
        QualityTier.TIER_2,
        QualityTier.TIER_3,
    },
    ComplexityLabel.COMPLEX: {
        QualityTier.TIER_3,
    },
}


# ── Filter-rejection reasons ──────────────────────────────────────────────────

class RejectionReason(str, Enum):
    """Why a model was excluded from the candidate set."""
    DISABLED            = "disabled"
    TIER_TOO_LOW        = "tier_too_low"
    REASONING_SCORE_LOW = "reasoning_score_low"
    CODING_SCORE_LOW    = "coding_score_low"
    NO_STRUCTURED_OUTPUT = "no_structured_output"
    CONTEXT_WINDOW_TOO_SMALL = "context_window_too_small"


# ── Per-model selection outcome ───────────────────────────────────────────────

@dataclass
class ModelSelectionOutcome:
    """Records why a specific model was accepted or rejected."""
    model: ModelConfig
    accepted: bool
    rejection_reason: Optional[RejectionReason] = None

    @property
    def model_id(self) -> str:
        return self.model.model_id


# ── Selection result ──────────────────────────────────────────────────────────

@dataclass
class CandidateSelectionResult:
    """
    Structured output of the candidate selector.

    Consumed downstream by:
        - Routing Score Calculator (Part 4)
        - Intelligent Router (Part 5)

    Fields:
        candidates:          Models that passed ALL filters (sorted by model_id).
        rejected:            Models that failed at least one filter, with reason.
        request_analysis:    The RequestAnalysis that drove selection.
        complexity:          The ComplexityLabel that drove tier filtering.
        applied_filters:     Human-readable description of the filters applied.
        has_candidates:      Convenience bool — True if candidates is non-empty.
    """
    candidates: list[ModelConfig]
    rejected: list[ModelSelectionOutcome]
    request_analysis: RequestAnalysis
    complexity: ComplexityLabel
    applied_filters: list[str]

    @property
    def has_candidates(self) -> bool:
        return len(self.candidates) > 0

    @property
    def candidate_ids(self) -> list[str]:
        return [m.model_id for m in self.candidates]

    @property
    def rejected_ids(self) -> list[str]:
        return [r.model_id for r in self.rejected]

    def summary(self) -> str:
        return (
            f"CandidateSelectionResult("
            f"complexity={self.complexity.value}, "
            f"candidates={len(self.candidates)}, "
            f"rejected={len(self.rejected)})"
        )


# ── Candidate Selector ────────────────────────────────────────────────────────

class CandidateSelector:
    """
    Stateless, deterministic model candidate selector.

    Usage::

        selector = CandidateSelector(registry)
        result = selector.select(analysis, complexity_prediction)

        if result.has_candidates:
            # pass result.candidates to Part 4 (routing score)
        else:
            # no eligible models — handle gracefully

    The selector is intentionally stateless and holds no mutable per-request
    state, so a single shared instance is safe across threads.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        if not isinstance(registry, ModelRegistry):
            raise TypeError(
                f"registry must be a ModelRegistry instance, got {type(registry)}"
            )
        self._registry = registry

    def select(
        self,
        analysis: RequestAnalysis,
        complexity_prediction: ComplexityPrediction,
    ) -> CandidateSelectionResult:
        """
        Apply capability filters and return eligible model candidates.

        Args:
            analysis:              Output of RequestAnalyzer.analyze().
            complexity_prediction: Output of a trained ComplexityClassifier.predict().

        Returns:
            CandidateSelectionResult with accepted candidates and rejection log.
        """
        complexity = complexity_prediction.complexity
        allowed_tiers = _COMPLEXITY_MIN_TIERS[complexity]
        applied_filters: list[str] = []
        outcomes: list[ModelSelectionOutcome] = []

        # Retrieve ALL models from registry (including disabled) so we can
        # explain why each was excluded.
        all_models = list(self._registry._models.values())
        # Sort deterministically by model_id
        all_models.sort(key=lambda m: m.model_id)

        for model in all_models:
            outcome = self._evaluate_model(
                model, analysis, complexity, allowed_tiers
            )
            outcomes.append(outcome)

        candidates = [o.model for o in outcomes if o.accepted]
        rejected   = [o        for o in outcomes if not o.accepted]

        # Build human-readable filter description
        applied_filters = _describe_filters(analysis, complexity)

        return CandidateSelectionResult(
            candidates=candidates,
            rejected=rejected,
            request_analysis=analysis,
            complexity=complexity,
            applied_filters=applied_filters,
        )

    # ── Internal filter logic ─────────────────────────────────────────────────

    def _evaluate_model(
        self,
        model: ModelConfig,
        analysis: RequestAnalysis,
        complexity: ComplexityLabel,
        allowed_tiers: set[QualityTier],
    ) -> ModelSelectionOutcome:
        """
        Run all filters for one model.  Returns on the first failing filter
        (early exit) so the rejection_reason is the primary/first failure.
        """

        # ── Filter 1: Enabled ────────────────────────────────────────────────
        if not model.enabled:
            return ModelSelectionOutcome(
                model=model,
                accepted=False,
                rejection_reason=RejectionReason.DISABLED,
            )

        # ── Filter 2: Complexity / Quality Tier ─────────────────────────────
        if model.quality_tier not in allowed_tiers:
            return ModelSelectionOutcome(
                model=model,
                accepted=False,
                rejection_reason=RejectionReason.TIER_TOO_LOW,
            )

        # ── Filter 3: Reasoning requirement ─────────────────────────────────
        if analysis.requires_reasoning:
            if model.reasoning_score < REASONING_SCORE_THRESHOLD:
                return ModelSelectionOutcome(
                    model=model,
                    accepted=False,
                    rejection_reason=RejectionReason.REASONING_SCORE_LOW,
                )

        # ── Filter 4: Coding requirement ─────────────────────────────────────
        if analysis.requires_code:
            if model.coding_score < CODING_SCORE_THRESHOLD:
                return ModelSelectionOutcome(
                    model=model,
                    accepted=False,
                    rejection_reason=RejectionReason.CODING_SCORE_LOW,
                )

        # ── Filter 5: Structured output requirement ───────────────────────────
        if analysis.requires_structured_output:
            if not model.supports_structured_output:
                return ModelSelectionOutcome(
                    model=model,
                    accepted=False,
                    rejection_reason=RejectionReason.NO_STRUCTURED_OUTPUT,
                )

        # ── Filter 6: Context window ──────────────────────────────────────────
        # Require the model's context window to be at least
        # approx_token_count × CONTEXT_SAFETY_FACTOR tokens, so there is
        # headroom for both the prompt and the expected output.
        required_tokens = analysis.approx_token_count * CONTEXT_SAFETY_FACTOR
        if model.context_window < required_tokens:
            return ModelSelectionOutcome(
                model=model,
                accepted=False,
                rejection_reason=RejectionReason.CONTEXT_WINDOW_TOO_SMALL,
            )

        # ── All filters passed ────────────────────────────────────────────────
        return ModelSelectionOutcome(model=model, accepted=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _describe_filters(
    analysis: RequestAnalysis,
    complexity: ComplexityLabel,
) -> list[str]:
    """Return a human-readable list of which filters were applied."""
    filters = [
        "enabled=True",
        f"quality_tier ∈ {[t.value for t in _COMPLEXITY_MIN_TIERS[complexity]]}",
    ]
    if analysis.requires_reasoning:
        filters.append(f"reasoning_score >= {REASONING_SCORE_THRESHOLD}")
    if analysis.requires_code:
        filters.append(f"coding_score >= {CODING_SCORE_THRESHOLD}")
    if analysis.requires_structured_output:
        filters.append("supports_structured_output=True")
    filters.append(
        f"context_window >= approx_token_count × {CONTEXT_SAFETY_FACTOR}"
    )
    return filters
