"""
app/routing/scoring.py
───────────────────────
Phase 2 Part 4 — Routing Score & Model Selection.

Takes the candidates from Part 3 (CandidateSelectionResult) and scores each
one to determine the best model to handle the request.

==================================================
PIPELINE POSITION
==================================================

LLMRequest
    ↓ Part 1
RequestAnalysis
    ↓ Part 2
ComplexityPrediction
    ↓ Part 3
CandidateSelectionResult      ← input to this module
    ↓ Part 4 (this file)
RoutingScoreResult
    ↓
selected_model: ModelConfig   ← output; Part 5 will call the provider

==================================================
SCORING FORMULA
==================================================

For each candidate model, five normalised factor scores are computed,
all in the range [0.0, 1.0] where higher is always better:

  cost_score      = 1 - normalise(estimated_cost, min_cost, max_cost)
                    (cheapest model scores 1.0; most expensive scores 0.0)

  latency_score   = 1 - normalise(latency_ms, min_latency, max_latency)
                    (fastest model scores 1.0; slowest scores 0.0)

  quality_score   = model.quality_score
                    (as configured in models.yaml; already in [0, 1])

  task_score      = task-specific model score for the detected task type
                    (e.g., reasoning_score for REASONING tasks)

  complexity_score = 1.0 for tier-3 on complex; 0.67 for tier-2 on moderate;
                     1.0 for any tier on simple (all already passed filter)
                     (rewards matching tier without over-penalising)

The final routing score is the weighted average:

  routing_score = (
      w_cost      * cost_score      +
      w_latency   * latency_score   +
      w_quality   * quality_score   +
      w_task      * task_score      +
      w_complexity * complexity_score
  ) / sum(weights)

Default weights (sum to 1.0):
  cost       = 0.25
  latency    = 0.20
  quality    = 0.25
  task       = 0.20
  complexity = 0.10

==================================================
NORMALIZATION
==================================================

Cost and latency are normalised across the candidate set using
min-max scaling.  When all candidates have the same value (min == max),
every candidate scores 0.5 (neutral — no differentiation possible).

This approach:
  - Is deterministic (same inputs → same scores).
  - Does not require external data.
  - Works correctly with a single candidate.

==================================================
COST ESTIMATION
==================================================

Estimated cost = CostCalculator.calculate(model,
    input_tokens  = analysis.approx_token_count,
    output_tokens = _DEFAULT_OUTPUT_TOKENS  # 256 tokens constant fallback
)

Using analysis.approx_token_count for input tokens and a fixed
default for output tokens gives a deterministic cost estimate
before the actual response is generated.  The value is clearly
labelled as an estimate and is not presented as a real measured cost.

==================================================
LATENCY
==================================================

Uses model.average_latency_ms from ModelConfig (populated from models.yaml).
This is a configuration estimate, not a real measurement.  No provider
calls are made.

==================================================
TASK SUITABILITY SCORE
==================================================

Mapped from TaskType to the relevant ModelConfig score field:

  REASONING / GENERAL_QA / DATA_ANALYSIS → reasoning_score
  CODE_GENERATION / CODE_DEBUGGING       → coding_score
  SUMMARIZATION                          → summarization_score
  STRUCTURED_GENERATION / CLASSIFICATION → extraction_score
  TRANSLATION / CREATIVE_WRITING / UNKNOWN → quality_score  (general fallback)

==================================================
COMPLEXITY COMPATIBILITY SCORE
==================================================

Maps (complexity, model_tier) → [0.0, 1.0]:

  SIMPLE   + any tier   → 1.0  (all passed filter; equally compatible)
  MODERATE + tier_2     → 0.80
  MODERATE + tier_3     → 1.0  (over-spec is OK but tier_2 is ideal)
  COMPLEX  + tier_3     → 1.0  (only tier possible; all score equally)

==================================================
TIE-BREAKING
==================================================

If two candidates have identical routing_score (to full float64 precision),
the model with the lexicographically smaller model_id is ranked first.
This is stable and deterministic.

==================================================
NO CANDIDATES
==================================================

If Part 3 returns zero candidates, RoutingScoreResult.selected_model is None
and scored_candidates is empty.  The caller must handle this case; Part 4
never fabricates a candidate.

==================================================
WEIGHTS VALIDATION
==================================================

ScoringWeights raises ValueError if:
  - any weight is negative
  - all weights are zero (sum == 0.0)

Invalid weight names (not in the known factor set) are also rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.classification.complexity_classifier import ComplexityLabel
from app.cost.calculator import CostCalculator
from app.models.config import ModelConfig, QualityTier
from app.routing.analyzer import RequestAnalysis, TaskType
from app.routing.candidate_selector import CandidateSelectionResult


# ── Constants ─────────────────────────────────────────────────────────────────

# Fallback output-token estimate used when no real response is available yet.
# This is explicitly a configuration constant, not a measured value.
_DEFAULT_OUTPUT_TOKENS: int = 256

# Neutral score when min == max (no differentiation possible)
_NEUTRAL_NORMALISED_SCORE: float = 0.5


# ── Task type → score field mapping ──────────────────────────────────────────

_TASK_SCORE_MAP: dict[TaskType, str] = {
    TaskType.REASONING:            "reasoning_score",
    TaskType.GENERAL_QA:           "reasoning_score",
    TaskType.DATA_ANALYSIS:        "reasoning_score",
    TaskType.CODE_GENERATION:      "coding_score",
    TaskType.CODE_DEBUGGING:       "coding_score",
    TaskType.SUMMARIZATION:        "summarization_score",
    TaskType.STRUCTURED_GENERATION:"extraction_score",
    TaskType.CLASSIFICATION:       "extraction_score",
    TaskType.TRANSLATION:          "quality_score",
    TaskType.CREATIVE_WRITING:     "quality_score",
    TaskType.UNKNOWN:              "quality_score",
}


# ── Complexity × tier → compatibility score ───────────────────────────────────

_COMPLEXITY_TIER_SCORE: dict[tuple[ComplexityLabel, QualityTier], float] = {
    # SIMPLE: any tier is equally compatible (all passed the filter)
    (ComplexityLabel.SIMPLE,   QualityTier.TIER_1): 1.0,
    (ComplexityLabel.SIMPLE,   QualityTier.TIER_2): 1.0,
    (ComplexityLabel.SIMPLE,   QualityTier.TIER_3): 1.0,
    # MODERATE: tier_2 is the ideal match; tier_3 is capable but over-spec
    (ComplexityLabel.MODERATE, QualityTier.TIER_2): 1.0,
    (ComplexityLabel.MODERATE, QualityTier.TIER_3): 0.80,
    # COMPLEX: only tier_3 passes the filter; all equally compatible
    (ComplexityLabel.COMPLEX,  QualityTier.TIER_3): 1.0,
}


# ── Scoring weights ───────────────────────────────────────────────────────────

_VALID_WEIGHT_NAMES = frozenset({"cost", "latency", "quality", "task", "complexity"})


@dataclass
class ScoringWeights:
    """
    Configurable weights for the routing score formula.

    All weights must be non-negative and their sum must be > 0.

    Default values are chosen to balance cost-efficiency and quality:
        cost:       0.25 — cost matters but is not the only concern
        latency:    0.20 — speed matters for interactive use
        quality:    0.25 — general output quality
        task:       0.20 — specialised task suitability
        complexity: 0.10 — tier compatibility bonus
    """
    cost:       float = 0.25
    latency:    float = 0.20
    quality:    float = 0.25
    task:       float = 0.20
    complexity: float = 0.10

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        values = {
            "cost":       self.cost,
            "latency":    self.latency,
            "quality":    self.quality,
            "task":       self.task,
            "complexity": self.complexity,
        }
        for name, val in values.items():
            if val < 0.0:
                raise ValueError(
                    f"ScoringWeights.{name} must be >= 0.0, got {val}"
                )
        if sum(values.values()) == 0.0:
            raise ValueError(
                "At least one ScoringWeights factor must be > 0.0 "
                "(all weights are zero)."
            )

    @property
    def total(self) -> float:
        return self.cost + self.latency + self.quality + self.task + self.complexity

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "ScoringWeights":
        """
        Build ScoringWeights from a dict.

        Raises:
            ValueError: If any key is not a known factor name.
        """
        unknown = set(d.keys()) - _VALID_WEIGHT_NAMES
        if unknown:
            raise ValueError(
                f"Unknown scoring weight factor(s): {sorted(unknown)}. "
                f"Valid factors: {sorted(_VALID_WEIGHT_NAMES)}"
            )
        return cls(**{k: float(v) for k, v in d.items()})


# ── Per-candidate result ──────────────────────────────────────────────────────

@dataclass
class CandidateScore:
    """
    Full scoring breakdown for one candidate model.

    All factor_scores values are in [0.0, 1.0] (higher = better).
    routing_score is the weighted average of factor scores.
    rank is 1-based (1 = best).
    """
    model: ModelConfig
    factor_scores: dict[str, float]          # {"cost": 0.9, "latency": 0.7, ...}
    estimated_cost_usd: float                # Raw estimated cost (for transparency)
    latency_ms: float                        # Raw latency value used for scoring
    routing_score: float                     # Final weighted score in [0.0, 1.0]
    rank: int = 0                            # Set after ranking all candidates

    @property
    def model_id(self) -> str:
        return self.model.model_id

    def explain(self) -> str:
        """Return a human-readable explanation of this candidate's score."""
        factors = " | ".join(
            f"{k}={v:.3f}" for k, v in self.factor_scores.items()
        )
        return (
            f"[rank={self.rank}] {self.model_id} "
            f"-> score={self.routing_score:.4f} "
            f"({factors}) "
            f"| est_cost=${self.estimated_cost_usd:.6f} "
            f"| latency={self.latency_ms:.0f}ms"
        )


# ── Routing result ────────────────────────────────────────────────────────────

@dataclass
class RoutingScoreResult:
    """
    Full output of the RoutingScorer.

    Consumed downstream by:
        - Intelligent Router (Part 5)

    Fields:
        scored_candidates: All candidates scored and ranked (rank 1 = best).
        selected_model:    The rank-1 model, or None if no candidates.
        weights_used:      ScoringWeights used for this decision (for tracing).
        request_id:        Traces back to the originating request.
        complexity:        The complexity label that drove tier filtering.
    """
    scored_candidates: list[CandidateScore]
    selected_model: Optional[ModelConfig]
    weights_used: ScoringWeights
    request_id: str
    complexity: ComplexityLabel

    @property
    def has_selection(self) -> bool:
        return self.selected_model is not None

    @property
    def best(self) -> Optional[CandidateScore]:
        """Return the rank-1 CandidateScore, or None if empty."""
        if self.scored_candidates:
            return self.scored_candidates[0]
        return None

    def explain(self) -> str:
        """Human-readable summary of the routing decision."""
        if not self.has_selection:
            return (
                f"[{self.request_id}] No eligible candidates found "
                f"(complexity={self.complexity.value})."
            )
        lines = [
            f"[{self.request_id}] Routing decision "
            f"(complexity={self.complexity.value}, "
            f"selected={self.selected_model.model_id}):",
        ]
        for cs in self.scored_candidates:
            lines.append(f"  {cs.explain()}")
        return "\n".join(lines)


# ── Routing Scorer ────────────────────────────────────────────────────────────

class RoutingScorer:
    """
    Stateless, deterministic routing scorer.

    Accepts Part 3's CandidateSelectionResult and produces a ranked
    RoutingScoreResult.  No network calls, no API keys, no randomness.

    Usage::

        scorer = RoutingScorer()
        result = scorer.score(candidate_result, analysis)

        if result.has_selection:
            model = result.selected_model
        else:
            # no candidates — handle gracefully

    Thread-safe (stateless).
    """

    def __init__(self, weights: Optional[ScoringWeights] = None) -> None:
        """
        Args:
            weights: ScoringWeights to use.  Defaults to ScoringWeights().
        """
        self._weights = weights or ScoringWeights()

    @property
    def weights(self) -> ScoringWeights:
        return self._weights

    def score(
        self,
        candidate_result: CandidateSelectionResult,
        analysis: RequestAnalysis,
    ) -> RoutingScoreResult:
        """
        Score and rank all candidates in the selection result.

        Args:
            candidate_result: Output of CandidateSelector.select().
            analysis:         Output of RequestAnalyzer.analyze().

        Returns:
            RoutingScoreResult with scored, ranked candidates and selected model.
        """
        candidates = candidate_result.candidates
        complexity  = candidate_result.complexity

        if not candidates:
            return RoutingScoreResult(
                scored_candidates=[],
                selected_model=None,
                weights_used=self._weights,
                request_id=analysis.request_id,
                complexity=complexity,
            )

        # ── Step 1: compute raw values for all candidates ──────────────────
        raw_costs    = [_estimate_cost(m, analysis) for m in candidates]
        raw_latencies = [m.average_latency_ms for m in candidates]

        # ── Step 2: normalise cost and latency (min-max, inverted) ─────────
        cost_scores    = _invert_normalise(raw_costs)
        latency_scores = _invert_normalise(raw_latencies)

        # ── Step 3: compute per-candidate scores ───────────────────────────
        scored: list[CandidateScore] = []
        for i, model in enumerate(candidates):
            quality_score  = model.quality_score
            task_score     = _task_score(model, analysis.task_type)
            complexity_scr = _complexity_score(model, complexity)

            factors = {
                "cost":       cost_scores[i],
                "latency":    latency_scores[i],
                "quality":    quality_score,
                "task":       task_score,
                "complexity": complexity_scr,
            }

            routing_score = _weighted_score(factors, self._weights)

            scored.append(CandidateScore(
                model=model,
                factor_scores=factors,
                estimated_cost_usd=raw_costs[i],
                latency_ms=raw_latencies[i],
                routing_score=routing_score,
            ))

        # ── Step 4: rank (descending score; model_id as tie-breaker) ──────
        scored.sort(key=lambda cs: (-cs.routing_score, cs.model_id))
        for i, cs in enumerate(scored):
            cs.rank = i + 1

        return RoutingScoreResult(
            scored_candidates=scored,
            selected_model=scored[0].model,
            weights_used=self._weights,
            request_id=analysis.request_id,
            complexity=complexity,
        )


# ── Internal scoring utilities ────────────────────────────────────────────────

def _estimate_cost(model: ModelConfig, analysis: RequestAnalysis) -> float:
    """
    Deterministic cost estimate using CostCalculator.

    Input tokens  = analysis.approx_token_count
    Output tokens = _DEFAULT_OUTPUT_TOKENS (256, fixed constant)

    This is an estimate before the response is generated; it is NOT a
    measured cost and is clearly labelled as such throughout.
    """
    breakdown = CostCalculator.calculate(
        model=model,
        input_tokens=analysis.approx_token_count,
        output_tokens=_DEFAULT_OUTPUT_TOKENS,
    )
    return breakdown.total_cost


def _invert_normalise(values: list[float]) -> list[float]:
    """
    Min-max normalise and invert a list of values so that lower raw
    values produce higher scores (higher is better).

    When min == max (all identical or single candidate), every element
    returns _NEUTRAL_NORMALISED_SCORE (0.5) — no artificial differentiation.

    Returns:
        List of floats in [0.0, 1.0], same length as input.
    """
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [_NEUTRAL_NORMALISED_SCORE] * len(values)
    return [1.0 - (v - min_v) / (max_v - min_v) for v in values]


def _task_score(model: ModelConfig, task_type: TaskType) -> float:
    """Return the model score field most relevant to the detected task type."""
    field_name = _TASK_SCORE_MAP.get(task_type, "quality_score")
    return float(getattr(model, field_name))


def _complexity_score(model: ModelConfig, complexity: ComplexityLabel) -> float:
    """Return the complexity-tier compatibility score for this (model, complexity) pair."""
    return _COMPLEXITY_TIER_SCORE.get(
        (complexity, model.quality_tier),
        1.0,   # default: compatible (should not occur after Part 3 filtering)
    )


def _weighted_score(factors: dict[str, float], weights: ScoringWeights) -> float:
    """
    Compute the weighted average score.

    routing_score = sum(weight_i * score_i) / sum(weights)

    Dividing by total weight normalises correctly when weights don't sum to 1.
    """
    total_weight = weights.total
    if total_weight == 0.0:
        return 0.0
    return (
        weights.cost       * factors["cost"]       +
        weights.latency    * factors["latency"]     +
        weights.quality    * factors["quality"]     +
        weights.task       * factors["task"]        +
        weights.complexity * factors["complexity"]
    ) / total_weight
