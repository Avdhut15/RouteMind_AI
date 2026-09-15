"""
tests/test_routing_scorer.py
─────────────────────────────
Unit tests for Phase 2 Part 4 — Routing Score & Model Selection.

All tests are:
    - Offline (no API calls, no network)
    - Deterministic (fixed inputs, no randomness)
    - Independent of OpenRouter, Ollama, or any live service

Test structure uses small in-memory ModelConfig fixtures and minimal
RequestAnalysis constructions so tests run in isolation from config/models.yaml.

Coverage:
    1.  Individual factor score calculation
    2.  Cost normalisation/scoring
    3.  Latency normalisation/scoring
    4.  Task-suitability scoring
    5.  Complexity compatibility scoring
    6.  Weighted final score calculation
    7.  Candidate ranking order
    8.  Correct best-model selection
    9.  Deterministic tie-breaking
    10. Multiple candidates
    11. Single candidate
    12. No candidates (empty CandidateSelectionResult)
    13. Invalid ScoringWeights (negative, all-zero, unknown keys)
    14. Zero-cost models
    15. Missing/unavailable latency (all same)
    16. Deterministic repeated scoring
    17. Explainability / explain() output
    18. ScoringWeights validation
    19. ScoringWeights.from_dict
    20. RoutingScoreResult structure
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from app.classification.complexity_classifier import ComplexityLabel, ComplexityPrediction
from app.cost.calculator import CostCalculator
from app.models.config import ModelConfig, QualityTier
from app.models.registry import ModelRegistry
from app.providers.base import LLMRequest
from app.routing.analyzer import RequestAnalysis, RequestAnalyzer, ResponseLengthCategory, TaskType
from app.routing.candidate_selector import CandidateSelectionResult, CandidateSelector
from app.routing.scoring import (
    CandidateScore,
    RoutingScoreResult,
    RoutingScorer,
    ScoringWeights,
    _complexity_score,
    _estimate_cost,
    _invert_normalise,
    _task_score,
    _weighted_score,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_ANALYZER = RequestAnalyzer()


def _make_model(
    model_id: str = "test/model",
    provider: str = "openrouter",
    quality_tier: QualityTier = QualityTier.TIER_2,
    quality_score: float = 0.70,
    reasoning_score: float = 0.70,
    coding_score: float = 0.65,
    summarization_score: float = 0.72,
    extraction_score: float = 0.70,
    input_cost_per_1k: float = 0.001,
    output_cost_per_1k: float = 0.002,
    average_latency_ms: float = 2000.0,
    context_window: int = 16000,
    supports_structured_output: bool = True,
    supports_function_calling: bool = True,
    enabled: bool = True,
) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider=provider,
        display_name=f"Test ({model_id})",
        input_cost_per_1k_tokens=input_cost_per_1k,
        output_cost_per_1k_tokens=output_cost_per_1k,
        average_latency_ms=average_latency_ms,
        quality_tier=quality_tier,
        quality_score=quality_score,
        reasoning_score=reasoning_score,
        coding_score=coding_score,
        summarization_score=summarization_score,
        extraction_score=extraction_score,
        context_window=context_window,
        supports_structured_output=supports_structured_output,
        supports_function_calling=supports_function_calling,
        enabled=enabled,
    )


def _make_analysis(prompt: str = "What is 2 + 2?", max_tokens: int = 1024) -> RequestAnalysis:
    req = LLMRequest(prompt=prompt, model_id="__test__", max_tokens=max_tokens)
    return _ANALYZER.analyze(req)


def _make_prediction(complexity: ComplexityLabel, confidence: float = 0.9) -> ComplexityPrediction:
    prob = {lbl.value: 0.0 for lbl in ComplexityLabel}
    prob[complexity.value] = confidence
    remaining = (1.0 - confidence) / 2
    for k in prob:
        if k != complexity.value:
            prob[k] = remaining
    return ComplexityPrediction(
        complexity=complexity,
        confidence=confidence,
        probabilities=prob,
        model_id="test_clf",
    )


def _make_candidate_result(
    models: list[ModelConfig],
    analysis: RequestAnalysis,
    complexity: ComplexityLabel,
) -> CandidateSelectionResult:
    """Build a CandidateSelectionResult directly from a model list."""
    return CandidateSelectionResult(
        candidates=sorted(models, key=lambda m: m.model_id),
        rejected=[],
        request_analysis=analysis,
        complexity=complexity,
        applied_filters=["test"],
    )


# ── 1. ScoringWeights validation ──────────────────────────────────────────────

class TestScoringWeights:

    def test_default_weights_valid(self):
        w = ScoringWeights()
        assert w.cost == 0.25
        assert w.latency == 0.20
        assert w.quality == 0.25
        assert w.task == 0.20
        assert w.complexity == 0.10

    def test_default_weights_sum_to_one(self):
        w = ScoringWeights()
        assert abs(w.total - 1.0) < 1e-9

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="cost"):
            ScoringWeights(cost=-0.1)

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError, match="latency"):
            ScoringWeights(latency=-0.5)

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError, match="all weights are zero"):
            ScoringWeights(cost=0.0, latency=0.0, quality=0.0, task=0.0, complexity=0.0)

    def test_single_nonzero_weight_valid(self):
        w = ScoringWeights(cost=1.0, latency=0.0, quality=0.0, task=0.0, complexity=0.0)
        assert w.total == 1.0

    def test_from_dict_valid(self):
        d = {"cost": 0.5, "quality": 0.5}
        w = ScoringWeights.from_dict(d)
        assert w.cost == 0.5
        assert w.quality == 0.5

    def test_from_dict_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            ScoringWeights.from_dict({"cost": 0.5, "magic": 0.5})

    def test_custom_weights_valid(self):
        w = ScoringWeights(cost=0.5, latency=0.1, quality=0.2, task=0.1, complexity=0.1)
        assert abs(w.total - 1.0) < 1e-9


# ── 2. _invert_normalise ──────────────────────────────────────────────────────

class TestInvertNormalise:

    def test_empty_returns_empty(self):
        assert _invert_normalise([]) == []

    def test_single_value_returns_neutral(self):
        result = _invert_normalise([100.0])
        assert result == [0.5]

    def test_all_same_values_returns_neutral(self):
        result = _invert_normalise([50.0, 50.0, 50.0])
        assert result == [0.5, 0.5, 0.5]

    def test_min_value_scores_highest(self):
        result = _invert_normalise([1.0, 5.0, 10.0])
        assert result[0] > result[1] > result[2]

    def test_max_value_scores_zero(self):
        result = _invert_normalise([1.0, 5.0, 10.0])
        assert abs(result[2] - 0.0) < 1e-9

    def test_min_value_scores_one(self):
        result = _invert_normalise([1.0, 5.0, 10.0])
        assert abs(result[0] - 1.0) < 1e-9

    def test_two_values(self):
        result = _invert_normalise([100.0, 200.0])
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[1] - 0.0) < 1e-9

    def test_scores_in_zero_one_range(self):
        for val in _invert_normalise([3.0, 7.0, 15.0, 100.0]):
            assert 0.0 <= val <= 1.0


# ── 3. _task_score ────────────────────────────────────────────────────────────

class TestTaskScore:

    def test_reasoning_uses_reasoning_score(self):
        model = _make_model(reasoning_score=0.88)
        assert _task_score(model, TaskType.REASONING) == pytest.approx(0.88)

    def test_general_qa_uses_reasoning_score(self):
        model = _make_model(reasoning_score=0.75)
        assert _task_score(model, TaskType.GENERAL_QA) == pytest.approx(0.75)

    def test_data_analysis_uses_reasoning_score(self):
        model = _make_model(reasoning_score=0.80)
        assert _task_score(model, TaskType.DATA_ANALYSIS) == pytest.approx(0.80)

    def test_code_generation_uses_coding_score(self):
        model = _make_model(coding_score=0.91)
        assert _task_score(model, TaskType.CODE_GENERATION) == pytest.approx(0.91)

    def test_code_debugging_uses_coding_score(self):
        model = _make_model(coding_score=0.85)
        assert _task_score(model, TaskType.CODE_DEBUGGING) == pytest.approx(0.85)

    def test_summarization_uses_summarization_score(self):
        model = _make_model(summarization_score=0.77)
        assert _task_score(model, TaskType.SUMMARIZATION) == pytest.approx(0.77)

    def test_structured_generation_uses_extraction_score(self):
        model = _make_model(extraction_score=0.83)
        assert _task_score(model, TaskType.STRUCTURED_GENERATION) == pytest.approx(0.83)

    def test_classification_uses_extraction_score(self):
        model = _make_model(extraction_score=0.79)
        assert _task_score(model, TaskType.CLASSIFICATION) == pytest.approx(0.79)

    def test_translation_uses_quality_score(self):
        model = _make_model(quality_score=0.70)
        assert _task_score(model, TaskType.TRANSLATION) == pytest.approx(0.70)

    def test_creative_writing_uses_quality_score(self):
        model = _make_model(quality_score=0.68)
        assert _task_score(model, TaskType.CREATIVE_WRITING) == pytest.approx(0.68)

    def test_unknown_uses_quality_score(self):
        model = _make_model(quality_score=0.60)
        assert _task_score(model, TaskType.UNKNOWN) == pytest.approx(0.60)


# ── 4. _complexity_score ──────────────────────────────────────────────────────

class TestComplexityScore:

    def test_simple_tier1(self):
        model = _make_model(quality_tier=QualityTier.TIER_1)
        assert _complexity_score(model, ComplexityLabel.SIMPLE) == pytest.approx(1.0)

    def test_simple_tier2(self):
        model = _make_model(quality_tier=QualityTier.TIER_2)
        assert _complexity_score(model, ComplexityLabel.SIMPLE) == pytest.approx(1.0)

    def test_simple_tier3(self):
        model = _make_model(quality_tier=QualityTier.TIER_3)
        assert _complexity_score(model, ComplexityLabel.SIMPLE) == pytest.approx(1.0)

    def test_moderate_tier2_is_ideal(self):
        model = _make_model(quality_tier=QualityTier.TIER_2)
        assert _complexity_score(model, ComplexityLabel.MODERATE) == pytest.approx(1.0)

    def test_moderate_tier3_is_over_spec(self):
        model = _make_model(quality_tier=QualityTier.TIER_3)
        score = _complexity_score(model, ComplexityLabel.MODERATE)
        assert score < 1.0
        assert score > 0.0

    def test_complex_tier3(self):
        model = _make_model(quality_tier=QualityTier.TIER_3)
        assert _complexity_score(model, ComplexityLabel.COMPLEX) == pytest.approx(1.0)


# ── 5. _estimate_cost ─────────────────────────────────────────────────────────

class TestEstimateCost:

    def test_zero_cost_model(self):
        model = _make_model(input_cost_per_1k=0.0, output_cost_per_1k=0.0)
        analysis = _make_analysis("hello")
        cost = _estimate_cost(model, analysis)
        assert cost == pytest.approx(0.0)

    def test_cost_is_positive_for_priced_model(self):
        model = _make_model(input_cost_per_1k=0.001, output_cost_per_1k=0.002)
        analysis = _make_analysis("A " * 200)  # reasonably long prompt
        cost = _estimate_cost(model, analysis)
        assert cost > 0.0

    def test_cost_uses_model_pricing(self):
        model_cheap = _make_model(model_id="test/cheap", input_cost_per_1k=0.0001, output_cost_per_1k=0.0002)
        model_expensive = _make_model(model_id="test/exp",   input_cost_per_1k=0.01,   output_cost_per_1k=0.02)
        analysis = _make_analysis("hello world")
        cost_cheap = _estimate_cost(model_cheap, analysis)
        cost_expensive = _estimate_cost(model_expensive, analysis)
        assert cost_cheap < cost_expensive

    def test_cost_is_deterministic(self):
        model = _make_model()
        analysis = _make_analysis("test prompt")
        c1 = _estimate_cost(model, analysis)
        c2 = _estimate_cost(model, analysis)
        assert c1 == c2


# ── 6. _weighted_score ────────────────────────────────────────────────────────

class TestWeightedScore:

    def test_all_ones_returns_one(self):
        factors = {"cost": 1.0, "latency": 1.0, "quality": 1.0, "task": 1.0, "complexity": 1.0}
        w = ScoringWeights()
        assert _weighted_score(factors, w) == pytest.approx(1.0)

    def test_all_zeros_returns_zero(self):
        factors = {"cost": 0.0, "latency": 0.0, "quality": 0.0, "task": 0.0, "complexity": 0.0}
        w = ScoringWeights()
        assert _weighted_score(factors, w) == pytest.approx(0.0)

    def test_single_factor_weight(self):
        factors = {"cost": 1.0, "latency": 0.0, "quality": 0.0, "task": 0.0, "complexity": 0.0}
        w = ScoringWeights(cost=1.0, latency=0.0, quality=0.0, task=0.0, complexity=0.0)
        assert _weighted_score(factors, w) == pytest.approx(1.0)

    def test_asymmetric_weights(self):
        factors = {"cost": 1.0, "latency": 0.0, "quality": 0.5, "task": 0.0, "complexity": 0.0}
        w = ScoringWeights(cost=0.8, latency=0.0, quality=0.2, task=0.0, complexity=0.0)
        expected = (0.8 * 1.0 + 0.2 * 0.5) / (0.8 + 0.2)
        assert _weighted_score(factors, w) == pytest.approx(expected)


# ── 7. RoutingScorer — no candidates ─────────────────────────────────────────

class TestNoCandidates:

    def test_empty_candidates_returns_result(self):
        analysis = _make_analysis()
        cr = CandidateSelectionResult(
            candidates=[],
            rejected=[],
            request_analysis=analysis,
            complexity=ComplexityLabel.SIMPLE,
            applied_filters=[],
        )
        scorer = RoutingScorer()
        result = scorer.score(cr, analysis)
        assert isinstance(result, RoutingScoreResult)

    def test_no_selection_when_empty(self):
        analysis = _make_analysis()
        cr = CandidateSelectionResult(
            candidates=[], rejected=[], request_analysis=analysis,
            complexity=ComplexityLabel.SIMPLE, applied_filters=[],
        )
        result = RoutingScorer().score(cr, analysis)
        assert result.has_selection is False
        assert result.selected_model is None

    def test_scored_candidates_empty(self):
        analysis = _make_analysis()
        cr = CandidateSelectionResult(
            candidates=[], rejected=[], request_analysis=analysis,
            complexity=ComplexityLabel.SIMPLE, applied_filters=[],
        )
        result = RoutingScorer().score(cr, analysis)
        assert result.scored_candidates == []

    def test_best_is_none_when_empty(self):
        analysis = _make_analysis()
        cr = CandidateSelectionResult(
            candidates=[], rejected=[], request_analysis=analysis,
            complexity=ComplexityLabel.SIMPLE, applied_filters=[],
        )
        result = RoutingScorer().score(cr, analysis)
        assert result.best is None

    def test_explain_mentions_no_candidates(self):
        analysis = _make_analysis()
        cr = CandidateSelectionResult(
            candidates=[], rejected=[], request_analysis=analysis,
            complexity=ComplexityLabel.SIMPLE, applied_filters=[],
        )
        result = RoutingScorer().score(cr, analysis)
        text = result.explain()
        assert "No eligible" in text


# ── 8. RoutingScorer — single candidate ──────────────────────────────────────

class TestSingleCandidate:

    def test_single_candidate_is_selected(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert result.selected_model.model_id == model.model_id

    def test_single_candidate_rank_is_one(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert result.scored_candidates[0].rank == 1

    def test_single_candidate_routing_score_in_range(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        score = result.scored_candidates[0].routing_score
        assert 0.0 <= score <= 1.0

    def test_single_candidate_factor_scores_have_all_keys(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        factors = result.scored_candidates[0].factor_scores
        assert set(factors.keys()) == {"cost", "latency", "quality", "task", "complexity"}

    def test_single_candidate_neutral_cost_score(self):
        """With only one candidate, cost normalisation gives 0.5."""
        model = _make_model(input_cost_per_1k=0.01, output_cost_per_1k=0.02)
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert result.scored_candidates[0].factor_scores["cost"] == pytest.approx(0.5)


# ── 9. RoutingScorer — multiple candidates ────────────────────────────────────

class TestMultipleCandidates:

    def test_returns_all_candidates_scored(self):
        m1 = _make_model(model_id="test/a")
        m2 = _make_model(model_id="test/b")
        m3 = _make_model(model_id="test/c")
        analysis = _make_analysis()
        cr = _make_candidate_result([m1, m2, m3], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert len(result.scored_candidates) == 3

    def test_ranks_are_sequential_from_one(self):
        m1 = _make_model(model_id="test/a")
        m2 = _make_model(model_id="test/b")
        analysis = _make_analysis()
        cr = _make_candidate_result([m1, m2], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        ranks = [cs.rank for cs in result.scored_candidates]
        assert sorted(ranks) == [1, 2]

    def test_selected_model_is_rank_one(self):
        m1 = _make_model(model_id="test/cheap", input_cost_per_1k=0.0001, output_cost_per_1k=0.0002)
        m2 = _make_model(model_id="test/expensive", input_cost_per_1k=0.01, output_cost_per_1k=0.02)
        analysis = _make_analysis()
        # Use cost-only weights
        w = ScoringWeights(cost=1.0, latency=0.0, quality=0.0, task=0.0, complexity=0.0)
        cr = _make_candidate_result([m1, m2], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer(weights=w).score(cr, analysis)
        assert result.selected_model.model_id == "test/cheap"

    def test_scores_descending(self):
        m1 = _make_model(model_id="test/a")
        m2 = _make_model(model_id="test/b")
        m3 = _make_model(model_id="test/c")
        analysis = _make_analysis()
        cr = _make_candidate_result([m1, m2, m3], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        scores = [cs.routing_score for cs in result.scored_candidates]
        assert scores == sorted(scores, reverse=True)


# ── 10. Ranking correctness ───────────────────────────────────────────────────

class TestRanking:

    def test_cheapest_model_wins_cost_only_weights(self):
        cheap = _make_model(model_id="test/cheap", input_cost_per_1k=0.0, output_cost_per_1k=0.0)
        exp = _make_model(model_id="test/exp",   input_cost_per_1k=0.1, output_cost_per_1k=0.2)
        analysis = _make_analysis("hello world from a test prompt")
        w = ScoringWeights(cost=1.0, latency=0.0, quality=0.0, task=0.0, complexity=0.0)
        cr = _make_candidate_result([cheap, exp], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer(weights=w).score(cr, analysis)
        assert result.selected_model.model_id == "test/cheap"

    def test_fastest_model_wins_latency_only_weights(self):
        fast = _make_model(model_id="test/fast",  average_latency_ms=500.0)
        slow = _make_model(model_id="test/slow", average_latency_ms=5000.0)
        analysis = _make_analysis()
        w = ScoringWeights(cost=0.0, latency=1.0, quality=0.0, task=0.0, complexity=0.0)
        cr = _make_candidate_result([fast, slow], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer(weights=w).score(cr, analysis)
        assert result.selected_model.model_id == "test/fast"

    def test_best_quality_wins_quality_only_weights(self):
        good = _make_model(model_id="test/good", quality_score=0.95)
        bad  = _make_model(model_id="test/bad",  quality_score=0.40)
        analysis = _make_analysis()
        w = ScoringWeights(cost=0.0, latency=0.0, quality=1.0, task=0.0, complexity=0.0)
        cr = _make_candidate_result([good, bad], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer(weights=w).score(cr, analysis)
        assert result.selected_model.model_id == "test/good"

    def test_best_task_score_wins_task_only_weights(self):
        specialist = _make_model(model_id="test/specialist", coding_score=0.95)
        generalist = _make_model(model_id="test/generalist", coding_score=0.55)
        analysis = _make_analysis("Write a function to sort a list.")  # → CODE_GENERATION
        w = ScoringWeights(cost=0.0, latency=0.0, quality=0.0, task=1.0, complexity=0.0)
        cr = _make_candidate_result([specialist, generalist], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer(weights=w).score(cr, analysis)
        assert result.selected_model.model_id == "test/specialist"

    def test_tier2_beats_tier3_on_moderate_complexity_only(self):
        """For MODERATE, tier_2 scores 1.0 vs tier_3 scores 0.8 on complexity."""
        tier2 = _make_model(model_id="test/t2", quality_tier=QualityTier.TIER_2)
        tier3 = _make_model(model_id="test/t3", quality_tier=QualityTier.TIER_3,
                            quality_score=0.70, reasoning_score=0.70, coding_score=0.65,
                            summarization_score=0.72, extraction_score=0.70)
        analysis = _make_analysis()
        w = ScoringWeights(cost=0.0, latency=0.0, quality=0.0, task=0.0, complexity=1.0)
        cr = _make_candidate_result([tier2, tier3], analysis, ComplexityLabel.MODERATE)
        result = RoutingScorer(weights=w).score(cr, analysis)
        assert result.selected_model.model_id == "test/t2"


# ── 11. Tie-breaking ─────────────────────────────────────────────────────────

class TestTieBreaking:

    def test_tie_broken_by_model_id_lexicographic(self):
        """If scores are identical, the lexicographically first model_id wins."""
        # Make two models with exactly the same specs → same score
        m_zzz = _make_model(model_id="test/zzz")
        m_aaa = _make_model(model_id="test/aaa")
        analysis = _make_analysis()
        cr = _make_candidate_result([m_zzz, m_aaa], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        # Both have identical cost (same pricing), latency, quality scores
        # Tie-break: 'test/aaa' < 'test/zzz'
        assert result.selected_model.model_id == "test/aaa"

    def test_tie_breaking_is_deterministic(self):
        m1 = _make_model(model_id="test/aaa")
        m2 = _make_model(model_id="test/zzz")
        analysis = _make_analysis()
        cr = _make_candidate_result([m1, m2], analysis, ComplexityLabel.SIMPLE)
        scorer = RoutingScorer()
        r1 = scorer.score(cr, analysis)
        r2 = scorer.score(cr, analysis)
        assert r1.selected_model.model_id == r2.selected_model.model_id


# ── 12. Determinism ───────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_inputs_same_result(self):
        m1 = _make_model(model_id="test/a", input_cost_per_1k=0.001)
        m2 = _make_model(model_id="test/b", input_cost_per_1k=0.005)
        analysis = _make_analysis("What is the capital of France?")
        cr = _make_candidate_result([m1, m2], analysis, ComplexityLabel.SIMPLE)
        scorer = RoutingScorer()
        r1 = scorer.score(cr, analysis)
        r2 = scorer.score(cr, analysis)
        assert r1.selected_model.model_id == r2.selected_model.model_id
        assert r1.scored_candidates[0].routing_score == r2.scored_candidates[0].routing_score

    def test_two_scorer_instances_same_result(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        r1 = RoutingScorer().score(cr, analysis)
        r2 = RoutingScorer().score(cr, analysis)
        assert r1.selected_model.model_id == r2.selected_model.model_id


# ── 13. Explainability ────────────────────────────────────────────────────────

class TestExplainability:

    def test_explain_returns_string(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert isinstance(result.explain(), str)

    def test_explain_contains_selected_model_id(self):
        model = _make_model(model_id="test/selected")
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert "test/selected" in result.explain()

    def test_candidate_score_explain_contains_rank(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        cs = result.scored_candidates[0]
        assert "rank=1" in cs.explain()

    def test_candidate_score_explain_contains_model_id(self):
        model = _make_model(model_id="test/explain-me")
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert "test/explain-me" in result.scored_candidates[0].explain()

    def test_candidate_score_explain_contains_factor_names(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        explanation = result.scored_candidates[0].explain()
        for name in ("cost", "latency", "quality", "task", "complexity"):
            assert name in explanation


# ── 14. RoutingScoreResult structure ──────────────────────────────────────────

class TestRoutingScoreResultStructure:

    def test_request_id_preserved(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert result.request_id == analysis.request_id

    def test_complexity_preserved(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.MODERATE)
        result = RoutingScorer().score(cr, analysis)
        assert result.complexity == ComplexityLabel.MODERATE

    def test_weights_used_preserved(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        w = ScoringWeights(cost=0.5, latency=0.5, quality=0.0, task=0.0, complexity=0.0)
        result = RoutingScorer(weights=w).score(cr, analysis)
        assert result.weights_used is w

    def test_candidate_score_has_all_fields(self):
        model = _make_model()
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        cs = result.scored_candidates[0]
        assert cs.model is not None
        assert isinstance(cs.factor_scores, dict)
        assert isinstance(cs.estimated_cost_usd, float)
        assert isinstance(cs.latency_ms, float)
        assert isinstance(cs.routing_score, float)
        assert cs.rank == 1

    def test_routing_score_in_range(self):
        m1 = _make_model(model_id="test/a")
        m2 = _make_model(model_id="test/b", input_cost_per_1k=0.01)
        analysis = _make_analysis()
        cr = _make_candidate_result([m1, m2], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        for cs in result.scored_candidates:
            assert 0.0 <= cs.routing_score <= 1.0

    def test_factor_scores_in_range(self):
        m1 = _make_model(model_id="test/a")
        m2 = _make_model(model_id="test/b")
        analysis = _make_analysis()
        cr = _make_candidate_result([m1, m2], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        for cs in result.scored_candidates:
            for name, val in cs.factor_scores.items():
                assert 0.0 <= val <= 1.0, f"{name}={val} out of range"


# ── 15. Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_cost_all_models(self):
        """When all models are free, cost_score should be neutral (0.5)."""
        m1 = _make_model(model_id="test/a", input_cost_per_1k=0.0, output_cost_per_1k=0.0)
        m2 = _make_model(model_id="test/b", input_cost_per_1k=0.0, output_cost_per_1k=0.0)
        analysis = _make_analysis()
        cr = _make_candidate_result([m1, m2], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        for cs in result.scored_candidates:
            assert cs.factor_scores["cost"] == pytest.approx(0.5)

    def test_same_latency_all_models_gives_neutral_score(self):
        m1 = _make_model(model_id="test/a", average_latency_ms=2000.0)
        m2 = _make_model(model_id="test/b", average_latency_ms=2000.0)
        analysis = _make_analysis()
        cr = _make_candidate_result([m1, m2], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        for cs in result.scored_candidates:
            assert cs.factor_scores["latency"] == pytest.approx(0.5)

    def test_estimated_cost_is_labelled(self):
        """Confirm the raw estimated_cost_usd field is accessible on CandidateScore."""
        model = _make_model(input_cost_per_1k=0.001, output_cost_per_1k=0.002)
        analysis = _make_analysis()
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        cs = result.scored_candidates[0]
        assert cs.estimated_cost_usd >= 0.0

    def test_empty_prompt_does_not_crash(self):
        model = _make_model()
        analysis = _make_analysis("")
        cr = _make_candidate_result([model], analysis, ComplexityLabel.SIMPLE)
        result = RoutingScorer().score(cr, analysis)
        assert isinstance(result, RoutingScoreResult)


# ── 16. Real registry integration ────────────────────────────────────────────

class TestRealRegistryIntegration:

    def test_end_to_end_no_crash(self):
        registry = ModelRegistry(pathlib.Path("config/models.yaml"))
        analyzer = RequestAnalyzer()
        req = LLMRequest(prompt="What is the capital of France?", model_id="test")
        analysis = analyzer.analyze(req)
        prob = {"simple": 0.9, "moderate": 0.05, "complex": 0.05}
        pred = ComplexityPrediction(
            complexity=ComplexityLabel.SIMPLE,
            confidence=0.9,
            probabilities=prob,
            model_id="test_clf",
        )
        selector = CandidateSelector(registry)
        cr = selector.select(analysis, pred)
        scorer = RoutingScorer()
        result = scorer.score(cr, analysis)
        assert isinstance(result, RoutingScoreResult)

    def test_selected_model_is_in_registry(self):
        registry = ModelRegistry(pathlib.Path("config/models.yaml"))
        analyzer = RequestAnalyzer()
        req = LLMRequest(prompt="What is the capital of France?", model_id="test")
        analysis = analyzer.analyze(req)
        prob = {"simple": 0.9, "moderate": 0.05, "complex": 0.05}
        pred = ComplexityPrediction(
            complexity=ComplexityLabel.SIMPLE,
            confidence=0.9,
            probabilities=prob,
            model_id="test_clf",
        )
        cr = CandidateSelector(registry).select(analysis, pred)
        result = RoutingScorer().score(cr, analysis)
        if result.has_selection:
            assert result.selected_model.model_id in registry._models
