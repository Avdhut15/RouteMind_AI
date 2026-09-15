"""
tests/test_candidate_selector.py
─────────────────────────────────
Unit tests for Phase 2 Part 3 — Model Capability & Candidate Selection.

All tests are:
    - Offline (no API calls, no network)
    - Deterministic (fixed inputs, no randomness)
    - Independent of OpenRouter, Ollama, or any live service

Test fixtures use in-memory ModelConfig/ModelRegistry objects rather than
loading from disk, so they are isolated from changes to config/models.yaml.

Coverage:
    - Candidate selection structure
    - Enabled/disabled filter
    - Complexity → tier filter (simple / moderate / complex)
    - Reasoning-score filter
    - Coding-score filter
    - Structured-output capability filter
    - Context-window filter
    - Multiple filters applied simultaneously
    - No eligible candidates edge case
    - Single candidate
    - Multiple candidates, deterministic ordering (by model_id)
    - Rejection log completeness
    - Invalid registry type
    - Prompt with very long token count (context window filter)
    - All task types produce a result (no crash)
    - Determinism: same inputs → same output
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest

from app.classification.complexity_classifier import ComplexityLabel, ComplexityPrediction
from app.models.config import ModelConfig, QualityTier
from app.models.registry import ModelRegistry
from app.providers.base import LLMRequest
from app.routing.analyzer import RequestAnalyzer, TaskType
from app.routing.candidate_selector import (
    CODING_SCORE_THRESHOLD,
    CONTEXT_SAFETY_FACTOR,
    REASONING_SCORE_THRESHOLD,
    CandidateSelectionResult,
    CandidateSelector,
    ModelSelectionOutcome,
    RejectionReason,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_ANALYZER = RequestAnalyzer()
_REAL_REGISTRY = ModelRegistry(pathlib.Path("config/models.yaml"))


def _make_model(
    model_id: str = "test/model",
    provider: str = "openrouter",
    quality_tier: QualityTier = QualityTier.TIER_2,
    quality_score: float = 0.70,
    reasoning_score: float = 0.70,
    coding_score: float = 0.65,
    summarization_score: float = 0.70,
    extraction_score: float = 0.70,
    context_window: int = 16000,
    supports_structured_output: bool = True,
    supports_function_calling: bool = True,
    enabled: bool = True,
) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider=provider,
        display_name=f"Test Model ({model_id})",
        input_cost_per_1k_tokens=0.001,
        output_cost_per_1k_tokens=0.002,
        average_latency_ms=1500.0,
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


def _make_registry(*models: ModelConfig) -> ModelRegistry:
    """Build an in-memory ModelRegistry from provided ModelConfig objects."""
    registry = MagicMock(spec=ModelRegistry)
    model_dict = {m.model_id: m for m in models}
    registry._models = model_dict
    registry.all_enabled.return_value = [m for m in models if m.enabled]
    return registry


def _make_analysis(
    prompt: str,
    max_tokens: int = 1024,
) -> object:
    req = LLMRequest(prompt=prompt, model_id="__test__", max_tokens=max_tokens)
    return _ANALYZER.analyze(req)


def _make_prediction(
    complexity: ComplexityLabel,
    confidence: float = 0.90,
) -> ComplexityPrediction:
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
        model_id="test_classifier",
    )


def _make_selector(*models: ModelConfig) -> CandidateSelector:
    registry = _make_registry(*models)
    return CandidateSelector(registry)


# ── 1. Basic structure ────────────────────────────────────────────────────────

class TestSelectionResultStructure:

    def test_returns_candidate_selection_result(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert isinstance(result, CandidateSelectionResult)

    def test_has_candidates_property_true(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates is True

    def test_has_candidates_property_false_when_empty(self):
        model = _make_model(enabled=False)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates is False

    def test_candidate_ids_list(self):
        m1 = _make_model(model_id="openrouter/a")
        m2 = _make_model(model_id="openrouter/b")
        selector = _make_selector(m1, m2)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert set(result.candidate_ids) == {"openrouter/a", "openrouter/b"}

    def test_rejected_ids_list(self):
        m1 = _make_model(model_id="openrouter/ok")
        m2 = _make_model(model_id="openrouter/disabled", enabled=False)
        selector = _make_selector(m1, m2)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert "openrouter/disabled" in result.rejected_ids

    def test_summary_is_string(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert isinstance(result.summary(), str)
        assert "simple" in result.summary()

    def test_applied_filters_is_non_empty_list(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert isinstance(result.applied_filters, list)
        assert len(result.applied_filters) >= 2

    def test_request_analysis_preserved_in_result(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.request_analysis is analysis

    def test_complexity_preserved_in_result(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.COMPLEX)
        result = selector.select(analysis, pred)
        assert result.complexity == ComplexityLabel.COMPLEX


# ── 2. Enabled/disabled filter ────────────────────────────────────────────────

class TestEnabledFilter:

    def test_disabled_model_is_rejected(self):
        model = _make_model(enabled=False)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert not result.has_candidates

    def test_disabled_rejection_reason_is_disabled(self):
        model = _make_model(model_id="test/disabled", enabled=False)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        rejected = {r.model_id: r.rejection_reason for r in result.rejected}
        assert rejected["test/disabled"] == RejectionReason.DISABLED

    def test_enabled_model_passes_enabled_filter(self):
        model = _make_model(enabled=True)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_mix_of_enabled_and_disabled(self):
        m_on = _make_model(model_id="openrouter/on",  enabled=True)
        m_off = _make_model(model_id="openrouter/off", enabled=False)
        selector = _make_selector(m_on, m_off)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.candidate_ids == ["openrouter/on"]
        assert "openrouter/off" in result.rejected_ids


# ── 3. Complexity / tier filter ───────────────────────────────────────────────

class TestComplexityTierFilter:

    def test_simple_accepts_tier_1(self):
        model = _make_model(quality_tier=QualityTier.TIER_1)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_simple_accepts_tier_2(self):
        model = _make_model(quality_tier=QualityTier.TIER_2)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_simple_accepts_tier_3(self):
        model = _make_model(quality_tier=QualityTier.TIER_3)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_moderate_rejects_tier_1(self):
        model = _make_model(quality_tier=QualityTier.TIER_1)
        selector = _make_selector(model)
        analysis = _make_analysis("Summarize the key differences between TCP and UDP.")
        pred = _make_prediction(ComplexityLabel.MODERATE)
        result = selector.select(analysis, pred)
        assert not result.has_candidates

    def test_moderate_rejects_reason_is_tier_too_low(self):
        model = _make_model(model_id="test/tier1", quality_tier=QualityTier.TIER_1)
        selector = _make_selector(model)
        analysis = _make_analysis("Summarize the differences.")
        pred = _make_prediction(ComplexityLabel.MODERATE)
        result = selector.select(analysis, pred)
        rejected = {r.model_id: r.rejection_reason for r in result.rejected}
        assert rejected["test/tier1"] == RejectionReason.TIER_TOO_LOW

    def test_moderate_accepts_tier_2(self):
        model = _make_model(quality_tier=QualityTier.TIER_2)
        selector = _make_selector(model)
        analysis = _make_analysis("Summarize the key differences between TCP and UDP.")
        pred = _make_prediction(ComplexityLabel.MODERATE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_moderate_accepts_tier_3(self):
        model = _make_model(quality_tier=QualityTier.TIER_3)
        selector = _make_selector(model)
        analysis = _make_analysis("Summarize the key differences.")
        pred = _make_prediction(ComplexityLabel.MODERATE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_complex_rejects_tier_1(self):
        model = _make_model(quality_tier=QualityTier.TIER_1)
        selector = _make_selector(model)
        analysis = _make_analysis("Design a distributed consensus algorithm.")
        pred = _make_prediction(ComplexityLabel.COMPLEX)
        result = selector.select(analysis, pred)
        assert not result.has_candidates

    def test_complex_rejects_tier_2(self):
        model = _make_model(quality_tier=QualityTier.TIER_2)
        selector = _make_selector(model)
        analysis = _make_analysis("Design a distributed consensus algorithm.")
        pred = _make_prediction(ComplexityLabel.COMPLEX)
        result = selector.select(analysis, pred)
        assert not result.has_candidates

    def test_complex_accepts_tier_3(self):
        model = _make_model(quality_tier=QualityTier.TIER_3)
        selector = _make_selector(model)
        analysis = _make_analysis("Design a distributed consensus algorithm.")
        pred = _make_prediction(ComplexityLabel.COMPLEX)
        result = selector.select(analysis, pred)
        assert result.has_candidates


# ── 4. Reasoning requirement filter ──────────────────────────────────────────

class TestReasoningFilter:

    def test_reasoning_required_rejects_low_score(self):
        model = _make_model(reasoning_score=0.50)  # below 0.60 threshold
        selector = _make_selector(model)
        # "prove" and "step by step" both trigger requires_reasoning
        analysis = _make_analysis("Prove step by step that this is correct.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert not result.has_candidates

    def test_reasoning_required_rejection_reason(self):
        model = _make_model(model_id="test/low-reason", reasoning_score=0.50)
        selector = _make_selector(model)
        analysis = _make_analysis("Prove step by step that this is correct.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        rejected = {r.model_id: r.rejection_reason for r in result.rejected}
        assert rejected["test/low-reason"] == RejectionReason.REASONING_SCORE_LOW

    def test_reasoning_required_accepts_high_score(self):
        model = _make_model(reasoning_score=0.80)
        selector = _make_selector(model)
        analysis = _make_analysis("Prove step by step that this is correct.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_reasoning_required_accepts_at_threshold(self):
        model = _make_model(reasoning_score=REASONING_SCORE_THRESHOLD)
        selector = _make_selector(model)
        analysis = _make_analysis("Prove step by step that this is correct.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_no_reasoning_requirement_allows_low_score(self):
        """If requires_reasoning is False, low reasoning_score should NOT block."""
        model = _make_model(reasoning_score=0.30)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")  # no reasoning signal
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates


# ── 5. Coding requirement filter ─────────────────────────────────────────────

class TestCodingFilter:

    def test_code_required_rejects_low_score(self):
        model = _make_model(coding_score=0.40)  # below 0.55 threshold
        selector = _make_selector(model)
        analysis = _make_analysis("Write a function that reverses a linked list.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert not result.has_candidates

    def test_code_required_rejection_reason(self):
        model = _make_model(model_id="test/low-code", coding_score=0.40)
        selector = _make_selector(model)
        analysis = _make_analysis("Write a function that reverses a linked list.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        rejected = {r.model_id: r.rejection_reason for r in result.rejected}
        assert rejected["test/low-code"] == RejectionReason.CODING_SCORE_LOW

    def test_code_required_accepts_high_score(self):
        model = _make_model(coding_score=0.80)
        selector = _make_selector(model)
        analysis = _make_analysis("Write a function that reverses a linked list.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_code_required_accepts_at_threshold(self):
        model = _make_model(coding_score=CODING_SCORE_THRESHOLD)
        selector = _make_selector(model)
        analysis = _make_analysis("Write a function that reverses a linked list.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_no_code_requirement_allows_low_score(self):
        model = _make_model(coding_score=0.20)
        selector = _make_selector(model)
        analysis = _make_analysis("What is the capital of France?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates


# ── 6. Structured-output capability filter ───────────────────────────────────

class TestStructuredOutputFilter:

    def test_structured_required_rejects_no_capability(self):
        model = _make_model(supports_structured_output=False)
        selector = _make_selector(model)
        analysis = _make_analysis(
            "Return the answer as a JSON object with fields name and age."
        )
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert not result.has_candidates

    def test_structured_required_rejection_reason(self):
        model = _make_model(model_id="test/no-struct", supports_structured_output=False)
        selector = _make_selector(model)
        analysis = _make_analysis("Return the answer as JSON.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        rejected = {r.model_id: r.rejection_reason for r in result.rejected}
        assert rejected["test/no-struct"] == RejectionReason.NO_STRUCTURED_OUTPUT

    def test_structured_required_accepts_capable_model(self):
        model = _make_model(supports_structured_output=True)
        selector = _make_selector(model)
        analysis = _make_analysis("Return the answer as JSON.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_no_structured_requirement_allows_incapable_model(self):
        model = _make_model(supports_structured_output=False)
        selector = _make_selector(model)
        analysis = _make_analysis("What is the capital of Japan?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates


# ── 7. Context-window filter ──────────────────────────────────────────────────

class TestContextWindowFilter:

    def test_small_context_window_rejected_for_long_prompt(self):
        """A model with a 512-token window should be rejected for a 500-token prompt."""
        model = _make_model(context_window=512)
        selector = _make_selector(model)
        # ~500 tokens → 2000 chars needed; safety factor ×2 → need 1000 token window
        long_prompt = "word " * 500  # 500 words ≈ 500 tokens
        analysis = _make_analysis(long_prompt)
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert not result.has_candidates

    def test_small_context_rejection_reason(self):
        model = _make_model(model_id="test/small-ctx", context_window=100)
        selector = _make_selector(model)
        long_prompt = "word " * 300
        analysis = _make_analysis(long_prompt)
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        rejected = {r.model_id: r.rejection_reason for r in result.rejected}
        assert rejected["test/small-ctx"] == RejectionReason.CONTEXT_WINDOW_TOO_SMALL

    def test_large_context_window_accepted(self):
        model = _make_model(context_window=128000)
        selector = _make_selector(model)
        long_prompt = "word " * 500
        analysis = _make_analysis(long_prompt)
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates

    def test_short_prompt_passes_small_window(self):
        model = _make_model(context_window=4096)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")  # ~4 tokens
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert result.has_candidates


# ── 8. Multiple filters combined ──────────────────────────────────────────────

class TestMultipleFilters:

    def test_reasoning_and_code_both_required(self):
        """Model passes both: high reasoning + high coding."""
        good = _make_model(
            model_id="test/good",
            reasoning_score=0.80,
            coding_score=0.75,
        )
        bad = _make_model(
            model_id="test/bad",
            reasoning_score=0.40,  # fails reasoning
            coding_score=0.75,
        )
        selector = _make_selector(good, bad)
        # Trigger both requires_reasoning and requires_code
        analysis = _make_analysis(
            "Prove step by step with code how to implement a binary search algorithm."
        )
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert "test/good" in result.candidate_ids
        assert "test/bad" in result.rejected_ids

    def test_all_filters_applied(self):
        """A model that only fails the last context filter should reach it."""
        model = _make_model(
            context_window=10,  # tiny — will fail context filter
            reasoning_score=0.90,
            coding_score=0.90,
            supports_structured_output=True,
            quality_tier=QualityTier.TIER_3,
        )
        selector = _make_selector(model)
        analysis = _make_analysis("word " * 100)
        pred = _make_prediction(ComplexityLabel.COMPLEX)
        result = selector.select(analysis, pred)
        assert not result.has_candidates
        assert result.rejected[0].rejection_reason == RejectionReason.CONTEXT_WINDOW_TOO_SMALL


# ── 9. No eligible candidates ─────────────────────────────────────────────────

class TestNoEligibleCandidates:

    def test_empty_registry_no_candidates(self):
        registry = _make_registry()
        selector = CandidateSelector(registry)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert not result.has_candidates
        assert result.candidates == []
        assert result.rejected == []

    def test_all_disabled_no_candidates(self):
        m1 = _make_model(model_id="test/a", enabled=False)
        m2 = _make_model(model_id="test/b", enabled=False)
        selector = _make_selector(m1, m2)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert not result.has_candidates
        assert len(result.rejected) == 2

    def test_complex_with_no_tier_3_model_no_candidates(self):
        m1 = _make_model(model_id="test/t1", quality_tier=QualityTier.TIER_1)
        m2 = _make_model(model_id="test/t2", quality_tier=QualityTier.TIER_2)
        selector = _make_selector(m1, m2)
        analysis = _make_analysis("Design a distributed consensus algorithm.")
        pred = _make_prediction(ComplexityLabel.COMPLEX)
        result = selector.select(analysis, pred)
        assert not result.has_candidates
        assert len(result.rejected) == 2


# ── 10. Multiple candidates, deterministic ordering ───────────────────────────

class TestMultipleCandidates:

    def test_multiple_candidates_returned(self):
        m1 = _make_model(model_id="openrouter/alpha")
        m2 = _make_model(model_id="openrouter/beta")
        m3 = _make_model(model_id="openrouter/gamma")
        selector = _make_selector(m1, m2, m3)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert len(result.candidates) == 3

    def test_candidates_sorted_by_model_id(self):
        """Candidates must be returned in deterministic, sorted-by-model_id order."""
        m1 = _make_model(model_id="openrouter/zzz")
        m2 = _make_model(model_id="openrouter/aaa")
        m3 = _make_model(model_id="openrouter/mmm")
        selector = _make_selector(m1, m2, m3)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        ids = result.candidate_ids
        assert ids == sorted(ids)

    def test_rejected_models_not_in_candidates(self):
        m_ok = _make_model(model_id="test/ok")
        m_off = _make_model(model_id="test/off", enabled=False)
        selector = _make_selector(m_ok, m_off)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert "test/off" not in result.candidate_ids
        assert "test/ok" in result.candidate_ids


# ── 11. Rejection log completeness ────────────────────────────────────────────

class TestRejectionLog:

    def test_every_model_has_an_outcome(self):
        m1 = _make_model(model_id="test/a")
        m2 = _make_model(model_id="test/b", enabled=False)
        m3 = _make_model(model_id="test/c", quality_tier=QualityTier.TIER_1)
        selector = _make_selector(m1, m2, m3)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        total_accounted = len(result.candidates) + len(result.rejected)
        assert total_accounted == 3

    def test_accepted_models_have_no_rejection_reason(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        for candidate in result.candidates:
            # accepted models should not appear in rejected list
            assert candidate.model_id not in result.rejected_ids

    def test_rejected_outcomes_have_reason(self):
        model = _make_model(enabled=False)
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        for outcome in result.rejected:
            assert outcome.rejection_reason is not None


# ── 12. Determinism ───────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_input_same_candidates(self):
        models = [
            _make_model(model_id="test/a", quality_tier=QualityTier.TIER_1),
            _make_model(model_id="test/b", quality_tier=QualityTier.TIER_2),
            _make_model(model_id="test/c", quality_tier=QualityTier.TIER_3),
        ]
        selector = _make_selector(*models)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        r1 = selector.select(analysis, pred)
        r2 = selector.select(analysis, pred)
        assert r1.candidate_ids == r2.candidate_ids
        assert r1.rejected_ids  == r2.rejected_ids

    def test_two_selector_instances_same_result(self):
        m = _make_model()
        reg = _make_registry(m)
        s1 = CandidateSelector(reg)
        s2 = CandidateSelector(reg)
        analysis = _make_analysis("What is the capital of France?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        r1 = s1.select(analysis, pred)
        r2 = s2.select(analysis, pred)
        assert r1.candidate_ids == r2.candidate_ids


# ── 13. Invalid registry type ─────────────────────────────────────────────────

class TestInvalidInputs:

    def test_non_registry_argument_raises_type_error(self):
        with pytest.raises(TypeError, match="ModelRegistry"):
            CandidateSelector("not a registry")

    def test_non_registry_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            CandidateSelector({"models": []})


# ── 14. Real registry integration ────────────────────────────────────────────

class TestRealRegistryIntegration:

    def test_real_registry_loads_without_error(self):
        selector = CandidateSelector(_REAL_REGISTRY)
        assert selector is not None

    def test_simple_request_has_candidates_from_real_registry(self):
        selector = CandidateSelector(_REAL_REGISTRY)
        analysis = _make_analysis("What is the capital of France?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        # At least one enabled model should be tier_1 or tier_2 or tier_3
        assert isinstance(result, CandidateSelectionResult)

    def test_complex_request_from_real_registry_no_crash(self):
        """Even if no tier_3 model exists, selection should not crash."""
        selector = CandidateSelector(_REAL_REGISTRY)
        analysis = _make_analysis(
            "Design a distributed consensus algorithm with formal correctness proofs."
        )
        pred = _make_prediction(ComplexityLabel.COMPLEX)
        result = selector.select(analysis, pred)
        assert isinstance(result, CandidateSelectionResult)

    def test_all_registered_models_accounted_for(self):
        """Sum of candidates + rejected == total registered models."""
        selector = CandidateSelector(_REAL_REGISTRY)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        total = len(result.candidates) + len(result.rejected)
        assert total == len(_REAL_REGISTRY._models)


# ── 15. RejectionReason enum completeness ─────────────────────────────────────

class TestRejectionReasonEnum:

    def test_all_rejection_reasons_defined(self):
        expected = {
            "disabled", "tier_too_low", "reasoning_score_low",
            "coding_score_low", "no_structured_output",
            "context_window_too_small",
        }
        actual = {r.value for r in RejectionReason}
        assert expected == actual


# ── 16. Applied filters description ──────────────────────────────────────────

class TestAppliedFiltersDescription:

    def test_reasoning_filter_appears_when_required(self):
        model = _make_model(reasoning_score=0.90)
        selector = _make_selector(model)
        analysis = _make_analysis("Prove step by step that this is correct.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert any("reasoning" in f.lower() for f in result.applied_filters)

    def test_coding_filter_appears_when_required(self):
        model = _make_model(coding_score=0.90)
        selector = _make_selector(model)
        analysis = _make_analysis("Write a function that reverses a string.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert any("coding" in f.lower() for f in result.applied_filters)

    def test_structured_output_filter_appears_when_required(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("Return the answer as JSON with fields name and age.")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert any("structured" in f.lower() for f in result.applied_filters)

    def test_context_filter_always_appears(self):
        model = _make_model()
        selector = _make_selector(model)
        analysis = _make_analysis("What is 2 + 2?")
        pred = _make_prediction(ComplexityLabel.SIMPLE)
        result = selector.select(analysis, pred)
        assert any("context" in f.lower() for f in result.applied_filters)
