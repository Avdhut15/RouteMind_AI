"""
tests/test_deterministic_evaluator.py
─────────────────────────────────────
Unit tests for Phase 3 Part 2 — Deterministic Quality Metrics.

Covers metric logic, overall score aggregation, and safe error handling.
"""

import pytest

from app.providers.base import LLMRequest, LLMResponse
from app.evaluation.models import EvaluationStatus, EvaluationSource, EvaluationResult
from app.evaluation.deterministic import DeterministicMetrics, DeterministicEvaluator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_request():
    return LLMRequest(
        request_id="req-test",
        prompt="Tell me a story.",
        model_id="test/model",
    )


@pytest.fixture
def base_response():
    return LLMResponse(
        request_id="req-test",
        model_id="test/model",
        provider="dummy",
        output="Once upon a time in a faraway land, there was a tiny village.",
        finish_reason="stop",
    )


# ── Tests: Individual Metrics ─────────────────────────────────────────────────

def test_measure_presence():
    assert DeterministicMetrics.measure_presence("Some valid text") == 1.0
    assert DeterministicMetrics.measure_presence("") == 0.0
    assert DeterministicMetrics.measure_presence("   \n  ") == 0.0
    assert DeterministicMetrics.measure_presence(None) == 0.0


def test_measure_truncation(base_response):
    # Perfect sentence ending
    base_response.output = "This is complete."
    assert DeterministicMetrics.measure_truncation(base_response, base_response.output) == 1.0
    
    # Missing punctuation at the end suggests cutoff
    base_response.output = "This is cut off abruptly"
    assert DeterministicMetrics.measure_truncation(base_response, base_response.output) == 0.8
    
    # Unclosed markdown block
    base_response.output = "Here is some code:\n```python\nprint('hello')\n"
    assert DeterministicMetrics.measure_truncation(base_response, base_response.output) == 0.2
    
    # Explicit limit hit
    base_response.finish_reason = "length"
    assert DeterministicMetrics.measure_truncation(base_response, base_response.output) == 0.0


def test_measure_repetition():
    # Short texts are ignored (1.0)
    assert DeterministicMetrics.measure_repetition("Short text") == 1.0
    
    # Natural text with variety
    natural_text = "The quick brown fox jumps over the lazy dog perfectly well."
    assert DeterministicMetrics.measure_repetition(natural_text) == 1.0
    
    # Extreme degenerate repetition (ratio < 0.1)
    # E.g., 20 words, only 1 unique word = 1/20 = 0.05
    degenerate = "loop " * 20
    assert DeterministicMetrics.measure_repetition(degenerate) == 0.0
    
    # Moderate repetition (ratio between 0.1 and 0.4)
    # E.g. 20 words, 5 unique words = 5/20 = 0.25 -> (0.25 - 0.1) / 0.3 = 0.5
    moderate = "one two three four five " * 4
    score = DeterministicMetrics.measure_repetition(moderate)
    assert 0.0 < score < 1.0
    assert round(score, 2) == 0.5


def test_measure_format_compliance(base_request):
    # Not asking for JSON
    base_request.prompt = "Write a poem."
    assert DeterministicMetrics.measure_format_compliance(base_request, "Roses are red...") == 1.0

    # Asking for JSON, giving valid JSON
    base_request.prompt = "Output your response as JSON."
    valid_json = '```json\n{"key": "value"}\n```'
    assert DeterministicMetrics.measure_format_compliance(base_request, valid_json) == 1.0

    # Asking for JSON, giving valid JSON array
    valid_array = '[{"item": 1}, {"item": 2}]'
    assert DeterministicMetrics.measure_format_compliance(base_request, valid_array) == 1.0

    # Asking for JSON, giving invalid JSON
    invalid_json = "Here is the JSON: {key: value without quotes}"
    assert DeterministicMetrics.measure_format_compliance(base_request, invalid_json) == 0.0


# ── Tests: Evaluator Pipeline ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluator_success_baseline(base_request, base_response):
    """Test a fully compliant, healthy response."""
    evaluator = DeterministicEvaluator()
    result = await evaluator.evaluate(base_request, base_response)
    
    assert result.status == EvaluationStatus.SUCCESS
    assert result.source == EvaluationSource.DETERMINISTIC
    assert result.overall_quality_score == 1.0
    
    assert result.individual_scores["presence"] == 1.0
    assert result.individual_scores["truncation"] == 1.0
    assert result.individual_scores["repetition"] == 1.0
    assert result.individual_scores["format"] == 1.0


@pytest.mark.asyncio
async def test_evaluator_success_penalties(base_request, base_response):
    """Test how degenerate behavior reduces the overall score."""
    base_request.prompt = "Return a JSON payload."
    base_response.output = "loop " * 20 # Repetition penalty
    base_response.finish_reason = "length" # Truncation penalty
    # Format penalty because it's not JSON
    
    evaluator = DeterministicEvaluator()
    result = await evaluator.evaluate(base_request, base_response)
    
    assert result.status == EvaluationStatus.SUCCESS
    assert result.individual_scores["presence"] == 1.0  # Present
    assert result.individual_scores["truncation"] == 0.0 # Length hit
    assert result.individual_scores["repetition"] == 0.0 # Degenerate
    assert result.individual_scores["format"] == 0.0     # Invalid JSON
    
    # With default weights (presence:0.3, truncation:0.2, repetition:0.2, format:0.3)
    # The overall score should just be 0.3 (since only presence got 1.0)
    assert round(result.overall_quality_score, 2) == 0.3


@pytest.mark.asyncio
async def test_evaluator_handles_exceptions_safely(base_request, base_response):
    """Ensure unexpected evaluation failures do not crash the pipeline."""
    evaluator = DeterministicEvaluator()
    
    # Pass None as request to force a crash
    result = await evaluator.evaluate(None, base_response) # type: ignore
    
    assert result.status == EvaluationStatus.FAILURE
    assert result.source == EvaluationSource.DETERMINISTIC
    assert result.overall_quality_score is None
    assert "ValueError" in result.error
    assert "Both request and response must be provided" in result.error
    
    # Make sure we preserved identity where possible
    assert result.model_id == base_response.model_id
    assert result.provider == base_response.provider


@pytest.mark.asyncio
async def test_evaluator_custom_weights(base_request, base_response):
    """Ensure custom weights correctly adjust the overall score."""
    base_request.prompt = "Give me JSON."
    base_response.output = "Invalid JSON."
    
    # The format will fail (score 0.0), presence/truncation/repetition pass (score 1.0)
    # Give format 90% weight. The score should drop dramatically.
    evaluator = DeterministicEvaluator(weights={
        "presence": 0.05,
        "truncation": 0.05,
        "repetition": 0.0,
        "format": 0.9,
    })
    
    result = await evaluator.evaluate(base_request, base_response)
    assert result.individual_scores["format"] == 0.0
    assert result.individual_scores["presence"] == 1.0
    assert result.overall_quality_score == 0.1 # 0.05 + 0.05
