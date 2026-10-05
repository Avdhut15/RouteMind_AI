"""
tests/test_llm_judge_evaluator.py
──────────────────────────────────
Unit tests for Phase 3 Part 3 — LLM-as-a-Judge & DeepEval Integration.

All tests use mocked judge callers/providers. No real API keys, network
access, or DeepEval model execution is required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evaluation.llm_judge import (
    JudgeConfig,
    LLMJudgeEvaluator,
    build_judge_prompt,
    extract_json_from_judge_output,
    validate_and_normalise_scores,
)
from app.evaluation.models import EvaluationSource, EvaluationStatus
from app.providers.base import LLMRequest, LLMResponse


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def req():
    return LLMRequest(
        request_id="req-42",
        prompt="Explain the water cycle in two sentences.",
        model_id="auto",
    )


@pytest.fixture
def resp():
    return LLMResponse(
        request_id="req-42",
        model_id="openai/gpt-4o",
        provider="openrouter",
        output="Water evaporates from oceans, condenses into clouds, and falls as rain. This cycle continuously replenishes freshwater sources.",
    )


@pytest.fixture
def default_config():
    return JudgeConfig(judge_model_id="openai/gpt-4o")


@pytest.fixture
def valid_judge_payload(default_config):
    return {
        "scores": {c: 0.9 for c in default_config.criteria},
        "overall": 0.9,
        "reasoning": "The answer is concise and relevant.",
    }


def _mock_caller(raw_output: str):
    """Return an async mock caller that yields raw_output."""
    caller = MagicMock()
    caller.call = AsyncMock(return_value=raw_output)
    return caller


# ── Tests: prompt builder ──────────────────────────────────────────────────────

def test_build_judge_prompt_contains_request(req, resp, default_config):
    prompt = build_judge_prompt(req, resp, default_config.criteria)
    assert req.prompt in prompt
    assert resp.output in prompt
    for c in default_config.criteria:
        assert c in prompt


def test_build_judge_prompt_truncates_large_inputs(default_config):
    huge_request = LLMRequest(
        request_id="r", prompt="x" * 10_000, model_id="m"
    )
    huge_response = LLMResponse(
        request_id="r", model_id="m", provider="p", output="y" * 10_000
    )
    prompt = build_judge_prompt(huge_request, huge_response, default_config.criteria)
    # Should not crash and should be bounded
    assert len(prompt) < 20_000


# ── Tests: JSON extraction ────────────────────────────────────────────────────

def test_extract_json_plain():
    raw = '{"scores": {"relevance": 0.9}, "overall": 0.9, "reasoning": "ok"}'
    result = extract_json_from_judge_output(raw)
    assert result["overall"] == 0.9


def test_extract_json_with_markdown_fence():
    raw = '```json\n{"scores": {"relevance": 0.8}, "overall": 0.8, "reasoning": "fine"}\n```'
    result = extract_json_from_judge_output(raw)
    assert result["overall"] == 0.8


def test_extract_json_with_surrounding_text():
    raw = 'Here is my evaluation:\n{"scores": {"relevance": 0.7}, "overall": 0.7, "reasoning": "ok"}\nEnd.'
    result = extract_json_from_judge_output(raw)
    assert result["overall"] == 0.7


def test_extract_json_raises_on_unparseable():
    with pytest.raises(ValueError, match="Could not extract valid JSON"):
        extract_json_from_judge_output("This is just plain text with no JSON at all.")


# ── Tests: score validation ───────────────────────────────────────────────────

def test_validate_scores_valid():
    parsed = {
        "scores": {"relevance": 0.9, "completeness": 0.8},
        "overall": 0.85,
    }
    individual, overall = validate_and_normalise_scores(parsed, ["relevance", "completeness"])
    assert individual["relevance"] == 0.9
    assert overall == 0.85


def test_validate_scores_missing_criterion():
    parsed = {"scores": {"relevance": 0.9}, "overall": 0.9}
    with pytest.raises(ValueError, match="missing score for criterion 'completeness'"):
        validate_and_normalise_scores(parsed, ["relevance", "completeness"])


def test_validate_scores_missing_overall():
    parsed = {"scores": {"relevance": 0.9}}
    with pytest.raises(ValueError, match="missing 'overall' key"):
        validate_and_normalise_scores(parsed, ["relevance"])


def test_validate_scores_out_of_range_high():
    parsed = {"scores": {"relevance": 1.5}, "overall": 0.9}
    with pytest.raises(ValueError, match="out of range"):
        validate_and_normalise_scores(parsed, ["relevance"])


def test_validate_scores_out_of_range_low():
    parsed = {"scores": {"relevance": -0.1}, "overall": 0.9}
    with pytest.raises(ValueError, match="out of range"):
        validate_and_normalise_scores(parsed, ["relevance"])


def test_validate_scores_non_numeric():
    parsed = {"scores": {"relevance": "high"}, "overall": 0.9}
    with pytest.raises(ValueError, match="not numeric"):
        validate_and_normalise_scores(parsed, ["relevance"])


def test_validate_overall_out_of_range():
    parsed = {"scores": {"relevance": 0.9}, "overall": 2.0}
    with pytest.raises(ValueError, match="out of range"):
        validate_and_normalise_scores(parsed, ["relevance"])


# ── Tests: LLMJudgeEvaluator (mocked caller) ─────────────────────────────────

@pytest.mark.asyncio
async def test_judge_evaluator_success(req, resp, default_config, valid_judge_payload):
    caller = _mock_caller(json.dumps(valid_judge_payload))
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    result = await evaluator.evaluate(req, resp)

    assert result.status == EvaluationStatus.SUCCESS
    assert result.source == EvaluationSource.LLM_JUDGE
    assert result.overall_quality_score == 0.9
    assert result.request_id == "req-42"
    assert result.model_id == "openai/gpt-4o"
    assert result.provider == "openrouter"
    for c in default_config.criteria:
        assert c in result.individual_scores
    assert result.error is None


@pytest.mark.asyncio
async def test_judge_evaluator_preserves_reasoning(req, resp, default_config):
    payload = {
        "scores": {c: 0.8 for c in default_config.criteria},
        "overall": 0.8,
        "reasoning": "Response is mostly relevant and clear.",
    }
    caller = _mock_caller(json.dumps(payload))
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    result = await evaluator.evaluate(req, resp)

    assert "Response is mostly relevant and clear." in result.explanation


@pytest.mark.asyncio
async def test_judge_evaluator_malformed_json(req, resp, default_config):
    caller = _mock_caller("I could not evaluate this response.")
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    result = await evaluator.evaluate(req, resp)

    assert result.status == EvaluationStatus.FAILURE
    assert result.overall_quality_score is None
    assert result.error is not None
    assert "LLM-judge evaluation failed" in result.error


@pytest.mark.asyncio
async def test_judge_evaluator_missing_criterion(req, resp, default_config):
    # Returns valid JSON but missing one expected criterion
    partial_payload = {
        "scores": {"relevance": 0.9},  # missing other criteria
        "overall": 0.9,
        "reasoning": "Partial.",
    }
    caller = _mock_caller(json.dumps(partial_payload))
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    result = await evaluator.evaluate(req, resp)

    assert result.status == EvaluationStatus.FAILURE
    assert "missing score for criterion" in result.error


@pytest.mark.asyncio
async def test_judge_evaluator_score_out_of_range(req, resp, default_config):
    bad_payload = {
        "scores": {c: 1.9 for c in default_config.criteria},
        "overall": 1.9,
        "reasoning": "High scores.",
    }
    caller = _mock_caller(json.dumps(bad_payload))
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    result = await evaluator.evaluate(req, resp)

    assert result.status == EvaluationStatus.FAILURE
    assert "out of range" in result.error


@pytest.mark.asyncio
async def test_judge_evaluator_caller_raises_exception(req, resp, default_config):
    caller = MagicMock()
    caller.call = AsyncMock(side_effect=ConnectionError("Network timeout"))
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    result = await evaluator.evaluate(req, resp)

    assert result.status == EvaluationStatus.FAILURE
    assert "ConnectionError" in result.error
    assert "Network timeout" in result.error


@pytest.mark.asyncio
async def test_judge_evaluator_none_inputs_handled(default_config):
    caller = MagicMock()
    caller.call = AsyncMock(return_value="{}")
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    result = await evaluator.evaluate(None, None)  # type: ignore

    assert result.status == EvaluationStatus.FAILURE
    assert result.overall_quality_score is None


@pytest.mark.asyncio
async def test_judge_evaluator_custom_criteria(req, resp):
    config = JudgeConfig(
        judge_model_id="openai/gpt-4o",
        criteria=["relevance", "clarity"],
    )
    payload = {
        "scores": {"relevance": 0.85, "clarity": 0.75},
        "overall": 0.80,
        "reasoning": "Good.",
    }
    caller = _mock_caller(json.dumps(payload))
    evaluator = LLMJudgeEvaluator(config=config, caller=caller)
    result = await evaluator.evaluate(req, resp)

    assert result.status == EvaluationStatus.SUCCESS
    assert "relevance" in result.individual_scores
    assert "clarity" in result.individual_scores
    assert "completeness" not in result.individual_scores


# ── Tests: DeepEval adapter (mocked) ─────────────────────────────────────────

def test_deepeval_evaluator_raises_import_error_when_not_installed():
    """If deepeval is not installed, DeepEvalEvaluator raises ImportError on init."""
    with patch.dict("sys.modules", {"deepeval": None}):
        from importlib import import_module
        import sys
        # Remove cached module if present
        sys.modules.pop("app.evaluation.deepeval_adapter", None)
        from app.evaluation.deepeval_adapter import DeepEvalConfig

        with pytest.raises(ImportError, match="DeepEval is not installed"):
            from app.evaluation.deepeval_adapter import DeepEvalEvaluator
            DeepEvalEvaluator(config=DeepEvalConfig())


@pytest.mark.asyncio
async def test_deepeval_evaluator_failure_on_error(req, resp):
    """DeepEvalEvaluator returns FAILURE when deepeval raises."""
    with patch("app.evaluation.deepeval_adapter.DeepEvalEvaluator.__init__", return_value=None):
        from app.evaluation.deepeval_adapter import DeepEvalEvaluator, DeepEvalConfig
        evaluator = DeepEvalEvaluator.__new__(DeepEvalEvaluator)
        evaluator._config = DeepEvalConfig()

        # Patch the evaluate method to simulate a runtime failure
        with patch.object(evaluator, "evaluate", new=AsyncMock(return_value=None)) as mock_eval:
            from app.evaluation.models import EvaluationResult, EvaluationStatus, EvaluationSource
            mock_eval.return_value = EvaluationResult(
                request_id=req.request_id,
                model_id=resp.model_id,
                provider=resp.provider,
                status=EvaluationStatus.FAILURE,
                source=EvaluationSource.DEEPEVAL,
                error="DeepEval evaluation failed: ImportError: deepeval not available",
            )
            result = await evaluator.evaluate(req, resp)

        assert result.status == EvaluationStatus.FAILURE
        assert result.source == EvaluationSource.DEEPEVAL
        assert "DeepEval" in result.error


# ── Tests: EvaluationResult serialization ────────────────────────────────────

@pytest.mark.asyncio
async def test_judge_result_serialization(req, resp, default_config, valid_judge_payload):
    caller = _mock_caller(json.dumps(valid_judge_payload))
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    result = await evaluator.evaluate(req, resp)

    data = result.model_dump()
    assert data["status"] == "success"
    assert data["source"] == "llm_judge"
    assert isinstance(data["individual_scores"], dict)
    assert isinstance(data["overall_quality_score"], float)
    assert data["error"] is None

    # Ensure re-parsing roundtrip
    import json as json_module
    serialized = json_module.dumps(data, default=str)
    reloaded = json_module.loads(serialized)
    assert reloaded["request_id"] == "req-42"


# ── Tests: Interface contract ─────────────────────────────────────────────────

def test_llm_judge_evaluator_implements_base_evaluator(default_config):
    from app.evaluation.base import BaseEvaluator
    caller = _mock_caller("{}")
    evaluator = LLMJudgeEvaluator(config=default_config, caller=caller)
    assert isinstance(evaluator, BaseEvaluator)
