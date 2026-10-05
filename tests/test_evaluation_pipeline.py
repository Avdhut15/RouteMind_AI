"""
tests/test_evaluation_pipeline.py
───────────────────────────────────
Unit tests for Phase 3 Part 5 — Evaluation Pipeline Integration.

All tests are fully offline. No API keys, network access, or real LLM
calls are required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.evaluation.base import BaseEvaluator
from app.evaluation.decision_engine import ThresholdConfig
from app.evaluation.models import (
    EvaluationResult,
    EvaluationSource,
    EvaluationStatus,
    QualityDecision,
)
from app.evaluation.pipeline import EvaluationPipeline, EvaluationPipelineResult
from app.providers.base import LLMRequest, LLMResponse


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_request(prompt: str = "Explain gravity.") -> LLMRequest:
    return LLMRequest(
        request_id="req-pipeline-1",
        prompt=prompt,
        model_id="auto",
    )


def _make_response(output: str = "Gravity is a fundamental force.") -> LLMResponse:
    return LLMResponse(
        request_id="req-pipeline-1",
        model_id="test/model",
        provider="dummy",
        output=output,
        finish_reason="stop",
    )


def _mock_evaluator(
    score: float | None = 0.85,
    status: EvaluationStatus = EvaluationStatus.SUCCESS,
    error: str | None = None,
    source: EvaluationSource = EvaluationSource.DETERMINISTIC,
) -> BaseEvaluator:
    """Return a mock evaluator that produces the given EvaluationResult."""
    eval_kwargs: dict = dict(
        request_id="req-pipeline-1",
        model_id="test/model",
        provider="dummy",
        status=status,
        source=source,
    )
    if score is not None:
        eval_kwargs["overall_quality_score"] = score
    if error is not None:
        eval_kwargs["error"] = error

    mock = MagicMock(spec=BaseEvaluator)
    mock.evaluate = AsyncMock(return_value=EvaluationResult(**eval_kwargs))
    return mock


# ── Constructor validation ────────────────────────────────────────────────────

def test_pipeline_requires_base_evaluator():
    with pytest.raises(TypeError, match="BaseEvaluator"):
        EvaluationPipeline(evaluator="not_an_evaluator")  # type: ignore


def test_pipeline_accepts_valid_evaluator():
    evaluator = _mock_evaluator()
    pipeline = EvaluationPipeline(evaluator=evaluator)
    assert pipeline is not None


# ── Core pipeline runs ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_pass_decision():
    """High score → PASS."""
    pipeline = EvaluationPipeline(
        evaluator=_mock_evaluator(score=0.9),
        threshold_config=ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.4),
    )
    result = await pipeline.run(_make_request(), _make_response())

    assert isinstance(result, EvaluationPipelineResult)
    assert result.decision == QualityDecision.PASS
    assert result.quality_score == 0.9


@pytest.mark.asyncio
async def test_pipeline_escalate_decision():
    """Score between thresholds → ESCALATE."""
    pipeline = EvaluationPipeline(
        evaluator=_mock_evaluator(score=0.55),
        threshold_config=ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.4),
    )
    result = await pipeline.run(_make_request(), _make_response())

    assert result.decision == QualityDecision.ESCALATE
    assert result.quality_score == 0.55


@pytest.mark.asyncio
async def test_pipeline_fail_decision():
    """Low score → FAIL."""
    pipeline = EvaluationPipeline(
        evaluator=_mock_evaluator(score=0.2),
        threshold_config=ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.4),
    )
    result = await pipeline.run(_make_request(), _make_response())

    assert result.decision == QualityDecision.FAIL
    assert result.quality_score == 0.2


# ── Failure and skip states ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_evaluation_failure_yields_fail():
    """Evaluator returning FAILURE → pipeline decision is FAIL."""
    pipeline = EvaluationPipeline(
        evaluator=_mock_evaluator(
            score=None, status=EvaluationStatus.FAILURE, error="Judge timeout"
        ),
    )
    result = await pipeline.run(_make_request(), _make_response())

    assert result.decision == QualityDecision.FAIL
    assert result.quality_score is None
    assert result.evaluation_status == EvaluationStatus.FAILURE
    assert "Judge timeout" in result.reason


@pytest.mark.asyncio
async def test_pipeline_evaluation_skipped_yields_escalate():
    """Evaluator returning SKIPPED → pipeline decision is ESCALATE."""
    pipeline = EvaluationPipeline(
        evaluator=_mock_evaluator(score=None, status=EvaluationStatus.SKIPPED),
    )
    result = await pipeline.run(_make_request(), _make_response())

    assert result.decision == QualityDecision.ESCALATE
    assert result.evaluation_status == EvaluationStatus.SKIPPED


@pytest.mark.asyncio
async def test_pipeline_missing_score_yields_escalate():
    """SUCCESS status with no score → conservative ESCALATE."""
    pipeline = EvaluationPipeline(
        evaluator=_mock_evaluator(score=None, status=EvaluationStatus.SUCCESS),
    )
    result = await pipeline.run(_make_request(), _make_response())

    assert result.decision == QualityDecision.ESCALATE
    assert result.quality_score is None


# ── Exception containment ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_evaluator_exception_is_contained():
    """Unexpected evaluator exception → structured FAILURE result, not crash."""
    bad_evaluator = MagicMock(spec=BaseEvaluator)
    bad_evaluator.evaluate = AsyncMock(side_effect=RuntimeError("Unexpected crash!"))

    pipeline = EvaluationPipeline(evaluator=bad_evaluator)
    result = await pipeline.run(_make_request(), _make_response())

    assert isinstance(result, EvaluationPipelineResult)
    assert result.decision == QualityDecision.FAIL          # FAILURE → FAIL
    assert result.evaluation_status == EvaluationStatus.FAILURE
    assert "Unexpected crash!" in result.reason


# ── Identity propagation ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_preserves_request_id():
    pipeline = EvaluationPipeline(evaluator=_mock_evaluator(score=0.8))
    result = await pipeline.run(_make_request(), _make_response())
    assert result.request_id == "req-pipeline-1"


@pytest.mark.asyncio
async def test_pipeline_preserves_model_id():
    pipeline = EvaluationPipeline(evaluator=_mock_evaluator(score=0.8))
    result = await pipeline.run(_make_request(), _make_response())
    assert result.model_id == "test/model"


@pytest.mark.asyncio
async def test_pipeline_preserves_provider():
    pipeline = EvaluationPipeline(evaluator=_mock_evaluator(score=0.8))
    result = await pipeline.run(_make_request(), _make_response())
    assert result.provider == "dummy"


@pytest.mark.asyncio
async def test_pipeline_preserves_evaluation_source():
    pipeline = EvaluationPipeline(
        evaluator=_mock_evaluator(score=0.8, source=EvaluationSource.LLM_JUDGE)
    )
    result = await pipeline.run(_make_request(), _make_response())
    assert result.evaluation_source == EvaluationSource.LLM_JUDGE


# ── Full result structure ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_result_contains_evaluation_result():
    pipeline = EvaluationPipeline(evaluator=_mock_evaluator(score=0.8))
    result = await pipeline.run(_make_request(), _make_response())

    assert isinstance(result.evaluation_result, EvaluationResult)
    assert result.evaluation_result.overall_quality_score == 0.8


@pytest.mark.asyncio
async def test_pipeline_result_has_reason():
    pipeline = EvaluationPipeline(
        evaluator=_mock_evaluator(score=0.8),
        threshold_config=ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.4),
    )
    result = await pipeline.run(_make_request(), _make_response())
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


# ── Determinism ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_is_deterministic():
    """Same inputs via mock evaluator → same output every time."""
    evaluator = _mock_evaluator(score=0.65)
    pipeline = EvaluationPipeline(
        evaluator=evaluator,
        threshold_config=ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.4),
    )
    req, resp = _make_request(), _make_response()

    r1 = await pipeline.run(req, resp)
    r2 = await pipeline.run(req, resp)

    assert r1.decision == r2.decision
    assert r1.quality_score == r2.quality_score
    assert r1.reason == r2.reason


# ── Dependency injection with DeterministicEvaluator ─────────────────────────

@pytest.mark.asyncio
async def test_pipeline_with_real_deterministic_evaluator():
    """Integration: wire an actual DeterministicEvaluator through the pipeline."""
    from app.evaluation.deterministic import DeterministicEvaluator

    evaluator = DeterministicEvaluator()
    pipeline = EvaluationPipeline(
        evaluator=evaluator,
        threshold_config=ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.4),
    )
    request = _make_request("Explain the water cycle.")
    response = _make_response(
        "Water evaporates from oceans, condenses into clouds, and falls as rain."
    )

    result = await pipeline.run(request, response)

    assert isinstance(result, EvaluationPipelineResult)
    assert result.evaluation_status == EvaluationStatus.SUCCESS
    assert result.decision in {
        QualityDecision.PASS, QualityDecision.ESCALATE, QualityDecision.FAIL
    }
    assert result.quality_score is not None
