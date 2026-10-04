"""
tests/test_evaluation_models.py
───────────────────────────────
Unit tests for Phase 3 Part 1 — Evaluation Architecture & Result Schema.

Ensures proper validation of scores, NaN/Inf checks, and state constraints 
for the EvaluationResult schema.
"""

import math
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.evaluation.models import (
    EvaluationStatus,
    EvaluationSource,
    QualityDecision,
    EvaluationResult,
)
from app.evaluation.base import BaseEvaluator
from app.providers.base import LLMRequest, LLMResponse


# ── Fixtures & Mocks ──────────────────────────────────────────────────────────

@pytest.fixture
def base_kwargs():
    """Minimal valid keyword arguments for a successful evaluation result."""
    return {
        "request_id": "req-123",
        "model_id": "test/model",
        "provider": "openrouter",
        "status": EvaluationStatus.SUCCESS,
        "source": EvaluationSource.DETERMINISTIC,
    }


class DummyEvaluator(BaseEvaluator):
    """A dummy evaluator that always succeeds, used to test the interface contract."""
    async def evaluate(self, request: LLMRequest, response: LLMResponse) -> EvaluationResult:
        return EvaluationResult(
            request_id=request.request_id,
            model_id=response.model_id,
            provider=response.provider,
            status=EvaluationStatus.SUCCESS,
            source=EvaluationSource.DETERMINISTIC,
            overall_quality_score=0.95,
            decision=QualityDecision.PASS,
        )


# ── Tests: Interface ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluator_interface():
    """Ensure the BaseEvaluator can be implemented and executed correctly."""
    request = LLMRequest(
        request_id="test-req-id",
        prompt="Test prompt",
        model_id="test/model",
    )
    response = LLMResponse(
        request_id="test-req-id",
        model_id="test/model",
        provider="dummy_provider",
        output="Test output",
    )
    
    evaluator = DummyEvaluator()
    result = await evaluator.evaluate(request, response)
    
    assert isinstance(result, EvaluationResult)
    assert result.request_id == "test-req-id"
    assert result.model_id == "test/model"
    assert result.status == EvaluationStatus.SUCCESS
    assert result.overall_quality_score == 0.95


# ── Tests: Schema Validation ──────────────────────────────────────────────────

def test_valid_successful_evaluation(base_kwargs):
    """Test standard success case."""
    result = EvaluationResult(
        **base_kwargs,
        overall_quality_score=0.8,
        individual_scores={"coherence": 0.9, "accuracy": 0.75},
        decision=QualityDecision.PASS,
        explanation="Looked good.",
    )
    
    assert result.status == EvaluationStatus.SUCCESS
    assert result.overall_quality_score == 0.8
    assert result.decision == QualityDecision.PASS
    assert result.error is None
    assert isinstance(result.evaluated_at, datetime)


def test_valid_failed_evaluation(base_kwargs):
    """Test standard failure case where the evaluation itself failed."""
    kwargs = base_kwargs.copy()
    kwargs.update({
        "status": EvaluationStatus.FAILURE,
        "error": "Timeout while calling deep-eval service",
    })
    result = EvaluationResult(**kwargs)
    
    assert result.status == EvaluationStatus.FAILURE
    assert result.error == "Timeout while calling deep-eval service"
    assert result.overall_quality_score is None
    assert result.decision is None


def test_failure_must_not_have_score(base_kwargs):
    kwargs = base_kwargs.copy()
    kwargs.update({
        "status": EvaluationStatus.FAILURE,
        "error": "Something went wrong",
        "overall_quality_score": 0.5,
    })
    with pytest.raises(ValidationError, match="cannot have an overall_quality_score"):
        EvaluationResult(**kwargs)


def test_failure_must_not_have_decision(base_kwargs):
    kwargs = base_kwargs.copy()
    kwargs.update({
        "status": EvaluationStatus.FAILURE,
        "error": "Something went wrong",
        "decision": QualityDecision.FAIL,
    })
    with pytest.raises(ValidationError, match="cannot yield a QualityDecision"):
        EvaluationResult(**kwargs)


def test_failure_must_have_error(base_kwargs):
    kwargs = base_kwargs.copy()
    kwargs.update({
        "status": EvaluationStatus.FAILURE,
        # error is implicitly None
    })
    with pytest.raises(ValidationError, match="must provide an error message"):
        EvaluationResult(**kwargs)


def test_success_must_not_have_error(base_kwargs):
    kwargs = base_kwargs.copy()
    kwargs.update({
        "error": "This should not be here",
    })
    with pytest.raises(ValidationError, match="should not contain an error message"):
        EvaluationResult(**kwargs)


def test_overall_score_out_of_range(base_kwargs):
    kwargs = base_kwargs.copy()
    
    kwargs["overall_quality_score"] = 1.1
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        EvaluationResult(**kwargs)
        
    kwargs["overall_quality_score"] = -0.1
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        EvaluationResult(**kwargs)


def test_individual_scores_out_of_range(base_kwargs):
    kwargs = base_kwargs.copy()
    
    kwargs["individual_scores"] = {"coherence": 1.5}
    with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
        EvaluationResult(**kwargs)
        
    kwargs["individual_scores"] = {"coherence": -0.5}
    with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
        EvaluationResult(**kwargs)


def test_nan_and_inf_overall_score_rejected(base_kwargs):
    kwargs = base_kwargs.copy()
    
    kwargs["overall_quality_score"] = float('nan')
    with pytest.raises(ValidationError):
        EvaluationResult(**kwargs)
        
    kwargs["overall_quality_score"] = float('inf')
    with pytest.raises(ValidationError):
        EvaluationResult(**kwargs)


def test_nan_and_inf_individual_scores_rejected(base_kwargs):
    kwargs = base_kwargs.copy()
    
    kwargs["individual_scores"] = {"metric": float('nan')}
    with pytest.raises(ValidationError):
        EvaluationResult(**kwargs)
        
    kwargs["individual_scores"] = {"metric": float('inf')}
    with pytest.raises(ValidationError):
        EvaluationResult(**kwargs)


def test_enum_serialization(base_kwargs):
    """Test that enum values correctly dump as strings."""
    result = EvaluationResult(
        **base_kwargs,
        decision=QualityDecision.ESCALATE
    )
    data = result.model_dump()
    assert data["status"] == "success"
    assert data["source"] == "deterministic"
    assert data["decision"] == "escalate"


def test_missing_required_fields():
    """Ensure missing required fields raise Pydantic ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(
            request_id="req-123",
            model_id="test",
            # provider missing
            status=EvaluationStatus.SUCCESS,
            source=EvaluationSource.DETERMINISTIC,
        )
    assert "provider" in str(exc_info.value)
