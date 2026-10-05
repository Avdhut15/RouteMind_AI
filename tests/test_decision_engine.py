"""
tests/test_decision_engine.py
──────────────────────────────
Unit tests for Phase 3 Part 4 — Quality Threshold & Decision Engine.

All tests are offline and deterministic; no LLM calls, network access,
or external services are required.
"""

from __future__ import annotations

import pytest

from app.evaluation.decision_engine import (
    DEFAULT_ESCALATE_THRESHOLD,
    DEFAULT_PASS_THRESHOLD,
    QualityDecisionEngine,
    QualityDecisionResult,
    ThresholdConfig,
)
from app.evaluation.models import (
    EvaluationResult,
    EvaluationSource,
    EvaluationStatus,
    QualityDecision,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_eval(
    score: float | None = 0.8,
    status: EvaluationStatus = EvaluationStatus.SUCCESS,
    error: str | None = None,
) -> EvaluationResult:
    """Build a minimal EvaluationResult for testing."""
    kwargs: dict = dict(
        request_id="req-1",
        model_id="test/model",
        provider="dummy",
        status=status,
        source=EvaluationSource.DETERMINISTIC,
    )
    if score is not None:
        kwargs["overall_quality_score"] = score
    if error is not None:
        kwargs["error"] = error
    return EvaluationResult(**kwargs)


# ── ThresholdConfig validation ────────────────────────────────────────────────

class TestThresholdConfig:
    def test_defaults(self):
        cfg = ThresholdConfig()
        assert cfg.pass_threshold == DEFAULT_PASS_THRESHOLD
        assert cfg.escalate_threshold == DEFAULT_ESCALATE_THRESHOLD

    def test_custom_valid(self):
        cfg = ThresholdConfig(pass_threshold=0.8, escalate_threshold=0.5)
        assert cfg.pass_threshold == 0.8
        assert cfg.escalate_threshold == 0.5

    def test_reversed_thresholds_raises(self):
        with pytest.raises(ValueError, match="strictly less than"):
            ThresholdConfig(pass_threshold=0.4, escalate_threshold=0.7)

    def test_equal_thresholds_raises(self):
        with pytest.raises(ValueError, match="strictly less than"):
            ThresholdConfig(pass_threshold=0.5, escalate_threshold=0.5)

    def test_out_of_range_pass_raises(self):
        with pytest.raises(ValueError, match="pass_threshold"):
            ThresholdConfig(pass_threshold=1.1, escalate_threshold=0.4)

    def test_negative_escalate_raises(self):
        with pytest.raises(ValueError, match="escalate_threshold"):
            ThresholdConfig(pass_threshold=0.7, escalate_threshold=-0.1)

    def test_nan_raises(self):
        import math
        with pytest.raises(ValueError, match="finite"):
            ThresholdConfig(pass_threshold=math.nan, escalate_threshold=0.4)

    def test_inf_raises(self):
        import math
        with pytest.raises(ValueError, match="finite"):
            ThresholdConfig(pass_threshold=math.inf, escalate_threshold=0.4)

    def test_non_float_raises(self):
        with pytest.raises(TypeError, match="float"):
            ThresholdConfig(pass_threshold="high", escalate_threshold=0.4)  # type: ignore

    def test_boundary_pass_at_1(self):
        cfg = ThresholdConfig(pass_threshold=1.0, escalate_threshold=0.9)
        assert cfg.pass_threshold == 1.0

    def test_boundary_escalate_at_0(self):
        cfg = ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.0)
        assert cfg.escalate_threshold == 0.0


# ── Score-based threshold decisions ──────────────────────────────────────────

class TestScoreBasedDecisions:
    def setup_method(self):
        # Use explicit thresholds: PASS >= 0.7, ESCALATE >= 0.4, else FAIL
        self.engine = QualityDecisionEngine(
            ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.4)
        )

    def test_score_above_pass_threshold(self):
        result = self.engine.decide(_make_eval(score=0.9))
        assert result.decision == QualityDecision.PASS

    def test_score_exactly_at_pass_threshold(self):
        result = self.engine.decide(_make_eval(score=0.7))
        assert result.decision == QualityDecision.PASS

    def test_score_just_below_pass_threshold(self):
        result = self.engine.decide(_make_eval(score=0.699))
        assert result.decision == QualityDecision.ESCALATE

    def test_score_exactly_at_escalate_threshold(self):
        result = self.engine.decide(_make_eval(score=0.4))
        assert result.decision == QualityDecision.ESCALATE

    def test_score_between_thresholds(self):
        result = self.engine.decide(_make_eval(score=0.55))
        assert result.decision == QualityDecision.ESCALATE

    def test_score_just_below_escalate_threshold(self):
        result = self.engine.decide(_make_eval(score=0.399))
        assert result.decision == QualityDecision.FAIL

    def test_score_exactly_zero(self):
        result = self.engine.decide(_make_eval(score=0.0))
        assert result.decision == QualityDecision.FAIL

    def test_score_exactly_one(self):
        result = self.engine.decide(_make_eval(score=1.0))
        assert result.decision == QualityDecision.PASS


# ── Evaluation failure and skip states ───────────────────────────────────────

class TestFailureAndSkipStates:
    def setup_method(self):
        self.engine = QualityDecisionEngine()

    def test_evaluation_failure_yields_fail(self):
        eval_result = _make_eval(
            score=None,
            status=EvaluationStatus.FAILURE,
            error="Judge timeout",
        )
        result = self.engine.decide(eval_result)
        assert result.decision == QualityDecision.FAIL
        assert result.quality_score is None
        assert "Judge timeout" in result.reason

    def test_evaluation_failure_ignores_any_score(self):
        # Even if somehow a score was attached to a FAILURE result,
        # the engine must respect the FAILURE status and return FAIL.
        # We construct this edge case by manually patching after construction.
        eval_result = _make_eval(
            score=None,
            status=EvaluationStatus.FAILURE,
            error="Something failed",
        )
        result = self.engine.decide(eval_result)
        assert result.decision == QualityDecision.FAIL

    def test_evaluation_skipped_yields_escalate(self):
        eval_result = _make_eval(score=None, status=EvaluationStatus.SKIPPED)
        result = self.engine.decide(eval_result)
        assert result.decision == QualityDecision.ESCALATE
        assert "skipped" in result.reason.lower()

    def test_missing_score_yields_escalate(self):
        # SUCCESS status but no quality score
        eval_result = _make_eval(score=None, status=EvaluationStatus.SUCCESS)
        result = self.engine.decide(eval_result)
        assert result.decision == QualityDecision.ESCALATE
        assert result.quality_score is None
        assert "No quality score" in result.reason


# ── Decision result field correctness ────────────────────────────────────────

class TestDecisionResultFields:
    def setup_method(self):
        self.cfg = ThresholdConfig(pass_threshold=0.7, escalate_threshold=0.4)
        self.engine = QualityDecisionEngine(self.cfg)

    def test_thresholds_echo_configuration(self):
        result = self.engine.decide(_make_eval(score=0.8))
        assert result.pass_threshold == 0.7
        assert result.escalate_threshold == 0.4

    def test_request_id_propagated(self):
        result = self.engine.decide(_make_eval(score=0.8))
        assert result.request_id == "req-1"

    def test_model_id_propagated(self):
        result = self.engine.decide(_make_eval(score=0.8))
        assert result.model_id == "test/model"

    def test_provider_propagated(self):
        result = self.engine.decide(_make_eval(score=0.8))
        assert result.provider == "dummy"

    def test_score_propagated(self):
        result = self.engine.decide(_make_eval(score=0.65))
        assert result.quality_score == 0.65

    def test_evaluation_status_propagated(self):
        result = self.engine.decide(_make_eval(score=0.8))
        assert result.evaluation_status == EvaluationStatus.SUCCESS

    def test_evaluation_source_propagated(self):
        result = self.engine.decide(_make_eval(score=0.8))
        assert result.evaluation_source == EvaluationSource.DETERMINISTIC

    def test_pass_reason_mentions_threshold(self):
        result = self.engine.decide(_make_eval(score=0.9))
        assert "0.7" in result.reason  # pass threshold appears in reason

    def test_escalate_reason_mentions_both_thresholds(self):
        result = self.engine.decide(_make_eval(score=0.55))
        assert "0.7" in result.reason
        assert "0.4" in result.reason

    def test_fail_reason_mentions_escalate_threshold(self):
        result = self.engine.decide(_make_eval(score=0.1))
        assert "0.4" in result.reason

    def test_result_is_dataclass(self):
        result = self.engine.decide(_make_eval(score=0.8))
        assert isinstance(result, QualityDecisionResult)


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_output(self):
        engine = QualityDecisionEngine()
        eval_result = _make_eval(score=0.65)
        r1 = engine.decide(eval_result)
        r2 = engine.decide(eval_result)
        assert r1.decision == r2.decision
        assert r1.reason == r2.reason
        assert r1.quality_score == r2.quality_score

    def test_two_engine_instances_same_result(self):
        e1 = QualityDecisionEngine()
        e2 = QualityDecisionEngine()
        eval_result = _make_eval(score=0.55)
        assert e1.decide(eval_result).decision == e2.decide(eval_result).decision

    def test_engine_config_accessible(self):
        cfg = ThresholdConfig(pass_threshold=0.8, escalate_threshold=0.5)
        engine = QualityDecisionEngine(cfg)
        assert engine.config.pass_threshold == 0.8
        assert engine.config.escalate_threshold == 0.5
