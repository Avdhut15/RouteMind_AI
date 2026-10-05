"""
tests/test_phase3_benchmarks.py
───────────────────────────────
Unit tests for Phase 3 Part 6 — Quality Evaluation Validation & Benchmarking.

Validates the complete evaluation subsystem via the benchmark runner, dataset,
and mock execution of alternative evaluators.

All tests are strictly offline and do not require API keys or external services.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evaluation.decision_engine import ThresholdConfig
from app.evaluation.models import EvaluationResult, EvaluationSource, EvaluationStatus, QualityDecision
from app.evaluation.pipeline import EvaluationPipeline
from app.providers.base import LLMRequest, LLMResponse
from benchmarks.phase3_runner import load_dataset, run_benchmark


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dataset_path(tmp_path) -> Path:
    """Provides a temporary, minimal valid dataset for testing."""
    data = {
        "_meta": {
            "dataset_version": "1.0",
            "evaluation_mode": "offline_deterministic"
        },
        "cases": [
            {
                "id": "test-pass",
                "category": "high_quality",
                "expected_decision": "pass",
                "request": {"prompt": "Hello", "model_id": "auto"},
                "response": {"output": "Hi there! How can I help you?", "finish_reason": "stop"}
            },
            {
                "id": "test-fail",
                "category": "bad_response",
                "expected_decision": "fail",
                "request": {"prompt": "Return JSON", "model_id": "auto"},
                "response": {"output": "bad bad bad bad bad bad bad bad bad bad bad bad bad", "finish_reason": "length"}
            }
        ]
    }
    path = tmp_path / "test_dataset.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def output_dir(tmp_path) -> Path:
    out = tmp_path / "results"
    out.mkdir()
    return out


# ── Dataset Loading ───────────────────────────────────────────────────────────

def test_load_dataset_valid(dataset_path):
    dataset = load_dataset(dataset_path)
    assert "_meta" in dataset
    assert len(dataset["cases"]) == 2


def test_load_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("does_not_exist.json"))


# ── Benchmark Execution & Output Generation ───────────────────────────────────

@pytest.mark.asyncio
async def test_run_benchmark_creates_output(dataset_path, output_dir):
    out_file = await run_benchmark(dataset_path, output_dir)
    assert out_file.exists()
    assert out_file.suffix == ".json"


@pytest.mark.asyncio
async def test_benchmark_output_structure_and_no_secrets(dataset_path, output_dir):
    out_file = await run_benchmark(dataset_path, output_dir)
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "metadata" in data
    assert "summary" in data
    assert "cases" in data
    
    meta = data["metadata"]
    assert "timestamp" in meta
    assert meta["evaluation_mode"] == "offline_deterministic"
    
    # Ensure no secrets leak
    text_content = out_file.read_text(encoding="utf-8")
    assert "api_key" not in text_content.lower()
    assert "secret" not in text_content.lower()


@pytest.mark.asyncio
async def test_benchmark_aggregate_summary(dataset_path, output_dir):
    out_file = await run_benchmark(dataset_path, output_dir)
    with out_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)["summary"]
        
    assert summary["total_cases"] == 2
    assert summary["evaluator_source"] == "DETERMINISTIC"
    assert summary["status_distribution"]["SUCCESS"] == 2
    assert summary["status_distribution"]["FAILURE"] == 0
    assert summary["decision_distribution"]["PASS"] == 1
    assert summary["decision_distribution"]["FAIL"] == 1
    
    stats = summary["score_statistics"]
    assert stats["count"] == 2
    assert stats["average"] is not None
    assert stats["min"] is not None
    assert stats["max"] is not None


# ── Deterministic Reproducibility ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deterministic_reproducibility(dataset_path, output_dir):
    # Run benchmark twice and verify exact identical scores and decisions
    out1 = await run_benchmark(dataset_path, output_dir)
    out2 = await run_benchmark(dataset_path, output_dir)
    
    with out1.open("r") as f1, out2.open("r") as f2:
        d1 = json.load(f1)
        d2 = json.load(f2)
        
    cases1 = d1["cases"]
    cases2 = d2["cases"]
    
    assert len(cases1) == len(cases2)
    for c1, c2 in zip(cases1, cases2):
        assert c1["case_id"] == c2["case_id"]
        assert c1["quality_score"] == c2["quality_score"]
        assert c1["actual_decision"] == c2["actual_decision"]
        assert c1["individual_scores"] == c2["individual_scores"]


# ── Threshold Boundary Behavior (Pipeline Level) ──────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_threshold_boundaries():
    # Use a mock evaluator to inject exact scores
    from app.evaluation.base import BaseEvaluator
    
    mock_eval = MagicMock(spec=BaseEvaluator)
    # Define exact threshold config
    config = ThresholdConfig(pass_threshold=0.8, escalate_threshold=0.5)
    pipeline = EvaluationPipeline(evaluator=mock_eval, threshold_config=config)
    
    req = LLMRequest(request_id="1", prompt="test", model_id="auto")
    resp = LLMResponse(request_id="1", model_id="test", provider="test", output="test")
    
    async def run_with_score(score: float):
        mock_eval.evaluate = AsyncMock(return_value=EvaluationResult(
            request_id="1", model_id="test", provider="test",
            status=EvaluationStatus.SUCCESS, source=EvaluationSource.DETERMINISTIC,
            overall_quality_score=score
        ))
        return await pipeline.run(req, resp)

    # Above pass
    res = await run_with_score(0.85)
    assert res.decision == QualityDecision.PASS
    
    # Exactly pass
    res = await run_with_score(0.80)
    assert res.decision == QualityDecision.PASS
    
    # Below pass, above escalate
    res = await run_with_score(0.79)
    assert res.decision == QualityDecision.ESCALATE
    
    # Exactly escalate
    res = await run_with_score(0.50)
    assert res.decision == QualityDecision.ESCALATE
    
    # Below escalate
    res = await run_with_score(0.49)
    assert res.decision == QualityDecision.FAIL


# ── Mock LLM-Judge & DeepEval Behavior ────────────────────────────────────────

@pytest.mark.asyncio
async def test_mock_llm_judge_pipeline():
    from app.evaluation.llm_judge import LLMJudgeEvaluator, JudgeConfig
    from app.evaluation.decision_engine import QualityDecisionEngine
    
    # Setup mock caller
    mock_caller = MagicMock()
    # Returns valid JSON structure
    mock_caller.call = AsyncMock(return_value='{"scores": {"relevance": 0.9, "completeness": 0.9, "clarity": 0.9, "instruction_adherence": 0.9}, "overall": 0.9, "reasoning": "Good"}')
    
    evaluator = LLMJudgeEvaluator(config=JudgeConfig(judge_model_id="test"), caller=mock_caller)
    pipeline = EvaluationPipeline(evaluator=evaluator)
    
    req = LLMRequest(request_id="judge-1", prompt="test", model_id="auto")
    resp = LLMResponse(request_id="judge-1", model_id="test", provider="test", output="test")
    
    result = await pipeline.run(req, resp)
    
    assert result.evaluation_status == EvaluationStatus.SUCCESS
    assert result.evaluation_source == EvaluationSource.LLM_JUDGE
    assert result.quality_score == 0.9
    assert result.decision == QualityDecision.PASS
    mock_caller.call.assert_called_once()


@pytest.mark.asyncio
async def test_mock_deepeval_pipeline():
    from app.evaluation.deepeval_adapter import DeepEvalEvaluator, DeepEvalConfig
    
    # We must patch DeepEvalEvaluator.evaluate to avoid trying to import/run real deepeval
    with patch("app.evaluation.deepeval_adapter.DeepEvalEvaluator.__init__", return_value=None):
        evaluator = DeepEvalEvaluator.__new__(DeepEvalEvaluator)
        evaluator._config = DeepEvalConfig()
        
        with patch.object(evaluator, "evaluate", new=AsyncMock()) as mock_eval:
            mock_eval.return_value = EvaluationResult(
                request_id="de-1", model_id="test", provider="test",
                status=EvaluationStatus.SUCCESS, source=EvaluationSource.DEEPEVAL,
                overall_quality_score=0.45,  # escalates with default 0.4 escalate threshold
                individual_scores={"faithfulness": 0.45}
            )
            
            pipeline = EvaluationPipeline(evaluator=evaluator)
            req = LLMRequest(request_id="de-1", prompt="test", model_id="auto")
            resp = LLMResponse(request_id="de-1", model_id="test", provider="test", output="test")
            
            result = await pipeline.run(req, resp)
            
            assert result.evaluation_status == EvaluationStatus.SUCCESS
            assert result.evaluation_source == EvaluationSource.DEEPEVAL
            assert result.decision == QualityDecision.ESCALATE


# ── Pipeline Failure Cases ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_evaluator_failure_distribution(dataset_path, output_dir):
    """If the evaluator fails constantly, the benchmark must record failures without crashing."""
    # We patch the DeterministicEvaluator.evaluate to always fail
    with patch("app.evaluation.deterministic.DeterministicEvaluator.evaluate", new=AsyncMock()) as mock_eval:
        mock_eval.return_value = EvaluationResult(
            request_id="err", model_id="err", provider="err",
            status=EvaluationStatus.FAILURE, source=EvaluationSource.DETERMINISTIC,
            error="Mock failure"
        )
        
        out_file = await run_benchmark(dataset_path, output_dir)
        
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    summary = data["summary"]
    # All cases should fail evaluation
    assert summary["status_distribution"]["FAILURE"] == 2
    # Failed evaluations lead to FAIL decisions
    assert summary["decision_distribution"]["FAIL"] == 2
    
    # Ensure reason string mentions the failure
    for case in data["cases"]:
        assert "Evaluation failed" in case["reason"]
