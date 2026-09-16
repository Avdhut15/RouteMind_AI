"""
tests/test_phase2_runner.py
─────────────────────────────
Unit tests for the Phase 2 Benchmark Runner.

Ensures the IntelligentRouter validation framework works as expected without
making actual network calls.

Coverage:
    - End-to-end router + benchmark integration
    - Result schema and serialisation
    - Successful prompt processing
    - Failure handling (routing failure, provider failure)
    - Summary computation (tiers, models)
    - Output generation
"""

import json
import pathlib
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.classification.complexity_classifier import BaseComplexityClassifier, ComplexityPrediction, ComplexityLabel
from app.models.config import ModelConfig, QualityTier
from app.models.registry import ModelRegistry
from app.providers.base import LLMRequest, LLMResponse
from app.providers.provider_registry import ProviderRegistry
from app.routing.analyzer import RequestAnalysis, TaskType
from app.routing.candidate_selector import CandidateSelectionResult
from app.routing.result import RouterResult
from app.routing.router import IntelligentRouter
from benchmarks.phase2_runner import (
    _compute_summary,
    _empty_prompt_result,
    load_prompts,
    run_benchmark,
    run_single_prompt,
    save_results,
)


# ── Mocks and Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_router():
    router = MagicMock(spec=IntelligentRouter)
    # Default success response
    success_result = RouterResult(
        request_id="test_req",
        status="success",
        request_analysis=MagicMock(spec=RequestAnalysis, task_type=TaskType.GENERAL_QA),
        complexity=ComplexityLabel.SIMPLE,
        classifier_confidence=0.9,
        classifier_model_id="mock_clf",
        candidate_model_ids=["mock_model_1", "mock_model_2"],
        selected_model_id="mock_model_1",
        selected_provider="mock_provider",
        routing_score=0.85,
        factor_scores={"cost": 0.5, "quality": 0.8},
        llm_response=MagicMock(spec=LLMResponse),
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        estimated_cost=0.0001,
        latency_ms=150.0,
        finish_reason="stop",
    )
    router.route = AsyncMock(return_value=success_result)
    return router


@pytest.fixture
def temp_prompts_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(
            {
                "meta": {"version": "1.0"},
                "prompts": [
                    {"id": "p1", "tier": "tier_1", "task_type": "general_qa", "prompt": "Test 1"},
                    {"id": "p2", "tier": "tier_2", "task_type": "reasoning", "prompt": "Test 2"},
                ],
            },
            f,
        )
        path = pathlib.Path(f.name)
    yield path
    path.unlink()


@pytest.fixture
def temp_results_dir():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadPrompts:
    def test_loads_valid_json(self, temp_prompts_file):
        prompts = load_prompts(temp_prompts_file)
        assert len(prompts) == 2
        assert prompts[0]["id"] == "p1"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_prompts(pathlib.Path("nonexistent.json"))

    def test_raises_on_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("{ invalid json")
            path = pathlib.Path(f.name)
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(path)
        path.unlink()

    def test_raises_on_missing_prompts_key(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump({"meta": "only"}, f)
            path = pathlib.Path(f.name)
            
        with pytest.raises(ValueError, match="must contain a non-empty"):
            load_prompts(path)
        path.unlink()


class TestRunSinglePrompt:
    @pytest.mark.asyncio
    async def test_success_path(self, mock_router):
        prompt_entry = {"id": "p1", "tier": "tier_1", "task_type": "qa", "prompt": "hello"}
        result = await run_single_prompt(prompt_entry, mock_router, 100, 0.5)

        assert result["prompt_id"] == "p1"
        assert result["status"] == "success"
        assert result["success"] is True
        assert result["predicted_complexity"] == "simple"
        assert result["candidate_models"] == ["mock_model_1", "mock_model_2"]
        assert result["selected_model_id"] == "mock_model_1"
        assert result["total_tokens"] == 30
        assert result["latency_ms"] == 150.0

    @pytest.mark.asyncio
    async def test_empty_prompt_returns_error(self, mock_router):
        prompt_entry = {"id": "p1", "tier": "tier_1", "task_type": "qa", "prompt": ""}
        result = await run_single_prompt(prompt_entry, mock_router, 100, 0.5)

        assert result["status"] == "dataset_error"
        assert result["success"] is False
        assert "Empty prompt" in result["error"]
        assert not mock_router.route.called

    @pytest.mark.asyncio
    async def test_routing_failure_handled(self, mock_router):
        fail_result = RouterResult(
            request_id="test",
            status="routing_failure",
            error="No eligible candidates",
            complexity=ComplexityLabel.COMPLEX,
        )
        mock_router.route = AsyncMock(return_value=fail_result)
        
        prompt_entry = {"id": "p1", "prompt": "hello"}
        result = await run_single_prompt(prompt_entry, mock_router, 100, 0.5)
        
        assert result["status"] == "routing_failure"
        assert result["success"] is False
        assert result["selected_model_id"] is None
        assert "No eligible candidates" in result["error"]

    @pytest.mark.asyncio
    async def test_unexpected_exception_caught(self, mock_router):
        mock_router.route.side_effect = Exception("Boom")
        
        prompt_entry = {"id": "p1", "prompt": "hello"}
        result = await run_single_prompt(prompt_entry, mock_router, 100, 0.5)
        
        assert result["status"] == "unexpected_error"
        assert result["success"] is False
        assert "Boom" in result["error"]


class TestComputeSummary:
    def test_computes_correct_aggregates(self):
        results = [
            {"success": True, "status": "success", "latency_ms": 100, "estimated_cost_usd": 0.01, "total_tokens": 10, "input_tokens": 5, "output_tokens": 5, "tier": "tier_1", "selected_model_id": "model_A"},
            {"success": True, "status": "success", "latency_ms": 200, "estimated_cost_usd": 0.02, "total_tokens": 20, "input_tokens": 10, "output_tokens": 10, "tier": "tier_2", "selected_model_id": "model_B"},
            {"success": False, "status": "routing_failure", "latency_ms": 0, "estimated_cost_usd": 0, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0, "tier": "tier_3", "selected_model_id": None},
        ]
        
        summary = _compute_summary(results)
        
        assert summary["total_prompts"] == 3
        assert summary["succeeded"] == 2
        assert summary["failed"] == 1
        assert summary["routing_failures"] == 1
        assert summary["success_rate"] == round(2/3, 4)
        assert summary["avg_latency_ms"] == 150.0
        assert summary["total_tokens"] == 30
        assert summary["total_estimated_cost_usd"] == 0.03
        assert summary["by_model"]["model_A"] == 1
        assert summary["by_model"]["model_B"] == 1
        assert summary["by_tier"]["tier_1"]["succeeded"] == 1
        assert summary["by_tier"]["tier_3"]["failed"] == 1


class TestSaveResults:
    def test_creates_file_and_valid_schema(self, temp_results_dir):
        results = [
            _empty_prompt_result("p1", "tier_1", "qa")
        ]
        started = datetime.now(timezone.utc)
        finished = datetime.now(timezone.utc)
        
        out_path = save_results(results, temp_results_dir, started, finished)
        
        assert out_path.exists()
        
        data = json.loads(out_path.read_text())
        assert data["_meta"]["benchmark_version"] == "2.0"
        assert len(data["results"]) == 1
        assert "summary" in data


class TestRunBenchmark:
    @patch("benchmarks.phase2_runner.load_prompts")
    @patch("benchmarks.phase2_runner.ModelRegistry")
    @patch("benchmarks.phase2_runner.ProviderRegistry")
    @patch("benchmarks.phase2_runner.IntelligentRouter")
    @pytest.mark.asyncio
    async def test_end_to_end(
        self,
        mock_router_cls,
        mock_provider_reg,
        mock_model_reg,
        mock_load,
        temp_results_dir
    ):
        # Setup mocks
        mock_load.return_value = [{"id": "p1", "prompt": "test", "tier": "tier_1"}]
        
        mock_router_instance = MagicMock()
        mock_router_instance.route = AsyncMock(return_value=RouterResult(
            request_id="test",
            status="success",
            complexity=ComplexityLabel.SIMPLE,
            selected_model_id="test_model",
        ))
        mock_router_cls.return_value = mock_router_instance
        
        # Run
        out_path = await run_benchmark(
            prompts_file=pathlib.Path("dummy.json"),
            results_dir=temp_results_dir
        )
        
        assert out_path.exists()
        
        data = json.loads(out_path.read_text())
        assert data["summary"]["total_prompts"] == 1
        assert data["results"][0]["selected_model_id"] == "test_model"
