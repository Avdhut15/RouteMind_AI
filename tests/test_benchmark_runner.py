"""
tests/test_benchmark_runner.py
────────────────────────────────
Unit tests for benchmarks/runner.py.

All provider interactions are mocked — no real API key, network access,
or running Ollama server is required.

Test coverage:
    - load_prompts() loads the dataset and returns 30 entries.
    - load_prompts() raises FileNotFoundError for missing file.
    - load_prompts() raises ValueError for malformed JSON.
    - load_prompts() raises ValueError when 'prompts' key is absent.
    - build_provider() returns OpenRouterProvider for 'openrouter'.
    - build_provider() returns OllamaProvider for 'ollama'.
    - build_provider() raises ValueError for unsupported provider.
    - run_single_prompt() records success fields correctly.
    - run_single_prompt() records failure without raising.
    - run_single_prompt() handles empty prompt_text gracefully.
    - run_benchmark() executes all 30 prompts.
    - run_benchmark() records provider and model_id in every result.
    - run_benchmark() records token, cost, and latency fields.
    - run_benchmark() continues after individual prompt failure.
    - run_benchmark() saves a JSON result file.
    - run_benchmark() result filename is timestamped.
    - save_results() writes parseable JSON.
    - save_results() result file contains expected top-level keys.
    - _compute_summary() counts succeeded/failed correctly.
    - _safe_filename_part() removes characters unsafe for filenames.
"""

import json
import pathlib
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import ProviderUnavailableError
from app.models.config import ModelConfig, QualityTier
from app.providers.base import LLMRequest, LLMResponse
from benchmarks.runner import (
    _compute_summary,
    _empty_prompt_result,
    _safe_filename_part,
    build_provider,
    load_prompts,
    run_benchmark,
    run_single_prompt,
    save_results,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

REAL_PROMPTS_FILE = pathlib.Path("benchmarks/prompts.json")

@pytest.fixture
def openrouter_model_config() -> ModelConfig:
    return ModelConfig(
        model_id="openai/gpt-oss-20b:free",
        provider="openrouter",
        display_name="Test OpenRouter Model",
        input_cost_per_1k_tokens=0.0001,
        output_cost_per_1k_tokens=0.0002,
        quality_tier=QualityTier.TIER_2,
        context_window=16000,
        enabled=True,
    )


@pytest.fixture
def ollama_model_config() -> ModelConfig:
    return ModelConfig(
        model_id="gemma3:4b",
        provider="ollama",
        display_name="Test Ollama Model",
        input_cost_per_1k_tokens=0.0,
        output_cost_per_1k_tokens=0.0,
        quality_tier=QualityTier.TIER_1,
        context_window=8192,
        enabled=True,
    )


@pytest.fixture
def sample_prompt_entry() -> dict:
    return {
        "id": "s001",
        "tier": "tier_1",
        "task_type": "extraction",
        "prompt": "Extract the email from: contact john@example.com for details.",
    }


def _make_success_response(model_id: str = "openai/gpt-oss-20b:free") -> LLMResponse:
    return LLMResponse(
        request_id="req-test",
        output="Extracted: john@example.com",
        model_id=model_id,
        provider="openrouter",
        input_tokens=25,
        output_tokens=10,
        total_tokens=35,
        latency_ms=450.0,
        estimated_cost=0.000005,
        finish_reason="stop",
        error=None,
    )


def _make_minimal_prompts(n: int = 5) -> list[dict]:
    """Generate n minimal prompt entries for testing."""
    return [
        {
            "id": f"t{i:03d}",
            "tier": "tier_1",
            "task_type": "qa",
            "prompt": f"Test prompt number {i}.",
        }
        for i in range(1, n + 1)
    ]


# ── load_prompts() tests ──────────────────────────────────────────────────────

class TestLoadPrompts:

    def test_loads_real_dataset_returns_30_entries(self):
        """load_prompts() with the real file returns exactly 30 prompts."""
        prompts = load_prompts(REAL_PROMPTS_FILE)
        assert len(prompts) == 30

    def test_loads_real_dataset_returns_list_of_dicts(self):
        """Each entry in the loaded dataset is a dict."""
        prompts = load_prompts(REAL_PROMPTS_FILE)
        assert all(isinstance(p, dict) for p in prompts)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """load_prompts() raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_prompts(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_value_error(self, tmp_path):
        """load_prompts() raises ValueError for malformed JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{{", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(bad_file)

    def test_missing_prompts_key_raises_value_error(self, tmp_path):
        """load_prompts() raises ValueError when 'prompts' key is absent."""
        bad_file = tmp_path / "no_prompts.json"
        bad_file.write_text(json.dumps({"_meta": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="'prompts'"):
            load_prompts(bad_file)

    def test_empty_prompts_list_raises_value_error(self, tmp_path):
        """load_prompts() raises ValueError when 'prompts' is an empty list."""
        bad_file = tmp_path / "empty.json"
        bad_file.write_text(json.dumps({"prompts": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_prompts(bad_file)


# ── build_provider() tests ────────────────────────────────────────────────────

class TestBuildProvider:

    def test_openrouter_returns_openrouter_provider(self, openrouter_model_config):
        """build_provider() returns OpenRouterProvider for openrouter config."""
        from app.providers.openrouter import OpenRouterProvider
        with patch("app.providers.openrouter.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-test-fake"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            provider = build_provider(openrouter_model_config, timeout_s=30.0)
        assert isinstance(provider, OpenRouterProvider)

    def test_ollama_returns_ollama_provider(self, ollama_model_config):
        """build_provider() returns OllamaProvider for ollama config."""
        from app.providers.ollama import OllamaProvider
        with patch("app.providers.ollama.settings") as mock_settings:
            mock_settings.ollama_base_url = "http://localhost:11434"
            provider = build_provider(ollama_model_config, timeout_s=120.0)
        assert isinstance(provider, OllamaProvider)

    def test_unsupported_provider_raises_value_error(self):
        """build_provider() raises ValueError for unsupported provider name."""
        fake_config = ModelConfig(
            model_id="some-model",
            provider="groq",
            display_name="Groq Test",
            quality_tier=QualityTier.TIER_2,
        )
        with pytest.raises(ValueError, match="groq"):
            build_provider(fake_config, timeout_s=30.0)


# ── run_single_prompt() tests ─────────────────────────────────────────────────

class TestRunSinglePrompt:

    @pytest.mark.asyncio
    async def test_success_records_correct_fields(
        self, sample_prompt_entry, openrouter_model_config
    ):
        """Successful provider call fills all metric fields in result."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=_make_success_response()
        )

        result = await run_single_prompt(
            prompt_entry=sample_prompt_entry,
            provider=mock_provider,
            model_config=openrouter_model_config,
            max_tokens=256,
            temperature=0.2,
        )

        assert result["success"] is True
        assert result["prompt_id"] == "s001"
        assert result["tier"] == "tier_1"
        assert result["task_type"] == "extraction"
        assert result["model_id"] == openrouter_model_config.model_id
        assert result["provider"] == openrouter_model_config.provider
        assert result["input_tokens"] == 25
        assert result["output_tokens"] == 10
        assert result["total_tokens"] == 35
        assert result["latency_ms"] == 450.0
        assert result["estimated_cost_usd"] == 0.000005
        assert result["finish_reason"] == "stop"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_provider_exception_records_failure_does_not_raise(
        self, sample_prompt_entry, openrouter_model_config
    ):
        """Provider exception is caught; result is recorded as failure."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            side_effect=ProviderUnavailableError("server down")
        )

        result = await run_single_prompt(
            prompt_entry=sample_prompt_entry,
            provider=mock_provider,
            model_config=openrouter_model_config,
            max_tokens=256,
            temperature=0.2,
        )

        assert result["success"] is False
        assert "ProviderUnavailableError" in result["error"]
        assert "server down" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_prompt_text_records_failure(
        self, openrouter_model_config
    ):
        """Empty prompt text returns a failure result without calling the provider."""
        bad_entry = {"id": "bad001", "tier": "tier_1", "task_type": "qa", "prompt": ""}
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock()

        result = await run_single_prompt(
            prompt_entry=bad_entry,
            provider=mock_provider,
            model_config=openrouter_model_config,
            max_tokens=256,
            temperature=0.2,
        )

        assert result["success"] is False
        assert result["error"] is not None
        mock_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_does_not_contain_raw_prompt(
        self, sample_prompt_entry, openrouter_model_config
    ):
        """The result dict must not contain the raw prompt text."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=_make_success_response())

        result = await run_single_prompt(
            prompt_entry=sample_prompt_entry,
            provider=mock_provider,
            model_config=openrouter_model_config,
            max_tokens=256,
            temperature=0.2,
        )

        # The raw prompt text must not appear as a value in the result dict
        raw_prompt = sample_prompt_entry["prompt"]
        assert raw_prompt not in result.values()


# ── run_benchmark() integration tests ─────────────────────────────────────────

class TestRunBenchmark:

    def _make_mock_provider(self, response: LLMResponse | Exception) -> MagicMock:
        mock = MagicMock()
        if isinstance(response, Exception):
            mock.generate = AsyncMock(side_effect=response)
        else:
            mock.generate = AsyncMock(return_value=response)
        return mock

    def _mock_registry_and_provider(self, model_config: ModelConfig, mock_provider):
        """Patch ModelRegistry.get() and build_provider() together."""
        registry_patch = patch(
            "benchmarks.runner.ModelRegistry",
            return_value=MagicMock(get=MagicMock(return_value=model_config)),
        )
        provider_patch = patch(
            "benchmarks.runner.build_provider",
            return_value=mock_provider,
        )
        return registry_patch, provider_patch

    @pytest.mark.asyncio
    async def test_all_30_prompts_are_executed(
        self, openrouter_model_config, tmp_path
    ):
        """run_benchmark() executes all 30 prompts from the real dataset."""
        success_response = _make_success_response()
        mock_provider = self._make_mock_provider(success_response)
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        assert mock_provider.generate.call_count == 30

    @pytest.mark.asyncio
    async def test_result_file_is_created(
        self, openrouter_model_config, tmp_path
    ):
        """run_benchmark() creates exactly one result JSON file."""
        mock_provider = self._make_mock_provider(_make_success_response())
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        assert output_path.exists()
        assert output_path.suffix == ".json"

    @pytest.mark.asyncio
    async def test_result_filename_contains_timestamp(
        self, openrouter_model_config, tmp_path
    ):
        """Result filename contains a timestamp fragment (YYYYMMDD_HHMMSS)."""
        import re
        mock_provider = self._make_mock_provider(_make_success_response())
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        assert re.search(r"\d{8}_\d{6}", output_path.name), (
            f"Expected timestamp in filename: {output_path.name}"
        )

    @pytest.mark.asyncio
    async def test_result_json_has_expected_top_level_keys(
        self, openrouter_model_config, tmp_path
    ):
        """Result JSON contains '_meta', 'summary', and 'results' keys."""
        mock_provider = self._make_mock_provider(_make_success_response())
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "_meta" in data
        assert "summary" in data
        assert "results" in data

    @pytest.mark.asyncio
    async def test_result_contains_30_prompt_results(
        self, openrouter_model_config, tmp_path
    ):
        """result['results'] contains exactly 30 entries."""
        mock_provider = self._make_mock_provider(_make_success_response())
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data["results"]) == 30

    @pytest.mark.asyncio
    async def test_every_result_has_provider_and_model_id(
        self, openrouter_model_config, tmp_path
    ):
        """All result entries have provider and model_id fields set."""
        mock_provider = self._make_mock_provider(_make_success_response())
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        data = json.loads(output_path.read_text(encoding="utf-8"))
        for r in data["results"]:
            assert r.get("provider") == openrouter_model_config.provider
            assert r.get("model_id") == openrouter_model_config.model_id

    @pytest.mark.asyncio
    async def test_every_result_has_token_cost_latency_fields(
        self, openrouter_model_config, tmp_path
    ):
        """All result entries have input_tokens, output_tokens, estimated_cost_usd, latency_ms."""
        mock_provider = self._make_mock_provider(_make_success_response())
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        data = json.loads(output_path.read_text(encoding="utf-8"))
        for r in data["results"]:
            assert "input_tokens" in r
            assert "output_tokens" in r
            assert "estimated_cost_usd" in r
            assert "latency_ms" in r

    @pytest.mark.asyncio
    async def test_failed_prompt_recorded_run_continues(
        self, openrouter_model_config, tmp_path
    ):
        """A single provider failure is recorded; remaining prompts still execute."""
        call_count = 0
        async def flaky_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count == 5:
                raise ProviderUnavailableError("simulated failure on prompt 5")
            return _make_success_response()

        mock_provider = MagicMock()
        mock_provider.generate = flaky_generate
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data["results"]) == 30
        assert data["summary"]["failed"] == 1
        assert data["summary"]["succeeded"] == 29

    @pytest.mark.asyncio
    async def test_all_failures_does_not_crash_runner(
        self, openrouter_model_config, tmp_path
    ):
        """If all prompts fail, the runner still completes and writes a result file."""
        mock_provider = self._make_mock_provider(
            ProviderUnavailableError("all down")
        )
        registry_patch, provider_patch = self._mock_registry_and_provider(
            openrouter_model_config, mock_provider
        )

        with registry_patch, provider_patch:
            output_path = await run_benchmark(
                model_id=openrouter_model_config.model_id,
                provider_name="openrouter",
                prompts_file=REAL_PROMPTS_FILE,
                results_dir=tmp_path / "results",
            )

        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["summary"]["failed"] == 30
        assert data["summary"]["succeeded"] == 0


# ── save_results() tests ──────────────────────────────────────────────────────

class TestSaveResults:

    def test_creates_results_dir_if_absent(self, tmp_path):
        """save_results() creates results_dir if it does not exist."""
        new_dir = tmp_path / "new_results_dir"
        assert not new_dir.exists()

        results = [_empty_prompt_result("p1", "tier_1", "qa", "model-x", "openrouter")]
        now = datetime.now(timezone.utc)
        save_results(results, "model-x", "openrouter", new_dir, now, now)

        assert new_dir.exists()

    def test_output_is_valid_json(self, tmp_path):
        """Written result file parses as valid JSON."""
        results = [_empty_prompt_result("p1", "tier_1", "qa", "model-x", "openrouter")]
        now = datetime.now(timezone.utc)
        output_path = save_results(results, "model-x", "openrouter", tmp_path, now, now)

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_file_contains_meta_summary_results(self, tmp_path):
        """Result file has _meta, summary, and results keys."""
        results = [_empty_prompt_result("p1", "tier_1", "qa", "model-x", "openrouter")]
        now = datetime.now(timezone.utc)
        output_path = save_results(results, "model-x", "openrouter", tmp_path, now, now)

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "_meta" in data
        assert "summary" in data
        assert "results" in data


# ── _compute_summary() tests ──────────────────────────────────────────────────

class TestComputeSummary:

    def test_summary_counts_succeeded_and_failed(self):
        """_compute_summary() correctly counts success and failure."""
        results = [
            {**_empty_prompt_result(f"p{i}", "tier_1", "qa", "m", "p"), "success": i % 2 == 0}
            for i in range(6)
        ]
        summary = _compute_summary(results)
        assert summary["total_prompts"] == 6
        assert summary["succeeded"] == 3
        assert summary["failed"] == 3

    def test_success_rate_calculation(self):
        """success_rate is succeeded / total."""
        results = [
            {**_empty_prompt_result(f"p{i}", "tier_1", "qa", "m", "p"), "success": True}
            for i in range(4)
        ] + [
            {**_empty_prompt_result(f"f{i}", "tier_1", "qa", "m", "p"), "success": False}
            for i in range(1)
        ]
        summary = _compute_summary(results)
        assert abs(summary["success_rate"] - 0.8) < 0.001

    def test_tier_breakdown_present(self):
        """by_tier key groups counts by tier."""
        results = [
            {**_empty_prompt_result("t1", "tier_1", "qa", "m", "p"), "success": True},
            {**_empty_prompt_result("t2", "tier_2", "qa", "m", "p"), "success": False},
        ]
        summary = _compute_summary(results)
        assert "tier_1" in summary["by_tier"]
        assert "tier_2" in summary["by_tier"]


# ── _safe_filename_part() tests ───────────────────────────────────────────────

class TestSafeFilenamePart:

    def test_simple_string_unchanged(self):
        """Simple alphanumeric strings pass through unchanged."""
        assert _safe_filename_part("openrouter") == "openrouter"

    def test_slashes_replaced(self):
        """Forward slashes are replaced with underscores."""
        result = _safe_filename_part("openai/gpt-oss-20b:free")
        assert "/" not in result

    def test_colons_replaced(self):
        """Colons are replaced with underscores."""
        result = _safe_filename_part("model:version")
        assert ":" not in result

    def test_safe_characters_preserved(self):
        """Dots, hyphens, underscores are preserved."""
        result = _safe_filename_part("my-model_v1.0")
        assert result == "my-model_v1.0"
