"""
tests/test_ollama_provider.py
──────────────────────────────
Unit tests for OllamaProvider.

All HTTP interactions are mocked via unittest.mock — Ollama does NOT need
to be running to execute these tests.

Test coverage:
    - Successful inference response (happy path).
    - Token extraction (prompt_eval_count / eval_count).
    - Zero-cost calculation for local models.
    - finish_reason derived from done=True (done_reason absent).
    - finish_reason taken from done_reason when present.
    - system_prompt included in messages payload.
    - No system_prompt → only user message sent.
    - stream=False is always set in payload.
    - options dict contains temperature and num_predict.
    - provider_name is 'ollama'.
    - latency_ms is populated.
    - HTTP 404 → ProviderResponseError (model not pulled).
    - HTTP 500 → ProviderUnavailableError.
    - HTTP 400 → ProviderResponseError.
    - Timeout → ProviderTimeoutError.
    - ConnectError → ProviderUnavailableError (Ollama not running).
    - Non-JSON body → ProviderResponseError.
    - Missing 'message' field → ProviderResponseError.
    - health_check True when server responds.
    - health_check False on ConnectError.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.models.config import ModelConfig, QualityTier
from app.providers.base import LLMRequest, LLMResponse
from app.providers.ollama import OllamaProvider


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def model_config() -> ModelConfig:
    """A minimal ModelConfig for a local Ollama model with zero-cost pricing."""
    return ModelConfig(
        model_id="gemma3:4b",
        provider="ollama",
        display_name="Gemma 3 4B (Local Test)",
        input_cost_per_1k_tokens=0.0,
        output_cost_per_1k_tokens=0.0,
        average_latency_ms=3000.0,
        quality_tier=QualityTier.TIER_1,
        context_window=8192,
        enabled=True,
    )


@pytest.fixture
def llm_request() -> LLMRequest:
    """A minimal LLMRequest targeting an Ollama model."""
    return LLMRequest(
        request_id="test-ollama-001",
        prompt="What is the capital of France?",
        model_id="gemma3:4b",
        temperature=0.0,
        max_tokens=128,
    )


@pytest.fixture
def llm_request_with_system(llm_request: LLMRequest) -> LLMRequest:
    return llm_request.model_copy(
        update={"system_prompt": "You are a helpful assistant.", "request_id": "test-ollama-002"}
    )


def _make_provider(model_config: ModelConfig) -> OllamaProvider:
    """Create an OllamaProvider with the test base URL patched in."""
    with patch("app.providers.ollama.settings") as mock_settings:
        mock_settings.ollama_base_url = "http://localhost:11434"
        provider = OllamaProvider(model_config=model_config)
    return provider


def _ok_response_body(
    content: str = "Paris",
    model_id: str = "gemma3:4b",
    prompt_eval_count: int = 10,
    eval_count: int = 5,
    done: bool = True,
    done_reason: str | None = None,
) -> dict:
    """Build a minimal valid Ollama /api/chat non-streaming response."""
    body = {
        "model": model_id,
        "message": {"role": "assistant", "content": content},
        "done": done,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    if done_reason is not None:
        body["done_reason"] = done_reason
    return body


def _mock_httpx_post(
    status_code: int,
    body: dict | str | None = None,
    raise_exc: Exception | None = None,
) -> AsyncMock:
    """
    Build a mock replacing httpx.AsyncClient used in OllamaProvider._post().

    The context-manager usage pattern is:
        async with httpx.AsyncClient(...) as client:
            resp = await client.post(...)
    """
    mock_client = AsyncMock()

    if raise_exc is not None:
        mock_client.post = AsyncMock(side_effect=raise_exc)
    else:
        mock_response = MagicMock()
        mock_response.status_code = status_code

        if isinstance(body, dict):
            mock_response.json.return_value = body
            mock_response.text = json.dumps(body)
        elif isinstance(body, str):
            mock_response.json.side_effect = ValueError("Not JSON")
            mock_response.text = body
        else:
            mock_response.json.return_value = {}
            mock_response.text = ""

        mock_client.post = AsyncMock(return_value=mock_response)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


# ── Happy-path tests ──────────────────────────────────────────────────────────

class TestOllamaProviderHappyPath:

    @pytest.mark.asyncio
    async def test_successful_response_returns_llm_response(
        self, model_config, llm_request
    ):
        """generate() returns a valid LLMResponse on a 200 OK."""
        provider = _make_provider(model_config)
        body = _ok_response_body(content="Paris")

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert isinstance(response, LLMResponse)
        assert response.output == "Paris"
        assert response.provider == "ollama"
        assert response.request_id == "test-ollama-001"
        assert response.is_success is True
        assert response.is_error is False

    @pytest.mark.asyncio
    async def test_token_counts_extracted_correctly(self, model_config, llm_request):
        """prompt_eval_count → input_tokens, eval_count → output_tokens."""
        provider = _make_provider(model_config)
        body = _ok_response_body(prompt_eval_count=42, eval_count=17)

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert response.input_tokens == 42
        assert response.output_tokens == 17
        assert response.total_tokens == 59

    @pytest.mark.asyncio
    async def test_zero_cost_for_local_model(self, model_config, llm_request):
        """estimated_cost is $0.00 when model pricing is 0.0/0.0."""
        provider = _make_provider(model_config)
        body = _ok_response_body(prompt_eval_count=500, eval_count=200)

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert response.estimated_cost == 0.0

    @pytest.mark.asyncio
    async def test_finish_reason_from_done_true(self, model_config, llm_request):
        """When done=True and done_reason absent, finish_reason is 'stop'."""
        provider = _make_provider(model_config)
        body = _ok_response_body(done=True, done_reason=None)

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_finish_reason_from_done_reason_field(self, model_config, llm_request):
        """When done_reason is present, it is used directly."""
        provider = _make_provider(model_config)
        body = _ok_response_body(done=True, done_reason="length")

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert response.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_finish_reason_none_when_not_done(self, model_config, llm_request):
        """When done=False and done_reason absent, finish_reason is None."""
        provider = _make_provider(model_config)
        body = _ok_response_body(done=False, done_reason=None)

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert response.finish_reason is None

    @pytest.mark.asyncio
    async def test_model_id_from_response_used(self, model_config, llm_request):
        """model_id returned by Ollama is reflected in the response."""
        provider = _make_provider(model_config)
        body = _ok_response_body(model_id="gemma3:4b-instruct")

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert response.model_id == "gemma3:4b-instruct"

    @pytest.mark.asyncio
    async def test_latency_ms_is_positive(self, model_config, llm_request):
        """latency_ms is a non-negative float."""
        provider = _make_provider(model_config)
        body = _ok_response_body()

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert response.latency_ms >= 0.0

    def test_provider_name_is_ollama(self, model_config):
        """provider_name property returns 'ollama'."""
        provider = _make_provider(model_config)
        assert provider.provider_name == "ollama"

    @pytest.mark.asyncio
    async def test_system_prompt_included_in_messages(
        self, model_config, llm_request_with_system
    ):
        """system_prompt → first message with role=system."""
        provider = _make_provider(model_config)
        body = _ok_response_body()
        captured: dict = {}

        async def capture_post(url, json, headers):
            captured.update(json)
            mock_r = MagicMock()
            mock_r.status_code = 200
            mock_r.json.return_value = body
            return mock_r

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            await provider.generate(llm_request_with_system)

        messages = captured.get("messages", [])
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_system_prompt_sends_only_user(self, model_config, llm_request):
        """Without system_prompt, only one user message is sent."""
        provider = _make_provider(model_config)
        body = _ok_response_body()
        captured: dict = {}

        async def capture_post(url, json, headers):
            captured.update(json)
            mock_r = MagicMock()
            mock_r.status_code = 200
            mock_r.json.return_value = body
            return mock_r

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            await provider.generate(llm_request)

        messages = captured.get("messages", [])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_stream_is_false_in_payload(self, model_config, llm_request):
        """stream=False is always sent in the payload."""
        provider = _make_provider(model_config)
        body = _ok_response_body()
        captured: dict = {}

        async def capture_post(url, json, headers):
            captured.update(json)
            mock_r = MagicMock()
            mock_r.status_code = 200
            mock_r.json.return_value = body
            return mock_r

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            await provider.generate(llm_request)

        assert captured.get("stream") is False

    @pytest.mark.asyncio
    async def test_options_contains_temperature_and_num_predict(
        self, model_config, llm_request
    ):
        """options dict carries temperature and num_predict."""
        provider = _make_provider(model_config)
        body = _ok_response_body()
        captured: dict = {}

        async def capture_post(url, json, headers):
            captured.update(json)
            mock_r = MagicMock()
            mock_r.status_code = 200
            mock_r.json.return_value = body
            return mock_r

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            await provider.generate(llm_request)

        options = captured.get("options", {})
        assert options.get("temperature") == llm_request.temperature
        assert options.get("num_predict") == llm_request.max_tokens

    @pytest.mark.asyncio
    async def test_missing_token_fields_default_to_zero(
        self, model_config, llm_request
    ):
        """When prompt_eval_count and eval_count are absent, tokens default to 0."""
        provider = _make_provider(model_config)
        body = {
            "model": "gemma3:4b",
            "message": {"role": "assistant", "content": "Paris"},
            "done": True,
            # prompt_eval_count and eval_count intentionally omitted
        }

        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, body)):
            response = await provider.generate(llm_request)

        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.total_tokens == 0


# ── Error-handling tests ──────────────────────────────────────────────────────

class TestOllamaProviderErrorHandling:

    @pytest.mark.asyncio
    async def test_http_404_raises_response_error(self, model_config, llm_request):
        """HTTP 404 (model not pulled) → ProviderResponseError."""
        provider = _make_provider(model_config)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(404, "Not found")):
            with pytest.raises(ProviderResponseError, match="not found"):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_http_500_raises_unavailable_error(self, model_config, llm_request):
        """HTTP 500 → ProviderUnavailableError."""
        provider = _make_provider(model_config)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(500, "Server Error")):
            with pytest.raises(ProviderUnavailableError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_http_400_raises_response_error(self, model_config, llm_request):
        """HTTP 400 → ProviderResponseError."""
        provider = _make_provider(model_config)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(400, {})):
            with pytest.raises(ProviderResponseError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self, model_config, llm_request):
        """httpx.TimeoutException → ProviderTimeoutError."""
        provider = _make_provider(model_config)
        exc = httpx.TimeoutException("timed out")
        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(0, raise_exc=exc)):
            with pytest.raises(ProviderTimeoutError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_connect_error_raises_unavailable(self, model_config, llm_request):
        """httpx.ConnectError → ProviderUnavailableError (Ollama not running)."""
        provider = _make_provider(model_config)
        exc = httpx.ConnectError("connection refused")
        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(0, raise_exc=exc)):
            with pytest.raises(ProviderUnavailableError, match="Ollama"):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_non_json_response_raises_response_error(
        self, model_config, llm_request
    ):
        """Non-JSON body → ProviderResponseError."""
        provider = _make_provider(model_config)
        with patch(
            "httpx.AsyncClient",
            return_value=_mock_httpx_post(200, "not json at all"),
        ):
            with pytest.raises(ProviderResponseError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_missing_message_field_raises_response_error(
        self, model_config, llm_request
    ):
        """Response body without 'message' → ProviderResponseError."""
        provider = _make_provider(model_config)
        bad_body = {"model": "gemma3:4b", "done": True}  # No 'message'
        with patch("httpx.AsyncClient", return_value=_mock_httpx_post(200, bad_body)):
            with pytest.raises(ProviderResponseError, match="'message'"):
                await provider.generate(llm_request)


# ── Health check tests ────────────────────────────────────────────────────────

class TestOllamaProviderHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_reachable(self, model_config):
        """health_check() returns True when Ollama server responds."""
        provider = _make_provider(model_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await provider.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_connect_error(self, model_config):
        """health_check() returns False when Ollama is not running."""
        provider = _make_provider(model_config)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await provider.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_timeout(self, model_config):
        """health_check() returns False on timeout."""
        provider = _make_provider(model_config)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await provider.health_check()

        assert result is False
