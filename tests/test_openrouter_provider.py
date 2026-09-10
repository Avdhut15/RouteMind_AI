"""
tests/test_openrouter_provider.py
──────────────────────────────────
Unit tests for OpenRouterProvider.

All HTTP interactions are mocked via unittest.mock — no real API key or
network connection is required to run these tests.

Test coverage:
    - Successful inference response (happy path).
    - Token extraction and cost calculation.
    - HTTP 401 → ProviderAuthError.
    - HTTP 403 → ProviderAuthError.
    - HTTP 429 → ProviderRateLimitError.
    - HTTP 500 → ProviderUnavailableError.
    - HTTP 400 → ProviderResponseError.
    - Timeout → ProviderTimeoutError.
    - Connection error → ProviderUnavailableError.
    - Non-JSON response → ProviderResponseError.
    - Missing 'choices' in response → ProviderResponseError.
    - Missing API key → ConfigurationError.
    - health_check returns True on reachable endpoint.
    - health_check returns False on connection error.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.exceptions import (
    ConfigurationError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.models.config import ModelConfig, QualityTier
from app.providers.base import LLMRequest, LLMResponse
from app.providers.openrouter import OpenRouterProvider


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def model_config() -> ModelConfig:
    """A minimal ModelConfig for a free OpenRouter model."""
    return ModelConfig(
        model_id="openai/gpt-oss-20b:free",
        provider="openrouter",
        display_name="GPT OSS 20B (Test)",
        input_cost_per_1k_tokens=0.0001,
        output_cost_per_1k_tokens=0.0002,
        average_latency_ms=2000.0,
        quality_tier=QualityTier.TIER_2,
        context_window=16000,
        enabled=True,
    )


@pytest.fixture
def llm_request() -> LLMRequest:
    """A minimal LLMRequest."""
    return LLMRequest(
        request_id="test-req-001",
        prompt="What is 2 + 2?",
        model_id="openai/gpt-oss-20b:free",
        temperature=0.0,
        max_tokens=64,
    )


@pytest.fixture
def llm_request_with_system(llm_request: LLMRequest) -> LLMRequest:
    """An LLMRequest that includes a system prompt."""
    return llm_request.model_copy(
        update={"system_prompt": "You are a helpful assistant.", "request_id": "test-req-002"}
    )


def _make_provider(model_config: ModelConfig) -> OpenRouterProvider:
    """Create a provider instance with a fake API key injected via settings patch."""
    with patch("app.providers.openrouter.settings") as mock_settings:
        mock_settings.openrouter_api_key = "sk-or-test-fake-key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        provider = OpenRouterProvider(model_config=model_config)
    # Patch settings on the live instance so all method calls use the fake key.
    provider._headers["Authorization"] = "Bearer sk-or-test-fake-key"
    return provider


def _ok_response_body(
    output: str = "4",
    model_id: str = "openai/gpt-oss-20b:free",
    prompt_tokens: int = 12,
    completion_tokens: int = 3,
) -> dict:
    """Build a minimal valid OpenRouter /chat/completions response body."""
    return {
        "id": "gen-test-123",
        "model": model_id,
        "choices": [
            {
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _mock_httpx_response(
    status_code: int,
    body: dict | str | None = None,
    raise_exc: Exception | None = None,
) -> AsyncMock:
    """
    Build a mock that replaces httpx.AsyncClient used inside OpenRouterProvider.

    The mock patches the async context manager returned by httpx.AsyncClient(...)
    and its .post() method.
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

    # AsyncClient used as: async with httpx.AsyncClient(...) as client:
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOpenRouterProviderHappyPath:

    @pytest.mark.asyncio
    async def test_successful_response_returns_llm_response(
        self, model_config, llm_request
    ):
        """generate() returns a valid LLMResponse on a 200 OK."""
        provider = _make_provider(model_config)
        body = _ok_response_body(output="4", prompt_tokens=12, completion_tokens=3)

        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, body)):
            response = await provider.generate(llm_request)

        assert isinstance(response, LLMResponse)
        assert response.output == "4"
        assert response.finish_reason == "stop"
        assert response.provider == "openrouter"
        assert response.request_id == "test-req-001"
        assert response.is_success is True
        assert response.is_error is False

    @pytest.mark.asyncio
    async def test_token_counts_are_extracted(self, model_config, llm_request):
        """Token counts from 'usage' field are correctly extracted."""
        provider = _make_provider(model_config)
        body = _ok_response_body(prompt_tokens=50, completion_tokens=20)

        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, body)):
            response = await provider.generate(llm_request)

        assert response.input_tokens == 50
        assert response.output_tokens == 20
        assert response.total_tokens == 70

    @pytest.mark.asyncio
    async def test_estimated_cost_is_calculated(self, model_config, llm_request):
        """estimated_cost is computed from token usage and ModelConfig pricing."""
        provider = _make_provider(model_config)
        body = _ok_response_body(prompt_tokens=1000, completion_tokens=1000)

        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, body)):
            response = await provider.generate(llm_request)

        # input:  (1000/1000) * 0.0001 = 0.0001
        # output: (1000/1000) * 0.0002 = 0.0002
        # total:  0.0003
        assert abs(response.estimated_cost - 0.0003) < 1e-9

    @pytest.mark.asyncio
    async def test_model_id_from_response_is_used(self, model_config, llm_request):
        """The model_id returned by OpenRouter (may differ) is used in the response."""
        provider = _make_provider(model_config)
        body = _ok_response_body(model_id="openai/gpt-oss-20b")  # Slightly different

        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, body)):
            response = await provider.generate(llm_request)

        assert response.model_id == "openai/gpt-oss-20b"

    @pytest.mark.asyncio
    async def test_latency_ms_is_populated(self, model_config, llm_request):
        """latency_ms is a positive float."""
        provider = _make_provider(model_config)
        body = _ok_response_body()

        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, body)):
            response = await provider.generate(llm_request)

        assert response.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_system_prompt_is_included_in_payload(
        self, model_config, llm_request_with_system
    ):
        """When system_prompt is set, it appears first in the messages array."""
        provider = _make_provider(model_config)
        body = _ok_response_body()
        captured_payload: dict = {}

        async def capture_post(url, json, headers):
            captured_payload.update(json)
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

        messages = captured_payload.get("messages", [])
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_system_prompt_sends_only_user_message(
        self, model_config, llm_request
    ):
        """When system_prompt is None, only the user message is sent."""
        provider = _make_provider(model_config)
        body = _ok_response_body()
        captured_payload: dict = {}

        async def capture_post(url, json, headers):
            captured_payload.update(json)
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

        messages = captured_payload.get("messages", [])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_provider_name_is_openrouter(self, model_config):
        """provider_name property returns 'openrouter'."""
        provider = _make_provider(model_config)
        assert provider.provider_name == "openrouter"


class TestOpenRouterProviderErrorHandling:

    @pytest.mark.asyncio
    async def test_http_401_raises_auth_error(self, model_config, llm_request):
        """HTTP 401 → ProviderAuthError."""
        provider = _make_provider(model_config)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(401, {})):
            with pytest.raises(ProviderAuthError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_http_403_raises_auth_error(self, model_config, llm_request):
        """HTTP 403 → ProviderAuthError."""
        provider = _make_provider(model_config)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(403, {})):
            with pytest.raises(ProviderAuthError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_http_429_raises_rate_limit_error(self, model_config, llm_request):
        """HTTP 429 → ProviderRateLimitError."""
        provider = _make_provider(model_config)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(429, {})):
            with pytest.raises(ProviderRateLimitError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_http_500_raises_unavailable_error(self, model_config, llm_request):
        """HTTP 500 → ProviderUnavailableError."""
        provider = _make_provider(model_config)
        with patch(
            "httpx.AsyncClient",
            return_value=_mock_httpx_response(500, "Internal Server Error"),
        ):
            with pytest.raises(ProviderUnavailableError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_http_400_raises_response_error(self, model_config, llm_request):
        """HTTP 400 → ProviderResponseError."""
        provider = _make_provider(model_config)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(400, {})):
            with pytest.raises(ProviderResponseError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self, model_config, llm_request):
        """httpx.TimeoutException → ProviderTimeoutError."""
        provider = _make_provider(model_config)
        exc = httpx.TimeoutException("timed out")
        with patch(
            "httpx.AsyncClient",
            return_value=_mock_httpx_response(0, raise_exc=exc),
        ):
            with pytest.raises(ProviderTimeoutError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_connect_error_raises_unavailable(self, model_config, llm_request):
        """httpx.ConnectError → ProviderUnavailableError."""
        provider = _make_provider(model_config)
        exc = httpx.ConnectError("connection refused")
        with patch(
            "httpx.AsyncClient",
            return_value=_mock_httpx_response(0, raise_exc=exc),
        ):
            with pytest.raises(ProviderUnavailableError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_non_json_response_raises_response_error(
        self, model_config, llm_request
    ):
        """Non-JSON response body → ProviderResponseError."""
        provider = _make_provider(model_config)
        with patch(
            "httpx.AsyncClient",
            return_value=_mock_httpx_response(200, "not json at all"),
        ):
            with pytest.raises(ProviderResponseError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_missing_choices_raises_response_error(
        self, model_config, llm_request
    ):
        """Response body without 'choices' → ProviderResponseError."""
        provider = _make_provider(model_config)
        bad_body = {"id": "gen-test", "model": "some-model"}  # No 'choices'
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, bad_body)):
            with pytest.raises(ProviderResponseError):
                await provider.generate(llm_request)

    @pytest.mark.asyncio
    async def test_empty_choices_raises_response_error(
        self, model_config, llm_request
    ):
        """Response body with empty 'choices' list → ProviderResponseError."""
        provider = _make_provider(model_config)
        bad_body = {"id": "gen-test", "model": "some-model", "choices": []}
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, bad_body)):
            with pytest.raises(ProviderResponseError):
                await provider.generate(llm_request)


class TestOpenRouterProviderConfiguration:

    def test_missing_api_key_raises_configuration_error(self, model_config):
        """Constructing provider without API key raises ConfigurationError."""
        with patch("app.providers.openrouter.settings") as mock_settings:
            mock_settings.openrouter_api_key = ""
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
                OpenRouterProvider(model_config=model_config)


class TestOpenRouterHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_reachable(self, model_config):
        """health_check() returns True when endpoint responds."""
        provider = _make_provider(model_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_response)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await provider.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_connect_error(self, model_config):
        """health_check() returns False when the endpoint is unreachable."""
        provider = _make_provider(model_config)

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await provider.health_check()

        assert result is False
