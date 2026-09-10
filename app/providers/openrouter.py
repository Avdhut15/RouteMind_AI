"""
app/providers/openrouter.py
────────────────────────────
OpenRouter provider implementation.

OpenRouter exposes an OpenAI-compatible /chat/completions endpoint, so the
request format follows the standard messages array convention.

Responsibilities of this class:
    - Read the API key and base URL from existing Settings (never hardcoded).
    - Convert LLMRequest → OpenRouter-compatible JSON payload.
    - POST the payload via async httpx.
    - Parse the response into the standardized LLMResponse.
    - Extract token counts from the 'usage' field.
    - Compute estimated_cost via CostCalculator.
    - Raise domain-specific exceptions for every failure mode.
    - Log inference events using the project logger (no raw key/prompt logged).

This class contains NO routing logic, NO model selection, NO classification,
NO quality evaluation, and NO escalation.
"""

import time
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.core.exceptions import (
    ConfigurationError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.cost.calculator import CostCalculator
from app.logging.logger import get_logger, hash_prompt, log_inference_event
from app.models.config import ModelConfig
from app.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = get_logger(__name__)

# Default timeout in seconds for a full OpenRouter round-trip.
_DEFAULT_TIMEOUT_S: float = 60.0


class OpenRouterProvider(LLMProvider):
    """
    Concrete implementation of LLMProvider for the OpenRouter API.

    OpenRouter is a unified gateway that proxies many models through an
    OpenAI-compatible /v1/chat/completions endpoint.

    Usage:
        provider = OpenRouterProvider(model_config=model_cfg)
        response = await provider.generate(request)

    The instance is intended to be constructed once per model and reused
    across multiple requests (the underlying httpx.AsyncClient is managed
    per-call to avoid connection-state issues between requests).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """
        Args:
            model_config: ModelConfig for the target model loaded from the registry.
            timeout_s:    HTTP request timeout in seconds.

        Raises:
            ConfigurationError: If OPENROUTER_API_KEY is not set.
        """
        if not settings.openrouter_api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )

        self._model_config = model_config
        self._base_url = settings.openrouter_base_url.rstrip("/")
        self._timeout_s = timeout_s

        # Build headers once — the key is never logged or re-read after init.
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Avdhut15/RouteMind_AI",
            "X-Title": "RouteMind_AI",
        }

    # ── LLMProvider interface ─────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "openrouter"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Execute an LLM inference call against OpenRouter and return a
        normalized LLMResponse.

        Raises:
            ProviderAuthError:       HTTP 401 / 403.
            ProviderRateLimitError:  HTTP 429.
            ProviderTimeoutError:    Request exceeds timeout.
            ProviderUnavailableError: Connection errors / HTTP 5xx.
            ProviderResponseError:   Malformed or unexpected response shape.
        """
        payload = self._build_payload(request)
        endpoint = f"{self._base_url}/chat/completions"

        t_start = time.perf_counter()
        raw: dict[str, Any] = await self._post(endpoint, payload, request.request_id)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return self._parse_response(raw, request, latency_ms)

    async def health_check(self) -> bool:
        """
        Verify that the OpenRouter endpoint is reachable.

        Returns True if a HEAD request to the base URL succeeds (or returns
        any HTTP response, even 4xx — that means the server is up). Returns
        False on connection/timeout errors.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.head(
                    self._base_url,
                    headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                )
            # Any HTTP response means the server is reachable.
            return response.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """
        Convert LLMRequest into an OpenAI-compatible messages payload.

        OpenRouter follows the OpenAI Chat Completions format exactly.
        """
        messages: list[dict[str, str]] = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        return {
            "model": request.model_id,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

    async def _post(
        self,
        url: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """
        Execute the async HTTP POST and handle all transport-level errors.

        Returns the parsed JSON body on success.
        Raises domain-specific ProviderError subclasses on failure.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                http_response = await client.post(
                    url,
                    json=payload,
                    headers=self._headers,
                )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"[{request_id}] OpenRouter request timed out after {self._timeout_s}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"[{request_id}] Could not connect to OpenRouter: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                f"[{request_id}] Network error while calling OpenRouter: {exc}"
            ) from exc

        # ── HTTP error handling ───────────────────────────────────────────────
        if http_response.status_code == 401:
            raise ProviderAuthError(
                f"[{request_id}] OpenRouter rejected the API key (HTTP 401)."
            )
        if http_response.status_code == 403:
            raise ProviderAuthError(
                f"[{request_id}] OpenRouter access forbidden (HTTP 403)."
            )
        if http_response.status_code == 429:
            raise ProviderRateLimitError(
                f"[{request_id}] OpenRouter rate limit exceeded (HTTP 429)."
            )
        if http_response.status_code >= 500:
            raise ProviderUnavailableError(
                f"[{request_id}] OpenRouter server error "
                f"(HTTP {http_response.status_code}): {http_response.text[:200]}"
            )
        if http_response.status_code >= 400:
            raise ProviderResponseError(
                f"[{request_id}] Unexpected HTTP {http_response.status_code} "
                f"from OpenRouter: {http_response.text[:200]}"
            )

        # ── Parse JSON ────────────────────────────────────────────────────────
        try:
            return http_response.json()
        except Exception as exc:
            raise ProviderResponseError(
                f"[{request_id}] OpenRouter returned non-JSON response: "
                f"{http_response.text[:200]}"
            ) from exc

    def _parse_response(
        self,
        raw: dict[str, Any],
        request: LLMRequest,
        latency_ms: float,
    ) -> LLMResponse:
        """
        Normalize an OpenRouter JSON response body into LLMResponse.

        OpenRouter follows the standard OpenAI shape:
        {
          "id": "...",
          "model": "...",
          "choices": [{"message": {"content": "..."}, "finish_reason": "stop"}],
          "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
        }

        Raises:
            ProviderResponseError: If the expected shape is absent.
        """
        # ── Validate shape ────────────────────────────────────────────────────
        choices: Optional[list] = raw.get("choices")
        if not choices or not isinstance(choices, list):
            raise ProviderResponseError(
                f"[{request.request_id}] OpenRouter response missing 'choices': "
                f"{str(raw)[:300]}"
            )

        first_choice: dict[str, Any] = choices[0]
        message = first_choice.get("message", {})
        output: Optional[str] = message.get("content")
        finish_reason: Optional[str] = first_choice.get("finish_reason")

        # ── Token usage ───────────────────────────────────────────────────────
        usage: dict[str, Any] = raw.get("usage") or {}
        input_tokens: int = int(usage.get("prompt_tokens", 0))
        output_tokens: int = int(usage.get("completion_tokens", 0))
        total_tokens: int = int(usage.get("total_tokens", input_tokens + output_tokens))

        # ── Cost ──────────────────────────────────────────────────────────────
        breakdown = CostCalculator.calculate(
            self._model_config, input_tokens, output_tokens
        )

        # ── Model ID returned by OpenRouter ──────────────────────────────────
        # OpenRouter may return the actual routed model_id; prefer that.
        returned_model_id: str = raw.get("model") or request.model_id

        # ── Structured log ────────────────────────────────────────────────────
        log_inference_event(
            logger=logger,
            request_id=request.request_id,
            model_id=returned_model_id,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=breakdown.total_cost,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            error=None,
            prompt_hash=hash_prompt(request.prompt),
        )

        return LLMResponse(
            request_id=request.request_id,
            output=output,
            model_id=returned_model_id,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost=breakdown.total_cost,
            finish_reason=finish_reason,
        )
