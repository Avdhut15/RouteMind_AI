"""
app/providers/ollama.py
────────────────────────
Ollama local provider implementation.

Communicates with a locally running Ollama server via its native REST API.
Uses the /api/chat endpoint (available since Ollama 0.1.14+) which accepts
a messages array, making the request shape consistent with the OpenRouter
provider.

Ollama API reference:
    POST /api/chat
    Body: {"model": "...", "messages": [...], "stream": false, "options": {...}}
    Response: {"message": {"content": "..."}, "done": true,
               "prompt_eval_count": N, "eval_count": N, ...}

Key differences from OpenRouter:
    - No API key required (local server).
    - Token fields are: prompt_eval_count (input) and eval_count (output).
    - finish_reason is derived from the "done" flag (not an explicit field).
    - Cost is $0.00 for local inference (per ModelConfig pricing).
    - stream is always set to false — this provider does not stream.

Responsibilities:
    - Read the Ollama base URL from existing Settings.
    - Convert LLMRequest → Ollama /api/chat payload.
    - POST via async httpx and measure latency.
    - Parse the response into the standardized LLMResponse.
    - Extract token counts from prompt_eval_count / eval_count.
    - Compute estimated_cost via CostCalculator (will be $0.00 for local models).
    - Raise domain-specific exceptions for every failure mode.
    - Emit a structured inference log (no raw prompts logged).

This class contains NO routing logic, NO model selection, NO classification,
NO quality evaluation, and NO escalation.
"""

import time
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.core.exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.cost.calculator import CostCalculator
from app.logging.logger import get_logger, hash_prompt, log_inference_event
from app.models.config import ModelConfig
from app.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = get_logger(__name__)

# Default timeout for a local Ollama round-trip.
# Local models can be slow on CPU — 120s is generous but safe.
_DEFAULT_TIMEOUT_S: float = 120.0

# Finish reason string returned when Ollama signals done=True.
_FINISH_REASON_STOP = "stop"


class OllamaProvider(LLMProvider):
    """
    Concrete implementation of LLMProvider for a locally running Ollama server.

    No API key is required — Ollama is accessed over localhost HTTP.

    Usage:
        provider = OllamaProvider(model_config=model_cfg)
        response = await provider.generate(request)

    The instance is intended to be constructed once per model and reused
    across multiple requests.
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
                          Defaults to 120s to accommodate slow local inference.
        """
        self._model_config = model_config
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._timeout_s = timeout_s

    # ── LLMProvider interface ─────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Execute an LLM inference call against the local Ollama server and
        return a normalized LLMResponse.

        Raises:
            ProviderTimeoutError:    Request exceeds timeout.
            ProviderUnavailableError: Connection errors or HTTP 5xx.
            ProviderResponseError:   HTTP 4xx, malformed JSON, or missing fields.
        """
        payload = self._build_payload(request)
        endpoint = f"{self._base_url}/api/chat"

        t_start = time.perf_counter()
        raw: dict[str, Any] = await self._post(endpoint, payload, request.request_id)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return self._parse_response(raw, request, latency_ms)

    async def health_check(self) -> bool:
        """
        Verify that the local Ollama server is reachable.

        GETs the Ollama root endpoint (/). Returns True if the server
        responds with any HTTP status. Returns False on connection or
        timeout errors.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self._base_url)
            return response.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """
        Convert LLMRequest into an Ollama /api/chat payload.

        Ollama's chat endpoint accepts a messages array identical in shape
        to the OpenAI format, with an additional top-level "stream" field
        (set to False — this provider does not handle streaming).

        Generation parameters (temperature, max tokens) are passed via the
        nested "options" object.
        """
        messages: list[dict[str, str]] = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        return {
            "model": request.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
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
                    headers={"Content-Type": "application/json"},
                )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"[{request_id}] Ollama request timed out after {self._timeout_s}s. "
                "The model may still be loading. Try increasing timeout_s."
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"[{request_id}] Could not connect to Ollama at '{self._base_url}'. "
                "Ensure Ollama is running locally: https://ollama.ai"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                f"[{request_id}] Network error while calling Ollama: {exc}"
            ) from exc

        # ── HTTP error handling ───────────────────────────────────────────────
        if http_response.status_code == 404:
            raise ProviderResponseError(
                f"[{request_id}] Ollama model not found (HTTP 404). "
                f"Run: ollama pull {self._model_config.model_id}"
            )
        if http_response.status_code >= 500:
            raise ProviderUnavailableError(
                f"[{request_id}] Ollama server error "
                f"(HTTP {http_response.status_code}): {http_response.text[:200]}"
            )
        if http_response.status_code >= 400:
            raise ProviderResponseError(
                f"[{request_id}] Unexpected HTTP {http_response.status_code} "
                f"from Ollama: {http_response.text[:200]}"
            )

        # ── Parse JSON ────────────────────────────────────────────────────────
        try:
            return http_response.json()
        except Exception as exc:
            raise ProviderResponseError(
                f"[{request_id}] Ollama returned non-JSON response: "
                f"{http_response.text[:200]}"
            ) from exc

    def _parse_response(
        self,
        raw: dict[str, Any],
        request: LLMRequest,
        latency_ms: float,
    ) -> LLMResponse:
        """
        Normalize an Ollama /api/chat response body into LLMResponse.

        Ollama /api/chat (non-streaming) response shape:
        {
          "model": "gemma3:4b",
          "message": {"role": "assistant", "content": "..."},
          "done": true,
          "prompt_eval_count": N,   # input tokens
          "eval_count": N,          # output tokens
          "done_reason": "stop"     # available in newer Ollama versions
        }

        Raises:
            ProviderResponseError: If the expected shape is absent.
        """
        # ── Validate shape ────────────────────────────────────────────────────
        message: Optional[dict[str, Any]] = raw.get("message")
        if not message or not isinstance(message, dict):
            raise ProviderResponseError(
                f"[{request.request_id}] Ollama response missing 'message' field: "
                f"{str(raw)[:300]}"
            )

        output: Optional[str] = message.get("content")

        # ── Finish reason ─────────────────────────────────────────────────────
        # Newer Ollama builds expose done_reason; fall back to "stop" when done=True.
        done: bool = bool(raw.get("done", False))
        finish_reason: Optional[str] = raw.get("done_reason")
        if finish_reason is None:
            finish_reason = _FINISH_REASON_STOP if done else None

        # ── Token usage ───────────────────────────────────────────────────────
        # Ollama uses prompt_eval_count for input tokens, eval_count for output.
        # These fields may be absent if Ollama did not evaluate tokens (e.g. cached).
        input_tokens: int = int(raw.get("prompt_eval_count", 0))
        output_tokens: int = int(raw.get("eval_count", 0))
        total_tokens: int = input_tokens + output_tokens

        # ── Cost ──────────────────────────────────────────────────────────────
        # For local Ollama models, ModelConfig sets cost to $0.00, so the
        # CostCalculator will correctly return $0.00 without special-casing.
        breakdown = CostCalculator.calculate(
            self._model_config, input_tokens, output_tokens
        )

        # ── Model ID ─────────────────────────────────────────────────────────
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
