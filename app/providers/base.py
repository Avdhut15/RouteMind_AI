"""
app/providers/base.py
─────────────────────
Abstract LLM Provider interface.

Every provider (OpenRouter, Ollama, Groq, Gemini, Mistral...) must implement
this interface. The router communicates ONLY with this abstraction — never
directly with provider-specific API code.

Design rules:
  - The provider receives a fully-formed LLMRequest.
  - The provider returns a fully-formed LLMResponse.
  - The provider does NOT decide which model to use.
  - The provider handles all provider-specific authentication, HTTP, and parsing.
  - The provider must normalize token counts and populate estimated_cost.

Phase 1 scope:
    - Abstract base class definition.
    - LLMRequest schema.
    - LLMResponse schema.

Implementations (openrouter.py, ollama.py, etc.) are added in the next step.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Unified Request Schema ────────────────────────────────────────────────────

class LLMRequest(BaseModel):
    """
    Provider-independent representation of an LLM inference request.

    The router constructs this and hands it to a provider.
    The provider must NOT modify routing-level fields.
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier for tracing.",
    )
    prompt: str = Field(..., description="The user's input prompt.")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system-level instruction prompt.",
    )
    model_id: str = Field(
        ...,
        description="Target model identifier (set by the router, not the client).",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum output tokens.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional context (task_type, complexity_tier, etc.).",
    )


# ── Unified Response Schema ───────────────────────────────────────────────────

class LLMResponse(BaseModel):
    """
    Standardized response returned by every provider implementation.

    Every provider MUST normalize its raw API response into this schema.
    The rest of the platform operates only on LLMResponse — never on
    provider-specific response formats.
    """

    request_id: str = Field(..., description="Matches the originating LLMRequest.request_id.")
    output: Optional[str] = Field(default=None, description="Generated text output.")
    model_id: str = Field(..., description="Model that generated this response.")
    provider: str = Field(..., description="Provider that served this response.")

    # ── Token usage ───────────────────────────────────────────────────────────
    input_tokens: int = Field(default=0, ge=0, description="Tokens in the input prompt.")
    output_tokens: int = Field(default=0, ge=0, description="Tokens in the generated output.")
    total_tokens: int = Field(default=0, ge=0, description="input_tokens + output_tokens.")

    # ── Performance ───────────────────────────────────────────────────────────
    latency_ms: float = Field(default=0.0, ge=0.0, description="End-to-end latency in ms.")
    estimated_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated USD cost calculated from token usage and model pricing.",
    )

    # ── Status ────────────────────────────────────────────────────────────────
    finish_reason: Optional[str] = Field(
        default=None,
        description="Provider-reported stop reason (stop, length, error, etc.).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of response creation.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the request failed. None on success.",
    )

    # ── Routing Metadata (populated by router, not provider) ──────────────────
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Routing context (complexity_tier, routing_reason, etc.).",
    )

    @property
    def is_error(self) -> bool:
        """True if the response contains an error."""
        return self.error is not None

    @property
    def is_success(self) -> bool:
        """True if the response was generated successfully."""
        return self.error is None and self.output is not None


# ── Abstract Provider ─────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract base class for all LLM provider implementations.

    Adding a new provider requires only:
        1. Creating a new file in app/providers/.
        2. Implementing this interface.
        3. Registering the provider in the provider factory.

    The router never changes when new providers are added.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The canonical provider name (must match ModelConfig.provider)."""
        ...

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Execute an LLM inference request and return a standardized response.

        Args:
            request: Fully-formed LLMRequest constructed by the router.

        Returns:
            LLMResponse with all token, cost, and latency fields populated.

        Raises:
            ProviderAuthError: On authentication failure.
            ProviderUnavailableError: When the endpoint is unreachable.
            ProviderRateLimitError: On HTTP 429 responses.
            ProviderTimeoutError: When the request exceeds timeout.
            ProviderResponseError: For malformed or unexpected responses.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify that the provider endpoint is reachable.

        Returns:
            True if healthy, False otherwise.
        """
        ...
