"""
app/providers/provider_registry.py
────────────────────────────────────
Provider Registry — maps provider name strings to LLMProvider instances.

The IntelligentRouter (Phase 2 Part 5) must resolve a concrete LLMProvider
from the string held in ModelConfig.provider without being coupled to any
specific provider class.  This registry is the indirection layer.

Design rules:
    - The registry only stores already-constructed LLMProvider instances.
    - Registration is explicit; nothing is auto-discovered.
    - get() raises ProviderNotRegisteredError, never returns None.
    - The registry is intentionally simple so it can be populated in tests
      with mock providers without touching real API code.

Usage (production)::

    from app.providers.provider_registry import ProviderRegistry
    from app.providers.openrouter import OpenRouterProvider
    from app.providers.ollama import OllamaProvider

    registry = ProviderRegistry()
    registry.register("openrouter", OpenRouterProvider(model_config=...))
    registry.register("ollama", OllamaProvider(model_config=...))

Usage (tests)::

    from unittest.mock import AsyncMock
    registry = ProviderRegistry()
    mock_provider = AsyncMock(spec=LLMProvider)
    mock_provider.provider_name = "openrouter"
    registry.register("openrouter", mock_provider)

Phase 2 Part 5 scope:
    - ProviderRegistry class with register() / get() / registered_names().
    - Raises ProviderNotRegisteredError on missing provider.
"""

from __future__ import annotations

from app.core.exceptions import ProviderNotRegisteredError
from app.providers.base import LLMProvider


class ProviderRegistry:
    """
    In-memory registry mapping provider name → LLMProvider instance.

    Thread-safe for reads once fully populated at startup.
    Mutations (register) should occur only during application initialisation.
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    # ── Mutation ──────────────────────────────────────────────────────────────

    def register(self, name: str, provider: LLMProvider) -> None:
        """
        Register a provider instance under the given name.

        Args:
            name:     The canonical provider name string (e.g. "openrouter",
                      "ollama").  Must match ModelConfig.provider values.
            provider: A fully-constructed LLMProvider instance.

        Raises:
            TypeError: If provider is not an LLMProvider instance.
        """
        if not isinstance(provider, LLMProvider):
            raise TypeError(
                f"provider must be an LLMProvider instance, got {type(provider)}"
            )
        self._providers[name.lower()] = provider

    # ── Query ─────────────────────────────────────────────────────────────────

    def get(self, name: str) -> LLMProvider:
        """
        Retrieve the registered LLMProvider for a provider name.

        Args:
            name: Provider name string (case-insensitive).

        Returns:
            The registered LLMProvider instance.

        Raises:
            ProviderNotRegisteredError: If the provider name is not registered.
        """
        key = name.lower()
        if key not in self._providers:
            registered = sorted(self._providers.keys())
            raise ProviderNotRegisteredError(
                f"Provider '{name}' is not registered. "
                f"Registered providers: {registered}. "
                "Call ProviderRegistry.register() during application startup."
            )
        return self._providers[key]

    def registered_names(self) -> list[str]:
        """Return a sorted list of all registered provider names."""
        return sorted(self._providers.keys())

    def __len__(self) -> int:
        return len(self._providers)

    def __repr__(self) -> str:
        return f"ProviderRegistry(registered={self.registered_names()})"
