"""
app/core/exceptions.py
───────────────────────
Domain-specific exceptions for RouteMind_AI.

All custom exceptions inherit from RouteMindError so callers can catch
platform-level errors with a single except clause when needed.

Phase 1 scope:
    - Provider-level errors (auth, timeout, rate limit, unavailable).
    - Configuration errors (missing model, bad YAML, missing key).

Future phases will add:
    - RoutingError (Phase 2)
    - QualityVerificationError (Phase 3)
    - EscalationError (Phase 3)
    - DatabaseError (Phase 4)
"""


class RouteMindError(Exception):
    """Base exception for all RouteMind_AI platform errors."""


# ── Configuration & Registry ──────────────────────────────────────────────────

class ConfigurationError(RouteMindError):
    """Raised when a required configuration value is missing or invalid."""


class ModelNotFoundError(RouteMindError):
    """Raised when a requested model_id is not present in the registry."""


# ── Provider Errors ───────────────────────────────────────────────────────────

class ProviderError(RouteMindError):
    """Base class for provider-related errors."""


class ProviderAuthError(ProviderError):
    """Raised when a provider rejects the API key or credentials."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider or model endpoint cannot be reached."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider returns a rate-limit (429) response."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request exceeds the configured timeout."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns an unexpected or malformed response."""


class TokenExtractionError(ProviderError):
    """Raised when token counts cannot be extracted from a provider response."""
