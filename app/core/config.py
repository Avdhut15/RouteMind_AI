"""
app/core/config.py
──────────────────
Global application settings loaded from environment variables and .env file.

Uses pydantic-settings so all config is type-safe, validated at startup, and
sourced from environment variables — never hardcoded.

Phase 1 scope:
    - Provider API keys and base URLs.
    - Application environment and log level.
    - Paths to YAML config files.

Future phases will extend Settings with:
    - DATABASE_URL (Phase 4)
    - Verification mode (Phase 3)
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    All values are loaded from environment variables (or .env file).
    See .env.example for required variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Logging level")

    # ── Config file paths ─────────────────────────────────────────────────────
    models_config_path: Path = Field(
        default=Path("config/models.yaml"),
        description="Path to models.yaml",
    )
    routing_config_path: Path = Field(
        default=Path("config/routing.yaml"),
        description="Path to routing.yaml",
    )

    # ── OpenRouter ────────────────────────────────────────────────────────────
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama local server base URL",
    )

    # ── Groq (Phase 1+ integration) ───────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key")

    # ── Google Gemini (Phase 1+ integration) ──────────────────────────────────
    gemini_api_key: str = Field(default="", description="Google Gemini API key")

    # ── Mistral (Phase 1+ integration) ────────────────────────────────────────
    mistral_api_key: str = Field(default="", description="Mistral API key")

    # ── Database (Phase 4+) ───────────────────────────────────────────────────
    # database_url: str = Field(default="", description="PostgreSQL connection URL")


# Singleton instance — import this throughout the app.
settings = Settings()
