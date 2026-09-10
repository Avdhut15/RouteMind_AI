"""
app/models/registry.py
───────────────────────
Model Registry — loads and provides access to all model configurations.

The registry is the single source of truth for model metadata at runtime.
It reads config/models.yaml on startup and exposes lookup methods used
by the router, cost calculator, and provider factory.

Phase 1 scope:
    - Load ModelConfig entries from YAML.
    - Validate all entries against the ModelConfig schema.
    - Provide lookup by model_id.
    - Provide filtering by provider and quality tier.
    - Provide a list of all enabled models.

The registry does NOT make routing decisions.
"""

from pathlib import Path
from typing import Optional

import yaml

from app.core.exceptions import ConfigurationError, ModelNotFoundError
from app.models.config import ModelConfig, QualityTier


class ModelRegistry:
    """
    Centralized in-memory registry of LLM model configurations.

    Loaded once from config/models.yaml at startup.
    All routing, cost, and provider code queries this registry.
    """

    def __init__(self, config_path: Path) -> None:
        """
        Initialize the registry by loading and validating models.yaml.

        Args:
            config_path: Path to the models.yaml configuration file.

        Raises:
            ConfigurationError: If the file is missing, unreadable, or invalid.
        """
        self._models: dict[str, ModelConfig] = {}
        self._load(config_path)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        """Parse YAML and validate every model entry against ModelConfig."""
        if not path.exists():
            raise ConfigurationError(
                f"models.yaml not found at '{path}'. "
                "Ensure config/models.yaml exists."
            )

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw or "models" not in raw:
            raise ConfigurationError(
                "models.yaml must contain a top-level 'models' list."
            )

        for entry in raw["models"]:
            model = ModelConfig(**entry)
            self._models[model.model_id] = model

    # ── Query Interface ───────────────────────────────────────────────────────

    def get(self, model_id: str) -> ModelConfig:
        """
        Retrieve a model config by its ID.

        Args:
            model_id: The provider-specific model identifier.

        Returns:
            ModelConfig for the requested model.

        Raises:
            ModelNotFoundError: If the model_id is not in the registry.
        """
        if model_id not in self._models:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found in registry. "
                "Check config/models.yaml."
            )
        return self._models[model_id]

    def all_enabled(self) -> list[ModelConfig]:
        """Return all models where enabled=True."""
        return [m for m in self._models.values() if m.enabled]

    def by_provider(self, provider: str) -> list[ModelConfig]:
        """Return all enabled models for a specific provider."""
        return [
            m for m in self.all_enabled()
            if m.provider == provider.lower()
        ]

    def by_tier(self, tier: QualityTier) -> list[ModelConfig]:
        """Return all enabled models matching the given quality tier."""
        return [
            m for m in self.all_enabled()
            if m.quality_tier == tier
        ]

    def list_model_ids(self) -> list[str]:
        """Return model IDs of all enabled models."""
        return [m.model_id for m in self.all_enabled()]

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return (
            f"ModelRegistry("
            f"total={len(self._models)}, "
            f"enabled={len(self.all_enabled())})"
        )
