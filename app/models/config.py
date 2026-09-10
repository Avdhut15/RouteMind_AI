"""
app/models/config.py
─────────────────────
Pydantic model for a single LLM model's configuration entry.

ModelConfig is the canonical data structure loaded from config/models.yaml.
It describes everything the platform needs to know about a model:
  - Identity (model_id, provider, display_name)
  - Pricing (cost per 1k input/output tokens — may be simulated for free models)
  - Performance estimates (latency, quality scores per task type)
  - Capability tier (tier_1 / tier_2 / tier_3)
  - Availability flag

Phase 1 scope:
    - Full ModelConfig definition.
    - QualityTier enum.

NOTE: Score fields (quality_score, reasoning_score, etc.) populated initially
from config/models.yaml are ESTIMATES. They will be updated with measured
benchmark results after Phase 1 experiments complete.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class QualityTier(str, Enum):
    """
    Model capability tier used by the router to filter candidates.

    tier_1 = low-cost, suitable for simple tasks.
    tier_2 = mid-capability, suitable for moderate tasks.
    tier_3 = high-capability, suitable for complex tasks.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class ModelConfig(BaseModel):
    """
    Configuration for a single LLM model.

    Loaded from config/models.yaml — never hardcoded in Python source.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    model_id: str = Field(..., description="Provider-specific model identifier.")
    provider: str = Field(..., description="Provider name (openrouter, ollama, groq, etc.)")
    display_name: str = Field(..., description="Human-readable name for logs and UI.")

    # ── Pricing ───────────────────────────────────────────────────────────────
    # For free/local models, set simulated values for benchmarking purposes.
    input_cost_per_1k_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="USD cost per 1,000 input tokens.",
    )
    output_cost_per_1k_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="USD cost per 1,000 output tokens.",
    )

    # ── Performance (Estimates — updated from benchmarks) ─────────────────────
    average_latency_ms: float = Field(
        default=2000.0,
        ge=0.0,
        description="Expected average response latency in milliseconds (config estimate).",
    )

    # ── Capability Tier & Scores ──────────────────────────────────────────────
    quality_tier: QualityTier = Field(..., description="Tier classification for routing.")
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_score: float = Field(default=0.5, ge=0.0, le=1.0)
    coding_score: float = Field(default=0.5, ge=0.0, le=1.0)
    summarization_score: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Context & Availability ────────────────────────────────────────────────
    context_window: int = Field(default=4096, gt=0, description="Max context window in tokens.")
    enabled: bool = Field(default=True, description="Whether this model is active.")

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, v: str) -> str:
        known = {"openrouter", "ollama", "groq", "gemini", "mistral", "openai", "anthropic"}
        if v.lower() not in known:
            raise ValueError(
                f"Unknown provider '{v}'. Add it to known providers or check spelling."
            )
        return v.lower()
