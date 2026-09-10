"""
app/cost/calculator.py
───────────────────────
Isolated token cost calculation module.

Cost logic is deliberately separated from provider implementations.
This ensures:
  - Provider code never mixes billing logic with HTTP/API logic.
  - Cost formula can be updated in one place without touching providers.
  - Cost calculation can be independently unit-tested.

Formula:
    input_cost  = (input_tokens  / 1000) * input_cost_per_1k_tokens
    output_cost = (output_tokens / 1000) * output_cost_per_1k_tokens
    total_cost  = input_cost + output_cost

Phase 1 scope:
    - CostCalculator.calculate() given token counts and model pricing.
    - CostBreakdown dataclass for structured cost reporting.
"""

from dataclasses import dataclass

from app.models.config import ModelConfig


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a single inference call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model_id: str
    provider: str

    def __str__(self) -> str:
        return (
            f"[{self.model_id}] "
            f"{self.total_tokens} tokens | "
            f"${self.total_cost:.6f} "
            f"(in=${self.input_cost:.6f}, out=${self.output_cost:.6f})"
        )


class CostCalculator:
    """
    Calculates inference cost from token counts and model pricing config.

    Stateless — instantiate once and reuse, or call as static methods.
    """

    @staticmethod
    def calculate(
        model: ModelConfig,
        input_tokens: int,
        output_tokens: int,
    ) -> CostBreakdown:
        """
        Compute a full cost breakdown for one inference call.

        Args:
            model:         ModelConfig loaded from the registry.
            input_tokens:  Number of tokens in the input prompt.
            output_tokens: Number of tokens in the generated output.

        Returns:
            CostBreakdown with per-segment and total cost in USD.
        """
        input_cost = (input_tokens / 1000.0) * model.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000.0) * model.output_cost_per_1k_tokens
        total_cost = input_cost + output_cost
        total_tokens = input_tokens + output_tokens

        return CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            model_id=model.model_id,
            provider=model.provider,
        )

    @staticmethod
    def estimate_input_cost(model: ModelConfig, input_tokens: int) -> float:
        """Quick estimate of input-only cost (useful before response is received)."""
        return (input_tokens / 1000.0) * model.input_cost_per_1k_tokens
