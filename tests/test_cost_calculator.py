"""
tests/test_cost_calculator.py
──────────────────────────────
Unit tests for CostCalculator and CostBreakdown.

No real APIs or Ollama server required.

Audit findings (before writing tests):
    - CostCalculator.calculate() reads pricing exclusively from ModelConfig.
    - No pricing is hardcoded inside CostCalculator.
    - Formula: input_cost = (input_tokens / 1000) * input_cost_per_1k_tokens
               output_cost = (output_tokens / 1000) * output_cost_per_1k_tokens
               total_cost = input_cost + output_cost
    - CostCalculator.estimate_input_cost() provides a pre-response estimate.
    - No changes to the existing implementation were required.

Test coverage:
    - Standard token counts with paid model pricing.
    - Zero tokens (both input and output).
    - Zero input tokens only.
    - Zero output tokens only.
    - Zero-priced model (local Ollama) produces $0.00 cost.
    - Large token counts (no overflow or precision loss).
    - Fractional pricing values.
    - CostBreakdown fields match inputs exactly.
    - CostBreakdown __str__ output format.
    - estimate_input_cost() returns correct value.
    - estimate_input_cost() returns 0.0 for zero-priced model.
    - estimate_input_cost() with zero tokens.
    - CostCalculator.calculate() returns a CostBreakdown instance.
    - total_tokens == input_tokens + output_tokens.
    - CostBreakdown model_id and provider match ModelConfig.
"""

import pytest

from app.cost.calculator import CostBreakdown, CostCalculator
from app.models.config import ModelConfig, QualityTier


# ── Test fixtures ─────────────────────────────────────────────────────────────

def make_paid_model(
    input_cost: float = 0.0001,
    output_cost: float = 0.0002,
) -> ModelConfig:
    """ModelConfig with explicit simulated paid pricing."""
    return ModelConfig(
        model_id="openai/gpt-oss-20b:free",
        provider="openrouter",
        display_name="Test Paid Model",
        input_cost_per_1k_tokens=input_cost,
        output_cost_per_1k_tokens=output_cost,
        quality_tier=QualityTier.TIER_2,
        context_window=16000,
        enabled=True,
    )


def make_zero_cost_model() -> ModelConfig:
    """ModelConfig for a local Ollama model with $0.00 pricing."""
    return ModelConfig(
        model_id="gemma3:4b",
        provider="ollama",
        display_name="Local Ollama Model",
        input_cost_per_1k_tokens=0.0,
        output_cost_per_1k_tokens=0.0,
        quality_tier=QualityTier.TIER_1,
        context_window=8192,
        enabled=True,
    )


# ── CostCalculator.calculate() tests ─────────────────────────────────────────

class TestCostCalculatorCalculate:

    def test_returns_cost_breakdown_instance(self):
        """calculate() returns a CostBreakdown dataclass."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 100, 50)
        assert isinstance(result, CostBreakdown)

    def test_standard_token_counts(self):
        """Standard case: paid model with both input and output tokens."""
        model = make_paid_model(input_cost=0.0001, output_cost=0.0002)
        result = CostCalculator.calculate(model, 1000, 500)

        # input:  (1000 / 1000) * 0.0001 = 0.0001
        # output: (500  / 1000) * 0.0002 = 0.0001
        # total:  0.0002
        assert abs(result.input_cost - 0.0001) < 1e-10
        assert abs(result.output_cost - 0.0001) < 1e-10
        assert abs(result.total_cost - 0.0002) < 1e-10

    def test_total_tokens_is_sum_of_input_and_output(self):
        """total_tokens must always equal input_tokens + output_tokens."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 300, 150)
        assert result.total_tokens == 300 + 150

    def test_zero_input_and_output_tokens(self):
        """Both token counts zero → all costs $0.00."""
        model = make_paid_model(input_cost=0.001, output_cost=0.002)
        result = CostCalculator.calculate(model, 0, 0)
        assert result.input_cost == 0.0
        assert result.output_cost == 0.0
        assert result.total_cost == 0.0
        assert result.total_tokens == 0

    def test_zero_input_tokens_only(self):
        """Zero input tokens → input_cost is $0.00; output_cost is non-zero."""
        model = make_paid_model(input_cost=0.001, output_cost=0.002)
        result = CostCalculator.calculate(model, 0, 1000)
        assert result.input_cost == 0.0
        assert abs(result.output_cost - 0.002) < 1e-10
        assert abs(result.total_cost - 0.002) < 1e-10

    def test_zero_output_tokens_only(self):
        """Zero output tokens → output_cost is $0.00; input_cost is non-zero."""
        model = make_paid_model(input_cost=0.001, output_cost=0.002)
        result = CostCalculator.calculate(model, 1000, 0)
        assert abs(result.input_cost - 0.001) < 1e-10
        assert result.output_cost == 0.0
        assert abs(result.total_cost - 0.001) < 1e-10

    def test_zero_priced_model_always_zero_cost(self):
        """Local Ollama model with 0.0/0.0 pricing → $0.00 regardless of tokens."""
        model = make_zero_cost_model()
        result = CostCalculator.calculate(model, 5000, 2000)
        assert result.input_cost == 0.0
        assert result.output_cost == 0.0
        assert result.total_cost == 0.0

    def test_large_token_counts(self):
        """Large token counts (context-window-sized) are handled without overflow."""
        model = make_paid_model(input_cost=0.0001, output_cost=0.0002)
        # 128k input, 4k output
        result = CostCalculator.calculate(model, 128_000, 4_000)
        expected_input = (128_000 / 1000.0) * 0.0001   # 0.0128
        expected_output = (4_000 / 1000.0) * 0.0002     # 0.0008
        assert abs(result.input_cost - expected_input) < 1e-9
        assert abs(result.output_cost - expected_output) < 1e-9
        assert abs(result.total_cost - (expected_input + expected_output)) < 1e-9

    def test_fractional_pricing(self):
        """Fractional per-1k prices are handled correctly."""
        model = make_paid_model(input_cost=0.00015, output_cost=0.00030)
        result = CostCalculator.calculate(model, 2000, 1000)
        expected_input = (2000 / 1000.0) * 0.00015   # 0.0003
        expected_output = (1000 / 1000.0) * 0.00030  # 0.0003
        assert abs(result.input_cost - expected_input) < 1e-10
        assert abs(result.output_cost - expected_output) < 1e-10

    def test_pricing_comes_from_model_config_not_hardcoded(self):
        """
        Two models with different pricing should produce different costs
        for the same token counts, confirming pricing is read from ModelConfig.
        """
        cheap = make_paid_model(input_cost=0.00005, output_cost=0.00010)
        expensive = make_paid_model(input_cost=0.003, output_cost=0.006)

        result_cheap = CostCalculator.calculate(cheap, 1000, 1000)
        result_expensive = CostCalculator.calculate(expensive, 1000, 1000)

        assert result_cheap.total_cost < result_expensive.total_cost

    def test_cost_breakdown_model_id_matches_config(self):
        """CostBreakdown.model_id matches the ModelConfig.model_id."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 100, 50)
        assert result.model_id == model.model_id

    def test_cost_breakdown_provider_matches_config(self):
        """CostBreakdown.provider matches the ModelConfig.provider."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 100, 50)
        assert result.provider == model.provider

    def test_cost_breakdown_input_tokens_field(self):
        """CostBreakdown.input_tokens matches the argument passed."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 777, 333)
        assert result.input_tokens == 777

    def test_cost_breakdown_output_tokens_field(self):
        """CostBreakdown.output_tokens matches the argument passed."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 777, 333)
        assert result.output_tokens == 333

    def test_cost_breakdown_str_contains_model_id(self):
        """CostBreakdown.__str__() includes the model_id."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 1000, 500)
        assert model.model_id in str(result)

    def test_cost_breakdown_str_contains_total_cost(self):
        """CostBreakdown.__str__() includes a dollar-sign cost figure."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 1000, 500)
        assert "$" in str(result)

    def test_cost_breakdown_str_contains_token_count(self):
        """CostBreakdown.__str__() includes the total token count."""
        model = make_paid_model()
        result = CostCalculator.calculate(model, 300, 200)
        assert "500" in str(result)  # 300 + 200 = 500


# ── CostCalculator.estimate_input_cost() tests ───────────────────────────────

class TestCostCalculatorEstimateInputCost:

    def test_standard_estimate(self):
        """estimate_input_cost() returns correct value for paid model."""
        model = make_paid_model(input_cost=0.001)
        result = CostCalculator.estimate_input_cost(model, 2000)
        expected = (2000 / 1000.0) * 0.001  # 0.002
        assert abs(result - expected) < 1e-10

    def test_zero_tokens_estimate(self):
        """estimate_input_cost() returns 0.0 when tokens=0."""
        model = make_paid_model(input_cost=0.001)
        result = CostCalculator.estimate_input_cost(model, 0)
        assert result == 0.0

    def test_zero_priced_model_estimate(self):
        """estimate_input_cost() returns 0.0 for zero-priced model."""
        model = make_zero_cost_model()
        result = CostCalculator.estimate_input_cost(model, 50_000)
        assert result == 0.0

    def test_estimate_is_input_only(self):
        """estimate_input_cost() does not include output pricing."""
        model = make_paid_model(input_cost=0.001, output_cost=999.0)
        # Despite huge output_cost, estimate should only use input_cost
        result = CostCalculator.estimate_input_cost(model, 1000)
        assert abs(result - 0.001) < 1e-10
        assert result < 1.0  # Must not include output pricing
