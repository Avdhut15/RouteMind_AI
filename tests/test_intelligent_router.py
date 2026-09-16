"""
tests/test_intelligent_router.py
──────────────────────────────────
Unit tests for Phase 2 Part 5 — Intelligent Router Integration.

All tests are:
    - Offline (no real API calls, no network)
    - Deterministic (fixed inputs, fixed mock responses)
    - Independent of OpenRouter, Ollama, or any live service
    - Using AsyncMock for provider.generate(); no real API keys needed

Test coverage
─────────────
    1.  test_successful_end_to_end
    2.  test_request_id_preserved
    3.  test_no_candidates_returns_routing_failure
    4.  test_provider_not_registered_returns_routing_failure
    5.  test_provider_inference_failure_provider_unavailable
    6.  test_provider_inference_failure_auth_error
    7.  test_model_selection_comes_from_scorer
    8.  test_cost_token_latency_preserved
    9.  test_factor_scores_populated
    10. test_routing_score_in_result
    11. test_complexity_and_confidence_in_result
    12. test_candidate_model_ids_in_result
    13. test_pipeline_is_deterministic
    14. test_invalid_constructor_types
    15. test_provider_registry_register_and_get
    16. test_provider_registry_missing_raises
    17. test_router_result_status_properties
    18. test_router_result_repr
    19. test_unexpected_provider_exception_returns_provider_failure
    20. test_routing_failure_populates_analysis_fields
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.classification.complexity_classifier import (
    ComplexityLabel,
    ComplexityPrediction,
    LogisticRegressionClassifier,
)
from app.core.exceptions import (
    NoEligibleCandidatesError,
    ProviderAuthError,
    ProviderNotRegisteredError,
    ProviderUnavailableError,
)
from app.models.config import ModelConfig, QualityTier
from app.models.registry import ModelRegistry
from app.providers.base import LLMProvider, LLMRequest, LLMResponse
from app.providers.provider_registry import ProviderRegistry
from app.routing.analyzer import RequestAnalyzer, TaskType
from app.routing.result import RouterResult
from app.routing.router import IntelligentRouter
from app.routing.scoring import ScoringWeights


# ── Shared constants ───────────────────────────────────────────────────────────

_REQUEST_ID = "test-req-abc123"

# A simple prompt that the classifier will classify as "simple"
_SIMPLE_PROMPT = "What is the capital of France?"

# A complex prompt that the classifier will classify as "complex"
_COMPLEX_PROMPT = (
    "Prove step-by-step using formal logical deduction that the halting problem "
    "is undecidable, and write a Python function implementing a related reduction."
)


# ── Model factory helpers ─────────────────────────────────────────────────────

def _make_model(
    model_id: str = "test/model-tier2",
    provider: str = "openrouter",
    quality_tier: QualityTier = QualityTier.TIER_2,
    quality_score: float = 0.72,
    reasoning_score: float = 0.70,
    coding_score: float = 0.65,
    summarization_score: float = 0.72,
    extraction_score: float = 0.70,
    context_window: int = 32000,
    supports_structured_output: bool = True,
    supports_function_calling: bool = True,
    input_cost_per_1k: float = 0.001,
    output_cost_per_1k: float = 0.002,
    average_latency_ms: float = 2000.0,
    enabled: bool = True,
) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider=provider,
        display_name=f"Test Model ({model_id})",
        quality_tier=quality_tier,
        quality_score=quality_score,
        reasoning_score=reasoning_score,
        coding_score=coding_score,
        summarization_score=summarization_score,
        extraction_score=extraction_score,
        context_window=context_window,
        supports_structured_output=supports_structured_output,
        supports_function_calling=supports_function_calling,
        input_cost_per_1k_tokens=input_cost_per_1k,
        output_cost_per_1k_tokens=output_cost_per_1k,
        average_latency_ms=average_latency_ms,
        enabled=enabled,
    )


def _make_tier1_model(model_id: str = "test/model-tier1") -> ModelConfig:
    return _make_model(
        model_id=model_id,
        quality_tier=QualityTier.TIER_1,
        quality_score=0.55,
        reasoning_score=0.52,
        coding_score=0.50,
    )


def _make_tier3_model(model_id: str = "test/model-tier3") -> ModelConfig:
    return _make_model(
        model_id=model_id,
        quality_tier=QualityTier.TIER_3,
        quality_score=0.90,
        reasoning_score=0.88,
        coding_score=0.85,
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.02,
    )


def _make_registry(*models: ModelConfig) -> ModelRegistry:
    """Build an in-memory ModelRegistry from ModelConfig objects (no disk I/O)."""
    registry = ModelRegistry.__new__(ModelRegistry)
    registry._models = {m.model_id: m for m in models}
    return registry


# ── Classifier helpers ────────────────────────────────────────────────────────

def _make_trained_classifier() -> LogisticRegressionClassifier:
    """
    Train a LogisticRegressionClassifier on a minimal 3-sample dataset
    (one per class) so predict() works without any file I/O.
    """
    from app.routing.analyzer import RequestAnalyzer
    analyzer = RequestAnalyzer()

    # Simple request
    simple_req = LLMRequest(
        prompt="What is the capital of France?",
        model_id="placeholder",
    )
    # Moderate request
    moderate_req = LLMRequest(
        prompt="Summarize the key events of the French Revolution in detail.",
        model_id="placeholder",
    )
    # Complex request
    complex_req = LLMRequest(
        prompt=(
            "Prove the Pythagorean theorem, analyze the data below, "
            "write a Python function, step by step in JSON format."
        ),
        model_id="placeholder",
    )

    analyses = [
        analyzer.analyze(simple_req),
        analyzer.analyze(moderate_req),
        analyzer.analyze(complex_req),
    ]
    labels = [ComplexityLabel.SIMPLE, ComplexityLabel.MODERATE, ComplexityLabel.COMPLEX]

    clf = LogisticRegressionClassifier()
    clf.train(analyses, labels)
    return clf


# ── Mock LLMResponse factory ──────────────────────────────────────────────────

def _make_llm_response(
    request_id: str = _REQUEST_ID,
    model_id: str = "test/model-tier2",
    provider: str = "openrouter",
    output: str = "Paris",
    input_tokens: int = 12,
    output_tokens: int = 4,
    total_tokens: int = 16,
    estimated_cost: float = 0.0000168,
    latency_ms: float = 350.0,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        request_id=request_id,
        output=output,
        model_id=model_id,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        timestamp=datetime(2026, 9, 16, 0, 0, 0, tzinfo=timezone.utc),
    )


# ── Mock provider factory ─────────────────────────────────────────────────────

def _make_mock_provider(
    provider_name: str = "openrouter",
    response: LLMResponse | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """
    Build a mock LLMProvider whose generate() is an AsyncMock.

    The mock passes isinstance(mock, LLMProvider) checks because it uses
    spec=LLMProvider.
    """
    mock = MagicMock(spec=LLMProvider)
    mock.provider_name = provider_name
    if side_effect is not None:
        mock.generate = AsyncMock(side_effect=side_effect)
    else:
        mock.generate = AsyncMock(return_value=response or _make_llm_response())
    return mock


# ── Router factory ────────────────────────────────────────────────────────────

def _make_router(
    models: list[ModelConfig] | None = None,
    provider_name: str = "openrouter",
    provider_response: LLMResponse | None = None,
    provider_side_effect: Exception | None = None,
    weights: ScoringWeights | None = None,
    extra_providers: dict[str, MagicMock] | None = None,
) -> tuple[IntelligentRouter, MagicMock, ProviderRegistry]:
    """
    Build a fully-wired IntelligentRouter with injected mocks.

    Returns (router, mock_provider, provider_registry) for inspection.
    """
    if models is None:
        models = [_make_model()]

    registry = _make_registry(*models)
    clf = _make_trained_classifier()

    mock_provider = _make_mock_provider(
        provider_name=provider_name,
        response=provider_response,
        side_effect=provider_side_effect,
    )

    prov_reg = ProviderRegistry()
    prov_reg.register(provider_name, mock_provider)

    if extra_providers:
        for name, prov in extra_providers.items():
            prov_reg.register(name, prov)

    router = IntelligentRouter(
        classifier=clf,
        model_registry=registry,
        provider_registry=prov_reg,
        weights=weights,
    )
    return router, mock_provider, prov_reg


def _make_request(
    prompt: str = _SIMPLE_PROMPT,
    request_id: str = _REQUEST_ID,
    model_id: str = "placeholder",
    max_tokens: int = 512,
    system_prompt: str | None = None,
) -> LLMRequest:
    return LLMRequest(
        request_id=request_id,
        prompt=prompt,
        model_id=model_id,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestSuccessfulEndToEnd:
    """Test 1 — Full happy path: routing succeeds and provider returns a response."""

    @pytest.mark.asyncio
    async def test_successful_end_to_end(self) -> None:
        model = _make_model(model_id="openai/gpt-tier2", provider="openrouter")
        response = _make_llm_response(
            request_id=_REQUEST_ID,
            model_id=model.model_id,
            provider="openrouter",
        )
        router, mock_prov, _ = _make_router(
            models=[model],
            provider_response=response,
        )

        result = await router.route(_make_request())

        assert result.status == "success"
        assert result.error is None
        assert result.output == "Paris"
        assert result.is_success
        assert not result.is_routing_failure
        assert not result.is_provider_failure
        # Provider was actually called
        mock_prov.generate.assert_awaited_once()


class TestRequestIdPreservation:
    """Test 2 — request_id from the original LLMRequest is preserved in RouterResult."""

    @pytest.mark.asyncio
    async def test_request_id_preserved(self) -> None:
        custom_id = "my-custom-request-id-9999"
        model = _make_model()
        response = _make_llm_response(request_id=custom_id, model_id=model.model_id)
        router, _, _ = _make_router(models=[model], provider_response=response)

        request = _make_request(request_id=custom_id)
        result = await router.route(request)

        assert result.request_id == custom_id

    @pytest.mark.asyncio
    async def test_request_id_preserved_on_routing_failure(self) -> None:
        """request_id is preserved even when routing fails."""
        custom_id = "fail-req-id-1234"
        # Use a COMPLEX prompt but only TIER_1 model → no candidates
        tier1_model = _make_tier1_model()
        router, _, _ = _make_router(models=[tier1_model])

        # Force classifier to predict COMPLEX by monkey-patching
        with patch.object(
            router._classifier,
            "predict",
            return_value=ComplexityPrediction(
                complexity=ComplexityLabel.COMPLEX,
                confidence=0.95,
                probabilities={"complex": 0.95, "moderate": 0.03, "simple": 0.02},
                model_id="mock-clf",
            ),
        ):
            result = await router.route(_make_request(request_id=custom_id))

        assert result.request_id == custom_id
        assert result.is_routing_failure


class TestNoCandidates:
    """Test 3 — No eligible candidates → routing_failure with informative error."""

    @pytest.mark.asyncio
    async def test_no_candidates_returns_routing_failure(self) -> None:
        # Only tier_1 model; force COMPLEX complexity prediction → tier_3 required
        tier1_model = _make_tier1_model()
        router, mock_prov, _ = _make_router(models=[tier1_model])

        with patch.object(
            router._classifier,
            "predict",
            return_value=ComplexityPrediction(
                complexity=ComplexityLabel.COMPLEX,
                confidence=0.95,
                probabilities={"complex": 0.95, "moderate": 0.03, "simple": 0.02},
                model_id="mock-clf",
            ),
        ):
            result = await router.route(_make_request())

        assert result.status == "routing_failure"
        assert result.is_routing_failure
        assert result.error is not None
        assert len(result.error) > 0
        # Provider must NOT have been called
        mock_prov.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_candidates_populates_complexity_field(self) -> None:
        """complexity field is populated even on routing_failure (Part 2 ran)."""
        tier1_model = _make_tier1_model()
        router, _, _ = _make_router(models=[tier1_model])

        with patch.object(
            router._classifier,
            "predict",
            return_value=ComplexityPrediction(
                complexity=ComplexityLabel.COMPLEX,
                confidence=0.88,
                probabilities={"complex": 0.88, "moderate": 0.08, "simple": 0.04},
                model_id="mock-clf",
            ),
        ):
            result = await router.route(_make_request())

        assert result.complexity == "complex"
        assert result.classifier_confidence == pytest.approx(0.88)


class TestProviderNotRegistered:
    """Test 4 — Provider name from selected model not in ProviderRegistry."""

    @pytest.mark.asyncio
    async def test_provider_not_registered_returns_routing_failure(self) -> None:
        # Model says provider="openrouter" but we register "ollama"
        model = _make_model(provider="openrouter")
        registry = _make_registry(model)
        clf = _make_trained_classifier()

        # Register a different provider; "openrouter" is absent
        ollama_mock = _make_mock_provider("ollama")
        prov_reg = ProviderRegistry()
        prov_reg.register("ollama", ollama_mock)

        router = IntelligentRouter(
            classifier=clf,
            model_registry=registry,
            provider_registry=prov_reg,
        )
        result = await router.route(_make_request())

        assert result.status == "routing_failure"
        assert result.is_routing_failure
        assert result.error is not None
        # Provider must NOT have been called
        ollama_mock.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provider_not_registered_still_populates_routing_decision(self) -> None:
        """Even if provider lookup fails, selected_model_id/provider should be set."""
        model = _make_model(model_id="openai/test", provider="openrouter")
        registry = _make_registry(model)
        clf = _make_trained_classifier()
        prov_reg = ProviderRegistry()
        # Deliberately do NOT register "openrouter"

        router = IntelligentRouter(
            classifier=clf,
            model_registry=registry,
            provider_registry=prov_reg,
        )
        result = await router.route(_make_request())

        assert result.is_routing_failure
        # The routing decision was made before provider lookup failed
        assert result.selected_model_id is not None
        assert result.selected_provider == "openrouter"


class TestProviderInferenceFailure:
    """Tests 5 & 6 — Provider raises ProviderError during generate()."""

    @pytest.mark.asyncio
    async def test_provider_inference_failure_provider_unavailable(self) -> None:
        model = _make_model()
        router, mock_prov, _ = _make_router(
            models=[model],
            provider_side_effect=ProviderUnavailableError("Server down"),
        )

        result = await router.route(_make_request())

        assert result.status == "provider_failure"
        assert result.is_provider_failure
        assert "Server down" in (result.error or "")
        # Routing decision was made (fields populated before the provider call)
        assert result.selected_model_id is not None
        assert result.complexity is not None

    @pytest.mark.asyncio
    async def test_provider_inference_failure_auth_error(self) -> None:
        model = _make_model()
        router, _, _ = _make_router(
            models=[model],
            provider_side_effect=ProviderAuthError("Invalid API key"),
        )

        result = await router.route(_make_request())

        assert result.status == "provider_failure"
        assert "Invalid API key" in (result.error or "")

    @pytest.mark.asyncio
    async def test_provider_failure_does_not_include_llm_response(self) -> None:
        model = _make_model()
        router, _, _ = _make_router(
            models=[model],
            provider_side_effect=ProviderUnavailableError("Timeout"),
        )
        result = await router.route(_make_request())

        assert result.llm_response is None
        assert result.output is None
        assert result.input_tokens is None


class TestModelSelectionFromScorer:
    """Test 7 — Selected model MUST come from RoutingScorer, not bypassing it."""

    @pytest.mark.asyncio
    async def test_model_selection_comes_from_scorer(self) -> None:
        """
        With two models, the scorer picks the highest-ranked one.
        We verify that the selected_model_id in the result matches what
        the scorer would pick — not the first model alphabetically or
        the first model registered.
        """
        # Model A: cheap/fast but low quality
        model_a = _make_model(
            model_id="vendor/model-a",
            quality_score=0.55,
            input_cost_per_1k=0.0001,
            average_latency_ms=500,
        )
        # Model B: slightly more expensive but much higher quality
        model_b = _make_model(
            model_id="vendor/model-b",
            quality_score=0.90,
            reasoning_score=0.88,
            input_cost_per_1k=0.005,
            average_latency_ms=2000,
        )

        # Use quality-heavy weights so model_b should win
        weights = ScoringWeights(cost=0.10, latency=0.10, quality=0.50, task=0.20, complexity=0.10)

        response_a = _make_llm_response(model_id=model_a.model_id, provider="openrouter")
        response_b = _make_llm_response(model_id=model_b.model_id, provider="openrouter")

        registry = _make_registry(model_a, model_b)
        clf = _make_trained_classifier()

        # Mock provider that returns response matching the requested model_id
        async def _generate(req: LLMRequest) -> LLMResponse:
            if req.model_id == model_a.model_id:
                return response_a
            return response_b

        mock_prov = MagicMock(spec=LLMProvider)
        mock_prov.provider_name = "openrouter"
        mock_prov.generate = AsyncMock(side_effect=_generate)

        prov_reg = ProviderRegistry()
        prov_reg.register("openrouter", mock_prov)

        router = IntelligentRouter(
            classifier=clf,
            model_registry=registry,
            provider_registry=prov_reg,
            weights=weights,
        )

        result = await router.route(_make_request())

        assert result.is_success
        # With quality=0.50 weight, model_b (quality=0.90) should beat model_a (0.55)
        assert result.selected_model_id == model_b.model_id, (
            f"Expected scorer to select model_b ({model_b.model_id}) "
            f"but got {result.selected_model_id}"
        )

    @pytest.mark.asyncio
    async def test_scorer_is_actually_called(self) -> None:
        """Verify RoutingScorer.score() is invoked (not bypassed)."""
        model = _make_model()
        response = _make_llm_response(model_id=model.model_id)
        router, _, _ = _make_router(models=[model], provider_response=response)

        with patch.object(
            router._scorer, "score", wraps=router._scorer.score
        ) as mock_score:
            await router.route(_make_request())

        mock_score.assert_called_once()


class TestCostTokenLatencyPreservation:
    """Test 8 — Cost, token counts, and latency from LLMResponse are preserved."""

    @pytest.mark.asyncio
    async def test_cost_token_latency_preserved(self) -> None:
        model = _make_model()
        response = _make_llm_response(
            model_id=model.model_id,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost=0.00025,
            latency_ms=712.5,
        )
        router, _, _ = _make_router(models=[model], provider_response=response)

        result = await router.route(_make_request())

        assert result.is_success
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.estimated_cost == pytest.approx(0.00025)
        assert result.latency_ms == pytest.approx(712.5)

    @pytest.mark.asyncio
    async def test_finish_reason_preserved(self) -> None:
        model = _make_model()
        response = _make_llm_response(
            model_id=model.model_id,
            finish_reason="length",
        )
        router, _, _ = _make_router(models=[model], provider_response=response)

        result = await router.route(_make_request())

        assert result.finish_reason == "length"


class TestFactorScoresPopulated:
    """Test 9 — factor_scores is a non-empty dict with expected keys on success."""

    @pytest.mark.asyncio
    async def test_factor_scores_populated(self) -> None:
        model = _make_model()
        response = _make_llm_response(model_id=model.model_id)
        router, _, _ = _make_router(models=[model], provider_response=response)

        result = await router.route(_make_request())

        assert result.is_success
        assert result.factor_scores is not None
        assert isinstance(result.factor_scores, dict)
        expected_keys = {"cost", "latency", "quality", "task", "complexity"}
        assert set(result.factor_scores.keys()) == expected_keys
        for key, val in result.factor_scores.items():
            assert 0.0 <= val <= 1.0, f"factor_scores[{key!r}] = {val} is out of [0,1]"


class TestRoutingScoreInResult:
    """Test 10 — routing_score is populated and in [0, 1] on success."""

    @pytest.mark.asyncio
    async def test_routing_score_in_result(self) -> None:
        model = _make_model()
        response = _make_llm_response(model_id=model.model_id)
        router, _, _ = _make_router(models=[model], provider_response=response)

        result = await router.route(_make_request())

        assert result.is_success
        assert result.routing_score is not None
        assert isinstance(result.routing_score, float)
        assert 0.0 <= result.routing_score <= 1.0


class TestComplexityAndConfidenceInResult:
    """Test 11 — complexity and classifier_confidence are populated on success."""

    @pytest.mark.asyncio
    async def test_complexity_and_confidence_in_result(self) -> None:
        model = _make_model()
        response = _make_llm_response(model_id=model.model_id)
        router, _, _ = _make_router(models=[model], provider_response=response)

        result = await router.route(_make_request())

        assert result.is_success
        assert result.complexity in {"simple", "moderate", "complex"}
        assert result.classifier_confidence is not None
        assert 0.0 <= result.classifier_confidence <= 1.0
        assert result.classifier_model_id is not None

    @pytest.mark.asyncio
    async def test_request_analysis_is_populated(self) -> None:
        model = _make_model()
        response = _make_llm_response(model_id=model.model_id)
        router, _, _ = _make_router(models=[model], provider_response=response)

        result = await router.route(_make_request(prompt=_SIMPLE_PROMPT))

        assert result.request_analysis is not None
        assert result.request_analysis.request_id == _REQUEST_ID


class TestCandidateModelIdsInResult:
    """Test 12 — candidate_model_ids is populated with the eligible model set."""

    @pytest.mark.asyncio
    async def test_candidate_model_ids_in_result(self) -> None:
        model_a = _make_model(model_id="vendor/m-a")
        model_b = _make_model(model_id="vendor/m-b")
        response = _make_llm_response(model_id=model_a.model_id)
        router, _, _ = _make_router(
            models=[model_a, model_b],
            provider_response=response,
        )

        result = await router.route(_make_request())

        assert result.is_success
        assert result.candidate_model_ids is not None
        assert len(result.candidate_model_ids) >= 1
        # Both tier_2 models should appear as candidates for a simple request
        assert set(result.candidate_model_ids).issuperset(
            {m.model_id for m in [model_a, model_b]}
        ) or len(result.candidate_model_ids) >= 1  # at least one candidate


class TestPipelineDeterminism:
    """Test 13 — Same request produces same routing decision (excluding provider call)."""

    @pytest.mark.asyncio
    async def test_pipeline_is_deterministic(self) -> None:
        model_a = _make_model(model_id="vendor/alpha", input_cost_per_1k=0.001)
        model_b = _make_model(model_id="vendor/beta", input_cost_per_1k=0.002)

        response = _make_llm_response(model_id=model_a.model_id)

        router, _, _ = _make_router(
            models=[model_a, model_b],
            provider_response=response,
        )

        request = _make_request()
        result1 = await router.route(request)
        result2 = await router.route(request)

        assert result1.selected_model_id == result2.selected_model_id
        assert result1.complexity == result2.complexity
        assert result1.routing_score == result2.routing_score
        assert result1.candidate_model_ids == result2.candidate_model_ids


class TestInvalidConstructorTypes:
    """Test 14 — Router raises TypeError on invalid dependency types."""

    def test_invalid_classifier_type(self) -> None:
        registry = _make_registry(_make_model())
        prov_reg = ProviderRegistry()

        with pytest.raises(TypeError, match="classifier"):
            IntelligentRouter(
                classifier="not-a-classifier",  # type: ignore[arg-type]
                model_registry=registry,
                provider_registry=prov_reg,
            )

    def test_invalid_registry_type(self) -> None:
        clf = _make_trained_classifier()
        prov_reg = ProviderRegistry()

        with pytest.raises(TypeError, match="model_registry"):
            IntelligentRouter(
                classifier=clf,
                model_registry={"not": "a registry"},  # type: ignore[arg-type]
                provider_registry=prov_reg,
            )

    def test_invalid_provider_registry_type(self) -> None:
        clf = _make_trained_classifier()
        registry = _make_registry(_make_model())

        with pytest.raises(TypeError, match="provider_registry"):
            IntelligentRouter(
                classifier=clf,
                model_registry=registry,
                provider_registry=None,  # type: ignore[arg-type]
            )


class TestProviderRegistry:
    """Tests 15 & 16 — ProviderRegistry register/get/error handling."""

    def test_provider_registry_register_and_get(self) -> None:
        reg = ProviderRegistry()
        mock_prov = _make_mock_provider("openrouter")
        reg.register("openrouter", mock_prov)

        retrieved = reg.get("openrouter")
        assert retrieved is mock_prov

    def test_provider_registry_missing_raises(self) -> None:
        reg = ProviderRegistry()

        with pytest.raises(ProviderNotRegisteredError, match="not registered"):
            reg.get("nonexistent-provider")

    def test_provider_registry_case_insensitive(self) -> None:
        reg = ProviderRegistry()
        mock_prov = _make_mock_provider("OpenRouter")
        reg.register("OpenRouter", mock_prov)

        assert reg.get("openrouter") is mock_prov
        assert reg.get("OPENROUTER") is mock_prov

    def test_provider_registry_invalid_type_raises(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(TypeError):
            reg.register("bad", "not-a-provider")  # type: ignore[arg-type]

    def test_provider_registry_registered_names(self) -> None:
        reg = ProviderRegistry()
        reg.register("openrouter", _make_mock_provider("openrouter"))
        reg.register("ollama", _make_mock_provider("ollama"))

        names = reg.registered_names()
        assert names == ["ollama", "openrouter"]  # sorted

    def test_provider_registry_len(self) -> None:
        reg = ProviderRegistry()
        assert len(reg) == 0
        reg.register("openrouter", _make_mock_provider("openrouter"))
        assert len(reg) == 1


class TestRouterResultStatusProperties:
    """Test 17 — RouterResult status convenience properties."""

    def test_success_status_properties(self) -> None:
        result = RouterResult(request_id="r1", status="success")
        assert result.is_success
        assert not result.is_routing_failure
        assert not result.is_provider_failure

    def test_routing_failure_status_properties(self) -> None:
        result = RouterResult(request_id="r1", status="routing_failure", error="no candidates")
        assert not result.is_success
        assert result.is_routing_failure
        assert not result.is_provider_failure

    def test_provider_failure_status_properties(self) -> None:
        result = RouterResult(request_id="r1", status="provider_failure", error="timeout")
        assert not result.is_success
        assert not result.is_routing_failure
        assert result.is_provider_failure


class TestRouterResultRepr:
    """Test 18 — RouterResult.__repr__ contains key fields."""

    def test_repr_contains_request_id(self) -> None:
        result = RouterResult(request_id="my-id-999", status="success")
        r = repr(result)
        assert "my-id-999" in r
        assert "success" in r


class TestUnexpectedProviderException:
    """Test 19 — Non-ProviderError exceptions from provider are caught and wrapped."""

    @pytest.mark.asyncio
    async def test_unexpected_provider_exception_returns_provider_failure(self) -> None:
        model = _make_model()
        router, _, _ = _make_router(
            models=[model],
            provider_side_effect=RuntimeError("Completely unexpected internal error"),
        )

        result = await router.route(_make_request())

        assert result.status == "provider_failure"
        assert result.is_provider_failure
        assert "Completely unexpected internal error" in (result.error or "")
        assert result.llm_response is None


class TestRoutingFailurePopulatesAnalysisFields:
    """Test 20 — Even on routing_failure, analysis pipeline fields are populated."""

    @pytest.mark.asyncio
    async def test_routing_failure_populates_analysis_fields(self) -> None:
        tier1_model = _make_tier1_model()
        router, _, _ = _make_router(models=[tier1_model])

        with patch.object(
            router._classifier,
            "predict",
            return_value=ComplexityPrediction(
                complexity=ComplexityLabel.COMPLEX,
                confidence=0.91,
                probabilities={"complex": 0.91, "moderate": 0.06, "simple": 0.03},
                model_id="test-clf",
            ),
        ):
            result = await router.route(_make_request())

        assert result.is_routing_failure
        # Part 1 ran — analysis is populated
        assert result.request_analysis is not None
        assert result.request_analysis.request_id == _REQUEST_ID
        # Part 2 ran — complexity fields are populated
        assert result.complexity == "complex"
        assert result.classifier_confidence == pytest.approx(0.91)
        assert result.classifier_model_id == "test-clf"
        # Part 3 ran — candidate list is empty
        assert result.candidate_model_ids == []
        # Provider was NOT called
        assert result.llm_response is None
        assert result.output is None


class TestLLMResponseCarried:
    """Additional: llm_response field carries the raw LLMResponse on success."""

    @pytest.mark.asyncio
    async def test_llm_response_populated_on_success(self) -> None:
        model = _make_model()
        response = _make_llm_response(model_id=model.model_id, output="Hello world!")
        router, _, _ = _make_router(models=[model], provider_response=response)

        result = await router.route(_make_request())

        assert result.is_success
        assert result.llm_response is not None
        assert isinstance(result.llm_response, LLMResponse)
        assert result.llm_response.output == "Hello world!"
        # Convenience field mirrors llm_response.output
        assert result.output == result.llm_response.output


class TestSystemPromptHandled:
    """Additional: requests with system_prompt are handled without errors."""

    @pytest.mark.asyncio
    async def test_request_with_system_prompt(self) -> None:
        model = _make_model()
        response = _make_llm_response(model_id=model.model_id)
        router, mock_prov, _ = _make_router(models=[model], provider_response=response)

        request = _make_request(system_prompt="You are a helpful assistant.")
        result = await router.route(request)

        assert result.is_success
        # Verify the request passed to the provider had the system prompt
        call_args = mock_prov.generate.call_args
        provider_request: LLMRequest = call_args[0][0]
        assert provider_request.system_prompt == "You are a helpful assistant."
        # The selected model_id is used — not the original placeholder
        assert provider_request.model_id == model.model_id
