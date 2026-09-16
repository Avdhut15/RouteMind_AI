"""
app/routing/router.py
──────────────────────
Phase 2 Part 5 — IntelligentRouter: end-to-end routing orchestrator.

Wires Parts 1–4 together into a single deterministic routing pipeline,
then drives provider execution through the Phase 1 LLMProvider interface.

Pipeline
────────
    LLMRequest
        │
        ▼ Part 1
    RequestAnalyzer.analyze()  →  RequestAnalysis
        │
        ▼ Part 2
    BaseComplexityClassifier.predict()  →  ComplexityPrediction
        │
        ▼ Part 3
    CandidateSelector.select()  →  CandidateSelectionResult
        │  (no candidates?)  →  RouterResult(status="routing_failure")
        │
        ▼ Part 4
    RoutingScorer.score()  →  RoutingScoreResult
        │  (no selection?)  →  RouterResult(status="routing_failure")
        │
        ▼  Resolve provider
    ProviderRegistry.get(provider_name)
        │  (not registered?) →  RouterResult(status="routing_failure")
        │
        ▼  Mutate request model_id → selected model_id
    LLMProvider.generate(request)
        │  (ProviderError?)  →  RouterResult(status="provider_failure")
        │
        ▼
    RouterResult(status="success", ...)

Design rules
────────────
    - The routing decision (Parts 1–4) is deterministic and runs BEFORE any
      provider call. Provider execution only happens after a valid model is
      selected.
    - All failure modes return a RouterResult — they never propagate exceptions
      to the caller unless there is a programming error (TypeError, etc.).
    - Dependencies (classifier, registry, provider_registry, weights) are all
      injected via the constructor so the router can be unit-tested with mocks.
    - The router itself holds no per-request mutable state; route() is re-entrant.
    - RequestAnalyzer, CandidateSelector, and RoutingScorer are created once
      from the injected dependencies.
    - The request_id on the incoming LLMRequest is preserved throughout and
      echoed in every RouterResult.

Logging
───────
    Structured JSON log entries are emitted at each stage using the project
    logger (get_logger(__name__)).  No raw prompt text is logged — only the
    prompt hash via hash_prompt().

Phase 2 Part 5 scope:
    - IntelligentRouter class.
    - route() async method.
    - Internal helpers for each failure mode.
"""

from __future__ import annotations

import time
from typing import Optional

from app.classification.complexity_classifier import BaseComplexityClassifier
from app.core.exceptions import (
    ProviderError,
    ProviderNotRegisteredError,
    RoutingError,
)
from app.logging.logger import get_logger, hash_prompt, log_inference_event
from app.models.registry import ModelRegistry
from app.providers.base import LLMRequest, LLMResponse
from app.providers.provider_registry import ProviderRegistry
from app.routing.analyzer import RequestAnalyzer
from app.routing.candidate_selector import CandidateSelector
from app.routing.result import RouterResult
from app.routing.scoring import RoutingScorer, ScoringWeights

logger = get_logger(__name__)


class IntelligentRouter:
    """
    End-to-end intelligent routing orchestrator for Phase 2 Part 5.

    Accepts an LLMRequest, runs the full routing pipeline (analysis →
    complexity → candidates → scoring → model selection), resolves the
    appropriate provider, executes inference, and returns a RouterResult.

    All dependencies are injected at construction time so the router is
    unit-testable without API keys or a running Ollama instance.

    Usage::

        from app.routing.router import IntelligentRouter

        router = IntelligentRouter(
            classifier=trained_classifier,
            model_registry=registry,
            provider_registry=provider_reg,
        )
        result = await router.route(request)

        if result.is_success:
            print(result.output)
        elif result.is_routing_failure:
            print(f"Routing failed: {result.error}")
        else:
            print(f"Provider failed: {result.error}")

    Thread-safety: route() is re-entrant (no per-request mutable state on
    the router itself). The underlying provider implementations may or may
    not be thread-safe — check their documentation.
    """

    def __init__(
        self,
        classifier: BaseComplexityClassifier,
        model_registry: ModelRegistry,
        provider_registry: ProviderRegistry,
        weights: Optional[ScoringWeights] = None,
    ) -> None:
        """
        Initialise the router with its injected dependencies.

        Args:
            classifier:        A fully-trained complexity classifier instance.
                               Must have been trained before calling route().
            model_registry:    ModelRegistry loaded from config/models.yaml.
            provider_registry: ProviderRegistry mapping provider names to
                               LLMProvider instances.
            weights:           Optional ScoringWeights for Part 4.  Defaults
                               to ScoringWeights() (equal default weights).

        Raises:
            TypeError: If classifier is not a BaseComplexityClassifier.
            TypeError: If model_registry is not a ModelRegistry.
            TypeError: If provider_registry is not a ProviderRegistry.
        """
        if not isinstance(classifier, BaseComplexityClassifier):
            raise TypeError(
                f"classifier must be a BaseComplexityClassifier, got {type(classifier)}"
            )
        if not isinstance(model_registry, ModelRegistry):
            raise TypeError(
                f"model_registry must be a ModelRegistry, got {type(model_registry)}"
            )
        if not isinstance(provider_registry, ProviderRegistry):
            raise TypeError(
                f"provider_registry must be a ProviderRegistry, got {type(provider_registry)}"
            )

        self._classifier = classifier
        self._model_registry = model_registry
        self._provider_registry = provider_registry
        self._weights = weights

        # Build stateless sub-components once.
        self._analyzer = RequestAnalyzer()
        self._selector = CandidateSelector(model_registry)
        self._scorer = RoutingScorer(weights)

    # ── Public API ────────────────────────────────────────────────────────────

    async def route(self, request: LLMRequest) -> RouterResult:
        """
        Execute the full routing pipeline for a single LLMRequest.

        The routing decision is made BEFORE any provider call.  If no valid
        model can be selected, a RouterResult with status="routing_failure"
        is returned immediately — no provider is contacted.

        Args:
            request: The incoming LLMRequest.  The router will set
                     request.model_id to the selected model before calling
                     the provider (a copy is made so the caller's object is
                     not mutated).

        Returns:
            RouterResult with one of three statuses:
                "success"          – inference completed; output is populated.
                "routing_failure"  – pipeline could not pick a model.
                "provider_failure" – model selected but provider call failed.
        """
        request_id = request.request_id

        logger.info(
            "Routing pipeline started",
            extra={
                "event": "routing_start",
                "request_id": request_id,
                "prompt_hash": hash_prompt(request.prompt),
                "max_tokens": request.max_tokens,
            },
        )

        # ── Part 1: Feature extraction ────────────────────────────────────────
        analysis = self._analyzer.analyze(request)
        logger.debug(
            "Request analysis complete",
            extra={
                "event": "analysis_complete",
                "request_id": request_id,
                "task_type": analysis.task_type.value,
                "approx_token_count": analysis.approx_token_count,
                "requires_reasoning": analysis.requires_reasoning,
                "requires_code": analysis.requires_code,
            },
        )

        # ── Part 2: Complexity classification ─────────────────────────────────
        complexity_prediction = self._classifier.predict(analysis)
        logger.debug(
            "Complexity classified",
            extra={
                "event": "complexity_classified",
                "request_id": request_id,
                "complexity": complexity_prediction.complexity.value,
                "confidence": complexity_prediction.confidence,
                "classifier_model_id": complexity_prediction.model_id,
            },
        )

        # ── Part 3: Candidate selection ───────────────────────────────────────
        candidate_result = self._selector.select(analysis, complexity_prediction)
        logger.debug(
            "Candidate selection complete",
            extra={
                "event": "candidates_selected",
                "request_id": request_id,
                "n_candidates": len(candidate_result.candidates),
                "n_rejected": len(candidate_result.rejected),
                "candidate_ids": candidate_result.candidate_ids,
            },
        )

        if not candidate_result.has_candidates:
            return self._routing_failure(
                request_id=request_id,
                analysis=analysis,
                complexity_prediction=complexity_prediction,
                candidate_result_ids=[],
                error=(
                    f"No eligible models for request '{request_id}' "
                    f"(complexity={complexity_prediction.complexity.value}, "
                    f"filters={candidate_result.applied_filters})."
                ),
            )

        # ── Part 4: Routing score ─────────────────────────────────────────────
        score_result = self._scorer.score(candidate_result, analysis)
        logger.debug(
            "Routing score complete",
            extra={
                "event": "routing_scored",
                "request_id": request_id,
                "selected_model": (
                    score_result.selected_model.model_id
                    if score_result.selected_model else None
                ),
                "routing_score": (
                    score_result.best.routing_score if score_result.best else None
                ),
            },
        )

        if not score_result.has_selection:
            # Should not occur after has_candidates check, but guard defensively.
            return self._routing_failure(
                request_id=request_id,
                analysis=analysis,
                complexity_prediction=complexity_prediction,
                candidate_result_ids=candidate_result.candidate_ids,
                error=(
                    f"RoutingScorer returned no selection for request '{request_id}' "
                    "despite non-empty candidates — invalid routing state."
                ),
            )

        selected_model = score_result.selected_model
        best_score = score_result.best

        # ── Resolve provider ──────────────────────────────────────────────────
        try:
            provider = self._provider_registry.get(selected_model.provider)
        except ProviderNotRegisteredError as exc:
            return self._routing_failure(
                request_id=request_id,
                analysis=analysis,
                complexity_prediction=complexity_prediction,
                candidate_result_ids=candidate_result.candidate_ids,
                selected_model_id=selected_model.model_id,
                selected_provider=selected_model.provider,
                routing_score=best_score.routing_score if best_score else None,
                factor_scores=best_score.factor_scores if best_score else None,
                error=str(exc),
            )

        # ── Execute inference ─────────────────────────────────────────────────
        # Build a provider request with the selected model_id.
        # We create a new LLMRequest so the caller's object is not mutated.
        provider_request = request.model_copy(
            update={"model_id": selected_model.model_id}
        )

        logger.info(
            "Provider call starting",
            extra={
                "event": "provider_call_start",
                "request_id": request_id,
                "model_id": selected_model.model_id,
                "provider": selected_model.provider,
            },
        )

        t_start = time.perf_counter()
        try:
            llm_response: LLMResponse = await provider.generate(provider_request)
        except ProviderError as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            error_str = str(exc)
            logger.error(
                "Provider call failed",
                extra={
                    "event": "provider_call_failed",
                    "request_id": request_id,
                    "model_id": selected_model.model_id,
                    "provider": selected_model.provider,
                    "error": error_str,
                    "elapsed_ms": elapsed_ms,
                },
            )
            return RouterResult(
                request_id=request_id,
                status="provider_failure",
                error=error_str,
                request_analysis=analysis,
                complexity=complexity_prediction.complexity.value,
                classifier_confidence=complexity_prediction.confidence,
                classifier_model_id=complexity_prediction.model_id,
                candidate_model_ids=candidate_result.candidate_ids,
                selected_model_id=selected_model.model_id,
                selected_provider=selected_model.provider,
                routing_score=best_score.routing_score if best_score else None,
                factor_scores=best_score.factor_scores if best_score else None,
            )
        except Exception as exc:
            # Catch unexpected non-ProviderError exceptions defensively so the
            # router always returns a structured result rather than propagating
            # an uncaught exception to the caller.
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            error_str = f"Unexpected provider error: {exc!r}"
            logger.error(
                "Unexpected provider exception",
                extra={
                    "event": "provider_unexpected_error",
                    "request_id": request_id,
                    "model_id": selected_model.model_id,
                    "provider": selected_model.provider,
                    "error": error_str,
                    "elapsed_ms": elapsed_ms,
                },
            )
            return RouterResult(
                request_id=request_id,
                status="provider_failure",
                error=error_str,
                request_analysis=analysis,
                complexity=complexity_prediction.complexity.value,
                classifier_confidence=complexity_prediction.confidence,
                classifier_model_id=complexity_prediction.model_id,
                candidate_model_ids=candidate_result.candidate_ids,
                selected_model_id=selected_model.model_id,
                selected_provider=selected_model.provider,
                routing_score=best_score.routing_score if best_score else None,
                factor_scores=best_score.factor_scores if best_score else None,
            )

        # ── Build structured success result ───────────────────────────────────
        logger.info(
            "Routing pipeline complete",
            extra={
                "event": "routing_complete",
                "request_id": request_id,
                "model_id": llm_response.model_id,
                "provider": llm_response.provider,
                "complexity": complexity_prediction.complexity.value,
                "routing_score": best_score.routing_score if best_score else None,
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
                "estimated_cost_usd": llm_response.estimated_cost,
                "latency_ms": llm_response.latency_ms,
            },
        )

        return RouterResult(
            request_id=request_id,
            status="success",
            error=None,
            # Analysis pipeline
            request_analysis=analysis,
            complexity=complexity_prediction.complexity.value,
            classifier_confidence=complexity_prediction.confidence,
            classifier_model_id=complexity_prediction.model_id,
            # Candidate selection
            candidate_model_ids=candidate_result.candidate_ids,
            # Routing decision
            selected_model_id=llm_response.model_id,
            selected_provider=llm_response.provider,
            routing_score=best_score.routing_score if best_score else None,
            factor_scores=best_score.factor_scores if best_score else None,
            # Provider execution
            llm_response=llm_response,
            output=llm_response.output,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            total_tokens=llm_response.total_tokens,
            estimated_cost=llm_response.estimated_cost,
            latency_ms=llm_response.latency_ms,
            finish_reason=llm_response.finish_reason,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _routing_failure(
        self,
        *,
        request_id: str,
        analysis: object,
        complexity_prediction: object,
        candidate_result_ids: list[str],
        error: str,
        selected_model_id: Optional[str] = None,
        selected_provider: Optional[str] = None,
        routing_score: Optional[float] = None,
        factor_scores: Optional[dict[str, float]] = None,
    ) -> RouterResult:
        """
        Build a RouterResult for any routing-stage failure.

        Populates as many fields as are available up to the point of failure
        so callers get maximum diagnostic context even on the failure path.
        """
        # complexity_prediction and analysis may be partially typed here;
        # access attributes carefully.
        complexity_val: Optional[str] = None
        confidence_val: Optional[float] = None
        classifier_model_id_val: Optional[str] = None
        request_analysis_val = None

        try:
            complexity_val = complexity_prediction.complexity.value  # type: ignore[union-attr]
            confidence_val = complexity_prediction.confidence         # type: ignore[union-attr]
            classifier_model_id_val = complexity_prediction.model_id # type: ignore[union-attr]
        except AttributeError:
            pass

        try:
            request_analysis_val = analysis  # type: ignore[assignment]
        except Exception:
            pass

        logger.warning(
            "Routing failure",
            extra={
                "event": "routing_failure",
                "request_id": request_id,
                "complexity": complexity_val,
                "n_candidates": len(candidate_result_ids),
                "error": error,
            },
        )

        return RouterResult(
            request_id=request_id,
            status="routing_failure",
            error=error,
            request_analysis=request_analysis_val,
            complexity=complexity_val,
            classifier_confidence=confidence_val,
            classifier_model_id=classifier_model_id_val,
            candidate_model_ids=candidate_result_ids,
            selected_model_id=selected_model_id,
            selected_provider=selected_provider,
            routing_score=routing_score,
            factor_scores=factor_scores,
        )
