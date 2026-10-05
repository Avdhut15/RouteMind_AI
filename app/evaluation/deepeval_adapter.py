"""
app/evaluation/deepeval_adapter.py
───────────────────────────────────
Phase 3 Part 3 — DeepEval Adapter.

Wraps DeepEval's evaluation metrics behind RouteMind_AI's BaseEvaluator
contract so the rest of the platform never depends on DeepEval internals.

Design rules:
  - DeepEval is imported lazily so the package is optional.
  - If deepeval is not installed, the evaluator raises ImportError at
    construction time with a clear message.
  - All metric scores are converted from DeepEval's result structure
    into EvaluationResult using the project's existing models.
  - Network/model failures are caught and returned as FAILURE results.
  - No API keys are embedded here — credentials come from environment vars
    consumed by DeepEval internally (OPENAI_API_KEY, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.models import (
    EvaluationResult,
    EvaluationSource,
    EvaluationStatus,
)
from app.providers.base import LLMRequest, LLMResponse


@dataclass
class DeepEvalConfig:
    """
    Configuration for the DeepEval adapter.

    Attributes:
        model:           Judge model name passed to DeepEval metrics
                         (e.g. 'gpt-4o').
        threshold:       Not used for scoring decisions here (that belongs to
                         Part 4); passed to DeepEval metric constructors that
                         require it, defaulting to 0.5 as a neutral midpoint.
        include_reason:  Whether to request per-metric reasoning from DeepEval.
        metric_names:    Which DeepEval metrics to run.  Defaults to a
                         lightweight set that avoids heavy network calls.
    """
    model: str = "gpt-4o-mini"
    threshold: float = 0.5
    include_reason: bool = True
    metric_names: list[str] = field(default_factory=lambda: [
        "answer_relevancy",
        "faithfulness",
    ])


class DeepEvalEvaluator(BaseEvaluator):
    """
    DeepEval-backed evaluator.

    Delegates metric computation to the ``deepeval`` library and converts
    the resulting ``TestCase`` / metric objects into the project's unified
    ``EvaluationResult``.

    Example::

        cfg = DeepEvalConfig(model="gpt-4o-mini")
        evaluator = DeepEvalEvaluator(config=cfg)
        result = await evaluator.evaluate(request, response)
    """

    def __init__(self, config: Optional[DeepEvalConfig] = None) -> None:
        try:
            import deepeval  # noqa: F401 – presence check
        except ImportError as exc:
            raise ImportError(
                "DeepEval is not installed. "
                "Install it with: pip install deepeval"
            ) from exc

        self._config = config or DeepEvalConfig()

    async def evaluate(self, request: LLMRequest, response: LLMResponse) -> EvaluationResult:
        req_id = getattr(request, "request_id", "unknown")
        mod_id = getattr(response, "model_id", "unknown")
        prov = getattr(response, "provider", "unknown")

        try:
            if request is None or response is None:
                raise ValueError("Both request and response must be provided.")

            from deepeval import evaluate as deepeval_evaluate
            from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
            from deepeval.test_case import LLMTestCase

            # Build a DeepEval test case from our request/response
            test_case = LLMTestCase(
                input=request.prompt,
                actual_output=response.output or "",
                # context and retrieval_context are optional in DeepEval
            )

            # Construct the requested metrics
            metric_map: dict[str, object] = {}
            for name in self._config.metric_names:
                if name == "answer_relevancy":
                    metric_map[name] = AnswerRelevancyMetric(
                        threshold=self._config.threshold,
                        model=self._config.model,
                        include_reason=self._config.include_reason,
                    )
                elif name == "faithfulness":
                    metric_map[name] = FaithfulnessMetric(
                        threshold=self._config.threshold,
                        model=self._config.model,
                        include_reason=self._config.include_reason,
                    )
                else:
                    raise ValueError(
                        f"Unsupported DeepEval metric: '{name}'. "
                        f"Add it to DeepEvalEvaluator if needed."
                    )

            # Run evaluation synchronously inside async context
            # DeepEval's evaluate() is synchronous; wrap safely
            import asyncio

            loop = asyncio.get_event_loop()
            metrics_list = list(metric_map.values())
            await loop.run_in_executor(
                None,
                lambda: deepeval_evaluate([test_case], metrics=metrics_list, print_results=False),
            )

            # Collect results
            individual: dict[str, float] = {}
            reasons: list[str] = []
            for name, metric in metric_map.items():
                score = getattr(metric, "score", None)
                if score is None:
                    raise ValueError(f"DeepEval metric '{name}' returned no score.")
                clamped = max(0.0, min(1.0, float(score)))
                individual[name] = round(clamped, 4)

                reason = getattr(metric, "reason", None)
                if reason:
                    reasons.append(f"{name}: {reason}")

            overall = round(sum(individual.values()) / len(individual), 4) if individual else 0.0
            explanation = "; ".join(reasons) if reasons else "DeepEval evaluation completed."

            return EvaluationResult(
                request_id=request.request_id,
                model_id=response.model_id,
                provider=response.provider,
                status=EvaluationStatus.SUCCESS,
                source=EvaluationSource.DEEPEVAL,
                overall_quality_score=overall,
                individual_scores=individual,
                explanation=explanation,
            )

        except Exception as exc:
            return EvaluationResult(
                request_id=req_id,
                model_id=mod_id,
                provider=prov,
                status=EvaluationStatus.FAILURE,
                source=EvaluationSource.DEEPEVAL,
                error=f"DeepEval evaluation failed: {type(exc).__name__}: {exc}",
            )
