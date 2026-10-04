"""
app/evaluation/base.py
──────────────────────
Phase 3 Part 1 — Evaluator Interface.

Abstract protocol for response quality evaluators.
"""

from abc import ABC, abstractmethod

from app.providers.base import LLMRequest, LLMResponse
from app.evaluation.models import EvaluationResult


class BaseEvaluator(ABC):
    """
    Abstract base class for all quality evaluators.

    All evaluators (Deterministic, LLM-as-a-Judge, DeepEval, etc.) must implement
    this uniform interface. The pipeline routes the LLMRequest and LLMResponse
    to an evaluator, which yields a standardized EvaluationResult.
    """

    @abstractmethod
    async def evaluate(self, request: LLMRequest, response: LLMResponse) -> EvaluationResult:
        """
        Evaluate the quality of an LLMResponse.

        Implementations should catch their internal errors and return an
        EvaluationResult with status=FAILURE rather than raising exceptions
        to the caller.

        Args:
            request: The original request sent to the provider.
            response: The response returned by the provider.

        Returns:
            A fully constructed EvaluationResult detailing the quality scores,
            decisions, and potential errors.
        """
        pass
