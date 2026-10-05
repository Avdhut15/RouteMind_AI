"""
app/evaluation/__init__.py
──────────────────────────
Phase 3 Part 1 — Evaluation Package Exports.
"""

from app.evaluation.models import (
    EvaluationStatus,
    EvaluationSource,
    QualityDecision,
    EvaluationResult,
)
from app.evaluation.base import BaseEvaluator
from app.evaluation.deterministic import DeterministicEvaluator
from app.evaluation.llm_judge import JudgeConfig, JudgeCaller, LLMJudgeEvaluator
from app.evaluation.deepeval_adapter import DeepEvalConfig, DeepEvalEvaluator
from app.evaluation.decision_engine import (
    ThresholdConfig,
    QualityDecisionResult,
    QualityDecisionEngine,
    DEFAULT_PASS_THRESHOLD,
    DEFAULT_ESCALATE_THRESHOLD,
)

__all__ = [
    "EvaluationStatus",
    "EvaluationSource",
    "QualityDecision",
    "EvaluationResult",
    "BaseEvaluator",
    "DeterministicEvaluator",
    "JudgeConfig",
    "JudgeCaller",
    "LLMJudgeEvaluator",
    "DeepEvalConfig",
    "DeepEvalEvaluator",
    "ThresholdConfig",
    "QualityDecisionResult",
    "QualityDecisionEngine",
    "DEFAULT_PASS_THRESHOLD",
    "DEFAULT_ESCALATE_THRESHOLD",
]
