"""
app/classification/complexity_classifier.py
────────────────────────────────────────────
Phase 2 Part 2 — Complexity Classification.

Accepts a RequestAnalysis (from Part 1) and predicts request complexity:
    simple | moderate | complex

Pipeline:
    LLMRequest
        ↓
    RequestAnalyzer  (Part 1)
        ↓
    RequestAnalysis
        ↓
    to_feature_vector()
        ↓
    ComplexityClassifier  ← this module
        ↓
    ComplexityPrediction

Two models are provided:
    - LogisticRegressionClassifier  (baseline)
    - LightGBMClassifier            (primary candidate)

Both share:
    - Identical input feature representation (RequestAnalysis.to_feature_vector())
    - Identical ComplexityLabel controlled vocabulary
    - Identical ComplexityPrediction output schema
    - Identical training/evaluation interface

Design rules:
    - No API keys required.
    - No external network access.
    - Deterministic with fixed random_state.
    - No routing logic, no model selection, no escalation.
"""

from __future__ import annotations

import json
import pathlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTGBM_AVAILABLE = False

from app.routing.analyzer import RequestAnalysis


# ── Controlled complexity vocabulary ─────────────────────────────────────────

class ComplexityLabel(str, Enum):
    """
    Controlled vocabulary for request complexity.

    Do NOT use raw strings for complexity levels anywhere in the codebase —
    always reference this enum so that Part 3 (candidate selection) and
    Part 4 (routing score) receive consistent types.
    """
    SIMPLE   = "simple"
    MODERATE = "moderate"
    COMPLEX  = "complex"


# Label ordering for LabelEncoder (alphabetical → consistent integer mapping)
_LABEL_ORDER = [ComplexityLabel.COMPLEX, ComplexityLabel.MODERATE, ComplexityLabel.SIMPLE]
_LABEL_STRINGS = [lbl.value for lbl in _LABEL_ORDER]  # ["complex", "moderate", "simple"]


# ── Prediction result schema ──────────────────────────────────────────────────

@dataclass
class ComplexityPrediction:
    """
    Structured output from any complexity classifier.

    Consumed downstream by:
        - Candidate Model Selection (Part 3)
        - Routing Score Calculator (Part 4)
    """
    complexity: ComplexityLabel
    confidence: float                    # Probability of the predicted class [0, 1]
    probabilities: dict[str, float]      # {label_value: probability}
    model_id: str                        # Which classifier produced this

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        for lbl in ComplexityLabel:
            if lbl.value not in self.probabilities:
                raise ValueError(
                    f"probabilities dict missing key '{lbl.value}'"
                )


# ── Evaluation result ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Evaluation metrics for a single model on a held-out test set."""
    model_id: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    classification_report: str
    n_test_samples: int
    label_order: list[str] = field(default_factory=lambda: _LABEL_STRINGS)

    def summary(self) -> str:
        return (
            f"{self.model_id} | n={self.n_test_samples} | "
            f"acc={self.accuracy:.3f} | f1={self.f1_macro:.3f}"
        )


# ── Abstract base classifier ──────────────────────────────────────────────────

class BaseComplexityClassifier(ABC):
    """
    Common interface for all complexity classifiers.

    Subclasses implement train() and _raw_predict_proba().
    The rest of the prediction pipeline (feature extraction, label decoding,
    ComplexityPrediction construction) is shared here.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._is_trained = False
        self._feature_names: list[str] = []
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(_LABEL_STRINGS)

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        analyses: list[RequestAnalysis],
        labels: list[ComplexityLabel],
    ) -> None:
        """
        Train the classifier on a list of (RequestAnalysis, ComplexityLabel) pairs.

        Args:
            analyses: Feature sources (one per training sample).
            labels:   Corresponding ground-truth complexity labels.
        """
        if len(analyses) != len(labels):
            raise ValueError(
                f"analyses ({len(analyses)}) and labels ({len(labels)}) must have "
                "the same length."
            )
        if not analyses:
            raise ValueError("Cannot train on an empty dataset.")

        X, feature_names = _build_feature_matrix(analyses)
        y = self._label_encoder.transform([lbl.value for lbl in labels])

        self._feature_names = feature_names
        self._fit(X, y)
        self._is_trained = True

    def predict(self, analysis: RequestAnalysis) -> ComplexityPrediction:
        """
        Predict the complexity of a single request.

        Args:
            analysis: Output of RequestAnalyzer.analyze().

        Returns:
            ComplexityPrediction with label, confidence, and probabilities.

        Raises:
            RuntimeError: If the classifier has not been trained yet.
        """
        self._require_trained()
        X = _analysis_to_row(analysis, self._feature_names)
        proba = self._predict_proba_row(X)  # shape: (n_classes,)

        # Map back using the LabelEncoder's class ordering
        classes = list(self._label_encoder.classes_)  # e.g. ["complex","moderate","simple"]
        prob_dict = {cls: float(proba[i]) for i, cls in enumerate(classes)}

        predicted_str = classes[int(np.argmax(proba))]
        predicted_label = ComplexityLabel(predicted_str)
        confidence = float(proba[np.argmax(proba)])

        return ComplexityPrediction(
            complexity=predicted_label,
            confidence=confidence,
            probabilities=prob_dict,
            model_id=self.model_id,
        )

    def evaluate(
        self,
        analyses: list[RequestAnalysis],
        labels: list[ComplexityLabel],
    ) -> EvaluationResult:
        """
        Evaluate the classifier on a held-out test set.

        Args:
            analyses: Test feature sources.
            labels:   Ground-truth labels.

        Returns:
            EvaluationResult with accuracy, precision, recall, F1, confusion matrix.
        """
        self._require_trained()
        if not analyses:
            raise ValueError("Cannot evaluate on an empty dataset.")

        X, _ = _build_feature_matrix(analyses, expected_features=self._feature_names)
        y_true = self._label_encoder.transform([lbl.value for lbl in labels])
        proba_matrix = self._predict_proba_matrix(X)
        y_pred = np.argmax(proba_matrix, axis=1)

        acc    = float(accuracy_score(y_true, y_pred))
        prec   = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1     = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm     = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(
            y_true, y_pred,
            target_names=self._label_encoder.classes_,
            zero_division=0,
        )

        return EvaluationResult(
            model_id=self.model_id,
            accuracy=acc,
            precision_macro=prec,
            recall_macro=rec,
            f1_macro=f1,
            confusion_matrix=cm,
            classification_report=report,
            n_test_samples=len(analyses),
            label_order=list(self._label_encoder.classes_),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: pathlib.Path) -> None:
        """Persist the trained classifier to a pickle file."""
        self._require_trained()
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_id": self.model_id,
            "feature_names": self._feature_names,
            "label_encoder": self._label_encoder,
            "model_state": self._get_model_state(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: pathlib.Path) -> None:
        """Restore a previously saved classifier."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model_id       = state["model_id"]
        self._feature_names = state["feature_names"]
        self._label_encoder = state["label_encoder"]
        self._restore_model_state(state["model_state"])
        self._is_trained = True

    # ── Abstract hooks ────────────────────────────────────────────────────────

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the underlying model. Called by train()."""

    @abstractmethod
    def _predict_proba_row(self, X_row: np.ndarray) -> np.ndarray:
        """Return class probabilities for a single sample (shape: n_classes,)."""

    @abstractmethod
    def _predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for multiple samples (shape: n_samples, n_classes)."""

    @abstractmethod
    def _get_model_state(self) -> Any:
        """Return serialisable model state for save()."""

    @abstractmethod
    def _restore_model_state(self, state: Any) -> None:
        """Restore model from state produced by _get_model_state()."""

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                f"{self.model_id} has not been trained. Call train() first."
            )


# ── Logistic Regression classifier ───────────────────────────────────────────

class LogisticRegressionClassifier(BaseComplexityClassifier):
    """
    Baseline complexity classifier using scikit-learn's LogisticRegression.

    Reproducible via random_state=42.
    Uses L2 regularisation with the 'lbfgs' solver and max 1000 iterations.
    """

    MODEL_ID = "logistic_regression_v1"

    def __init__(self) -> None:
        super().__init__(model_id=self.MODEL_ID)
        self._model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            C=1.0,
        )

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def _predict_proba_row(self, X_row: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_row.reshape(1, -1))[0]

    def _predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def _get_model_state(self) -> Any:
        return self._model

    def _restore_model_state(self, state: Any) -> None:
        self._model = state


# ── LightGBM classifier ───────────────────────────────────────────────────────

class LightGBMClassifier(BaseComplexityClassifier):
    """
    Primary complexity classifier using LightGBM.

    Reproducible via seed=42.
    Configured for small datasets: shallow trees, few leaves.
    """

    MODEL_ID = "lightgbm_v1"

    def __init__(self) -> None:
        if not _LIGHTGBM_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "lightgbm is not installed. Install it with: pip install lightgbm"
            )
        super().__init__(model_id=self.MODEL_ID)
        self._model: Optional[lgb.Booster] = None
        self._lgb_params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "num_leaves": 8,
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_samples": 1,   # allow small datasets
            "seed": 42,
            "verbose": -1,
        }

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        clf = lgb.LGBMClassifier(**self._lgb_params)
        clf.fit(X, y)
        self._model = clf

    def _predict_proba_row(self, X_row: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_row.reshape(1, -1))[0]

    def _predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def _get_model_state(self) -> Any:
        return self._model

    def _restore_model_state(self, state: Any) -> None:
        self._model = state


# ── Feature matrix utilities ──────────────────────────────────────────────────

def _get_feature_names(analysis: RequestAnalysis) -> list[str]:
    """Return a consistent, sorted list of feature names from one analysis."""
    return sorted(analysis.to_feature_vector().keys())


def _analysis_to_row(
    analysis: RequestAnalysis,
    feature_names: list[str],
) -> np.ndarray:
    """Convert a single RequestAnalysis to a 1D numpy array in feature_names order."""
    fv = analysis.to_feature_vector()
    return np.array([fv[name] for name in feature_names], dtype=float)


def _build_feature_matrix(
    analyses: list[RequestAnalysis],
    expected_features: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a (n_samples, n_features) numpy matrix from a list of RequestAnalysis.

    Args:
        analyses:          Feature sources.
        expected_features: If provided, use this feature ordering (for consistency
                           between train and predict).

    Returns:
        (X matrix, feature_names list)
    """
    feature_names = expected_features or _get_feature_names(analyses[0])
    rows = [_analysis_to_row(a, feature_names) for a in analyses]
    return np.array(rows, dtype=float), feature_names
