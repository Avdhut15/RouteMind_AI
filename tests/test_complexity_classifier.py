"""
tests/test_complexity_classifier.py
─────────────────────────────────────
Unit tests for Phase 2 Part 2 — Complexity Classification.

All tests are:
    - Offline (no API calls, no network)
    - Reproducible (fixed random_state / seeds throughout)
    - Independent of OpenRouter, Ollama, or any LLM

Coverage (see block docstrings below):
    - Dataset loading and validation
    - Valid complexity labels
    - Feature preparation / matrix building
    - Logistic Regression: training, prediction, evaluation
    - LightGBM: training, prediction, evaluation
    - Prediction output structure and valid values
    - Probability / confidence output
    - Model persistence (save / load)
    - Deterministic / reproducible behavior
    - Invalid input handling
    - Empty / minimal input handling
    - ComplexityLabel enum completeness
    - DatasetSplit structure
"""

import json
import pathlib
import tempfile

import numpy as np
import pytest

from app.classification.complexity_classifier import (
    ComplexityLabel,
    ComplexityPrediction,
    EvaluationResult,
    LightGBMClassifier,
    LogisticRegressionClassifier,
    _build_feature_matrix,
    _get_feature_names,
)
from app.classification.dataset import (
    DatasetSplit,
    LabeledSample,
    load_labeled_samples,
    make_dataset_split,
)
from app.providers.base import LLMRequest
from app.routing.analyzer import RequestAnalyzer


# ── Constants ─────────────────────────────────────────────────────────────────

_REAL_DATASET_PATH = pathlib.Path("app/classification/complexity_dataset.json")
_ANALYZER = RequestAnalyzer()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_analysis(prompt: str, max_tokens: int = 1024):
    req = LLMRequest(prompt=prompt, model_id="__test__", max_tokens=max_tokens)
    return _ANALYZER.analyze(req)


def _minimal_dataset() -> tuple[list, list]:
    """Return a minimal but balanced dataset: 3 samples per class."""
    prompts_labels = [
        ("What is 2 + 2?", ComplexityLabel.SIMPLE),
        ("What is the capital of France?", ComplexityLabel.SIMPLE),
        ("List the planets in one sentence.", ComplexityLabel.SIMPLE),
        ("Write a Python function to sort a list.", ComplexityLabel.MODERATE),
        ("Summarize the key differences between TCP and UDP.", ComplexityLabel.MODERATE),
        ("Debug this code: def f(): return 1/0", ComplexityLabel.MODERATE),
        (
            "Design a distributed consensus algorithm with formal proofs and benchmarks.",
            ComplexityLabel.COMPLEX,
        ),
        (
            "Analyze trade-offs in transformer vs recurrent architectures for NLP.",
            ComplexityLabel.COMPLEX,
        ),
        (
            "Implement a production-ready rate limiter with Redis-backed state.",
            ComplexityLabel.COMPLEX,
        ),
    ]
    analyses = [_make_analysis(p) for p, _ in prompts_labels]
    labels = [lbl for _, lbl in prompts_labels]
    return analyses, labels


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_samples() -> list[LabeledSample]:
    """Load the real dataset once for the module."""
    return load_labeled_samples()


@pytest.fixture(scope="module")
def full_split(full_samples) -> DatasetSplit:
    return make_dataset_split(full_samples)


@pytest.fixture(scope="module")
def trained_lr(full_split) -> LogisticRegressionClassifier:
    clf = LogisticRegressionClassifier()
    clf.train(full_split.train_analyses, full_split.train_labels)
    return clf


@pytest.fixture(scope="module")
def trained_lgbm(full_split) -> LightGBMClassifier:
    clf = LightGBMClassifier()
    clf.train(full_split.train_analyses, full_split.train_labels)
    return clf


# ── 1. Dataset loading and validation ─────────────────────────────────────────

class TestDatasetLoading:

    def test_loads_real_dataset(self, full_samples):
        assert len(full_samples) == 90

    def test_all_samples_have_analysis(self, full_samples):
        for s in full_samples:
            assert s.analysis is not None

    def test_all_samples_have_valid_labels(self, full_samples):
        valid = {lbl for lbl in ComplexityLabel}
        for s in full_samples:
            assert s.label in valid

    def test_label_distribution_is_balanced(self, full_samples):
        from collections import Counter
        counts = Counter(s.label for s in full_samples)
        assert counts[ComplexityLabel.SIMPLE]   == 30
        assert counts[ComplexityLabel.MODERATE] == 30
        assert counts[ComplexityLabel.COMPLEX]  == 30

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_labeled_samples(dataset_path=tmp_path / "nonexistent.json")

    def test_invalid_json_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json {{", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_labeled_samples(dataset_path=bad)

    def test_invalid_label_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad_label.json"
        bad.write_text(
            json.dumps({"samples": [{"prompt": "hello", "label": "super_hard"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid label"):
            load_labeled_samples(dataset_path=bad)

    def test_empty_samples_raises_value_error(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"samples": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_labeled_samples(dataset_path=empty)

    def test_missing_samples_key_raises_value_error(self, tmp_path):
        bad = tmp_path / "no_key.json"
        bad.write_text(json.dumps({"_meta": {}}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_labeled_samples(dataset_path=bad)


# ── 2. Valid complexity labels ────────────────────────────────────────────────

class TestComplexityLabel:

    def test_enum_has_three_values(self):
        assert len(list(ComplexityLabel)) == 3

    def test_simple_value(self):
        assert ComplexityLabel.SIMPLE.value == "simple"

    def test_moderate_value(self):
        assert ComplexityLabel.MODERATE.value == "moderate"

    def test_complex_value(self):
        assert ComplexityLabel.COMPLEX.value == "complex"

    def test_from_string(self):
        assert ComplexityLabel("simple")   == ComplexityLabel.SIMPLE
        assert ComplexityLabel("moderate") == ComplexityLabel.MODERATE
        assert ComplexityLabel("complex")  == ComplexityLabel.COMPLEX

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            ComplexityLabel("ultra_hard")


# ── 3. Feature preparation ────────────────────────────────────────────────────

class TestFeaturePreparation:

    def test_feature_names_are_sorted_strings(self, full_samples):
        names = _get_feature_names(full_samples[0].analysis)
        assert names == sorted(names)
        assert all(isinstance(n, str) for n in names)

    def test_feature_matrix_shape(self, full_samples):
        analyses = [s.analysis for s in full_samples[:10]]
        X, names = _build_feature_matrix(analyses)
        assert X.shape == (10, len(names))

    def test_feature_matrix_dtype_is_float(self, full_samples):
        analyses = [s.analysis for s in full_samples[:5]]
        X, _ = _build_feature_matrix(analyses)
        assert X.dtype == float

    def test_feature_matrix_no_nan(self, full_samples):
        analyses = [s.analysis for s in full_samples]
        X, _ = _build_feature_matrix(analyses)
        assert not np.isnan(X).any()

    def test_feature_ordering_consistent_across_samples(self, full_samples):
        names1 = _get_feature_names(full_samples[0].analysis)
        names2 = _get_feature_names(full_samples[-1].analysis)
        assert names1 == names2


# ── 4. DatasetSplit structure ──────────────────────────────────────────────────

class TestDatasetSplit:

    def test_split_sizes_sum_to_total(self, full_split, full_samples):
        assert full_split.n_train + full_split.n_test == len(full_samples)

    def test_test_size_approximately_30_percent(self, full_split, full_samples):
        ratio = full_split.n_test / len(full_samples)
        assert abs(ratio - 0.30) < 0.05

    def test_train_labels_count_matches_analyses(self, full_split):
        assert len(full_split.train_analyses) == len(full_split.train_labels)

    def test_test_labels_count_matches_analyses(self, full_split):
        assert len(full_split.test_analyses) == len(full_split.test_labels)

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            make_dataset_split([])

    def test_split_is_stratified(self, full_split):
        """Each class should appear in the test set."""
        test_labels = set(full_split.test_labels)
        for lbl in ComplexityLabel:
            assert lbl in test_labels


# ── 5. Logistic Regression training ──────────────────────────────────────────

class TestLogisticRegressionTraining:

    def test_train_does_not_raise(self):
        analyses, labels = _minimal_dataset()
        clf = LogisticRegressionClassifier()
        clf.train(analyses, labels)
        assert clf._is_trained

    def test_model_id(self):
        clf = LogisticRegressionClassifier()
        assert clf.model_id == "logistic_regression_v1"

    def test_mismatched_lengths_raises(self):
        analyses, labels = _minimal_dataset()
        clf = LogisticRegressionClassifier()
        with pytest.raises(ValueError, match="same length"):
            clf.train(analyses, labels[:-1])

    def test_empty_dataset_raises(self):
        clf = LogisticRegressionClassifier()
        with pytest.raises(ValueError, match="empty"):
            clf.train([], [])

    def test_predicting_before_training_raises(self):
        clf = LogisticRegressionClassifier()
        analysis = _make_analysis("hello")
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(analysis)


# ── 6. Logistic Regression prediction ────────────────────────────────────────

class TestLogisticRegressionPrediction:

    def test_returns_complexity_prediction(self, trained_lr, full_split):
        pred = trained_lr.predict(full_split.test_analyses[0])
        assert isinstance(pred, ComplexityPrediction)

    def test_complexity_is_valid_label(self, trained_lr, full_split):
        pred = trained_lr.predict(full_split.test_analyses[0])
        assert pred.complexity in ComplexityLabel

    def test_confidence_in_range(self, trained_lr, full_split):
        for analysis in full_split.test_analyses:
            pred = trained_lr.predict(analysis)
            assert 0.0 <= pred.confidence <= 1.0

    def test_probabilities_sum_to_one(self, trained_lr, full_split):
        pred = trained_lr.predict(full_split.test_analyses[0])
        total = sum(pred.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_probabilities_has_all_classes(self, trained_lr, full_split):
        pred = trained_lr.predict(full_split.test_analyses[0])
        for lbl in ComplexityLabel:
            assert lbl.value in pred.probabilities

    def test_model_id_in_prediction(self, trained_lr, full_split):
        pred = trained_lr.predict(full_split.test_analyses[0])
        assert pred.model_id == "logistic_regression_v1"

    def test_simple_prompt_not_complex(self, trained_lr):
        """A trivially simple prompt should not be predicted as complex."""
        analysis = _make_analysis("What is 2 + 2?")
        pred = trained_lr.predict(analysis)
        assert pred.complexity != ComplexityLabel.COMPLEX


# ── 7. LightGBM training ──────────────────────────────────────────────────────

class TestLightGBMTraining:

    def test_train_does_not_raise(self):
        analyses, labels = _minimal_dataset()
        clf = LightGBMClassifier()
        clf.train(analyses, labels)
        assert clf._is_trained

    def test_model_id(self):
        clf = LightGBMClassifier()
        assert clf.model_id == "lightgbm_v1"

    def test_mismatched_lengths_raises(self):
        analyses, labels = _minimal_dataset()
        clf = LightGBMClassifier()
        with pytest.raises(ValueError, match="same length"):
            clf.train(analyses, labels[:-1])

    def test_predicting_before_training_raises(self):
        clf = LightGBMClassifier()
        analysis = _make_analysis("hello")
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(analysis)


# ── 8. LightGBM prediction ────────────────────────────────────────────────────

class TestLightGBMPrediction:

    def test_returns_complexity_prediction(self, trained_lgbm, full_split):
        pred = trained_lgbm.predict(full_split.test_analyses[0])
        assert isinstance(pred, ComplexityPrediction)

    def test_complexity_is_valid_label(self, trained_lgbm, full_split):
        pred = trained_lgbm.predict(full_split.test_analyses[0])
        assert pred.complexity in ComplexityLabel

    def test_confidence_in_range(self, trained_lgbm, full_split):
        for analysis in full_split.test_analyses:
            pred = trained_lgbm.predict(analysis)
            assert 0.0 <= pred.confidence <= 1.0

    def test_probabilities_sum_to_one(self, trained_lgbm, full_split):
        pred = trained_lgbm.predict(full_split.test_analyses[0])
        total = sum(pred.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_probabilities_has_all_classes(self, trained_lgbm, full_split):
        pred = trained_lgbm.predict(full_split.test_analyses[0])
        for lbl in ComplexityLabel:
            assert lbl.value in pred.probabilities

    def test_model_id_in_prediction(self, trained_lgbm, full_split):
        pred = trained_lgbm.predict(full_split.test_analyses[0])
        assert pred.model_id == "lightgbm_v1"


# ── 9. Evaluation output structure ───────────────────────────────────────────

class TestEvaluationOutput:

    def test_lr_evaluation_returns_result(self, trained_lr, full_split):
        result = trained_lr.evaluate(full_split.test_analyses, full_split.test_labels)
        assert isinstance(result, EvaluationResult)

    def test_lgbm_evaluation_returns_result(self, trained_lgbm, full_split):
        result = trained_lgbm.evaluate(full_split.test_analyses, full_split.test_labels)
        assert isinstance(result, EvaluationResult)

    def test_accuracy_in_range(self, trained_lr, full_split):
        result = trained_lr.evaluate(full_split.test_analyses, full_split.test_labels)
        assert 0.0 <= result.accuracy <= 1.0

    def test_f1_in_range(self, trained_lr, full_split):
        result = trained_lr.evaluate(full_split.test_analyses, full_split.test_labels)
        assert 0.0 <= result.f1_macro <= 1.0

    def test_confusion_matrix_shape(self, trained_lr, full_split):
        result = trained_lr.evaluate(full_split.test_analyses, full_split.test_labels)
        cm = result.confusion_matrix
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_confusion_matrix_sums_to_n_test(self, trained_lr, full_split):
        result = trained_lr.evaluate(full_split.test_analyses, full_split.test_labels)
        total = sum(result.confusion_matrix[i][j] for i in range(3) for j in range(3))
        assert total == full_split.n_test

    def test_n_test_samples_is_correct(self, trained_lr, full_split):
        result = trained_lr.evaluate(full_split.test_analyses, full_split.test_labels)
        assert result.n_test_samples == full_split.n_test

    def test_classification_report_is_string(self, trained_lr, full_split):
        result = trained_lr.evaluate(full_split.test_analyses, full_split.test_labels)
        assert isinstance(result.classification_report, str)
        assert "complex" in result.classification_report

    def test_empty_eval_set_raises(self, trained_lr):
        with pytest.raises(ValueError, match="empty"):
            trained_lr.evaluate([], [])

    def test_summary_is_string(self, trained_lr, full_split):
        result = trained_lr.evaluate(full_split.test_analyses, full_split.test_labels)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "logistic_regression" in summary


# ── 10. Deterministic / reproducible behavior ─────────────────────────────────

class TestDeterminism:

    def test_lr_same_data_same_prediction(self):
        """Two separately trained LR models on identical data yield identical predictions."""
        analyses, labels = _minimal_dataset()
        clf1 = LogisticRegressionClassifier()
        clf1.train(analyses, labels)
        clf2 = LogisticRegressionClassifier()
        clf2.train(analyses, labels)

        test_analysis = _make_analysis("What is 2 + 2?")
        p1 = clf1.predict(test_analysis)
        p2 = clf2.predict(test_analysis)
        assert p1.complexity == p2.complexity
        assert abs(p1.confidence - p2.confidence) < 1e-10

    def test_lgbm_same_data_same_prediction(self):
        """Two separately trained LightGBM models on identical data yield identical predictions."""
        analyses, labels = _minimal_dataset()
        clf1 = LightGBMClassifier()
        clf1.train(analyses, labels)
        clf2 = LightGBMClassifier()
        clf2.train(analyses, labels)

        test_analysis = _make_analysis("Implement a distributed rate limiter.")
        p1 = clf1.predict(test_analysis)
        p2 = clf2.predict(test_analysis)
        assert p1.complexity == p2.complexity

    def test_dataset_split_is_reproducible(self, full_samples):
        split1 = make_dataset_split(full_samples, random_state=42)
        split2 = make_dataset_split(full_samples, random_state=42)
        assert split1.n_train == split2.n_train
        assert [l.value for l in split1.train_labels] == [l.value for l in split2.train_labels]

    def test_repeated_predictions_are_identical(self, trained_lr, full_split):
        analysis = full_split.test_analyses[0]
        p1 = trained_lr.predict(analysis)
        p2 = trained_lr.predict(analysis)
        assert p1.complexity == p2.complexity
        assert abs(p1.confidence - p2.confidence) < 1e-12


# ── 11. Model persistence ─────────────────────────────────────────────────────

class TestModelPersistence:

    def test_lr_save_and_load(self, trained_lr, full_split, tmp_path):
        """Saved and reloaded LR produces the same prediction as the original."""
        save_path = tmp_path / "lr_model.pkl"
        trained_lr.save(save_path)
        assert save_path.exists()

        loaded = LogisticRegressionClassifier()
        loaded.load(save_path)

        analysis = full_split.test_analyses[0]
        p_orig = trained_lr.predict(analysis)
        p_loaded = loaded.predict(analysis)
        assert p_orig.complexity == p_loaded.complexity
        assert abs(p_orig.confidence - p_loaded.confidence) < 1e-10

    def test_lgbm_save_and_load(self, trained_lgbm, full_split, tmp_path):
        """Saved and reloaded LightGBM produces the same prediction as the original."""
        save_path = tmp_path / "lgbm_model.pkl"
        trained_lgbm.save(save_path)
        assert save_path.exists()

        loaded = LightGBMClassifier()
        loaded.load(save_path)

        analysis = full_split.test_analyses[0]
        p_orig = trained_lgbm.predict(analysis)
        p_loaded = loaded.predict(analysis)
        assert p_orig.complexity == p_loaded.complexity

    def test_save_before_training_raises(self, tmp_path):
        clf = LogisticRegressionClassifier()
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.save(tmp_path / "never.pkl")

    def test_load_creates_parent_dir(self, trained_lr, tmp_path):
        nested = tmp_path / "models" / "subdir" / "lr.pkl"
        trained_lr.save(nested)
        assert nested.exists()


# ── 12. ComplexityPrediction validation ───────────────────────────────────────

class TestComplexityPredictionValidation:

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            ComplexityPrediction(
                complexity=ComplexityLabel.SIMPLE,
                confidence=1.5,
                probabilities={"simple": 1.0, "moderate": 0.0, "complex": 0.0},
                model_id="test",
            )

    def test_missing_probability_key_raises(self):
        with pytest.raises(ValueError, match="missing key"):
            ComplexityPrediction(
                complexity=ComplexityLabel.SIMPLE,
                confidence=0.9,
                probabilities={"simple": 0.9, "moderate": 0.1},  # missing 'complex'
                model_id="test",
            )


# ── 13. Empty / minimal prompt handling ──────────────────────────────────────

class TestMinimalInput:

    def test_empty_prompt_does_not_crash_lr(self, trained_lr):
        analysis = _make_analysis("")
        pred = trained_lr.predict(analysis)
        assert isinstance(pred, ComplexityPrediction)
        assert pred.complexity in ComplexityLabel

    def test_empty_prompt_does_not_crash_lgbm(self, trained_lgbm):
        analysis = _make_analysis("")
        pred = trained_lgbm.predict(analysis)
        assert isinstance(pred, ComplexityPrediction)
        assert pred.complexity in ComplexityLabel

    def test_single_word_prompt_lr(self, trained_lr):
        analysis = _make_analysis("hello")
        pred = trained_lr.predict(analysis)
        assert pred.complexity in ComplexityLabel

    def test_single_word_prompt_lgbm(self, trained_lgbm):
        analysis = _make_analysis("hello")
        pred = trained_lgbm.predict(analysis)
        assert pred.complexity in ComplexityLabel


# ── 14. Both models expose same interface ─────────────────────────────────────

class TestSharedInterface:

    def test_both_models_return_same_output_type(self, trained_lr, trained_lgbm, full_split):
        analysis = full_split.test_analyses[0]
        p_lr = trained_lr.predict(analysis)
        p_lgb = trained_lgbm.predict(analysis)
        assert type(p_lr) is type(p_lgb)

    def test_both_models_have_all_probability_keys(self, trained_lr, trained_lgbm, full_split):
        analysis = full_split.test_analyses[0]
        for clf in [trained_lr, trained_lgbm]:
            pred = clf.predict(analysis)
            for lbl in ComplexityLabel:
                assert lbl.value in pred.probabilities

    def test_both_evaluations_produce_evaluation_result(self, trained_lr, trained_lgbm, full_split):
        for clf in [trained_lr, trained_lgbm]:
            result = clf.evaluate(full_split.test_analyses, full_split.test_labels)
            assert isinstance(result, EvaluationResult)
