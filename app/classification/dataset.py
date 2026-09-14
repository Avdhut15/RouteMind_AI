"""
app/classification/dataset.py
──────────────────────────────
Dataset loading and train/test splitting for complexity classification.

This module is the single source of truth for:
    - Loading labeled complexity samples from complexity_dataset.json
    - Converting raw prompts → RequestAnalysis (via RequestAnalyzer)
    - Producing train/test splits with no leakage

IMPORTANT DESIGN DECISIONS:
    - The 30-prompt benchmark dataset (benchmarks/prompts.json) is NOT used
      for training or evaluation to prevent train/test leakage.
    - The complexity_dataset.json file is intentionally separate from
      all benchmark data.
    - random_state=42 makes splits reproducible.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Optional

from sklearn.model_selection import train_test_split

from app.classification.complexity_classifier import ComplexityLabel
from app.providers.base import LLMRequest
from app.routing.analyzer import RequestAnalysis, RequestAnalyzer

# Path to the labeled dataset relative to the project root
_DATASET_PATH = pathlib.Path("app/classification/complexity_dataset.json")

# Default train/test split ratio
_DEFAULT_TEST_SIZE = 0.30

# Default random state for reproducibility
_DEFAULT_RANDOM_STATE = 42


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class LabeledSample:
    """One labeled complexity training/evaluation sample."""
    prompt: str
    label: ComplexityLabel
    analysis: RequestAnalysis   # Pre-computed feature extraction


@dataclass
class DatasetSplit:
    """
    Stratified train/test split of labeled samples.

    All downstream code should consume this rather than loading data directly
    to ensure consistent splitting.
    """
    train_analyses: list[RequestAnalysis]
    train_labels: list[ComplexityLabel]
    test_analyses: list[RequestAnalysis]
    test_labels: list[ComplexityLabel]

    @property
    def n_train(self) -> int:
        return len(self.train_labels)

    @property
    def n_test(self) -> int:
        return len(self.test_labels)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_test


# ── Valid label set ───────────────────────────────────────────────────────────

_VALID_LABELS = frozenset(lbl.value for lbl in ComplexityLabel)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_labeled_samples(
    dataset_path: pathlib.Path = _DATASET_PATH,
    analyzer: Optional[RequestAnalyzer] = None,
) -> list[LabeledSample]:
    """
    Load and validate the complexity dataset.

    Each raw prompt is converted into a RequestAnalysis using the provided
    (or default) RequestAnalyzer.

    Args:
        dataset_path: Path to complexity_dataset.json.
        analyzer:     RequestAnalyzer instance. Creates one if not provided.

    Returns:
        List of LabeledSample objects, one per dataset entry.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If the JSON structure is invalid or any label is unknown.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Complexity dataset not found at '{dataset_path}'. "
            "Ensure app/classification/complexity_dataset.json exists."
        )

    raw = dataset_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in complexity dataset: {exc}") from exc

    samples_raw = data.get("samples")
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ValueError(
            "complexity_dataset.json must contain a non-empty 'samples' list."
        )

    # Validate all labels before processing
    for i, entry in enumerate(samples_raw):
        label_str = entry.get("label", "")
        if label_str not in _VALID_LABELS:
            raise ValueError(
                f"Sample #{i} has invalid label '{label_str}'. "
                f"Valid labels: {sorted(_VALID_LABELS)}"
            )

    if analyzer is None:
        analyzer = RequestAnalyzer()

    labeled: list[LabeledSample] = []
    for entry in samples_raw:
        prompt = entry["prompt"]
        label = ComplexityLabel(entry["label"])
        # Build a minimal LLMRequest (model_id is irrelevant for analysis)
        request = LLMRequest(prompt=prompt, model_id="__dataset_loader__")
        analysis = analyzer.analyze(request)
        labeled.append(LabeledSample(prompt=prompt, label=label, analysis=analysis))

    return labeled


def make_dataset_split(
    samples: list[LabeledSample],
    test_size: float = _DEFAULT_TEST_SIZE,
    random_state: int = _DEFAULT_RANDOM_STATE,
) -> DatasetSplit:
    """
    Create a stratified train/test split from labeled samples.

    Stratification ensures each complexity class is proportionally represented
    in both the train and test sets.

    Args:
        samples:      List of LabeledSample (from load_labeled_samples).
        test_size:    Fraction of data for the test set (default 0.30).
        random_state: Seed for reproducibility (default 42).

    Returns:
        DatasetSplit with separate train/test analyses and labels.
    """
    if not samples:
        raise ValueError("Cannot split an empty sample list.")

    analyses = [s.analysis for s in samples]
    labels   = [s.label    for s in samples]
    label_strs = [lbl.value for lbl in labels]

    (
        train_analyses, test_analyses,
        train_label_strs, test_label_strs,
    ) = train_test_split(
        analyses, label_strs,
        test_size=test_size,
        random_state=random_state,
        stratify=label_strs,
    )

    return DatasetSplit(
        train_analyses=train_analyses,
        train_labels=[ComplexityLabel(l) for l in train_label_strs],
        test_analyses=test_analyses,
        test_labels=[ComplexityLabel(l) for l in test_label_strs],
    )
