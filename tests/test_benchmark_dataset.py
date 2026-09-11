"""
tests/test_benchmark_dataset.py
─────────────────────────────────
Unit tests validating the structure and integrity of benchmarks/prompts.json.

Audit findings (before writing tests):
    - benchmarks/prompts.json is valid JSON.
    - Total prompts: 30.
    - Distribution: 10 tier_1, 10 tier_2, 10 tier_3.
    - All prompts have required fields: id, tier, task_type, prompt.
    - No duplicate IDs.
    - No duplicate prompt texts.
    - Tier values are consistently: 'tier_1', 'tier_2', 'tier_3'.
    - No changes to the dataset were required.

NOTE: These tests do not call any LLM APIs, run inference,
      or implement any routing or classification logic.
"""

import json
import pathlib
from collections import Counter
from typing import Any

import pytest

PROMPTS_PATH = pathlib.Path("benchmarks/prompts.json")

# Required field set for every prompt entry.
REQUIRED_FIELDS = {"id", "tier", "task_type", "prompt"}

# The three valid tier names.
VALID_TIERS = {"tier_1", "tier_2", "tier_3"}

# Expected count per tier.
EXPECTED_TIER_COUNT = 10

# Expected total.
EXPECTED_TOTAL = 30


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset() -> dict[str, Any]:
    """Load benchmarks/prompts.json once for the entire module."""
    assert PROMPTS_PATH.exists(), (
        f"benchmarks/prompts.json not found at '{PROMPTS_PATH.resolve()}'"
    )
    raw = PROMPTS_PATH.read_text(encoding="utf-8")
    return json.loads(raw)


@pytest.fixture(scope="module")
def prompts(dataset) -> list[dict[str, Any]]:
    """Return the list of prompt entries."""
    assert "prompts" in dataset, "Top-level 'prompts' key is missing from dataset."
    return dataset["prompts"]


# ── JSON validity ─────────────────────────────────────────────────────────────

class TestDatasetJSONValidity:

    def test_file_exists(self):
        """benchmarks/prompts.json exists on disk."""
        assert PROMPTS_PATH.exists()

    def test_file_is_valid_json(self):
        """benchmarks/prompts.json parses as valid JSON without raising."""
        raw = PROMPTS_PATH.read_text(encoding="utf-8")
        parsed = json.loads(raw)  # Raises json.JSONDecodeError if invalid
        assert isinstance(parsed, dict)

    def test_top_level_has_prompts_key(self, dataset):
        """Parsed JSON has a top-level 'prompts' list."""
        assert "prompts" in dataset
        assert isinstance(dataset["prompts"], list)

    def test_top_level_has_meta_key(self, dataset):
        """Parsed JSON has a top-level '_meta' key."""
        assert "_meta" in dataset


# ── Count and distribution ────────────────────────────────────────────────────

class TestDatasetDistribution:

    def test_total_prompt_count_is_30(self, prompts):
        """Dataset contains exactly 30 prompts."""
        assert len(prompts) == EXPECTED_TOTAL

    def test_tier_1_count_is_10(self, prompts):
        """Exactly 10 prompts are labelled tier_1 (simple)."""
        count = sum(1 for p in prompts if p.get("tier") == "tier_1")
        assert count == EXPECTED_TIER_COUNT

    def test_tier_2_count_is_10(self, prompts):
        """Exactly 10 prompts are labelled tier_2 (moderate)."""
        count = sum(1 for p in prompts if p.get("tier") == "tier_2")
        assert count == EXPECTED_TIER_COUNT

    def test_tier_3_count_is_10(self, prompts):
        """Exactly 10 prompts are labelled tier_3 (complex)."""
        count = sum(1 for p in prompts if p.get("tier") == "tier_3")
        assert count == EXPECTED_TIER_COUNT

    def test_no_unlabelled_prompts(self, prompts):
        """Every prompt has a tier value in the valid set."""
        unlabelled = [p.get("id") for p in prompts if p.get("tier") not in VALID_TIERS]
        assert unlabelled == [], f"Prompts with invalid tier: {unlabelled}"


# ── Schema / structure ────────────────────────────────────────────────────────

class TestDatasetSchema:

    def test_every_prompt_has_required_fields(self, prompts):
        """Each prompt entry contains: id, tier, task_type, prompt."""
        violations = []
        for p in prompts:
            missing = REQUIRED_FIELDS - set(p.keys())
            if missing:
                violations.append((p.get("id", "?"), sorted(missing)))
        assert violations == [], f"Prompts with missing fields: {violations}"

    def test_id_field_is_string(self, prompts):
        """All 'id' values are strings."""
        non_strings = [p.get("id") for p in prompts if not isinstance(p.get("id"), str)]
        assert non_strings == []

    def test_tier_field_is_string(self, prompts):
        """All 'tier' values are strings."""
        non_strings = [p.get("id") for p in prompts if not isinstance(p.get("tier"), str)]
        assert non_strings == []

    def test_task_type_field_is_string(self, prompts):
        """All 'task_type' values are strings."""
        non_strings = [p.get("id") for p in prompts if not isinstance(p.get("task_type"), str)]
        assert non_strings == []

    def test_prompt_field_is_non_empty_string(self, prompts):
        """All 'prompt' values are non-empty strings."""
        invalid = [
            p.get("id") for p in prompts
            if not isinstance(p.get("prompt"), str) or len(p["prompt"].strip()) == 0
        ]
        assert invalid == [], f"Prompts with empty/missing prompt text: {invalid}"

    def test_tier_values_are_valid(self, prompts):
        """All tier values are one of: tier_1, tier_2, tier_3."""
        invalid = [
            (p.get("id"), p.get("tier")) for p in prompts
            if p.get("tier") not in VALID_TIERS
        ]
        assert invalid == []

    def test_prompt_text_is_meaningful_length(self, prompts):
        """Each prompt text is at least 10 characters long."""
        too_short = [
            p.get("id") for p in prompts
            if isinstance(p.get("prompt"), str) and len(p["prompt"].strip()) < 10
        ]
        assert too_short == [], f"Suspiciously short prompts: {too_short}"


# ── ID and prompt uniqueness ──────────────────────────────────────────────────

class TestDatasetUniqueness:

    def test_no_duplicate_ids(self, prompts):
        """All prompt IDs are unique."""
        ids = [p.get("id") for p in prompts]
        duplicates = [i for i, count in Counter(ids).items() if count > 1]
        assert duplicates == [], f"Duplicate IDs found: {duplicates}"

    def test_no_duplicate_prompt_texts(self, prompts):
        """All prompt text values are unique (no copy-paste duplicates)."""
        texts = [p.get("prompt") for p in prompts if isinstance(p.get("prompt"), str)]
        duplicates = [t[:60] for t, count in Counter(texts).items() if count > 1]
        assert duplicates == [], f"Duplicate prompt texts (first 60 chars): {duplicates}"


# ── Complexity differentiation ────────────────────────────────────────────────

class TestDatasetComplexityDifferentiation:
    """
    Lightweight heuristic checks that the three tiers are meaningfully differentiated.
    These checks do not call any LLM API — they inspect observable text properties.
    """

    def test_tier_1_prompts_are_shorter_on_average_than_tier_3(self, prompts):
        """
        Simple prompts should be meaningfully shorter than complex prompts on average.
        This is a heuristic sanity check, not a hard requirement.
        """
        tier_1 = [p for p in prompts if p.get("tier") == "tier_1"]
        tier_3 = [p for p in prompts if p.get("tier") == "tier_3"]

        avg_t1 = sum(len(p["prompt"]) for p in tier_1) / len(tier_1)
        avg_t3 = sum(len(p["prompt"]) for p in tier_3) / len(tier_3)

        assert avg_t1 < avg_t3, (
            f"Expected tier_1 avg length ({avg_t1:.0f}) < tier_3 avg length ({avg_t3:.0f})"
        )

    def test_task_type_variety(self, prompts):
        """Dataset spans multiple distinct task types (not all the same)."""
        task_types = set(p.get("task_type") for p in prompts)
        assert len(task_types) >= 5, (
            f"Expected at least 5 distinct task types, found: {sorted(task_types)}"
        )

    def test_tier_3_contains_reasoning_or_coding(self, prompts):
        """Complex tier includes reasoning, coding, or architecture tasks."""
        tier_3_tasks = {p.get("task_type") for p in prompts if p.get("tier") == "tier_3"}
        demanding_types = {"reasoning", "coding", "architecture", "analysis"}
        overlap = tier_3_tasks & demanding_types
        assert overlap, (
            f"tier_3 should include demanding task types. Found: {sorted(tier_3_tasks)}"
        )

    def test_tier_1_does_not_contain_complex_task_types(self, prompts):
        """Simple tier should not contain reasoning, coding, or architecture tasks."""
        tier_1_tasks = {p.get("task_type") for p in prompts if p.get("tier") == "tier_1"}
        complex_types = {"reasoning", "coding", "architecture"}
        overlap = tier_1_tasks & complex_types
        assert not overlap, (
            f"tier_1 contains unexpectedly complex task types: {overlap}"
        )
