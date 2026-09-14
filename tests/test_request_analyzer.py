"""
tests/test_request_analyzer.py
─────────────────────────────────
Unit tests for Phase 2 Part 1 — Request Analysis & Feature Extraction.

All tests are:
    - Deterministic (no randomness)
    - Offline (no API calls, no network)
    - Independent of OpenRouter, Ollama, or any real model

Coverage:
    1.  General question / QA
    2.  Summarization request
    3.  Code generation request
    4.  Code debugging request
    5.  Reasoning-heavy request
    6.  Structured JSON output request
    7.  Context-heavy request (explicit signal)
    8.  Context-heavy request (long prompt threshold)
    9.  Short-response request
    10. Long-response request
    11. Empty / minimal prompt
    12. Determinism: same input always produces the same output
    13. Correct feature values for representative prompts
    14. Feature vector has no string values (ML-ready)
    15. task_type uses controlled TaskType enum
    16. RequestAnalysis.request_id mirrors LLMRequest.request_id
    17. has_system_prompt is True when system_prompt is supplied
    18. has_system_prompt is False when system_prompt is absent
    19. max_tokens mirrors LLMRequest.max_tokens
    20. approx_token_count is > 0 for non-empty prompt
    21. Translation detection
    22. Data-analysis detection
    23. Creative-writing detection
    24. Classification detection
    25. requires_code is True for code generation request
    26. requires_code is True for debugging request
    27. requires_structured_output is True for JSON request
    28. requires_reasoning is True for step-by-step request
    29. expected_response_length is SHORT when max_tokens <= 150
    30. expected_response_length is LONG when max_tokens >= 600
"""

import pytest

from app.providers.base import LLMRequest
from app.routing.analyzer import (
    RequestAnalysis,
    RequestAnalyzer,
    ResponseLengthCategory,
    TaskType,
)


# ── Shared analyzer instance ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def analyzer() -> RequestAnalyzer:
    return RequestAnalyzer()


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_request(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 1024,
    model_id: str = "openai/gpt-oss-20b:free",
) -> LLMRequest:
    return LLMRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        model_id=model_id,
    )


# ── 1. General question / QA ─────────────────────────────────────────────────

class TestGeneralQA:

    def test_task_type_is_general_qa(self, analyzer):
        req = _make_request("What is the capital of France?")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.GENERAL_QA

    def test_question_mark_triggers_general_qa(self, analyzer):
        req = _make_request("How does photosynthesis work?")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.GENERAL_QA

    def test_wh_question_without_mark(self, analyzer):
        req = _make_request("What is machine learning")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.GENERAL_QA


# ── 2. Summarization ──────────────────────────────────────────────────────────

class TestSummarization:

    def test_summarize_keyword(self, analyzer):
        req = _make_request("Summarize the following article in bullet points.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.SUMMARIZATION

    def test_summary_keyword(self, analyzer):
        req = _make_request("Write a summary of the document below.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.SUMMARIZATION

    def test_tldr_keyword(self, analyzer):
        req = _make_request("tldr: The future of renewable energy in Europe")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.SUMMARIZATION


# ── 3. Code generation ────────────────────────────────────────────────────────

class TestCodeGeneration:

    def test_write_function_keyword(self, analyzer):
        req = _make_request(
            "Write a function that sorts a list of integers using merge sort."
        )
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CODE_GENERATION

    def test_implement_keyword(self, analyzer):
        req = _make_request(
            "Implement a binary search tree with insert, delete, and search methods."
        )
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CODE_GENERATION

    def test_requires_code_is_true(self, analyzer):
        req = _make_request("Write a function to reverse a string in Python.")
        result = analyzer.analyze(req)
        assert result.requires_code is True


# ── 4. Code debugging ─────────────────────────────────────────────────────────

class TestCodeDebugging:

    def test_debug_keyword(self, analyzer):
        req = _make_request(
            "Debug this Python code: def foo(): return 1/0"
        )
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CODE_DEBUGGING

    def test_fix_this_keyword(self, analyzer):
        req = _make_request("Fix this code — it throws an AttributeError.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CODE_DEBUGGING

    def test_debugging_requires_code(self, analyzer):
        req = _make_request("Why does this code raise a KeyError? `d['missing']`")
        result = analyzer.analyze(req)
        assert result.requires_code is True

    def test_debugging_prioritized_over_generation(self, analyzer):
        """Debugging signals should beat code-generation signals when both present."""
        req = _make_request(
            "Write a function to fix this bug in my implementation."
        )
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CODE_DEBUGGING


# ── 5. Reasoning-heavy request ────────────────────────────────────────────────

class TestReasoning:

    def test_step_by_step_reasoning(self, analyzer):
        req = _make_request(
            "Explain step by step how to prove the Pythagorean theorem."
        )
        result = analyzer.analyze(req)
        assert result.requires_reasoning is True

    def test_analyze_keyword(self, analyzer):
        req = _make_request("Analyze the trade-offs between SQL and NoSQL databases.")
        result = analyzer.analyze(req)
        assert result.requires_reasoning is True

    def test_prove_keyword(self, analyzer):
        req = _make_request("Prove that the square root of 2 is irrational.")
        result = analyzer.analyze(req)
        assert result.requires_reasoning is True
        assert result.task_type == TaskType.REASONING

    def test_compare_contrast_keyword(self, analyzer):
        req = _make_request(
            "Compare and contrast supervised and unsupervised learning."
        )
        result = analyzer.analyze(req)
        assert result.requires_reasoning is True


# ── 6. Structured output / JSON ───────────────────────────────────────────────

class TestStructuredOutput:

    def test_json_keyword(self, analyzer):
        req = _make_request(
            "Return a JSON object with keys: name, age, and email."
        )
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.STRUCTURED_GENERATION
        assert result.requires_structured_output is True

    def test_table_keyword(self, analyzer):
        req = _make_request(
            "Present the results as a markdown table with three columns."
        )
        result = analyzer.analyze(req)
        assert result.requires_structured_output is True

    def test_schema_keyword(self, analyzer):
        req = _make_request(
            "Generate a JSON schema for a user profile object."
        )
        result = analyzer.analyze(req)
        assert result.requires_structured_output is True

    def test_yaml_keyword(self, analyzer):
        req = _make_request("Output the configuration in YAML format.")
        result = analyzer.analyze(req)
        assert result.requires_structured_output is True


# ── 7. Context-heavy (explicit signal) ───────────────────────────────────────

class TestContextHeavyExplicit:

    def test_based_on_following_signal(self, analyzer):
        req = _make_request(
            "Based on the following passage, answer the question: ..."
        )
        result = analyzer.analyze(req)
        assert result.requires_context is True

    def test_given_the_text_signal(self, analyzer):
        req = _make_request(
            "Given the text below, identify the main argument."
        )
        result = analyzer.analyze(req)
        assert result.requires_context is True

    def test_from_the_document_signal(self, analyzer):
        req = _make_request(
            "From the document, extract all dates mentioned."
        )
        result = analyzer.analyze(req)
        assert result.requires_context is True


# ── 8. Context-heavy (long prompt threshold) ──────────────────────────────────

class TestContextHeavyLongPrompt:

    def test_long_prompt_triggers_context_flag(self, analyzer):
        """Prompts longer than 800 chars should set requires_context=True."""
        long_prompt = "Tell me about machine learning. " * 30  # ~930 chars
        req = _make_request(long_prompt)
        result = analyzer.analyze(req)
        assert result.requires_context is True
        assert result.prompt_length > 800

    def test_short_prompt_no_context_flag(self, analyzer):
        """Short prompts without explicit signals should not flag requires_context."""
        req = _make_request("What is 2 + 2?")
        result = analyzer.analyze(req)
        assert result.requires_context is False


# ── 9. Short-response expectation ────────────────────────────────────────────

class TestShortResponseExpectation:

    def test_in_one_sentence_signal(self, analyzer):
        req = _make_request(
            "Explain gravity in one sentence.", max_tokens=1024
        )
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.SHORT

    def test_yes_or_no_signal(self, analyzer):
        req = _make_request("Is Python a compiled language? Yes or no.")
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.SHORT

    def test_max_tokens_150_or_less_is_short(self, analyzer):
        req = _make_request("Describe a tree.", max_tokens=100)
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.SHORT


# ── 10. Long-response expectation ────────────────────────────────────────────

class TestLongResponseExpectation:

    def test_detailed_signal(self, analyzer):
        req = _make_request(
            "Write a detailed analysis of the causes of World War I."
        )
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.LONG

    def test_comprehensive_signal(self, analyzer):
        req = _make_request(
            "Provide a comprehensive overview of machine learning algorithms."
        )
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.LONG

    def test_max_tokens_600_or_more_is_long(self, analyzer):
        req = _make_request("Tell me about Python.", max_tokens=2048)
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.LONG


# ── 11. Empty / minimal prompt ────────────────────────────────────────────────

class TestEmptyMinimalPrompt:

    def test_empty_prompt_does_not_crash(self, analyzer):
        req = _make_request("")
        result = analyzer.analyze(req)
        assert isinstance(result, RequestAnalysis)

    def test_empty_prompt_word_count_is_zero(self, analyzer):
        req = _make_request("")
        result = analyzer.analyze(req)
        assert result.word_count == 0

    def test_empty_prompt_length_is_zero(self, analyzer):
        req = _make_request("")
        result = analyzer.analyze(req)
        assert result.prompt_length == 0

    def test_empty_prompt_token_count_is_zero(self, analyzer):
        req = _make_request("")
        result = analyzer.analyze(req)
        assert result.approx_token_count == 0

    def test_whitespace_only_prompt_word_count_is_zero(self, analyzer):
        req = _make_request("   ")
        result = analyzer.analyze(req)
        assert result.word_count == 0

    def test_single_word_prompt(self, analyzer):
        req = _make_request("hello")
        result = analyzer.analyze(req)
        assert result.word_count == 1
        assert result.prompt_length == 5
        assert result.approx_token_count >= 1


# ── 12. Determinism ───────────────────────────────────────────────────────────

class TestDeterminism:

    def test_identical_inputs_produce_identical_outputs(self, analyzer):
        """Same prompt must always produce the same analysis (except request_id)."""
        prompt = "Summarize the quarterly earnings report in bullet points."
        req1 = LLMRequest(prompt=prompt, model_id="test-model")
        req2 = LLMRequest(prompt=prompt, model_id="test-model")

        r1 = analyzer.analyze(req1)
        r2 = analyzer.analyze(req2)

        # All feature fields must match (request_id differs — it is auto-generated)
        assert r1.prompt_length           == r2.prompt_length
        assert r1.word_count              == r2.word_count
        assert r1.approx_token_count      == r2.approx_token_count
        assert r1.task_type               == r2.task_type
        assert r1.requires_reasoning      == r2.requires_reasoning
        assert r1.requires_code           == r2.requires_code
        assert r1.requires_structured_output == r2.requires_structured_output
        assert r1.requires_context        == r2.requires_context
        assert r1.expected_response_length == r2.expected_response_length

    def test_repeated_calls_same_instance_same_result(self, analyzer):
        req = _make_request("What is the speed of light?")
        r1 = analyzer.analyze(req)
        r2 = analyzer.analyze(req)
        assert r1.model_dump() == r2.model_dump()

    def test_different_analyzer_instances_same_result(self):
        """Statelessness: two separate instances yield identical output."""
        req = _make_request("Implement a quicksort algorithm in Python.")
        r1 = RequestAnalyzer().analyze(req)
        r2 = RequestAnalyzer().analyze(req)
        assert r1.task_type == r2.task_type
        assert r1.requires_code == r2.requires_code


# ── 13. Correct feature values for representative prompts ────────────────────

class TestFeatureValues:

    def test_prompt_length_is_correct(self, analyzer):
        prompt = "Hello world"
        req = _make_request(prompt)
        result = analyzer.analyze(req)
        assert result.prompt_length == len(prompt)

    def test_word_count_is_correct(self, analyzer):
        req = _make_request("one two three four five")
        result = analyzer.analyze(req)
        assert result.word_count == 5

    def test_approx_token_count_reasonable(self, analyzer):
        """40 chars / 4 = 10 tokens."""
        prompt = "a" * 40
        req = _make_request(prompt)
        result = analyzer.analyze(req)
        assert result.approx_token_count == 10

    def test_request_id_mirrors_llm_request(self, analyzer):
        req = _make_request("What is entropy?")
        result = analyzer.analyze(req)
        assert result.request_id == req.request_id

    def test_max_tokens_mirrors_llm_request(self, analyzer):
        req = _make_request("Tell me something.", max_tokens=512)
        result = analyzer.analyze(req)
        assert result.max_tokens == 512


# ── 14. Feature vector — ML-ready ────────────────────────────────────────────

class TestFeatureVector:

    def test_feature_vector_has_no_string_values(self, analyzer):
        req = _make_request("Generate a Python function to parse JSON.")
        result = analyzer.analyze(req)
        fv = result.to_feature_vector()
        for key, val in fv.items():
            assert isinstance(val, (int, float)), (
                f"Feature '{key}' has non-numeric value: {val!r}"
            )

    def test_feature_vector_contains_expected_keys(self, analyzer):
        req = _make_request("Translate this text to Spanish.")
        result = analyzer.analyze(req)
        fv = result.to_feature_vector()
        required_keys = {
            "prompt_length", "word_count", "approx_token_count",
            "requires_reasoning", "requires_code", "requires_structured_output",
            "requires_context", "has_system_prompt", "max_tokens",
            "expected_response_length_short",
            "expected_response_length_medium",
            "expected_response_length_long",
        }
        assert required_keys.issubset(fv.keys())

    def test_feature_vector_task_one_hot(self, analyzer):
        """Exactly one task_* feature should be 1."""
        req = _make_request("Summarize the article.")
        result = analyzer.analyze(req)
        fv = result.to_feature_vector()
        task_values = [v for k, v in fv.items() if k.startswith("task_")]
        assert sum(task_values) == 1, (
            f"Expected exactly one active task feature, got: "
            f"{[k for k,v in fv.items() if k.startswith('task_') and v]}"
        )


# ── 15. TaskType enum usage ───────────────────────────────────────────────────

class TestTaskTypeEnum:

    def test_task_type_is_enum_instance(self, analyzer):
        req = _make_request("Translate 'hello' into French.")
        result = analyzer.analyze(req)
        assert isinstance(result.task_type, TaskType)

    def test_all_task_types_accessible(self):
        """All 11 documented task types exist in the enum."""
        expected = {
            "general_qa", "summarization", "code_generation", "code_debugging",
            "translation", "data_analysis", "reasoning", "creative_writing",
            "structured_generation", "classification", "unknown",
        }
        actual = {t.value for t in TaskType}
        assert expected == actual


# ── 16–18. system_prompt / request_id passthrough ────────────────────────────

class TestPassthroughFields:

    def test_has_system_prompt_true(self, analyzer):
        req = _make_request("Hello?", system_prompt="You are a helpful assistant.")
        result = analyzer.analyze(req)
        assert result.has_system_prompt is True

    def test_has_system_prompt_false_when_none(self, analyzer):
        req = _make_request("Hello?", system_prompt=None)
        result = analyzer.analyze(req)
        assert result.has_system_prompt is False

    def test_request_id_propagated(self, analyzer):
        req = _make_request("Explain recursion.")
        result = analyzer.analyze(req)
        assert result.request_id == req.request_id


# ── 19–20. max_tokens thresholds ─────────────────────────────────────────────

class TestMaxTokensThresholds:

    def test_max_tokens_150_short(self, analyzer):
        req = _make_request("Briefly explain sorting.", max_tokens=150)
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.SHORT

    def test_max_tokens_600_long(self, analyzer):
        req = _make_request("Explain sorting.", max_tokens=600)
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.LONG

    def test_max_tokens_300_medium(self, analyzer):
        req = _make_request("Explain sorting.", max_tokens=300)
        result = analyzer.analyze(req)
        assert result.expected_response_length == ResponseLengthCategory.MEDIUM


# ── 21. Translation ───────────────────────────────────────────────────────────

class TestTranslation:

    def test_translate_keyword(self, analyzer):
        req = _make_request("Translate the following paragraph into Spanish.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.TRANSLATION

    def test_to_french_phrase(self, analyzer):
        req = _make_request("How do you say 'good morning' in French?")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.TRANSLATION


# ── 22. Data analysis ─────────────────────────────────────────────────────────

class TestDataAnalysis:

    def test_data_analysis_keyword(self, analyzer):
        req = _make_request(
            "Perform a data analysis on the sales dataset and report the statistics."
        )
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.DATA_ANALYSIS

    def test_statistics_keyword(self, analyzer):
        req = _make_request("Compute the mean, median, and standard deviation.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.DATA_ANALYSIS


# ── 23. Creative writing ──────────────────────────────────────────────────────

class TestCreativeWriting:

    def test_write_a_story(self, analyzer):
        req = _make_request("Write a story about a dragon who learns to code.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CREATIVE_WRITING

    def test_write_a_poem(self, analyzer):
        req = _make_request("Write a poem about autumn leaves falling.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CREATIVE_WRITING


# ── 24. Classification ────────────────────────────────────────────────────────

class TestClassification:

    def test_classify_keyword(self, analyzer):
        req = _make_request("Classify the following text as positive, negative, or neutral.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CLASSIFICATION

    def test_categorize_keyword(self, analyzer):
        req = _make_request("Categorize these emails by topic.")
        result = analyzer.analyze(req)
        assert result.task_type == TaskType.CLASSIFICATION


# ── 25–28. Individual indicators ─────────────────────────────────────────────

class TestIndividualIndicators:

    def test_requires_code_for_generation(self, analyzer):
        req = _make_request("Implement a linked list in Python.")
        result = analyzer.analyze(req)
        assert result.requires_code is True

    def test_requires_code_for_debugging(self, analyzer):
        req = _make_request("Fix this bug in my Python function.")
        result = analyzer.analyze(req)
        assert result.requires_code is True

    def test_requires_structured_output_for_json(self, analyzer):
        req = _make_request("Return the answer as JSON with fields: name, age.")
        result = analyzer.analyze(req)
        assert result.requires_structured_output is True

    def test_requires_reasoning_for_step_by_step(self, analyzer):
        req = _make_request(
            "Walk me through the proof step by step."
        )
        result = analyzer.analyze(req)
        assert result.requires_reasoning is True

    def test_non_coding_prompt_does_not_require_code(self, analyzer):
        req = _make_request("What is the population of Japan?")
        result = analyzer.analyze(req)
        assert result.requires_code is False

    def test_simple_prompt_does_not_require_reasoning(self, analyzer):
        req = _make_request("What color is the sky?")
        result = analyzer.analyze(req)
        assert result.requires_reasoning is False
