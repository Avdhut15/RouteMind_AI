"""
app/routing/analyzer.py
────────────────────────
Phase 2 Part 1 — Request Analysis & Feature Extraction.

Accepts an LLMRequest and returns a structured RequestAnalysis containing
routing-relevant features extracted deterministically, without calling any
external API, LLM, or network resource.

Design contract:
    - Deterministic: same input always yields the same output.
    - No external calls: no OpenRouter, no Ollama, no tokenizer downloads.
    - Consumed by: Complexity Classifier (Part 2), Candidate Selection (Part 3),
      Routing Score (Part 4).
    - Does NOT classify complexity; that is Part 2's responsibility.

Token approximation strategy:
    A well-established rule of thumb for English text is ~4 characters per
    token (based on GPT tokenizer averages).  This is intentionally a *local,
    deterministic approximation* — exact tokenizer accuracy is not required at
    the feature-extraction stage.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.providers.base import LLMRequest


# ── Controlled task-type vocabulary ──────────────────────────────────────────

class TaskType(str, Enum):
    """
    Controlled vocabulary for detected task types.

    Phase 2 Part 2 (classifier) and Part 3 (candidate selection) rely on
    these labels — never use raw strings for task classification.
    """
    GENERAL_QA          = "general_qa"
    SUMMARIZATION        = "summarization"
    CODE_GENERATION      = "code_generation"
    CODE_DEBUGGING       = "code_debugging"
    TRANSLATION          = "translation"
    DATA_ANALYSIS        = "data_analysis"
    REASONING            = "reasoning"
    CREATIVE_WRITING     = "creative_writing"
    STRUCTURED_GENERATION = "structured_generation"
    CLASSIFICATION       = "classification"
    UNKNOWN              = "unknown"


class ResponseLengthCategory(str, Enum):
    """Expected output length bucket."""
    SHORT  = "short"
    MEDIUM = "medium"
    LONG   = "long"


# ── Analysis result schema ────────────────────────────────────────────────────

class RequestAnalysis(BaseModel):
    """
    Structured, routing-relevant feature set extracted from an LLMRequest.

    Consumed downstream by:
        - Complexity Classifier (Part 2)
        - Candidate Model Selection (Part 3)
        - Routing Score Calculator (Part 4)

    All fields are deterministically computed from the request — no ML,
    no external calls.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id: str = Field(
        description="Mirrors LLMRequest.request_id for tracing."
    )

    # ── Lexical features ──────────────────────────────────────────────────────
    prompt_length: int = Field(
        ge=0,
        description="Character count of the user prompt.",
    )
    word_count: int = Field(
        ge=0,
        description="Whitespace-delimited word count of the prompt.",
    )
    approx_token_count: int = Field(
        ge=0,
        description=(
            "Deterministic token count approximation using the ~4-chars/token "
            "rule of thumb. No external tokenizer."
        ),
    )

    # ── Task classification ───────────────────────────────────────────────────
    task_type: TaskType = Field(
        description="Most likely task type, determined by rule-based heuristics.",
    )

    # ── Capability requirement indicators (booleans) ──────────────────────────
    requires_reasoning: bool = Field(
        description=(
            "True when the prompt contains signals of multi-step or logical "
            "reasoning (e.g., 'prove', 'why', 'analyze', 'compare', 'logical')."
        ),
    )
    requires_code: bool = Field(
        description=(
            "True when the prompt is code-related (e.g., contains 'function', "
            "'bug', 'implement', code blocks, programming keywords)."
        ),
    )
    requires_structured_output: bool = Field(
        description=(
            "True when the prompt explicitly asks for structured formats such as "
            "JSON, XML, tables, schemas, or lists."
        ),
    )
    requires_context: bool = Field(
        description=(
            "True when the prompt depends on or references substantial context, "
            "e.g., 'based on the following', 'given the text', long documents."
        ),
    )

    # ── Response length expectation ───────────────────────────────────────────
    expected_response_length: ResponseLengthCategory = Field(
        description=(
            "Estimated output length: short (<~150 tokens), medium (150–600), "
            "long (>600).  Derived from prompt signals and max_tokens."
        ),
    )

    # ── Pass-through metadata from the original request ───────────────────────
    max_tokens: int = Field(
        description="max_tokens from the originating LLMRequest."
    )
    has_system_prompt: bool = Field(
        description="True when a system_prompt was provided."
    )

    def to_feature_vector(self) -> dict[str, Any]:
        """
        Return a flat dict of numeric/boolean features suitable for ML input.

        Strings and enums are mapped to integers so that Part 2 classifiers
        (Logistic Regression, LightGBM) can consume this without additional
        preprocessing.
        """
        return {
            "prompt_length": self.prompt_length,
            "word_count": self.word_count,
            "approx_token_count": self.approx_token_count,
            "requires_reasoning": int(self.requires_reasoning),
            "requires_code": int(self.requires_code),
            "requires_structured_output": int(self.requires_structured_output),
            "requires_context": int(self.requires_context),
            "expected_response_length_short": int(
                self.expected_response_length == ResponseLengthCategory.SHORT
            ),
            "expected_response_length_medium": int(
                self.expected_response_length == ResponseLengthCategory.MEDIUM
            ),
            "expected_response_length_long": int(
                self.expected_response_length == ResponseLengthCategory.LONG
            ),
            "has_system_prompt": int(self.has_system_prompt),
            "max_tokens": self.max_tokens,
            # One-hot task type columns
            **{
                f"task_{t.value}": int(self.task_type == t)
                for t in TaskType
            },
        }


# ── Heuristic keyword banks ───────────────────────────────────────────────────

# Each set contains lowercase tokens / substrings.
# Order of _TASK_RULES matters: first match wins.

_SUMMARIZATION_SIGNALS = frozenset({
    "summarize", "summarise", "summary", "tldr", "tl;dr",
    "brief overview", "key points", "main points", "recap",
    "condense", "shorten", "abstract",
})

_CODE_GENERATION_SIGNALS = frozenset({
    "write a function", "implement", "create a class", "write code",
    "write a script", "generate code", "write a program",
    "code that", "function that", "method that", "class that",
    "algorithm", "snippet",
})

_CODE_DEBUGGING_SIGNALS = frozenset({
    "fix this", "debug", "bug", "error in", "traceback",
    "exception", "why does this fail", "not working", "broken code",
    "syntax error", "runtime error", "fix the code", "what's wrong with",
    "what is wrong with",
})

_TRANSLATION_SIGNALS = frozenset({
    "translate", "translation", "in french", "in spanish", "in german",
    "in japanese", "in chinese", "in arabic", "in portuguese",
    "from english to", "to english",
})

_DATA_ANALYSIS_SIGNALS = frozenset({
    "analyze the data", "analyse the data", "data analysis",
    "statistics", "statistical", "distribution", "correlation",
    "regression", "dataset", "csv", "dataframe", "pandas",
    "mean", "median", "variance", "standard deviation",
})

_REASONING_SIGNALS = frozenset({
    "prove", "proof", "logical", "deduce", "infer", "inference",
    "reasoning", "reason why", "explain why", "because", "therefore",
    "if then", "if...then", "analyze", "analyse", "compare and contrast",
    "what would happen if", "step by step",
})

_CREATIVE_WRITING_SIGNALS = frozenset({
    "write a story", "write a poem", "write a song", "write an essay",
    "creative writing", "fiction", "narrative", "once upon a time",
    "write a blog", "write a letter", "character", "plot",
})

_STRUCTURED_OUTPUT_SIGNALS = frozenset({
    "json", "xml", "yaml", "schema", "in table", "as a table",
    "structured format", "output format", "return a list",
    "return json", "markdown table", "bullet points", "numbered list",
    "key-value", "csv format",
})

_CLASSIFICATION_SIGNALS = frozenset({
    "classify", "categorize", "categorise", "label", "tag",
    "which category", "what type of", "identify the type",
    "is this a", "determine if",
})

# Reasoning requirement (separate from task-type reasoning)
_REASONING_REQUIREMENT_SIGNALS = _REASONING_SIGNALS | frozenset({
    "step-by-step", "think through", "walk me through",
    "explain your reasoning", "justify", "argue", "counterargument",
    "evaluate", "assess", "critique",
})

# Code requirement (union of generation + debugging signals + misc)
_CODE_REQUIREMENT_SIGNALS = _CODE_GENERATION_SIGNALS | _CODE_DEBUGGING_SIGNALS | frozenset({
    "def ", "class ", "import ", "return ", "```python", "```js",
    "```javascript", "```typescript", "```java", "```c++", "```cpp",
    "```go", "```rust", "code", "programming", "developer", "compile",
    "syntax", "api endpoint", "function", "variable", "loop",
})

# Context requirement signals
_CONTEXT_REQUIREMENT_SIGNALS = frozenset({
    "based on the following", "given the following", "given the text",
    "the following text", "the following document", "using the context",
    "from the passage", "from the document", "the article",
    "the paper", "the report", "the transcript",
    "from the data below", "the provided", "above", "below",
})

# Short response signals
_SHORT_RESPONSE_SIGNALS = frozenset({
    "in one sentence", "one word", "yes or no", "briefly",
    "in a few words", "give me just", "short answer",
    "single word", "one line",
})

# Long response signals
_LONG_RESPONSE_SIGNALS = frozenset({
    "detailed", "comprehensive", "in depth", "in-depth", "thoroughly",
    "step by step", "step-by-step", "exhaustive", "full analysis",
    "complete guide", "write a full", "write an entire",
    "write a detailed", "elaborate", "long-form",
})

# Approximate chars per token for English text (GPT tokenizer average)
_CHARS_PER_TOKEN: float = 4.0

# Token thresholds for expected response length buckets
_SHORT_TOKEN_THRESHOLD  = 150
_LONG_TOKEN_THRESHOLD   = 600

# Prompt character length above which we flag context-dependency
_LONG_PROMPT_CHAR_THRESHOLD = 800


# ── Feature extraction helpers ────────────────────────────────────────────────

def _lower(text: str) -> str:
    return text.lower()


def _contains_any(text_lower: str, signals: frozenset[str]) -> bool:
    """Return True if any signal phrase is a substring of text_lower."""
    return any(sig in text_lower for sig in signals)


def _detect_task_type(prompt_lower: str) -> TaskType:
    """
    Determine the most likely task type using ordered heuristic matching.

    The first matching rule wins.  The ordering is designed so that more
    specific signals (debugging, code generation) are checked before broader
    ones (general_qa).
    """
    # Code debugging is checked before code generation (subset of code signals)
    if _contains_any(prompt_lower, _CODE_DEBUGGING_SIGNALS):
        return TaskType.CODE_DEBUGGING

    if _contains_any(prompt_lower, _CODE_GENERATION_SIGNALS):
        return TaskType.CODE_GENERATION

    if _contains_any(prompt_lower, _TRANSLATION_SIGNALS):
        return TaskType.TRANSLATION

    if _contains_any(prompt_lower, _SUMMARIZATION_SIGNALS):
        return TaskType.SUMMARIZATION

    if _contains_any(prompt_lower, _DATA_ANALYSIS_SIGNALS):
        return TaskType.DATA_ANALYSIS

    if _contains_any(prompt_lower, _STRUCTURED_OUTPUT_SIGNALS):
        return TaskType.STRUCTURED_GENERATION

    if _contains_any(prompt_lower, _CLASSIFICATION_SIGNALS):
        return TaskType.CLASSIFICATION

    if _contains_any(prompt_lower, _REASONING_SIGNALS):
        return TaskType.REASONING

    if _contains_any(prompt_lower, _CREATIVE_WRITING_SIGNALS):
        return TaskType.CREATIVE_WRITING

    if _contains_any(prompt_lower, _SUMMARIZATION_SIGNALS):
        return TaskType.SUMMARIZATION

    # Weak general-QA signal: question mark or starts with wh-/how/is/are/can
    if re.search(r"\?", prompt_lower) or re.match(
        r"(what|who|where|when|why|how|is|are|can|could|would|should)\b",
        prompt_lower,
    ):
        return TaskType.GENERAL_QA

    return TaskType.UNKNOWN


def _detect_expected_response_length(
    prompt_lower: str,
    max_tokens: int,
) -> ResponseLengthCategory:
    """
    Estimate output length category from prompt signals and max_tokens.

    Priority:
      1. Explicit short-response signals in prompt.
      2. Explicit long-response signals in prompt.
      3. max_tokens threshold.
    """
    if _contains_any(prompt_lower, _SHORT_RESPONSE_SIGNALS):
        return ResponseLengthCategory.SHORT

    if _contains_any(prompt_lower, _LONG_RESPONSE_SIGNALS):
        return ResponseLengthCategory.LONG

    if max_tokens <= _SHORT_TOKEN_THRESHOLD:
        return ResponseLengthCategory.SHORT
    if max_tokens >= _LONG_TOKEN_THRESHOLD:
        return ResponseLengthCategory.LONG
    return ResponseLengthCategory.MEDIUM


# ── Public analyzer API ───────────────────────────────────────────────────────

class RequestAnalyzer:
    """
    Stateless, deterministic feature extractor for LLMRequest objects.

    Usage::

        from app.routing.analyzer import RequestAnalyzer

        analyzer = RequestAnalyzer()
        analysis = analyzer.analyze(request)
        features = analysis.to_feature_vector()

    The analyzer is intentionally stateless — instantiation is cheap and it
    holds no mutable state, so a single shared instance is safe across threads.
    """

    # No __init__ required — stateless.

    def analyze(self, request: LLMRequest) -> RequestAnalysis:
        """
        Analyze an LLMRequest and return a structured RequestAnalysis.

        Args:
            request: The incoming LLMRequest (Phase 1 schema).

        Returns:
            RequestAnalysis populated with all nine feature groups.
        """
        prompt = request.prompt or ""
        prompt_lower = _lower(prompt)

        # ── Lexical features ───────────────────────────────────────────────
        prompt_length     = len(prompt)
        word_count        = len(prompt.split()) if prompt.strip() else 0
        approx_token_count = max(1, round(prompt_length / _CHARS_PER_TOKEN)) if prompt else 0

        # ── Task type ──────────────────────────────────────────────────────
        task_type = _detect_task_type(prompt_lower)

        # ── Capability indicators ──────────────────────────────────────────
        requires_reasoning = _contains_any(
            prompt_lower, _REASONING_REQUIREMENT_SIGNALS
        )
        requires_code = _contains_any(
            prompt_lower, _CODE_REQUIREMENT_SIGNALS
        )
        requires_structured_output = _contains_any(
            prompt_lower, _STRUCTURED_OUTPUT_SIGNALS
        )
        # Context: explicit signals OR very long prompt
        requires_context = (
            _contains_any(prompt_lower, _CONTEXT_REQUIREMENT_SIGNALS)
            or prompt_length > _LONG_PROMPT_CHAR_THRESHOLD
        )

        # ── Expected response length ───────────────────────────────────────
        expected_response_length = _detect_expected_response_length(
            prompt_lower, request.max_tokens
        )

        return RequestAnalysis(
            request_id=request.request_id,
            prompt_length=prompt_length,
            word_count=word_count,
            approx_token_count=approx_token_count,
            task_type=task_type,
            requires_reasoning=requires_reasoning,
            requires_code=requires_code,
            requires_structured_output=requires_structured_output,
            requires_context=requires_context,
            expected_response_length=expected_response_length,
            max_tokens=request.max_tokens,
            has_system_prompt=request.system_prompt is not None,
        )
