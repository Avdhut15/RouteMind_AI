"""
app/evaluation/llm_judge.py
───────────────────────────
Phase 3 Part 3 — LLM-as-a-Judge Evaluator.

Sends a structured evaluation prompt to an LLM judge and parses
the structured JSON response into an EvaluationResult.

Design rules:
  - The judge is configured via JudgeConfig (model, provider, temperature).
  - The actual judge call is delegated to a JudgeCaller protocol so the
    evaluator can be tested with a mock without a real provider.
  - API credentials are read from environment/settings — never hard-coded.
  - Malformed or out-of-range judge responses are caught and converted to
    EvaluationResult(status=FAILURE). The routing pipeline never crashes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from app.evaluation.base import BaseEvaluator
from app.evaluation.models import (
    EvaluationResult,
    EvaluationSource,
    EvaluationStatus,
)
from app.providers.base import LLMRequest, LLMResponse


# ── Judge configuration ───────────────────────────────────────────────────────

@dataclass
class JudgeConfig:
    """
    Configuration for the LLM judge call.

    Attributes:
        judge_model_id:  Model ID used as the judge (e.g. 'openai/gpt-4o').
        judge_provider:  Provider name (e.g. 'openrouter').
        temperature:     Sampling temperature for the judge (low = deterministic).
        max_tokens:      Token budget for the judge's structured response.
        criteria:        List of evaluation criteria to assess.
    """
    judge_model_id: str
    judge_provider: str = "openrouter"
    temperature: float = 0.1
    max_tokens: int = 512
    criteria: list[str] = field(default_factory=lambda: [
        "relevance",
        "completeness",
        "clarity",
        "instruction_adherence",
    ])


# ── Judge caller protocol (allows dependency injection for testing) ────────────

class JudgeCaller(Protocol):
    """
    A protocol (interface) for making the actual call to the judge LLM.

    Implementations can be:
      - A real provider wrapper using the Phase 1 LLMProvider abstraction.
      - A mock/stub for unit tests.
    """
    async def call(self, prompt: str, config: JudgeConfig) -> str:
        """
        Send the evaluation prompt to the judge model.

        Args:
            prompt: The fully assembled evaluation prompt.
            config: Judge configuration specifying model and parameters.

        Returns:
            The raw text output from the judge model.

        Raises:
            Exception: Any exception is caught by LLMJudgeEvaluator.
        """
        ...


# ── Prompt construction ───────────────────────────────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """You are an objective, rigorous LLM evaluator. Your task is to evaluate the quality of an AI assistant's response.

== ORIGINAL REQUEST ==
{request_text}

== ASSISTANT RESPONSE ==
{response_text}

== EVALUATION INSTRUCTIONS ==
Evaluate the response on the following criteria. For each criterion, assign a score from 0.0 to 1.0 (where 1.0 is perfect).

Criteria:
{criteria_list}

Rules:
- Be objective and concise. Do not fabricate certainty about facts you cannot verify.
- Each score MUST be a decimal number between 0.0 and 1.0 inclusive.
- Provide a brief (1-2 sentence) reason for each score.
- Do NOT be lenient — give low scores when genuinely warranted.

Return your evaluation as a valid JSON object with this exact structure:
{{
  "scores": {{
    "relevance": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>,
    "clarity": <float 0.0-1.0>,
    "instruction_adherence": <float 0.0-1.0>
  }},
  "overall": <float 0.0-1.0>,
  "reasoning": "<concise overall reasoning>"
}}

Only include the criteria listed in "scores". Return ONLY the JSON object. Do not add any text before or after it."""


def build_judge_prompt(
    request: LLMRequest,
    response: LLMResponse,
    criteria: list[str],
) -> str:
    """Build the structured evaluation prompt sent to the judge model."""
    criteria_list = "\n".join(f"- {c}" for c in criteria)
    return _JUDGE_PROMPT_TEMPLATE.format(
        request_text=request.prompt[:4000],          # guard against huge inputs
        response_text=(response.output or "")[:4000],
        criteria_list=criteria_list,
    )


# ── Judge response parsing ────────────────────────────────────────────────────

def extract_json_from_judge_output(raw: str) -> dict[str, Any]:
    """
    Attempt to parse the judge's raw text output as JSON.

    The judge is instructed to return only a JSON object, but may occasionally
    wrap it in markdown fences. This function handles common wrapping patterns.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    text = raw.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Attempt direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to locate the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from judge output: {text[:200]!r}")


def validate_and_normalise_scores(
    parsed: dict[str, Any],
    expected_criteria: list[str],
) -> tuple[dict[str, float], float]:
    """
    Validate the parsed judge output and normalise all scores to [0.0, 1.0].

    Returns:
        Tuple of (individual_scores dict, overall_score float).

    Raises:
        ValueError: If required keys are missing or scores are out of range.
    """
    if "scores" not in parsed:
        raise ValueError("Judge response missing 'scores' key.")
    if "overall" not in parsed:
        raise ValueError("Judge response missing 'overall' key.")

    raw_scores: dict[str, Any] = parsed["scores"]
    individual: dict[str, float] = {}

    for criterion in expected_criteria:
        if criterion not in raw_scores:
            raise ValueError(f"Judge response missing score for criterion '{criterion}'.")
        raw_val = raw_scores[criterion]
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            raise ValueError(f"Score for '{criterion}' is not numeric: {raw_val!r}")
        if not (0.0 <= val <= 1.0):
            raise ValueError(
                f"Score for '{criterion}' is out of range [0, 1]: {val}"
            )
        individual[criterion] = round(val, 4)

    overall_raw = parsed["overall"]
    try:
        overall = float(overall_raw)
    except (TypeError, ValueError):
        raise ValueError(f"Overall score is not numeric: {overall_raw!r}")
    if not (0.0 <= overall <= 1.0):
        raise ValueError(f"Overall score is out of range [0, 1]: {overall}")
    overall = round(overall, 4)

    return individual, overall


# ── LLM-as-a-Judge evaluator ─────────────────────────────────────────────────

class LLMJudgeEvaluator(BaseEvaluator):
    """
    LLM-as-a-Judge evaluator.

    Sends a structured evaluation prompt to a configured judge model and
    converts the structured JSON response into an EvaluationResult.

    Usage::

        config = JudgeConfig(judge_model_id="openai/gpt-4o")
        caller = MyJudgeCallerImpl(api_key=settings.openrouter_api_key)
        evaluator = LLMJudgeEvaluator(config=config, caller=caller)
        result = await evaluator.evaluate(request, response)
    """

    def __init__(self, config: JudgeConfig, caller: JudgeCaller) -> None:
        self._config = config
        self._caller = caller

    async def evaluate(self, request: LLMRequest, response: LLMResponse) -> EvaluationResult:
        req_id = getattr(request, "request_id", "unknown")
        mod_id = getattr(response, "model_id", "unknown")
        prov = getattr(response, "provider", "unknown")

        try:
            if request is None or response is None:
                raise ValueError("Both request and response must be provided.")

            prompt = build_judge_prompt(request, response, self._config.criteria)
            raw_output = await self._caller.call(prompt, self._config)

            parsed = extract_json_from_judge_output(raw_output)
            individual_scores, overall_score = validate_and_normalise_scores(
                parsed, self._config.criteria
            )

            reasoning = parsed.get("reasoning") or "No reasoning provided by judge."

            return EvaluationResult(
                request_id=request.request_id,
                model_id=response.model_id,
                provider=response.provider,
                status=EvaluationStatus.SUCCESS,
                source=EvaluationSource.LLM_JUDGE,
                overall_quality_score=overall_score,
                individual_scores=individual_scores,
                explanation=str(reasoning),
            )

        except Exception as exc:
            return EvaluationResult(
                request_id=req_id,
                model_id=mod_id,
                provider=prov,
                status=EvaluationStatus.FAILURE,
                source=EvaluationSource.LLM_JUDGE,
                error=f"LLM-judge evaluation failed: {type(exc).__name__}: {exc}",
            )
