"""
app/evaluation/deterministic.py
───────────────────────────────
Phase 3 Part 2 — Deterministic Quality Metrics.

Implements a purely offline, rule-based evaluator that assesses response
quality based on observable characteristics without calling an external API.
"""

import json
from typing import Dict

from app.providers.base import LLMRequest, LLMResponse
from app.evaluation.base import BaseEvaluator
from app.evaluation.models import EvaluationResult, EvaluationStatus, EvaluationSource


class DeterministicMetrics:
    """Modular deterministic metric calculations."""

    @staticmethod
    def measure_presence(output: str | None) -> float:
        """
        Evaluates whether the response contains meaningful textual output.
        Returns 1.0 if non-empty, 0.0 if missing or only whitespace.
        """
        if not output or not output.strip():
            return 0.0
        return 1.0

    @staticmethod
    def measure_truncation(response: LLMResponse, output: str) -> float:
        """
        Evaluates if the response was excessively truncated or incomplete.
        0.0 if explicitly truncated by the provider's token limit.
        0.2 if markdown code blocks are left unclosed.
        0.8 if it ends abruptly without standard punctuation.
        1.0 otherwise.
        """
        if getattr(response, "finish_reason", None) == "length":
            return 0.0
        
        stripped = output.strip()
        if not stripped:
            return 1.0  # Handled by presence metric
        
        # Unclosed markdown code blocks highly suggest arbitrary truncation
        if stripped.count("```") % 2 != 0:
            return 0.2
            
        # Check for abrupt endings without punctuation
        if stripped[-1] not in ".!?\"'}`]*)>":
            return 0.8
            
        return 1.0

    @staticmethod
    def measure_repetition(output: str) -> float:
        """
        Detects degenerate output / endless looping via word uniqueness ratio.
        If a response repeats a small set of words over and over, the ratio drops.
        """
        words = output.lower().split()
        total_words = len(words)
        
        if total_words < 10:
            return 1.0  # Too short to reliably detect looping
            
        unique_words = len(set(words))
        ratio = unique_words / total_words
        
        # Natural language typically has a unique word ratio > 0.4.
        # Below 0.1 indicates extreme repetition or degeneracy.
        if ratio >= 0.4:
            return 1.0
        elif ratio <= 0.1:
            return 0.0
        else:
            # Linearly scale the interval (0.1, 0.4) onto (0.0, 1.0)
            return (ratio - 0.1) / 0.3

    @staticmethod
    def measure_format_compliance(request: LLMRequest, output: str) -> float:
        """
        Evaluates compliance if a structured format (JSON) is requested.
        1.0 if format wasn't requested OR if it successfully parses.
        0.0 if requested but fails to parse.
        """
        prompt_text = (request.prompt or "").lower()
        sys_text = (request.system_prompt or "").lower()
        combined = f"{prompt_text} {sys_text}"
        
        if "json" in combined:
            clean_output = output.strip()
            # Strip markdown formatting often returned by models
            if clean_output.startswith("```json"):
                clean_output = clean_output.strip("`").replace("json\n", "", 1).strip()
            elif clean_output.startswith("```"):
                clean_output = clean_output.strip("`").strip()
                
            try:
                # Attempt to extract JSON payload if wrapped in text
                start_obj = clean_output.find("{")
                end_obj = clean_output.rfind("}")
                start_arr = clean_output.find("[")
                end_arr = clean_output.rfind("]")
                
                # Check which structure encapsulates the other/appears outermost
                if start_obj >= 0 and end_obj >= 0 and end_obj >= start_obj:
                    if start_arr == -1 or (start_obj < start_arr and end_obj > end_arr):
                        json.loads(clean_output[start_obj:end_obj+1])
                        return 1.0
                
                if start_arr >= 0 and end_arr >= 0 and end_arr >= start_arr:
                    json.loads(clean_output[start_arr:end_arr+1])
                    return 1.0
                    
                # Direct parse as fallback
                json.loads(clean_output)
                return 1.0
            except json.JSONDecodeError:
                return 0.0
                
        # If no specific format was requested, it is inherently compliant
        return 1.0


class DeterministicEvaluator(BaseEvaluator):
    """
    Offline, deterministic evaluator that produces a quality score based on
    fast, observable heuristics rather than external semantic verification.
    """
    
    def __init__(self, weights: Dict[str, float] | None = None):
        """
        Initialize with optional custom metric weights.
        """
        # Default explicit weights prioritizing completeness and compliance
        self.weights = weights or {
            "presence": 0.3,
            "truncation": 0.2,
            "repetition": 0.2,
            "format": 0.3,
        }

    async def evaluate(self, request: LLMRequest, response: LLMResponse) -> EvaluationResult:
        try:
            if request is None or response is None:
                raise ValueError("Both request and response must be provided.")

            output = response.output or ""
            
            # Execute modular deterministic checks
            scores = {
                "presence": DeterministicMetrics.measure_presence(output),
                "truncation": DeterministicMetrics.measure_truncation(response, output),
                "repetition": DeterministicMetrics.measure_repetition(output),
                "format": DeterministicMetrics.measure_format_compliance(request, output),
            }
            
            # Compute weighted overall score
            total_weight = sum(self.weights.values())
            if total_weight <= 0:
                raise ValueError("Total weight of metrics must be greater than zero.")
                
            raw_overall = sum(scores[k] * self.weights.get(k, 0.0) for k in scores) / total_weight
            overall_score = max(0.0, min(1.0, raw_overall))
            
            explanation = (
                f"Deterministic evaluation completed. "
                f"Presence: {scores['presence']:.2f}, "
                f"Truncation: {scores['truncation']:.2f}, "
                f"Repetition: {scores['repetition']:.2f}, "
                f"Format: {scores['format']:.2f}."
            )
            
            return EvaluationResult(
                request_id=request.request_id,
                model_id=response.model_id,
                provider=response.provider,
                status=EvaluationStatus.SUCCESS,
                source=EvaluationSource.DETERMINISTIC,
                overall_quality_score=overall_score,
                individual_scores=scores,
                explanation=explanation,
            )
            
        except Exception as e:
            # Handle malformed data gracefully without crashing the pipeline
            req_id = getattr(request, "request_id", "unknown")
            mod_id = getattr(response, "model_id", "unknown")
            prov = getattr(response, "provider", "unknown")
            
            return EvaluationResult(
                request_id=req_id,
                model_id=mod_id,
                provider=prov,
                status=EvaluationStatus.FAILURE,
                source=EvaluationSource.DETERMINISTIC,
                error=f"Deterministic evaluation failed: {type(e).__name__}: {str(e)}",
            )
