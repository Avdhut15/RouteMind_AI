"""
tests/test_escalation_policy.py
────────────────────────────────
Phase 4 Part 1 — Comprehensive unit tests for the escalation policy
and its result schemas.

All tests are offline and deterministic — no network calls, no LLM
providers, no database access.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from app.evaluation.models import QualityDecision
from app.routing.escalation_policy import (
    EscalationOutcome,
    EscalationPolicy,
    EscalationPolicyConfig,
    EscalationPolicyResult,
    NextStepIntent,
)
from app.core.exceptions import EscalationError, EscalationPolicyError, RouteMindError


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EscalationPolicyConfig — validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalationPolicyConfigDefaults:
    """Default config should be valid and have sensible values."""

    def test_defaults_are_valid(self):
        cfg = EscalationPolicyConfig()
        assert cfg.escalation_enabled is True
        assert cfg.max_total_attempts == 3
        assert cfg.max_escalation_attempts == 2

    def test_default_relationship_holds(self):
        cfg = EscalationPolicyConfig()
        assert cfg.max_escalation_attempts <= cfg.max_total_attempts - 1


class TestEscalationPolicyConfigValidation:
    """Config rejects invalid values loudly."""

    # ── max_total_attempts ───────────────────────────────────────────

    def test_max_total_attempts_zero_raises(self):
        with pytest.raises(ValueError, match="max_total_attempts must be >= 1"):
            EscalationPolicyConfig(max_total_attempts=0)

    def test_max_total_attempts_negative_raises(self):
        with pytest.raises(ValueError, match="max_total_attempts must be >= 1"):
            EscalationPolicyConfig(max_total_attempts=-1)

    def test_max_total_attempts_float_raises(self):
        with pytest.raises(TypeError, match="max_total_attempts must be an int"):
            EscalationPolicyConfig(max_total_attempts=3.5)

    def test_max_total_attempts_string_raises(self):
        with pytest.raises(TypeError, match="max_total_attempts must be an int"):
            EscalationPolicyConfig(max_total_attempts="3")

    def test_max_total_attempts_bool_raises(self):
        """bool is a subclass of int; we reject it explicitly."""
        with pytest.raises(TypeError, match="max_total_attempts must be an int"):
            EscalationPolicyConfig(max_total_attempts=True)

    # ── max_escalation_attempts ──────────────────────────────────────

    def test_max_escalation_attempts_negative_raises(self):
        with pytest.raises(ValueError, match="max_escalation_attempts must be >= 0"):
            EscalationPolicyConfig(max_escalation_attempts=-1)

    def test_max_escalation_attempts_float_raises(self):
        with pytest.raises(TypeError, match="max_escalation_attempts must be an int"):
            EscalationPolicyConfig(max_escalation_attempts=2.0)

    def test_max_escalation_attempts_bool_raises(self):
        with pytest.raises(TypeError, match="max_escalation_attempts must be an int"):
            EscalationPolicyConfig(max_escalation_attempts=False)

    # ── Relationship between total and escalation limits ─────────────

    def test_escalation_exceeds_total_minus_one_raises(self):
        with pytest.raises(ValueError, match="max_escalation_attempts.*must be <= max_total_attempts - 1"):
            EscalationPolicyConfig(max_total_attempts=2, max_escalation_attempts=2)

    def test_escalation_equals_total_minus_one_valid(self):
        cfg = EscalationPolicyConfig(max_total_attempts=3, max_escalation_attempts=2)
        assert cfg.max_escalation_attempts == 2

    def test_single_attempt_no_escalation(self):
        """max_total_attempts=1 forces max_escalation_attempts=0."""
        cfg = EscalationPolicyConfig(max_total_attempts=1, max_escalation_attempts=0)
        assert cfg.max_total_attempts == 1
        assert cfg.max_escalation_attempts == 0

    def test_single_attempt_with_escalation_raises(self):
        with pytest.raises(ValueError, match="max_escalation_attempts.*must be <= max_total_attempts - 1"):
            EscalationPolicyConfig(max_total_attempts=1, max_escalation_attempts=1)

    # ── escalation_enabled type check ────────────────────────────────

    def test_escalation_enabled_non_bool_raises(self):
        with pytest.raises(TypeError, match="escalation_enabled must be a bool"):
            EscalationPolicyConfig(escalation_enabled=1)

    def test_escalation_enabled_string_raises(self):
        with pytest.raises(TypeError, match="escalation_enabled must be a bool"):
            EscalationPolicyConfig(escalation_enabled="true")

    # ── Valid edge cases ─────────────────────────────────────────────

    def test_large_values_valid(self):
        cfg = EscalationPolicyConfig(max_total_attempts=100, max_escalation_attempts=99)
        assert cfg.max_total_attempts == 100

    def test_zero_escalation_attempts_valid(self):
        """Escalation enabled but 0 escalation attempts means no actual escalation."""
        cfg = EscalationPolicyConfig(
            escalation_enabled=True,
            max_total_attempts=3,
            max_escalation_attempts=0,
        )
        assert cfg.max_escalation_attempts == 0

    def test_config_is_frozen(self):
        cfg = EscalationPolicyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_total_attempts = 5


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EscalationOutcome & NextStepIntent — enum coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalationOutcomeEnum:
    """Verify all expected enum members exist."""

    def test_all_members(self):
        assert set(EscalationOutcome) == {
            EscalationOutcome.ACCEPTED,
            EscalationOutcome.ESCALATION_REQUESTED,
            EscalationOutcome.ESCALATION_UNAVAILABLE,
            EscalationOutcome.ESCALATION_EXHAUSTED,
            EscalationOutcome.TERMINAL_FAILURE,
        }

    def test_string_values(self):
        assert EscalationOutcome.ACCEPTED == "accepted"
        assert EscalationOutcome.ESCALATION_REQUESTED == "escalation_requested"
        assert EscalationOutcome.ESCALATION_UNAVAILABLE == "escalation_unavailable"
        assert EscalationOutcome.ESCALATION_EXHAUSTED == "escalation_exhausted"
        assert EscalationOutcome.TERMINAL_FAILURE == "terminal_failure"

    def test_invalid_member_raises(self):
        with pytest.raises(ValueError):
            EscalationOutcome("nonexistent")


class TestNextStepIntentEnum:
    """Verify all expected enum members exist."""

    def test_all_members(self):
        assert set(NextStepIntent) == {
            NextStepIntent.RETURN_RESPONSE,
            NextStepIntent.SELECT_STRONGER_MODEL,
            NextStepIntent.FAIL_REQUEST,
        }

    def test_string_values(self):
        assert NextStepIntent.RETURN_RESPONSE == "return_response"
        assert NextStepIntent.SELECT_STRONGER_MODEL == "select_stronger_model"
        assert NextStepIntent.FAIL_REQUEST == "fail_request"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EscalationPolicy.evaluate() — PASS decision
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyPassDecision:
    """PASS always results in ACCEPTED, regardless of attempt state."""

    def test_pass_on_first_attempt(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-001",
            current_model_id="gpt-4o",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=QualityDecision.PASS,
        )
        assert result.outcome == EscalationOutcome.ACCEPTED
        assert result.is_terminal is True
        assert result.next_step_intent == NextStepIntent.RETURN_RESPONSE
        assert result.quality_decision == QualityDecision.PASS

    def test_pass_on_later_attempt(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-002",
            current_model_id="claude-3-opus",
            current_provider="openrouter",
            current_attempt=2,
            quality_decision=QualityDecision.PASS,
        )
        assert result.outcome == EscalationOutcome.ACCEPTED
        assert result.is_terminal is True

    def test_pass_on_max_attempt(self):
        cfg = EscalationPolicyConfig(max_total_attempts=3, max_escalation_attempts=2)
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-003",
            current_model_id="gemini-pro",
            current_provider="gemini",
            current_attempt=3,
            quality_decision=QualityDecision.PASS,
        )
        assert result.outcome == EscalationOutcome.ACCEPTED
        assert result.is_terminal is True

    def test_pass_with_escalation_disabled(self):
        """PASS should still be ACCEPTED even when escalation is disabled."""
        cfg = EscalationPolicyConfig(
            escalation_enabled=False,
            max_total_attempts=1,
            max_escalation_attempts=0,
        )
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-004",
            current_model_id="llama-3",
            current_provider="ollama",
            current_attempt=1,
            quality_decision=QualityDecision.PASS,
        )
        assert result.outcome == EscalationOutcome.ACCEPTED


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EscalationPolicy.evaluate() — FAIL decision
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyFailDecision:
    """FAIL always results in TERMINAL_FAILURE — no automatic retry."""

    def test_fail_on_first_attempt(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-010",
            current_model_id="gpt-3.5-turbo",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=QualityDecision.FAIL,
        )
        assert result.outcome == EscalationOutcome.TERMINAL_FAILURE
        assert result.is_terminal is True
        assert result.next_step_intent == NextStepIntent.FAIL_REQUEST
        assert result.quality_decision == QualityDecision.FAIL

    def test_fail_on_later_attempt(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-011",
            current_model_id="claude-3-haiku",
            current_provider="openrouter",
            current_attempt=2,
            quality_decision=QualityDecision.FAIL,
        )
        assert result.outcome == EscalationOutcome.TERMINAL_FAILURE
        assert result.is_terminal is True
        assert result.next_step_intent == NextStepIntent.FAIL_REQUEST

    def test_fail_does_not_retry_even_when_escalation_enabled(self):
        """FAIL is never retried, even when escalation budget remains."""
        cfg = EscalationPolicyConfig(
            escalation_enabled=True,
            max_total_attempts=5,
            max_escalation_attempts=4,
        )
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-012",
            current_model_id="mistral-large",
            current_provider="mistral",
            current_attempt=1,
            quality_decision=QualityDecision.FAIL,
        )
        assert result.outcome == EscalationOutcome.TERMINAL_FAILURE
        assert result.is_terminal is True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EscalationPolicy.evaluate() — ESCALATE when permitted
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyEscalatePermitted:
    """ESCALATE with budget remaining → ESCALATION_REQUESTED."""

    def test_escalate_first_attempt_permitted(self):
        policy = EscalationPolicy()  # defaults: total=3, esc=2, enabled=True
        result = policy.evaluate(
            request_id="req-020",
            current_model_id="gpt-3.5-turbo",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert result.outcome == EscalationOutcome.ESCALATION_REQUESTED
        assert result.is_terminal is False
        assert result.next_step_intent == NextStepIntent.SELECT_STRONGER_MODEL

    def test_escalate_second_attempt_permitted(self):
        policy = EscalationPolicy()  # total=3, esc=2
        result = policy.evaluate(
            request_id="req-021",
            current_model_id="gpt-4o-mini",
            current_provider="openrouter",
            current_attempt=2,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert result.outcome == EscalationOutcome.ESCALATION_REQUESTED
        assert result.is_terminal is False

    def test_escalate_preserves_model_identity(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-022",
            current_model_id="my-model-v2",
            current_provider="groq",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert result.current_model_id == "my-model-v2"
        assert result.current_provider == "groq"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EscalationPolicy.evaluate() — escalation disabled
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyEscalationDisabled:
    """When escalation_enabled=False, ESCALATE → ESCALATION_UNAVAILABLE."""

    def test_escalation_disabled_first_attempt(self):
        cfg = EscalationPolicyConfig(
            escalation_enabled=False,
            max_total_attempts=3,
            max_escalation_attempts=0,
        )
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-030",
            current_model_id="llama-3",
            current_provider="ollama",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert result.outcome == EscalationOutcome.ESCALATION_UNAVAILABLE
        assert result.is_terminal is True
        assert result.next_step_intent == NextStepIntent.RETURN_RESPONSE
        assert result.escalation_enabled is False

    def test_escalation_disabled_reason_mentions_disabled(self):
        cfg = EscalationPolicyConfig(
            escalation_enabled=False,
            max_total_attempts=1,
            max_escalation_attempts=0,
        )
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-031",
            current_model_id="tiny-llm",
            current_provider="ollama",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert "disabled" in result.reason.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EscalationPolicy.evaluate() — total attempt limit reached
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyTotalLimitReached:
    """When current_attempt == max_total_attempts, ESCALATE → EXHAUSTED."""

    def test_total_limit_reached(self):
        cfg = EscalationPolicyConfig(max_total_attempts=3, max_escalation_attempts=2)
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-040",
            current_model_id="gpt-4-turbo",
            current_provider="openrouter",
            current_attempt=3,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert result.outcome == EscalationOutcome.ESCALATION_EXHAUSTED
        assert result.is_terminal is True
        assert result.next_step_intent == NextStepIntent.RETURN_RESPONSE

    def test_single_attempt_limit_escalate(self):
        """Only 1 attempt allowed: ESCALATE on attempt 1 → EXHAUSTED."""
        cfg = EscalationPolicyConfig(
            escalation_enabled=True,
            max_total_attempts=1,
            max_escalation_attempts=0,
        )
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-041",
            current_model_id="phi-3",
            current_provider="ollama",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        # Total limit is 1, current attempt is 1 → no room for another attempt
        assert result.outcome == EscalationOutcome.ESCALATION_EXHAUSTED
        assert result.is_terminal is True


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EscalationPolicy.evaluate() — escalation-attempt limit reached
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyEscalationLimitReached:
    """When escalation attempts used >= max_escalation_attempts → EXHAUSTED."""

    def test_escalation_limit_reached_before_total(self):
        """Escalation limit hit while total budget remains."""
        cfg = EscalationPolicyConfig(max_total_attempts=5, max_escalation_attempts=1)
        policy = EscalationPolicy(cfg)
        # Attempt 1 = initial. Attempt 2 = first escalation (uses 1 of 1).
        # Attempt 2: escalation_attempts_used = 1 == max_escalation_attempts → EXHAUSTED
        result = policy.evaluate(
            request_id="req-050",
            current_model_id="gpt-4",
            current_provider="openrouter",
            current_attempt=2,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert result.outcome == EscalationOutcome.ESCALATION_EXHAUSTED
        assert result.is_terminal is True

    def test_zero_escalation_attempts_allowed(self):
        """max_escalation_attempts=0: first ESCALATE on attempt 1 hits limit."""
        cfg = EscalationPolicyConfig(
            escalation_enabled=True,
            max_total_attempts=3,
            max_escalation_attempts=0,
        )
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-051",
            current_model_id="llama-3",
            current_provider="ollama",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        # current_attempt=1, total limit not reached (1 < 3), but
        # escalation_attempts_used=0 >= max_escalation_attempts=0 → EXHAUSTED
        assert result.outcome == EscalationOutcome.ESCALATION_EXHAUSTED
        assert result.is_terminal is True


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Invalid attempt counters
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyInvalidAttemptCounters:
    """Policy rejects invalid attempt values."""

    def test_attempt_zero_raises(self):
        policy = EscalationPolicy()
        with pytest.raises(ValueError, match="current_attempt must be >= 1"):
            policy.evaluate(
                request_id="req-060",
                current_model_id="m",
                current_provider="p",
                current_attempt=0,
                quality_decision=QualityDecision.PASS,
            )

    def test_attempt_negative_raises(self):
        policy = EscalationPolicy()
        with pytest.raises(ValueError, match="current_attempt must be >= 1"):
            policy.evaluate(
                request_id="req-061",
                current_model_id="m",
                current_provider="p",
                current_attempt=-1,
                quality_decision=QualityDecision.PASS,
            )

    def test_attempt_exceeds_max_raises(self):
        cfg = EscalationPolicyConfig(max_total_attempts=3, max_escalation_attempts=2)
        policy = EscalationPolicy(cfg)
        with pytest.raises(ValueError, match="current_attempt.*exceeds.*max_total_attempts"):
            policy.evaluate(
                request_id="req-062",
                current_model_id="m",
                current_provider="p",
                current_attempt=4,
                quality_decision=QualityDecision.PASS,
            )

    def test_attempt_bool_raises(self):
        policy = EscalationPolicy()
        with pytest.raises(TypeError, match="current_attempt must be an int"):
            policy.evaluate(
                request_id="req-063",
                current_model_id="m",
                current_provider="p",
                current_attempt=True,
                quality_decision=QualityDecision.PASS,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Invalid quality_decision
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyInvalidQualityDecision:
    """Policy rejects invalid quality_decision types."""

    def test_string_raises(self):
        policy = EscalationPolicy()
        with pytest.raises(TypeError, match="quality_decision must be a QualityDecision"):
            policy.evaluate(
                request_id="req-070",
                current_model_id="m",
                current_provider="p",
                current_attempt=1,
                quality_decision="pass",
            )

    def test_none_raises(self):
        policy = EscalationPolicy()
        with pytest.raises(TypeError, match="quality_decision must be a QualityDecision"):
            policy.evaluate(
                request_id="req-071",
                current_model_id="m",
                current_provider="p",
                current_attempt=1,
                quality_decision=None,
            )

    def test_int_raises(self):
        policy = EscalationPolicy()
        with pytest.raises(TypeError, match="quality_decision must be a QualityDecision"):
            policy.evaluate(
                request_id="req-072",
                current_model_id="m",
                current_provider="p",
                current_attempt=1,
                quality_decision=1,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Request ID and model identity preservation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyIdentityPreservation:
    """Policy preserves request_id, model_id, and provider in every result."""

    @pytest.mark.parametrize("decision", list(QualityDecision))
    def test_request_id_preserved(self, decision):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="unique-id-abc-123",
            current_model_id="test-model",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=decision,
        )
        assert result.request_id == "unique-id-abc-123"

    @pytest.mark.parametrize("decision", list(QualityDecision))
    def test_model_id_preserved(self, decision):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req",
            current_model_id="specific-model-id",
            current_provider="groq",
            current_attempt=1,
            quality_decision=decision,
        )
        assert result.current_model_id == "specific-model-id"

    @pytest.mark.parametrize("decision", list(QualityDecision))
    def test_provider_preserved(self, decision):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req",
            current_model_id="m",
            current_provider="custom-provider",
            current_attempt=1,
            quality_decision=decision,
        )
        assert result.current_provider == "custom-provider"


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Deterministic repeated decisions
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyDeterminism:
    """Identical inputs always produce identical outputs."""

    def test_same_inputs_same_output(self):
        policy = EscalationPolicy()
        kwargs = dict(
            request_id="req-det-01",
            current_model_id="gpt-4o",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        result1 = policy.evaluate(**kwargs)
        result2 = policy.evaluate(**kwargs)
        assert result1 == result2

    def test_deterministic_across_policy_instances(self):
        """Two independently constructed policies produce the same result."""
        cfg = EscalationPolicyConfig(max_total_attempts=2, max_escalation_attempts=1)
        policy_a = EscalationPolicy(cfg)
        policy_b = EscalationPolicy(EscalationPolicyConfig(
            max_total_attempts=2, max_escalation_attempts=1,
        ))
        kwargs = dict(
            request_id="req-det-02",
            current_model_id="claude-3-sonnet",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert policy_a.evaluate(**kwargs) == policy_b.evaluate(**kwargs)

    def test_many_iterations_deterministic(self):
        policy = EscalationPolicy()
        kwargs = dict(
            request_id="req-det-03",
            current_model_id="m",
            current_provider="p",
            current_attempt=1,
            quality_decision=QualityDecision.PASS,
        )
        results = [policy.evaluate(**kwargs) for _ in range(50)]
        assert all(r == results[0] for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Terminal vs non-terminal outcomes
# ═══════════════════════════════════════════════════════════════════════════════


class TestTerminalOutcomes:
    """Only ESCALATION_REQUESTED is non-terminal."""

    TERMINAL_OUTCOMES = {
        EscalationOutcome.ACCEPTED,
        EscalationOutcome.ESCALATION_UNAVAILABLE,
        EscalationOutcome.ESCALATION_EXHAUSTED,
        EscalationOutcome.TERMINAL_FAILURE,
    }

    NON_TERMINAL_OUTCOMES = {
        EscalationOutcome.ESCALATION_REQUESTED,
    }

    def test_pass_is_terminal(self):
        policy = EscalationPolicy()
        r = policy.evaluate(
            request_id="r", current_model_id="m", current_provider="p",
            current_attempt=1, quality_decision=QualityDecision.PASS,
        )
        assert r.is_terminal is True
        assert r.outcome in self.TERMINAL_OUTCOMES

    def test_fail_is_terminal(self):
        policy = EscalationPolicy()
        r = policy.evaluate(
            request_id="r", current_model_id="m", current_provider="p",
            current_attempt=1, quality_decision=QualityDecision.FAIL,
        )
        assert r.is_terminal is True
        assert r.outcome in self.TERMINAL_OUTCOMES

    def test_escalate_permitted_is_non_terminal(self):
        policy = EscalationPolicy()
        r = policy.evaluate(
            request_id="r", current_model_id="m", current_provider="p",
            current_attempt=1, quality_decision=QualityDecision.ESCALATE,
        )
        assert r.is_terminal is False
        assert r.outcome in self.NON_TERMINAL_OUTCOMES

    def test_escalate_exhausted_is_terminal(self):
        cfg = EscalationPolicyConfig(max_total_attempts=2, max_escalation_attempts=1)
        policy = EscalationPolicy(cfg)
        r = policy.evaluate(
            request_id="r", current_model_id="m", current_provider="p",
            current_attempt=2, quality_decision=QualityDecision.ESCALATE,
        )
        assert r.is_terminal is True

    def test_escalate_disabled_is_terminal(self):
        cfg = EscalationPolicyConfig(
            escalation_enabled=False, max_total_attempts=1, max_escalation_attempts=0,
        )
        policy = EscalationPolicy(cfg)
        r = policy.evaluate(
            request_id="r", current_model_id="m", current_provider="p",
            current_attempt=1, quality_decision=QualityDecision.ESCALATE,
        )
        assert r.is_terminal is True


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Schema serialization and validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaSerialisation:
    """EscalationPolicyResult is a frozen dataclass — test its fields."""

    def test_result_has_all_expected_fields(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-ser-01",
            current_model_id="gpt-4o",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=QualityDecision.PASS,
        )
        expected_fields = {
            "request_id", "current_model_id", "current_provider",
            "current_attempt", "max_total_attempts", "max_escalation_attempts",
            "quality_decision", "escalation_enabled", "outcome", "reason",
            "is_terminal", "next_step_intent",
        }
        actual_fields = {f.name for f in dataclasses.fields(result)}
        assert actual_fields == expected_fields

    def test_result_is_frozen(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-ser-02",
            current_model_id="m",
            current_provider="p",
            current_attempt=1,
            quality_decision=QualityDecision.PASS,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.outcome = EscalationOutcome.TERMINAL_FAILURE

    def test_result_to_dict(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-ser-03",
            current_model_id="m",
            current_provider="p",
            current_attempt=1,
            quality_decision=QualityDecision.PASS,
        )
        d = dataclasses.asdict(result)
        assert d["request_id"] == "req-ser-03"
        assert d["outcome"] == "accepted"
        assert d["quality_decision"] == "pass"
        assert d["is_terminal"] is True
        assert d["next_step_intent"] == "return_response"

    def test_result_dict_is_json_serialisable(self):
        """Verifies the result can be serialised to JSON (e.g. for logging)."""
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-ser-04",
            current_model_id="m",
            current_provider="p",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        d = dataclasses.asdict(result)
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["outcome"] == result.outcome.value


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Policy config propagation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyConfigPropagation:
    """Policy results carry the active config values for downstream inspection."""

    def test_config_values_in_result(self):
        cfg = EscalationPolicyConfig(
            escalation_enabled=True,
            max_total_attempts=5,
            max_escalation_attempts=3,
        )
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-cfg-01",
            current_model_id="m",
            current_provider="p",
            current_attempt=1,
            quality_decision=QualityDecision.PASS,
        )
        assert result.max_total_attempts == 5
        assert result.max_escalation_attempts == 3
        assert result.escalation_enabled is True

    def test_default_config_exposed(self):
        policy = EscalationPolicy()
        assert policy.config == EscalationPolicyConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Exception hierarchy
# ═══════════════════════════════════════════════════════════════════════════════


class TestExceptionHierarchy:
    """Escalation exceptions follow the RouteMindError hierarchy."""

    def test_escalation_error_is_routemind_error(self):
        assert issubclass(EscalationError, RouteMindError)

    def test_escalation_policy_error_is_escalation_error(self):
        assert issubclass(EscalationPolicyError, EscalationError)

    def test_escalation_policy_error_is_routemind_error(self):
        assert issubclass(EscalationPolicyError, RouteMindError)

    def test_escalation_error_catchable_as_routemind(self):
        with pytest.raises(RouteMindError):
            raise EscalationPolicyError("test")


# ═══════════════════════════════════════════════════════════════════════════════
# 17. No-eligible-alternative scenario
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoEligibleAlternative:
    """
    The policy does not check for eligible alternatives (that is Part 2).
    When escalation is permitted, the policy returns ESCALATION_REQUESTED
    with next_step_intent=SELECT_STRONGER_MODEL.

    If no stronger model exists, that will be resolved by Part 2 logic.
    The policy's job is to say "try to find one" — not to verify one exists.

    However, the policy CAN produce terminal outcomes that effectively
    prevent escalation (EXHAUSTED, UNAVAILABLE), which covers the case
    where escalation will never succeed anyway.
    """

    def test_escalation_requested_does_not_guarantee_alternative(self):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="req-noalt-01",
            current_model_id="already-top-tier",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        # The policy still says "try to escalate" — it doesn't know
        # whether a better model exists.
        assert result.outcome == EscalationOutcome.ESCALATION_REQUESTED
        assert result.next_step_intent == NextStepIntent.SELECT_STRONGER_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Edge case: escalation enabled but zero max_escalation_attempts
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalationEnabledButZeroEscalations:
    """
    escalation_enabled=True but max_escalation_attempts=0 means the switch
    is on but the budget is exhausted before any escalation can happen.
    """

    def test_escalate_with_zero_budget_is_exhausted(self):
        cfg = EscalationPolicyConfig(
            escalation_enabled=True,
            max_total_attempts=2,
            max_escalation_attempts=0,
        )
        policy = EscalationPolicy(cfg)
        result = policy.evaluate(
            request_id="req-edge-01",
            current_model_id="m",
            current_provider="p",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        # escalation_enabled is checked first → passes.
        # Then total limit: 1 < 2 → not exhausted by total.
        # Then escalation limit: 0 >= 0 → EXHAUSTED.
        assert result.outcome == EscalationOutcome.ESCALATION_EXHAUSTED
        assert result.is_terminal is True


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Comprehensive multi-step escalation scenario
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiStepEscalationScenario:
    """Simulate a full escalation chain across multiple attempts."""

    def test_three_attempt_chain(self):
        cfg = EscalationPolicyConfig(max_total_attempts=3, max_escalation_attempts=2)
        policy = EscalationPolicy(cfg)

        # Attempt 1: ESCALATE → permitted (0/2 escalations used)
        r1 = policy.evaluate(
            request_id="req-chain",
            current_model_id="model-a",
            current_provider="ollama",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert r1.outcome == EscalationOutcome.ESCALATION_REQUESTED
        assert r1.is_terminal is False

        # Attempt 2: ESCALATE → permitted (1/2 escalations used)
        r2 = policy.evaluate(
            request_id="req-chain",
            current_model_id="model-b",
            current_provider="openrouter",
            current_attempt=2,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert r2.outcome == EscalationOutcome.ESCALATION_REQUESTED
        assert r2.is_terminal is False

        # Attempt 3: ESCALATE → exhausted (2/2 escalations used AND total limit)
        r3 = policy.evaluate(
            request_id="req-chain",
            current_model_id="model-c",
            current_provider="openrouter",
            current_attempt=3,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert r3.outcome == EscalationOutcome.ESCALATION_EXHAUSTED
        assert r3.is_terminal is True

    def test_chain_with_early_pass(self):
        cfg = EscalationPolicyConfig(max_total_attempts=3, max_escalation_attempts=2)
        policy = EscalationPolicy(cfg)

        # Attempt 1: ESCALATE
        r1 = policy.evaluate(
            request_id="req-early",
            current_model_id="model-a",
            current_provider="ollama",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert r1.is_terminal is False

        # Attempt 2: PASS → accepted even though budget remains
        r2 = policy.evaluate(
            request_id="req-early",
            current_model_id="model-b",
            current_provider="openrouter",
            current_attempt=2,
            quality_decision=QualityDecision.PASS,
        )
        assert r2.outcome == EscalationOutcome.ACCEPTED
        assert r2.is_terminal is True

    def test_chain_with_early_fail(self):
        cfg = EscalationPolicyConfig(max_total_attempts=3, max_escalation_attempts=2)
        policy = EscalationPolicy(cfg)

        # Attempt 1: ESCALATE
        r1 = policy.evaluate(
            request_id="req-fail-chain",
            current_model_id="model-a",
            current_provider="ollama",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )
        assert r1.is_terminal is False

        # Attempt 2: FAIL → terminal even though escalation budget remains
        r2 = policy.evaluate(
            request_id="req-fail-chain",
            current_model_id="model-b",
            current_provider="openrouter",
            current_attempt=2,
            quality_decision=QualityDecision.FAIL,
        )
        assert r2.outcome == EscalationOutcome.TERMINAL_FAILURE
        assert r2.is_terminal is True
        assert r2.next_step_intent == NextStepIntent.FAIL_REQUEST


# ═══════════════════════════════════════════════════════════════════════════════
# 20. Reason field is always non-empty
# ═══════════════════════════════════════════════════════════════════════════════


class TestReasonField:
    """Every outcome produces a non-empty, descriptive reason string."""

    @pytest.mark.parametrize("decision", list(QualityDecision))
    def test_reason_is_non_empty(self, decision):
        policy = EscalationPolicy()
        result = policy.evaluate(
            request_id="r", current_model_id="m", current_provider="p",
            current_attempt=1, quality_decision=decision,
        )
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
