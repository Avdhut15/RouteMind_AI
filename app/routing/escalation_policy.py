"""
app/routing/escalation_policy.py
─────────────────────────────────
Phase 4 Part 1 — Escalation Policy & Result Schema.

Defines the **policy** and **data contracts** for handling PASS, FAIL, and
ESCALATE decisions produced by the Phase 3 evaluation subsystem.

This module establishes *what* the system should do for each decision
outcome.  It does NOT execute retries, select stronger models, call
providers, or modify any Phase 2/Phase 3 behaviour.

Design rules
─────────────
  - Stateless:      no global mutable state, no caches, no persistence.
  - Deterministic:  identical inputs always produce identical outputs.
  - Configurable:   policy limits are set via EscalationPolicyConfig.
  - Offline:        no network calls, no provider dependencies.
  - Explicit:       every expected condition has a structured outcome;
                    normal control flow never relies on exceptions.

Attempt-counter semantics
──────────────────────────
  - ``max_total_attempts``:  Upper bound on the total number of inference
    attempts for a single request, *including* the initial attempt.  A
    value of 1 means the initial attempt is the only one — no escalation
    can occur.  Minimum valid value: 1.

  - ``max_escalation_attempts``:  Upper bound on the number of
    *escalation* steps (attempts beyond the first).  Must satisfy
    ``0 <= max_escalation_attempts < max_total_attempts``.

  - ``current_attempt``:  1-indexed counter of the attempt currently
    being evaluated.  The initial attempt is attempt 1.

Relationship constraint:
    max_escalation_attempts <= max_total_attempts - 1

    If max_total_attempts = 3 and max_escalation_attempts = 2, then the
    system may make 1 initial attempt + up to 2 escalation attempts = 3
    total attempts.

Limitations
───────────
  - The policy does not select a stronger model; that belongs to
    Phase 4 Part 2.
  - The policy does not execute retries or provider calls; that belongs
    to Phase 4 Part 3.
  - A model's tier label alone is NOT treated as proof that escalation
    to a higher-tier model will produce a better response.
  - The ``next_step_intent`` field in EscalationPolicyResult is
    informational; the actual execution is deferred to later parts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.evaluation.models import QualityDecision


# ── Enums ─────────────────────────────────────────────────────────────────────

class EscalationOutcome(str, Enum):
    """
    Terminal or non-terminal outcome produced by the escalation policy.

    Values
    ──────
    ACCEPTED
        The response passed quality evaluation.  No further action is
        needed.  Terminal.

    ESCALATION_REQUESTED
        The evaluation returned ESCALATE, the policy allows escalation,
        and attempt limits have not been reached.  Non-terminal — a
        downstream component should attempt to find a stronger model.

    ESCALATION_UNAVAILABLE
        Escalation was requested but is globally disabled via policy
        configuration.  Terminal.

    ESCALATION_EXHAUSTED
        Escalation was requested but the attempt limit (total or
        escalation-specific) has been reached.  Terminal.

    TERMINAL_FAILURE
        The evaluation returned FAIL.  The configured failure outcome
        is a hard stop — no automatic retry.  Terminal.
    """
    ACCEPTED = "accepted"
    ESCALATION_REQUESTED = "escalation_requested"
    ESCALATION_UNAVAILABLE = "escalation_unavailable"
    ESCALATION_EXHAUSTED = "escalation_exhausted"
    TERMINAL_FAILURE = "terminal_failure"


class NextStepIntent(str, Enum):
    """
    Advisory intent describing the recommended next step.

    This is purely informational — the policy does NOT execute any of
    these intents.  Execution belongs to Phase 4 Part 3.

    Values
    ──────
    RETURN_RESPONSE
        Deliver the current response to the caller (PASS or terminal
        failure with no retry).

    SELECT_STRONGER_MODEL
        Attempt to find an eligible alternative model and retry.

    FAIL_REQUEST
        Terminate the request pipeline with a failure result.
    """
    RETURN_RESPONSE = "return_response"
    SELECT_STRONGER_MODEL = "select_stronger_model"
    FAIL_REQUEST = "fail_request"


# ── Policy Configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class EscalationPolicyConfig:
    """
    Configuration knobs for the escalation policy.

    All values are validated at construction time.  Invalid
    configurations raise ``ValueError`` or ``TypeError`` — they are
    never silently repaired.

    Attributes
    ──────────
    escalation_enabled
        Master switch.  When False, any ESCALATE decision is converted
        to ESCALATION_UNAVAILABLE immediately.

    max_total_attempts
        Maximum inference attempts for one request, including the
        initial attempt.  Must be >= 1.

    max_escalation_attempts
        Maximum escalation steps (beyond the initial attempt).
        Must be >= 0 and <= max_total_attempts - 1.
    """
    escalation_enabled: bool = True
    max_total_attempts: int = 3
    max_escalation_attempts: int = 2

    def __post_init__(self) -> None:
        # ── Type checks ──────────────────────────────────────────────
        if not isinstance(self.escalation_enabled, bool):
            raise TypeError(
                f"escalation_enabled must be a bool, "
                f"got {type(self.escalation_enabled).__name__}."
            )

        if not isinstance(self.max_total_attempts, int) or isinstance(
            self.max_total_attempts, bool
        ):
            raise TypeError(
                f"max_total_attempts must be an int, "
                f"got {type(self.max_total_attempts).__name__}."
            )

        if not isinstance(self.max_escalation_attempts, int) or isinstance(
            self.max_escalation_attempts, bool
        ):
            raise TypeError(
                f"max_escalation_attempts must be an int, "
                f"got {type(self.max_escalation_attempts).__name__}."
            )

        # ── Value checks ─────────────────────────────────────────────
        if self.max_total_attempts < 1:
            raise ValueError(
                f"max_total_attempts must be >= 1, got {self.max_total_attempts}. "
                f"A value of 1 means only the initial attempt is allowed."
            )

        if self.max_escalation_attempts < 0:
            raise ValueError(
                f"max_escalation_attempts must be >= 0, "
                f"got {self.max_escalation_attempts}."
            )

        if self.max_escalation_attempts > self.max_total_attempts - 1:
            raise ValueError(
                f"max_escalation_attempts ({self.max_escalation_attempts}) must be "
                f"<= max_total_attempts - 1 ({self.max_total_attempts - 1}). "
                f"The initial attempt is not an escalation."
            )


# ── Policy Result ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EscalationPolicyResult:
    """
    Structured output of the escalation policy.

    Carries the full context needed by downstream Phase 4 components
    without duplicating entire RouterResult or EvaluationPipelineResult
    trees.

    Attributes
    ──────────
    request_id
        Mirrors the original LLMRequest.request_id — preserved
        throughout the escalation chain.

    current_model_id
        Model that produced the response being evaluated.

    current_provider
        Provider that served the response.

    current_attempt
        1-indexed attempt number that was just evaluated.

    max_total_attempts
        Maximum total attempts from the active policy configuration.

    max_escalation_attempts
        Maximum escalation attempts from the active policy
        configuration.

    quality_decision
        The QualityDecision (PASS/FAIL/ESCALATE) from the Phase 3
        evaluation subsystem that triggered this policy evaluation.

    escalation_enabled
        Whether escalation is enabled in the active policy.

    outcome
        The EscalationOutcome produced by the policy.

    reason
        Human-readable explanation of the outcome.

    is_terminal
        True when the outcome is final and no further attempts should
        be made.  False only when outcome is ESCALATION_REQUESTED.

    next_step_intent
        Advisory intent for the downstream executor.  Informational
        only — the policy does NOT execute this intent.
    """
    request_id: str
    current_model_id: str
    current_provider: str
    current_attempt: int
    max_total_attempts: int
    max_escalation_attempts: int
    quality_decision: QualityDecision
    escalation_enabled: bool
    outcome: EscalationOutcome
    reason: str
    is_terminal: bool
    next_step_intent: NextStepIntent


# ── Escalation Policy ─────────────────────────────────────────────────────────

class EscalationPolicy:
    """
    Stateless, deterministic policy that converts a QualityDecision and
    attempt context into an EscalationPolicyResult.

    The policy answers: "Given this evaluation decision and the current
    attempt state, what should happen next?"

    It does NOT:
      - select a stronger model (Part 2)
      - execute retries or provider calls (Part 3)
      - modify routing scores or evaluation thresholds

    Usage::

        config = EscalationPolicyConfig(
            escalation_enabled=True,
            max_total_attempts=3,
            max_escalation_attempts=2,
        )
        policy = EscalationPolicy(config)

        result = policy.evaluate(
            request_id="abc-123",
            current_model_id="gpt-4o",
            current_provider="openrouter",
            current_attempt=1,
            quality_decision=QualityDecision.ESCALATE,
        )

        if result.is_terminal:
            # Return the current response or fail the request
            ...
        else:
            # result.next_step_intent == NextStepIntent.SELECT_STRONGER_MODEL
            ...
    """

    def __init__(self, config: Optional[EscalationPolicyConfig] = None) -> None:
        self._config = config or EscalationPolicyConfig()

    @property
    def config(self) -> EscalationPolicyConfig:
        """Return the active policy configuration."""
        return self._config

    def evaluate(
        self,
        *,
        request_id: str,
        current_model_id: str,
        current_provider: str,
        current_attempt: int,
        quality_decision: QualityDecision,
    ) -> EscalationPolicyResult:
        """
        Evaluate the escalation policy for the given decision context.

        Args:
            request_id:        Original LLMRequest.request_id.
            current_model_id:  Model that produced the evaluated response.
            current_provider:  Provider that served the response.
            current_attempt:   1-indexed attempt number (initial = 1).
            quality_decision:  PASS, FAIL, or ESCALATE from Phase 3.

        Returns:
            EscalationPolicyResult with the structured outcome.

        Raises:
            ValueError: If current_attempt < 1 or current_attempt >
                        max_total_attempts.
            TypeError:  If quality_decision is not a QualityDecision.
        """
        # ── Input validation ─────────────────────────────────────────
        if not isinstance(quality_decision, QualityDecision):
            raise TypeError(
                f"quality_decision must be a QualityDecision enum member, "
                f"got {type(quality_decision).__name__}."
            )

        if not isinstance(current_attempt, int) or isinstance(current_attempt, bool):
            raise TypeError(
                f"current_attempt must be an int, "
                f"got {type(current_attempt).__name__}."
            )

        if current_attempt < 1:
            raise ValueError(
                f"current_attempt must be >= 1 (1-indexed), "
                f"got {current_attempt}."
            )

        if current_attempt > self._config.max_total_attempts:
            raise ValueError(
                f"current_attempt ({current_attempt}) exceeds "
                f"max_total_attempts ({self._config.max_total_attempts})."
            )

        # ── Common result fields ─────────────────────────────────────
        def _result(
            outcome: EscalationOutcome,
            reason: str,
            is_terminal: bool,
            next_step_intent: NextStepIntent,
        ) -> EscalationPolicyResult:
            return EscalationPolicyResult(
                request_id=request_id,
                current_model_id=current_model_id,
                current_provider=current_provider,
                current_attempt=current_attempt,
                max_total_attempts=self._config.max_total_attempts,
                max_escalation_attempts=self._config.max_escalation_attempts,
                quality_decision=quality_decision,
                escalation_enabled=self._config.escalation_enabled,
                outcome=outcome,
                reason=reason,
                is_terminal=is_terminal,
                next_step_intent=next_step_intent,
            )

        # ── Rule 1: PASS — accept the response ──────────────────────
        if quality_decision == QualityDecision.PASS:
            return _result(
                outcome=EscalationOutcome.ACCEPTED,
                reason=(
                    f"Response from '{current_model_id}' passed quality "
                    f"evaluation on attempt {current_attempt}."
                ),
                is_terminal=True,
                next_step_intent=NextStepIntent.RETURN_RESPONSE,
            )

        # ── Rule 2: FAIL — terminal, no automatic retry ─────────────
        if quality_decision == QualityDecision.FAIL:
            return _result(
                outcome=EscalationOutcome.TERMINAL_FAILURE,
                reason=(
                    f"Response from '{current_model_id}' failed quality "
                    f"evaluation on attempt {current_attempt}. "
                    f"The configured failure outcome is a hard stop; "
                    f"automatic retry is not permitted for FAIL decisions."
                ),
                is_terminal=True,
                next_step_intent=NextStepIntent.FAIL_REQUEST,
            )

        # ── Rule 3: ESCALATE — check policy constraints ─────────────
        # quality_decision == QualityDecision.ESCALATE at this point.

        # 3a. Escalation globally disabled
        if not self._config.escalation_enabled:
            return _result(
                outcome=EscalationOutcome.ESCALATION_UNAVAILABLE,
                reason=(
                    f"Escalation requested for '{current_model_id}' on "
                    f"attempt {current_attempt}, but escalation is disabled "
                    f"in the policy configuration."
                ),
                is_terminal=True,
                next_step_intent=NextStepIntent.RETURN_RESPONSE,
            )

        # 3b. Total attempt limit reached
        if current_attempt >= self._config.max_total_attempts:
            return _result(
                outcome=EscalationOutcome.ESCALATION_EXHAUSTED,
                reason=(
                    f"Escalation requested for '{current_model_id}' on "
                    f"attempt {current_attempt}, but the total attempt "
                    f"limit ({self._config.max_total_attempts}) has been "
                    f"reached."
                ),
                is_terminal=True,
                next_step_intent=NextStepIntent.RETURN_RESPONSE,
            )

        # 3c. Escalation-specific attempt limit reached
        # Escalation attempts used = current_attempt - 1 (the first
        # attempt is not an escalation).
        escalation_attempts_used = current_attempt - 1
        if escalation_attempts_used >= self._config.max_escalation_attempts:
            return _result(
                outcome=EscalationOutcome.ESCALATION_EXHAUSTED,
                reason=(
                    f"Escalation requested for '{current_model_id}' on "
                    f"attempt {current_attempt}, but the escalation attempt "
                    f"limit ({self._config.max_escalation_attempts}) has "
                    f"been reached ({escalation_attempts_used} escalation "
                    f"attempt(s) already used)."
                ),
                is_terminal=True,
                next_step_intent=NextStepIntent.RETURN_RESPONSE,
            )

        # 3d. Escalation permitted
        return _result(
            outcome=EscalationOutcome.ESCALATION_REQUESTED,
            reason=(
                f"Escalation requested for '{current_model_id}' on "
                f"attempt {current_attempt}. Policy allows escalation: "
                f"{escalation_attempts_used}/{self._config.max_escalation_attempts} "
                f"escalation attempt(s) used, "
                f"{current_attempt}/{self._config.max_total_attempts} "
                f"total attempt(s) used."
            ),
            is_terminal=False,
            next_step_intent=NextStepIntent.SELECT_STRONGER_MODEL,
        )
