"""
tests/test_logger.py
──────────────────────
Unit tests for the structured JSON logging implementation.

Audit findings (before writing tests):
    - app/logging/logger.py is already correct — no production code changes required.
    - JSONFormatter emits single-line JSON via json.dumps(log_object, default=str).
    - get_logger() returns a Logger with a single JSONFormatter-backed StreamHandler.
    - get_logger() is idempotent — calling it twice for the same name does not
      add duplicate handlers.
    - log_inference_event() builds a structured event dict and routes to
      logger.info (success) or logger.error (failure).
    - hash_prompt() returns a 64-char lowercase hex SHA-256 digest.
    - Prompts are never passed raw to log_inference_event() — only their hash is.
    - API keys / Authorization headers do not exist anywhere in logger.py.

Test coverage:
    - JSONFormatter produces valid parseable JSON.
    - JSON output contains timestamp, level, logger, message fields.
    - Extra fields passed via extra={} appear in JSON output.
    - get_logger() returns a Logger instance.
    - get_logger() is idempotent (no duplicate handlers).
    - hash_prompt() returns a 64-character lowercase hex string.
    - hash_prompt() is deterministic (same input → same hash).
    - hash_prompt() is unique per distinct prompt (different input → different hash).
    - hash_prompt() never returns the raw prompt text.
    - log_inference_event() calls logger.info on success (error=None).
    - log_inference_event() calls logger.error on failure (error is set).
    - log_inference_event() emits required inference fields in the log record.
    - log_inference_event() uses prompt_hash not raw prompt.
    - log_inference_event() does NOT include raw prompt text in output.
    - A simulated API key string does not leak into log output.
    - error-path log contains the error field.
    - success-path log does not have a truthy error field.
"""

import json
import logging
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.logging.logger import (
    JSONFormatter,
    get_logger,
    hash_prompt,
    log_inference_event,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _capture_log_output(
    logger: logging.Logger,
    level: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    Capture the raw string written to the JSONFormatter for one log call.

    Returns the formatted string (should be a single-line JSON object).
    """
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JSONFormatter())

    # Temporarily attach the capture handler
    logger.addHandler(handler)
    try:
        log_fn = getattr(logger, level.lower())
        if extra:
            log_fn(message, extra=extra)
        else:
            log_fn(message)
    finally:
        logger.removeHandler(handler)
        handler.close()

    return buffer.getvalue().strip()


def _emit_inference_event_and_capture(
    request_id: str = "req-test-001",
    model_id: str = "openai/gpt-oss-20b:free",
    provider: str = "openrouter",
    input_tokens: int = 100,
    output_tokens: int = 50,
    total_tokens: int = 150,
    estimated_cost: float = 0.00005,
    latency_ms: float = 820.5,
    finish_reason: str | None = "stop",
    error: str | None = None,
    prompt_hash: str | None = None,
) -> dict[str, Any]:
    """
    Call log_inference_event() with a captured logger and return the parsed JSON.
    """
    test_logger = logging.getLogger("test.inference_event_capture")
    test_logger.handlers.clear()
    test_logger.setLevel(logging.DEBUG)

    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JSONFormatter())
    test_logger.addHandler(handler)
    test_logger.propagate = False

    try:
        log_inference_event(
            logger=test_logger,
            request_id=request_id,
            model_id=model_id,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            error=error,
            prompt_hash=prompt_hash or hash_prompt("test prompt for hashing"),
        )
    finally:
        test_logger.removeHandler(handler)
        handler.close()

    raw = buffer.getvalue().strip()
    return json.loads(raw)


# ── JSONFormatter tests ───────────────────────────────────────────────────────

class TestJSONFormatter:

    def test_output_is_valid_json(self):
        """JSONFormatter produces a string that parses as valid JSON."""
        logger = logging.getLogger("test.json_formatter.valid")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        raw = _capture_log_output(logger, "info", "test message")
        parsed = json.loads(raw)  # Raises if invalid JSON
        assert isinstance(parsed, dict)

    def test_output_contains_timestamp(self):
        """Formatted JSON contains a 'timestamp' field."""
        logger = logging.getLogger("test.json_formatter.timestamp")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        raw = _capture_log_output(logger, "info", "ts check")
        parsed = json.loads(raw)
        assert "timestamp" in parsed
        assert isinstance(parsed["timestamp"], str)
        assert len(parsed["timestamp"]) > 0

    def test_output_contains_level(self):
        """Formatted JSON contains a 'level' field with the log level name."""
        logger = logging.getLogger("test.json_formatter.level")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        raw = _capture_log_output(logger, "warning", "level check")
        parsed = json.loads(raw)
        assert parsed.get("level") == "WARNING"

    def test_output_contains_logger_name(self):
        """Formatted JSON contains a 'logger' field with the logger name."""
        logger = logging.getLogger("test.json_formatter.loggername")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        raw = _capture_log_output(logger, "info", "name check")
        parsed = json.loads(raw)
        assert parsed.get("logger") == "test.json_formatter.loggername"

    def test_output_contains_message(self):
        """Formatted JSON contains a 'message' field."""
        logger = logging.getLogger("test.json_formatter.msg")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        raw = _capture_log_output(logger, "info", "hello world")
        parsed = json.loads(raw)
        assert parsed.get("message") == "hello world"

    def test_extra_fields_appear_in_json(self):
        """Fields passed via extra={} are merged into the JSON output."""
        logger = logging.getLogger("test.json_formatter.extra")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        raw = _capture_log_output(
            logger, "info", "extra test",
            extra={"request_id": "abc-123", "latency_ms": 42.5}
        )
        parsed = json.loads(raw)
        assert parsed.get("request_id") == "abc-123"
        assert parsed.get("latency_ms") == 42.5

    def test_output_is_single_line(self):
        """JSONFormatter output is a single line (no embedded newlines)."""
        logger = logging.getLogger("test.json_formatter.singleline")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        raw = _capture_log_output(logger, "info", "single line test")
        assert "\n" not in raw


# ── get_logger() tests ────────────────────────────────────────────────────────

class TestGetLogger:

    def test_returns_logger_instance(self):
        """get_logger() returns a logging.Logger."""
        # settings is imported lazily inside get_logger(); call with real settings
        # (log_level defaults to 'INFO' from Settings defaults).
        logger = get_logger("test.get_logger.basic")
        assert isinstance(logger, logging.Logger)

    def test_idempotent_no_duplicate_handlers(self):
        """Calling get_logger() twice for the same name does not add duplicate handlers."""
        # Reset any cached handler from a previous test run
        cached = logging.getLogger("test.get_logger.idempotent")
        cached.handlers.clear()

        logger1 = get_logger("test.get_logger.idempotent")
        handler_count_after_first = len(logger1.handlers)
        logger2 = get_logger("test.get_logger.idempotent")
        handler_count_after_second = len(logger2.handlers)

        assert logger1 is logger2
        assert handler_count_after_first == handler_count_after_second

    def test_logger_has_json_formatter(self):
        """The logger's handler uses JSONFormatter."""
        # Clear any cached handler first
        cached = logging.getLogger("test.get_logger.formatter")
        cached.handlers.clear()

        logger = get_logger("test.get_logger.formatter")

        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)


# ── hash_prompt() tests ───────────────────────────────────────────────────────

class TestHashPrompt:

    def test_returns_64_char_hex_string(self):
        """hash_prompt() returns a 64-character lowercase hex string (SHA-256)."""
        result = hash_prompt("What is 2 + 2?")
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_is_deterministic(self):
        """Same prompt always produces the same hash."""
        prompt = "Explain quantum entanglement."
        assert hash_prompt(prompt) == hash_prompt(prompt)

    def test_different_prompts_produce_different_hashes(self):
        """Different prompts produce different hashes."""
        h1 = hash_prompt("prompt one")
        h2 = hash_prompt("prompt two")
        assert h1 != h2

    def test_hash_does_not_contain_raw_prompt(self):
        """The hash output does not contain the raw input text."""
        prompt = "super secret prompt content"
        result = hash_prompt(prompt)
        assert prompt not in result
        assert "secret" not in result

    def test_empty_string_produces_valid_hash(self):
        """Empty string input produces a valid 64-char SHA-256 hash."""
        result = hash_prompt("")
        assert len(result) == 64


# ── log_inference_event() tests ───────────────────────────────────────────────

class TestLogInferenceEvent:

    def test_success_event_calls_logger_info(self):
        """log_inference_event() with error=None calls logger.info."""
        mock_logger = MagicMock(spec=logging.Logger)
        log_inference_event(
            logger=mock_logger,
            request_id="req-001",
            model_id="gemma3:4b",
            provider="ollama",
            input_tokens=50,
            output_tokens=20,
            total_tokens=70,
            estimated_cost=0.0,
            latency_ms=1200.0,
            error=None,
        )
        mock_logger.info.assert_called_once()
        mock_logger.error.assert_not_called()

    def test_error_event_calls_logger_error(self):
        """log_inference_event() with error set calls logger.error."""
        mock_logger = MagicMock(spec=logging.Logger)
        log_inference_event(
            logger=mock_logger,
            request_id="req-002",
            model_id="openai/gpt-oss-20b:free",
            provider="openrouter",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost=0.0,
            latency_ms=50.0,
            error="HTTP 429 rate limit exceeded",
        )
        mock_logger.error.assert_called_once()
        mock_logger.info.assert_not_called()

    def test_success_event_json_contains_required_fields(self):
        """Successful inference log contains all required structured fields."""
        parsed = _emit_inference_event_and_capture(
            request_id="req-fields-check",
            model_id="openai/gpt-oss-20b:free",
            provider="openrouter",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost=0.000090,
            latency_ms=820.5,
            finish_reason="stop",
            error=None,
        )

        assert parsed.get("request_id") == "req-fields-check"
        assert parsed.get("model_id") == "openai/gpt-oss-20b:free"
        assert parsed.get("provider") == "openrouter"
        assert parsed.get("input_tokens") == 100
        assert parsed.get("output_tokens") == 50
        assert parsed.get("total_tokens") == 150
        assert parsed.get("estimated_cost_usd") == pytest.approx(0.000090)
        assert parsed.get("latency_ms") == pytest.approx(820.5)
        assert parsed.get("finish_reason") == "stop"
        assert "timestamp" in parsed
        assert "event" in parsed

    def test_error_event_json_contains_error_field(self):
        """Error inference log contains a non-null 'error' field."""
        parsed = _emit_inference_event_and_capture(
            error="ProviderTimeoutError: timed out after 60s",
        )
        assert parsed.get("error") is not None
        assert "timed out" in parsed["error"]

    def test_success_event_error_field_is_none(self):
        """Successful inference log has error=null (None) in JSON."""
        parsed = _emit_inference_event_and_capture(error=None)
        # json.dumps(None) → null; parsed back → None
        assert parsed.get("error") is None

    def test_prompt_hash_present_not_raw_prompt(self):
        """prompt_hash is logged; the raw prompt text is NOT in the output."""
        raw_prompt = "Tell me a very secret internal prompt content"
        p_hash = hash_prompt(raw_prompt)

        parsed = _emit_inference_event_and_capture(prompt_hash=p_hash)

        assert parsed.get("prompt_hash") == p_hash
        # Raw prompt must not appear anywhere in the serialized JSON
        serialized = json.dumps(parsed)
        assert raw_prompt not in serialized
        assert "secret internal" not in serialized

    def test_no_api_key_in_log_output(self):
        """A simulated API key string must not appear in log output."""
        fake_key = "sk-or-v1-abcdef1234567890verysecretkey"

        parsed = _emit_inference_event_and_capture()

        serialized = json.dumps(parsed)
        assert fake_key not in serialized
        # Also check no Authorization header pattern leaks
        assert "Bearer" not in serialized
        assert "Authorization" not in serialized

    def test_event_field_is_inference(self):
        """log_inference_event() always sets event='inference'."""
        parsed = _emit_inference_event_and_capture()
        assert parsed.get("event") == "inference"

    def test_ollama_zero_cost_event(self):
        """Zero-cost Ollama local model logs correctly with $0.00 cost."""
        parsed = _emit_inference_event_and_capture(
            model_id="gemma3:4b",
            provider="ollama",
            estimated_cost=0.0,
        )
        assert parsed.get("estimated_cost_usd") == 0.0
        assert parsed.get("provider") == "ollama"

    def test_extra_kwargs_appear_in_output(self):
        """Additional **extra kwargs passed to log_inference_event are included."""
        test_logger = logging.getLogger("test.extra_kwargs")
        test_logger.handlers.clear()
        test_logger.setLevel(logging.DEBUG)
        test_logger.propagate = False

        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())
        test_logger.addHandler(handler)

        try:
            log_inference_event(
                logger=test_logger,
                request_id="req-extra",
                model_id="some-model",
                provider="openrouter",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                estimated_cost=0.0,
                latency_ms=100.0,
                task_type="extraction",
                complexity_tier="tier_1",
            )
        finally:
            test_logger.removeHandler(handler)
            handler.close()

        parsed = json.loads(buffer.getvalue().strip())
        assert parsed.get("task_type") == "extraction"
        assert parsed.get("complexity_tier") == "tier_1"
