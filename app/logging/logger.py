"""
app/logging/logger.py
──────────────────────
Structured JSON logging for RouteMind_AI.

Every inference event, routing decision, error, and escalation should be
logged using this module. Structured JSON format ensures logs are:
  - Machine-parseable for future log aggregation.
  - Traceable via request_id.
  - Safe (no raw API keys, no full prompts unless explicitly enabled).

Phase 1 scope:
    - get_logger() factory using standard Python logging.
    - log_inference_event() for structured per-request logs.
    - Safe prompt hashing for log records.

Usage:
    from app.logging.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Provider called", extra={"request_id": "...", "model": "..."})
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_object: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields passed via extra={...}
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                log_object[key] = value

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_object, default=str)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger configured with JSON output to stdout.

    Args:
        name:  Module name (pass __name__ from the calling module).
        level: Optional override log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured Logger instance.
    """
    from app.core.config import settings

    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — return existing logger to avoid duplicate handlers.
        return logger

    effective_level = level or settings.log_level
    logger.setLevel(effective_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


# ── Structured Inference Event Logging ────────────────────────────────────────

def hash_prompt(prompt: str) -> str:
    """
    Return a SHA-256 hex digest of the prompt.

    Used instead of storing raw prompt text in logs and database records,
    preserving privacy while still allowing deduplication.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def log_inference_event(
    logger: logging.Logger,
    request_id: str,
    model_id: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    estimated_cost: float,
    latency_ms: float,
    finish_reason: Optional[str] = None,
    error: Optional[str] = None,
    prompt_hash: Optional[str] = None,
    **extra: Any,
) -> None:
    """
    Emit a structured JSON log record for a single inference call.

    Args:
        logger:         The module logger from get_logger().
        request_id:     Unique request identifier.
        model_id:       Model used for inference.
        provider:       Provider that served the request.
        input_tokens:   Prompt token count.
        output_tokens:  Generated token count.
        total_tokens:   Total token count.
        estimated_cost: Estimated USD cost.
        latency_ms:     Round-trip latency in milliseconds.
        finish_reason:  Provider stop reason.
        error:          Error message if request failed.
        prompt_hash:    SHA-256 hash of the prompt (not the raw prompt).
        **extra:        Any additional structured fields to include.
    """
    event: dict[str, Any] = {
        "event": "inference",
        "request_id": request_id,
        "model_id": model_id,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "error": error,
        "prompt_hash": prompt_hash,
        **extra,
    }

    if error:
        logger.error("Inference failed", extra=event)
    else:
        logger.info("Inference completed", extra=event)
