"""
benchmarks/runner.py
─────────────────────
Benchmark runner for Phase 1 — Unified Model Interface.

WHAT THIS DOES:
    - Loads benchmarks/prompts.json.
    - Executes every prompt against a single, explicitly specified model/provider.
    - Records per-prompt results: tokens, latency, cost, success/failure.
    - Saves a timestamped JSON result file to benchmarks/results/.

WHAT THIS DOES NOT DO:
    - Select models automatically.
    - Route, classify, or evaluate prompts.
    - Compare models or rank results.
    - Call multiple models.
    - Optimize cost or escalate failures.

USAGE (from the project root):

    python -m benchmarks.runner \\
        --model-id "openai/gpt-oss-20b:free" \\
        --provider openrouter

    python -m benchmarks.runner \\
        --model-id "gemma3:4b" \\
        --provider ollama

Optional flags:
    --prompts-file     Path to prompts JSON (default: benchmarks/prompts.json)
    --results-dir      Path to results directory (default: benchmarks/results)
    --max-tokens       Max output tokens per prompt (default: 512)
    --temperature      Sampling temperature (default: 0.2)
    --timeout          Provider timeout in seconds (default: 60)

Result file: benchmarks/results/<provider>_<model>_<timestamp>.json
"""

import argparse
import asyncio
import json
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.config import settings
from app.logging.logger import get_logger
from app.models.config import ModelConfig, QualityTier
from app.models.registry import ModelRegistry
from app.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_PROMPTS_FILE = pathlib.Path("benchmarks/prompts.json")
_DEFAULT_RESULTS_DIR = pathlib.Path("benchmarks/results")
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_TIMEOUT_S = 60.0


# ── Result dataclass (plain dict — no external dep needed) ────────────────────

def _empty_prompt_result(
    prompt_id: str,
    tier: str,
    task_type: str,
    model_id: str,
    provider: str,
) -> dict[str, Any]:
    """Return a zero-valued result record for a single prompt."""
    return {
        "prompt_id": prompt_id,
        "tier": tier,
        "task_type": task_type,
        "model_id": model_id,
        "provider": provider,
        "success": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "latency_ms": 0.0,
        "finish_reason": None,
        "error": None,
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_prompts(prompts_file: pathlib.Path) -> list[dict[str, Any]]:
    """
    Load and validate the benchmark prompt dataset from a JSON file.

    Args:
        prompts_file: Path to prompts.json.

    Returns:
        List of prompt entry dicts.

    Raises:
        FileNotFoundError: If the prompts file does not exist.
        ValueError: If the JSON structure is invalid.
    """
    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Prompts file not found: '{prompts_file}'. "
            "Ensure benchmarks/prompts.json exists."
        )

    raw = prompts_file.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in prompts file: {exc}") from exc

    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts.json must contain a non-empty 'prompts' list.")

    return prompts


# ── Provider factory ──────────────────────────────────────────────────────────

def build_provider(model_config: ModelConfig, timeout_s: float) -> LLMProvider:
    """
    Instantiate the correct provider for the given ModelConfig.

    Centralises provider construction so the runner never imports provider
    classes conditionally in the main body.

    Args:
        model_config: The model to run.
        timeout_s:    HTTP request timeout.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If the provider name is not supported.
    """
    provider_name = model_config.provider

    if provider_name == "openrouter":
        from app.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(model_config=model_config, timeout_s=timeout_s)

    if provider_name == "ollama":
        from app.providers.ollama import OllamaProvider
        return OllamaProvider(model_config=model_config, timeout_s=timeout_s)

    raise ValueError(
        f"Provider '{provider_name}' is not supported by the benchmark runner. "
        "Add it to build_provider() in benchmarks/runner.py."
    )


# ── Single-prompt execution ───────────────────────────────────────────────────

async def run_single_prompt(
    prompt_entry: dict[str, Any],
    provider: LLMProvider,
    model_config: ModelConfig,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """
    Execute one benchmark prompt and return a result record.

    Never raises — failures are caught and recorded in the result.

    Args:
        prompt_entry:  One entry from prompts.json.
        provider:      Instantiated LLMProvider.
        model_config:  ModelConfig for the target model.
        max_tokens:    Maximum output tokens.
        temperature:   Sampling temperature.

    Returns:
        A result dict with all recorded benchmark fields.
    """
    prompt_id = prompt_entry.get("id", "unknown")
    tier = prompt_entry.get("tier", "unknown")
    task_type = prompt_entry.get("task_type", "unknown")
    prompt_text = prompt_entry.get("prompt", "")

    result = _empty_prompt_result(
        prompt_id=prompt_id,
        tier=tier,
        task_type=task_type,
        model_id=model_config.model_id,
        provider=model_config.provider,
    )

    if not prompt_text:
        result["error"] = "Empty prompt text in dataset."
        return result

    request = LLMRequest(
        prompt=prompt_text,
        model_id=model_config.model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        metadata={"tier": tier, "task_type": task_type, "benchmark": True},
    )

    try:
        response: LLMResponse = await provider.generate(request)

        result["success"] = response.is_success
        result["input_tokens"] = response.input_tokens
        result["output_tokens"] = response.output_tokens
        result["total_tokens"] = response.total_tokens
        result["estimated_cost_usd"] = response.estimated_cost
        result["latency_ms"] = response.latency_ms
        result["finish_reason"] = response.finish_reason
        result["error"] = response.error  # None on success

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Benchmark prompt failed",
            extra={
                "prompt_id": prompt_id,
                "tier": tier,
                "model_id": model_config.model_id,
                "provider": model_config.provider,
                "error": str(exc),
            },
        )

    return result


# ── Benchmark summary ─────────────────────────────────────────────────────────

def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics over all prompt results."""
    total = len(results)
    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    latencies = [r["latency_ms"] for r in succeeded if r.get("latency_ms", 0) > 0]
    costs = [r["estimated_cost_usd"] for r in succeeded]

    return {
        "total_prompts": total,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "success_rate": round(len(succeeded) / total, 4) if total else 0.0,
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_estimated_cost_usd": round(sum(costs), 8),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "by_tier": _tier_summary(results),
    }


def _tier_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tier aggregation of success counts."""
    tiers: dict[str, dict[str, int]] = {}
    for r in results:
        tier = r.get("tier", "unknown")
        if tier not in tiers:
            tiers[tier] = {"total": 0, "succeeded": 0, "failed": 0}
        tiers[tier]["total"] += 1
        if r.get("success"):
            tiers[tier]["succeeded"] += 1
        else:
            tiers[tier]["failed"] += 1
    return tiers


# ── Result persistence ────────────────────────────────────────────────────────

def _safe_filename_part(text: str) -> str:
    """Strip characters unsafe for filenames, replacing them with underscores."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", text)


def save_results(
    results: list[dict[str, Any]],
    model_id: str,
    provider: str,
    results_dir: pathlib.Path,
    started_at: datetime,
    finished_at: datetime,
) -> pathlib.Path:
    """
    Persist the benchmark results as a timestamped JSON file.

    Args:
        results:      List of per-prompt result dicts.
        model_id:     Model that was benchmarked.
        provider:     Provider name.
        results_dir:  Directory to write into.
        started_at:   UTC datetime when the run started.
        finished_at:  UTC datetime when the run completed.

    Returns:
        Path of the written file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = started_at.strftime("%Y%m%d_%H%M%S")
    safe_provider = _safe_filename_part(provider)
    safe_model = _safe_filename_part(model_id)
    filename = f"{safe_provider}_{safe_model}_{ts}.json"
    output_path = results_dir / filename

    summary = _compute_summary(results)
    duration_s = (finished_at - started_at).total_seconds()

    payload = {
        "_meta": {
            "benchmark_version": "1.0",
            "phase": "Phase 1 — Unified Model Interface",
            "model_id": model_id,
            "provider": provider,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round(duration_s, 2),
            "note": (
                "Results contain estimated/simulated cost values "
                "for free models. No real cost was incurred for free-tier models."
            ),
        },
        "summary": summary,
        "results": results,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


# ── Main runner ───────────────────────────────────────────────────────────────

async def run_benchmark(
    model_id: str,
    provider_name: str,
    prompts_file: pathlib.Path = _DEFAULT_PROMPTS_FILE,
    results_dir: pathlib.Path = _DEFAULT_RESULTS_DIR,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> pathlib.Path:
    """
    Execute the full benchmark run.

    Args:
        model_id:      Model identifier to benchmark.
        provider_name: Provider to use ('openrouter' or 'ollama').
        prompts_file:  Path to prompts.json.
        results_dir:   Directory for output files.
        max_tokens:    Max output tokens per prompt.
        temperature:   Sampling temperature.
        timeout_s:     Per-request timeout in seconds.

    Returns:
        Path to the saved result JSON file.
    """
    logger.info(
        "Benchmark run starting",
        extra={"model_id": model_id, "provider": provider_name},
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    prompts = load_prompts(prompts_file)
    logger.info("Dataset loaded", extra={"prompt_count": len(prompts)})

    # ── Load model config from registry ───────────────────────────────────────
    registry = ModelRegistry(settings.models_config_path)
    model_config = registry.get(model_id)

    # ── Build provider ────────────────────────────────────────────────────────
    provider = build_provider(model_config, timeout_s=timeout_s)

    # ── Execute prompts ───────────────────────────────────────────────────────
    started_at = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []

    for i, prompt_entry in enumerate(prompts, start=1):
        prompt_id = prompt_entry.get("id", f"p{i}")
        logger.info(
            "Running prompt",
            extra={
                "progress": f"{i}/{len(prompts)}",
                "prompt_id": prompt_id,
                "tier": prompt_entry.get("tier"),
            },
        )

        result = await run_single_prompt(
            prompt_entry=prompt_entry,
            provider=provider,
            model_config=model_config,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        results.append(result)

    finished_at = datetime.now(timezone.utc)

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = save_results(
        results=results,
        model_id=model_id,
        provider=provider_name,
        results_dir=results_dir,
        started_at=started_at,
        finished_at=finished_at,
    )

    summary = _compute_summary(results)
    logger.info(
        "Benchmark run complete",
        extra={
            "output_file": str(output_path),
            "total": summary["total_prompts"],
            "succeeded": summary["succeeded"],
            "failed": summary["failed"],
        },
    )
    return output_path


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RouteMind_AI Phase 1 Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID to benchmark (must exist in config/models.yaml).",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openrouter", "ollama"],
        help="Provider to use.",
    )
    parser.add_argument(
        "--prompts-file",
        default=str(_DEFAULT_PROMPTS_FILE),
        help=f"Path to prompts JSON (default: {_DEFAULT_PROMPTS_FILE}).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(_DEFAULT_RESULTS_DIR),
        help=f"Directory for results (default: {_DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_DEFAULT_MAX_TOKENS,
        help=f"Max output tokens per prompt (default: {_DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {_DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_S,
        help=f"Per-request timeout in seconds (default: {_DEFAULT_TIMEOUT_S}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    output_path = asyncio.run(
        run_benchmark(
            model_id=args.model_id,
            provider_name=args.provider,
            prompts_file=pathlib.Path(args.prompts_file),
            results_dir=pathlib.Path(args.results_dir),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
        )
    )
    print(f"\nBenchmark complete. Results saved to:\n  {output_path}")


if __name__ == "__main__":
    main()
