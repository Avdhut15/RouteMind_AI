"""
benchmarks/phase3_runner.py
───────────────────────────
Phase 3 Part 6 — Quality Evaluation Validation & Benchmarking.

Runs the synthetic Phase 3 evaluation dataset through the complete evaluation
pipeline and generates a structured benchmark report.

This runner uses the DeterministicEvaluator by default to ensure offline,
reproducible benchmarking without requiring API keys or incurring costs.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.deterministic import DeterministicEvaluator
from app.evaluation.pipeline import EvaluationPipeline
from app.providers.base import LLMRequest, LLMResponse


def load_dataset(path: Path) -> dict[str, Any]:
    """Load the evaluation dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


async def run_benchmark(dataset_path: Path, output_dir: Path) -> Path:
    """
    Execute the evaluation benchmark and write results to disk.
    
    Args:
        dataset_path: Path to phase3_eval_dataset.json.
        output_dir: Directory where results should be saved.
        
    Returns:
        Path to the generated benchmark results file.
    """
    dataset = load_dataset(dataset_path)
    cases = dataset.get("cases", [])
    
    # Initialize the pipeline using the DeterministicEvaluator.
    # This guarantees offline, reproducible validation for the benchmark.
    evaluator = DeterministicEvaluator()
    pipeline = EvaluationPipeline(evaluator=evaluator)
    
    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0
    decision_counts = {"PASS": 0, "FAIL": 0, "ESCALATE": 0}
    
    start_time = time.perf_counter()
    
    for case in cases:
        req_data = case["request"]
        resp_data = case["response"]
        
        # Reconstruct internal models
        request = LLMRequest(
            request_id=case["id"],
            prompt=req_data["prompt"],
            model_id=req_data.get("model_id", "auto"),
        )
        response = LLMResponse(
            request_id=case["id"],
            model_id=resp_data.get("model_id", "unknown"),
            provider=resp_data.get("provider", "unknown"),
            output=resp_data.get("output", ""),
            finish_reason=resp_data.get("finish_reason", "stop"),
        )
        
        # Execute pipeline
        t0 = time.perf_counter()
        pipeline_result = await pipeline.run(request, response)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        # Track statistics
        status_name = pipeline_result.evaluation_status.name
        if status_name == "SUCCESS":
            success_count += 1
        elif status_name == "FAILURE":
            fail_count += 1
        elif status_name == "SKIPPED":
            skip_count += 1
            
        decision_name = pipeline_result.decision.name
        decision_counts[decision_name] = decision_counts.get(decision_name, 0) + 1
        
        eval_result = pipeline_result.evaluation_result
        
        # Record structured case result
        results.append({
            "case_id": case["id"],
            "category": case["category"],
            "expected_decision": case["expected_decision"],
            "actual_decision": decision_name,
            "decision_match": decision_name.lower() in case["expected_decision"].lower(),
            "evaluation_status": status_name,
            "evaluation_source": pipeline_result.evaluation_source.name,
            "quality_score": pipeline_result.quality_score,
            "individual_scores": eval_result.individual_scores if hasattr(eval_result, "individual_scores") else None,
            "reason": pipeline_result.reason,
            "error": eval_result.error if hasattr(eval_result, "error") else None,
            "latency_ms": round(latency_ms, 2)
        })
        
    total_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Calculate score statistics where meaningful
    scores = [r["quality_score"] for r in results if r["quality_score"] is not None]
    avg_score = sum(scores) / len(scores) if scores else None
    
    summary = {
        "total_cases": len(cases),
        "total_time_ms": round(total_time_ms, 2),
        "evaluator_source": "DETERMINISTIC",
        "status_distribution": {
            "SUCCESS": success_count,
            "FAILURE": fail_count,
            "SKIPPED": skip_count
        },
        "decision_distribution": decision_counts,
        "score_statistics": {
            "count": len(scores),
            "average": round(avg_score, 4) if avg_score is not None else None,
            "min": round(min(scores), 4) if scores else None,
            "max": round(max(scores), 4) if scores else None
        }
    }
    
    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset.get("_meta", {}).get("dataset_version", "unknown"),
            "evaluation_mode": "offline_deterministic"
        },
        "summary": summary,
        "cases": results
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"phase3_benchmark_{timestamp}.json"
    
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    return out_file


def main():
    dataset_path = Path("benchmarks/phase3_eval_dataset.json")
    output_dir = Path("benchmarks/results")
    
    print("Starting Phase 3 Evaluation Benchmark...")
    print(f"Dataset: {dataset_path}")
    
    out_file = asyncio.run(run_benchmark(dataset_path, output_dir))
    
    print(f"\nBenchmark complete. Results saved to:\n{out_file}")
    
    # Print a brief summary
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        summary = data["summary"]
        print("\nSummary:")
        print(f"Total Cases: {summary['total_cases']}")
        print(f"Decisions: {summary['decision_distribution']}")
        print(f"Average Score: {summary['score_statistics']['average']}")


if __name__ == "__main__":
    main()
