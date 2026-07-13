"""Evaluate an exported model: CPU latency benchmark (and, in future, test-set metrics).

Usage:
    uv run python scripts/evaluate.py --benchmark-latency [--onnx-model ...]
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from waste_classification.config import settings
from waste_classification.logging_config import setup_logging

logger = logging.getLogger(__name__)

LATENCY_WARMUP_RUNS = 10
LATENCY_BENCHMARK_RUNS = 100
LATENCY_TARGET_MS = 300.0


def benchmark_latency(
    onnx_path: Path,
    image_size: int,
    warmup_runs: int = LATENCY_WARMUP_RUNS,
    benchmark_runs: int = LATENCY_BENCHMARK_RUNS,
) -> dict[str, float]:
    """Benchmark single-image CPU inference latency for an ONNX model.

    Args:
        onnx_path: Path to the exported ONNX model.
        image_size: Square input resolution used to build the dummy input.
        warmup_runs: Number of untimed warmup inferences.
        benchmark_runs: Number of timed inferences.

    Returns:
        Dict with mean_ms, p50_ms, p95_ms.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    for _ in range(warmup_runs):
        session.run(None, {"input": dummy_input})

    durations_ms = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        session.run(None, {"input": dummy_input})
        durations_ms.append((time.perf_counter() - start) * 1000)

    durations_ms.sort()
    return {
        "mean_ms": float(np.mean(durations_ms)),
        "p50_ms": float(np.percentile(durations_ms, 50)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate an exported model")
    parser.add_argument(
        "--benchmark-latency",
        action="store_true",
        help="Run the CPU-only latency benchmark against the exported ONNX model",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Path to the .onnx model (default: models/onnx/<checkpoint stem>.onnx)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=settings.model_image_size,
        help=f"Square input resolution (default: {settings.model_image_size})",
    )
    args = parser.parse_args()

    setup_logging(settings.log_level, settings.log_file)

    if not args.benchmark_latency:
        logger.error("Nothing to do: pass --benchmark-latency")
        return 1

    onnx_path = args.onnx_model or Path("models/onnx") / f"{Path(settings.model_weights).stem}.onnx"
    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path} (run scripts/export.py first)")
        return 1

    logger.info(f"Benchmarking latency for {onnx_path} on CPU")
    logger.info(f"Warmup: {LATENCY_WARMUP_RUNS} runs, benchmark: {LATENCY_BENCHMARK_RUNS} runs")
    results = benchmark_latency(onnx_path, args.image_size)

    logger.info(
        f"Latency (ms) — mean={results['mean_ms']:.2f} "
        f"p50={results['p50_ms']:.2f} p95={results['p95_ms']:.2f}"
    )
    if results["p95_ms"] <= LATENCY_TARGET_MS:
        logger.info(f"✓ p95 latency meets the ≤{LATENCY_TARGET_MS:.0f}ms target")
    else:
        logger.warning(
            f"✗ p95 latency {results['p95_ms']:.2f}ms exceeds the ≤{LATENCY_TARGET_MS:.0f}ms target"
        )

    return 0


if __name__ == "__main__":
    exit(main())
