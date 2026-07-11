"""Load evaluation benchmarks by name."""

from __future__ import annotations

from evaluation.datasets.contract_benchmark import CONTRACT_BENCHMARK
from evaluation.datasets.schema import EvalBenchmark

_BENCHMARKS: dict[str, EvalBenchmark] = {
    CONTRACT_BENCHMARK.name: CONTRACT_BENCHMARK,
}


def load_benchmark(name: str = "contract") -> EvalBenchmark:
    """Return a registered benchmark dataset."""
    try:
        return _BENCHMARKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BENCHMARKS))
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}") from exc


def list_benchmarks() -> list[str]:
    return sorted(_BENCHMARKS)
