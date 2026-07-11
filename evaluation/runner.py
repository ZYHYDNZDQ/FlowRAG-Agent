"""
FlowRAG-Agent evaluation runner.

Independent CLI — does not start Streamlit or modify production ``data/``.

Usage:
    python -m evaluation.runner
    python -m evaluation.runner --fake --benchmark contract
    flowrag-eval --output-dir evaluation/reports
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agent.runtime import AgentRuntime
from evaluation.datasets.fake_responses import FAKE_AGENT_RESPONSES, FAKE_JUDGE_RESPONSES
from evaluation.datasets.loader import list_benchmarks, load_benchmark
from evaluation.harness import build_eval_environment
from evaluation.metrics.generation import evaluate_generation
from evaluation.metrics.report import build_markdown_report, write_report
from evaluation.metrics.retrieval import evaluate_retrieval
from llm.factory import get_embeddings, get_llm


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FlowRAG-Agent offline evaluation and write a Markdown report.",
    )
    parser.add_argument(
        "--benchmark",
        default="contract",
        choices=list_benchmarks(),
        help="Benchmark dataset name (default: contract)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/reports"),
        help="Directory for Markdown reports (default: evaluation/reports)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Isolated KB cache dir (default: <output-dir>/.cache/<run_id>)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval hit rate")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation + LLM-as-judge evaluation",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use FakeEmbeddings/FakeListChatModel (offline, no Ollama/OpenAI required)",
    )
    return parser.parse_args(argv)


def _build_models(use_fake: bool):
    if use_fake:
        from langchain_community.chat_models.fake import FakeListChatModel
        from langchain_community.embeddings import FakeEmbeddings

        embeddings = FakeEmbeddings(size=128)
        llm = FakeListChatModel(responses=list(FAKE_AGENT_RESPONSES))
        judge_llm = FakeListChatModel(responses=list(FAKE_JUDGE_RESPONSES))
        return embeddings, llm, judge_llm

    embeddings = get_embeddings()
    llm = get_llm()
    return embeddings, llm, llm


def run_evaluation(args: argparse.Namespace) -> Path:
    """Execute evaluation pipeline and return report path."""
    benchmark = load_benchmark(args.benchmark)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:6]
    work_dir = args.work_dir or (args.output_dir / ".cache" / run_id)

    embeddings, llm, judge_llm = _build_models(args.fake)
    env = build_eval_environment(benchmark, work_dir=work_dir, embeddings=embeddings)

    retrieval_summary = evaluate_retrieval(
        benchmark.retrieval_cases,
        store=env.store,
        scope=env.scope,
        embeddings=embeddings,
        settings=env.settings,
        top_k=args.top_k,
    )

    generation_summary = None
    if not args.skip_generation:
        runtime = AgentRuntime()
        generation_summary = evaluate_generation(
            benchmark.generation_cases,
            runtime=runtime,
            judge_llm=judge_llm,
            settings=env.settings,
            store=env.store,
            llm=llm,
            embeddings=embeddings,
            scope=env.scope,
            doc_id=env.doc_id,
        )

    report_body = build_markdown_report(
        benchmark=benchmark,
        retrieval=retrieval_summary,
        generation=generation_summary,
        run_id=run_id,
        work_dir=work_dir,
        used_fake_models=args.fake,
    )
    report_path = args.output_dir / f"{run_id}_report.md"
    write_report(report_path, report_body)
    return report_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_path = run_evaluation(args)
    print(f"Evaluation report written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
