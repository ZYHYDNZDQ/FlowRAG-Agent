"""Markdown report builder for evaluation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from evaluation.datasets.schema import EvalBenchmark
from evaluation.metrics.generation import GenerationEvalSummary
from evaluation.metrics.retrieval import RetrievalEvalSummary


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_markdown_report(
    *,
    benchmark: EvalBenchmark,
    retrieval: RetrievalEvalSummary,
    generation: GenerationEvalSummary | None,
    run_id: str,
    work_dir: Path,
    used_fake_models: bool,
) -> str:
    """Render evaluation results as Markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        f"# FlowRAG-Agent Evaluation Report",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Benchmark**: {benchmark.name} — {benchmark.description}",
        f"- **Generated**: {now}",
        f"- **Isolated work dir**: `{work_dir}`",
        f"- **Fake models**: {'yes' if used_fake_models else 'no'}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Retrieval Top-{retrieval.top_k} Hit Rate | **{_pct(retrieval.hit_rate)}** ({retrieval.hits}/{retrieval.total}) |",
    ]

    if generation is not None:
        lines.extend(
            [
                f"| Generation Pass Rate (score ≥ 4) | **{_pct(generation.pass_rate)}** ({generation.passed}/{generation.total}) |",
                f"| Generation Average Judge Score | **{generation.average_score:.2f}** / 5.0 |",
            ]
        )
    else:
        lines.append("| Generation | skipped |")

    lines.extend(["", "## Retrieval Details", ""])
    lines.extend(
        [
            "| Case ID | Query | Expected Page | Hit | Top-k Pages |",
            "|---------|-------|---------------|-----|-------------|",
        ]
    )
    for row in retrieval.case_results:
        hit = "✅" if row.hit else "❌"
        pages = ", ".join(str(p) for p in row.top_k_pages) or "—"
        lines.append(
            f"| `{row.case_id}` | {row.query} | {row.expected_page} | {hit} | {pages} |"
        )

    if generation is not None:
        lines.extend(["", "## Generation Details (LLM-as-Judge)", ""])
        lines.extend(
            [
                "| Case ID | Query | Score | Pass | Citations |",
                "|---------|-------|-------|------|-----------|",
            ]
        )
        for row in generation.case_results:
            passed = "✅" if row.verdict.passed else "❌"
            lines.append(
                f"| `{row.case_id}` | {row.query} | {row.verdict.score}/5 | {passed} | {row.citation_count} |"
            )

        lines.extend(["", "### Judge Reasoning", ""])
        for row in generation.case_results:
            lines.extend(
                [
                    f"#### `{row.case_id}`",
                    "",
                    f"**Model answer:** {row.model_answer}",
                    "",
                    f"**Judge:** {row.verdict.reasoning}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Notes",
            "",
            "- Evaluation uses an **isolated temporary knowledge base**; production `data/` is not modified.",
            "- Retrieval hit = expected page **or** keyword appears in Top-k retrieved chunks.",
            "- Generation pass threshold: LLM judge score ≥ 4.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
