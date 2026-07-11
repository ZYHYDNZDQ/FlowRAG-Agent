"""Evaluation module tests."""

from __future__ import annotations

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings import FakeEmbeddings

from evaluation.datasets.loader import load_benchmark
from evaluation.harness import build_eval_environment
from evaluation.metrics.generation import _parse_judge_response, judge_answer
from evaluation.metrics.report import build_markdown_report
from evaluation.metrics.retrieval import top_k_hit
from evaluation.runner import run_evaluation
from models.schemas import ChunkMetadata, RetrievedChunk

pytestmark = pytest.mark.unit


def _chunk(page: int, text: str, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            doc_id="d1",
            source_file="t.pdf",
            source_path="data/t.pdf",
            page=page,
            chunk_index=0,
            chunk_id=chunk_id,
        ),
        score=1.0,
    )


def test_top_k_hit_by_page():
    chunks = [_chunk(1, "payment terms", "c1"), _chunk(2, "penalty", "c2")]
    assert top_k_hit(chunks, expected_page=1, keyword=None, top_k=2) is True
    assert top_k_hit(chunks, expected_page=3, keyword=None, top_k=1) is False


def test_top_k_hit_by_keyword_fallback():
    chunks = [_chunk(5, "contains penalty clause", "c1")]
    assert top_k_hit(chunks, expected_page=1, keyword="penalty", top_k=1) is True


def test_parse_judge_response_json():
    verdict = _parse_judge_response('{"score": 5, "pass": true, "reasoning": "good"}')
    assert verdict.score == 5
    assert verdict.passed is True


def test_judge_answer_with_fake_llm():
    judge = FakeListChatModel(
        responses=['{"score": 4, "pass": true, "reasoning": "acceptable"}']
    )
    verdict = judge_answer(
        judge_llm=judge,
        query="q",
        reference_answer="ref",
        criteria="crit",
        model_answer="ans",
    )
    assert verdict.passed is True


def test_offline_evaluation_pipeline(tmp_path):
    from argparse import Namespace

    args = Namespace(
        benchmark="contract",
        output_dir=tmp_path / "reports",
        work_dir=tmp_path / "cache",
        top_k=3,
        skip_generation=False,
        fake=True,
    )
    report_path = run_evaluation(args)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Top-3 Hit Rate" in content
    assert "LLM-as-Judge" in content
    assert "100.0%" in content or "Hit Rate" in content


def test_build_markdown_report_retrieval_only():
    from pathlib import Path

    from evaluation.metrics.retrieval import RetrievalCaseResult, RetrievalEvalSummary

    benchmark = load_benchmark("contract")
    retrieval = RetrievalEvalSummary(
        top_k=3,
        total=1,
        hits=1,
        hit_rate=1.0,
        case_results=[
            RetrievalCaseResult(
                case_id="r1",
                query="q",
                expected_page=1,
                hit=True,
                top_k_pages=[1],
                top_k_chunk_ids=["c1"],
            )
        ],
    )
    md = build_markdown_report(
        benchmark=benchmark,
        retrieval=retrieval,
        generation=None,
        run_id="test-run",
        work_dir=Path("."),
        used_fake_models=True,
    )
    assert "Retrieval Top-3 Hit Rate" in md
    assert "Generation | skipped" in md
