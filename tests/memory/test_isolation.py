"""Memory manager and architectural isolation tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings import FakeEmbeddings

from agent.runtime import AgentRuntime, ExecuteRequest
from memory import InMemoryStorage, MemoryManager
from memory.storage_interface import JsonFileStorage, RedisStorage
from models.schemas import AnswerResult, IntentType

pytestmark = pytest.mark.memory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestMemoryManager:
    def test_session_lifecycle(self):
        storage = InMemoryStorage()
        manager = MemoryManager(storage=storage)
        manager.begin_session("s1")
        assert manager.has_session("s1")
        manager.end_session("s1")
        assert not manager.has_session("s1")

    def test_record_turn(self):
        manager = MemoryManager()
        manager.record_turn("s2", "hi", AnswerResult(answer="hello", intent=IntentType.QA))
        memory = manager.get_session("s2")
        assert memory is not None
        assert memory.turns[0].query == "hi"

    def test_recent_turns_limit(self):
        manager = MemoryManager()
        for index in range(8):
            manager.record_turn(
                "s3",
                f"q{index}",
                AnswerResult(answer=f"a{index}", intent=IntentType.QA),
            )
        memory = manager.get_session("s3")
        assert memory is not None
        assert len(memory.recent_turns(limit=3)) == 3

    def test_runtime_cleanup_on_end_session(
        self, rag_settings, chroma_store, fake_llm, fake_embeddings
    ):
        storage = InMemoryStorage()
        runtime = AgentRuntime(memory_manager=MemoryManager(storage=storage))
        runtime.execute(
            ExecuteRequest(
                query="付款周期是多少",
                session_id="cleanup-test",
                end_session=True,
            ),
            settings=rag_settings,
            store=chroma_store,
            llm=fake_llm,
            embeddings=fake_embeddings,
        )
        assert not storage.exists("cleanup-test")

    def test_reserved_backends_not_implemented(self):
        with pytest.raises(NotImplementedError):
            RedisStorage()
        with pytest.raises(NotImplementedError):
            JsonFileStorage()


class TestMemoryIsolation:
    """Tools and Skills must not import memory — access only via Runtime."""

    @pytest.mark.parametrize(
        "package",
        ["tools", "skills"],
    )
    def test_packages_do_not_import_memory(self, package: str):
        violations: list[str] = []
        root = PROJECT_ROOT / package
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "memory" or alias.name.startswith("memory."):
                            violations.append(f"{path}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "memory" or node.module.startswith("memory."):
                        violations.append(f"{path}: from {node.module}")
        assert not violations, "Memory imports found:\n" + "\n".join(violations)

    def test_separate_sessions_isolated(self):
        manager = MemoryManager()
        manager.record_turn(
            "session-a",
            "q1",
            AnswerResult(answer="a1", intent=IntentType.QA),
        )
        manager.record_turn(
            "session-b",
            "q2",
            AnswerResult(answer="a2", intent=IntentType.SUMMARIZE),
        )
        mem_a = manager.get_session("session-a")
        mem_b = manager.get_session("session-b")
        assert mem_a is not None and mem_b is not None
        assert mem_a.turns[0].query == "q1"
        assert mem_b.turns[0].query == "q2"
        assert mem_a.turns[0].intent == IntentType.QA
        assert mem_b.turns[0].intent == IntentType.SUMMARIZE
