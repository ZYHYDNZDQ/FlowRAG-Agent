"""
Streamlit session state management.

Centralizes keys and accessors for:
  - messages: chat history
  - selected_doc_ids: sidebar document selection
  - trace_steps: current Agent execution trace
  - ingest_jobs: upload progress state

Implementation planned for Day 4.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from models.schemas import AgentStep, DocRecord


# Session state keys
KEY_MESSAGES = "messages"
KEY_SELECTED_DOCS = "selected_doc_ids"
KEY_TRACE_STEPS = "trace_steps"
KEY_DOC_LIST = "doc_list"


def init_session_state() -> None:
    """Initialize default session state on app load."""
    defaults: dict[str, Any] = {
        KEY_MESSAGES: [],
        KEY_SELECTED_DOCS: [],
        KEY_TRACE_STEPS: [],
        KEY_DOC_LIST: [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_messages() -> list[dict[str, str]]:
    return st.session_state.get(KEY_MESSAGES, [])


def append_message(
    role: str,
    content: str,
    *,
    citations: list | None = None,
    trace: list | None = None,
) -> None:
    message: dict[str, Any] = {"role": role, "content": content}
    if citations is not None:
        message["citations"] = citations
    if trace is not None:
        message["trace"] = trace
    st.session_state[KEY_MESSAGES].append(message)


def get_selected_doc_ids() -> list[str]:
    return st.session_state.get(KEY_SELECTED_DOCS, [])


def set_selected_doc_ids(doc_ids: list[str]) -> None:
    st.session_state[KEY_SELECTED_DOCS] = doc_ids


def get_trace_steps() -> list[AgentStep]:
    return st.session_state.get(KEY_TRACE_STEPS, [])


def append_trace_step(step: AgentStep) -> None:
    st.session_state[KEY_TRACE_STEPS].append(step)


def clear_trace() -> None:
    st.session_state[KEY_TRACE_STEPS] = []


def set_doc_list(docs: list[DocRecord]) -> None:
    st.session_state[KEY_DOC_LIST] = docs


def get_doc_list() -> list[DocRecord]:
    return st.session_state.get(KEY_DOC_LIST, [])
