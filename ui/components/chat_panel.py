"""
Chat panel — user input and answer display.
"""

from __future__ import annotations

import streamlit as st

from agent.runtime import ExecuteRequest, get_runtime
from ui.components.agent_trace import render_agent_trace
from ui.components.citation_view import render_citations
from ui.state import (
    append_message,
    append_trace_step,
    clear_trace,
    get_messages,
    get_selected_doc_ids,
    get_trace_steps,
)


def render_chat_panel() -> None:
    """Render the main chat interface."""
    for message in get_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                render_citations(message["citations"])
            if message.get("trace"):
                render_agent_trace(message["trace"])

    if prompt := st.chat_input("输入问题或任务（问答 / 总结 / 分析）"):
        _handle_user_message(prompt)


def _handle_user_message(prompt: str) -> None:
    append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    clear_trace()
    selected = get_selected_doc_ids()

    with st.chat_message("assistant"):
        with st.spinner("Agent 执行中…"):
            try:
                execute_result = get_runtime().execute(
                    ExecuteRequest(
                        query=prompt,
                        session_id="streamlit",
                        selected_doc_ids=selected,
                        on_step=append_trace_step,
                    ),
                )
                result = execute_result.answer
            except Exception as exc:
                error_msg = f"Agent 执行失败：{exc}"
                append_message("assistant", error_msg)
                st.error(error_msg)
                render_agent_trace(get_trace_steps())
                return

        append_message(
            "assistant",
            result.answer,
            citations=result.citations,
            trace=result.trace,
        )
        st.markdown(result.answer)
        render_citations(result.citations)
        render_agent_trace(result.trace)
