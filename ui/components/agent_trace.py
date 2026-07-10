"""
Agent trace panel — visualize execution steps.

Example output::

    [Router]
    任务类型: summary

    [Retriever]
    找到 3 个相关文档片段

    [Generator]
    生成最终答案
"""

from __future__ import annotations

import streamlit as st

from models.schemas import AgentStep


def render_agent_trace(steps: list[AgentStep] | None = None) -> None:
    """Render expandable trace of Agent execution."""
    if not steps:
        return

    with st.expander("Agent 执行轨迹", expanded=True):
        for step in steps:
            st.markdown(f"**[{step.name}]**")
            if step.detail:
                st.markdown(step.detail)
            if step.duration_ms is not None:
                st.caption(f"耗时 {step.duration_ms:.0f} ms")
            if step.metadata.get("hits"):
                with st.container(border=True):
                    for hit in step.metadata["hits"]:
                        score = hit.get("score")
                        score_txt = f" (score={score:.2f})" if score is not None else ""
                        st.caption(
                            f"· {hit.get('source_file')} 第{hit.get('page')}页{score_txt}"
                        )
            elif step.metadata and step.name not in ("Retriever",):
                with st.expander("详情", expanded=False):
                    st.json(step.metadata)
            st.divider()
