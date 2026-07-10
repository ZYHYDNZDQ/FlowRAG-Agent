"""
FlowRAG-Agent Streamlit entry point.

Layout:
  - Sidebar: upload, document list, settings
  - Main: chat panel + agent trace + citations

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import config.env_bootstrap  # noqa: F401  # must run before huggingface_hub import

import streamlit as st

from config.settings import get_settings
from ui.components.chat_panel import render_chat_panel
from ui.components.upload_panel import render_upload_panel
from ui.state import init_session_state

st.set_page_config(
    page_title="FlowRAG-Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    """Sidebar: branding, upload, document scope."""
    settings = get_settings()
    st.title("FlowRAG-Agent")
    st.caption("本地知识库 · RAG · 动态工作流")
    render_upload_panel()
    st.divider()
    with st.expander("运行配置", expanded=False):
        st.write(f"**Collection**: `{settings.chroma_collection_name}`")
        st.write(f"**Chunk**: {settings.chunk_size} / overlap {settings.chunk_overlap}")
        st.write(f"**LLM**: {settings.llm_provider.value} / {settings.ollama_model}")
        st.write(f"**Embedding**: {settings.embedding_model}")


def render_main() -> None:
    """Main area: chat interface."""
    st.header("对话")
    render_chat_panel()


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    init_session_state()

    with st.sidebar:
        render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
