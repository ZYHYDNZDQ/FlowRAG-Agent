"""
Upload panel — PDF file upload and ingestion progress.

Wires Streamlit file uploader to ``ingestion.indexer.ingest_uploaded_bytes``.
"""

from __future__ import annotations

import streamlit as st

from ingestion.indexer import ingest_uploaded_bytes
from models.doc_registry import DocRegistry
from models.schemas import DocStatus
from ui.state import set_doc_list, set_selected_doc_ids

_STAGE_LABELS = {
    "parse": "解析 PDF",
    "chunk": "文本切分",
    "embed": "向量化",
    "index": "写入 Chroma",
    "done": "完成",
}


def render_upload_panel() -> None:
    """Render the PDF upload section in the sidebar."""
    st.subheader("上传 PDF")
    uploaded_files = st.file_uploader(
        "选择 PDF 文件",
        type=["pdf"],
        accept_multiple_files=True,
        help="上传后自动执行：解析 → 切分 → Embedding → Chroma 入库",
    )

    if uploaded_files and st.button("开始入库", type="primary", use_container_width=True):
        progress = st.progress(0.0, text="准备中…")
        status = st.empty()

        for index, uploaded in enumerate(uploaded_files, start=1):
            status.info(f"正在处理 ({index}/{len(uploaded_files)}): {uploaded.name}")

            def on_progress(stage: str, value: float) -> None:
                label = _STAGE_LABELS.get(stage, stage)
                progress.progress(value, text=f"{uploaded.name} — {label}")

            result = ingest_uploaded_bytes(
                uploaded.name,
                uploaded.getvalue(),
                on_progress=on_progress,
            )

            if result.status == DocStatus.INDEXED:
                st.success(
                    f"✅ {result.source_file}：{result.page_count} 页，"
                    f"{result.chunk_count} 个 chunk"
                )
            else:
                st.error(f"❌ {uploaded.name}：{result.error_message}")

        progress.progress(1.0, text="全部完成")
        st.rerun()

    st.divider()
    _render_document_list()


def _render_document_list() -> None:
    """Show indexed documents from doc_registry."""
    st.subheader("已入库文档")
    registry = DocRegistry()
    docs = registry.list_all()
    set_doc_list(docs)

    if not docs:
        st.caption("暂无文档。上传 PDF 后将显示在此处。")
        return

    options = {f"{doc.source_file} ({doc.status.value})": doc.doc_id for doc in docs}
    selected_labels = st.multiselect(
        "检索范围",
        options=list(options.keys()),
        default=list(options.keys()),
        help="限定 Agent 检索的文档范围",
    )
    set_selected_doc_ids([options[label] for label in selected_labels])

    for doc in docs:
        with st.expander(f"📄 {doc.source_file}", expanded=False):
            st.write(f"**状态**: {doc.status.value}")
            st.write(f"**页数**: {doc.page_count}")
            st.write(f"**Chunks**: {doc.chunk_count}")
            st.write(f"**doc_id**: `{doc.doc_id[:8]}...`")
