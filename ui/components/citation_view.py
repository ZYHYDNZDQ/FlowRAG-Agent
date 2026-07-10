"""
Citation view — display source references for an answer.

Shows: source_file, page, similarity score, excerpt preview.
Grouped by document, sorted by page.
"""

from __future__ import annotations

import streamlit as st

from models.schemas import Citation
from retrieval.citation_builder import format_citation_label


def render_citations(citations: list[Citation] | None) -> None:
    """Render citation cards below the answer."""
    if not citations:
        return

    st.markdown("**引用来源**")
    for citation in citations:
        label = format_citation_label(citation)
        score_text = f" · 相似度 {citation.score:.2f}" if citation.score is not None else ""
        with st.container(border=True):
            st.markdown(f"📎 **{label}**{score_text}")
            if citation.excerpt:
                st.caption(citation.excerpt)
