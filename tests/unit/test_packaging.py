"""Packaging and import-path regression tests."""

from __future__ import annotations

import importlib.metadata as metadata

import pytest

pytestmark = pytest.mark.unit

_REQUIRED_TOP_LEVEL = ("skills", "tools", "memory", "agent")


def test_editable_install_exposes_top_level_packages():
    """Stale ``pip install -e .`` metadata omitted skills/tools/memory (pytest still passed)."""
    try:
        dist = metadata.distribution("flowrag-agent")
    except metadata.PackageNotFoundError:
        pytest.skip("flowrag-agent not installed; run: pip install -e .")
    top_level = (dist.read_text("top_level.txt") or "").split()
    missing = [name for name in _REQUIRED_TOP_LEVEL if name not in top_level]
    assert not missing, (
        f"Missing from editable install top_level.txt: {missing}. "
        "Run: pip install -e ."
    )


def test_streamlit_entry_import_chain():
    """Smoke test: same import chain as app/streamlit_app.py startup."""
    import config.env_bootstrap  # noqa: F401

    from agent.runtime import ExecuteRequest, get_runtime
    from skills.registry import get_skill_registry
    from tools.registry import get_tool_registry

    assert get_runtime() is not None
    assert get_skill_registry().list_skills()
    assert get_tool_registry().names()
