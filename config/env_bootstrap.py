"""
Bootstrap environment variables that must be set before third-party imports.

huggingface_hub reads HF_ENDPOINT into constants.ENDPOINT at import time.
If transformers/huggingface_hub is imported first (e.g. Streamlit file watcher),
later os.environ updates are ignored unless constants.ENDPOINT is patched too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_project_root_on_path() -> None:
    """Ensure repo root is importable when running Streamlit or stale editable installs."""
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _read_hf_endpoint_from_dotenv() -> str | None:
    env_file = _PROJECT_ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "HF_ENDPOINT":
            endpoint = value.strip()
            return endpoint or None
    return None


def apply_hf_endpoint(endpoint: str | None = None) -> None:
    """Apply HuggingFace mirror endpoint to os.environ and huggingface_hub."""
    resolved = endpoint or os.environ.get("HF_ENDPOINT") or _read_hf_endpoint_from_dotenv()
    if not resolved:
        return

    resolved = resolved.rstrip("/")
    os.environ["HF_ENDPOINT"] = resolved
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.ENDPOINT = resolved
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            f"{resolved}/{{repo_id}}/resolve/{{revision}}/{{filename}}"
        )
        hf_constants.HUGGINGFACE_CO_URL_HOME = f"{resolved}/"
    except ImportError:
        pass


# Run as early as possible when this module is imported.
ensure_project_root_on_path()
apply_hf_endpoint()
