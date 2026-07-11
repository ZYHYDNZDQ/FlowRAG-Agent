"""Static test data — separated from business code and test logic."""

from tests.fixtures.data.documents import (
    SAMPLE_CONTRACT_FILENAME,
    SAMPLE_CONTRACT_PAGES,
    SAMPLE_QUERIES,
)
from tests.fixtures.data.llm_responses import FAKE_LLM_RESPONSES
from tests.fixtures.data.qa_scenarios import QA_SCENARIOS
from tests.fixtures.data.router_cases import ROUTER_SCOPE_CASES, ROUTER_TEST_CASES

__all__ = [
    "FAKE_LLM_RESPONSES",
    "QA_SCENARIOS",
    "ROUTER_SCOPE_CASES",
    "ROUTER_TEST_CASES",
    "SAMPLE_CONTRACT_FILENAME",
    "SAMPLE_CONTRACT_PAGES",
    "SAMPLE_QUERIES",
]
