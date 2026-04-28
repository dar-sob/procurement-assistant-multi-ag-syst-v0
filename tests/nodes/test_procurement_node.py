# pytest tests/nodes/test_procurement_node.py -v

"""
tests/nodes/test_procurement_node.py

Unit tests for make_procurement_node() — the LangGraph node that wraps ProcurementAgent.

Strategy:
    - container.procurement_agent() is patched so no real agent is instantiated.
    - No LLM is called at any point.

Covered paths:
    1. Happy path             — agent returns a valid result, node passes it through.
    2. AgentError             — node catches it and routes to ESCALATE.
    3. ProcurementSystemError — node catches it and routes to ESCALATE.
    4. Unexpected Exception   — node catches it and routes to ESCALATE.
    5. Edge cases             — empty supplier list, missing procurement key.

Run:
    pytest tests/nodes/test_procurement_node.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from procurement_system.constants import RoutingDecision, StepName
from procurement_system.exceptions import AgentError, ProcurementSystemError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_agent():
    """A fresh MagicMock that stands in for ProcurementAgent."""
    return MagicMock()


@pytest.fixture()
def make_node(mock_agent):
    """
    Returns a factory that builds a procurement_node with the agent pre-injected.

    Usage inside a test:
        node = make_node()
        result = node(some_state)
    """
    def _factory():
        with patch(
            "procurement_system.nodes.procurement_node.container"
        ) as mock_container:
            mock_container.procurement_agent.return_value = mock_agent

            from procurement_system.nodes.procurement_node import make_procurement_node
            return make_procurement_node()

    return _factory


@pytest.fixture()
def base_state() -> dict:
    """Minimal SharedState sufficient for all tests."""
    return {
        "raw_request":  "I need 10 laptops for the dev team.",
        "category_id":  "IT_HARDWARE",
        "process_type": "standard",
        "decision_log": [],
        "errors":       [],
        "intake":       {},
        "procurement":  {},
        "analyst":      {},
        "orchestrator": {},
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_agent_result_ok(supplier_count: int = 3) -> dict:
    """
    Minimal agent result that signals successful procurement completion
    with a configurable number of supplier recommendations.
    """
    return {
        "routing_decision": "proceed",
        "current_step":     StepName.PROCUREMENT_COMPLETED.value,
        "decision_log":     ["procurement ok"],
        "errors":           [],
        "procurement": {
            "supplier_recommendations": [
                {"name": f"Supplier {i}", "score": round(0.9 - i * 0.1, 1)}
                for i in range(supplier_count)
            ],
            "strategy": "competitive_bidding",
        },
    }


def make_agent_result_no_suppliers() -> dict:
    """Agent result with an empty supplier list — edge case."""
    return {
        "routing_decision": "escalate",
        "current_step":     StepName.PROCUREMENT_COMPLETED.value,
        "decision_log":     ["no suppliers found"],
        "errors":           ["No suppliers available for this category"],
        "procurement": {
            "supplier_recommendations": [],
            "strategy": None,
        },
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestProcurementNodeHappyPath:
    """Agent returns a valid result — node must pass it through unchanged."""

    def test_returns_agent_result_directly(self, make_node, mock_agent, base_state):
        """Node output must equal the dict returned by agent.run()."""
        expected = make_agent_result_ok()
        mock_agent.run.return_value = expected

        node = make_node()
        result = node(base_state)

        assert result == expected

    def test_agent_run_called_once_with_state(self, make_node, mock_agent, base_state):
        """agent.run() must receive the exact state passed to the node."""
        mock_agent.run.return_value = make_agent_result_ok()

        node = make_node()
        node(base_state)

        mock_agent.run.assert_called_once_with(base_state)

    def test_routing_decision_preserved(self, make_node, mock_agent, base_state):
        """routing_decision from agent must survive node pass-through."""
        mock_agent.run.return_value = make_agent_result_ok()

        node = make_node()
        result = node(base_state)

        assert result["routing_decision"] == "proceed"

    def test_supplier_recommendations_preserved(self, make_node, mock_agent, base_state):
        """Supplier list must be returned intact — node must not modify it."""
        mock_agent.run.return_value = make_agent_result_ok(supplier_count=3)

        node = make_node()
        result = node(base_state)

        assert len(result["procurement"]["supplier_recommendations"]) == 3

    def test_empty_supplier_list_handled_gracefully(self, make_node, mock_agent, base_state):
        """
        Node must not crash when agent returns zero suppliers.
        Supplier count logging must handle an empty list safely.
        """
        mock_agent.run.return_value = make_agent_result_no_suppliers()

        node = make_node()
        result = node(base_state)   # must not raise

        assert result["procurement"]["supplier_recommendations"] == []

    def test_missing_procurement_key_does_not_crash(self, make_node, mock_agent, base_state):
        """
        If agent omits the procurement key entirely, node must not raise
        when reading supplier_recommendations for logging.
        """
        mock_agent.run.return_value = {
            "routing_decision": "proceed",
            "current_step":     StepName.PROCUREMENT_COMPLETED.value,
            "decision_log":     [],
            "errors":           [],
        }

        node = make_node()
        result = node(base_state)   # must not raise

        assert result["routing_decision"] == "proceed"


class TestProcurementNodeErrorHandling:
    """
    Node must catch known exceptions and route to ESCALATE
    without propagating the error to the graph.
    """

    def _assert_escalate(self, result: dict) -> None:
        assert result["routing_decision"] == RoutingDecision.ESCALATE.value
        assert result["errors"]
        assert result["current_step"] == StepName.PROCUREMENT_COMPLETED.value

    def test_agent_error_routes_to_escalate(self, make_node, mock_agent, base_state):
        """AgentError must be caught and result in ESCALATE routing."""
        mock_agent.run.side_effect = AgentError("LLM returned empty content")

        node = make_node()
        result = node(base_state)

        self._assert_escalate(result)

    def test_agent_error_message_in_errors(self, make_node, mock_agent, base_state):
        """AgentError message must appear in result['errors'] for auditability."""
        error_msg = "LLM returned empty content"
        mock_agent.run.side_effect = AgentError(error_msg)

        node = make_node()
        result = node(base_state)

        assert any(error_msg in e for e in result["errors"])

    def test_procurement_system_error_routes_to_escalate(self, make_node, mock_agent, base_state):
        """ProcurementSystemError must be caught and result in ESCALATE routing."""
        mock_agent.run.side_effect = ProcurementSystemError("Supplier DB unreachable")

        node = make_node()
        result = node(base_state)

        self._assert_escalate(result)

    def test_procurement_system_error_message_in_errors(self, make_node, mock_agent, base_state):
        """ProcurementSystemError message must appear in result['errors']."""
        error_msg = "Supplier DB unreachable"
        mock_agent.run.side_effect = ProcurementSystemError(error_msg)

        node = make_node()
        result = node(base_state)

        assert any(error_msg in e for e in result["errors"])

    def test_unexpected_exception_routes_to_escalate(self, make_node, mock_agent, base_state):
        """Any unknown exception must be caught and result in ESCALATE routing."""
        mock_agent.run.side_effect = RuntimeError("Connection timeout")

        node = make_node()
        result = node(base_state)

        self._assert_escalate(result)

    def test_unexpected_exception_message_in_errors(self, make_node, mock_agent, base_state):
        """Unexpected exception message must appear in result['errors']."""
        mock_agent.run.side_effect = RuntimeError("Connection timeout")

        node = make_node()
        result = node(base_state)

        assert any("Connection timeout" in e for e in result["errors"])

    def test_error_result_never_contains_stale_suppliers(self, make_node, mock_agent, base_state):
        """
        On error the node returns a minimal dict with no procurement section —
        caller must not see stale supplier data from a previous run.
        """
        mock_agent.run.side_effect = AgentError("timeout")

        node = make_node()
        result = node(base_state)

        assert "procurement" not in result
