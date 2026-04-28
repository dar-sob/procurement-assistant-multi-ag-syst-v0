# tests/nodes/test_orchestrator_node.py
# pytest tests/nodes/test_orchestrator_node.py -v

import pytest
from unittest.mock import Mock, patch

from procurement_system.nodes.orchestrator_node import make_orchestrator_node
from procurement_system.state import SharedState
from procurement_system.exceptions import AgentError, ProcurementSystemError
from procurement_system.constants import NodeName, StepName


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_orchestrator_agent():
    """Mock OrchestratorAgent with method .run()"""
    agent = Mock()
    agent.run.return_value = {
        "current_step": StepName.ORCHESTRATOR_COMPLETED.value,
        "orchestrator": {
            "final_report": {
                "report_id": "REP-20250427-001",
                "summary": "Rekomendacja zakupu 50 laptopów"
            }
        },
        "final_decision": "PROCEED",
        "message_to_user": "Oto finalna rekomendacja zakupu..."
    }
    return agent


@pytest.fixture
def mock_container(mock_orchestrator_agent):
    """Mock container DI"""
    container_mock = Mock()
    container_mock.orchestrator_agent.return_value = mock_orchestrator_agent
    return container_mock


@pytest.fixture
def orchestrator_node(mock_container):
    """A ready node with a mock container injected"""
    with patch("procurement_system.nodes.orchestrator_node.container", mock_container):
        return make_orchestrator_node()


@pytest.fixture
def sample_state() -> SharedState:
    """Example input state"""
    return {
        "procurement": {
            "items": [{"name": "Laptop Dell", "quantity": 50}],
            "supplier_recommendations": [
                {"supplier_id": 1, "name": "ABC Tech", "final_price_usd": 124500}
            ]
        },
        "analyst": {
            "risk_analysis": {"overall_risk_level": "LOW"}
        },
        "final_decision": "PROCEED",
        "current_step": "analyst_completed",
    }


# -----------------------------------------------------------------------------
# Initialization & Factory
# -----------------------------------------------------------------------------

class TestMakeOrchestratorNode:
    def test_creates_node_function(self, orchestrator_node):
        assert callable(orchestrator_node)

    def test_initializes_agent_only_once(self, mock_container):
        with patch("procurement_system.nodes.orchestrator_node.container", mock_container):
            make_orchestrator_node()
            mock_container.orchestrator_agent.assert_called_once()


# -----------------------------------------------------------------------------
# Happy Path
# -----------------------------------------------------------------------------

class TestOrchestratorNodeHappyPath:
    def test_calls_agent_run_with_state(self, orchestrator_node, sample_state, mock_orchestrator_agent):
        """The node should pass all state to the agent."""
        orchestrator_node(sample_state)
        mock_orchestrator_agent.run.assert_called_once_with(sample_state)

    def test_returns_result_from_agent(self, orchestrator_node, sample_state):
        """Node returns exactly what agent.run() returned"""
        result = orchestrator_node(sample_state)

        assert result["current_step"] == StepName.ORCHESTRATOR_COMPLETED.value
        assert result["orchestrator"]["final_report"]["report_id"] == "REP-20250427-001"
        assert result["final_decision"] == "PROCEED"

    def test_logs_initial_decision(self, orchestrator_node, sample_state, caplog):
        """Logs the decision on entry"""
        with caplog.at_level("INFO"):
            orchestrator_node(sample_state)

        assert "Compiling final report" in caplog.text
        assert "decision=PROCEED" in caplog.text
        assert NodeName.ORCHESTRATOR.value in caplog.text

    def test_logs_successful_completion(self, orchestrator_node, sample_state, caplog):
        """Logs successful completion with report_id"""
        with caplog.at_level("INFO"):
            orchestrator_node(sample_state)

        assert "Completed" in caplog.text
        assert "report_id=REP-20250427-001" in caplog.text


# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------

class TestOrchestratorNodeErrorHandling:
    def test_agent_error_returns_fallback_state(self, orchestrator_node, sample_state, mock_orchestrator_agent):
        """AgentError → returns status with message to user"""
        mock_orchestrator_agent.run.side_effect = AgentError("Failed to generate report")

        result = orchestrator_node(sample_state)

        assert result["current_step"] == StepName.ORCHESTRATOR_COMPLETED.value
        assert "errors" in result
        assert "Failed to generate report" in result["errors"][0]
        assert "Please contact the procurement team" in result.get("message_to_user", "")

    def test_procurement_system_error_returns_fallback(self, orchestrator_node, sample_state, mock_orchestrator_agent):
        mock_orchestrator_agent.run.side_effect = ProcurementSystemError("Database connection failed")

        result = orchestrator_node(sample_state)

        assert result["current_step"] == StepName.ORCHESTRATOR_COMPLETED.value
        assert "system error" in result.get("message_to_user", "").lower()

    def test_unexpected_exception_returns_fallback(self, orchestrator_node, sample_state, mock_orchestrator_agent, caplog):
        """Unexpected exception → fallback + logging with traceback"""
        mock_orchestrator_agent.run.side_effect = ZeroDivisionError("division by zero")

        result = orchestrator_node(sample_state)

        assert result["current_step"] == StepName.ORCHESTRATOR_COMPLETED.value
        assert "Unexpected error in orchestrator node" in result["errors"][0]
        assert "ZeroDivisionError" in caplog.text

    def test_error_logging(self, orchestrator_node, sample_state, mock_orchestrator_agent, caplog):
        with caplog.at_level("ERROR"):
            mock_orchestrator_agent.run.side_effect = AgentError("Critical orchestrator failure")
            orchestrator_node(sample_state)

        assert "Agent error" in caplog.text
        assert "Critical orchestrator failure" in caplog.text


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------

class TestOrchestratorNodeEdgeCases:
    def test_handles_missing_final_decision(self, orchestrator_node, mock_orchestrator_agent, caplog):
        """When there is no final_decision in state → logs UNKNOWN"""
        state: SharedState = {"procurement": {}}

        with caplog.at_level("INFO"):
            orchestrator_node(state)

        assert any("decision=UNKNOWN" in msg for msg in caplog.messages)

    def test_handles_empty_state(self, orchestrator_node, mock_orchestrator_agent):
        """A completely empty state should not cause an error"""
        state: SharedState = {}
        orchestrator_node(state)
        mock_orchestrator_agent.run.assert_called_once_with(state)

    def test_completion_logging_when_no_report_id(self, orchestrator_node, mock_orchestrator_agent, caplog):
        """When agent did not return report_id → logs N/A"""
        mock_orchestrator_agent.run.return_value = {
            "current_step": StepName.ORCHESTRATOR_COMPLETED.value,
            "orchestrator": {},
            "final_decision": "PROCEED"
        }

        with caplog.at_level("INFO"):
            orchestrator_node({})

        assert "report_id=N/A" in caplog.text
