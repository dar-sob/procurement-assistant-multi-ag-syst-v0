# tests/nodes/test_analyst_node.py
# pytest tests/nodes/test_analyst_node.py -v

import pytest
from unittest.mock import Mock, patch

from procurement_system.nodes.analyst_node import make_analyst_node
from procurement_system.state import SharedState
from procurement_system.exceptions import AgentError, ProcurementSystemError
from procurement_system.constants import NodeName, StepName, AnalystFinalDecision


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_analyst_agent():
    """Mock AnalystAgent z metodą .run()"""
    agent = Mock()
    agent.run.return_value = {
        "current_step": StepName.ANALYST_COMPLETED.value,
        "analyst": {
            "risk_analysis": {"overall_risk_level": "MEDIUM"},
            "tco_summary": {"total_usd": 125000.0}
        },
        "final_decision": AnalystFinalDecision.PROCEED.value,
    }
    return agent


@pytest.fixture
def mock_container(mock_analyst_agent):
    container_mock = Mock()
    container_mock.analyst_agent.return_value = mock_analyst_agent
    return container_mock


@pytest.fixture
def analyst_node(mock_container):
    with patch("procurement_system.nodes.analyst_node.container", mock_container):
        return make_analyst_node()


@pytest.fixture
def sample_state() -> SharedState:
    return {
        "procurement": {
            "supplier_recommendations": [
                {"supplier_id": 1, "name": "ABC Sp. z o.o."},
                {"supplier_id": 2, "name": "XYZ Corp"}
            ],
            "items": [{"name": "Laptop", "quantity": 50}]
        },
        "current_step": "supplier_selection_completed",
    }


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

class TestMakeAnalystNode:
    def test_creates_node_function(self, analyst_node):
        assert callable(analyst_node)

    def test_initializes_agent_only_once(self, mock_container):
        with patch("procurement_system.nodes.analyst_node.container", mock_container):
            make_analyst_node()
            mock_container.analyst_agent.assert_called_once()


# -----------------------------------------------------------------------------
# Happy Path
# -----------------------------------------------------------------------------

class TestAnalystNodeHappyPath:
    def test_calls_agent_run_with_state(self, analyst_node, sample_state, mock_analyst_agent):
        analyst_node(sample_state)
        mock_analyst_agent.run.assert_called_once_with(sample_state)

    def test_returns_result_from_agent(self, analyst_node, sample_state):
        result = analyst_node(sample_state)
        assert result["final_decision"] == AnalystFinalDecision.PROCEED.value
        assert result["analyst"]["risk_analysis"]["overall_risk_level"] == "MEDIUM"

    def test_logs_supplier_count(self, analyst_node, sample_state, caplog):
        """Logs the number of suppliers at the INFO level"""
        with caplog.at_level("INFO"):        
            analyst_node(sample_state)

        assert "suppliers=2" in caplog.text
        assert NodeName.ANALYST.value in caplog.text

    def test_logs_successful_completion(self, analyst_node, sample_state, caplog):
        """Logs successful completion of analysis"""
        with caplog.at_level("INFO"):
            analyst_node(sample_state)

        assert "Completed" in caplog.text
        assert "decision=PROCEED" in caplog.text
        assert "risk=MEDIUM" in caplog.text


# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------

class TestAnalystNodeErrorHandling:
    def test_agent_error_returns_escalate_state(self, analyst_node, sample_state, mock_analyst_agent):
        mock_analyst_agent.run.side_effect = AgentError("Model hallucinated prices")

        result = analyst_node(sample_state)

        assert result["final_decision"] == AnalystFinalDecision.ESCALATE.value
        assert "Model hallucinated prices" in str(result.get("errors", []))

    def test_procurement_system_error_returns_escalate(self, analyst_node, sample_state, mock_analyst_agent):
        mock_analyst_agent.run.side_effect = ProcurementSystemError("Database timeout")

        result = analyst_node(sample_state)

        assert result["final_decision"] == AnalystFinalDecision.ESCALATE.value

    def test_unexpected_exception_returns_escalate(self, analyst_node, sample_state, mock_analyst_agent, caplog):
        mock_analyst_agent.run.side_effect = ZeroDivisionError("division by zero")

        result = analyst_node(sample_state)

        assert result["final_decision"] == AnalystFinalDecision.ESCALATE.value
        assert "Unexpected error in analyst node" in str(result.get("errors", []))

    def test_error_logging(self, analyst_node, sample_state, mock_analyst_agent, caplog):
        with caplog.at_level("ERROR"):
            mock_analyst_agent.run.side_effect = AgentError("Critical failure")
            analyst_node(sample_state)

        assert "Agent error" in caplog.text


# -----------------------------------------------------------------------------
# Edge Cases & Robustness
# -----------------------------------------------------------------------------

class TestAnalystNodeEdgeCases:
    def test_handles_empty_supplier_list(self, analyst_node, mock_analyst_agent):
        state = {"procurement": {"supplier_recommendations": []}}
        analyst_node(state)
        mock_analyst_agent.run.assert_called_once()

    def test_handles_missing_procurement_key(self, analyst_node, mock_analyst_agent):
        state: SharedState = {}
        analyst_node(state)
        mock_analyst_agent.run.assert_called_once()

    def test_supplier_count_calculation_is_robust(self, analyst_node, mock_analyst_agent):
        """Node should safely handle different types of supplier recommendations"""
        test_cases = [
            {"procurement": {"supplier_recommendations": None}},
            {"procurement": {}},
            {"procurement": {"supplier_recommendations": "not a list"}},
            {"procurement": {"supplier_recommendations": 123}},
            {"procurement": {"supplier_recommendations": {"key": "value"}}},
            {"procurement": {"supplier_recommendations": []}},
        ]

        for state in test_cases:
            mock_analyst_agent.run.reset_mock()
            analyst_node(state)
            mock_analyst_agent.run.assert_called_once()
