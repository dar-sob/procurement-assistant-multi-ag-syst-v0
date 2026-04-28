# tests/nodes/test_intake_node.py
# pytest tests/nodes/test_intake_node.py -v
"""
tests/test_intake_node.py

Unit tests for make_intake_node() — the LangGraph node that wraps IntakeAgent.

Strategy:
    - container.intake_agent() is patched so no real agent is instantiated.
    - langgraph.types.interrupt is patched so no real graph checkpoint is needed.
    - No LLM is called at any point.

Covered paths:
    1. Happy path            — agent returns a valid result, node passes it through.
    2. Clarification path    — agent signals missing fields, node calls interrupt()
                               and returns an enriched state.
    3. AgentError            — node catches it and routes to ESCALATE.
    4. ProcurementSystemError— node catches it and routes to ESCALATE.
    5. Unexpected Exception  — node catches it and routes to ESCALATE.
    6. GraphInterrupt        — node re-raises it so LangGraph can handle it.

Run:
    pytest tests/test_intake_node.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from procurement_system.constants import RoutingDecision, StepName
from procurement_system.exceptions import AgentError, ProcurementSystemError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_agent():
    """A fresh MagicMock that stands in for IntakeAgent."""
    return MagicMock()


@pytest.fixture()
def make_node(mock_agent):
    """
    Returns a factory that builds an intake_node with the agent pre-injected.

    Usage inside a test:
        node = make_node()
        result = node(some_state)
    """
    def _factory():
        with patch(
            "procurement_system.nodes.intake_node.container"
        ) as mock_container:
            mock_container.intake_agent.return_value = mock_agent

            # Import *after* patching so the module picks up the mock.
            from procurement_system.nodes.intake_node import make_intake_node
            return make_intake_node()

    return _factory


@pytest.fixture()
def base_state() -> dict:
    """Minimal SharedState sufficient for all tests."""
    return {
        "raw_request": "I need 10 laptops for the dev team.",
        "decision_log": [],
        "errors": [],
        "intake": {},
        "procurement": {},
        "analyst": {},
        "orchestrator": {},
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_agent_result_ok(routing: str = "proceed") -> dict:
    """Minimal agent result that signals successful intake completion."""
    return {
        "routing_decision": routing,
        "process_type": "standard",
        "current_step": StepName.INTAKE_COMPLETED.value,
        "decision_log": ["intake ok"],
        "errors": [],
        "intake": {
            "clarification_question": None,
            "clarification_rounds": 0,
            "missing_fields": [],
        },
    }


def make_agent_result_clarification(
    question: str,
    missing: list,
    round_num: int,
) -> dict:
    """Agent result that requests clarification (no routing_decision yet)."""
    return {
        "current_step": StepName.INTAKE_CLARIFICATION.value,
        "decision_log": [f"clarification round {round_num}"],
        "errors": [],
        "intake": {
            "clarification_question": question,
            "clarification_rounds": round_num,
            "missing_fields": missing,
        },
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIntakeNodeHappyPath:
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
        mock_agent.run.return_value = make_agent_result_ok(routing="proceed")

        node = make_node()
        result = node(base_state)

        assert result["routing_decision"] == "proceed"


class TestIntakeNodeClarificationPath:
    """
    Agent signals missing fields.

    Node must:
        1. Call interrupt() with the right payload.
        2. Return an enriched state that merges the user answer into raw_request.
        3. Reset clarification_question and missing_fields so the next
           agent invocation starts clean.
    """

    QUESTION = "What is the budget for this purchase?"
    MISSING  = ["budget", "currency"]
    ROUND    = 1
    ANSWER   = "50 000 PLN"

    def _run_with_mocked_interrupt(self, make_node, mock_agent, base_state):
        """
        Helper: runs the node with interrupt() mocked to return ANSWER.
        Returns (node_result, captured_interrupt_payload).
        """
        agent_result = make_agent_result_clarification(
            self.QUESTION, self.MISSING, self.ROUND
        )
        mock_agent.run.return_value = agent_result

        captured_payload = {}

        def fake_interrupt(payload):
            captured_payload.update(payload)
            return self.ANSWER

        node = make_node()
        with patch(
            "procurement_system.nodes.intake_node.interrupt",
            side_effect=fake_interrupt,
        ):
            result = node(base_state)

        return result, captured_payload

    def test_interrupt_is_called(self, make_node, mock_agent, base_state):
        """interrupt() must be invoked exactly once."""
        mock_agent.run.return_value = make_agent_result_clarification(
            self.QUESTION, self.MISSING, self.ROUND
        )
        node = make_node()

        with patch(
            "procurement_system.nodes.intake_node.interrupt",
            return_value=self.ANSWER,
        ) as mock_interrupt:
            node(base_state)

        mock_interrupt.assert_called_once()

    def test_interrupt_payload_contains_question(self, make_node, mock_agent, base_state):
        """Payload passed to interrupt() must include the clarification question."""
        _, payload = self._run_with_mocked_interrupt(make_node, mock_agent, base_state)
        assert payload["question"] == self.QUESTION

    def test_interrupt_payload_contains_missing_fields(self, make_node, mock_agent, base_state):
        """Payload passed to interrupt() must list all missing fields."""
        _, payload = self._run_with_mocked_interrupt(make_node, mock_agent, base_state)
        assert payload["missing_fields"] == self.MISSING

    def test_interrupt_payload_contains_round(self, make_node, mock_agent, base_state):
        """Payload passed to interrupt() must include the current round number."""
        _, payload = self._run_with_mocked_interrupt(make_node, mock_agent, base_state)
        assert payload["round"] == self.ROUND

    def test_enriched_raw_request_contains_original(self, make_node, mock_agent, base_state):
        """
        The returned raw_request must start with the original request text
        so the agent retains full context on re-entry.
        """
        result, _ = self._run_with_mocked_interrupt(make_node, mock_agent, base_state)
        assert base_state["raw_request"] in result["raw_request"]

    def test_enriched_raw_request_contains_user_answer(self, make_node, mock_agent, base_state):
        """User's clarification answer must appear in the enriched raw_request."""
        result, _ = self._run_with_mocked_interrupt(make_node, mock_agent, base_state)
        assert self.ANSWER in result["raw_request"]

    def test_clarification_question_reset_to_none(self, make_node, mock_agent, base_state):
        """
        After interrupt() returns, clarification_question must be set to None
        so the next agent invocation does not re-trigger another interrupt.
        """
        result, _ = self._run_with_mocked_interrupt(make_node, mock_agent, base_state)
        assert result["intake"]["clarification_question"] is None

    def test_missing_fields_reset_to_empty(self, make_node, mock_agent, base_state):
        """missing_fields must be cleared so the agent recomputes them fresh."""
        result, _ = self._run_with_mocked_interrupt(make_node, mock_agent, base_state)
        assert result["intake"]["missing_fields"] == []

    def test_current_step_set_to_clarification(self, make_node, mock_agent, base_state):
        """current_step must be INTAKE_CLARIFICATION after an interrupt."""
        result, _ = self._run_with_mocked_interrupt(make_node, mock_agent, base_state)
        assert result["current_step"] == StepName.INTAKE_CLARIFICATION.value


class TestIntakeNodeErrorHandling:
    """
    Node must catch known exceptions and route to ESCALATE
    without propagating the error to the graph.
    """

    def _assert_escalate(self, result: dict) -> None:
        assert result["routing_decision"] == RoutingDecision.ESCALATE.value
        assert result["errors"]           # non-empty list
        assert result["current_step"]     == StepName.INTAKE_COMPLETED.value

    def test_agent_error_routes_to_escalate(self, make_node, mock_agent, base_state):
        """AgentError must be caught and result in ESCALATE routing."""
        mock_agent.run.side_effect = AgentError("LLM returned empty content")

        node = make_node()
        result = node(base_state)

        self._assert_escalate(result)

    def test_agent_error_message_included_in_errors(self, make_node, mock_agent, base_state):
        """The error message must appear in result['errors'] for auditability."""
        error_msg = "LLM returned empty content"
        mock_agent.run.side_effect = AgentError(error_msg)

        node = make_node()
        result = node(base_state)

        assert any(error_msg in e for e in result["errors"])

    def test_procurement_system_error_routes_to_escalate(self, make_node, mock_agent, base_state):
        """ProcurementSystemError must be caught and result in ESCALATE routing."""
        mock_agent.run.side_effect = ProcurementSystemError("Config missing")

        node = make_node()
        result = node(base_state)

        self._assert_escalate(result)

    def test_unexpected_exception_routes_to_escalate(self, make_node, mock_agent, base_state):
        """Any unknown exception must be caught and result in ESCALATE routing."""
        mock_agent.run.side_effect = RuntimeError("Disk full")

        node = make_node()
        result = node(base_state)

        self._assert_escalate(result)

    def test_graph_interrupt_is_re_raised(self, make_node, mock_agent, base_state):
        """
        GraphInterrupt must NOT be swallowed — it must propagate to LangGraph
        so the graph can pause and resume correctly.
        """
        mock_agent.run.side_effect = GraphInterrupt(
            (Interrupt(value={"question": "budget?"}),)
        )

        node = make_node()

        with pytest.raises(GraphInterrupt):
            node(base_state)


class TestIntakeNodeRealWorldFailure:
    """
    Reproduces the failure observed in production:
        model returns empty content → StructuredOutputError raised as AgentError
        → node must route to ESCALATE, not crash.

    This is the exact scenario that caused the missing clarification question.
    """

    def test_empty_llm_response_does_not_crash(self, make_node, mock_agent, base_state):
        """
        When the LLM returns empty content and both schemas fail,
        the node must return a valid ESCALATE state — never raise.
        """
        from procurement_system.exceptions import StructuredOutputError

        mock_agent.run.side_effect = AgentError(
            str(
                StructuredOutputError(
                    agent="intake",
                    schema="IntakeAgentOutput | ClarificationRequest",
                    reason=(
                        "Primary error: Model returned empty content; "
                        "cannot parse into IntakeAgentOutput.. "
                        "Fallback error: Model returned empty content; "
                        "cannot parse into ClarificationRequest."
                    ),
                )
            )
        )

        node = make_node()
        result = node(base_state)   # must not raise

        assert result["routing_decision"] == RoutingDecision.ESCALATE.value

    def test_empty_llm_response_never_calls_interrupt(self, make_node, mock_agent, base_state):
        """
        When both schemas fail, interrupt() must never be called —
        the question never reaches the user, which is the bug we diagnosed.
        """
        from procurement_system.exceptions import StructuredOutputError

        mock_agent.run.side_effect = AgentError(
            str(
                StructuredOutputError(
                    agent="intake",
                    schema="IntakeAgentOutput | ClarificationRequest",
                    reason="Primary error: Model returned empty content.",
                )
            )
        )

        node = make_node()

        with patch(
            "procurement_system.nodes.intake_node.interrupt"
        ) as mock_interrupt:
            node(base_state)

        mock_interrupt.assert_not_called()
