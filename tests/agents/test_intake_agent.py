# tests/agents/test_intake_agent.py
# pytest tests/agents/test_intake_agent.py -v

import pytest
from unittest.mock import Mock, patch

from procurement_system.agents.intake.agent import IntakeAgent
from procurement_system.exceptions import StructuredOutputError
from procurement_system.schemas.intake_schemas import (
    ClarificationRequest,
    DecisionLogEntry,
    IntakeAgentOutput,
    ValidatedRequest,
)
from procurement_system.state import SharedState
from procurement_system.constants import StepName  # agent.py imports StepName, not Steps


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_TIMESTAMP = "2025-01-01T00:00:00"
"""Canonical ISO-8601 timestamp used in all test fixtures.

Using one constant avoids Pydantic datetime-validation failures that occur
when a test passes an arbitrary string such as 'now' or 'today'.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_state(raw_request: str) -> SharedState:
    """Return a minimal SharedState suitable for intake-agent tests.

    SharedState is declared with ``total=False``, so every key is optional
    and the agent accesses all fields via ``.get()``.  A bare dict with only
    ``raw_request`` is therefore valid — no missing-key crashes can occur.

    Args:
        raw_request: The purchase request text the agent will process.

    Returns:
        A SharedState dict containing only the fields relevant to intake.
    """
    return {"raw_request": raw_request}


def make_validated_output(
    *,
    process_type: str = "rfq",
    routing_decision: str = "proceed",
    clarification_rounds: int = 0,
) -> IntakeAgentOutput:
    """Build a complete IntakeAgentOutput with sensible defaults.

    Centralising object construction here keeps individual tests short:
    each test overrides only the fields it actually cares about.

    Args:
        process_type:         Procurement process type string.
        routing_decision:     'proceed' or 'escalate'.
        clarification_rounds: Number of clarification rounds completed.

    Returns:
        A fully populated IntakeAgentOutput that passes Pydantic validation.
    """
    return IntakeAgentOutput(
        validated_request=ValidatedRequest(
            description="10 laptopów biznesowych",
            quantity=10,
            unit="sztuki",
        ),
        estimated_value_usd=15_000.0,
        process_type=process_type,
        category_id="IT_HARDWARE",
        routing_decision=routing_decision,
        routing_justification="Wartość poniżej progu strategicznego — reguła T2.",
        decision_log=DecisionLogEntry(
            timestamp=VALID_TIMESTAMP,
            process_type=process_type,
            routing_decision=routing_decision,
        ),
        clarification_rounds=clarification_rounds,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def agent_with_mocks():
    """Create a real IntakeAgent with all external dependencies replaced by Mocks.

    Construction strategy
    ---------------------
    1. ``build_agent_llm_bundle`` is patched at import time so the constructor
       never opens a network connection or reads GPU resources.
    2. After construction every LLM attribute is replaced individually,
       giving each test full, isolated control over return values and
       side effects without cross-test contamination.
    3. ``_record_metrics`` is replaced so tests can assert telemetry calls
       without needing a real metrics backend.

    Note on ``_metrics``
    --------------------
    The real ``_record_metrics`` method guards itself with
    ``if self._metrics is None: return``.  Replacing the *method* with a
    Mock bypasses that guard entirely, which is exactly what we want:
    tests can assert ``assert_called_once()`` without providing a metrics
    collector.

    Yields:
        A fully initialised IntakeAgent ready for testing.
    """
    with patch(
        "procurement_system.agents.intake.agent.build_agent_llm_bundle"
    ) as mock_bundle:
        mock_llm_bundle = Mock()
        mock_llm_bundle.structured_llm = Mock()
        mock_bundle.return_value = mock_llm_bundle
        agent = IntakeAgent()

    # Replace LLMs with controllable mocks
    agent._llm = Mock()
    agent._llm.structured_llm = Mock()
    agent._clarification_llm = Mock()

    # Replace telemetry method (not the _metrics attribute)
    agent._record_metrics = Mock()

    # Use a predictable round limit across all tests
    agent._max_clarification_rounds = 3

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Happy path: primary LLM succeeds on the first attempt
# ─────────────────────────────────────────────────────────────────────────────

def test_primary_success(agent_with_mocks):
    """Verify the agent maps IntakeAgentOutput correctly when the primary LLM succeeds.

    Scenario
    --------
    The primary LLM (``_llm.structured_llm``) returns a valid
    IntakeAgentOutput.  The agent must:

    - Surface ``process_type`` and ``routing_decision`` at the top level of
      the returned state dict.
    - Write ``clarification_question: None`` inside ``result["intake"]``
      (the agent always writes this key; on the success path its value is
      None — verified from ``_format_result`` in agent.py).
    - Never touch ``_clarification_llm`` (fallback must be skipped).
    - Call ``_record_metrics`` exactly once (telemetry fires on success).
    - Return no errors.
    """
    agent = agent_with_mocks
    state = make_state("I'm asking for 10 laptops")

    mock_output = make_validated_output(process_type="rfq", routing_decision="proceed")
    agent._llm.structured_llm.invoke.return_value = mock_output

    result = agent.run(state)

    # Top-level routing fields must be propagated from IntakeAgentOutput
    assert result["process_type"] == "rfq"
    assert result["routing_decision"] == "proceed"

    # On the success path _format_result always sets clarification_question to None.
    # We check the *value*, not key absence, because the key is always present.
    assert result["intake"]["clarification_question"] is None

    # No error payloads written
    assert result["errors"] == []

    # Primary was called once; fallback must never have been touched
    agent._llm.structured_llm.invoke.assert_called_once()
    agent._clarification_llm.invoke.assert_not_called()

    # Telemetry must fire on the happy path
    agent._record_metrics.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Fallback path: primary fails, clarification LLM takes over
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_on_primary_error(agent_with_mocks):
    """Verify the agent delegates to _clarification_llm when the primary LLM raises.

    Scenario
    --------
    ``_llm.structured_llm.invoke`` raises an Exception (simulating a timeout,
    schema-parse failure, or any other primary-path error).
    The agent must:

    - Catch the exception silently and log a warning (visible in captured logs).
    - Invoke ``_clarification_llm`` exactly once with a list of LangChain
      messages ``[SystemMessage, HumanMessage]`` — NOT the raw state dict.
      (The agent builds the message list in ``run()`` before calling the
      fallback; confirmed from agent.py source.)
    - Embed the original ``raw_request`` text inside the HumanMessage so
      the fallback LLM has full context.
    - Write ``clarification_question`` and ``missing_fields`` into
      ``result["intake"]``.
    - Set ``current_step`` to the value of ``StepName.INTAKE_CLARIFICATION``
      (a string literal, not the enum itself).
    - Return no errors.

    Note on assert_called_once_with
    --------------------------------
    ``assert_called_once_with(state)`` always fails here because the agent
    passes a constructed message list, not the state dict.  We instead use
    ``call_args`` inspection to verify the behavioural contract: the
    ``raw_request`` string must appear somewhere in the prompt.
    """
    agent = agent_with_mocks
    state = make_state("niekompletny request")

    agent._llm.structured_llm.invoke.side_effect = Exception("Primary LLM failed")

    mock_clarification = ClarificationRequest(
        question="Enter quantity and unit of measure.",
        missing_fields=["quantity", "unit"],
    )
    agent._clarification_llm.invoke.return_value = mock_clarification

    result = agent.run(state)

    # Clarification payload must be written into the intake section
    intake = result["intake"]
    assert intake["clarification_question"] == "Enter quantity and unit of measure."
    assert intake["missing_fields"] == ["quantity", "unit"]

    # Graph control step must use the canonical enum value (a string)
    assert result["current_step"] == StepName.INTAKE_CLARIFICATION.value

    # No errors surfaced to the caller
    assert result["errors"] == []

    # Fallback LLM was invoked exactly once
    agent._clarification_llm.invoke.assert_called_once()

    # Verify the raw_request is embedded in the prompt sent to the fallback.
    # The agent passes [SystemMessage, HumanMessage]; we stringify the list
    # and search for the original text — robust to any formatting changes.
    call_args = agent._clarification_llm.invoke.call_args
    prompt_messages = call_args.args[0]
    full_prompt_text = " ".join(str(msg) for msg in prompt_messages)
    assert state["raw_request"] in full_prompt_text, (
        "raw_request must be embedded in the prompt forwarded to _clarification_llm; "
        f"searched for {state['raw_request']!r} in the reconstructed prompt text."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Guard: fallback is skipped when primary succeeds
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_not_called_when_primary_works(agent_with_mocks):
    """Confirm _clarification_llm is never invoked when the primary LLM returns a result.

    Scenario
    --------
    The primary LLM returns a valid ``catalog_purchase`` output.
    The fallback must remain completely idle.  This is a regression guard:
    if someone accidentally wires the fallback to always run, this test
    will catch the breakage immediately.
    """
    agent = agent_with_mocks
    state = make_state("I would like 5 reams of A4 paper, please.")

    mock_output = make_validated_output(
        process_type="catalog_purchase",
        routing_decision="proceed",
    )
    agent._llm.structured_llm.invoke.return_value = mock_output

    result = agent.run(state)

    agent._clarification_llm.invoke.assert_not_called()
    assert result["process_type"] == "catalog_purchase"
    assert result["errors"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Boundary: max clarification rounds exceeded → StructuredOutputError
# ─────────────────────────────────────────────────────────────────────────────

def test_max_clarification_rounds_exceeded(agent_with_mocks):
    """Verify the agent raises StructuredOutputError when the round limit is reached.

    Scenario
    --------
    The incoming state already has ``clarification_rounds == 3``, which
    equals ``_max_clarification_rounds``.  The guard at the top of ``run()``
    fires before any LLM is called (confirmed from agent.py lines 286-296):

    .. code-block:: python

        if current_round >= self._max_clarification_rounds:
            raise StructuredOutputError(...)

    The agent must:

    - Raise ``StructuredOutputError`` immediately.
    - Never call ``_clarification_llm`` (no infinite clarification loop).
    - Include the round limit in the exception message so it is actionable
      in logs and monitoring dashboards.

    Why pytest.raises and not a result dict
    ----------------------------------------
    An earlier version of this test checked ``result["errors"]``.  Reading
    the actual source code shows the agent raises rather than returning —
    so ``pytest.raises`` is the correct assertion mechanism.
    """
    agent = agent_with_mocks
    state = make_state("nadal niekompletny request")

    # Simulate a state that has already exhausted all clarification rounds
    state["intake"] = {"clarification_rounds": agent._max_clarification_rounds}

    # Primary failure is configured but the guard fires before any LLM call
    agent._llm.structured_llm.invoke.side_effect = Exception("Primary LLM failed")

    with pytest.raises(StructuredOutputError) as exc_info:
        agent.run(state)

    # The exception must reference the limit so operators can act on it
    assert str(agent._max_clarification_rounds) in str(exc_info.value)

    # Neither LLM must have been touched — the guard must fire first
    agent._llm.structured_llm.invoke.assert_not_called()
    agent._clarification_llm.invoke.assert_not_called()
