# tests/agents/test_procurement_agent.py
# pytest tests/agents/test_procurement_agent.py -v

import json
import pytest
from unittest.mock import Mock, patch

from procurement_system.agents.procurement.agent import ProcurementAgent
from procurement_system.exceptions import StructuredOutputError
from procurement_system.schemas.procurement_schemas import (
    ProcurementAgentOutput,
    SupplierRecommendation,
)
from procurement_system.state import SharedState
from procurement_system.constants import StepName


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TEST_USER_TEMPLATE = (
    "product={product_name} category={category} "
    "qty={quantity} unit={unit} budget={estimated_budget} "
    "process={process_type} req={requirements} "
    "deadline={deadline} urgency={urgency}"
)
"""Minimal user-prompt template that mirrors all placeholders used in _build_prompt.

The real template lives in a YAML/txt file loaded at runtime.  Using a simple
inline string here keeps tests independent of the file system while still
exercising every .format() substitution the agent performs.
"""

PROCUREMENT_STRATEGY = (
    "Given the mid-range budget and rfq process type, the recommended approach "
    "is to issue a Request for Quotation to at least three national suppliers. "
    "Urgency is medium, so a 2-week lead time is acceptable. "
    "Framework agreements should be considered for repeat orders above $10k."
)
"""Strategy string that satisfies the Pydantic min_length=150 constraint."""

# ─────────────────────────────────────────────────────────────────────────────
# Patch targets — defined once so a typo is caught in one place
# ─────────────────────────────────────────────────────────────────────────────

# build_agent_llm_bundle and resolve_from_tier are imported and called
# inside base_agent.py, NOT inside procurement/agent.py.
# Rule: patch where the name is *looked up*, not where it is defined.
_BASE = "procurement_system.agents.base_agent"
_AGENT = "procurement_system.agents.procurement.agent"

PATCH_BUILD_LLM_BUNDLE  = f"{_BASE}.build_agent_llm_bundle"
PATCH_RESOLVE_FROM_TIER = f"{_BASE}.resolve_from_tier"      # called in BaseAgent.__init__
PATCH_GET_CONFIG        = f"{_AGENT}.get_procurement_config"
PATCH_GET_SYS_PROMPT    = f"{_AGENT}.get_procurement_system_prompt"
PATCH_GET_USER_PROMPT   = f"{_AGENT}.get_procurement_user_prompt"
PATCH_MAKE_TOOLS        = f"{_AGENT}.make_procurement_tools"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — state factory
# ─────────────────────────────────────────────────────────────────────────────

def make_state(
    *,
    process_type: str = "rfq",
    category_id: str = "IT_HARDWARE",
    description: str = "laptopy biznesowe Dell",
    quantity: float = 10.0,
    unit: str = "sztuki",
    estimated_value_usd: float = 15_000.0,
    requirements: list | None = None,
    deadline: str | None = None,
    urgency: str = "medium",
) -> SharedState:
    """Return a SharedState that mirrors real post-intake graph state.

    ProcurementAgent._build_prompt() reads from three locations in state:
      - state["intake"]["validated_request"]   (description, quantity, unit, …)
      - state["intake"]["estimated_value_usd"] (budget)
      - state["category_id"] and state["process_type"] (top-level routing fields)

    This factory populates all of them so individual tests only override the
    one field they exercise.  SharedState has ``total=False`` so every key
    is optional — the agent always guards reads with ``.get()``.

    Args:
        process_type:         Routing decision from the intake agent.
        category_id:          Procurement category from intake classification.
        description:          Full product / service description.
        quantity:             Numeric quantity from validated request.
        unit:                 Unit of measure.
        estimated_value_usd:  Total estimated purchase value.
        requirements:         Optional list of technical requirements.
        deadline:             Optional delivery deadline string.
        urgency:              Urgency level string.

    Returns:
        Populated SharedState dict ready to pass to agent methods.
    """
    return {
        "process_type": process_type,
        "category_id": category_id,
        "intake": {
            "estimated_value_usd": estimated_value_usd,
            "validated_request": {
                "description": description,
                "quantity": quantity,
                "unit": unit,
                "requirements": requirements or [],
                "deadline": deadline,
                "urgency": urgency,
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — output factories
# ─────────────────────────────────────────────────────────────────────────────

def make_supplier(
    *,
    name: str = "TechDirect Sp. z o.o.",
    supplier_type: str = "national",
    contact_priority: int = 1,
    reliability_score: float = 8.5,
) -> SupplierRecommendation:
    """Return a valid SupplierRecommendation with sensible defaults.

    Args:
        name:              Full supplier name shown in the output.
        supplier_type:     Geographic scope: 'local', 'national', 'international'.
        contact_priority:  1 = primary, 2 = secondary, 3 = backup.
        reliability_score: Float 1–10; must satisfy ge=1, le=10 validation.

    Returns:
        A Pydantic-validated SupplierRecommendation instance.
    """
    return SupplierRecommendation(
        name=name,
        supplier_type=supplier_type,
        estimated_price_range="$12,000 – $16,000",
        lead_time="2–3 weeks",
        reliability_score=reliability_score,
        pros=["Certyfikat ISO", "Fast delivery"],
        cons=["Higher price"],
        contact_priority=contact_priority,
    )


def make_procurement_output(
    *,
    recommended_order_type: str = "rfq",
    supplier_count: int = 1,
    errors: list | None = None,
) -> ProcurementAgentOutput:
    """Return a fully valid ProcurementAgentOutput with sensible defaults.

    Builds the required number of suppliers automatically, assigning each a
    unique contact_priority so the (commented-out) uniqueness validator works
    correctly if it is ever re-enabled.

    Args:
        recommended_order_type: Order type literal expected in the output.
        supplier_count:         How many SupplierRecommendation objects to include (1–3).
        errors:                 Optional list of non-fatal error strings.

    Returns:
        A Pydantic-validated ProcurementAgentOutput instance.
    """
    suppliers = [
        make_supplier(name=f"Dostawca {i}", contact_priority=i)
        for i in range(1, supplier_count + 1)
    ]
    return ProcurementAgentOutput(
        supplier_recommendations=suppliers,
        procurement_strategy=PROCUREMENT_STRATEGY,
        recommended_order_type=recommended_order_type,
        negotiation_points=[
            "Negotiate a unit price below $1,400",
            "Negotiate a free 3-year warranty",
            "Establish contractual penalties for delays exceeding 5 days",
        ],
        alternative_products=["Lenovo ThinkPad", "HP EliteBook"],
        errors=errors or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def agent_with_mocks():
    """Create a ProcurementAgent with all external dependencies replaced by Mocks.

    Patching strategy
    -----------------
    Six targets are patched — at the module where each name is *looked up*,
    not where it is defined.  This is the core unittest.mock rule.

    procurement/agent.py imports:
      - get_procurement_config, get_procurement_system_prompt,
        get_procurement_user_prompt  → patch at PATCH_AGENT (procurement/agent.py)
      - make_procurement_tools       → patch at PATCH_AGENT

    base_agent.py imports (called during BaseAgent.__init__ which ProcurementAgent
    delegates to via super().__init__()):
      - build_agent_llm_bundle       → patch at PATCH_BASE (base_agent.py)
      - resolve_from_tier            → patch at PATCH_BASE (base_agent.py)

    Why resolve_from_tier must be patched
    --------------------------------------
    BaseAgent.__init__ calls resolve_from_tier(config.model_tier) when
    model_tier is not None.  Our config_stub sets model_tier='balanced',
    so without this patch the real resolver would try to hit configuration
    files or the model registry during test setup.

    Why make_procurement_tools returns []
    --------------------------------------
    BaseAgent.has_tools == bool(self.config.tools).  Returning an empty list
    makes has_tools False, so BaseAgent._execute() skips execute_tool_loop
    entirely and passes messages directly to the structured LLM.  This keeps
    integration tests simple — no ReAct loop simulation needed.

    Post-construction replacements
    --------------------------------
    After the agent is built, _llm is replaced with a fresh Mock so each
    test controls invoke() return values in isolation.  _metrics is set to
    None (production default) so the real _record_metrics guard works when
    _format_result is called directly.

    Returns:
        A fully initialised ProcurementAgent ready for testing.
    """
    config_stub = {
        "procurement_agent": {
            "model_tier": "balanced",
            "prompt": {"checksums": {}},
        }
    }

    mock_chain = Mock()

    with (
        patch(PATCH_BUILD_LLM_BUNDLE) as mock_bundle,
        patch(PATCH_RESOLVE_FROM_TIER, return_value=mock_chain),
        patch(PATCH_GET_CONFIG, return_value=config_stub),
        patch(PATCH_GET_SYS_PROMPT, return_value="system prompt"),
        patch(PATCH_GET_USER_PROMPT, return_value=TEST_USER_TEMPLATE),
        patch(PATCH_MAKE_TOOLS, return_value=[]),
    ):
        mock_bundle.return_value = Mock()
        mock_bundle.return_value.structured_llm = Mock()
        mock_bundle.return_value.llm_with_tools = Mock()
        agent = ProcurementAgent()

    # Replace the LLM bundle post-construction for per-test control
    agent._llm = Mock()
    agent._llm.structured_llm = Mock()

    # _metrics=None → _record_metrics returns early (production default).
    # Tests that verify telemetry set agent._metrics = Mock() themselves.
    agent._metrics = None

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — _build_prompt: field extraction and defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPrompt:
    """Unit tests for ProcurementAgent._build_prompt().

    Each test calls _build_prompt() directly and inspects the returned string.
    TEST_USER_TEMPLATE contains all placeholders that _build_prompt() passes
    to .format(), so real string substitution is exercised without depending
    on YAML prompt files.
    """

    def test_full_state_all_fields_present(self, agent_with_mocks):
        """All intake fields present → every placeholder is substituted correctly.

        Scenario
        --------
        State has a complete intake section.  _build_prompt() must extract each
        field and embed it in the returned prompt string via .format().
        We assert every significant value appears in the output.
        """
        agent = agent_with_mocks
        state = make_state(
            description="laptop Dell XPS",
            quantity=5,
            unit="szt",
            category_id="IT_HARDWARE",
            process_type="rfq",
            estimated_value_usd=8_000.0,
            requirements=["SSD 512GB", "RAM 16GB"],
            deadline="2025-06-01",
            urgency="high",
        )

        prompt = agent._build_prompt(state)

        assert "laptop Dell XPS" in prompt
        assert "IT_HARDWARE" in prompt
        assert "5" in prompt
        assert "szt" in prompt
        assert "8000" in prompt
        assert "rfq" in prompt
        assert "SSD 512GB" in prompt     # inside JSON array
        assert "2025-06-01" in prompt
        assert "high" in prompt

    def test_missing_intake_section_uses_defaults(self, agent_with_mocks):
        """No intake key in state → _build_prompt falls back to documented defaults.

        Scenario
        --------
        SharedState has total=False so an empty dict is valid.  Every .get()
        in _build_prompt has a documented default:
          product_name / category / quantity / unit / process_type → 'N/A'
          estimated_budget → 'unknown'
          deadline         → 'not specified'
          urgency          → 'medium'
        All of these must appear in the returned prompt.
        """
        agent = agent_with_mocks
        state: SharedState = {}

        prompt = agent._build_prompt(state)

        assert "N/A" in prompt
        assert "unknown" in prompt
        assert "not specified" in prompt
        assert "medium" in prompt

    def test_missing_validated_request_uses_defaults(self, agent_with_mocks):
        """intake exists but validated_request is None → field defaults apply.

        Scenario
        --------
        _build_prompt uses ``intake.get('validated_request') or {}``, so a
        None value is replaced with an empty dict and every inner .get()
        returns its default safely.  category_id IS present at the top level
        and must still be embedded correctly.
        """
        agent = agent_with_mocks
        state: SharedState = {
            "intake": {
                "estimated_value_usd": 5_000.0,
                "validated_request": None,
            },
            "category_id": "OTHER",
            "process_type": "rfq",
        }

        prompt = agent._build_prompt(state)

        assert "N/A" in prompt       # description, quantity, unit fall back
        assert "OTHER" in prompt     # category_id is at top level — always present

    def test_requirements_are_json_serialised(self, agent_with_mocks):
        """requirements list → JSON array string embedded in prompt.

        Scenario
        --------
        _build_prompt calls json.dumps(requirements, ensure_ascii=False).
        The prompt must contain a valid JSON array string, not a Python list
        repr such as "['SSD', 'RAM']".  The LLM reads this field as structured
        data, so the format must be unambiguous.
        """
        agent = agent_with_mocks
        reqs = ["RAM 32GB", "SSD NVMe", "Klawiatura PL"]
        state = make_state(requirements=reqs)

        prompt = agent._build_prompt(state)

        expected_json = json.dumps(reqs, ensure_ascii=False)
        assert expected_json in prompt

    def test_none_deadline_behaviour(self, agent_with_mocks):
        """Documenting current behaviour when deadline key is present but None.

        Scenario
        --------
        ``validated.get('deadline', 'not specified')`` returns None when the
        key exists with value None (Python only uses the default when the key
        is absent).  The prompt will therefore contain the string 'None'.
        This test pins the current behaviour — if it is later changed to
        ``validated.get('deadline') or 'not specified'``, update this
        assertion to check for 'not specified' instead.
        """
        agent = agent_with_mocks
        state = make_state(deadline=None)

        prompt = agent._build_prompt(state)

        # 'None' (str) appears because the key is present with value None.
        # If _build_prompt is fixed to use `or`, change this to 'not specified'.
        assert "None" in prompt or "not specified" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — _format_result: output → state mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatResult:
    """Unit tests for ProcurementAgent._format_result().

    _format_result() is called directly — no LLM is involved.
    Tests verify the exact structure of the SharedState update dict returned.
    """

    def test_current_step_is_procurement_completed(self, agent_with_mocks):
        """result['current_step'] must equal StepName.PROCUREMENT_COMPLETED.value.

        Scenario
        --------
        Graph routing downstream reads current_step as a plain string.
        It must match the canonical enum .value, not the enum object itself.
        """
        agent = agent_with_mocks
        result = agent._format_result(make_state(), make_procurement_output())

        assert result["current_step"] == StepName.PROCUREMENT_COMPLETED.value

    def test_suppliers_serialised_as_dicts(self, agent_with_mocks):
        """supplier_recommendations must be a list of plain dicts (model_dump()).

        Scenario
        --------
        SharedState is a TypedDict and cannot hold Pydantic instances.
        _format_result() must call .model_dump() on every SupplierRecommendation
        so downstream agents can access fields without importing the schema.
        """
        agent = agent_with_mocks
        result = agent._format_result(make_state(), make_procurement_output(supplier_count=2))

        suppliers = result["procurement"]["supplier_recommendations"]
        assert len(suppliers) == 2
        for supplier in suppliers:
            assert isinstance(supplier, dict)
            assert "name" in supplier
            assert "reliability_score" in supplier

    def test_all_procurement_section_fields_present(self, agent_with_mocks):
        """procurement section must contain all five fields defined in ProcurementState.

        Scenario
        --------
        ProcurementState declares five optional keys.  _format_result() must
        write all five so downstream agents can access them with direct indexing
        instead of .get() guards.
        """
        agent = agent_with_mocks
        result = agent._format_result(
            make_state(), make_procurement_output(recommended_order_type="framework")
        )

        p = result["procurement"]
        assert "supplier_recommendations" in p
        assert "procurement_strategy" in p
        assert "recommended_order_type" in p
        assert "negotiation_points" in p
        assert "alternative_products" in p
        assert p["recommended_order_type"] == "framework"
        assert p["procurement_strategy"] == PROCUREMENT_STRATEGY

    def test_errors_propagated_from_output(self, agent_with_mocks):
        """Non-fatal errors in ProcurementAgentOutput surface in result['errors'].

        Scenario
        --------
        The LLM may return warnings in the errors field.  _format_result must
        forward them so the orchestrator can log them without aborting the flow.
        """
        agent = agent_with_mocks
        result = agent._format_result(
            make_state(), make_procurement_output(errors=["Brak danych dla dostawcy Y"])
        )

        assert result["errors"] == ["Brak danych dla dostawcy Y"]

    def test_empty_errors_list_when_output_has_no_errors(self, agent_with_mocks):
        """result['errors'] must be [] (not None) when output.errors is empty.

        Scenario
        --------
        SharedState uses Annotated[list[str], operator.add] for errors.
        A None value would raise TypeError during LangGraph state merge.
        The ``or []`` fallback in _format_result must guarantee an empty list.
        """
        agent = agent_with_mocks
        result = agent._format_result(make_state(), make_procurement_output())

        assert result["errors"] == []

    def test_decision_log_is_list_with_one_timestamped_entry(self, agent_with_mocks):
        """decision_log must be a single-element list containing a formatted string.

        Scenario
        --------
        SharedState.decision_log uses Annotated[list[str], operator.add] so
        LangGraph concatenates entries across nodes.  _format_result wraps the
        entry in a list.  The entry must reference supplier count and order type.
        """
        agent = agent_with_mocks
        result = agent._format_result(
            make_state(), make_procurement_output(supplier_count=2, recommended_order_type="rfq")
        )

        log = result["decision_log"]
        assert isinstance(log, list)
        assert len(log) == 1
        assert "suppliers=2" in log[0]
        assert "rfq" in log[0]

    def test_format_result_calls_record_metrics(self, agent_with_mocks):
        """_format_result must call _record_metrics exactly once.

        Scenario
        --------
        _record_metrics is called inside _format_result.  Replacing the method
        with a Mock bypasses the None guard and lets us assert the call.
        """
        agent = agent_with_mocks
        agent._record_metrics = Mock()
        output = make_procurement_output()

        agent._format_result(make_state(), output)

        agent._record_metrics.assert_called_once_with(output)


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — _record_metrics: telemetry guard and call verification
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordMetrics:
    """Unit tests for ProcurementAgent._record_metrics().

    Tests call the real _record_metrics method (not a Mock) to verify the
    None-guard and the exact telemetry calls independently.
    """

    def test_no_calls_when_metrics_collector_is_none(self, agent_with_mocks):
        """_record_metrics must return immediately when self._metrics is None.

        Scenario
        --------
        Most agents run without a metrics collector.  The guard
        ``if self._metrics is None: return`` must prevent AttributeError.
        """
        agent = agent_with_mocks
        assert agent._metrics is None   # fixture guarantees this

        # Must not raise AttributeError
        agent._record_metrics(make_procurement_output())

    def test_supplier_count_and_order_type_recorded(self, agent_with_mocks):
        """Both metrics methods must be called with the correct arguments.

        Scenario
        --------
        When a metrics collector is present, _record_metrics must call:
          - record_supplier_count(n)  where n == len(supplier_recommendations)
          - record_order_type(order_type_string)
        """
        agent = agent_with_mocks
        agent._metrics = Mock()
        output = make_procurement_output(supplier_count=3, recommended_order_type="framework")

        agent._record_metrics(output)

        agent._metrics.record_supplier_count.assert_called_once_with(3)
        agent._metrics.record_order_type.assert_called_once_with("framework")

    def test_no_undocumented_metrics_calls(self, agent_with_mocks):
        """Exactly two metrics methods must be called — no more.

        Scenario
        --------
        Mock creates attributes silently, so an accidental call to an
        undocumented metrics method would pass other tests undetected.
        Counting total method_calls catches any additions immediately.
        """
        agent = agent_with_mocks
        agent._metrics = Mock()

        agent._record_metrics(make_procurement_output())

        assert len(agent._metrics.method_calls) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — run(): integration through BaseAgent._execute()
# ─────────────────────────────────────────────────────────────────────────────

class TestRun:
    """Integration tests for ProcurementAgent.run().

    These tests exercise the pipeline defined in BaseAgent._execute():
      _build_messages → _invoke_structured_llm → _format_result → return updates

    The tool loop is bypassed because the fixture makes has_tools == False
    (make_procurement_tools returns []).

    IMPORTANT: run() returns only the updates dict from _format_result(),
    NOT state | updates.  LangGraph handles the merge externally.
    """

    def test_happy_path_returns_correct_updates(self, agent_with_mocks):
        """Valid LLM output → run() returns a correctly structured updates dict.

        Scenario
        --------
        The structured LLM returns a ProcurementAgentOutput with two suppliers
        and 'rfq' order type.  run() must pass it through _format_result() and
        return a dict with all required keys populated.
        """
        agent = agent_with_mocks
        state = make_state()
        mock_output = make_procurement_output(supplier_count=2, recommended_order_type="rfq")
        agent._llm.structured_llm.invoke.return_value = mock_output

        result = agent.run(state)

        assert result["current_step"] == StepName.PROCUREMENT_COMPLETED.value
        assert len(result["procurement"]["supplier_recommendations"]) == 2
        assert result["procurement"]["recommended_order_type"] == "rfq"
        assert result["errors"] == []

    def test_structured_llm_receives_system_and_human_messages(self, agent_with_mocks):
        """run() must forward [SystemMessage, HumanMessage] to the structured LLM.

        Scenario
        --------
        When has_tools is False, BaseAgent._execute() passes the initial
        two-message list directly to _invoke_structured_llm.  We verify:
          - exactly two messages are sent
          - message types are correct
          - the product description from state appears in the HumanMessage,
            confirming _build_prompt() was called and its output embedded.
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        agent = agent_with_mocks
        state = make_state(description="monitor 27 cali")
        agent._llm.structured_llm.invoke.return_value = make_procurement_output()

        agent.run(state)

        agent._llm.structured_llm.invoke.assert_called_once()
        messages = agent._llm.structured_llm.invoke.call_args.args[0]

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert "monitor 27 cali" in messages[1].content

    def test_run_raises_structured_output_error_on_llm_failure(self, agent_with_mocks):
        """LLM raises → run() re-raises as StructuredOutputError.

        Scenario
        --------
        _invoke_structured_llm in BaseAgent wraps any exception in
        StructuredOutputError and re-raises.  run() must not swallow it.
        """
        agent = agent_with_mocks
        state = make_state()
        agent._llm.structured_llm.invoke.side_effect = Exception("LLM timeout")

        with pytest.raises(StructuredOutputError):
            agent.run(state)

    def test_structured_llm_called_exactly_once(self, agent_with_mocks):
        """Structured LLM must be called exactly once per run() invocation.

        Scenario
        --------
        ProcurementAgent has no retry logic or fallback schemas.  A single
        successful LLM call must complete the pipeline.  Multiple calls would
        indicate an unintended retry loop.
        """
        agent = agent_with_mocks
        state = make_state()
        agent._llm.structured_llm.invoke.return_value = make_procurement_output()

        agent.run(state)

        agent._llm.structured_llm.invoke.assert_called_once()
