# tests/agents/test_analyst_agent.py
# pytest tests/agents/test_analyst_agent.py -v
# pytest tests/agents/test_analyst_agent.py -v --cache-clear

import json
import pytest
from unittest.mock import Mock, patch

from procurement_system.agents.analyst.agent import AnalystAgent
from procurement_system.exceptions import StructuredOutputError
from procurement_system.schemas.analyst_schemas import (
    AnalystAgentOutput,
    CostAnalysis,
    CostScenarios,
    FinalRecommendation,
    RiskAnalysis,
    RiskItem,
)
from procurement_system.state import SharedState
from procurement_system.constants import StepName

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

TEST_USER_TEMPLATE = (
    "product={product_name} category={category} "
    "qty={quantity} unit={unit} budget={estimated_budget} "
    "process={process_type} req={requirements} "
    "deadline={deadline} urgency={urgency} "
    "suppliers={suppliers} auto_proceed={auto_proceed} auto_escalate={auto_escalate}"
)
"""Minimal user-prompt template matching all placeholders used in _build_prompt."""

# -----------------------------------------------------------------------------
# Patch targets – patch where names are looked up
# -----------------------------------------------------------------------------

_BASE = "procurement_system.agents.base_agent"
_AGENT = "procurement_system.agents.analyst.agent"

PATCH_BUILD_LLM_BUNDLE  = f"{_BASE}.build_agent_llm_bundle"
PATCH_RESOLVE_FROM_TIER = f"{_BASE}.resolve_from_tier"
PATCH_GET_CONFIG        = f"{_AGENT}.get_analyst_config"
PATCH_GET_SYS_PROMPT    = f"{_AGENT}.get_analyst_system_prompt"
PATCH_GET_USER_PROMPT   = f"{_AGENT}.get_analyst_user_prompt"
PATCH_MAKE_TOOLS        = f"{_AGENT}.make_analyst_tools"

# -----------------------------------------------------------------------------
# Helpers – state factory
# -----------------------------------------------------------------------------

def make_state(
    *,
    process_type: str = "rfq",
    category_id: str = "IT_HARDWARE",
    description: str = "laptop Dell XPS",
    quantity: float = 10.0,
    unit: str = "szt",
    estimated_value_usd: float = 15_000.0,
    requirements: list | None = None,
    deadline: str | None = None,
    urgency: str = "medium",
    supplier_recommendations: list | None = None,
) -> SharedState:
    """
    Return a SharedState that mirrors real post‑intake, post‑procurement state.

    AnalystAgent._build_prompt reads from:
      - state["intake"]["validated_request"]   (description, quantity, unit, …)
      - state["intake"]["estimated_value_usd"] (budget)
      - state["category_id"] and state["process_type"]
      - state["procurement"]["supplier_recommendations"] (or legacy top-level)
    """
    suppliers = supplier_recommendations or [
        {"name": "TechDirect", "reliability_score": 8.5}
    ]
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
        "procurement": {
            "supplier_recommendations": suppliers,
        },
    }

# -----------------------------------------------------------------------------
# Helpers – output factories
# -----------------------------------------------------------------------------

def make_cost_analysis(
    *,
    unit_cost: float = 1200.0,
    total_cost: float = 12_000.0,
    optimistic: float = 10_000.0,
    realistic: float = 12_000.0,
    pessimistic: float = 15_000.0,
    budget_adequacy: str = "sufficient",
) -> CostAnalysis:
    return CostAnalysis(
        estimated_unit_cost=unit_cost,
        total_estimated_cost=total_cost,
        tco_factors=["maintenance", "training", "support"],
        cost_scenarios=CostScenarios(
            optimistic=optimistic,
            realistic=realistic,
            pessimistic=pessimistic,
        ),
        budget_adequacy=budget_adequacy,  # type: ignore
        potential_savings=500.0,
    )

def make_risk_analysis(
    *,
    overall_risk: str = "low",
    risk_score: float = 2.5,
) -> RiskAnalysis:
    return RiskAnalysis(
        overall_risk_level=overall_risk,  # type: ignore
        risk_score=risk_score,
        risks=[
            RiskItem(
                risk_name="Dostawca opóźnia dostawę",
                category="delivery",
                probability=0.3,
                impact=4.0,
                mitigation="Wprowadzić kary umowne",
            )
        ],
    )

def make_final_recommendation(
    *,
    decision: str = "PROCEED",
    priority_supplier: str | None = "TechDirect",
) -> FinalRecommendation:
    return FinalRecommendation(
        decision=decision,  # type: ignore
        justification="Niski poziom ryzyka i koszt mieszczący się w budżecie. "
                      "Dostawca ma dobre referencje i krótki czas realizacji.",
        conditions=[] if decision != "PROCEED_WITH_CONDITIONS" else ["Przedłużyć gwarancję"],
        priority_supplier=priority_supplier,
        next_steps=["Podpisać umowę", "Zlecić dostawę"],
    )

def make_analyst_output(
    *,
    decision: str = "PROCEED",
    overall_risk: str = "low",
    risk_score: float = 2.5,
    errors: list | None = None,
) -> AnalystAgentOutput:
    return AnalystAgentOutput(
        cost_analysis=make_cost_analysis(),
        risk_analysis=make_risk_analysis(overall_risk=overall_risk, risk_score=risk_score),
        final_recommendation=make_final_recommendation(decision=decision),
        errors=errors or [],
    )

# -----------------------------------------------------------------------------
# Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def agent_with_mocks():
    """Create an AnalystAgent with all external dependencies replaced by Mocks."""
    config_stub = {
        "analyst_agent": {
            "model_tier": "reasoning_heavy",
            "risk_score": {
                "auto_proceed": 3.0,
                "auto_escalate": 7.0,
            },
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
        patch(PATCH_MAKE_TOOLS, return_value=[]),   # no tools => simple path
    ):
        mock_bundle.return_value = Mock()
        mock_bundle.return_value.structured_llm = Mock()
        mock_bundle.return_value.llm_with_tools = Mock()
        agent = AnalystAgent()

    # Replace LLM bundle for per‑test control
    agent._llm = Mock()
    agent._llm.structured_llm = Mock()
    agent._metrics = None   # default: no telemetry

    return agent

# -----------------------------------------------------------------------------
# Group 1 – _build_prompt: field extraction and defaults
# -----------------------------------------------------------------------------

class TestBuildPrompt:
    """Unit tests for AnalystAgent._build_prompt()."""

    def test_all_fields_present(self, agent_with_mocks):
        """All intake and procurement fields appear in the prompt."""
        agent = agent_with_mocks
        suppliers = [{"name": "IBM", "score": 9}]
        state = make_state(
            description="serwer rack",
            quantity=2,
            unit="szt",
            category_id="SERVER",
            process_type="rfq",
            estimated_value_usd=25_000,
            requirements=["CPU 32 rdzenie", "RAM 128GB"],
            deadline="2025-08-01",
            urgency="high",
            supplier_recommendations=suppliers,
        )

        prompt = agent._build_prompt(state)

        assert "serwer rack" in prompt
        assert "SERVER" in prompt
        assert "2" in prompt
        assert "25000" in prompt
        assert "rfq" in prompt
        assert "CPU 32 rdzenie" in prompt
        assert "2025-08-01" in prompt
        assert "high" in prompt

        # Match the agent's serialisation (indent=2)
        expected_suppliers = json.dumps(suppliers, ensure_ascii=False, indent=2)
        assert expected_suppliers in prompt
        assert "auto_proceed=3.0" in prompt
        assert "auto_escalate=7.0" in prompt

    def test_missing_intake_uses_defaults(self, agent_with_mocks):
        """No intake section → fallbacks (N/A, unknown, not specified, medium)."""
        agent = agent_with_mocks
        state: SharedState = {
            "category_id": "UNKNOWN",
            "process_type": "rfq",
        }

        prompt = agent._build_prompt(state)

        assert "N/A" in prompt
        assert "unknown" in prompt
        assert "not specified" in prompt
        assert "medium" in prompt

    def test_missing_validated_request_uses_defaults(self, agent_with_mocks):
        """intake exists but validated_request is None → defaults for product fields."""
        agent = agent_with_mocks
        state: SharedState = {
            "intake": {
                "estimated_value_usd": 10_000,
                "validated_request": None,
            },
            "category_id": "OTHER",
            "process_type": "rfq",
            "procurement": {"supplier_recommendations": []},
        }

        prompt = agent._build_prompt(state)

        assert "N/A" in prompt   # description, quantity, unit
        assert "OTHER" in prompt
        assert "auto_proceed=3.0" in prompt

    def test_suppliers_json_serialised(self, agent_with_mocks):
        agent = agent_with_mocks
        suppliers = [
            {"name": "Dell", "reliability_score": 9.2},
            {"name": "HP", "reliability_score": 8.5},
        ]
        state = make_state(supplier_recommendations=suppliers)
        prompt = agent._build_prompt(state)

        for supplier in suppliers:
            assert str(supplier["name"]) in prompt
            assert str(supplier["reliability_score"]) in prompt
        # Also verify JSON structure (optional)
        assert '"name": "Dell"' in prompt
        assert '"reliability_score": 9.2' in prompt

    def test_suppliers_json_serialised(self, agent_with_mocks):
        """supplier_recommendations list is JSON‑serialised."""
        agent = agent_with_mocks
        suppliers = [
            {"name": "Dell", "reliability_score": 9.2},
            {"name": "HP", "reliability_score": 8.5},
        ]
        state = make_state(supplier_recommendations=suppliers)

        prompt = agent._build_prompt(state)
        expected = json.dumps(suppliers, ensure_ascii=False, indent=2)

        assert expected in prompt

    def test_suppliers_from_legacy_state_key(self, agent_with_mocks):
        """If state["procurement"] missing, fall back to state["supplier_recommendations"]."""
        agent = agent_with_mocks
        state: SharedState = {
            "process_type": "rfq",
            "category_id": "IT",
            "intake": {"validated_request": {}, "estimated_value_usd": 1000},
            "supplier_recommendations": [{"name": "LegacySupplier"}],
        }

        prompt = agent._build_prompt(state)
        assert "LegacySupplier" in prompt

# -----------------------------------------------------------------------------
# Group 2 – _format_result: output → state mapping
# -----------------------------------------------------------------------------

class TestFormatResult:
    """Unit tests for AnalystAgent._format_result()."""

    def test_current_step_is_analyst_completed(self, agent_with_mocks):
        """result['current_step'] must equal StepName.ANALYST_COMPLETED.value."""
        agent = agent_with_mocks
        result = agent._format_result(make_state(), make_analyst_output())

        assert result["current_step"] == StepName.ANALYST_COMPLETED.value

    def test_final_decision_propagated(self, agent_with_mocks):
        """result['final_decision'] equals the decision from recommendation."""
        agent = agent_with_mocks
        result = agent._format_result(
            make_state(),
            make_analyst_output(decision="PROCEED_WITH_CONDITIONS")
        )
        assert result["final_decision"] == "PROCEED_WITH_CONDITIONS"

    def test_analyst_section_contains_three_dumped_models(self, agent_with_mocks):
        """analyst dict must contain cost_analysis, risk_analysis, final_recommendation as dicts."""
        agent = agent_with_mocks
        output = make_analyst_output()
        result = agent._format_result(make_state(), output)

        analyst = result["analyst"]
        assert "cost_analysis" in analyst
        assert "risk_analysis" in analyst
        assert "final_recommendation" in analyst
        assert isinstance(analyst["cost_analysis"], dict)
        assert isinstance(analyst["risk_analysis"], dict)
        assert isinstance(analyst["final_recommendation"], dict)
        assert analyst["cost_analysis"]["total_estimated_cost"] == 12_000

    def test_decision_log_contains_one_timestamped_entry(self, agent_with_mocks):
        """Log entry includes decision, risk level, risk score, total cost."""
        agent = agent_with_mocks
        output = make_analyst_output(
            decision="ESCALATE",
            overall_risk="critical",
            risk_score=9.0,
        )
        result = agent._format_result(make_state(), output)

        log = result["decision_log"]
        assert len(log) == 1
        entry = log[0]
        assert "ESCALATE" in entry
        assert "critical" in entry
        assert "risk_score=9.0/10" in entry
        assert "total_cost=$12,000" in entry

    def test_errors_propagated_from_output(self, agent_with_mocks):
        """Non‑fatal errors in output appear in result['errors']."""
        agent = agent_with_mocks
        result = agent._format_result(
            make_state(),
            make_analyst_output(errors=["Nie udało się oszacować kosztu serwisu"])
        )
        assert result["errors"] == ["Nie udało się oszacować kosztu serwisu"]

    def test_empty_errors_list_when_output_has_no_errors(self, agent_with_mocks):
        """result['errors'] must be [] (not None) when output.errors is empty."""
        agent = agent_with_mocks
        result = agent._format_result(make_state(), make_analyst_output())
        assert result["errors"] == []

    def test_format_result_calls_record_metrics(self, agent_with_mocks):
        """_format_result must call _record_metrics exactly once."""
        agent = agent_with_mocks
        agent._record_metrics = Mock()
        output = make_analyst_output()

        agent._format_result(make_state(), output)

        agent._record_metrics.assert_called_once_with(output)

# -----------------------------------------------------------------------------
# Group 3 – _record_metrics: telemetry guard and calls
# -----------------------------------------------------------------------------

class TestRecordMetrics:
    """Unit tests for AnalystAgent._record_metrics()."""

    def test_no_calls_when_metrics_collector_is_none(self, agent_with_mocks):
        """Guard returns early when self._metrics is None."""
        agent = agent_with_mocks
        assert agent._metrics is None
        # Should not raise AttributeError
        agent._record_metrics(make_analyst_output())

    def test_decision_and_risk_score_recorded(self, agent_with_mocks):
        """Both metrics methods are called with correct arguments."""
        agent = agent_with_mocks
        agent._metrics = Mock()
        output = make_analyst_output(decision="PROCEED", risk_score=4.5)

        agent._record_metrics(output)

        agent._metrics.record_decision.assert_called_once_with("PROCEED")
        agent._metrics.record_risk_score.assert_called_once_with(4.5)

    def test_no_extra_metrics_calls(self, agent_with_mocks):
        """Exactly two metrics methods must be called."""
        agent = agent_with_mocks
        agent._metrics = Mock()

        agent._record_metrics(make_analyst_output())

        assert len(agent._metrics.method_calls) == 2

# -----------------------------------------------------------------------------
# Group 4 – run(): integration through BaseAgent._execute()
# -----------------------------------------------------------------------------

class TestRun:
    """Integration tests for AnalystAgent.run()."""

    def test_happy_path_returns_correct_updates(self, agent_with_mocks):
        """Valid LLM output → run() returns updates dict with all keys."""
        agent = agent_with_mocks
        state = make_state()
        mock_output = make_analyst_output(decision="REJECT", overall_risk="high", risk_score=7.5)
        agent._llm.structured_llm.invoke.return_value = mock_output

        result = agent.run(state)

        assert result["current_step"] == StepName.ANALYST_COMPLETED.value
        assert result["final_decision"] == "REJECT"
        assert result["analyst"]["risk_analysis"]["overall_risk_level"] == "high"
        assert result["errors"] == []

    def test_structured_llm_receives_system_and_human_messages(self, agent_with_mocks):
        """run() forwards [SystemMessage, HumanMessage] to structured LLM."""
        from langchain_core.messages import SystemMessage, HumanMessage

        agent = agent_with_mocks
        state = make_state(description="drukarka 3D")
        agent._llm.structured_llm.invoke.return_value = make_analyst_output()

        agent.run(state)

        agent._llm.structured_llm.invoke.assert_called_once()
        messages = agent._llm.structured_llm.invoke.call_args.args[0]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert "drukarka 3D" in messages[1].content

    def test_run_raises_structured_output_error_on_llm_failure(self, agent_with_mocks):
        """LLM exception is wrapped as StructuredOutputError."""
        agent = agent_with_mocks
        state = make_state()
        agent._llm.structured_llm.invoke.side_effect = Exception("Model overload")

        with pytest.raises(StructuredOutputError):
            agent.run(state)

    def test_structured_llm_called_exactly_once(self, agent_with_mocks):
        """No accidental retries – single LLM call per run()."""
        agent = agent_with_mocks
        state = make_state()
        agent._llm.structured_llm.invoke.return_value = make_analyst_output()

        agent.run(state)

        agent._llm.structured_llm.invoke.assert_called_once()
