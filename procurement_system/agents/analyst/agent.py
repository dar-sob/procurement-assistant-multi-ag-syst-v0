# procurement_system/agents/analyst/agent.py
"""
Analyst Agent (Cost & Risk Analyst) — third agent in the procurement workflow.

This agent:
- Performs Total Cost of Ownership (TCO) analysis with three scenarios
  (optimistic / realistic / pessimistic)
- Identifies and scores individual risks + computes overall risk level
- Produces a final, auditable recommendation:
  PROCEED / PROCEED_WITH_CONDITIONS / ESCALATE / REJECT
- Provides concrete next steps for the requester

Uses the shared BaseAgent infrastructure for consistent prompt handling,
structured output parsing, tool-loop support (currently none), tracing,
and observability.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from procurement_system.schemas.analyst_schemas import AnalystAgentOutput
from procurement_system.agents.base_agent import AgentConfig, BaseAgent
from procurement_system.tools import make_analyst_tools
from procurement_system.state import SharedState

from procurement_system.constants import (
    NodeName, 
    StepName, 
    AUTO_PROCEED_MAX_THRESHOLD, 
    AUTO_ESCALATE_MIN_THRESHOLD
)

from procurement_system.settings import (
    get_analyst_config,
    get_analyst_system_prompt,
    get_analyst_user_prompt,
)


logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):

    """
    Production-grade Analyst Agent.

    Responsible for TCO calculation, comprehensive risk analysis,
    and issuing a final, auditable procurement decision with justification
    and next steps.
    """

    def __init__(
        self,
        repositories: Dict[str, Any] | None = None,
        services: Dict[str, Any] | None = None,
        tracer: Any | None = None,
        metrics: Any | None = None,
    ) -> None:

        """
        Initialize the Analyst Agent from centralized configuration.

        Model selection is driven by `model_tier` defined in
        config_analyst_agent.yaml. Uses 'reasoning_heavy' tier by default
        for high-stakes financial/risk decisions.
        """

        cfg = get_analyst_config()["analyst_agent"]

        # Prompt configuration and checksum validation
        prompt_cfg = cfg.get("prompt", {})
        checksums = prompt_cfg.get("checksums", {})
        system_prompt = get_analyst_system_prompt()
        user_template = get_analyst_user_prompt()
        tools=make_analyst_tools(services=services)

        risk_score = cfg.get("risk_score", {})
        self._auto_proceed_max_threshold = risk_score.get(
            "auto_proceed", 
            AUTO_PROCEED_MAX_THRESHOLD
        )
        self._auto_escalate_min_threshold = risk_score.get(
            "auto_escalate", 
            AUTO_ESCALATE_MIN_THRESHOLD
        )

        # Determine model tier with safe fallback
        model_tier = cfg.get("model_tier")
        if not model_tier:
            logger.warning(
                "[%s] 'model_tier' not defined in configuration. "
                "Falling back to 'reasoning_heavy' tier.",
                NodeName.ANALYST.value,
            )
            model_tier = "reasoning_heavy"

        # Build immutable agent configuration
        agent_config = AgentConfig(
            name=NodeName.ANALYST.value,
            output_schema=AnalystAgentOutput,
            system_prompt=system_prompt,
            user_template=user_template,
            model_tier=model_tier,
            tools=tools,
        )

        super().__init__(
            config=agent_config,
            repositories=repositories,
            services=services,
            tracer=tracer,
            metrics=metrics,
        )

        logger.info(
            "[%s] AnalystAgent initialized — model_tier='%s'",
            self.name,
            model_tier,
        )

    # ------------------------------------------------------------------
    # BaseAgent abstract method implementations
    # ------------------------------------------------------------------

    def _build_prompt(self, state: SharedState) -> str:
        """
        Construct the user prompt from the current SharedState.

        Extracts validated request data (supports both legacy direct key
        and new "intake" structure) and supplier recommendations
        from the previous procurement step.
        """
        # Support both possible state shapes for robustness
        intake = state.get("intake", {}) or {}
        validated = intake.get("validated_request") or state.get("validated_request", {})

        # Supplier recommendations may live under "procurement" (new structure)
        # or directly in state (legacy)
        procurement = state.get("procurement", {}) or {}
        suppliers = (
            procurement.get("supplier_recommendations")
            or state.get("supplier_recommendations")
            or []
        )

        return self.config.user_template.format(
            product_name=validated.get("description", "N/A"),
            category=state.get("category_id", "N/A"),
            quantity=validated.get("quantity", "N/A"),
            unit=validated.get("unit", "N/A"),
            estimated_budget=intake.get("estimated_value_usd", "unknown"),
            process_type=state.get("process_type", "N/A"),
            requirements=json.dumps(
                validated.get("requirements", []), ensure_ascii=False
            ),
            deadline=validated.get("deadline", "not specified"),
            urgency=validated.get("urgency", "medium"),
            suppliers=json.dumps(suppliers, ensure_ascii=False, indent=2),
            auto_proceed=self._auto_proceed_max_threshold,
            auto_escalate=self._auto_escalate_min_threshold, 
        )

    def _format_result(
        self, state: SharedState, parsed_output: AnalystAgentOutput
    ) -> Dict[str, Any]:
        """
        Map the structured LLM output to SharedState updates.

        Updates only the fields defined in the agent contract
        (see config_analyst_agent.yaml → output_state_fields).
        """
        decision = parsed_output.final_recommendation.decision

        log_entry = self._make_log_entry(
            message=(
                f"decision={decision} "
                f"risk={parsed_output.risk_analysis.overall_risk_level} "
                f"risk_score={parsed_output.risk_analysis.risk_score}/10 "
                f"total_cost=${getattr(parsed_output.cost_analysis, 'total_estimated_cost', 0):,.0f}"
            )
        )

        analyst_data: Dict[str, Any] = {
            "cost_analysis": parsed_output.cost_analysis.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "risk_analysis": parsed_output.risk_analysis.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "final_recommendation": parsed_output.final_recommendation.model_dump(
                mode="json",
                exclude_none=True,
            ),
        }

        # Record business metrics if collector is available
        self._record_metrics(parsed_output)

        return {
            "current_step": StepName.ANALYST_COMPLETED.value,
            "final_decision": decision,
            "decision_log": [log_entry],
            "errors": parsed_output.errors or [],
            "analyst": analyst_data,
        }

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _record_metrics(self, parsed_output: AnalystAgentOutput) -> None:
        """Record analyst-specific metrics if a metrics collector is provided."""
        if self._metrics is None:
            return

        self._metrics.record_decision(parsed_output.final_recommendation.decision)
        self._metrics.record_risk_score(parsed_output.risk_analysis.risk_score)

    def _make_log_entry(self, message: str) -> str:
        """Create a standardized, timestamped log entry for decision audit trail."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return f"[{ts} UTC] {self.name.upper()}: {message}"
