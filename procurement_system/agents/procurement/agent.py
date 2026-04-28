# procurement_system/agents/procurement/agent.py
"""
Procurement Agent — responsible for supplier discovery, ranking,
and building a procurement strategy.

This agent:
- Leverages tools to search and retrieve supplier information
- Analyzes data in the context of the intake request
- Produces a structured, validated output using ProcurementAgentOutput schema
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from procurement_system.schemas.procurement_schemas import ProcurementAgentOutput
from procurement_system.agents.base_agent import AgentConfig, BaseAgent
from procurement_system.constants import NodeName, StepName
from procurement_system.tools import make_procurement_tools
from procurement_system.state import SharedState

from procurement_system.settings import (
    get_procurement_config,
    get_procurement_system_prompt,
    get_procurement_user_prompt,
)


logger = logging.getLogger(__name__)


class ProcurementAgent(BaseAgent):
    """
    Production-grade Procurement Agent.

    Responsible for identifying suitable suppliers, recommending an order type,
    defining a procurement strategy, and preparing negotiation points.
    """

    def __init__(
        self,
        repositories: Dict[str, Any] | None = None,
        services: Dict[str, Any] | None = None,
        tracer: Any | None = None,
        metrics: Any | None = None,
    ) -> None:
        """
        Initialize the Procurement Agent from centralized configuration.

        Model selection is driven by `model_tier` defined in
        config_procurement_agent.yaml. If the tier is not specified,
        a safe default is used with a warning logged.
        """
        cfg = get_procurement_config()["procurement_agent"]

        # Prompt configuration and checksum validation
        prompt_cfg = cfg.get("prompt", {})        
        checksums = prompt_cfg.get("checksums", {})

        system_prompt = get_procurement_system_prompt()

        user_template = get_procurement_user_prompt()

        # Determine model tier with safe fallback
        model_tier = cfg.get("model_tier")

        if not model_tier:
            logger.warning(
                "[%s] 'model_tier' not defined in configuration. "
                "Falling back to 'balanced' tier.",
                NodeName.PROCUREMENT.value,
            )
            model_tier = "balanced"  # Safe, well-balanced default for this agent

        tools = make_procurement_tools(services=services)

        # Build immutable agent configuration
        agent_config = AgentConfig(
            name=NodeName.PROCUREMENT.value,
            output_schema=ProcurementAgentOutput,
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
            "[%s] ProcurementAgent initialized — model_tier='%s', tools=%d",
            self.name,
            model_tier,
            len(tools),
        )

    # ------------------------------------------------------------------
    # BaseAgent abstract method implementations
    # ------------------------------------------------------------------

    def _build_prompt(self, state: SharedState) -> str:
        """
        Construct the user prompt from the current SharedState.

        Extracts relevant data from the intake section (validated request)
        and formats it using the configured user template.
        """
        intake = state.get("intake", {}) or {}
        validated = intake.get("validated_request") or {}

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
        )

    def _format_result(
        self, state: SharedState, parsed_output: ProcurementAgentOutput
    ) -> Dict[str, Any]:
        """
        Map the structured LLM output to SharedState updates.

        Updates only the fields defined in the agent contract.
        """
        log_entry = self._make_log_entry(
            message=(
                f"suppliers={len(parsed_output.supplier_recommendations)} "
                f"order_type={parsed_output.recommended_order_type}"
            )
        )

        procurement_data: Dict[str, Any] = {
            "supplier_recommendations": [
                supplier.model_dump() for supplier in parsed_output.supplier_recommendations
            ],
            "procurement_strategy": parsed_output.procurement_strategy,
            "recommended_order_type": parsed_output.recommended_order_type,
            "negotiation_points": parsed_output.negotiation_points,
            "alternative_products": parsed_output.alternative_products,
        }

        # Record business metrics if collector is available
        self._record_metrics(parsed_output)

        return {
            "current_step": StepName.PROCUREMENT_COMPLETED.value,
            "decision_log": [log_entry],
            "errors": parsed_output.errors or [],
            "procurement": procurement_data,
        }

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _record_metrics(self, parsed_output: ProcurementAgentOutput) -> None:
        """Record procurement-specific metrics if a metrics collector is provided."""
        if self._metrics is None:
            return

        self._metrics.record_supplier_count(len(parsed_output.supplier_recommendations))
        self._metrics.record_order_type(parsed_output.recommended_order_type)

    def _make_log_entry(self, message: str) -> str:
        """Create a standardized, timestamped log entry for decision audit trail."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return f"[{ts} UTC] {self.name.upper()}: {message}"
