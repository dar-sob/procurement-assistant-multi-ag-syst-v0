# procurement_system/agents/orchestrator/agent.py
"""
Orchestrator Agent — final agent in the procurement process.

Extends BaseAgent. No tools — compiles all upstream agent outputs into
a structured FinalReport and formulates a plain-language response.

Responsibilities:
    - Compile structured final report from all agent outputs
    - Formulate plain-language response for the requester
    - Write full audit decision log
    - Produce deterministic report_id with timestamp

Design note on report_id:
    report_id must appear both in the user prompt sent to the LLM and in
    the formatted state update. To ensure consistency across a single run,
    it is generated in _build_prompt() and stored in self._run_report_id.
    This pattern is safe because LangGraph graph nodes are never called
    concurrently on the same agent instance.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from procurement_system.constants import NodeName, REPORT_ID_PREFIX, StepName
from procurement_system.agents.base_agent import AgentConfig, BaseAgent
from procurement_system.schemas.orchestrator_schemas import LLMFinalReport, FinalReport
from procurement_system.tools import make_orchestrator_tools

from procurement_system.settings import (
    get_orchestrator_config,
    get_orchestrator_system_prompt,
    get_orchestrator_user_prompt,
)
from procurement_system.state import SharedState

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent — initialised once at graph build time.

    Uses the shared BaseAgent infrastructure with AgentConfig.
    No tools are used; the agent synthesises all upstream outputs via a
    single structured LLM call. All common LLM setup, tracing, and error
    handling are inherited from BaseAgent.
    """

    def __init__(
        self,
        repositories: Dict[str, Any] | None = None,
        services: Dict[str, Any] | None = None,
        tracer: Any | None = None,
        metrics: Any | None = None,
    ) -> None:
        """
        Initialise the Orchestrator Agent.

        Loads config, resolves prompts (with checksum validation), and
        delegates LLM setup to BaseAgent via AgentConfig.

        Args:
            repositories: Optional repository overrides.
            services: Optional service overrides.
            tracer: Optional OpenTelemetry tracer.
            metrics: Optional metrics collector.
        """
        cfg = get_orchestrator_config()["orchestrator"]

        # Prompt configuration and checksum validation
        prompt_cfg = cfg.get("prompt", {})
        checksums = prompt_cfg.get("checksums", {})
        system_prompt = get_orchestrator_system_prompt()
        user_template = get_orchestrator_user_prompt()
        tools=make_orchestrator_tools(services=services)

        report = cfg.get("report",{})
        self._report_id_prefix = report.get("id_prefix", REPORT_ID_PREFIX)
        # Generate and persist report_id for this run
        self._run_report_id = (
            f"{self._report_id_prefix}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )
        # Determine model tier with safe fallback
        model_tier = cfg.get("model_tier")

        if not model_tier:
            logger.warning(
                "[%s] 'model_tier' not defined in configuration. "
                "Falling back to 'fast_cheap' tier.",
                NodeName.ORCHESTRATOR.value,
            )
            model_tier = "fast_cheap"

        # Build immutable agent configuration
        agent_config = AgentConfig(
            name=NodeName.ORCHESTRATOR.value,
            output_schema=LLMFinalReport,
            system_prompt=system_prompt,
            user_template=user_template,
            model_tier=model_tier,
            tools=tools,  # Currently empty list, but kept for consistency
        )

        super().__init__(
            config=agent_config,
            repositories=repositories,
            services=services,
            tracer=tracer,
            metrics=metrics,
        )

        # Populated in _build_prompt() and consumed in _format_result().
        # Safe for single-threaded graph execution — see module docstring.
        self._run_report_id: str = ""

        logger.info(
            "[%s] OrchestratorAgent initialized — model_tier='%s'",
            self.name,
            model_tier,
        )

    # ------------------------------------------------------------------
    # BaseAgent abstract method implementations
    # ------------------------------------------------------------------

    def _build_prompt(self, state: SharedState) -> str:
        """
        Build user prompt from all upstream agent sections of SharedState.
        """

        intake_section = state.get("intake", {})
        validated: Dict[str, Any] = intake_section.get("validated_request") or {}

        procurement_section = state.get("procurement", {})

        analyst_section = state.get("analyst", {})

        return self.config.user_template.format(
            product_name=validated.get("description", "N/A"),
            category=state.get("category_id", "N/A"),
            quantity=validated.get("quantity", "N/A"),
            unit=validated.get("unit", "N/A"),
            estimated_value_usd=intake_section.get("estimated_value_usd", "unknown"),
            process_type=state.get("process_type", "N/A"),
            framework_agreement_id=intake_section.get("framework_agreement_id", "none"),
            urgency=validated.get("urgency", "medium"),
            deadline=validated.get("deadline", "not specified"),
            procurement_strategy=procurement_section.get("procurement_strategy", "N/A"),
            supplier_recommendations=json.dumps(
                procurement_section.get("supplier_recommendations", []),
                ensure_ascii=False,
                indent=2,
            ),            
            negotiation_points=json.dumps(
                procurement_section.get("negotiation_points", []),
                ensure_ascii=False,
                indent=2,
            ),
            cost_analysis=json.dumps(
                analyst_section.get("cost_analysis", {}), 
                ensure_ascii=False, 
                indent=2
            ),
            risk_analysis=json.dumps(
                analyst_section.get("risk_analysis", {}), 
                ensure_ascii=False, 
                indent=2
            ),
            final_recommendation=json.dumps(
                analyst_section.get("final_recommendation", {}),
                ensure_ascii=False,
                indent=2,
            ),
            final_decision=state.get("final_decision", "UNKNOWN"),
            decision_log=json.dumps(
                state.get("decision_log", []), 
                ensure_ascii=False, 
                indent=2
            ),
        )

    def _format_result(
        self, state: SharedState, parsed_output: LLMFinalReport
    ) -> Dict[str, Any]:
        """
        Convert LLM-validated output to SharedState update with system fields enforced.

        System-controlled fields (report_id, generated_at) are added here to guarantee
        consistency and prevent any LLM hallucination or modification.
        """
        # Create final report with system metadata
        final_report = FinalReport(
            **parsed_output.model_dump(),
            report_id=self._run_report_id,
            generated_at=datetime.now(timezone.utc),
            # run_id=state.get("run_id") if you implement global run_id later
        )

        message_to_user = _build_message_to_user(final_report)

        log_entry = self._make_log_entry(
            message=(
                f"report_id={self._run_report_id} "
                f"decision={final_report.final_decision}"
            )
        )

        orchestrator_data: Dict[str, Any] = {
            "final_report": final_report.model_dump(mode="json"),
        }

        # Record business metrics if collector is available
        self._record_metrics(parsed_output)

        return {
            "current_step": StepName.ORCHESTRATOR_COMPLETED.value,
            "final_decision": final_report.final_decision,
            "message_to_user": message_to_user,
            "decision_log": [log_entry],
            "errors": [],  # No errors from successful formatting
            "orchestrator": orchestrator_data,
        }

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _record_metrics(self, parsed_output: FinalReport) -> None:
        """Record orchestrator-specific metrics if metrics collector is present."""
        if self._metrics is None:
            return
        self._metrics.record_decision(parsed_output.final_decision)
        self._metrics.record_report_generated(self._run_report_id)

    def _make_log_entry(self, message: str) -> str:
        """Create a standardized, timestamped log entry for decision audit trail."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return f"[{ts} UTC] {self.name.upper()}: {message}"


# ------------------------------------------------------------------
# Module-level helpers (pure functions, easily testable)
# ------------------------------------------------------------------

def _build_message_to_user(final_report: FinalReport) -> str:
    """
    Assemble a plain-language response string from a FinalReport.

    Extracted as a module-level function so it can be unit-tested
    independently of the OrchestratorAgent class.
    """
    next_steps_text = "\n".join(
        f"{i + 1}. {step}" for i, step in enumerate(final_report.next_steps)
    )
    return (
        f"Your procurement request has been processed.\n\n"
        f"Decision: {final_report.final_decision}\n"
        f"Recommended supplier: {final_report.recommended_supplier or 'N/A'}\n"
        # f"Estimated cost: ${final_report.cost_analysis["total_estimated_cost"]:,.0f}\n\n"
        f"Estimated cost: ${final_report.cost_analysis}\n\n"
        f"Next steps:\n{next_steps_text}"
    )
