# procurement_system/agents/intake/agent.py
"""
Intake Agent — first agent in the procurement process.

Extends BaseAgent. Overrides run() to support a dual-schema LLM flow:
    - Primary schema:  IntakeAgentOutput     (normal completion)
    - Fallback schema: ClarificationRequest  (missing required fields)

Responsibilities:
    - Parse and validate the purchase request
    - Apply enterprise buying rules (injected into the prompt at runtime)
    - Classify the procurement category
    - Determine the process type from value thresholds
    - Check framework agreements
    - Apply escalation rules
    - Route: proceed to Procurement Agent or escalate
    - Request clarification via interrupt() when required fields are absent
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from procurement_system.state import SharedState
from procurement_system.utils.buying_rules_prompt_builder import build_buying_rules_text
from procurement_system.utils.agent_llm_bundle import build_agent_llm_bundle
from procurement_system.agents.base_agent import AgentConfig, BaseAgent
from procurement_system.exceptions import StructuredOutputError
from procurement_system.tools import make_intake_tools

from procurement_system.constants import (
    MAX_CLARIFICATION_ROUNDS,
    NodeName,
    StepName,
)
from procurement_system.schemas.intake_schemas import (
    ClarificationRequest,
    IntakeAgentOutput,
)
from procurement_system.settings import (
    get_buying_rules,
    get_intake_config,
    get_intake_system_prompt,
    get_intake_user_prompt,
)


logger = logging.getLogger(__name__)


class IntakeAgent(BaseAgent):

    """
    Intake Agent — initialized once at graph build time.

    Inherits common infrastructure from BaseAgent (LLM initialization,
    model-chain resolution, tracing, tool loop, structured output) and
    overrides run() to implement the dual-schema fallback logic specific
    to intake processing.
    """

    def __init__(
        self,
        repositories: Dict[str, Any] | None = None,
        services: Dict[str, Any] | None = None,
        tracer: Any | None = None,
        metrics: Any | None = None,
    ) -> None:

        """
        Initialize the Intake Agent from centralized configuration.

        Model selection is driven by ``model_tier`` defined in
        config_intake_agent.yaml. If the tier is absent, a safe default
        is used with a warning logged.

        After BaseAgent setup, a secondary ClarificationRequest LLM is
        built by reusing the already-resolved model-chain primary config,
        avoiding a redundant registry lookup.

        Args:
            repositories: Optional repository overrides (forwarded to BaseAgent).
            services:     Optional service overrides (forwarded to BaseAgent).
            tracer:       Optional OpenTelemetry tracer (forwarded to BaseAgent).
            metrics:      Optional metrics collector (forwarded to BaseAgent).
        """

        cfg = get_intake_config()["intake_agent"]
        prompt_cfg = cfg.get("prompt", {})
        checksums = prompt_cfg.get("checksums", {})
        clarification_cfg = cfg.get("clarification", {})

        system_prompt = get_intake_system_prompt()
        user_template = get_intake_user_prompt()

        # Determine model tier with safe fallback
        model_tier = cfg.get("model_tier")
        if not model_tier:
            logger.warning(
                "[%s] 'model_tier' not defined in configuration. "
                "Falling back to 'fast_cheap' tier.",
                NodeName.INTAKE.value,
            )
            model_tier = "fast_cheap"

        tools = make_intake_tools(services=services)

        # Build immutable agent configuration
        agent_config = AgentConfig(
            name=NodeName.INTAKE.value,
            output_schema=IntakeAgentOutput,
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

        # Secondary LLM — ClarificationRequest fallback schema        
        self._clarification_bundle = build_agent_llm_bundle(
            output_schema=ClarificationRequest,
            chain=self._model_chain,   # the same with fallback
            tools=None,                # clarification not use tool
        )
        self._clarification_llm = self._clarification_bundle.structured_llm

        # Buying rules — loaded once and injected into every prompt at runtime
        self._buying_rules_text: str = build_buying_rules_text(get_buying_rules())

        # Clarification round limit (YAML value overrides constant)
        self._max_clarification_rounds: int = clarification_cfg.get(
            "max_rounds", MAX_CLARIFICATION_ROUNDS
        )

        logger.info(
            "[%s] IntakeAgent initialized — model_tier='%s', tools=%d, dual-schema mode",
            self.name,
            model_tier,
            len(tools),
        )

    # ── BaseAgent abstract method implementations ─────────────────────────────

    def _build_prompt(self, state: SharedState) -> str:
        """
        Build the user prompt for the intake LLM call.

        Reads ``raw_request`` and the current clarification round from state.
        Buying rules are injected at runtime so the prompt always reflects
        the latest enterprise policy without requiring a redeploy.
        """
        intake_section = state.get("intake", {}) or {}
        current_round: int = intake_section.get("clarification_rounds", 0)

        return self.config.user_template.format(
            raw_request=state.get("raw_request", ""),
            buying_rules_text=self._buying_rules_text,
            clarification_round=current_round,
            max_clarification_rounds=self._max_clarification_rounds,
        )

    def _format_result(
        self,
        state: SharedState,
        parsed_output: Any,
    ) -> Dict[str, Any]:
        """
        Map a successful IntakeAgentOutput to a SharedState update dict.

        Called by run() on the primary schema path only.
        For the ClarificationRequest path see _format_clarification_result().
        """
        result: IntakeAgentOutput = parsed_output
        log_entry = _make_log_entry(
            node=self.name,
            message=(
                f"process_type={result.process_type} "
                f"category={result.category_id} "
                f"routing={result.routing_decision} "
                f"value=${result.estimated_value_usd or 'unknown'}"
            ),
        )

        intake_data: Dict[str, Any] = {
            "validated_request": result.validated_request.model_dump(),
            "estimated_value_usd": result.estimated_value_usd,
            "framework_agreement_id": result.framework_agreement_id,
            "routing_justification": result.routing_justification,
            "escalation_rule_triggered": result.escalation_rule_triggered,
            "clarification_rounds": 0,          # reset after successful completion
            "clarification_question": None,
            "missing_fields": result.missing_fields or [],
        }

        return {
            "process_type": result.process_type,
            "category_id": result.category_id,
            "routing_decision": result.routing_decision,
            "current_step": StepName.INTAKE_COMPLETED.value,
            "decision_log": [log_entry],
            "errors": result.errors or [],
            "intake": intake_data,
        }

    # ── Clarification-specific helpers ───────────────────────────────────────

    def _format_clarification_result(
        self,
        state: SharedState,
        clarification: ClarificationRequest,
        current_round: int,
    ) -> Dict[str, Any]:
        """
        Map a ClarificationRequest to a SharedState update dict.

        Increments the clarification round counter and signals to the graph
        that an interrupt() is required before the intake node is re-entered.
        """
        new_round = current_round + 1
        log_entry = _make_log_entry(
            node=self.name,
            message=(
                f"clarification requested "
                f"(round {new_round}/{self._max_clarification_rounds}) "
                f"missing={clarification.missing_fields}"
            ),
        )

        intake_data: Dict[str, Any] = {
            "validated_request": None,
            "estimated_value_usd": None,
            "framework_agreement_id": None,
            "routing_justification": None,
            "escalation_rule_triggered": None,
            "clarification_rounds": new_round,
            "clarification_question": clarification.question,
            "missing_fields": list(clarification.missing_fields),
        }

        return {
            "current_step": StepName.INTAKE_CLARIFICATION.value,
            "decision_log": [log_entry],
            "errors": [],
            "intake": intake_data,
        }

    # ── run() override — dual-schema fallback logic ───────────────────────────

    def run(self, state: SharedState) -> SharedState:
        """
        Execute intake processing with dual-schema LLM fallback.

        Flow:
            1. Build messages (system + user prompt).
            2. Invoke primary LLM   → IntakeAgentOutput.
            3. On failure, fall back to secondary LLM → ClarificationRequest.
            4. Format via _format_result() or _format_clarification_result().

        Args:
            state: Current SharedState.

        Returns:
            Updated SharedState with intake results.

        Raises:
            StructuredOutputError: If both the primary and fallback LLM calls fail.
        """
        # span = self._start_span("run")
        exc_to_report: Exception | None = None

        try:
            intake_section = state.get("intake", {}) or {}
            current_round: int = intake_section.get("clarification_rounds", 0)

            # --------
            # Guard: max clarification rounds exceeded
            # --------
            if current_round >= self._max_clarification_rounds:

                raise StructuredOutputError(
                    agent=self.name,
                    schema="ClarificationRequest, IntakeAgentOutput",
                    reason=(
                        f"Max clarification rounds reached"
                        f"({current_round}/{self._max_clarification_rounds})."
                        f"Cannot complete intake without required fields."
                    )
                )
            messages = [
                SystemMessage(content=self.config.system_prompt),
                HumanMessage(content=self._build_prompt(state)),
            ]
            # --------
            # ── First - Primary schema: IntakeAgentOutput ────────────────────────────
            # --------
            primary_error: Exception | None = None
            try:
                result: IntakeAgentOutput = self._llm.structured_llm.invoke(messages) # invoke - router_llm.py
                state_update = self._format_result(state, result)
                self._record_metrics(result)
                # First - Primary result
                return state | state_update

            except Exception as exc:
                primary_error = exc
                logger.warning(
                    "[%s] Primary schema (IntakeAgentOutput) failed: %s. "
                    "Attempting ClarificationRequest fallback.",
                    self.name,
                    exc,
                )
            # --------
            # ── Second- Fallback schema: ClarificationRequest ─────────────────────────
            # --------
            try:
                clarification: ClarificationRequest = (
                    self._clarification_llm.invoke(messages)
                )
                state_update = self._format_clarification_result(
                    state, clarification, current_round
                )
                # Second - Crairication result
                return state | state_update

            except Exception as fallback_error:
                structured_error = StructuredOutputError(
                    agent=self.name,
                    schema="IntakeAgentOutput | ClarificationRequest",
                    reason=(
                        f"Primary error: {primary_error}. "
                        f"Fallback error: {fallback_error}"
                    ),
                )
                exc_to_report = structured_error
                raise structured_error from fallback_error

        except StructuredOutputError:
            raise
        except Exception as exc:
            exc_to_report = exc
            logger.exception("[%s] Unexpected error in run(): %s", self.name, exc)
            raise StructuredOutputError(
                agent=self.name,
                schema="IntakeAgentOutput | ClarificationRequest",
                reason=str(exc),
            ) from exc
        finally:
            # self._end_span(span, exc_to_report)
            pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _record_metrics(self, parsed_output: IntakeAgentOutput) -> None:
        """Record intake-specific metrics if a metrics collector is provided."""
        if self._metrics is None:
            return

        self._metrics.record_process_type(parsed_output.process_type)
        self._metrics.record_routing_decision(parsed_output.routing_decision)

        if parsed_output.estimated_value_usd is not None:
            self._metrics.record_estimated_value(parsed_output.estimated_value_usd)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _make_log_entry(*, node: str, message: str) -> str:
    """Return a timestamped, standardized log entry for the decision audit trail."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    return f"[{ts} UTC] {node.upper()}: {message}"
