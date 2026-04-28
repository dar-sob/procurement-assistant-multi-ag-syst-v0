"""
nodes/intake_node.py

LangGraph integration node for the Intake Agent.

Responsibility:
    Thin wrapper only — receives SharedState, delegates to
    IntakeAgent.run(), returns updated SharedState.

What this node does NOT do:
    - No LLM calls
    - No prompt loading
    - No business logic
    - No buying rules evaluation

What lives here:
    - Node factory function (make_intake_node)
    - Routing decision after agent run
    - Error handling and logging at the node level
    - interrupt() call when agent requests clarification
"""

import logging
from typing import Callable

from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt

from procurement_system.state import SharedState
from procurement_system.constants import (
    RoutingDecision,
    NodeName,
    StepName,
)
from procurement_system.exceptions import (
    AgentError,
    ProcurementSystemError,
)
from procurement_system.container import container

logger = logging.getLogger(__name__)


def make_intake_node() -> Callable[[SharedState], SharedState]:
    """
    Factory function for the Intake Agent node.

    Imports and initialises IntakeAgent at build time (once).
    Returns a node function that LangGraph calls per request.

    Returns:
        intake_node function ready to register in the graph.
    """

    # ── Import agent at build time — not at every invocation ──
    #from procurement_system.agents.intake.agent import IntakeAgent
    #agent = IntakeAgent()
    agent = container.intake_agent()

    logger.info(f"[{NodeName.INTAKE.value}] Node initialised")

    def intake_node(state: SharedState) -> SharedState:
        """
        Intake Agent node — called by LangGraph for every request.

        Delegates entirely to IntakeAgent.run(state).
        Handles clarification interrupt and error routing.

        Args:
            state: SharedState — complete graph state

        Returns:
            Updated SharedState with intake section populated.
        """
        raw_request = state.get("raw_request", "")
        logger.info(
            f"[{NodeName.INTAKE.value}] Processing request: "
            f"'{raw_request[:80]}...'"
        )

        try:
            result = agent.run(state)

        except GraphInterrupt:
            raise

        except AgentError as e:
            # Known agent error — log and route to error handler
            logger.error(f"[{NodeName.INTAKE.value}] Agent error: {e}")
            return {
                "current_step": StepName.INTAKE_COMPLETED.value,
                "errors":       [str(e)],
                "routing_decision": RoutingDecision.ESCALATE.value,
            }

        except ProcurementSystemError as e:
            # Known system error — log and route to error handler
            logger.error(f"[{NodeName.INTAKE.value}] System error: {e}")
            return {
                "current_step": StepName.INTAKE_COMPLETED.value,
                "errors":       [str(e)],
                "routing_decision": RoutingDecision.ESCALATE.value,
            }

        except Exception as e:
            # Unexpected error — log full traceback
            logger.exception(f"[{NodeName.INTAKE.value}] Unexpected error: {e}")
            return {
                "current_step": StepName.INTAKE_COMPLETED.value,
                "errors":       [f"Unexpected error in intake node: {e}"],
                "routing_decision": RoutingDecision.ESCALATE.value,
            }

        # ── Clarification requested — pause graph ─────────
        intake_section = result.get("intake", {})
        clarification_question = intake_section.get("clarification_question")

        if clarification_question:
            current_round = intake_section.get("clarification_rounds", 0)
            missing_fields = intake_section.get("missing_fields", [])

            logger.info(
                f"[{NodeName.INTAKE.value}] Clarification needed — "
                f"round {current_round}"
            )

            # Graph pauses here — state saved to checkpointer.
            # Execution resumes when user responds via API.
            user_answer = interrupt({
                "type":             "clarification_needed",
                "question":         clarification_question,
                "missing_fields":   missing_fields,
                "round":            current_round,
            })

            # Enrich raw_request with user's answer and re-enter node
            enriched_request = (
                f"{state.get('raw_request', '')}\n\n"
                f"[Clarification round {current_round}]\n"
                f"Additional information: {user_answer}"
            )

            logger.info(
                f"[{NodeName.INTAKE.value}] Clarification received — "
                f"re-entering with enriched request"
            )

            # Prepare updated state, preserving existing intake section
            # but resetting the question for the next iteration.
            updated_intake = dict(intake_section)
            updated_intake["clarification_question"] = None   # will be re‑set by agent
            updated_intake["missing_fields"] = []            # will be re‑computed

            return {
                "raw_request":         enriched_request,
                "intake":              updated_intake,
                "current_step":        StepName.INTAKE_CLARIFICATION.value,
                "decision_log":        result.get("decision_log", []),
            }

        # ── Normal completion ─────────────────────────────
        logger.info(
            f"[{NodeName.INTAKE.value}] Completed — "
            f"process_type={result.get('process_type')} "
            f"routing={result.get('routing_decision')}"
        )

        return result

    return intake_node
