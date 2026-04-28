"""
nodes/orchestrator_node.py

LangGraph integration node for the Orchestrator.

Responsibility:
    Thin wrapper only — receives SharedState, delegates to
    OrchestratorAgent.run(), returns updated SharedState.

What this node does NOT do:
    - No LLM calls
    - No report compilation logic
    - No response formatting

What lives here:
    - Node factory function (make_orchestrator_node)
    - Error handling and logging at the node level
"""

import logging
from typing import Callable

from procurement_system.state import SharedState
from procurement_system.constants import (
    NodeName,
    StepName,
)
from procurement_system.exceptions import (
    AgentError,
    ProcurementSystemError,
)
from procurement_system.container import container

logger = logging.getLogger(__name__)


def make_orchestrator_node() -> Callable[[SharedState], SharedState]:
    """
    Factory function for the Orchestrator node.

    Imports and initialises OrchestratorAgent at build time (once).
    Returns a node function that LangGraph calls per request.

    Returns:
        orchestrator_node function ready to register in the graph.
    """

    agent = container.orchestrator_agent()

    logger.info(f"[{NodeName.ORCHESTRATOR.value}] Node initialised")

    def orchestrator_node(state: SharedState) -> SharedState:
        """
        Orchestrator node — final node in the graph.

        Delegates entirely to OrchestratorAgent.run(state).
        Compiles final report and formulates user response.

        Args:
            state: SharedState — complete graph state

        Returns:
            Updated SharedState with orchestrator section and top-level fields.
        """
        # Log using top-level final_decision if already present
        final_decision = state.get("final_decision", "UNKNOWN")
        logger.info(
            f"[{NodeName.ORCHESTRATOR.value}] Compiling final report — "
            f"decision={final_decision}"
        )

        try:
            result = agent.run(state)

        except AgentError as e:
            logger.error(f"[{NodeName.ORCHESTRATOR.value}] Agent error: {e}")
            return {
                "current_step":   StepName.ORCHESTRATOR_COMPLETED.value,
                "errors":         [str(e)],
                "message_to_user": (
                    "Your request has been processed but an error occurred "
                    "while compiling the final report. "
                    "Please contact the procurement team for details."
                ),
            }

        except ProcurementSystemError as e:
            logger.error(f"[{NodeName.ORCHESTRATOR.value}] System error: {e}")
            return {
                "current_step":   StepName.ORCHESTRATOR_COMPLETED.value,
                "errors":         [str(e)],
                "message_to_user": (
                    "Your request has been processed but a system error occurred. "
                    "Please contact the procurement team."
                ),
            }

        except Exception as e:
            logger.exception(f"[{NodeName.ORCHESTRATOR.value}] Unexpected error: {e}")
            return {
                "current_step":   StepName.ORCHESTRATOR_COMPLETED.value,
                "errors":         [f"Unexpected error in orchestrator node: {e}"],
                "message_to_user": (
                    "An unexpected error occurred. "
                    "Please contact the procurement team."
                ),
            }

        # Log completion using orchestrator section if available
        orchestrator_section = result.get("orchestrator", {})
        final_report = orchestrator_section.get("final_report", {})
        logger.info(
            f"[{NodeName.ORCHESTRATOR.value}] Completed — "
            f"report_id={final_report.get('report_id', 'N/A')}"
        )

        return result

    return orchestrator_node
