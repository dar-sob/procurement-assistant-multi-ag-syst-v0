"""
nodes/procurement_node.py

LangGraph integration node for the Procurement Agent.

Responsibility:
    Thin wrapper only — receives SharedState, delegates to
    ProcurementAgent.run(), returns updated SharedState.

What this node does NOT do:
    - No LLM calls
    - No supplier search logic
    - No strategy decisions

What lives here:
    - Node factory function (make_procurement_node)
    - Error handling and logging at the node level
"""

import logging
from typing import Callable

from procurement_system.state import SharedState
from procurement_system.constants import (
    NodeName,
    StepName,
    RoutingDecision,
)
from procurement_system.exceptions import (
    AgentError,
    ProcurementSystemError,
)
from procurement_system.container import container


logger = logging.getLogger(__name__)


def make_procurement_node() -> Callable[[SharedState], SharedState]:
    """
    Factory function for the Procurement Agent node.

    Imports and initialises ProcurementAgent at build time (once).
    Returns a node function that LangGraph calls per request.

    Returns:
        procurement_node function ready to register in the graph.
    """

    #from procurement_system.agents.procurement.agent import ProcurementAgent
    #agent = ProcurementAgent()

    #----
    # Agent from Dependency Injection Container 
    # with artefact (repositories, services, tracer, metrics)
    #----
    agent = container.procurement_agent() 

    logger.info(f"[{NodeName.PROCUREMENT.value}] Node initialised")

    def procurement_node(state: SharedState) -> SharedState:
        """
        Procurement Agent node — called by LangGraph for every request.

        Delegates entirely to ProcurementAgent.run(state).

        Args:
            state: SharedState — complete graph state

        Returns:
            Updated SharedState with procurement section populated.
        """
        logger.info(
            f"[{NodeName.PROCUREMENT.value}] Identifying suppliers — "
            f"category={state.get('category_id')} "
            f"process_type={state.get('process_type')}"
        )

        try:
            result = agent.run(state)

        except AgentError as e:
            logger.error(f"[{NodeName.PROCUREMENT.value}] Agent error: {e}")
            return {
                "current_step":     StepName.PROCUREMENT_COMPLETED.value,
                "errors":           [str(e)],
                "routing_decision": RoutingDecision.ESCALATE.value,
            }

        except ProcurementSystemError as e:
            logger.error(f"[{NodeName.PROCUREMENT.value}] System error: {e}")
            return {
                "current_step":     StepName.PROCUREMENT_COMPLETED.value,
                "errors":           [str(e)],
                "routing_decision": RoutingDecision.ESCALATE.value,
            }

        except Exception as e:
            logger.exception(f"[{NodeName.PROCUREMENT.value}] Unexpected error: {e}")
            return {
                "current_step":     StepName.PROCUREMENT_COMPLETED.value,
                "errors":           [f"Unexpected error in procurement node: {e}"],
                "routing_decision": RoutingDecision.ESCALATE.value,
            }

        # Count suppliers from the new procurement section
        procurement_section = result.get("procurement", {})
        supplier_count = len(procurement_section.get("supplier_recommendations", []))
        logger.info(
            f"[{NodeName.PROCUREMENT.value}] Completed — "
            f"{supplier_count} suppliers found"
        )

        return result

    return procurement_node
