"""
nodes/analyst_node.py

LangGraph integration node for the Analyst Agent.

Responsibility:
    Thin wrapper only — receives AnalystInput state, delegates to
    AnalystAgent.run(), returns updated SharedState.

What this node does NOT do:
    - No LLM calls
    - No TCO calculations
    - No risk scoring logic

What lives here:
    - Node factory function (make_analyst_node)
    - Error handling and logging at the node level
"""

import logging
from typing import Callable

from procurement_system.state import SharedState
from procurement_system.constants import (
    NodeName,
    StepName,
    AnalystFinalDecision,
)
from procurement_system.exceptions import (
    AgentError,
    ProcurementSystemError,
)
from procurement_system.container import container

logger = logging.getLogger(__name__)


def make_analyst_node() -> Callable[[SharedState], SharedState]:
    """
    Factory function for the Analyst Agent node.

    Imports and initialises AnalystAgent at build time (once).
    Returns a node function that LangGraph calls per request.

    Returns:
        analyst_node function ready to register in the graph.
    """

    agent = container.analyst_agent()

    logger.info(f"[{NodeName.ANALYST.value}] Node initialised")

    def analyst_node(state: SharedState) -> SharedState:
        """
        Analyst Agent node — called by LangGraph for every request.

        Delegates entirely to AnalystAgent.run(state).

        Args:
            state: SharedState — complete graph state (with nested sections)

        Returns:
            Updated SharedState with analyst section populated.
        """
        procurement_state = state.get("procurement", {})
        recomendations = procurement_state.get("supplier_recommendations",[]) 
        supplier_count = len(recomendations) if isinstance (recomendations, (list, tuple)) else 0

        logger.info(
            f"[{NodeName.ANALYST.value}] Running cost and risk analysis — "
            f"suppliers={supplier_count}"
        )

        try:
            result = agent.run(state)

        except AgentError as e:
            logger.error(f"[{NodeName.ANALYST.value}] Agent error: {e}")
            return {
                "current_step":  StepName.ANALYST_COMPLETED.value,
                "errors":        [str(e)],
                "final_decision": AnalystFinalDecision.ESCALATE.value,
            }

        except ProcurementSystemError as e:
            logger.error(f"[{NodeName.ANALYST.value}] System error: {e}")
            return {
                "current_step":  StepName.ANALYST_COMPLETED.value,
                "errors":        [str(e)],
                "final_decision": AnalystFinalDecision.ESCALATE.value,
            }

        except Exception as e:
            logger.exception(f"[{NodeName.ANALYST.value}] Unexpected error: {e}")
            return {
                "current_step":  StepName.ANALYST_COMPLETED.value,
                "errors":        [f"Unexpected error in analyst node: {e}"],
                "final_decision": AnalystFinalDecision.ESCALATE.value,
            }

        logger.info(
            f"[{NodeName.ANALYST.value}] Completed — "
            f"decision={result.get('final_decision')} "
            f"risk={result.get('analyst', {}).get('risk_analysis', {}).get('overall_risk_level', 'N/A')}"
        )

        return result

    return analyst_node
