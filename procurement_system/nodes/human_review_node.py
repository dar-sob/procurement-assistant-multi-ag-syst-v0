
"""
nodes/human_review_node.py

Node for handling escalations that require human review.

When the Analyst Agent escalates a request (ESCALATE), this node is triggered.
It logs the escalation, optionally sends a notification, and then ends the graph.
"""

import logging
from typing import Callable

from procurement_system.state import SharedState
from procurement_system.constants import NodeName, StepName

logger = logging.getLogger(__name__)


def make_human_review_node() -> Callable[[SharedState], SharedState]:
    """
    Factory function for the Human Review node.

    Returns:
        human_review_node function ready to register in the graph.
    """

    def human_review_node(state: SharedState) -> SharedState:
        """
        Handles escalation: logs, notifies, and ends.

        Args:
            state: SharedState — complete graph state

        Returns:
            Updated state (with final step marked and a log entry).
        """
        # Extract relevant information for the log
        product = state.get("intake", {}).get("validated_request", {}).get("description", "N/A")
        decision = state.get("final_decision", "ESCALATE")
        reason = state.get("analyst", {}).get("final_recommendation", {}).get("justification", "No justification provided")

        logger.warning(
            f"[{NodeName.HUMAN_REVIEW.value}] Escalation triggered. "
            f"Product: {product}, Decision: {decision}, Reason: {reason}"
        )

        # Optional: send notification via notification_service
        # from procurement_system.services.notification_service import notify_ops
        # notify_ops(f"Escalation: {product} - {reason}")

        # Create log entry
        log_entry = (
            f"[{NodeName.HUMAN_REVIEW.value.upper()}] Escalated to human review. "
            f"Decision: {decision}, Reason: {reason}"
        )

        # Update state: mark step and add to decision log
        return {
            "current_step": StepName.HUMAN_REVIEW_COMPLETED.value,
            "decision_log": [log_entry],
            # The final_decision remains ESCALATE; no further routing needed
        }

    logger.info(f"[{NodeName.HUMAN_REVIEW.value}] Node initialised")
    return human_review_node
