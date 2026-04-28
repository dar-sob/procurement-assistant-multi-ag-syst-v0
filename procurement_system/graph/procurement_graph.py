# procurement_system/graph/procurement_graph.py

"""
LangGraph graph definition for the Multi-Agent Procurement System.

This module defines the complete workflow graph, including nodes and edges,
with conditional routing based on decisions made by agents.

Flow:
    START → intake_node
    intake_node → (proceed) → procurement_node → analyst_node → (conditional)
        - PROCEED / PROCEED_WITH_CONDITIONS → orchestrator_node → END
        - ESCALATE → human_review_node → END
        - REJECT → END
        - (error) → END
    intake_node → (escalate) → END
    procurement_node → (escalate) → END

Dependencies (repositories, services, tracer, metrics) are injected from the top level
(e.g., main.py) and passed to each agent node factory.
"""

import logging
from typing import Optional, Dict, Any

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import Runnable

from procurement_system.state import SharedState
from procurement_system.constants import NodeName
from procurement_system.nodes.intake_node import make_intake_node
from procurement_system.nodes.procurement_node import make_procurement_node
from procurement_system.nodes.analyst_node import make_analyst_node
from procurement_system.nodes.orchestrator_node import make_orchestrator_node
from procurement_system.nodes.human_review_node import make_human_review_node

logger = logging.getLogger(__name__)


def build_procurement_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    repositories: Optional[Dict[str, Any]] = None,
    services: Optional[Dict[str, Any]] = None,
    tracer: Optional[Any] = None,
    metrics: Optional[Any] = None,
) -> Runnable:
    """
    Build and compile the procurement workflow graph.

    Args:
        checkpointer: Optional checkpoint saver (e.g., MemorySaver, SqliteSaver).
                     If None, uses MemorySaver (suitable for development/testing).
        repositories: Dictionary of repository instances (e.g., {"supplier": SupplierRepository}).
        services: Dictionary of service instances (e.g., {"currency": CurrencyService}).
        tracer: Optional tracer for observability (e.g., OpenTelemetry tracer).
        metrics: Optional metrics collector.

    Returns:
        Compiled LangGraph ready for invocation (via .invoke() or .ainvoke()).
    """
    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.info("Using MemorySaver (in-memory checkpointer)")

    # Create graph with SharedState schema
    builder = StateGraph(SharedState)

    # Add nodes (factories receive dependencies)
    builder.add_node(
        NodeName.INTAKE.value,
        make_intake_node() # injected by container in agent: repositories=repositories, services=services, tracer=tracer, metrics=metrics
        # make_intake_node(repositories=repositories, services=services, tracer=tracer, metrics=metrics)
    )
    builder.add_node(
        NodeName.PROCUREMENT.value,
        make_procurement_node() # injected by container in agent: repositories=repositories, services=services, tracer=tracer, metrics=metrics
        # make_procurement_node(repositories=repositories, services=services, tracer=tracer, metrics=metrics)
    )
    builder.add_node(
        NodeName.ANALYST.value,
        make_analyst_node() # injected by container in agent: repositories=repositories, services=services, tracer=tracer, metrics=metrics
        # make_analyst_node(repositories=repositories, services=services, tracer=tracer, metrics=metrics)
    )
    builder.add_node(
        NodeName.ORCHESTRATOR.value,
        make_orchestrator_node() # injected by container in agent: repositories=repositories, services=services, tracer=tracer, metrics=metrics
        # make_orchestrator_node(repositories=repositories, services=services, tracer=tracer, metrics=metrics)
    )
    builder.add_node(
        NodeName.HUMAN_REVIEW.value,
        make_human_review_node()
    )

    # ---- Edge from START ----
    builder.add_edge(START, NodeName.INTAKE.value)

    # ---- Conditional edge from Intake ----
    def route_from_intake(state: SharedState) -> str:
        decision = state.get("routing_decision")
        if decision == "proceed":
            return NodeName.PROCUREMENT.value
        else:
            logger.warning(f"Intake routing decision '{decision}' -> ending graph")
            return END

    builder.add_conditional_edges(
        NodeName.INTAKE.value,
        route_from_intake,
        {
            NodeName.PROCUREMENT.value: NodeName.PROCUREMENT.value,
            END: END,
        }
    )

    # ---- Conditional edge from Procurement ----
    def route_from_procurement(state: SharedState) -> str:
        decision = state.get("routing_decision")
        if decision == "proceed":
            return NodeName.ANALYST.value
        else:
            logger.warning(f"Procurement routing decision '{decision}' -> ending graph")
            return END

    builder.add_conditional_edges(
        NodeName.PROCUREMENT.value,
        route_from_procurement,
        {
            NodeName.ANALYST.value: NodeName.ANALYST.value,
            END: END,
        }
    )

    # ---- Conditional edge from Analyst ----
    def route_from_analyst(state: SharedState) -> str:
        final = state.get("final_decision")
        if final in ("PROCEED", "PROCEED_WITH_CONDITIONS"):
            logger.info(f"Analyst decision '{final}' -> routing to Orchestrator")
            return NodeName.ORCHESTRATOR.value
        elif final == "ESCALATE":
            logger.info("Analyst decision 'ESCALATE' -> routing to Human Review")
            return NodeName.HUMAN_REVIEW.value
        elif final == "REJECT":
            logger.info("Analyst decision 'REJECT' -> ending graph")
            return END
        else:
            logger.error(f"Unknown final_decision '{final}' -> ending graph")
            return END

    builder.add_conditional_edges(
        NodeName.ANALYST.value,
        route_from_analyst,
        {
            NodeName.ORCHESTRATOR.value: NodeName.ORCHESTRATOR.value,
            NodeName.HUMAN_REVIEW.value: NodeName.HUMAN_REVIEW.value,
            END: END,
        }
    )

    # ---- Terminal edges ----
    builder.add_edge(NodeName.ORCHESTRATOR.value, END)
    builder.add_edge(NodeName.HUMAN_REVIEW.value, END)

    # Compile with checkpointer
    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Graph compiled successfully")
    return graph
