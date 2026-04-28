# procurement_system/state.py
"""
state.py

LangGraph state definitions for the Multi-Agent Procurement System.

ARCHITECTURE CHANGE (2026-03-28)
────────────────────────────────
Previously, we used separate PrivateState classes inheriting from SharedState,
which caused data visibility issues (child nodes could not see parent node outputs).

NEW STRUCTURE:
- Single SharedState TypedDict with nested domains: intake, procurement, analyst, orchestrator.
- Each domain contains fields written exclusively by one agent, but readable by all.
- Global fields (e.g., raw_request, final_decision) remain at the top level.

This approach guarantees that any node can access any previously written data,
while maintaining clear ownership of each field.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict, Any, Dict
from procurement_system.schemas.procurement_schemas import SupplierRecommendation



# ─────────────────────────────────────────────────────────
# DOMAIN-SPECIFIC STATE SECTIONS
# ─────────────────────────────────────────────────────────

class IntakeState(TypedDict, total=False):
    """State written by the Intake Agent."""
    validated_request:          Dict[str, Any] # ValidatedRequest
    estimated_value_usd:        float
    framework_agreement_id:     Optional[str]
    routing_justification:      str
    escalation_rule_triggered:  Optional[str]
    clarification_rounds:       int
    clarification_question:     Optional[str]
    missing_fields:             list[str]


class ProcurementState(TypedDict, total=False):
    """State written by the Procurement Agent."""
    supplier_recommendations:   list[SupplierRecommendation]
    procurement_strategy:       str
    recommended_order_type:     Literal["one_time", "framework", "rfq", "rfp"]
    negotiation_points:         list[str]
    alternative_products:       list[str]


class AnalystState(TypedDict, total=False):
    """State written by the Analyst Agent."""
    cost_analysis:              Dict[str, Any] # CostAnalysis
    risk_analysis:              Dict[str, Any] # RiskAnalysis
    final_recommendation:       Dict[str, Any] # FinalRecommendation


class OrchestratorState(TypedDict, total=False):
    """State written by the Orchestrator."""
    final_report:               Dict[str, Any] # FinalReport


# ─────────────────────────────────────────────────────────
# SHARED STATE — THE SINGLE SOURCE OF TRUTH
# ─────────────────────────────────────────────────────────

class SharedState(TypedDict, total=False):
    """
    Complete graph state.

    All agents receive and modify the same SharedState object.
    Data written by one agent is immediately visible to subsequent agents.
    """

    # ── Global fields (accessed by multiple agents) ────────
    raw_request:                str
    process_type:               Literal["catalog_purchase", "rfq", "formal_rfq", "strategic_sourcing"]
    category_id:                str
    routing_decision:           Literal["proceed", "escalate"]

    final_decision:             Literal["PROCEED", "PROCEED_WITH_CONDITIONS", "ESCALATE", "REJECT"]
    message_to_user:            str

    # Audit trail (append‑only)
    decision_log:               Annotated[list[str], operator.add]
    errors:                     Annotated[list[str], operator.add]

    # Graph control
    current_step:               str

    # ── Domain sections ────────────────────────────────────
    intake:       IntakeState
    procurement:  ProcurementState
    analyst:      AnalystState
    orchestrator: OrchestratorState


# ─────────────────────────────────────────────────────────
# TYPE ALIASES FOR AGENT INPUTS (optional, for readability)
# ─────────────────────────────────────────────────────────
# Each agent still receives the whole SharedState.
# These aliases can be used in type hints to clarify which fields
# the agent is expected to read/write, but they are not required.

IntakeInput = SharedState
ProcurementInput = SharedState
AnalystInput = SharedState
OrchestratorInput = SharedState
