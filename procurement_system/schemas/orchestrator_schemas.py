# procurement_system/schemas/orchestrator_schemas.py
"""
Orchestrator schemas for the procurement system.

Two models are used:
- LLMFinalReport : schema passed to the LLM (business fields only)
- FinalReport    : final model with system metadata and strict validation
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LLMFinalReport(BaseModel):
    """
    Schema used for structured output from the LLM.

    This model reflects exactly what the Orchestrator is allowed to do:
    copy values from upstream agents and assemble the final report.
    No new decisions are made here.
    """

    cost_analysis: Dict[str, Any] = Field(..., description="Copy full object from analyst.cost_analysis")
    risk_analysis: Dict[str, Any] = Field(..., description="Copy full object from analyst.risk_analysis")
    process_type: str = Field(..., description="Copy from state.process_type")

    summary: str = Field(
        ..., 
        description="One concise paragraph (max 50 words) covering product, decision, supplier and cost range."
    )

    framework_agreement_used: Optional[str] = Field(
        None, description="Copy from intake.framework_agreement_id or null"
    )

    recommended_supplier: Optional[str] = Field(
        None, description="Copy from final_recommendation.priority_supplier or null if ESCALATE/REJECT"
    )

    conditions: List[str] = Field(
        default_factory=list,
        description="Copy from final_recommendation.conditions"
    )

    next_steps: List[str] = Field(
        ..., 
        description="Copy and adapt from final_recommendation.next_steps into plain language"
    )

    decision_log: List[str] = Field(
        ..., 
        description="Copy ALL entries from state.decision_log without changes"
    )

    final_decision: Literal[
        "PROCEED",
        "PROCEED_WITH_CONDITIONS",
        "ESCALATE",
        "REJECT"
    ] = Field(
        ...,
        description="MUST copy the exact value from state.final_decision. Do not modify or override."
    )


class FinalReport(LLMFinalReport):
    """
    Final system model.

    Inherits business fields from LLMFinalReport and adds system-controlled metadata.
    Because final_decision is already Literal in the parent, it stays strictly validated.
    """

    report_id: str = Field(..., description="System-generated unique report identifier (PO-YYYYMMDD-HHMMSS)")
    generated_at: datetime = Field(..., description="UTC timestamp when the report was created")
    version: str = Field("1.0", description="Report schema version")
    run_id: Optional[str] = Field(None, description="Global run/correlation ID (optional)")
