"""
schemas/analyst_schemas.py

Domain models and Pydantic output schemas for the Analyst Agent.

Contents:
    CostScenarios       — domain model: three cost projections
    CostAnalysis        — domain model: full TCO analysis
    RiskItem            — domain model: single risk entry
    RiskAnalysis        — domain model: full risk assessment
    FinalRecommendation — domain model: procurement decision
    FinalReport         — domain model: compiled final report (Orchestrator)
    AnalystAgentOutput  — LLM output schema

Used with:
    llm.with_structured_output(AnalystAgentOutput)

Used by state.py:
    CostAnalysis, RiskAnalysis, FinalRecommendation, FinalReport
    are imported into private states.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# DOMAIN MODELS
# ─────────────────────────────────────────────────────────

class CostScenarios(BaseModel):
    """Three cost projections for the purchase."""

    optimistic:  float = Field(description="Best-case total cost in USD",    ge=0)
    realistic:   float = Field(description="Most likely total cost in USD",  ge=0)
    pessimistic: float = Field(description="Worst-case total cost in USD",   ge=0)


class CostAnalysis(BaseModel):
    """Full TCO cost analysis produced by the Analyst Agent."""

    estimated_unit_cost:  float       = Field(description="Estimated cost per unit in USD", ge=0)
    total_estimated_cost: float       = Field(description="Estimated total cost in USD",    ge=0)
    tco_factors:          list[str]   = Field(description="TCO factors beyond unit price")
    cost_scenarios:       CostScenarios = Field(description="Optimistic/realistic/pessimistic")
    budget_adequacy: Literal[
        "sufficient",
        "insufficient",
        "no_data",
    ] = Field(description="Whether stated budget covers the estimated cost")
    potential_savings: Optional[float] = Field(
        default=None,
        description="Savings achievable through negotiation or alternatives",
        ge=0,
    )


class RiskItem(BaseModel):
    """Single identified risk."""

    risk_name:   str = Field(description="Short name of the risk")
    category: Literal[
        "financial",
        "operational",
        "supplier",
        "quality",
        "delivery",
        "compliance",
    ] = Field(description="Risk category")
    probability: float = Field(
        description="Probability: 0.0 (none) to 1.0 (certain)",
        ge=0.0, le=1.0,
    )
    impact: float = Field(
        description="Business impact: 1 (low) to 10 (critical)",
        ge=1, le=10,
    )
    mitigation: str = Field(description="Recommended mitigation action")


class RiskAnalysis(BaseModel):
    """Risk assessment produced by the Analyst Agent."""

    overall_risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="Overall risk classification"
    )
    risk_score: float = Field(
        description="Composite risk score: 1 (minimal) to 10 (critical)",
        ge=1, le=10,
    )
    risks: list[RiskItem] = Field(description="Individual risks identified")


class FinalRecommendation(BaseModel):
    """Final procurement decision produced by the Analyst Agent."""

    decision: Literal[
        "PROCEED",
        "PROCEED_WITH_CONDITIONS",
        "ESCALATE",
        "REJECT",
    ] = Field(description="Final procurement decision")
    justification: str = Field(
        description="Justification referencing cost and risk findings",
        min_length=30,
    )
    conditions:        list[str]       = Field(
        default_factory=list,
        description="Conditions required if decision is PROCEED_WITH_CONDITIONS"
    )
    priority_supplier: Optional[str]   = Field(
        default=None,
        description="Recommended supplier name; null if REJECT"
    )
    next_steps:        list[str]       = Field(
        description="Concrete action items for the requester",
        min_length=1,
    )


# class FinalReport(BaseModel):
#     """
#     Compiled final report produced by the Orchestrator.
#     Aggregates outputs from all upstream agents.
#     """

#     report_id:                str                    = Field(description="Unique report ID, e.g. PO-20250315-143022")
#     generated_at:             str                    = Field(description="ISO-8601 generation timestamp")
#     summary:                  str                    = Field(description="One-paragraph executive summary")
#     process_type:             str                    = Field(description="Process type applied")
#     framework_agreement_used: Optional[str]          = Field(default=None, description="Agreement ID used, or null")
#     recommended_supplier:     Optional[str]          = Field(default=None, description="Supplier name, or null")
#     cost_analysis:            CostAnalysis           = Field(description="Full TCO analysis")
#     risk_analysis:            RiskAnalysis           = Field(description="Full risk assessment")
#     conditions:               list[str]              = Field(default_factory=list)
#     next_steps:               list[str]              = Field(description="Action items for the requester")
#     decision_log:             list[str]              = Field(description="Accumulated audit entries from all agents")
#     final_decision: Literal[
#     "PROCEED", 
#     "PROCEED_WITH_CONDITIONS", 
#     "ESCALATE", 
#     "REJECT"
# ]


# ─────────────────────────────────────────────────────────
# LLM OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────

class AnalystAgentOutput(BaseModel):
    """
    Returned by the Analyst Agent with full TCO analysis,
    risk assessment, and final decision.

    Mapped to: AnalystPrivateState in state.py
    """

    cost_analysis:        CostAnalysis        = Field(description="TCO analysis with three cost scenarios")
    risk_analysis:        RiskAnalysis        = Field(description="Risk assessment with scored risks")
    final_recommendation: FinalRecommendation = Field(description="Final decision with justification")
    errors:               list[str]           = Field(
        default_factory=list,
        description="Non-fatal errors or warnings encountered"
    )
