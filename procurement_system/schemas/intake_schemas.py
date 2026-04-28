"""
schemas/intake_schemas.py

Domain models and Pydantic output schemas for the Intake Agent.

Contents:
    ValidatedRequest     — domain model: structured purchase request
    DecisionLogEntry     — domain model: audit log entry
    IntakeAgentOutput    — LLM output schema: all required fields present
    ClarificationRequest — LLM output schema: required fields missing

Used with:
    llm.with_structured_output(IntakeAgentOutput)
    llm.with_structured_output(ClarificationRequest)

Used by state.py:
    ValidatedRequest is imported into IntakePrivateState.
    After LLM validation: result.model_dump() → TypedDict state.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# DOMAIN MODELS
# ─────────────────────────────────────────────────────────

class ValidatedRequest(BaseModel):
    """Structured purchase request extracted from user message."""

    description:              str   = Field(description="Full product or service description")
    quantity:                 float = Field(description="Numeric quantity requested", gt=0)
    unit:                     str   = Field(description="Unit of measure, e.g. 'units', 'kg'")
    estimated_unit_price_usd: Optional[float] = Field(
        default=None,
        description="Estimated unit price in USD; null if not stated",
        ge=0,
    )
    deadline:     Optional[str] = Field(
        default=None,
        description="Requested delivery date or timeframe; null if not stated"
    )
    urgency:      Optional[Literal["low", "medium", "high", "critical"]] = Field(
        default=None,
        description="Urgency level; null if not stated"
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Technical or quality requirements stated by the requester"
    )


class DecisionLogEntry(BaseModel):
    """Structured audit entry written to the shared decision log."""

    timestamp:        str = Field(description="ISO-8601 timestamp")
    process_type:     str = Field(description="Determined process type")
    routing_decision: str = Field(description="proceed or escalate")


# ─────────────────────────────────────────────────────────
# LLM OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────

class IntakeAgentOutput(BaseModel):
    """
    Returned when all required fields are present and the request
    can be classified and routed.

    Mapped to: IntakePrivateState + SharedState in state.py
    """

    validated_request:         ValidatedRequest = Field(
        description="Structured purchase request data"
    )
    estimated_value_usd:       Optional[float] = Field(
        default=None,
        description="Estimated total value in USD; null if unknown",
        ge=0,
    )
    process_type: Literal[
        "catalog_purchase",
        "rfq",
        "formal_rfq",
        "strategic_sourcing",
    ] = Field(
        description="Process type determined by applying buying rules thresholds"
    )
    category_id:               str = Field(
        description="Category ID exactly as defined in enterprise buying rules"
    )
    framework_agreement_id:    Optional[str] = Field(
        default=None,
        description="Matched framework agreement ID; null if none applies"
    )
    routing_decision:          Literal["proceed", "escalate"] = Field(
        description="Routing outcome for this request"
    )
    routing_justification:     str = Field(
        description="Justification citing the specific rule applied — min 20 chars",
        min_length=20,
    )
    escalation_rule_triggered: Optional[str] = Field(
        default=None,
        description="Name of the escalation rule triggered; null if none"
    )
    missing_fields:            list[str] = Field(
        default_factory=list,
        description="Optional fields absent from the request"
    )
    clarification_rounds:      int = Field(
        description="Number of clarification interrupts completed",
        ge=0,
    )
    current_step:              Literal["intake_completed"] = Field(
        default="intake_completed",
        description="Sentinel value confirming intake completion"
    )
    decision_log:              DecisionLogEntry = Field(
        description="Structured audit entry for this agent's decision"
    )
    errors:                    list[str] = Field(
        default_factory=list,
        description="Non-fatal errors or warnings encountered"
    )


class ClarificationRequest(BaseModel):
    """
    Returned when one or more required fields are missing.
    Python intercepts this and calls interrupt() to pause the graph.

    Note: round number is NOT included — Python tracks rounds,
    not the LLM. Round is injected via the user prompt template.
    """

    type:           Literal["clarification_request"] = Field(
        default="clarification_request",
        description="Discriminator — identifies this as a clarification"
    )
    question:       str = Field(
        description=(
            "Single question collecting ALL missing required fields at once. "
            "Plain, non-technical language. Never asks for optional fields."
        ),
        min_length=10,
    )
    missing_fields: list[Literal["description", "quantity", "unit"]] = Field(
        description="Required fields absent from the request"
    )
