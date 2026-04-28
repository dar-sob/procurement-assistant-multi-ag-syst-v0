"""
schemas/procurement_schemas.py

Domain models and Pydantic output schemas for the Procurement Agent.

Contents:
    SupplierRecommendation  — domain model: single supplier option
    ProcurementAgentOutput  — LLM output schema

Used with:
    llm.with_structured_output(ProcurementAgentOutput)

Used by state.py:
    SupplierRecommendation is imported into ProcurementPrivateState.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# DOMAIN MODELS
# ─────────────────────────────────────────────────────────

class SupplierRecommendation(BaseModel):
    """Single supplier option recommended by the Procurement Agent."""

    name:                   str   = Field(description="Full supplier name")
    supplier_type:          Literal["local", "national", "international"] = Field(
        description="Geographic scope of the supplier"
    )
    estimated_price_range:  str   = Field(description="e.g. '$10,000 – $15,000'")
    lead_time:              str   = Field(description="e.g. '2–3 weeks'")
    reliability_score:      float = Field(
        description="Score: 1 (poor) to 10 (excellent)",
        ge=1, le=10,
    )
    pros:                   list[str] = Field(description="Key advantages")
    cons:                   list[str] = Field(description="Key disadvantages or risks")
    contact_priority:       int = Field(
        description="1 = primary, 2 = secondary, 3 = backup",
        ge=1, le=3,
    )


# ─────────────────────────────────────────────────────────
# LLM OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────

class ProcurementAgentOutput(BaseModel):
    """
    Returned by the Procurement Agent with supplier recommendations
    and purchasing strategy.

    Mapped to: ProcurementPrivateState in state.py
    """

    supplier_recommendations: list[SupplierRecommendation] = Field(
        description="Between 1 and 5 supplier options from different segments",
        min_length=1,
        max_length=5,
    )
    procurement_strategy:     str = Field(
         description=(
            "Narrative description of the recommended purchasing strategy. "
            "Must explain: why these supplier segments were chosen, which process "
            "type applies and why, and how urgency and budget shape the approach. "
            "Minimum 150 characters."
        ),
        min_length=150,
    )
    recommended_order_type:   Literal["one_time", "framework", "rfq", "rfp"] = Field(
        description="Recommended order type — must be consistent with process_type from intake"
    )
    negotiation_points:       list[str] = Field(
        description=(
            "3 to 5 specific, actionable negotiation points. "
            "Each point must name what to negotiate and what outcome to target."
        ),
        min_length=3,
        max_length=5,
    )
    alternative_products:     list[str] = Field(
        default_factory=list,
        description="Alternative products or services if primary unavailable"
    )
    errors:                   list[str] = Field(
        default_factory=list,
        description="Non-fatal errors or warnings encountered"
    )


    # @model_validator(mode="after")
    # def check_unique_contact_priorities(self) -> "ProcurementAgentOutput":
    #     """
    #     Validate that contact_priority is unique across all supplier recommendations.

    #     Each supplier must have a distinct priority value (1 = primary,
    #     2 = secondary, 3 = backup). Duplicate priorities indicate a
    #     malformed response from the LLM.

    #     Raises:
    #         ValueError: If any two suppliers share the same contact_priority.
    #     """
    #     priorities = [s.contact_priority for s in self.supplier_recommendations]
    #     if len(priorities) != len(set(priorities)):
    #         raise ValueError(
    #             "contact_priority must be unique across all supplier recommendations. "
    #             f"Received: {priorities}"
    #         )
    #     return self
