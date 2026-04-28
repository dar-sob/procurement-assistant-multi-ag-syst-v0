"""
schemas/buying_rules_schemas.py

Pydantic models for enterprise_buying_rules.yaml structure validation.

Used by:
    utils/prompt_builder.py  — validates buying rules dict before serialization
    services/policy_service.py — validates rules before applying thresholds

Validated once when buying rules are first loaded via settings.get_buying_rules().
Any structural error in enterprise_buying_rules.yaml is caught here with a
clear Pydantic ValidationError — not deep in agent logic.
"""

from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────
# THRESHOLD
# ─────────────────────────────────────────────────────────

class ThresholdModel(BaseModel):
    """
    Single value threshold defining a procurement process type.

    Example:
        label:            rfq
        display_name:     RFQ — Request for Quotation
        min_value:        5000
        max_value:        49999
        required_quotes:  3
        approval_required: procurement_manager
    """

    label:             str   = Field(description="Process type label, e.g. 'rfq'")
    display_name:      str   = Field(description="Human-readable name")
    min_value:         float = Field(ge=0, description="Minimum value in USD (inclusive)")
    max_value:         Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum value in USD (inclusive); null = no upper limit",
    )
    description:       str   = Field(description="Description of the process type")
    required_quotes:   int   = Field(ge=1, description="Minimum number of quotes required")
    approval_required: str   = Field(description="Approval level required")

    @model_validator(mode="after")
    def max_must_exceed_min(self) -> "ThresholdModel":
        if self.max_value is not None and self.max_value <= self.min_value:
            raise ValueError(
                f"max_value ({self.max_value}) must exceed "
                f"min_value ({self.min_value}) "
                f"for threshold '{self.label}'"
            )
        return self


# ─────────────────────────────────────────────────────────
# CATEGORY
# ─────────────────────────────────────────────────────────

class CategoryModel(BaseModel):
    """
    Single procurement category.

    Example:
        id:                IT_SOFTWARE
        label:             IT / Software
        preferred_process: rfq
    """

    id:                str       = Field(description="Category ID — used in agent output")
    label:             str       = Field(description="Human-readable category label")
    examples:          list[str] = Field(
        default_factory=list,
        description="Example products or services in this category",
    )
    preferred_process: str       = Field(description="Default process type for this category")
    notes:             Optional[str] = Field(
        default=None,
        description="Additional notes for procurement team",
    )


# ─────────────────────────────────────────────────────────
# FRAMEWORK AGREEMENT
# ─────────────────────────────────────────────────────────

class AgreementModel(BaseModel):
    """
    Active framework agreement with a preferred supplier.

    Example:
        agreement_id:    FA-001
        supplier:        TechCorp Ltd.
        category:        IT_SOFTWARE
        valid_until:     2026-12-31
        max_order_value: 50000
    """

    agreement_id:    str   = Field(description="Unique agreement identifier, e.g. 'FA-001'")
    supplier:        str   = Field(description="Supplier name")
    category:        str   = Field(description="Category ID this agreement applies to")
    valid_until:     str   = Field(description="Expiry date in YYYY-MM-DD format")
    max_order_value: float = Field(ge=0, description="Maximum single order value in USD")
    notes:           Optional[str] = Field(
        default=None,
        description="Additional notes, e.g. direct PO conditions",
    )


# ─────────────────────────────────────────────────────────
# ESCALATION RULE
# ─────────────────────────────────────────────────────────

class EscalationRuleModel(BaseModel):
    """
    Single escalation rule evaluated by the Intake Agent.

    Example:
        trigger: estimated_value > 250000
        action:  route_to_strategic_sourcing
    """

    trigger: str = Field(description="Condition that triggers escalation")
    action:  str = Field(description="Action to take when triggered")


# ─────────────────────────────────────────────────────────
# BUYING RULES — ROOT MODEL
# ─────────────────────────────────────────────────────────

class BuyingRulesModel(BaseModel):
    """
    Full validated structure of enterprise_buying_rules.yaml.

    Validated once at startup via utils/prompt_builder.py and
    services/policy_service.py. Any structural error is caught here
    with a clear message before reaching agent logic.
    """

    thresholds:           list[ThresholdModel]
    categories:           list[CategoryModel]
    framework_agreements: list[AgreementModel]      = Field(default_factory=list)
    escalation:           list[EscalationRuleModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def thresholds_must_not_overlap(self) -> "BuyingRulesModel":
        """Detect overlapping threshold ranges."""
        sorted_t = sorted(self.thresholds, key=lambda t: t.min_value)
        for i in range(len(sorted_t) - 1):
            current = sorted_t[i]
            next_t  = sorted_t[i + 1]
            if (
                current.max_value is not None
                and current.max_value > next_t.min_value
            ):
                raise ValueError(
                    f"Threshold overlap detected: "
                    f"'{current.label}' max_value ({current.max_value}) "
                    f"exceeds '{next_t.label}' min_value ({next_t.min_value})"
                )
        return self

    @model_validator(mode="after")
    def categories_must_have_unique_ids(self) -> "BuyingRulesModel":
        """Detect duplicate category IDs."""
        ids = [c.id for c in self.categories]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate category IDs found: {sorted(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def agreements_must_have_unique_ids(self) -> "BuyingRulesModel":
        """Detect duplicate agreement IDs."""
        ids = [a.agreement_id for a in self.framework_agreements]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate agreement IDs found: {sorted(duplicates)}"
            )
        return self
