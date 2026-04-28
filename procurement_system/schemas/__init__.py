"""
schemas/__init__.py

Public API for the schemas package.
Import all Pydantic models from here for cleaner imports in agent code.

Usage:
    from procurement_system.schemas import (
        IntakeAgentOutput,
        ClarificationRequest,
        ProcurementAgentOutput,
        AnalystAgentOutput,
        BuyingRulesModel,
    )
"""

from procurement_system.schemas.intake_schemas import (
    ClarificationRequest,
    DecisionLogEntry,
    IntakeAgentOutput,
    ValidatedRequest,
)
from procurement_system.schemas.procurement_schemas import (
    ProcurementAgentOutput,
    SupplierRecommendation,
)
from procurement_system.schemas.analyst_schemas import (
    AnalystAgentOutput,
    CostAnalysis,
    CostScenarios,
    FinalRecommendation,
    RiskAnalysis,
    RiskItem,
)
from procurement_system.schemas.orchestrator_schemas import(
    LLMFinalReport,
    FinalReport,
)
from procurement_system.schemas.buying_rules_schemas import (
    AgreementModel,
    BuyingRulesModel,
    CategoryModel,
    EscalationRuleModel,
    ThresholdModel,
)
from procurement_system.schemas.tool_schemas import(
    CurrencyInput,
)

__all__ = [
    # Intake
    "IntakeAgentOutput",
    "ClarificationRequest",
    "ValidatedRequest",
    "DecisionLogEntry",
    # Procurement
    "ProcurementAgentOutput",
    "SupplierRecommendation",
    # Analyst
    "AnalystAgentOutput",
    "CostAnalysis",
    "CostScenarios",
    "FinalRecommendation",
    "RiskAnalysis",
    "RiskItem",
    #Orchestrator
    "LLMFinalReport",
    "FinalReport",
    # Buying Rules
    "BuyingRulesModel",
    "ThresholdModel",
    "CategoryModel",
    "AgreementModel",
    "EscalationRuleModel",
    # Tools
    "CurrencyInput"

]
