# procurement_system/container.py
"""
Dependency Injection Container — single composition root for the entire system.

Wiring order: repositories → services → agents.
Each provider is a Singleton: resolved once and shared across the application.

Testing:
    container.policy_service.override(MockPolicyService())
    container.intake_agent.reset_override()
"""

import logging
from dependency_injector import containers, providers

# Repositories
from procurement_system.repositories.currency_repository import CurrencyRepository
from procurement_system.repositories.pdf_repository import PDFRepository
from procurement_system.repositories.tavily_repository import TavilyRepository
#from procurement_system.repositories.supplier_repository import SupplierRepository
#from procurement_system.repositories.contract_repository import ContractRepository


# Services
from procurement_system.services.pdf_extraction_service import PDFExtractionService
from procurement_system.services.currency_service import CurrencyService
from procurement_system.services.supplier_web_search_service import SupplierWebSearchService
#from procurement_system.services.policy_service import PolicyService
#from procurement_system.services.supplier_service import SupplierService
#from procurement_system.services.risk_service import RiskService
#from procurement_system.services.notification_service import NotificationService

# Agents
from procurement_system.agents.intake.agent import IntakeAgent
from procurement_system.agents.procurement.agent import ProcurementAgent
from procurement_system.agents.analyst.agent import AnalystAgent
from procurement_system.agents.orchestrator.agent import OrchestratorAgent

logger = logging.getLogger(__name__)

class Container(containers.DeclarativeContainer):
    """
    Central Dependency Injection container for the procurement multi-agent system.

    Providers are declared from top to bottom, mirroring the dependency graph:
        repositories → services → agents.

    All providers are Singletons — each class is instantiated exactly once.
    Agent providers receive their dependencies as keyword arguments so that
    dependency_injector can automatically resolve and inject them. This also
    ensures that test overrides propagate correctly through the entire graph.
    """

    # ══════════════════════════════════════════════════════════════════════
    # 1. REPOSITORIES
    # ══════════════════════════════════════════════════════════════════════

    currency_repository: providers.Singleton = providers.Singleton(CurrencyRepository)
    pdf_repository: providers.Singleton = providers.Singleton(PDFRepository)
    tavily_repository: providers.Singleton = providers.Singleton(TavilyRepository)
    # supplier_repository: providers.Singleton = providers.Singleton(SupplierRepository)
    # contract_repository: providers.Singleton = providers.Singleton(ContractRepository)


    # ══════════════════════════════════════════════════════════════════════
    # 2. SERVICES
    # ══════════════════════════════════════════════════════════════════════

    # policy_service: providers.Singleton = providers.Singleton(PolicyService)
    # supplier_service: providers.Singleton = providers.Singleton(SupplierService)
    # risk_service: providers.Singleton = providers.Singleton(RiskService)
    # notification_service: providers.Singleton = providers.Singleton(NotificationService)

    pdf_extraction_service: providers.Singleton = providers.Singleton(
        PDFExtractionService,
        pdf_repo=pdf_repository,
    )

    currency_service: providers.Singleton = providers.Singleton(
        CurrencyService,
        repository=currency_repository,
    )

    supplier_web_search_service: providers.Singleton = providers.Singleton(
        SupplierWebSearchService,
        tavily_repo=tavily_repository,
    )

    # ══════════════════════════════════════════════════════════════════════
    # 3. OBSERVABILITY
    #    Replace providers.Object(None) with real implementations when ready.
    # ══════════════════════════════════════════════════════════════════════

    tracer: providers.Object = providers.Object(None)
    """Distributed tracing provider (e.g. OpenTelemetry). Currently disabled."""

    metrics: providers.Object = providers.Object(None)
    """Metrics collector. Currently disabled."""

    # ══════════════════════════════════════════════════════════════════════
    # 4. AGENTS
    #
    # Important: We pass PROVIDERS (not resolved instances) as arguments.
    # dependency_injector will resolve them at the right time.
    # This ensures that overrides in tests work correctly for all dependent agents.
    # ══════════════════════════════════════════════════════════════════════

    intake_agent: providers.Singleton = providers.Singleton(
        IntakeAgent,
        repositories=providers.Dict(
            # Add repositories here when needed
            # supplier=supplier_repository,
        ),
        services=providers.Dict(
            # policy=policy_service,
        ),
        tracer=tracer,
        metrics=metrics,
    )

    procurement_agent: providers.Singleton = providers.Singleton(
        ProcurementAgent,
        repositories=providers.Dict(),
        services=providers.Dict(
            pdf_extraction_service=pdf_extraction_service,
            currency_service=currency_service,
            supplier_web_search_service=supplier_web_search_service
        ),
        tracer=tracer,
        metrics=metrics,
    )

    analyst_agent: providers.Singleton = providers.Singleton(
        AnalystAgent,
        repositories=providers.Dict(),
        services=providers.Dict(
            pdf_extraction_service=pdf_extraction_service,
        ),
        tracer=tracer,
        metrics=metrics,
    )

    orchestrator_agent: providers.Singleton = providers.Singleton(
        OrchestratorAgent,
        repositories=providers.Dict(),
        services=providers.Dict(
            # notification=notification_service,
        ),
        tracer=tracer,
        metrics=metrics,
    )


# ── Module-level singleton instance ─────────────────────────────────────

container = Container()
"""
Single container instance shared across the entire application.

Import and usage example:

    from procurement_system.container import container

    agent = container.analyst_agent()
"""
