"""
constants.py

Business and technical constants for the Multi-Agent Procurement System.

Enums are used for all domain values — stronger type safety than raw strings.
Plain constants are used for technical settings (temperatures, timeouts, etc.)

Usage:
    from procurement_system.constants import ProcessType, RoutingDecision
    if state["process_type"] == ProcessType.RFQ:
    if result.routing == RoutingDecision.PROCEED:
"""

from enum import Enum


# ── LLM ───────────────────────────────────────────────────
DEFAULT_MODEL       = "claude-sonnet-4-6"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS  = 1024
DEFAULT_LLM_TIMEOUT_SECONDS = 60

# ── LLM with TOOLS ────────────────────────────────────────
DEFAULT_MAX_TOOL_ITERATIONS: int = 10
MIN_FINAL_MESSAGE_LENGTH: int = 10

# ── LLM Timeouts and Retries ──────────────────────────────
DEFAULT_LLM_TIMEOUT_SECONDS      = 60   # hard deadline per LLM call
DEFAULT_LLM_MAX_RETRIES          = 3    # tenacity retry attempts


# ── LLM Retry Policy ─────────────────────────────────────
LLM_RETRY_MAX_ATTEMPTS = 3
LLM_RETRY_WAIT_MULTIPLIER = 1
LLM_RETRY_WAIT_MIN = 2
LLM_RETRY_WAIT_MAX = 8


# ── LLM REQUESTS PER MINUTE, TOKENS PER MINUTE ────────────
DEFAULT_RPM: int | None = 60
DEFAULT_TPM: int | None = 5000


# ── Currency ──────────────────────────────────────────────
DEFAULT_CURRENCY = "USD"


# ── Process Types ─────────────────────────────────────────
# Must match 'label' values in enterprise_buying_rules.yaml
class ProcessType(str, Enum):
    CATALOG    = "catalog_purchase"
    RFQ        = "rfq"
    FORMAL_RFQ = "formal_rfq"
    STRATEGIC  = "strategic_sourcing"

    # Default when estimated value is unknown
    @classmethod
    def default(cls) -> "ProcessType":
        return cls.RFQ


# ── Urgency Levels ────────────────────────────────────────
class UrgencyLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ── Category IDs ──────────────────────────────────────────
# Must match 'id' values in enterprise_buying_rules.yaml
class CategoryID(str, Enum):
    IT                    = "IT_SOFTWARE"
    RAW_MATERIALS         = "RAW_MATERIALS"
    PROFESSIONAL_SERVICES = "PROFESSIONAL_SERVICES"
    OFFICE_SUPPLIES       = "OFFICE_SUPPLIES"
    LOGISTICS             = "LOGISTICS"
    MACHINERY             = "MACHINERY"
    MARKETING             = "MARKETING"
    OTHER                 = "OTHER"


# ── Routing Decisions ─────────────────────────────────────
class RoutingDecision(str, Enum):
    PROCEED  = "proceed"
    ESCALATE = "escalate"


# ── Final Decisions (Analyst Agent output) ────────────────
class AnalystFinalDecision(str, Enum):
    PROCEED             = "PROCEED"
    PROCEED_WITH_COND   = "PROCEED_WITH_CONDITIONS"
    ESCALATE            = "ESCALATE"
    REJECT              = "REJECT"


# ── THRESHOLDS Proceed Escalate (Analyst Agent Risc Score) ───
AUTO_PROCEED_MAX_THRESHOLD = 3.0
AUTO_ESCALATE_MIN_THRESHOLD = 7.5

# ── Graph Node Names ──────────────────────────────────────
# Must match node names registered in graph/procurement_graph.py
class NodeName(str, Enum):
    INTAKE       = "intake"
    PROCUREMENT  = "procurement"
    ANALYST      = "analyst"
    ORCHESTRATOR = "orchestrator"
    HUMAN_REVIEW = "human_review"


# ── Current Step Values ───────────────────────────────────
# Written to state["current_step"] by each agent
class StepName(str, Enum):
    INTAKE_CLARIFICATION   = "intake_clarification"
    INTAKE_COMPLETED       = "intake_completed"
    PROCUREMENT_COMPLETED  = "procurement_completed"
    ANALYST_COMPLETED      = "analyst_completed"
    ORCHESTRATOR_COMPLETED = "orchestrator_completed"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"


# ── Human-in-the-Loop ─────────────────────────────────────
MAX_CLARIFICATION_ROUNDS         = 2
CLARIFICATION_FALLBACK_ON_EXCEED = "continue_with_flags"


# ── Report ────────────────────────────────────────────────
REPORT_ID_PREFIX    = "PO"
LOG_DATETIME_FORMAT = "%H:%M:%S"


# ── Observability ─────────────────────────────────────────
TRACER_NAME      = "procurement_system"
METRICS_NAMESPACE = "procurement_system"


# ── Validation ────────────────────────────────────────────
DEFAULT_VALIDATION_MAX_RETRIES   = 2    # retries on schema validation failure


# ── Routing Retries ───────────────────────────────────────
DEFAULT_ROUTING_MAX_RETRIES      = 2    # retries on routing error
DEFAULT_RETRY_DELAY_S            = 5    # seconds between routing retries


# ── Buying Rules Injection ────────────────────────────────
BUYING_RULES_MAX_TOKENS_GUARD    = 2000  # max tokens for injected rules text


# ── Clarification (Human-in-the-Loop) ────────────────────
MAX_CLARIFICATION_ROUNDS         = 2    # max interrupt() rounds per request
CLARIFICATION_TIMEOUT_HOURS      = 24   # hours before unanswered request escalates
CLARIFICATION_FALLBACK           = "continue_with_flags"

# ── LLM Fallback Strategy ──────────────────────────────────
class FallbackStrategy(str, Enum):
    ON_ERROR    = "on_error"
    NEVER       = "never"
    FAIL_FAST   = "fail_fast"


# ── LLM Builder Mode      ──────────────────────────────────
# Use in llm_router
class BuilderMode(str, Enum):
    STRUCTURED  = "structured"
    TOOLS       = "tools"
    BASE        = "base"


# ── Tools                              ────────────────────


# ── Currency converter tool            ────────────────────
DEFAULT_FRANKFURTER_URL = "https://api.frankfurter.app"
DEFAULT_FRANKFURTER_TIMEOUT = 5
# Frankfurter API — supported currency codes (ECB, 31 currencies)
SUPPORTED_CURRENCIES = {
    "AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CZK",
    "DKK", "EUR", "GBP", "HKD", "HUF", "IDR", "ILS",
    "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK",
    "NZD", "PHP", "PLN", "RON", "SEK", "SGD", "THB",
    "TRY", "USD", "ZAR",
}


# ── Tavily search settings tool        ────────────────────
TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_DEFAULT_MAX_RESULTS = 5
TAVILY_DEFAULT_TIMEOUT = 10
TAVILY_SEARCH_SUPPLIER_QUERY_TEMPLATE = "{product_description} supplier"
MAX_CONTENT_LENGTH = 600


# ── PDF extraction settings            ────────────────────
PDF_DOWNLOAD_TIMEOUT = 30
PDF_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024   # 5 MB
PDF_MAX_PAGES = 10
PDF_OCR_LANGUAGE = "eng"
PDF_OCR_TIMEOUT_SECONDS = 60
PDF_STRICT_CONTENT_TYPE = False
PDF_CACHE_MAXSIZE = 16
PDF_MAX_PAGES = 10
PDF_MAX_CHARS = 6000
