"""
exceptions.py

Custom exceptions for the Multi-Agent Procurement System.

Hierarchy:
    ProcurementSystemError          <- base for all system exceptions
        |
        +-- ConfigurationError      <- config and file loading issues
        |       FileNotFoundError
        |       InvalidConfigError
        |       MissingFieldError
        |
        +-- ValidationError         <- request validation issues
        |       MissingRequiredField
        |       InvalidCategoryError
        |       InvalidProcessTypeError
        |
        +-- RoutingError            <- graph routing issues
        |       EscalationRequired
        |       MaxClarificationRoundsExceeded
        |
        +-- AgentError              <- agent execution issues
        |       LLMCallError
        |       StructuredOutputError
        |
        +-- RepositoryError         <- data access issues
                SupplierNotFoundError
                ContractNotFoundError
"""


# ── Base Exception ────────────────────────────────────────

class ProcurementSystemError(Exception):
    """
    Base exception for all procurement system errors.
    All custom exceptions inherit from this class.
    Catch this to handle any system error in one place.
    """
    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(
                f"{k}={v}" for k, v in self.context.items()
            )
            return f"{self.message} | context: {context_str}"
        return self.message


# ── Configuration Errors ──────────────────────────────────

class ConfigurationError(ProcurementSystemError):
    """Base for all configuration and file loading errors."""


class ConfigFileNotFoundError(ConfigurationError):
    """
    Raised when a required config or prompt file does not exist.

    Example:
        raise ConfigFileNotFoundError(
            path="procurement_system/config/enterprise_buying_rules.yaml"
        )
    """
    def __init__(self, path: str):
        super().__init__(
            message=f"Required file not found: {path}",
            context={"path": path}
        )
        self.path = path


class InvalidConfigError(ConfigurationError):
    """
    Raised when a config file exists but contains invalid or
    unexpected structure.

    Example:
        raise InvalidConfigError(
            file="config_intake_agent.yaml",
            reason="missing required key: 'prompts'"
        )
    """
    def __init__(self, file: str, reason: str):
        super().__init__(
            message=f"Invalid configuration in '{file}': {reason}",
            context={"file": file, "reason": reason}
        )


class MissingConfigFieldError(ConfigurationError):
    """
    Raised when a required field is absent in a config file.

    Example:
        raise MissingConfigFieldError(
            file="config_intake_agent.yaml",
            field="intake_agent.model_params.temperature"
        )
    """
    def __init__(self, file: str, field: str):
        super().__init__(
            message=f"Missing required field '{field}' in '{file}'",
            context={"file": file, "field": field}
        )


# ── Validation Errors ─────────────────────────────────────

class ValidationError(ProcurementSystemError):
    """Base for all request validation errors."""


class MissingRequiredFieldError(ValidationError):
    """
    Raised when a required field is absent in the purchase request.

    Example:
        raise MissingRequiredFieldError(
            fields=["quantity", "estimated_budget"]
        )
    """
    def __init__(self, fields: list[str]):
        super().__init__(
            message=f"Missing required fields in purchase request: {fields}",
            context={"missing_fields": fields}
        )
        self.fields = fields


class InvalidCategoryError(ValidationError):
    """
    Raised when a category ID does not match any entry
    in enterprise_buying_rules.yaml.

    Example:
        raise InvalidCategoryError(
            category="UNKNOWN_CAT",
            valid_categories=["IT_SOFTWARE", "MACHINERY", ...]
        )
    """
    def __init__(self, category: str, valid_categories: list[str]):
        super().__init__(
            message=(
                f"Invalid category '{category}'. "
                f"Must be one of: {valid_categories}"
            ),
            context={
                "category": category,
                "valid_categories": valid_categories
            }
        )


class InvalidProcessTypeError(ValidationError):
    """
    Raised when a process type does not match any defined threshold
    in enterprise_buying_rules.yaml.

    Example:
        raise InvalidProcessTypeError(process_type="unknown_process")
    """
    def __init__(self, process_type: str):
        super().__init__(
            message=f"Invalid process type: '{process_type}'",
            context={"process_type": process_type}
        )


# ── Routing Errors ────────────────────────────────────────

class RoutingError(ProcurementSystemError):
    """Base for all graph routing errors."""


class EscalationRequiredError(RoutingError):
    """
    Raised when an escalation rule is triggered and the request
    cannot continue through the standard process.

    Example:
        raise EscalationRequiredError(
            rule="estimated_value > 250000",
            justification="Value exceeds strategic sourcing threshold"
        )
    """
    def __init__(self, rule: str, justification: str):
        super().__init__(
            message=f"Escalation required — rule triggered: '{rule}'",
            context={"rule": rule, "justification": justification}
        )
        self.rule          = rule
        self.justification = justification


class MaxClarificationRoundsExceededError(RoutingError):
    """
    Raised when the maximum number of clarification rounds
    with the user has been reached without resolving missing fields.

    Example:
        raise MaxClarificationRoundsExceededError(
            max_rounds=2,
            missing_fields=["estimated_budget"]
        )
    """
    def __init__(self, max_rounds: int, missing_fields: list[str]):
        super().__init__(
            message=(
                f"Max clarification rounds ({max_rounds}) exceeded. "
                f"Still missing: {missing_fields}"
            ),
            context={
                "max_rounds": max_rounds,
                "missing_fields": missing_fields
            }
        )


# ── Agent Errors ──────────────────────────────────────────

class AgentError(ProcurementSystemError):
    """Base for all agent execution errors."""


class LLMCallError(AgentError):
    """
    Raised when the LLM call fails (timeout, API error, rate limit).

    Example:
        raise LLMCallError(
            agent="intake",
            reason="Anthropic API timeout after 30s"
        )
    """
    def __init__(self, agent: str, reason: str):
        super().__init__(
            message=f"LLM call failed in '{agent}' agent: {reason}",
            context={"agent": agent, "reason": reason}
        )


class StructuredOutputError(AgentError):
    """
    Raised when the LLM response cannot be parsed into
    the expected Pydantic schema.

    Example:
        raise StructuredOutputError(
            agent="intake",
            schema="IntakeAgentOutput",
            reason="missing field: routing_decision"
        )
    """
    def __init__(self, agent: str, schema: str, reason: str):
        super().__init__(
            message=(
                f"Failed to parse structured output in '{agent}' agent "
                f"into '{schema}': {reason}"
            ),
            context={"agent": agent, "schema": schema, "reason": reason}
        )


class ToolLoopError(AgentError):
    """
    Raised when the tool-calling loop cannot produce a valid final AIMessage.

    This can occur when:
        - No AIMessage without tool_calls exists in the message history.
        - The loop exits in an unexpected state despite the forced-summary guard.

    Example:
        raise ToolLoopError(
            agent="procurement",
            reason="No AIMessage without tool_calls found in message history."
        )
    """
    def __init__(self, agent: str, reason: str):
        super().__init__(
            message=f"Tool loop failed in '{agent}' agent: {reason}",
            context={"agent": agent, "reason": reason}
        )


# ── Repository Errors ─────────────────────────────────────

class RepositoryError(ProcurementSystemError):
    """Base for all data access errors."""


class SupplierNotFoundError(RepositoryError):
    """
    Raised when no supplier matching the given criteria
    is found in the repository.

    Example:
        raise SupplierNotFoundError(
            category="MACHINERY",
            filters={"min_rating": 8}
        )
    """
    def __init__(self, category: str, filters: dict | None = None):
        super().__init__(
            message=f"No supplier found for category '{category}'",
            context={"category": category, "filters": filters or {}}
        )


class ContractNotFoundError(RepositoryError):
    """
    Raised when no active framework agreement is found
    for the given category or supplier.

    Example:
        raise ContractNotFoundError(
            category="IT_SOFTWARE",
            supplier="TechCorp Ltd."
        )
    """
    def __init__(self, category: str, supplier: str | None = None):
        super().__init__(
            message=(
                f"No active framework agreement found "
                f"for category '{category}'"
                + (f" and supplier '{supplier}'" if supplier else "")
            ),
            context={"category": category, "supplier": supplier}
        )


# ── Tool Errors ─────────────────────────────────────


class TavilySearchError(ProcurementSystemError):
    """Raised when Tavily API search fails."""
    pass


class PDFProcessingError(ProcurementSystemError):
    """Raised when PDF processing fails."""
    pass
