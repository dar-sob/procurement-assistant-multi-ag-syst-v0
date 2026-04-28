# procurement_system/settings.py
"""
settings.py

Centralized settings and cached config getters for the
Multi-Agent Procurement System.

Responsibilities:
    1. Load environment variables from .env (via python-dotenv)
    2. Expose typed environment variables (API keys, DB URL, etc.)
    3. Provide lru_cache getters for all YAML config files
    4. Provide lru_cache getters for all prompt text files

Why lru_cache:
    Each getter reads from disk exactly once — on first call.
    Every subsequent call returns the cached result.
    No file I/O overhead during agent execution.

Usage:
    from procurement_system.settings import (
        get_anthropic_api_key,
        get_intake_config,
        get_buying_rules,
        get_intake_system_prompt,
    )
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

from procurement_system.project_paths import (
    # config
    ENTERPRISE_BUYING_RULES,
    CONFIG_INTAKE_AGENT,
    CONFIG_PROCUREMENT_AGENT,
    CONFIG_ANALYST_AGENT,
    CONFIG_ORCHESTRATOR,
    LOGGING_CONFIG,
    CONFIG_MODEL_REGISTRY,
    # prompts
    INTAKE_AGENT_SYSTEM_PROMPT,
    INTAKE_AGENT_USER_PROMPT,
    PROCUREMENT_AGENT_SYSTEM_PROMPT,
    PROCUREMENT_AGENT_USER_PROMPT,
    ANALYST_AGENT_SYSTEM_PROMPT,
    ANALYST_AGENT_USER_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_USER_PROMPT,
    FORCED_SUMMARY_LLM_TOOL_LOOP_PROMPT)

from procurement_system.constants import(
    DEFAULT_FRANKFURTER_URL,
    DEFAULT_FRANKFURTER_TIMEOUT
)
from procurement_system.utils.yaml_loader import load_yaml
from procurement_system.utils.prompt_loader import load_prompt
from procurement_system.utils.prompt_assembler import assemble_system_prompt
from procurement_system.exceptions import MissingConfigFieldError
from procurement_system.schemas.intake_schemas import IntakeAgentOutput
from procurement_system.schemas.procurement_schemas import ProcurementAgentOutput
from procurement_system.schemas.analyst_schemas import AnalystAgentOutput
from procurement_system.schemas.orchestrator_schemas import LLMFinalReport



# Load .env file into os.environ at import time
load_dotenv()


# ── Environment Variables ─────────────────────────────────

def get_anthropic_api_key() -> str:
    """
    Return the Anthropic API key from environment.
    Raises MissingConfigFieldError if not set.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise MissingConfigFieldError(
            file=".env",
            field="ANTHROPIC_API_KEY"
        )
    return key


def get_database_url() -> str | None:
    """
    Return the database URL from environment.
    Returns None if not set — optional for MVP (MemorySaver).
    Required for production (PostgresSaver checkpointer).
    """
    return os.getenv("DATABASE_URL")


# ── Config Getters (lru_cache) ────────────────────────────

@lru_cache
def get_buying_rules() -> dict:
    """
    Load and cache enterprise_buying_rules.yaml.
    Contains: thresholds, categories, framework agreements,
    escalation rules.
    """
    return load_yaml(ENTERPRISE_BUYING_RULES)


@lru_cache
def get_intake_config() -> dict:
    """Load and cache config_intake_agent.yaml."""
    return load_yaml(CONFIG_INTAKE_AGENT)


@lru_cache
def get_procurement_config() -> dict:
    """Load and cache config_procurement_agent.yaml."""
    return load_yaml(CONFIG_PROCUREMENT_AGENT)


@lru_cache
def get_analyst_config() -> dict:
    """Load and cache config_analyst_agent.yaml."""
    return load_yaml(CONFIG_ANALYST_AGENT)


@lru_cache
def get_orchestrator_config() -> dict:
    """Load and cache config_orchestrator.yaml."""
    return load_yaml(CONFIG_ORCHESTRATOR)


@lru_cache
def get_logging_config() -> dict:
    """Load and cache logging.yaml."""
    return load_yaml(LOGGING_CONFIG)


# ── Prompt Getters (lru_cache) ────────────────────────────

@lru_cache()
def get_forced_summary_llm_tool_loop() -> str:
    """Load and cache forced summary prompt template, after hit max loop for tool."""
    return load_prompt(FORCED_SUMMARY_LLM_TOOL_LOOP_PROMPT)

@lru_cache()
def get_intake_system_prompt() -> str: # *27.03.2026
    return assemble_system_prompt(
        template_path=str(INTAKE_AGENT_SYSTEM_PROMPT),
        output_schema=IntakeAgentOutput
    )

@lru_cache
def get_intake_user_prompt() -> str:
    """Load and cache intake agent user prompt template."""
    return load_prompt(INTAKE_AGENT_USER_PROMPT)


@lru_cache()
def get_procurement_system_prompt() -> str: # *27.03.2026
    return assemble_system_prompt(
        template_path=str(PROCUREMENT_AGENT_SYSTEM_PROMPT),
        output_schema=ProcurementAgentOutput
    )


@lru_cache
def get_procurement_user_prompt() -> str:
    """Load and cache procurement agent user prompt template."""
    return load_prompt(PROCUREMENT_AGENT_USER_PROMPT)


@lru_cache()
def get_analyst_system_prompt() -> str: # *27.03.2026
    return assemble_system_prompt(
        template_path=str(ANALYST_AGENT_SYSTEM_PROMPT),
        output_schema=AnalystAgentOutput
    )


@lru_cache
def get_analyst_user_prompt() -> str:
    """Load and cache analyst agent user prompt template."""
    return load_prompt(ANALYST_AGENT_USER_PROMPT)


@lru_cache()
def get_orchestrator_system_prompt() -> str: # *27.03.2026
    return assemble_system_prompt(
        template_path=str(ORCHESTRATOR_SYSTEM_PROMPT),
        output_schema=LLMFinalReport
    )


@lru_cache
def get_orchestrator_user_prompt() -> str:
    """Load and cache orchestrator user prompt template."""
    return load_prompt(ORCHESTRATOR_USER_PROMPT)

# ── Tools (lru_cache)          ────────────────────────────

def get_frankfurter_api_url() -> str:
    return os.getenv("FRANKFURTER_API_URL", DEFAULT_FRANKFURTER_URL)


def get_frankfurter_timeout() -> int:
    return int(os.getenv("FRANKFURTER_TIMEOUT", str(DEFAULT_FRANKFURTER_TIMEOUT)))


@lru_cache
def get_tavily_api_key() -> str:
    """Return the Tavily API key from environment."""
    return os.getenv("TAVILY_API_KEY", "")


@lru_cache
def get_model_registry() -> dict:
    """Load model registry from YAML file with caching."""
    path = CONFIG_MODEL_REGISTRY
    return load_yaml(path)

@lru_cache
def get_tesseract_cmd() -> str:
    import platform
    if platform.system() == "Windows":
        return r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return "tesseract" 


# ── Cache Management ──────────────────────────────────────

def clear_all_caches() -> None:
    """
    Clear all lru_cache caches.
    Use during development when prompt or config files are modified
    without restarting the kernel.

    Example:
        from procurement_system.settings import clear_all_caches
        clear_all_caches()
    """
    get_buying_rules.cache_clear()
    get_intake_config.cache_clear()
    get_procurement_config.cache_clear()
    get_analyst_config.cache_clear()
    get_orchestrator_config.cache_clear()
    get_logging_config.cache_clear()
    get_intake_system_prompt.cache_clear()
    get_intake_user_prompt.cache_clear()
    get_procurement_system_prompt.cache_clear()
    get_procurement_user_prompt.cache_clear()
    get_analyst_system_prompt.cache_clear()
    get_analyst_user_prompt.cache_clear()
    get_orchestrator_system_prompt.cache_clear()
    get_orchestrator_user_prompt.cache_clear()
    get_tavily_api_key()
    get_model_registry()
    get_tesseract_cmd()
