"""
project_paths.py

Single source of truth for all file and directory paths in the project.

All paths are resolved relative to this file's location (PACKAGE_ROOT),
so they work correctly regardless of where the script is launched from:
    - from project root:   python main.py
    - from any directory:  python /path/to/projekt/main.py
    - from Jupyter:        %run main.py
    - from tests:          pytest tests/
"""

from pathlib import Path

# ── Package and project root ──────────────────────────────
PACKAGE_ROOT = Path(__file__).parent        # procurement_system/
PROJECT_ROOT = PACKAGE_ROOT.parent          # projekt/


# ── Top-level directories ─────────────────────────────────
DOCS_DIR     = PROJECT_ROOT    / "docs"
SCRIPTS_DIR  = PROJECT_ROOT    / "scripts"
TESTS_DIR    = PROJECT_ROOT    / "tests"


# ── Package directories ───────────────────────────────────
GRAPH_DIR          = PACKAGE_ROOT / "graph"
NODES_DIR          = PACKAGE_ROOT / "nodes"
AGENTS_DIR         = PACKAGE_ROOT / "agents"
SERVICES_DIR       = PACKAGE_ROOT / "services"
REPOSITORIES_DIR   = PACKAGE_ROOT / "repositories"
SCHEMAS_DIR        = PACKAGE_ROOT / "schemas"
UTILS_DIR          = PACKAGE_ROOT / "utils"
OBSERVABILITY_DIR  = PACKAGE_ROOT / "observability"
PROMPTS_DIR        = PACKAGE_ROOT / "prompts"
CONFIG_DIR         = PACKAGE_ROOT / "config"


# ── Config: business policy ───────────────────────────────
ENTERPRISE_BUYING_RULES      = CONFIG_DIR / "enterprise_buying_rules.yaml"


# ── Config: agent configurations ─────────────────────────
CONFIG_INTAKE_AGENT          = CONFIG_DIR / "config_intake_agent.yaml"
CONFIG_PROCUREMENT_AGENT     = CONFIG_DIR / "config_procurement_agent.yaml"
CONFIG_ANALYST_AGENT         = CONFIG_DIR / "config_analyst_agent.yaml"
CONFIG_ORCHESTRATOR          = CONFIG_DIR / "config_orchestrator.yaml"


# ── Config: logging ───────────────────────────────────────
LOGGING_CONFIG               = CONFIG_DIR / "logging.yaml"


# ── Config: llm_models ────────────────────────────────────
CONFIG_MODEL_REGISTRY         = CONFIG_DIR / "model_registry.yaml"


# ── Prompts: intake agent ─────────────────────────────────
INTAKE_AGENT_SYSTEM_PROMPT      = PROMPTS_DIR / "intake_agent_system.txt"
INTAKE_AGENT_USER_PROMPT        = PROMPTS_DIR / "intake_agent_user.txt"


# ── Prompts: procurement agent ────────────────────────────
PROCUREMENT_AGENT_SYSTEM_PROMPT = PROMPTS_DIR / "procurement_agent_system.txt"
PROCUREMENT_AGENT_USER_PROMPT   = PROMPTS_DIR / "procurement_agent_user.txt"


# ── Prompts: analyst agent ────────────────────────────────
ANALYST_AGENT_SYSTEM_PROMPT     = PROMPTS_DIR / "analyst_agent_system.txt"
ANALYST_AGENT_USER_PROMPT       = PROMPTS_DIR / "analyst_agent_user.txt"


# ── Prompts: orchestrator ─────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT      = PROMPTS_DIR / "orchestrator_system.txt"
ORCHESTRATOR_USER_PROMPT        = PROMPTS_DIR / "orchestrator_user.txt"

# ── Prompts: forced sumaty llm too loop ────────────────────
FORCED_SUMMARY_LLM_TOOL_LOOP_PROMPT      = PROMPTS_DIR / "forced_summary_llm_tool_loop.txt"

# ── Schemas files ─────────────────────────────────────────
INTAKE_SCHEMAS     = SCHEMAS_DIR / "intake_schemas.py"
PROCUREMENT_SCHEMAS= SCHEMAS_DIR / "procurement_schemas.py"
ANALYST_SCHEMAS    = SCHEMAS_DIR / "analyst_schemas.py"
ORCHESTRATOR_SCHEMAS    = SCHEMAS_DIR / "orchestrator_schemas.py"


# ── Root files ────────────────────────────────────────────
ENV_FILE                     = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE             = PROJECT_ROOT / ".env.example"
REQUIREMENTS                 = PROJECT_ROOT / "requirements.txt"
PYPROJECT                    = PROJECT_ROOT / "pyproject.toml"
README                       = PROJECT_ROOT / "README.md"


# ── Validation ────────────────────────────────────────────
def validate_paths() -> None:
    """
    Verify that all critical config and prompt files exist.
    Call once at application startup in main.py.
    Raises FileNotFoundError with a clear message if anything is missing.
    """
    critical_files = {
        # config
        "Enterprise buying rules":    ENTERPRISE_BUYING_RULES,
        "Intake agent config":        CONFIG_INTAKE_AGENT,
        "Procurement agent config":   CONFIG_PROCUREMENT_AGENT,
        "Analyst agent config":       CONFIG_ANALYST_AGENT,
        "Orchestrator config":        CONFIG_ORCHESTRATOR,
        "Logging config":             LOGGING_CONFIG,
        # prompts
        "Intake system prompt":       INTAKE_AGENT_SYSTEM_PROMPT,
        "Intake user prompt":         INTAKE_AGENT_USER_PROMPT,
        "Procurement system prompt":  PROCUREMENT_AGENT_SYSTEM_PROMPT,
        "Procurement user prompt":    PROCUREMENT_AGENT_USER_PROMPT,
        "Analyst system prompt":      ANALYST_AGENT_SYSTEM_PROMPT,
        "Analyst user prompt":        ANALYST_AGENT_USER_PROMPT,
        "Orchestrator system prompt": ORCHESTRATOR_SYSTEM_PROMPT,
        "Orchestrator user prompt":   ORCHESTRATOR_USER_PROMPT,
        # schemas
        "Intake schemas":        INTAKE_SCHEMAS,
        "Procurement schemas":   PROCUREMENT_SCHEMAS,
        "Analyst schemas":       ANALYST_SCHEMAS,
    }

    missing = [
        f"{label}: {path}"
        for label, path in critical_files.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing critical files:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
