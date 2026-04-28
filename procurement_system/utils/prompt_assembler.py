"""
utils/prompt_assembler.py

Appends a human-readable output field summary to a static system prompt.

Architecture:
    System prompt (.txt)              — static, versioned, reviewed by team
    .with_structured_output(Schema)   — enforces JSON structure via Pydantic
    build_llm_friendly_output()       — tells LLM what fields to populate
                                        and what each field means

Why this split:
    .with_structured_output() passes the full JSON Schema to the API
    via the tool_use / function_calling mechanism — not via prompt text.
    The LLM does not need a raw JSON Schema in the prompt.
    It only needs a plain-language summary: field name + what it means.

What this module does NOT do:
    - Does not generate INPUT CONTRACT (written in .txt by the analyst)
    - Does not parse or validate schemas
    - Does not manage caching (handled by settings.py lru_cache getters)
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from procurement_system.exceptions import ConfigFileNotFoundError, InvalidConfigError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# OUTPUT FIELD SUMMARY – IMPROVED & CLEAN VERSION
# ─────────────────────────────────────────────────────────

def build_llm_friendly_output(model: Type[BaseModel]) -> str:
    """
    Generates a clear, well-formatted summary of all output fields for the LLM.
    Properly handles nested models (ValidatedRequest, DecisionLogEntry, etc.)
    and clearly marks required vs optional fields.
    """
    lines = [
        "## OUTPUT STRUCTURE – STRICT REQUIREMENTS",
        "",
        "You MUST return a COMPLETE JSON object with ALL fields listed below.",
        "Never omit any field, even if the value is null, empty list, or default.",
        "Use the exact field names shown (case-sensitive).",
        "The JSON structure is strictly enforced by the parser.",
        "",
        "=== FULL LIST OF FIELDS ===",
        ""
    ]

    def walk_fields(m: Type[BaseModel], prefix: str = "", depth: int = 0):
        indent = "    " * depth   # lepsze wcięcie (4 spacje)

        for name, field_info in m.model_fields.items():
            full_name = f"{prefix}{name}"

            description = field_info.description or "(no description)"
            is_required = field_info.is_required()

            req_marker = "★ REQUIRED" if is_required else "○ optional"

            # Clean field type for readability
            field_type = str(field_info.annotation)\
                .replace("typing.", "")\
                .replace("<class '", "")\
                .replace("'>", "")

            lines.append(f"{indent}• {full_name:<40} {req_marker:<15} → {description}")
            lines.append(f"{indent}    type: {field_type}")

            # === Handle nested models ===
            annotation = field_info.annotation

            # Case 1: Direct nested Pydantic model (e.g. decision_log: DecisionLogEntry)
            if hasattr(annotation, "model_fields"):
                lines.append(f"{indent}    → nested object containing:")
                walk_fields(annotation, prefix=f"{full_name}.", depth=depth + 1)

            # Case 2: List[SomeModel] or Optional[List[SomeModel]]
            elif hasattr(getattr(annotation, "__args__", None), "__len__") and len(annotation.__args__) > 0:
                inner_type = annotation.__args__[0]
                if hasattr(inner_type, "model_fields"):
                    lines.append(f"{indent}    → list of objects containing:")
                    walk_fields(inner_type, prefix=f"{full_name}[].", depth=depth + 1)

            lines.append("")  # empty line between fields

    walk_fields(model)

    # Final summary
    lines.extend([
        "SUMMARY AND RULES:",
        "→ Fields marked ★ REQUIRED must always be present.",
        "→ Nested fields (e.g. validated_request.description, decision_log.routing_decision) must also be included.",
        "→ Do not skip any key. If value is unknown → use null, [] or sensible default.",
        "→ Return ONLY the JSON object. No extra text or markdown."
    ])

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# HASH AND CACHE
# ─────────────────────────────────────────────────────────

def _hash_file(path: str) -> str:
    """SHA-256 hash of file content. Cache key for template changes."""
    file_path = Path(path)

    if not file_path.exists():
        raise ConfigFileNotFoundError(path=path)

    try:
        content = file_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise InvalidConfigError(
            file=path,
            reason=f"Cannot read template: {e}",
        ) from e

    if not content:
        raise InvalidConfigError(file=path, reason="Template file is empty")

    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _hash_model(model: Type[BaseModel]) -> str:
    """
    SHA-256 hash of model field names and descriptions.
    Cache key — changes when a field is added, removed, or
    its description changes.
    """
    fields_repr = str({
        name: field.description or ""
        for name, field in model.model_fields.items()
    })
    return hashlib.sha256(fields_repr.encode()).hexdigest()[:16]


@lru_cache(maxsize=16)
def _assemble_cached(
    template_path:  str,
    template_hash:  str,    # cache key — changes when .txt edited
    output_hash:    str,    # cache key — changes when model fields change
    output_section: str,    # pre-built output summary
) -> str:
    """
    Read template and append output field summary.
    Cached by (template_hash, output_hash) — rebuilt only when either changes.
    """
    try:
        template = Path(template_path).read_text(encoding="utf-8").strip()
    except OSError as e:
        raise InvalidConfigError(
            file=template_path,
            reason=f"Cannot read template during assembly: {e}",
        ) from e

    assembled = f"{template}\n\n{output_section}"

    logger.info(
        "System prompt assembled: template=%s output=%s chars=%s",
        template_hash,
        output_hash,
        len(assembled),
    )

    return assembled


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

def assemble_system_prompt(
    template_path: str,
    output_schema: Type[BaseModel],
) -> str:
    """
    Append a plain-language output field summary to a static system prompt.

    The static .txt file contains:
        ROLE, INPUT CONTRACT, APPROACH, STEPS, CONSTRAINTS, STYLE.

    This function appends:
        ## OUTPUT FIELDS — field name + description per output field.

    JSON structure is NOT included in the prompt — it is passed to the
    API via .with_structured_output(output_schema) separately.

    Cache behaviour:
        Rebuilt when template file changes (template_hash) or when
        output schema field names/descriptions change (output_hash).

    Args:
        template_path:
            Path to the static .txt system prompt file.

        output_schema:
            Pydantic BaseModel for this agent's structured output.
            Field descriptions from Field(description=...) are used.

    Returns:
        System prompt string with output field summary appended.

    Raises:
        ConfigFileNotFoundError: template_path does not exist.
        InvalidConfigError:      Template empty, unreadable.

    Example:
        class AnalystAgent:
            def __init__(self):
                self._system_prompt = assemble_system_prompt(
                    template_path = str(ANALYST_AGENT_SYSTEM_PROMPT),
                    output_schema = AnalystAgentOutput,
                )
                self._structured_llm = get_structured_llm(AnalystAgentOutput)
    """
    template_hash  = _hash_file(template_path)
    output_hash    = _hash_model(output_schema)
    output_section = build_llm_friendly_output(output_schema)

    return _assemble_cached(
        template_path  = template_path,
        template_hash  = template_hash,
        output_hash    = output_hash,
        output_section = output_section,
    )
