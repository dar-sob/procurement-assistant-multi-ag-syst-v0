"""
utils/buying_rules_prompt_builder.py

Serializes enterprise buying rules into prompt-ready text
for injection into agent user prompts at runtime.

Pydantic models for structure validation live in:
    schemas/buying_rules_schemas.py

Production features:
    - Pydantic validation via BuyingRulesModel (in schemas/)
    - Deterministic ordering (thresholds sorted by min_value ascending)
    - Section-by-section truncation with priority:
        1. thresholds  — always included (routing depends on this)
        2. categories  — truncated if budget exceeded
        3. agreements  — truncated if budget exceeded
        4. escalation  — truncated last
    - tiktoken for accurate token counting (not len // 4 approximation)
    - Lazy % formatting in all logger calls
"""

import logging

from procurement_system.schemas.buying_rules_schemas import (
    AgreementModel,
    BuyingRulesModel,
    CategoryModel,
    EscalationRuleModel,
    ThresholdModel,
)

logger = logging.getLogger(__name__)

# ── Token counting ────────────────────────────────────────
_CHARS_PER_TOKEN = 4  # fallback approximation only

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
    logger.debug("tiktoken loaded — using accurate token counting")
except ImportError:
    _ENCODER = None
    _USE_TIKTOKEN = False
    logger.warning(
        "tiktoken not installed — using char approximation (%s chars/token). "
        "Install with: pip install tiktoken",
        _CHARS_PER_TOKEN,
    )


def _count_tokens(text: str) -> int:
    """Count tokens accurately with tiktoken or fall back to approximation."""
    if _USE_TIKTOKEN and _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return len(text) // _CHARS_PER_TOKEN


# ── Section builders ──────────────────────────────────────

def _build_thresholds_section(thresholds: list[ThresholdModel]) -> str:
    """Always included. Sorted ascending by min_value — deterministic order."""
    sorted_thresholds = sorted(thresholds, key=lambda t: t.min_value)

    lines = ["VALUE THRESHOLDS (apply to estimated total purchase value):"]
    for t in sorted_thresholds:
        min_v = f"${t.min_value:,.0f}"
        max_v = f"${t.max_value:,.0f}" if t.max_value is not None else "no upper limit"
        lines.append(
            f"  {min_v} to {max_v}"
            f" → {t.label} | {t.display_name}"
            f" | min {t.required_quotes} quote(s)"
            f" | approval: {t.approval_required}"
        )
    return "\n".join(lines)


def _build_categories_section(categories: list[CategoryModel]) -> str:
    """Sorted alphabetically by id — deterministic order."""
    sorted_categories = sorted(categories, key=lambda c: c.id)

    lines = ["PROCUREMENT CATEGORIES (use id value exactly as listed):"]
    for c in sorted_categories:
        examples = ", ".join(c.examples[:3]) if c.examples else "see policy"
        lines.append(
            f"  id: {c.id}"
            f" | {c.label}"
            f" | examples: {examples}"
            f" | preferred process: {c.preferred_process}"
        )
    return "\n".join(lines)


def _build_agreements_section(agreements: list[AgreementModel]) -> str:
    """Sorted by agreement_id — deterministic order."""
    sorted_agreements = sorted(agreements, key=lambda a: a.agreement_id)

    lines = ["ACTIVE FRAMEWORK AGREEMENTS:"]
    for a in sorted_agreements:
        lines.append(
            f"  {a.agreement_id}"
            f" | supplier: {a.supplier}"
            f" | category: {a.category}"
            f" | valid until: {a.valid_until}"
            f" | max order: ${a.max_order_value:,.0f}"
            + (f" | note: {a.notes}" if a.notes else "")
        )
    return "\n".join(lines)


def _build_escalation_section(rules: list[EscalationRuleModel]) -> str:
    """Listed in YAML order — policy-defined sequence."""
    lines = ["ESCALATION RULES (evaluate each rule against the request):"]
    for r in rules:
        lines.append(f"  trigger: {r.trigger} → action: {r.action}")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────

def build_buying_rules_text(
    buying_rules_dict: dict,
    max_tokens: int = 2000,
) -> str:
    """
    Validate and serialize enterprise buying rules into prompt-ready text.

    Sections included with the following priority under token budget:
        1. thresholds  — always included (routing depends on this)
        2. categories  — included if budget allows
        3. agreements  — included if budget allows
        4. escalation  — included if budget allows

    Args:
        buying_rules_dict:
            Parsed content of enterprise_buying_rules.yaml.
            Validated against BuyingRulesModel (schemas/buying_rules_schemas.py).
        max_tokens:
            Maximum token budget for the serialized output.
            Default: 2000 — override via config/config_intake_agent.yaml.

    Returns:
        Formatted text string ready for injection into agent user prompt.

    Raises:
        pydantic.ValidationError: If buying_rules_dict structure is invalid.
    """

    # ── Validate via schema in schemas/ ──────────────────
    raw   = buying_rules_dict.get("procurement_strategy", buying_rules_dict)
    rules = BuyingRulesModel.model_validate(raw)

    logger.debug(
        "Building buying rules text: %s thresholds, %s categories, "
        "%s agreements, %s escalation rules",
        len(rules.thresholds),
        len(rules.categories),
        len(rules.framework_agreements),
        len(rules.escalation),
    )

    # ── Build sections with priority ─────────────────────
    sections: list[tuple[str, str]] = [
        ("thresholds", _build_thresholds_section(rules.thresholds)),
        ("categories", _build_categories_section(rules.categories)),
        ("agreements", _build_agreements_section(rules.framework_agreements)),
        ("escalation", _build_escalation_section(rules.escalation)),
    ]

    result_parts: list[str] = []
    tokens_used = 0

    for section_name, section_text in sections:
        section_tokens = _count_tokens(section_text)

        if section_name == "thresholds":
            result_parts.append(section_text)
            tokens_used += section_tokens
            logger.debug(
                "Section '%s' included: ~%s tokens (mandatory)",
                section_name,
                section_tokens,
            )
            continue

        if tokens_used + section_tokens <= max_tokens:
            result_parts.append(section_text)
            tokens_used += section_tokens
            logger.debug(
                "Section '%s' included: ~%s tokens (total so far: ~%s)",
                section_name,
                section_tokens,
                tokens_used,
            )
        else:
            logger.warning(
                "Section '%s' excluded — token budget exceeded "
                "(%s tokens needed, %s remaining)",
                section_name,
                section_tokens,
                max_tokens - tokens_used,
            )

    result = "\n\n".join(result_parts)

    logger.debug(
        "Buying rules serialized: %s chars, ~%s tokens, %s sections included",
        len(result),
        _count_tokens(result),
        len(result_parts),
    )

    return result
