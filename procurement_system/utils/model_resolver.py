# procurement_system/utils/model_resolver.py
"""
Model configuration resolution.

Translates a tier name (from model_registry.yaml) or explicit parameters
into a ModelChain — the single source of model truth consumed by llm_router.

Design decisions:
  - Pydantic BaseModel replaces @dataclass: validation, defaults, and
    model_validate() from dict eliminate the need for a manual _parse_config().
  - ModelChain.to_router_model_list() / to_fallback_groups() keep all
    litellm.Router serialisation concerns in one place.
  - Both resolve_* functions are the only public entry points;
    callers never construct ModelConfig / ModelChain by hand.
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from procurement_system.constants import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_RPM,
    DEFAULT_TPM,
    FallbackStrategy,
)
from procurement_system.settings import get_model_registry

logger = logging.getLogger(__name__)

# Group-name constants shared with llm_router to avoid string literals
PRIMARY_GROUP   = "primary"
FALLBACK_PREFIX = "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """
    Immutable configuration for a single LLM deployment.

    All fields are validated by Pydantic on construction.
    Defaults mirror the system-wide constants so that partial YAML entries
    (e.g. a fallback that only specifies 'model') are still valid.
    """

    model:       str   = DEFAULT_MODEL
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens:  int   = Field(DEFAULT_MAX_TOKENS,  ge=1)
    timeout:     int   = Field(DEFAULT_LLM_TIMEOUT_SECONDS, ge=1)
    rpm: Optional[int] = Field(default=DEFAULT_RPM, ge=1)
    tpm: Optional[int] = Field(default=DEFAULT_TPM, ge=1)

    model_config = {"frozen": True}          # immutable after construction


class ModelChain(BaseModel):
    """
    Ordered sequence of LLM deployments: primary first, then fallbacks.

    Attributes:
        primary:           The preferred deployment.
        fallbacks:         Ordered alternatives tried according to strategy.
        fallback_strategy: Controls when (if ever) fallbacks are activated.
    """

    primary:           ModelConfig
    fallbacks:         List[ModelConfig] = []
    fallback_strategy: FallbackStrategy  = FallbackStrategy.ON_ERROR

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _warn_unused_fallbacks(self) -> "ModelChain":
        """Emit a warning when fallbacks are defined but strategy ignores them."""
        if self.fallbacks and self.fallback_strategy != FallbackStrategy.ON_ERROR:
            logger.warning(
                "ModelChain has %d fallback(s) but strategy='%s' — "
                "they will never be used.",
                len(self.fallbacks),
                self.fallback_strategy.value,
            )
        return self

    # ── litellm.Router serialisation ─────────────────────────────────────── #

    def to_router_model_list(self) -> list[dict]:
        """
        Serialise the chain into the ``model_list`` format expected by
        ``litellm.Router``.

        Example output::

            [
              {"model_name": "primary",    "litellm_params": {...}},
              {"model_name": "fallback_0", "litellm_params": {...}},
            ]
        """
        entries = [_to_router_entry(PRIMARY_GROUP, self.primary)]
        for i, fb in enumerate(self.fallbacks):
            entries.append(_to_router_entry(f"{FALLBACK_PREFIX}_{i}", fb))
        return entries

    def to_fallback_groups(self) -> list[str]:
        """
        Ordered group names for ``fallbacks`` and ``context_window_fallbacks``
        arguments of ``litellm.Router``.  Returns an empty list when there are
        no fallbacks configured.
        """
        return [f"{FALLBACK_PREFIX}_{i}" for i in range(len(self.fallbacks))]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_router_entry(group: str, cfg: ModelConfig) -> dict:
    """Wrap a ModelConfig in the litellm router entry schema."""
    return {
        "model_name":    group,
        "litellm_params": cfg.model_dump(exclude_none=True),   # temperature, max_tokens, timeout, model
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def resolve_from_tier(tier_name: str) -> ModelChain:
    """
    Build a ``ModelChain`` from a named tier in ``model_registry.yaml``.

    Args:
        tier_name: Registry key, e.g. ``"reasoning_heavy"``.

    Returns:
        Fully validated ``ModelChain`` with primary and fallbacks.

    Raises:
        ValueError: Tier is absent from the registry.
        pydantic.ValidationError: A model entry in the YAML violates
            field constraints (e.g. negative timeout).
    """
    registry  = get_model_registry()
    available = [k for k in registry if not k.startswith("_")]

    if tier_name not in registry:
        raise ValueError(
            f"Unknown model tier '{tier_name}'. "
            f"Available tiers: {available}"
        )

    tier      = registry[tier_name]
    raw_strat = tier.get("fallback_strategy", FallbackStrategy.ON_ERROR.value)
    strategy  = FallbackStrategy(raw_strat) if isinstance(raw_strat, str) else raw_strat

    chain = ModelChain(
        primary=           ModelConfig.model_validate(tier["primary"]),
        fallbacks=         [ModelConfig.model_validate(f) for f in tier.get("fallback", [])],
        fallback_strategy= strategy,
    )

    logger.debug(
        "Resolved tier '%s': primary=%s fallbacks=%d strategy=%s",
        tier_name, chain.primary.model, len(chain.fallbacks), strategy,
    )
    return chain


def resolve_from_params(
    model:       Optional[str]   = None,
    temperature: Optional[float] = None,
    max_tokens:  Optional[int]   = None,
    timeout:     Optional[int]   = None,
) -> ModelChain:
    """
    Build a single-model ``ModelChain`` from explicit parameters.

    Missing parameters fall back to system-wide defaults.
    The resulting chain has no fallbacks and strategy ``NEVER``.

    Args:
        model:       LiteLLM model string, e.g. ``"anthropic/claude-sonnet-4-6"``.
        temperature: Sampling temperature (0.0 – 2.0).
        max_tokens:  Maximum output tokens (≥ 1).
        timeout:     Request timeout in seconds (≥ 1).

    Raises:
        pydantic.ValidationError: A supplied value violates field constraints.
    """
    primary = ModelConfig.model_validate(
        {
            k: v for k, v in {
                "model":       model,
                "temperature": temperature,
                "max_tokens":  max_tokens,
                "timeout":     timeout,
            }.items()
            if v is not None           # let Pydantic apply defaults for missing keys
        }
    )

    logger.debug("Resolved explicit config: model=%s", primary.model)

    return ModelChain(
        primary=           primary,
        fallbacks=         [],
        fallback_strategy= FallbackStrategy.NEVER,
    )
