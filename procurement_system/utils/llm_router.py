# procurement_system/utils/llm_router.py
"""
LLM routing via litellm.Router.

litellm.Router handles natively:
  - Fallbacks on any error               (fallbacks)
  - Fallbacks on context-window breach   (context_window_fallbacks)
  - Skipping models with too-small ctx   (enable_pre_call_checks)
  - Per-error-type retry counts          (RetryPolicy)
  - Per-error-type cooldown thresholds   (AllowedFailsPolicy)
  - Model cooldown + auto-recovery       (allowed_fails, cooldown_time)

This module is responsible for:
  1. Instantiating the Router from a ModelChain.
  2. Attaching mode-specific parameters (structured output / tools).
  3. Wrapping the call in a LangChain RunnableLambda.

LangChain BaseMessage → OpenAI dict conversion is intentional:
litellm.Router speaks OpenAI format; LangChain 'type' field ('human', 'ai')
differs from OpenAI 'role' field ('user', 'assistant').
No LiteLLM utility exists for this mapping (verified empirically).
"""

import json
import logging
from typing import Any, List, Optional, Type

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm import Router, completion_cost
from litellm.router import AllowedFailsPolicy, RetryPolicy

from procurement_system.constants import BuilderMode, FallbackStrategy
from procurement_system.utils.model_resolver import ModelChain, PRIMARY_GROUP

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Router-level reliability policy
# Tune these values to match your SLA and provider rate-limit characteristics.
# ─────────────────────────────────────────────────────────────────────────────

_RETRY_POLICY = RetryPolicy(
    RateLimitErrorRetries=1,               # exponential backoff on 429
    TimeoutErrorRetries=2,                 # retry on request timeout
    InternalServerErrorRetries=2,          # retry on provider 5xx
    ContentPolicyViolationErrorRetries=0,  # never retry; escalate immediately
    AuthenticationErrorRetries=0,          # never retry; bad key won't recover
)

_ALLOWED_FAILS_POLICY = AllowedFailsPolicy(
    RateLimitErrorAllowedFails=1,
    TimeoutErrorAllowedFails=3,
    InternalServerErrorAllowedFails=3,
    AuthenticationErrorAllowedFails=1,     # cooldown after first auth failure
    ContentPolicyViolationErrorAllowedFails=2,
)

# LangChain message type → OpenAI role mapping
_LC_ROLE: dict[str, str] = {
    "human":    "user",
    "ai":       "assistant",
    "system":   "system",
    "tool":     "tool",
    "function": "function",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_primary_with_fallback(
    chain:         ModelChain,
    mode:          BuilderMode,
    output_schema: Optional[Type]        = None,
    tools:         Optional[List[BaseTool]] = None,
) -> Runnable:
    """
    Return a LangChain ``Runnable`` backed by a ``litellm.Router``.

    The Router instance is created once per call to this function and
    captured in the closure — it is reused across all ``invoke()`` calls,
    preserving cooldown state and retry counters.

    Args:
        chain:         Resolved model chain (primary + fallbacks + strategy).
        mode:          ``STRUCTURED`` | ``TOOLS`` | ``BASE``
        output_schema: Pydantic model for structured output (``STRUCTURED`` only).
        tools:         LangChain tools to expose  (``TOOLS`` only).

    Returns:
        ``RunnableLambda`` compatible with any LangChain pipeline.

    Raises:
        ValueError: Required arguments missing for the requested mode.
    """
    if mode == BuilderMode.STRUCTURED and output_schema is None:
        raise ValueError("output_schema is required for STRUCTURED mode.")
    if mode == BuilderMode.TOOLS and not tools:
        raise ValueError("tools must be a non-empty list for TOOLS mode.")

    router = _build_router(chain)

    # Build extra kwargs once — convert_to_openai_tool is not free
    extra: dict[str, Any] = {}
    if mode == BuilderMode.STRUCTURED:
        extra["response_format"] = output_schema
    elif mode == BuilderMode.TOOLS:
        extra["tools"]       = [convert_to_openai_tool(t) for t in tools]
        extra["tool_choice"] = "auto"

    def invoke(messages: Any) -> Any:
        normalised = _normalise_messages(messages)

        logger.info(
            "Router dispatch — strategy=%s primary=%s mode=%s",
            chain.fallback_strategy.value,
            chain.primary.model,
            mode.value,
        )

        response = router.completion(
            model=PRIMARY_GROUP, messages=normalised, **extra
        )

        _log_usage(response)

        # litellm.Router always returns ModelResponse.
        # Each mode expects a different type downstream:
        #   STRUCTURED → Pydantic instance  (BaseAgent._invoke_structured_llm)
        #   TOOLS      → AIMessage          (execute_tool_loop / _extract_final_ai_message)
        #   BASE       → ModelResponse      (caller handles raw response)
        if mode == BuilderMode.STRUCTURED:
            return _parse_structured_response(response, output_schema)
        if mode == BuilderMode.TOOLS:
            return _to_ai_message(response)

        return response

    return RunnableLambda(invoke)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_router(chain: ModelChain) -> Router:
    """
    Instantiate a ``litellm.Router`` from a ``ModelChain``.

    Fallbacks and context-window fallbacks are only wired up when the
    strategy is ``ON_ERROR`` and at least one fallback model is defined.
    For ``NEVER`` and ``FAIL_FAST`` the router has a single deployment and
    will raise immediately on failure.
    """
    use_fallbacks   = (
        chain.fallback_strategy == FallbackStrategy.ON_ERROR
        and bool(chain.fallbacks)
    )
    fallback_groups = chain.to_fallback_groups() if use_fallbacks else None
    fallback_list   = _build_chained_fallbacks(fallback_groups) if fallback_groups else None

    return Router(
        model_list=chain.to_router_model_list(),

        # ── fallback wiring ───────────────────────────────────────── #
        fallbacks=fallback_list,
        context_window_fallbacks=fallback_list,   # also catch token-overflow errors
        enable_pre_call_checks=True,              # skip models with too-small context

        # ── retry / cooldown ─────────────────────────────────────── #
        retry_policy=        _RETRY_POLICY,
        allowed_fails_policy=_ALLOWED_FAILS_POLICY,
        allowed_fails=2,       # global: cooldown after 3 failures / minute
        cooldown_time=120,      # seconds before a cooled-down model recovers
        retry_after=10,         # minimum wait (s) between retries

        set_verbose=False,
    )


def _build_chained_fallbacks(groups: list[str]) -> list[dict]:
    """
    Build a chained fallback list from a flat list of group names.

    Example:
        Input:  ['fallback_0', 'fallback_1', 'fallback_2']
        Output: [
            {'primary':    ['fallback_0']},
            {'fallback_0': ['fallback_1']},
            {'fallback_1': ['fallback_2']},
        ]

    Each model knows only its immediate successor, ensuring retry policy
    and cooldown are exhausted per link before advancing the chain.
    """
    all_groups = [PRIMARY_GROUP] + groups
    return [
        {all_groups[i]: [all_groups[i + 1]]}
        for i in range(len(all_groups) - 1)
    ]


def _normalise_messages(messages: Any) -> list[dict]:
    """
    Convert LangChain ``BaseMessage`` objects to OpenAI-format dicts.

    litellm.Router speaks OpenAI format; plain dicts are passed through
    unchanged so that callers who already have OpenAI-format messages pay
    no conversion cost.

    Handles all message types produced by the tool loop:
      - HumanMessage / SystemMessage → {"role": ..., "content": ...}
      - AIMessage (no tool calls)    → {"role": "assistant", "content": ...}
      - AIMessage (with tool calls)  → {"role": "assistant", "content": None,
                                        "tool_calls": [...]}
      - ToolMessage                  → {"role": "tool", "content": ...,
                                        "tool_call_id": ...}
    """
    if isinstance(messages, (dict, BaseMessage)):
        messages = [messages]

    return [_lc_message_to_openai(m) if isinstance(m, BaseMessage) else m
            for m in messages]


def _lc_message_to_openai(m: BaseMessage) -> dict:
    """Serialise a single LangChain BaseMessage to an OpenAI-format dict."""
    role = _LC_ROLE.get(m.type, m.type)

    # ToolMessage — must carry tool_call_id for the model to correlate results
    if m.type == "tool":
        return {
            "role":         "tool",
            "content":      str(m.content),
            "tool_call_id": getattr(m, "tool_call_id", ""),
        }

    # AIMessage that contains tool calls — content must be None (not "")
    # and tool_calls must be serialised in OpenAI function-call format.
    if m.type == "ai" and getattr(m, "tool_calls", None):
        return {
            "role":       "assistant",
            "content":    None,
            "tool_calls": [
                {
                    "id":       tc["id"],
                    "type":     "function",
                    "function": {
                        "name":      tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in m.tool_calls
            ],
        }

    # All other messages (HumanMessage, SystemMessage, plain AIMessage)
    return {"role": role, "content": m.content}


def _to_ai_message(response: Any) -> AIMessage:
    """
    Convert a ``litellm.ModelResponse`` to a LangChain ``AIMessage``.

    Required for TOOLS mode: ``execute_tool_loop`` and
    ``_extract_final_ai_message`` in BaseAgent both work exclusively with
    LangChain ``BaseMessage`` / ``AIMessage`` objects.

    Handles two cases:
      - Model produced text only  → AIMessage(content=text)
      - Model called a tool       → AIMessage(content="", tool_calls=[...])
    """
    try:
        msg = response.choices[0].message
    except (AttributeError, IndexError) as exc:
        raise ValueError(
            f"Unexpected ModelResponse structure — cannot extract message: {exc}"
        ) from exc

    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append(
                {
                    "id":   tc.id,
                    "name": tc.function.name,
                    "args": args,
                    "type": "tool_call",
                }
            )

    return AIMessage(
        content=msg.content or "",
        tool_calls=tool_calls,
    )


def _parse_structured_response(response: Any, schema: Type) -> Any:
    """
    Extract and validate the JSON payload from a ``litellm.ModelResponse``.

    litellm.Router always returns a ``ModelResponse`` regardless of
    ``response_format``.  The structured content lives as a JSON string in
    ``choices[0].message.content``; this function parses it into the
    caller-supplied Pydantic schema.

    Raises:
        ValueError: The model returned empty content.
        pydantic.ValidationError: The JSON does not match the schema —
            indicates a model hallucination or wrong prompt.
    """
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise ValueError(
            f"Unexpected ModelResponse structure — cannot extract content: {exc}"
        ) from exc

    if not content:
        raise ValueError(
            f"Model returned empty content; cannot parse into {schema.__name__}."
        )

    logger.debug("Parsing structured response into %s", schema.__name__)

    # model_validate_json raises pydantic.ValidationError on mismatch —
    # let it propagate so the caller (agent retry logic) can handle it.
    return schema.model_validate_json(content)


def _log_usage(response: Any) -> None:
    """Log cost and token counts; best-effort — never raises."""
    try:
        cost  = completion_cost(completion_response=response)
        usage = response.usage
        logger.info(
            "LLM usage — model=%s cost=$%.6f input_tokens=%s output_tokens=%s",
            response.model,
            cost,
            usage.prompt_tokens    if usage else "?",
            usage.completion_tokens if usage else "?",
        )
    except Exception:
        logger.debug("Cost calculation unavailable.", exc_info=True)
