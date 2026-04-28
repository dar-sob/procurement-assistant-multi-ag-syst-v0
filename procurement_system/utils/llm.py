"""
utils/llm.py

LLM initialization using LiteLLM — single interface for all models.
Includes retry policy, timeout and error handling.
"""
import logging
from dotenv import load_dotenv
from typing import Type, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from langchain_litellm import ChatLiteLLM
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from litellm.exceptions import (
    Timeout,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    BadRequestError,
)

from procurement_system.constants import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    LLM_RETRY_MAX_ATTEMPTS,
    LLM_RETRY_WAIT_MULTIPLIER,
    LLM_RETRY_WAIT_MIN,
    LLM_RETRY_WAIT_MAX,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)
from procurement_system.exceptions import LLMCallError


logger = logging.getLogger(__name__)


# ── Retry Policy ──────────────────────────────────────────
# Transient errors — safe to retry
# Auth and bad request errors — must NOT retry

TRANSIENT_ERRORS = (
    Timeout,
    RateLimitError,
    ServiceUnavailableError,
)

retry_policy = retry(
    stop=stop_after_attempt(LLM_RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(
        multiplier=LLM_RETRY_WAIT_MULTIPLIER,
        min=LLM_RETRY_WAIT_MIN,
        max=LLM_RETRY_WAIT_MAX,
    ),
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# ── LLM Factory ───────────────────────────────────────────

def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> ChatLiteLLM:
    """
    Return a ChatLiteLLM instance for any supported model.

    Model is selected by name string — LiteLLM routes automatically:
        "claude-sonnet-4-6"      -> Anthropic
        "gpt-4o"                 -> OpenAI
        "gemini/gemini-pro"      -> Google
        "ollama/llama3"          -> Ollama (local)

    Args:
        model:       Model name string
        temperature: Sampling temperature
        max_tokens:  Maximum tokens in response
        timeout:     Request timeout in seconds

    Returns:
        ChatLiteLLM instance

    Raises:
        LLMCallError: On auth failure, bad request, or
                      transient error after 3 retries
    """
    logger.debug(f"Initializing LLM: model={model}, temperature={temperature}, timeout={timeout}")

    try:
        return _init_llm(model, temperature, max_tokens, timeout)

    except AuthenticationError as e:
        raise LLMCallError(
            agent="llm_factory",
            reason=f"Authentication failed for '{model}' — check API key: {e}"
        ) from e

    except BadRequestError as e:
        raise LLMCallError(
            agent="llm_factory",
            reason=f"Invalid request for model '{model}': {e}"
        ) from e

    except TRANSIENT_ERRORS as e:
        raise LLMCallError(
            agent="llm_factory",
            reason=f"Model '{model}' unavailable after {LLM_RETRY_MAX_ATTEMPTS} retries: {e}"
        ) from e

    except Exception as e:
        raise LLMCallError(
            agent="llm_factory",
            reason=f"Unexpected error initializing '{model}': {e}"
        ) from e


@retry_policy
def _init_llm(
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> ChatLiteLLM:
    """
    Internal — creates ChatLiteLLM instance with retry policy applied.
    Separated from get_llm() so retry decorates only the creation call.
    """
    return ChatLiteLLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


# ── Structured Output ─────────────────────────────────────

def get_structured_llm(
    schema: Type,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> Runnable:
    """
    Return LLM with structured output bound to a Pydantic schema.
    Used by all agents with .with_structured_output().

    Args:
        schema:      Pydantic BaseModel class (e.g. IntakeAgentOutput)
        model:       Model name string
        temperature: Sampling temperature
        max_tokens:  Maximum tokens in response
        timeout:     Request timeout in seconds

    Returns:
        LangChain Runnable returning parsed Pydantic schema instance

    Raises:
        LLMCallError: If LLM initialization fails
    """
    llm = get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return llm.with_structured_output(schema)


# ── Tool‑Enabled LLM ─────────────────────────────────────

def get_tool_llm(
    tools: List[BaseTool],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> Runnable:
    """
    Return an LLM instance with the given tools bound.

    This LLM can be used in a tool‑calling loop. The returned runnable
    will return messages that may contain tool calls.

    Args:
        tools:       List of LangChain tools.
        model:       Model name string.
        temperature: Sampling temperature.
        max_tokens:  Maximum tokens in response.
        timeout:     Request timeout in seconds.

    Returns:
        A runnable (LLM) that can be invoked with messages.

    Raises:
        LLMCallError: If LLM initialization fails.
    """
    try:
        llm = get_llm(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return llm.bind_tools(tools)
    except Exception as e:
        raise LLMCallError(
            agent="llm_factory",
            reason=f"Failed to bind tools to model '{model}': {e}"
        ) from e
