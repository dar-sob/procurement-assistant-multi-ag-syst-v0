# procurement_system/agents/base_agent.py
"""
Production-grade BaseAgent for the procurement system.

Architecture
------------
Subclasses implement exactly two methods:
    _build_prompt()   — constructs the user message from SharedState
    _format_result()  — maps structured LLM output back to SharedState

Everything else — model resolution, tool loop, token optimisation,
and structured output extraction — is handled here centrally.

Token optimisation
------------------
After the tool-calling loop completes, only the final AI message
(not the full tool-call history) is forwarded to the structured LLM.
This avoids re-sending large tool results and reduces token consumption
significantly on long tool chains.

If the tool loop hits the iteration ceiling, tool_utils.execute_tool_loop
requests a forced summary before returning. This guarantees that a
meaningful AIMessage is always available for the structured LLM —
even in degraded cases.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from procurement_system.agents.mixins import TracingMixin
from procurement_system.constants import DEFAULT_MAX_TOOL_ITERATIONS, MIN_FINAL_MESSAGE_LENGTH
from procurement_system.exceptions import StructuredOutputError, ToolLoopError
from procurement_system.state import SharedState
from procurement_system.utils.agent_llm_bundle import AgentLLMBundle, build_agent_llm_bundle
from procurement_system.utils.model_resolver import ModelChain, resolve_from_params, resolve_from_tier
from procurement_system.utils.tool_utils import execute_tool_loop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for a single agent instance."""

    name: str
    output_schema: Type[BaseModel]
    system_prompt: str
    user_template: str

    # Model selection — model_tier takes precedence over explicit model name.
    model_tier: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None

    # Maximum iterations for the ReAct tool-calling loop.
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS

    tools: List[BaseTool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model_tier and self.model:
            logger.warning(
                "[%s] Both 'model_tier' and 'model' were provided. "
                "'model_tier' takes precedence.",
                self.name,
            )


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(TracingMixin, ABC):
    """
    Abstract base class for all procurement agents.

    Subclasses must implement:
        _build_prompt(state)             → str
        _format_result(state, output)    → Dict[str, Any]
    """

    def __init__(
        self,
        config: AgentConfig,
        repositories: Optional[Dict[str, Any]] = None,
        services: Optional[Dict[str, Any]] = None,
        tracer: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self.config = config
        self._repositories = repositories or {}
        self._services = services or {}
        self._tracer = tracer
        self._metrics = metrics

        # Resolve model chain from tier or explicit parameters.
        self._model_chain: ModelChain = (
            resolve_from_tier(config.model_tier)
            if config.model_tier
            else resolve_from_params(
                config.model,
                config.temperature,
                config.max_tokens,
                config.timeout,
            )
        )

        # Build the LLM bundle: plain LLM with tools + structured-output LLM.
        self._llm: AgentLLMBundle = build_agent_llm_bundle(
            output_schema=config.output_schema,
            chain=self._model_chain,
            tools=config.tools or None,
        )

        self._tools_dict: Dict[str, BaseTool] = {t.name: t for t in config.tools}

        logger.info(
            "[%s] Initialized — tier=%s, tools=%d, max_iterations=%d",
            config.name,
            config.model_tier or "explicit",
            len(config.tools),
            config.max_tool_iterations,
        )

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_prompt(self, state: SharedState) -> str:
        """
        Construct the user-turn prompt from the current graph state.

        Args:
            state: The current SharedState.

        Returns:
            Formatted prompt string ready to wrap in a HumanMessage.
        """

    @abstractmethod
    def _format_result(
        self, state: SharedState, parsed_output: Any
    ) -> Dict[str, Any]:
        """
        Map the validated LLM output to SharedState update fields.

        Args:
            state:         The current SharedState.
            parsed_output: Validated Pydantic instance of output_schema.

        Returns:
            Dict of SharedState fields to update via state | updates.
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state: SharedState) -> SharedState:
        """
        Main entry point called by the LangGraph node.

        Args:
            state: The current SharedState.

        Returns:
            Updated SharedState with agent results merged in.

        Raises:
            Any unhandled exception is re-raised after logging.
        """
        try:
            return self._execute(state)
        except Exception:
            logger.error("[%s] Agent execution failed.", self.config.name, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Core execution pipeline
    # ------------------------------------------------------------------

    def _execute(self, state: SharedState) -> SharedState:
        """
        Orchestrate the full agent pipeline:
            build messages → tool loop → extract final message
            → structured LLM → format result → merge state.

        Args:
            state: The current SharedState.

        Returns:
            Updated SharedState.
        """
        messages = self._build_messages(state)
        self._log_estimated_tokens(messages)

        if self.has_tools:
            messages = execute_tool_loop(
                llm_with_tools=self._llm.llm_with_tools,
                messages=messages,
                tools_dict=self._tools_dict,
                max_iterations=self.config.max_tool_iterations,
                agent_name=self.name
            )
            final_ai_message = self._extract_final_ai_message(messages)
            structured_messages = self._build_structured_messages(final_ai_message)
        else:
            structured_messages = messages

        parsed_output = self._invoke_structured_llm(structured_messages)
        updates = self._format_result(state, parsed_output)

        return updates

    def _build_messages(self, state: SharedState) -> List[BaseMessage]:
        """
        Assemble the initial message list for the first LLM call.

        Args:
            state: The current SharedState.

        Returns:
            List containing SystemMessage and HumanMessage.
        """
        return [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=self._build_prompt(state)),
        ]

    def _build_structured_messages(
        self, final_ai_message: AIMessage
    ) -> List[BaseMessage]:
        """
        Build the minimal message list passed to the structured-output LLM.

        Only the system prompt and the final AI message are included.
        This avoids re-sending the full tool-call history and reduces
        token consumption significantly.

        Args:
            final_ai_message: The last meaningful AIMessage from the tool loop.

        Returns:
            Two-element list: [SystemMessage, AIMessage].
        """
        return [
            SystemMessage(content=self.config.system_prompt),
            final_ai_message,
        ]

    # ------------------------------------------------------------------
    # Final message extraction
    # ------------------------------------------------------------------

    def _extract_final_ai_message(
        self, messages: List[BaseMessage]
    ) -> AIMessage:
        """
        Extract the last meaningful AIMessage from the completed tool loop.

        Search order (most preferred → fallback):
            1. Last AIMessage with no tool calls and content ≥ MIN_FINAL_MESSAGE_LENGTH.
            2. Last AIMessage with no tool calls and any non-empty content.
            3. Last AIMessage regardless of content (logged as error).

        Note: execute_tool_loop guarantees that a forced-summary AIMessage is
        always appended when the iteration ceiling is reached, so a candidate
        of type (1) or (2) will almost always be found.

        Args:
            messages: Complete message history from the tool loop.

        Returns:
            The best available AIMessage.

        Raises:
            ToolLoopError: If no AIMessage exists in the history at all.
        """
        candidates = [
            m for m in messages
            if isinstance(m, AIMessage) and not m.tool_calls
        ]

        if not candidates:
            raise ToolLoopError(
                agent=self.name,
                reason="No AIMessage without tool_calls found in message history.",
            )

        # Preferred: last message with substantive content.
        for message in reversed(candidates):
            if len(str(message.content or "").strip()) >= MIN_FINAL_MESSAGE_LENGTH:
                return message

        # Fallback: last message with any non-empty content.
        for message in reversed(candidates):
            if str(message.content or "").strip():
                logger.warning(
                    "[%s] Final AI message is shorter than %d characters. "
                    "Structured output quality may be reduced.",
                    self.name,
                    MIN_FINAL_MESSAGE_LENGTH,
                )
                return message

        # Last resort: return whatever is available and log clearly.
        logger.error(
            "[%s] All candidate AIMessages have empty content. "
            "Structured output will likely fail validation.",
            self.name,
        )
        return candidates[-1]

    # ------------------------------------------------------------------
    # Structured LLM invocation
    # ------------------------------------------------------------------

    def _invoke_structured_llm(self, messages: List[BaseMessage]) -> Any:
        """
        Invoke the structured-output LLM and return a validated Pydantic instance.

        Args:
            messages: Message list to send to the structured LLM.

        Returns:
            Validated instance of config.output_schema.

        Raises:
            StructuredOutputError: Wraps any LLM or validation error with
                                   agent name context for upstream handling.
        """
        try:
            return self._llm.structured_llm.invoke(messages) # def invoke from router_llm
        except Exception as exc:
            raise StructuredOutputError(
                agent=self.config.name,
                schema=self.config.output_schema.__name__,
                reason=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_estimated_tokens(self, messages: List[BaseMessage]) -> None:
        """
        Log an estimated pre-tool-loop input token count when available.

        This is a diagnostic estimate only — it reflects the initial prompt
        before any tool calls are made. Logged at DEBUG level to avoid noise
        in production logs.

        Args:
            messages: Initial message list (system + user, before tool loop).
        """
        if not hasattr(self._llm.structured_llm, "get_num_tokens_from_messages"):
            return

        try:
            estimated = self._llm.structured_llm.get_num_tokens_from_messages(messages)
            logger.debug(
                "[%s] Estimated input tokens (pre-tool-loop): %d",
                self.name,
                estimated,
            )
        except Exception as exc:
            logger.debug("[%s] Token estimation unavailable: %s", self.name, exc)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Agent name as defined in AgentConfig."""
        return self.config.name

    @property
    def has_tools(self) -> bool:
        """True if this agent has at least one registered tool."""
        return bool(self.config.tools)

    @property
    def tools(self) -> List[BaseTool]:
        """List of tools registered with this agent."""
        return self.config.tools
