#  procurement_system/utils/agent_llm_bundle.py
"""
AgentLLMBundle: paired LLM surfaces used by every agent.

An agent needs exactly two LLM interfaces:
- A tool-capable LLM for the tool-calling loop.
- A structured-output LLM for final response extraction.

This module encapsulates their construction so that BaseAgent.__init__
stays clean and both surfaces are testable in isolation.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Type

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from procurement_system.utils.model_resolver import ModelConfig, ModelChain
from procurement_system.utils.llm_router import build_llm_primary_with_fallback
from procurement_system.utils.model_resolver import resolve_from_tier
from procurement_system.constants import BuilderMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentLLMBundle:
    """
    The two LLM surfaces an agent requires.

    Attributes:
        structured_llm: Produces typed, schema-validated output.
        llm_with_tools: Supports tool-calling loop. ``None`` when the agent
            has no tools registered.
        model_config: The resolved model configuration used to build both
            surfaces (useful for logging and metrics).
    """

    structured_llm: Runnable
    llm_with_tools: Optional[Runnable]
    model_config: ModelChain


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_agent_llm_bundle(
    chain: ModelChain,
    output_schema: Type[BaseModel],
    tools: Optional[List[BaseTool]] = None,
) -> AgentLLMBundle:
    """
    Construct an AgentLLMBundle from a resolved ModelConfig.

    Args:
        output_schema: Pydantic model used for structured output binding.
        config: Resolved model configuration (model name, temperature, …).
        tools: Optional list of LangChain tools to bind. When ``None`` or
            empty, ``llm_with_tools`` on the bundle is ``None``.

    Returns:
        Ready-to-use AgentLLMBundle.

    Raises:
        LLMCallError: Propagated from ``get_structured_llm`` / ``get_tool_llm``
            on authentication or initialisation failures.
    """    

    structured_llm = build_llm_primary_with_fallback(
        chain=chain,
        mode=BuilderMode.STRUCTURED,
        output_schema=output_schema
    )

    llm_with_tools: Optional[Runnable] = None
    if tools:
        llm_with_tools = build_llm_primary_with_fallback(
            chain=chain,
            mode=BuilderMode.TOOLS,
            tools=tools
        )
        logger.debug(
            "Bound %d tool(s) to model '%s': %s",
            len(tools),
            chain,
            [t.name for t in tools],
        )

    return AgentLLMBundle(
        structured_llm=structured_llm,
        llm_with_tools=llm_with_tools,
        model_config=chain,
    )
