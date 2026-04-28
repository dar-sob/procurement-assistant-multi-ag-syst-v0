# procurement_system/utils/tool_utils.py

"""
Utilities for executing tool calls in a ReAct-style loop.
"""

import logging
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from procurement_system.settings import get_forced_summary_llm_tool_loop

logger = logging.getLogger(__name__)


def execute_tool_loop(
    llm_with_tools: Runnable,
    messages: List[BaseMessage],
    tools_dict: Dict[str, BaseTool],
    max_iterations: int = 5,
    agent_name: Optional[str] = None,
) -> List[BaseMessage]:
    """
    Execute a ReAct-style tool-calling loop until the model stops requesting tools.

    Iterates up to max_iterations rounds. On each round the model response is
    appended to the message history, all requested tools are executed, and their
    results are appended as ToolMessages.

    If the model stops requesting tools before the ceiling is reached the loop
    exits normally and the message history ends with a clean AIMessage.

    If the iteration ceiling is reached a forced-summary request is sent as an
    additional LLM call. This guarantees the message history always ends with a
    meaningful AIMessage — never a bare ToolMessage — regardless of how the loop
    exits.

    Args:
        llm_with_tools: LangChain Runnable with tools bound via .bind_tools().
        messages:        Current message list (system + user prompt).
                         Extended in-place and returned.
        tools_dict:      Mapping of tool name → BaseTool instance.
        max_iterations:  Maximum tool-calling rounds before a forced summary
                         is requested. Defaults to 5.

    Returns:
        Updated message list. The final message is always an AIMessage.
    """
    prefix = f"{agent_name} " if agent_name else ""

    for iteration in range(1, max_iterations + 1):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        # Model produced a final answer — no further tool calls requested.
        if not response.tool_calls:            
            logger.debug("[%s] Tool loop finished after %d iteration(s).", prefix, iteration)
            return messages

        logger.debug(
            "[%s] Iteration %d/%d — executing %d tool call(s): %s",
            prefix,
            iteration,
            max_iterations,
            len(response.tool_calls),
            [tc["name"] for tc in response.tool_calls],
        )

        messages = _execute_tool_calls(messages, response.tool_calls, tools_dict, prefix=prefix)

    # Ceiling reached — force a summary so the history ends with a clean AIMessage.
    logger.warning(
        "[%s] Tool loop reached max iterations (%d). Requesting forced summary.",
        prefix,
        max_iterations,
    )
    return _request_forced_summary(llm_with_tools, messages)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _execute_tool_calls(
    messages: List[BaseMessage],
    tool_calls: List[dict],
    tools_dict: Dict[str, BaseTool],
    prefix: str = "",
) -> List[BaseMessage]:
    """
    Invoke all tool calls from a single model response and append results.

    Unknown tool names are surfaced as error strings in ToolMessage content
    so the model can reason about the failure rather than crashing the pipeline.

    Args:
        messages:   Current message history.
        tool_calls: Tool call descriptors from the AIMessage.
        tools_dict: Mapping of tool name → BaseTool instance.

    Returns:
        Updated message history with one ToolMessage appended per tool call.
    """
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool = tools_dict.get(tool_name)

        if tool is None:
            logger.warning("[%s] Tool '%s' not found in tools_dict.", prefix, tool_name)
            result = f"Error: tool '{tool_name}' is not registered."
        else:
            try:
                result = tool.invoke(tool_call["args"])
            except Exception as exc:
                logger.exception("[%s] Tool '%s' raised an exception.", prefix, tool_name)
                result = f"Error executing tool '{tool_name}': {exc}"

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    return messages


def _request_forced_summary(
    llm_with_tools: Runnable,
    messages: List[BaseMessage],
) -> List[BaseMessage]:
    """
    Request a forced summary after the tool loop iteration ceiling is hit.

    Appends a HumanMessage instructing the model to summarise all findings
    without invoking further tools, then invokes the LLM once more.
    Guarantees the message history ends with a clean AIMessage.

    Args:
        llm_with_tools: LangChain Runnable used for the summary call.
        messages:        Message history at the point of ceiling breach.

    Returns:
        Updated message history with the summary AIMessage appended.
    """
    messages.append(HumanMessage(content=get_forced_summary_llm_tool_loop()))
    summary: AIMessage = llm_with_tools.invoke(messages)
    messages.append(summary)
    return messages
