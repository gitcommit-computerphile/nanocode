"""Sub-agents.

A sub-agent is the same `create_agent` loop, given a narrower tool set and a
message history containing *only* its task string. Its reasoning, dead ends,
and intermediate tool calls never reach the parent's context — only a final
summary does.

`delegate` wipes the message history. It does not wipe the sub-agent's system
prompt, its access to the shared disk, or the underlying model. There is no
memory between calls either: delegating to `explorer` twice produces two
sub-agents that have never met, even though they share a name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from . import prompts
from .fs_tools import FileSystem, make_fs_tools
from .search_tool import make_search_tool
from .shell_tool import make_shell_tool
from .state import NanocodeState, event
from .usage import usage_of

SUBAGENT_RECURSION_LIMIT = 60


@dataclass(frozen=True)
class SubAgent:
    """One fixed sub-agent type. The registry is deliberately closed — the
    orchestrator picks from these rather than inventing roles per task."""

    name: str
    prompt: str
    build_tools: Callable[[FileSystem, Path], list[BaseTool]]


def _explorer_tools(fs: FileSystem, root: Path) -> list[BaseTool]:
    tools = make_fs_tools(fs, writable=False)
    search = make_search_tool(root)
    return [*tools, search] if search else tools


def _coder_tools(fs: FileSystem, root: Path) -> list[BaseTool]:
    # No shell: keeps the blast radius of an isolated sub-agent small, and
    # keeps "should we run the tests now" a decision the orchestrator makes.
    return make_fs_tools(fs, writable=True)


def _test_runner_tools(fs: FileSystem, root: Path) -> list[BaseTool]:
    return [make_shell_tool(root)]


REGISTRY: dict[str, SubAgent] = {
    "explorer": SubAgent("explorer", prompts.EXPLORER_PROMPT, _explorer_tools),
    "coder": SubAgent("coder", prompts.CODER_PROMPT, _coder_tools),
    "test-runner": SubAgent("test-runner", prompts.TEST_RUNNER_PROMPT, _test_runner_tools),
}


def make_delegate_tool(
    model: BaseChatModel,
    fs: FileSystem,
    root: str | Path,
    on_trace: Callable[[str, str], None] | None = None,
    on_usage: Callable[[int, int], None] | None = None,
    middleware: list | None = None,
) -> BaseTool:
    """Build the delegate tool.

    `on_trace(agent_type, line)` is an optional UI hook; it is called as the
    sub-agent works and has no effect on what the orchestrator sees.

    `on_usage(input_tokens, output_tokens)` reports what the sub-agent's run
    cost. It has to be reported explicitly here because a sub-agent's messages
    are discarded rather than returned to the parent — so nothing downstream
    would ever see them, and delegated work would look free.
    """
    project_root = Path(root).resolve()

    @tool("delegate", description=prompts.DELEGATE_DESCRIPTION)
    def delegate(
        task: str,
        agent_type: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        spec = REGISTRY.get(agent_type)
        if spec is None:
            known = ", ".join(REGISTRY)
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"unknown agent_type {agent_type!r}. Available: {known}.",
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                    ]
                }
            )

        if on_trace:
            on_trace(spec.name, task)

        agent = create_agent(
            model,
            tools=spec.build_tools(fs, project_root),
            system_prompt=spec.prompt,
            state_schema=NanocodeState,
            middleware=list(middleware or []),
            name=spec.name,
        )

        try:
            # Fresh context: one task string in, nothing else.
            result = agent.invoke(
                {"messages": [HumanMessage(task)]},
                config={"recursion_limit": SUBAGENT_RECURSION_LIMIT},
            )
        except Exception as exc:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"{spec.name} failed: {exc}",
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                    ]
                }
            )

        messages = result.get("messages", [])
        if on_usage:
            on_usage(*usage_of(messages))

        summary = _final_text(messages)
        if on_trace:
            on_trace(spec.name, f"returned: {summary[:120]}")

        return Command(
            update={
                "session_log": [event("delegate", f"{spec.name}: {summary[:200]}")],
                "messages": [ToolMessage(summary, tool_call_id=tool_call_id)],
            }
        )

    return delegate


def _final_text(messages: list) -> str:
    """Pull the sub-agent's last assistant message — its answer, and the only
    thing that crosses back into the parent's context."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
    return "(the sub-agent finished without producing a summary)"
