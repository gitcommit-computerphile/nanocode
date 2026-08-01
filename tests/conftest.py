from __future__ import annotations

from typing import Any, Callable

import pytest
from langchain_core.tools import BaseTool
from langgraph.types import Command


@pytest.fixture
def call() -> Callable[..., Any]:
    """Invoke a tool the way the agent runtime does.

    Tools taking an `InjectedToolCallId` must be handed a full ToolCall, not a
    bare args dict — the runtime fills the id in, so tests have to as well.
    """

    def _call(tool: BaseTool, **args: Any) -> Any:
        return tool.invoke(
            {"name": tool.name, "args": args, "id": "call-1", "type": "tool_call"}
        )

    return _call


@pytest.fixture
def message() -> Callable[[Command], Any]:
    """The single ToolMessage a Command-returning tool produced."""

    def _message(result: Command) -> Any:
        return result.update["messages"][0]

    return _message
