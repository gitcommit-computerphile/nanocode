"""Planning: the write_todos tool, plus the two middlewares that keep the
orchestrator's context bounded.

The tutorial recites the plan by *instructing* the model to re-read it, which
is easy to forget on a long run. Nanocode makes recitation structural instead:
the current plan is re-injected before every model call by middleware, not left
to the model's memory.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    before_model,
    wrap_model_call,
)
from langchain_core.messages import AnyMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from . import prompts
from .state import NanocodeState, Todo, format_constraints, format_todos

DEFAULT_CONTEXT_WINDOW = 200_000
COMPACT_AT = 0.9
KEEP_RECENT = 20


@tool("write_todos", description=prompts.WRITE_TODOS_DESCRIPTION)
def write_todos(
    todos: list[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    done = sum(1 for t in todos if t.get("status") == "completed")
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    f"plan updated — {len(todos)} steps, {done} complete",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@wrap_model_call
def recite_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Re-inject the plan and the standing constraints before every model call.

    This is the structural half of recitation: neither has to survive in the
    message list, and the model never has to remember to re-read them. It also
    means compaction cannot delete either one — they are rebuilt from state on
    the very next call, however much history was just thrown away.
    """
    state = request.state or {}
    todos = state.get("todos") or []
    constraints = state.get("constraints") or []
    if not todos and not constraints:
        return handler(request)

    blocks: list[str] = []
    if constraints:
        blocks.append(
            f"{format_constraints(constraints)}\n\n"
            "These hold for the whole project. Honour them without being reminded, "
            "and call write_constraints if the user adds, changes, or lifts one."
        )
    if todos:
        blocks.append(
            f"{format_todos(todos)}\n\n"
            "Keep exactly one step in_progress. Call write_todos whenever a status changes."
        )
    reminder = SystemMessage("\n\n".join(blocks))
    return handler(request.override(messages=[*request.messages, reminder]))


def make_compactor(context_window: int = DEFAULT_CONTEXT_WINDOW) -> AgentMiddleware:
    """Collapse old turns once the message list actually gets large.

    Two conditions have to hold, not one. Crossing the token threshold is
    necessary but not sufficient — compaction also waits for a clean
    checkpoint, meaning every tool call already produced its result and landed
    in `todos` / `session_log`. Collapsing mid-step would drop the reasoning
    behind an action nothing durable has recorded yet.
    """
    budget = int(context_window * COMPACT_AT)

    @before_model(name="compact_if_needed")
    def compact_if_needed(state: NanocodeState, runtime: Any) -> dict[str, Any] | None:
        messages: list[AnyMessage] = state.get("messages") or []
        if len(messages) <= KEEP_RECENT + 2:
            return None
        if count_tokens_approximately(messages) < budget:
            return None
        if not at_clean_checkpoint(messages):
            # A tool call is still in flight — let it land first.
            return None

        cut = _safe_cut(messages, len(messages) - KEEP_RECENT)
        if cut <= 1:
            return None

        kept = messages[cut:]
        digest = SystemMessage(
            f"[{cut - 1} earlier messages collapsed to save context. "
            "The plan in `todos`, the record in `session_log`, and the standing "
            "constraints are all current as of this point — the constraints are "
            "re-injected every turn and were not lost with the messages above. "
            "Every file edit is already on disk; re-read files if you need their "
            "contents.]"
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                messages[0],
                digest,
                *kept,
            ]
        }

    return compact_if_needed


def at_clean_checkpoint(messages: list[AnyMessage]) -> bool:
    """True when no issued tool call is still waiting for its result."""
    if not messages:
        return False
    last = messages[-1]
    return not getattr(last, "tool_calls", None)


def _safe_cut(messages: list[AnyMessage], index: int) -> int:
    """Move a cut point forward until it does not orphan a ToolMessage.

    A ToolMessage must be preceded by the AIMessage that requested it, so a cut
    landing on one would leave an unanswerable dangling result.
    """
    i = max(index, 1)
    while i < len(messages) and isinstance(messages[i], ToolMessage):
        i += 1
    return i
