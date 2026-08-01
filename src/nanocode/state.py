"""State schema.

Deliberately smaller than the tutorial's DeepAgentState. There is no ``files``
field — the disk already persists that. What remains is the plan and a running
log of what happened, which is enough to resume without ever holding a full
transcript.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain.agents.middleware import AgentState

TodoStatus = Literal["pending", "in_progress", "completed"]
EventKind = Literal["file_edit", "delegate", "shell", "decision", "search"]


class Todo(TypedDict):
    """One step of the plan."""

    content: str
    status: TodoStatus


class Event(TypedDict):
    """One thing that happened, durable across a resume."""

    kind: EventKind
    detail: str
    ts: str


class NanocodeState(AgentState):
    todos: NotRequired[list[Todo]]
    session_log: NotRequired[Annotated[list[Event], operator.add]]


def event(kind: EventKind, detail: str) -> Event:
    """Build an Event stamped with the current UTC time."""
    return Event(kind=kind, detail=detail, ts=datetime.now(timezone.utc).isoformat())


def format_todos(todos: list[Todo]) -> str:
    """Render the plan the way it gets recited back into context."""
    if not todos:
        return "No todos yet."
    marks = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
    lines = [f"{marks.get(t['status'], '[ ]')} {t['content']}" for t in todos]
    done = sum(1 for t in todos if t["status"] == "completed")
    return f"## Current plan ({done}/{len(todos)} complete)\n" + "\n".join(lines)
