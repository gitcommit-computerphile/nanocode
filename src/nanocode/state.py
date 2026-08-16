"""State schema.

Deliberately smaller than the tutorial's DeepAgentState. There is no ``files``
field — the disk already persists that. What remains is three durable things,
which together are enough to resume without ever holding a full transcript:

- ``todos``        — what to do, and how far along it is (intent)
- ``session_log``  — what actually happened (evidence)
- ``constraints``  — rules that outlive the task (standing orders)

The first two are scoped to one run. Constraints are not: they belong to the
project, they are never trimmed, and they are re-injected every turn — so
unlike anything said in conversation, compaction cannot delete them.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.agents.middleware import AgentState

TodoStatus = Literal["pending", "in_progress", "completed"]
EventKind = Literal["file_edit", "delegate", "shell", "decision", "search", "constraint"]


class Todo(TypedDict):
    """One step of the plan."""

    content: str
    status: TodoStatus


class Event(TypedDict):
    """One thing that happened, durable across a resume.

    `detail` is for a human to read. Anything a *program* needs goes in the
    optional fields below, because the alternative — formatting numbers into
    `detail` and regex-ing them back out — means two files must agree on
    exact wording with nothing checking that they do. A reworded string then
    breaks the summary silently, and a test that hand-writes the same string
    still passes.
    """

    kind: EventKind
    detail: str
    ts: str
    # Set on file_edit events. Read with .get(): older sessions predate them.
    path: NotRequired[str]
    added: NotRequired[int]
    removed: NotRequired[int]
    created: NotRequired[bool]


class NanocodeState(AgentState):
    todos: NotRequired[list[Todo]]
    session_log: NotRequired[Annotated[list[Event], operator.add]]
    constraints: NotRequired[list[str]]


def event(kind: EventKind, detail: str, **fields: Any) -> Event:
    """Build an Event stamped with the current UTC time.

    Extra keyword fields carry the structured half — see `file_edit_event`,
    which is the only caller that needs them today.
    """
    return Event(
        kind=kind, detail=detail, ts=datetime.now(timezone.utc).isoformat(), **fields
    )


def file_edit_event(
    path: str, added: int, removed: int, created: bool | None = None
) -> Event:
    """A file change, recorded as numbers *and* as text.

    `created=None` means a targeted edit (a diff); True or False means a
    whole-file write that respectively made or replaced the file.

    The text is built here, where the numbers are known, and is read by nobody
    but a human. Nothing parses it back.
    """
    if created is None:
        detail = f"{path}: +{added} -{removed}"
    else:
        detail = f"{path}: {'created' if created else 'overwrote'} ({added} lines)"
    return event(
        "file_edit",
        detail,
        path=path,
        added=added,
        removed=removed,
        created=bool(created),
    )


def format_todos(todos: list[Todo]) -> str:
    """Render the plan the way it gets recited back into context."""
    if not todos:
        return "No todos yet."
    marks = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
    lines = [f"{marks.get(t['status'], '[ ]')} {t['content']}" for t in todos]
    done = sum(1 for t in todos if t["status"] == "completed")
    return f"## Current plan ({done}/{len(todos)} complete)\n" + "\n".join(lines)


def format_constraints(constraints: list[str]) -> str:
    """Render the standing rules the way they get recited back into context."""
    if not constraints:
        return ""
    lines = "\n".join(f"- {c}" for c in constraints)
    return f"## Standing constraints (always apply)\n{lines}"
