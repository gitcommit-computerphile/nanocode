"""Planning: the write_todos tool, plus the two middlewares that keep the
orchestrator's context bounded.

The tutorial recites the plan by *instructing* the model to re-read it, which
is easy to forget on a long run. Nanocode makes recitation structural instead:
the current plan is re-injected before every model call by middleware, not left
to the model's memory.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from . import prompts
from .state import NanocodeState, Todo, format_constraints, format_todos

DEFAULT_CONTEXT_WINDOW = 200_000
COMPACT_AT = 0.9
KEEP_RECENT = 20
# Per-message cap when rendering the dropped span for the summariser. Tool
# output and file contents dominate the bulk and are already on disk; the notes
# are for what happened, not for the payloads.
TRANSCRIPT_CHARS = 1500
# And a cap on the whole transcript, because the per-message limit alone does
# not bound the total: a long run of many small messages never trips it, so the
# summary request would grow with the session. ~60k chars is ~15k tokens, which
# keeps compaction costing the same on turn 50 and turn 5,000.
TRANSCRIPT_TOTAL_CHARS = 60_000
# Of that cap, this much is reserved for the *start* of the span. The previous
# compaction's notes sit at the front, so trimming from there would discard
# them and the notes would erode a little more with every cycle.
TRANSCRIPT_HEAD_SHARE = 0.3
# User messages are kept whole and never summarised — see _user_messages.
USER_MESSAGE_CHARS = 2000


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


def _with_recitation(request: ModelRequest, git: Any = None) -> ModelRequest:
    """Build the request recite_context actually sends — a plan-and-constraints
    reminder appended, or the original request untouched if there's nothing to
    recite yet. Shared by the sync and async paths below so the two can't drift.
    """
    state = request.state or {}
    todos = state.get("todos") or []
    constraints = state.get("constraints") or []
    repository = git.as_prompt() if git is not None else ""
    if not todos and not constraints and not repository:
        return request

    blocks: list[str] = []
    if repository:
        # Cached text, refreshed once per ask — no subprocess runs here.
        blocks.append(repository)
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
    return request.override(messages=[*request.messages, reminder])


class _ReciteContext(AgentMiddleware):
    """Re-inject the plan, the standing constraints, and the repository state
    before every model call.

    This is the structural half of recitation: neither has to survive in the
    message list, and the model never has to remember to re-read them. It also
    means compaction cannot delete either one — they are rebuilt from state on
    the very next call, however much history was just thrown away.

    A plain `@wrap_model_call`-decorated function only gets you one of
    `wrap_model_call` / `awrap_model_call` — LangChain's decorator wires
    whichever slot matches the function's own sync/async-ness, never both. The
    CLI drives the graph synchronously (`.stream()`); a host driving it with
    `.astream()`/`.ainvoke()` (nanocode-web, for instance) needs the async slot
    too, so this is a class with both implemented explicitly, sharing the same
    request-building logic via `_with_recitation`.
    """

    name = "recite_context"

    def __init__(self, git: Any = None) -> None:
        super().__init__()
        self.git = git

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(_with_recitation(request, self.git))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        return await handler(_with_recitation(request, self.git))


def make_reciter(git: Any = None) -> AgentMiddleware:
    return _ReciteContext(git)


# The plain instance, for callers with no repository to report.
recite_context = _ReciteContext()


def make_compactor(
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    *,
    model: BaseChatModel | None = None,
    on_compact: Callable[[bool], None] | None = None,
) -> AgentMiddleware:
    """Collapse old turns once the message list actually gets large.

    Two conditions have to hold, not one. Crossing the token threshold is
    necessary but not sufficient — compaction also waits for a clean
    checkpoint, meaning every tool call already produced its result and landed
    in `todos` / `session_log`. Collapsing mid-step would drop the reasoning
    behind an action nothing durable has recorded yet.

    Given a `model`, the span being dropped is summarised into notes first, and
    those notes go into the digest. Without one, the digest is the bare header
    — the pre-existing behaviour, kept as the fallback because a failed summary
    must never cost the user their run.

    Why summarise at all: `todos` and `session_log` record *what was decided*
    and *what was done*, but not *why*. Reasoning that never became a todo —
    an approach tried and rejected, a constraint the user implied rather than
    stated — used to vanish here with no trace. One extra call at the moment of
    compaction is cheap; it happens rarely by construction.

    Sync and async graph execution each need their own hook here for the same
    reason `recite_context` is a class rather than a decorated function — see
    its docstring. Everything except the call itself is shared, so the two
    paths cannot drift.
    """
    budget = int(context_window * COMPACT_AT)

    def _plan(state: NanocodeState) -> tuple[AnyMessage, list[AnyMessage], list[AnyMessage]] | None:
        """Decide whether to compact, and what gets dropped versus kept."""
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
        return messages[0], messages[1:cut], messages[cut:]

    def _summary_request(dropped: list[AnyMessage]) -> list[AnyMessage]:
        return [
            SystemMessage(prompts.COMPACTION_PROMPT),
            HumanMessage(f"{_transcript(dropped)}\n\n{prompts.COMPACTION_REQUEST}"),
        ]

    def _notes(response: Any) -> str | None:
        # `.text` is a property on current message classes; older ones exposed
        # it as a method, and calling it now emits a deprecation warning.
        text = getattr(response, "text", None)
        if callable(text):
            text = None
        if not text:
            text = getattr(response, "content", "")
        return stripped if (stripped := str(text).strip()) else None

    class _Compactor(AgentMiddleware):
        name = "compact_if_needed"

        def before_model(self, state: NanocodeState, runtime: Any) -> dict[str, Any] | None:
            if (planned := _plan(state)) is None:
                return None
            first, dropped, kept = planned
            notes = None
            if model is not None:
                # An extra model call the user did not ask for, taking seconds
                # in the middle of their task — it says so rather than looking
                # like a stall.
                if on_compact:
                    on_compact(True)
                try:
                    notes = _notes(model.invoke(_summary_request(dropped)))
                except Exception:  # noqa: BLE001 — see _digest; never lose the run
                    notes = None
                finally:
                    if on_compact:
                        on_compact(False)
            return _compacted(first, dropped, kept, notes)

        async def abefore_model(self, state: NanocodeState, runtime: Any) -> dict[str, Any] | None:
            if (planned := _plan(state)) is None:
                return None
            first, dropped, kept = planned
            notes = None
            if model is not None:
                if on_compact:
                    on_compact(True)
                try:
                    notes = _notes(await model.ainvoke(_summary_request(dropped)))
                except Exception:  # noqa: BLE001
                    notes = None
                finally:
                    if on_compact:
                        on_compact(False)
            return _compacted(first, dropped, kept, notes)

    return _Compactor()


def _compacted(
    first: AnyMessage,
    dropped: list[AnyMessage],
    kept: list[AnyMessage],
    notes: str | None,
) -> dict[str, Any]:
    """The rewritten message list: the original task, a digest, recent turns."""
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first,
            _digest(len(dropped), notes, _user_messages(dropped)),
            *kept,
        ]
    }


def _digest(count: int, notes: str | None, asked: list[str]) -> SystemMessage:
    """The single message standing in for everything that was dropped.

    Three parts, in descending order of how badly they are missed: what the
    user actually said, quoted; the model's notes on the rest; and the header
    explaining what happened to everything else.

    The header alone is the fallback when no model was given, or when the
    summary call failed. Compaction fires precisely when context is nearly
    exhausted, so refusing to compact because a summary errored would strand
    the run at the exact moment it most needs the room.
    """
    blocks = [
        f"[{count} earlier messages collapsed to save context. "
        "The plan in `todos`, the record in `session_log`, and the standing "
        "constraints are all current as of this point — the constraints are "
        "re-injected every turn and were not lost with the messages above. "
        "Every file edit is already on disk; re-read files if you need their "
        "contents.]"
    ]
    if asked:
        quoted = "\n".join(f"{i}. {text}" for i, text in enumerate(asked, 1))
        blocks.append(
            "## What the user asked, in their own words\n"
            "Quoted exactly, not summarised. Treat these as the standing record "
            f"of intent for this session.\n\n{quoted}"
        )
    if notes:
        blocks.append(f"## Notes from the compacted conversation\n{notes}")
    return SystemMessage("\n\n".join(blocks))


def _transcript(
    messages: list[AnyMessage],
    per_message: int = TRANSCRIPT_CHARS,
    total: int = TRANSCRIPT_TOTAL_CHARS,
) -> str:
    """Render messages as plain text for the summariser.

    Text rather than a message list on purpose. Replaying raw messages invites
    the model to continue the conversation instead of describing it, and a span
    containing tool calls can fail provider-side validation once it is lifted
    out of its original context.

    Two limits apply. Each message is clipped, because the span is by
    definition near a full context window and most of that bulk is file
    contents and command output — already on disk, and not what the notes are
    for. Then the whole transcript is clipped, because the per-message limit
    bounds each line but not their sum: a long run of small messages never
    trips it. `_trim_middle` does the second one.
    """
    lines: list[str] = []
    for message in messages:
        role = type(message).__name__.removesuffix("Message").lower()
        body = _clip(str(message.content).strip(), per_message)
        if calls := getattr(message, "tool_calls", None):
            summary = ", ".join(f"{c['name']}({_clip(str(c.get('args', {})), 120)})" for c in calls)
            body = f"{body} calls: {summary}" if body else f"calls: {summary}"
        if body:
            lines.append(f"{role}: {body}")
    return _trim_middle(lines, total)


def _trim_middle(lines: list[str], total: int) -> str:
    """Fit `lines` into `total` characters by dropping from the middle.

    Both ends are load-bearing, for different reasons. The start holds the
    previous compaction's notes — trim there and each cycle loses the one
    before it, so the record erodes to nothing over a long session. The end
    holds the work in progress, which is what the next turn continues.

    The middle is where repetition concentrates: the same file read twice, a
    search refined three times. It is the right thing to lose.
    """
    if sum(len(line) + 1 for line in lines) <= total:
        return "\n".join(lines)

    head_budget = int(total * TRANSCRIPT_HEAD_SHARE)
    head, used = [], 0
    for line in lines:
        if used + len(line) + 1 > head_budget:
            break
        head.append(line)
        used += len(line) + 1

    tail_budget = total - used
    tail, used = [], 0
    for line in reversed(lines[len(head) :]):
        if used + len(line) + 1 > tail_budget:
            break
        tail.append(line)
        used += len(line) + 1
    tail.reverse()

    omitted = len(lines) - len(head) - len(tail)
    if omitted <= 0:
        return "\n".join(lines)
    return "\n".join([*head, f"[… {omitted} messages omitted from the middle …]", *tail])


def _user_messages(messages: list[AnyMessage]) -> list[str]:
    """The user's own words, pulled out to be carried through verbatim.

    They are a fraction of a session's tokens and the most expensive thing in
    it to lose: a summary of what someone asked for is strictly worse than what
    they actually said. So these skip summarisation entirely and are quoted
    into the digest as-is.
    """
    return [
        text
        for message in messages
        if isinstance(message, HumanMessage) and (text := _clip(str(message.content).strip(), USER_MESSAGE_CHARS))
    ]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… [{len(text) - limit} more chars]"


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
