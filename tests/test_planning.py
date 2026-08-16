from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from nanocode.state import Todo, format_todos
from nanocode.todo_tools import (
    KEEP_RECENT,
    TRANSCRIPT_TOTAL_CHARS,
    at_clean_checkpoint,
    make_compactor,
    write_todos,
)

pytestmark = pytest.mark.anyio


def test_write_todos_overwrites_the_whole_plan(call, message):
    todos = [
        Todo(content="find the pattern", status="completed"),
        Todo(content="write the middleware", status="in_progress"),
    ]
    result = call(write_todos, todos=todos)
    assert result.update["todos"] == todos
    assert "2 steps, 1 complete" in message(result).content


def test_format_todos_marks_status_and_progress():
    out = format_todos(
        [
            Todo(content="a", status="completed"),
            Todo(content="b", status="in_progress"),
            Todo(content="c", status="pending"),
        ]
    )
    assert "(1/3 complete)" in out
    assert "[x] a" in out and "[>] b" in out and "[ ] c" in out


# -- compaction: two conditions, not one ----------------------------------


def _bulk(n: int) -> list:
    """n message pairs, each big enough to blow past any sane budget."""
    body = "x " * 2000
    out = []
    for i in range(n):
        out.append(AIMessage(content=body, tool_calls=[{"name": "ls", "args": {}, "id": f"t{i}"}]))
        out.append(ToolMessage(content=body, tool_call_id=f"t{i}"))
    return out


def _run(messages, window=1000):
    return make_compactor(window).before_model({"messages": messages}, None)


def test_no_compaction_below_the_token_threshold():
    messages = [HumanMessage("task"), *_bulk(KEEP_RECENT + 5)]
    assert _run(messages, window=10_000_000) is None


def test_no_compaction_on_a_short_conversation():
    assert _run([HumanMessage("task"), AIMessage("done")]) is None


def test_no_compaction_mid_step_even_when_over_budget():
    """A tool call is still in flight — its result isn't on disk yet."""
    messages = [
        HumanMessage("task"),
        *_bulk(KEEP_RECENT + 5),
        AIMessage(content="", tool_calls=[{"name": "shell", "args": {}, "id": "pending"}]),
    ]
    assert not at_clean_checkpoint(messages)
    assert _run(messages) is None


def test_compaction_at_a_clean_checkpoint_keeps_task_and_recent_turns():
    messages = [HumanMessage("the original task"), *_bulk(KEEP_RECENT + 10)]
    assert at_clean_checkpoint(messages)

    update = _run(messages)
    assert update is not None

    kept = update["messages"]
    assert kept[1] is messages[0], "the original task survives compaction"
    assert isinstance(kept[2], SystemMessage)
    assert "collapsed" in kept[2].content
    assert kept[-1] is messages[-1], "the most recent turn survives"
    assert len(kept) < len(messages)


# -- compaction summarises what it drops ----------------------------------


class Summarizer(BaseChatModel):
    """Stands in for the summarising model, recording what it was asked."""

    reply: str = "decided X over Y because Z"
    seen: list = []
    fail: bool = False

    def __init__(self, reply: str = "decided X over Y because Z", fail: bool = False) -> None:
        super().__init__(reply=reply, seen=[], fail=fail)

    @property
    def _llm_type(self) -> str:
        return "summarizer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if self.fail:
            raise RuntimeError("summariser unavailable")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.reply))])


def _digest_of(update) -> str:
    return update["messages"][2].content


def test_the_dropped_span_is_summarised_into_the_digest():
    """Reasoning that never became a todo used to vanish here without trace."""
    model = Summarizer(reply="tried the regex approach first; it broke on nested quotes")
    messages = [HumanMessage("the original task"), *_bulk(KEEP_RECENT + 10)]

    update = make_compactor(1000, model=model).before_model({"messages": messages}, None)

    assert len(model.seen) == 1, "exactly one summary call per compaction"
    assert "tried the regex approach first" in _digest_of(update)
    assert "collapsed" in _digest_of(update), "the header still explains what happened"


def test_the_summariser_is_shown_the_dropped_span_as_text():
    """Text, not replayed messages — see _transcript's docstring for why."""
    model = Summarizer()
    messages = [HumanMessage("the original task"), *_bulk(KEEP_RECENT + 10)]
    make_compactor(1000, model=model).before_model({"messages": messages}, None)

    request = model.seen[0]
    assert len(request) == 2 and isinstance(request[0], SystemMessage)
    body = str(request[1].content)
    assert "ai:" in body and "tool:" in body, "the transcript should be rendered by role"
    assert "calls: ls" in body, "tool calls should survive into the transcript"


def test_huge_tool_output_is_clipped_before_summarising():
    """The dropped span is a near-full context window, mostly file contents."""
    model = Summarizer()
    messages = [
        HumanMessage("task"),
        *_bulk(KEEP_RECENT + 10),
    ]
    make_compactor(1000, model=model).before_model({"messages": messages}, None)

    body = str(model.seen[0][1].content)
    assert "more chars]" in body, "long messages should be clipped, not sent whole"
    # _bulk makes 4000-char bodies; nothing near that should survive intact.
    assert len(max(body.splitlines(), key=len)) < 2000


def test_a_failed_summary_still_compacts():
    """Compaction fires when context is nearly gone — refusing to compact
    because the summary errored would strand the run exactly when it can't
    afford it."""
    model = Summarizer(fail=True)
    messages = [HumanMessage("the original task"), *_bulk(KEEP_RECENT + 10)]

    update = make_compactor(1000, model=model).before_model({"messages": messages}, None)

    assert update is not None, "a summariser failure must not cancel compaction"
    assert "collapsed" in _digest_of(update)
    assert "Notes from the compacted conversation" not in _digest_of(update)
    assert len(update["messages"]) < len(messages)


def test_an_empty_summary_falls_back_to_the_bare_header():
    model = Summarizer(reply="   ")
    messages = [HumanMessage("task"), *_bulk(KEEP_RECENT + 10)]

    update = make_compactor(1000, model=model).before_model({"messages": messages}, None)
    assert "Notes from the compacted conversation" not in _digest_of(update)


def test_no_summary_call_when_nothing_is_compacted():
    """The call costs money — it must not fire on every turn."""
    model = Summarizer()
    compactor = make_compactor(10_000_000, model=model)  # budget never crossed
    assert compactor.before_model({"messages": [HumanMessage("task"), *_bulk(30)]}, None) is None
    assert model.seen == [], "summariser called without compacting"


async def test_the_async_path_summarises_too():
    """nanocode-web drives the graph with astream — same behaviour required."""
    model = Summarizer(reply="the async note")
    messages = [HumanMessage("task"), *_bulk(KEEP_RECENT + 10)]

    update = await make_compactor(1000, model=model).abefore_model({"messages": messages}, None)

    assert "the async note" in _digest_of(update)
    assert len(model.seen) == 1


# -- what the user said is never summarised -------------------------------


def test_user_messages_survive_compaction_word_for_word():
    """A summary of what someone asked for is strictly worse than what they said."""
    model = Summarizer()
    messages = [
        HumanMessage("the original task"),
        *_bulk(5),
        HumanMessage("actually, use the existing retry helper, don't write a new one"),
        *_bulk(KEEP_RECENT + 10),
    ]

    digest = _digest_of(make_compactor(1000, model=model).before_model({"messages": messages}, None))
    assert "use the existing retry helper, don't write a new one" in digest
    assert "in their own words" in digest


def test_the_summariser_is_told_not_to_restate_user_messages():
    """They're quoted verbatim, so summarising them too is wasted tokens."""
    from nanocode import prompts

    flat = " ".join(prompts.COMPACTION_PROMPT.lower().split())
    assert "quoted verbatim elsewhere" in flat
    assert "errors and fixes" in flat


def test_no_user_message_section_when_the_span_had_none():
    model = Summarizer()
    messages = [HumanMessage("task"), *_bulk(KEEP_RECENT + 10)]
    digest = _digest_of(make_compactor(1000, model=model).before_model({"messages": messages}, None))
    assert "in their own words" not in digest


# -- the transcript is bounded, and trimmed from the middle ---------------


def test_a_long_span_of_small_messages_is_capped():
    """The per-message clip alone doesn't bound the total — many small
    messages never trip it, so the request would grow with the session."""
    model = Summarizer()
    messages = [HumanMessage("task")]
    for i in range(1200):
        messages.append(AIMessage(content=f"step {i} " * 20, tool_calls=[{"name": "ls", "args": {}, "id": f"s{i}"}]))
        messages.append(ToolMessage(content=f"result {i}", tool_call_id=f"s{i}"))

    make_compactor(1000, model=model).before_model({"messages": messages}, None)

    sent = str(model.seen[0][1].content)
    assert len(sent) <= TRANSCRIPT_TOTAL_CHARS + 200, "the summary request is unbounded"
    assert "omitted from the middle" in sent


def test_trimming_keeps_the_previous_notes_at_the_front():
    """The erosion case: the last compaction's notes are the *first* thing in
    the next dropped span. Trim from the front and the record fades to nothing
    over a long session."""
    model = Summarizer()
    earlier_notes = SystemMessage(
        "[30 earlier messages collapsed...]\n\n## Notes from the compacted conversation\n"
        "decided to use middleware because the decorator only wires one of sync/async"
    )
    messages = [HumanMessage("task"), earlier_notes]
    for i in range(1200):
        messages.append(AIMessage(content=f"step {i} " * 20, tool_calls=[{"name": "ls", "args": {}, "id": f"s{i}"}]))
        messages.append(ToolMessage(content=f"result {i}", tool_call_id=f"s{i}"))

    make_compactor(1000, model=model).before_model({"messages": messages}, None)

    sent = str(model.seen[0][1].content)
    assert "decided to use middleware" in sent, "previous notes were trimmed away"
    # The last KEEP_RECENT messages stay in the conversation, so the dropped
    # span ends just short of them — 1189, not 1199.
    assert "step 1189" in sent, "the end of the dropped span should survive too"


def test_a_short_span_is_not_trimmed_at_all():
    model = Summarizer()
    messages = [HumanMessage("task"), *_bulk(KEEP_RECENT + 3)]
    make_compactor(1000, model=model).before_model({"messages": messages}, None)
    assert "omitted from the middle" not in str(model.seen[0][1].content)


def test_compaction_never_orphans_a_tool_result():
    """A ToolMessage must follow the AIMessage that requested it."""
    messages = [HumanMessage("task"), *_bulk(KEEP_RECENT + 10)]
    kept = _run(messages)["messages"]

    # Skip the RemoveMessage sentinel, the retained task, and the digest.
    body = kept[3:]
    assert not isinstance(body[0], ToolMessage), "cut landed on a dangling tool result"

    open_ids: set[str] = set()
    for message in body:
        if isinstance(message, AIMessage):
            open_ids |= {c["id"] for c in message.tool_calls or []}
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in open_ids, "orphaned tool result"
