from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nanocode.state import Todo, format_todos
from nanocode.todo_tools import (
    KEEP_RECENT,
    at_clean_checkpoint,
    make_compactor,
    write_todos,
)


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
