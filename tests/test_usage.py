"""Streaming, token accounting, and the context gauge.

The interesting one is `test_subagent_tokens_never_stream_to_the_user`:
LangChain propagates streaming callbacks into the sub-agent's own graph, so its
private reasoning genuinely does arrive on the parent's stream. Filtering it
out is what keeps sub-agent isolation true for the user as well as the model.
"""

from __future__ import annotations

import io
import itertools

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from rich.console import Console

from nanocode.cli import _drive
from nanocode.orchestrator import build_orchestrator
from nanocode.ui import NanocodeUI
from nanocode.usage import Usage, human, usage_of

_ids = itertools.count()


def tool_call(name: str, **args):
    return {"name": name, "args": args, "id": f"tc-{next(_ids)}", "type": "tool_call"}


class Streamer(BaseChatModel):
    """Emits word-by-word chunks, the way a real provider streams."""

    turns: list = []
    usage: dict | None = None

    def __init__(self, turns, usage=None):
        super().__init__(turns=list(turns), usage=usage)

    @property
    def _llm_type(self) -> str:
        return "streamer"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.turns.pop(0)
        if self.usage and not message.usage_metadata:
            # Sub-agents run through _generate, so usage has to be reported
            # here too or delegated work would look free.
            message = AIMessage(
                content=message.content,
                tool_calls=message.tool_calls,
                usage_metadata=self.usage,
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.turns.pop(0)
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": c["name"],
                            "args": __import__("json").dumps(c["args"]),
                            "id": c["id"],
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                        for c in message.tool_calls
                    ],
                    # Providers bill a tool-call turn like any other.
                    usage_metadata=self.usage,
                )
            )
            return
        words = str(message.content).split(" ")
        for i, word in enumerate(words):
            last = i == len(words) - 1
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=word + ("" if last else " "),
                    # Real providers report usage once, on the final chunk.
                    usage_metadata=self.usage if (last and self.usage) else None,
                )
            )


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def quiet():
    return Console(file=io.StringIO(), width=100, no_color=True)


# -- streaming ------------------------------------------------------------


def test_reply_text_is_streamed_then_committed_once(project):
    """Tokens appear live, and the finished text is printed exactly once."""
    model = Streamer([AIMessage("Here is the streamed answer.")])
    orch = build_orchestrator(model=model, root=project)

    streamed: list[str] = []
    committed: list[str] = []
    ui = NanocodeUI(live=False)
    ui.stream = lambda text: streamed.append(text)
    ui.assistant = lambda text: committed.append(text)

    _drive(orch, "ask", ui)

    assert len(streamed) > 1, "text should arrive as several deltas, not one blob"
    assert "".join(streamed) == "Here is the streamed answer."
    assert committed == ["Here is the streamed answer."], "committed exactly once"


def test_subagent_tokens_never_stream_to_the_user(project):
    """A sub-agent's reasoning reaches this stream — and must be dropped.

    Its private work is supposed to be invisible; only the final summary
    crosses back. Without the node filter it would be printed verbatim.
    """
    model = Streamer(
        [
            AIMessage(content="", tool_calls=[tool_call("delegate", task="find it", agent_type="explorer")]),
            AIMessage("SUBAGENT PRIVATE REASONING"),
            AIMessage("PARENT answer."),
        ]
    )
    orch = build_orchestrator(model=model, root=project)

    streamed: list[str] = []
    ui = NanocodeUI(live=False)
    ui.stream = lambda text: streamed.append(text)

    _drive(orch, "go", ui)

    shown = "".join(streamed)
    assert "SUBAGENT PRIVATE REASONING" not in shown, "sub-agent internals leaked to the user"
    assert "PARENT answer." in shown


def test_a_non_streaming_model_still_renders_its_reply(project):
    """Not every provider streams; the reply must not vanish when one doesn't."""

    class Plain(BaseChatModel):
        turns: list = []

        def __init__(self, turns):
            super().__init__(turns=list(turns))

        @property
        def _llm_type(self) -> str:
            return "plain"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=self.turns.pop(0))])

    model = Plain([AIMessage("no streaming here")])
    orch = build_orchestrator(model=model, root=project)
    committed: list[str] = []
    ui = NanocodeUI(live=False)
    ui.assistant = lambda text: committed.append(text)

    _drive(orch, "ask", ui)
    assert committed == ["no streaming here"]


def test_streaming_is_silent_outside_a_live_terminal(quiet):
    """`--plain > run.log` should get one clean block, not a stutter."""
    ui = NanocodeUI(quiet, live=False)
    ui.stream("partial ")
    ui.stream("text")
    assert quiet.file.getvalue() == "", "nothing should print before the reply is complete"

    assert ui.end_stream() is True
    assert "partial text" in quiet.file.getvalue()


def test_end_stream_is_a_no_op_with_nothing_buffered(quiet):
    ui = NanocodeUI(quiet, live=False)
    assert ui.end_stream() is False
    assert quiet.file.getvalue() == ""


# -- token accounting -----------------------------------------------------


def test_usage_accumulates_across_turns(project):
    model = Streamer(
        [AIMessage("first"), AIMessage("second")],
        usage={"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050},
    )
    orch = build_orchestrator(model=model, root=project)

    _drive(orch, "one", NanocodeUI(live=False))
    _drive(orch, "two", NanocodeUI(live=False))

    assert orch.usage.input_tokens == 2000
    assert orch.usage.output_tokens == 100
    assert orch.usage.calls == 2


def test_the_context_gauge_tracks_the_latest_call_not_the_total():
    """Context is how full the window is *now*, not what the session has spent."""
    usage = Usage()
    usage.add(1000, 50, is_context=True)
    usage.add(9000, 50, is_context=True)

    assert usage.input_tokens == 10_000, "spend accumulates"
    assert usage.context_tokens == 9000, "context is the most recent call"
    assert usage.context_percent(18_000) == pytest.approx(50.0)


def test_subagent_usage_is_billed_but_does_not_move_the_context_gauge():
    """Sub-agents cost real money in their own context — count the spend,
    but their size says nothing about how full the main conversation is."""
    usage = Usage()
    usage.add(1000, 50, is_context=True)
    usage.add(50_000, 400)  # a sub-agent chewing through files

    assert usage.total_tokens == 51_450, "sub-agent tokens are still billed"
    assert usage.context_tokens == 1000, "the main context gauge should not move"


def test_delegated_work_is_counted(project):
    """Sub-agent messages are discarded, so nothing else would ever see them."""
    model = Streamer(
        [
            AIMessage(content="", tool_calls=[tool_call("delegate", task="look", agent_type="explorer")]),
            AIMessage("sub-agent summary"),
            AIMessage("done"),
        ],
        usage={"input_tokens": 500, "output_tokens": 25, "total_tokens": 525},
    )
    orch = build_orchestrator(model=model, root=project)
    _drive(orch, "go", NanocodeUI(live=False))

    # Two orchestrator calls (the delegate turn, then the reply) plus the
    # sub-agent's own — which nothing else in the system would ever see.
    assert orch.usage.calls == 3
    assert orch.usage.input_tokens == 1500
    # The sub-agent ran in its own context, so the gauge tracks only the
    # orchestrator's last call.
    assert orch.usage.context_tokens == 500


def test_usage_survives_a_model_switch(project, monkeypatch):
    """Switching models mid-session shouldn't reset what you've spent."""
    from nanocode import commands
    from nanocode.commands import CommandContext

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    orch = build_orchestrator(model=Streamer([]), root=project)
    orch.usage.add(1234, 56, is_context=True)

    console = Console(file=io.StringIO(), no_color=True)
    ctx = CommandContext(console=console, ui=NanocodeUI(live=False), root=project, orch=orch)
    commands.dispatch("/model openai:gpt-5.4-mini", ctx)

    assert ctx.orch.usage is orch.usage
    assert ctx.orch.usage.input_tokens == 1234


def test_usage_of_sums_a_message_list():
    messages = [
        AIMessage("a", usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}),
        HumanMessage("b"),
        AIMessage("c", usage_metadata={"input_tokens": 20, "output_tokens": 2, "total_tokens": 22}),
    ]
    assert usage_of(messages) == (30, 3)


def test_a_provider_reporting_nothing_is_not_counted_as_a_call():
    usage = Usage()
    usage.record(AIMessage("no usage metadata here"))
    assert usage.calls == 0 and usage.total_tokens == 0


# -- display --------------------------------------------------------------


@pytest.mark.parametrize(
    "count,expected", [(0, "0"), (940, "940"), (12_400, "12.4k"), (1_200_000, "1.20M")]
)
def test_human_readable_token_counts(count, expected):
    assert human(count) == expected


def test_the_status_line_shows_spend_and_context(quiet):
    usage = Usage(input_tokens=12_400, output_tokens=1800, calls=3, context_tokens=100_000)
    NanocodeUI(quiet, live=False).status(usage, 200_000)

    out = quiet.file.getvalue()
    assert "12.4k in" in out and "1.8k out" in out
    assert "50% of 200.0k context" in out


def test_the_status_line_warns_when_compaction_is_near(quiet):
    usage = Usage(input_tokens=190_000, output_tokens=100, calls=9, context_tokens=190_000)
    NanocodeUI(quiet, live=False).status(usage, 200_000)
    assert "compaction imminent" in quiet.file.getvalue()


def test_no_status_line_before_the_first_call(quiet):
    NanocodeUI(quiet, live=False).status(Usage(), 200_000)
    assert quiet.file.getvalue() == ""