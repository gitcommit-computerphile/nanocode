"""The session loop: run it, ask for something, keep asking.

The point of interactive mode is that ask #2 knows what ask #1 did. These
tests pin that continuity, and pin `--once` being the same loop minus the
prompt-again step rather than a second code path.
"""

from __future__ import annotations

import itertools

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from rich.console import Console

from nanocode import session
from nanocode.cli import _drive, _prompt, _session_loop
from nanocode.orchestrator import build_orchestrator
from nanocode.ui import NanocodeUI


class Scripted(BaseChatModel):
    turns: list[AIMessage] = []
    calls: list[list] = []

    def __init__(self, turns: list[AIMessage]) -> None:
        super().__init__(turns=list(turns), calls=[])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.turns:
            raise AssertionError("scripted model ran out of turns")
        return ChatResult(generations=[ChatGeneration(message=self.turns.pop(0))])


_ids = itertools.count()


def tool_call(name: str, **args):
    return {"name": name, "args": args, "id": f"tc-{next(_ids)}", "type": "tool_call"}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def quiet():
    import io

    return Console(file=io.StringIO(), width=100, no_color=True)


# -- continuity between asks ----------------------------------------------


def test_a_follow_up_ask_sees_the_previous_one(project):
    """'now make it 3' has to know what 'it' is."""
    model = Scripted([AIMessage("set it to 2"), AIMessage("set it to 3")])
    orch = build_orchestrator(model=model, root=project)
    ui = NanocodeUI(live=False)

    _drive(orch, "set VALUE to 2", ui)
    _drive(orch, "now make it 3", ui)

    second_turn = model.calls[-1]
    text = "\n".join(str(m.content) for m in second_turn)
    assert "set VALUE to 2" in text, "the first ask is missing from the second turn"
    assert "set it to 2" in text, "the first answer is missing from the second turn"
    assert "now make it 3" in text


def test_the_second_ask_only_renders_its_own_output(project):
    """History was already shown when it happened — don't replay it."""
    model = Scripted([AIMessage("first answer"), AIMessage("second answer")])
    orch = build_orchestrator(model=model, root=project)

    lines: list[str] = []
    ui = NanocodeUI(live=False)
    ui.assistant = lambda text: lines.append(text)

    _drive(orch, "first ask", ui)
    _drive(orch, "second ask", ui)

    assert lines == ["first answer", "second answer"]


def test_state_accumulates_across_asks(project):
    plan_one = [{"content": "step one", "status": "completed"}]
    plan_two = [{"content": "step two", "status": "completed"}]
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=plan_one)]),
            AIMessage(content="", tool_calls=[tool_call("write_file", path="a.py", content="A\n")]),
            AIMessage("did the first thing"),
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=plan_two)]),
            AIMessage(content="", tool_calls=[tool_call("write_file", path="b.py", content="B\n")]),
            AIMessage("did the second thing"),
        ]
    )
    orch = build_orchestrator(model=model, root=project)
    ui = NanocodeUI(live=False)

    first = _drive(orch, "do the first thing", ui)
    assert len(first["session_log"]) == 1

    second = _drive(orch, "do the second thing", ui)
    # session_log is append-only across the whole session…
    assert len(second["session_log"]) == 2
    # …while todos are rewritten per ask.
    assert second["todos"] == plan_two
    assert (project / "a.py").exists() and (project / "b.py").exists()


# -- the loop itself ------------------------------------------------------


def test_once_runs_a_single_ask_and_returns(project, quiet, monkeypatch):
    model = Scripted([AIMessage("done")])
    orch = build_orchestrator(model=model, root=project)

    def explode(_console):
        raise AssertionError("--once must not prompt for another task")

    monkeypatch.setattr("nanocode.cli._prompt", explode)
    code = _session_loop(orch, NanocodeUI(quiet, live=False), quiet, project, "do it", "do it", once=True)

    assert code == 0
    assert len(model.calls) == 1


def test_interactive_keeps_asking_until_the_user_stops(project, quiet, monkeypatch):
    model = Scripted([AIMessage("one"), AIMessage("two"), AIMessage("three")])
    orch = build_orchestrator(model=model, root=project)

    asks = iter(["second ask", "third ask", None])  # None == user typed 'exit'
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: next(asks))

    code = _session_loop(
        orch, NanocodeUI(quiet, live=False), quiet, project, "first ask", "first ask", once=False
    )

    assert code == 0
    assert len(model.calls) == 3, "expected three asks in one session"


def test_a_failed_ask_ends_that_ask_not_the_session(project, quiet, monkeypatch):
    """One bad turn shouldn't drop you out of the session."""

    class Flaky(Scripted):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                raise RuntimeError("provider hiccup")
            return ChatResult(generations=[ChatGeneration(message=AIMessage("recovered"))])

    orch = build_orchestrator(model=Flaky([]), root=project)
    asks = iter(["second ask", None])
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: next(asks))

    code = _session_loop(
        orch, NanocodeUI(quiet, live=False), quiet, project, "first ask", "first ask", once=False
    )

    assert code == 0, "the session should end cleanly despite the failed first ask"
    assert "provider hiccup" in quiet.file.getvalue()


def test_the_session_file_is_written_after_every_ask(project, quiet, monkeypatch):
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_file", path="x.py", content="X\n")]),
            AIMessage("first done"),
            AIMessage("second done"),
        ]
    )
    orch = build_orchestrator(model=model, root=project)
    asks = iter(["second ask", None])
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: next(asks))

    _session_loop(
        orch, NanocodeUI(quiet, live=False), quiet, project, "first ask", "first ask", once=False
    )

    saved = session.load(project)
    assert saved is not None
    assert saved["task"] == "second ask", "the label should track the latest ask"
    assert len(saved["session_log"]) == 1


# -- the prompt -----------------------------------------------------------


@pytest.mark.parametrize("word", ["exit", "quit", ":q", "EXIT"])
def test_exit_words_end_the_session(monkeypatch, quiet, word):
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: word)
    assert _prompt(quiet) is None


def test_eof_ends_the_session(monkeypatch, quiet):
    def eof(self, *a, **k):
        raise EOFError

    monkeypatch.setattr(Console, "input", eof)
    assert _prompt(quiet) is None


def test_blank_input_re_prompts_rather_than_exiting(monkeypatch, quiet):
    replies = iter(["", "   ", "actually do this"])
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(replies))
    assert _prompt(quiet) == "actually do this"
