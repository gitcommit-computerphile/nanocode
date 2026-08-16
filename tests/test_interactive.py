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


def test_a_whole_session_writes_exactly_one_file(project, quiet, monkeypatch):
    """One run is one file — asks within it update it, they don't multiply it."""
    model = Scripted([AIMessage("one"), AIMessage("two"), AIMessage("three")])
    orch = build_orchestrator(model=model, root=project)
    asks = iter(["second", "third", None])
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: next(asks))

    _session_loop(orch, NanocodeUI(quiet, live=False), quiet, project, "first", "first", once=False)

    assert len(session.list_sessions(project)) == 1


def test_a_later_session_does_not_clobber_an_earlier_one(project, quiet, monkeypatch):
    """The bug this fixes: starting work used to destroy the previous record."""
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: None)

    first = build_orchestrator(model=Scripted([AIMessage("done")]), root=project)
    _session_loop(first, NanocodeUI(quiet, live=False), quiet, project, "first run", "first run", once=False)

    # A distinct id, the way a genuinely later run would have.
    monkeypatch.setattr(session, "new_session_id", lambda *_: "20991231T235959")
    second = build_orchestrator(model=Scripted([AIMessage("done")]), root=project)
    _session_loop(second, NanocodeUI(quiet, live=False), quiet, project, "second run", "second run", once=False)

    tasks = [session.load(project, path.stem)["task"] for path in session.list_sessions(project)]
    assert sorted(tasks) == ["first run", "second run"]


# -- constraints cross the session boundary -------------------------------


def test_constraints_are_seeded_into_the_session(project, quiet, monkeypatch):
    model = Scripted([AIMessage("understood")])
    orch = build_orchestrator(model=model, root=project)
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: None)

    _session_loop(
        orch,
        NanocodeUI(quiet, live=False),
        quiet,
        project,
        "do a thing",
        "do a thing",
        once=False,
        constraints=["do not modify the auth module"],
    )

    text = "\n".join(str(m.content) for m in model.calls[0])
    assert "do not modify the auth module" in text


# -- a follow-up question is not a second helping of the last task --------


def test_a_follow_up_question_does_not_inherit_the_previous_summary(project, quiet, monkeypatch):
    """Todos survive between asks, so a stale plan used to make "done?" print
    a full work report — elapsed time, log paths, "0/4 todos complete" — for a
    one-line answer that changed nothing."""
    plan = [{"content": "build the thing", "status": "in_progress"}]
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=plan)]),
            AIMessage(content="", tool_calls=[tool_call("write_file", path="a.py", content="A\n")]),
            AIMessage("started on it"),
            AIMessage("Not yet — still working on it."),  # the follow-up question
        ]
    )
    orch = build_orchestrator(model=model, root=project)
    asks = iter(["done?", None])
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: next(asks))

    _session_loop(orch, NanocodeUI(quiet, live=False), quiet, project, "build it", "build it", once=False)

    out = quiet.file.getvalue()
    assert out.count("todos complete") == 1, "the follow-up printed its own work summary"
    assert out.count("full logs") == 1, "only the ask that did work should report"


def test_a_real_second_task_still_gets_its_own_summary(project, quiet, monkeypatch):
    """The other direction — don't silence genuine work."""
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_file", path="a.py", content="A\n")]),
            AIMessage("first done"),
            AIMessage(content="", tool_calls=[tool_call("write_file", path="b.py", content="B\n")]),
            AIMessage("second done"),
        ]
    )
    orch = build_orchestrator(model=model, root=project)
    asks = iter(["write b.py", None])
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: next(asks))

    _session_loop(orch, NanocodeUI(quiet, live=False), quiet, project, "write a.py", "write a.py", once=False)

    out = quiet.file.getvalue()
    assert "a.py" in out and "b.py" in out, "both asks did real work and both should report it"


def test_a_follow_up_question_does_not_redisplay_the_old_plan(project):
    """The other half of the same bug: the summary was suppressed but the plan
    panel still rendered a finished checklist under an unrelated answer."""
    plan = [{"content": "build the thing", "status": "completed"}]
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=plan)]),
            AIMessage("built it"),
            AIMessage("Status: all done."),
        ]
    )
    orch = build_orchestrator(model=model, root=project)

    shown: list[list] = []
    ui = NanocodeUI(live=False)
    ui.set_todos = lambda todos: shown.append(todos)

    _drive(orch, "build it", ui)
    assert shown, "a real task should render its plan"

    shown.clear()
    _drive(orch, "tell me status", ui, plan_before=plan)
    assert shown == [], "an untouched plan should not reappear under a question"


def test_the_prompt_forbids_ticking_work_that_was_not_done():
    """Observed: "Add a small verification script or test" ticked complete
    while the summary showed one file changed and no test existed."""
    from nanocode import prompts

    flat = " ".join(prompts.ORCHESTRATOR_PROMPT.lower().split())
    assert "only mark a todo `completed` if you actually did the thing" in flat
    assert "is not complete because you added a print statement" in flat


def test_the_prompt_requires_verification_that_can_fail():
    """Observed: "ran python lex_sorter.py and it printed the expected output"
    — a command that succeeds identically whether or not the code works."""
    from nanocode import prompts

    flat = " ".join(prompts.ORCHESTRATOR_PROMPT.lower().split())
    assert "verification must be capable of failing" in flat
    assert 'if the answer is "the same thing", you have not verified it' in flat


def test_the_prompt_tells_it_to_build_rather_than_hunt():
    """An empty directory plus "write me an X" is unambiguous.

    Observed failure: in an empty folder it planned "locate the existing
    implementation", searched six times, then stopped with four pending todos
    and asked what to do.
    """
    from nanocode import prompts

    flat = " ".join(prompts.ORCHESTRATOR_PROMPT.lower().split())
    assert "is **from scratch**" in flat
    assert "never ask the user a question the directory already answers" in flat
    assert "discovering your premise was mistaken is a reason to **replan**" in flat
    assert "pending todos and a question is the worst outcome" in flat


# -- picking work up without being asked ----------------------------------


@pytest.fixture
def captured(monkeypatch):
    """Run the CLI down to the point the session loop would start."""
    seen: dict = {}

    def fake_loop(orch, ui, console, root, opening, task_label, once, constraints=None):
        seen.update(opening=opening, task_label=task_label, constraints=constraints or [])
        return 0

    class FakeOrch:
        """Enough of an Orchestrator for the startup path — which now reads
        `.git` to put the branch in the header."""

        git = None

    monkeypatch.setattr("nanocode.cli.build_orchestrator", lambda **kw: FakeOrch())
    monkeypatch.setattr("nanocode.cli._session_loop", fake_loop)
    return seen


def invoke(project, *args):
    from typer.testing import CliRunner

    from nanocode.cli import app

    return CliRunner().invoke(app, ["-C", str(project), *args])


def _unfinished(project):
    session.save(
        project,
        {
            "todos": [
                {"content": "find the handler", "status": "completed"},
                {"content": "update the tests", "status": "pending"},
            ],
            "session_log": [],
        },
        "add rate limiting",
        "20260814T120000",
    )


def test_unfinished_work_is_picked_up_without_a_flag(project, captured):
    """The flag that rescues you shouldn't be one you need before you need it."""
    _unfinished(project)
    invoke(project)

    assert "update the tests" in captured["opening"]
    assert captured["task_label"] == "add rate limiting"


def test_fresh_ignores_unfinished_work_and_constraints(project, captured):
    _unfinished(project)
    session.save_constraints(project, ["do not modify the auth module"])
    invoke(project, "--fresh", "something else")

    assert captured["opening"] == "something else"
    assert captured["constraints"] == []


def test_a_finished_run_does_not_haunt_the_next_one(project, captured):
    session.save(
        project,
        {"todos": [{"content": "all done", "status": "completed"}], "session_log": []},
        "yesterday's task",
        "20260814T120000",
    )
    invoke(project, "a new thing")

    assert captured["opening"] == "a new thing"


def test_a_new_ask_is_appended_to_the_briefing(project, captured):
    _unfinished(project)
    invoke(project, "actually, skip the tests")

    assert "update the tests" in captured["opening"]
    assert captured["opening"].endswith("actually, skip the tests")


def test_constraints_are_loaded_from_disk_at_startup(project, captured):
    session.save_constraints(project, ["do not modify the auth module"])
    invoke(project, "do a thing")

    assert captured["constraints"] == ["do not modify the auth module"]


def test_once_does_not_pick_up_unfinished_work(project, captured):
    """Scripted runs get what they asked for and nothing else."""
    _unfinished(project)
    invoke(project, "--once", "run the tests")

    assert captured["opening"] == "run the tests"


def test_resume_still_forces_a_pick_up_of_a_finished_run(project, captured):
    session.save(
        project,
        {"todos": [{"content": "all done", "status": "completed"}], "session_log": []},
        "yesterday's task",
        "20260814T120000",
    )
    invoke(project, "--resume")

    assert "yesterday's task" in captured["opening"]


def test_resume_with_nothing_to_resume_is_an_error(project, captured):
    result = invoke(project, "--resume")
    assert result.exit_code == 1
    assert "nothing to resume" in result.output


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
