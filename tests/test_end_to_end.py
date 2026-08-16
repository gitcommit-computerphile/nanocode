"""One full run through the real loop, against a scripted model.

No network, no API key — but every other part is the production path:
create_agent, the middlewares, the real disk backend, the tools, and the
session file. This is the test that would catch the wiring being wrong.
"""

from __future__ import annotations

import itertools

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from nanocode import session
from nanocode.cli import _drive
from nanocode.orchestrator import build_orchestrator
from nanocode.ui import NanocodeUI, _changed_files


class Scripted(BaseChatModel):
    """Replays a fixed sequence of assistant turns."""

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
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def handler(request):\n    return respond(request)\n", encoding="utf-8"
    )
    return tmp_path


def test_a_full_run_plans_edits_records_and_resumes(project):
    plan_v1 = [
        {"content": "find the handler", "status": "in_progress"},
        {"content": "add the rate limit", "status": "pending"},
    ]
    plan_v2 = [
        {"content": "find the handler", "status": "completed"},
        {"content": "add the rate limit", "status": "completed"},
    ]

    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=plan_v1)]),
            AIMessage(content="", tool_calls=[tool_call("grep", pattern="def handler")]),
            AIMessage(content="", tool_calls=[tool_call("read_file", path="src/app.py")]),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "edit_file",
                        path="src/app.py",
                        old_string="    return respond(request)",
                        new_string="    check_rate_limit(request)\n    return respond(request)",
                    )
                ],
            ),
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=plan_v2)]),
            AIMessage(content="Added a rate-limit check to the handler in src/app.py."),
        ]
    )

    orch = build_orchestrator(model=model, root=project)
    ui = NanocodeUI(live=False)
    state = _drive(orch, "add rate limiting to the handler", ui)

    # The edit actually landed on disk.
    assert "check_rate_limit(request)" in (project / "src" / "app.py").read_text(encoding="utf-8")

    # The plan is complete and the durable record captured the edit.
    assert state["todos"] == plan_v2
    kinds = [e["kind"] for e in state["session_log"]]
    assert "file_edit" in kinds
    assert _changed_files(state["session_log"]) == {"src/app.py": "+2 -1"}

    # The session is written, and resuming rebuilds context from it.
    session.save(project, state, "add rate limiting to the handler", "20260814T120000")
    saved = session.load(project)
    assert saved["todos"] == plan_v2
    prompt = session.resume_prompt(saved)
    assert "add rate limiting to the handler" in prompt
    assert "[x] add the rate limit" in prompt


def test_the_plan_is_recited_into_context_every_turn(project):
    """Recitation is structural — the model sees the plan without asking."""
    plan = [{"content": "do the thing", "status": "in_progress"}]
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=plan)]),
            AIMessage(content="", tool_calls=[tool_call("ls", path=".")]),
            AIMessage(content="done"),
        ]
    )

    orch = build_orchestrator(model=model, root=project)
    _drive(orch, "do the thing", NanocodeUI(live=False))

    first_prompt = "\n".join(str(m.content) for m in model.calls[0])
    assert "do the thing" not in first_prompt.split("do the thing", 1)[0] + "Current plan"

    # Every prompt after the plan exists carries it, without a read_todos call.
    for prompt in model.calls[1:]:
        text = "\n".join(str(m.content) for m in prompt)
        assert "## Current plan" in text
        assert "[>] do the thing" in text


def test_a_conversational_turn_leaves_no_plan_behind(project):
    """"hi" is not a task. The graph must not require a plan to complete a turn.

    The prompt is what decides this (a model applying a criterion, not a
    keyword list — see ORCHESTRATOR_PROMPT), so what's pinned here is the
    wiring around that decision: a turn that skips write_todos still runs,
    still renders, and leaves `todos` empty rather than half-built.
    """
    model = Scripted([AIMessage("Hi! What would you like me to work on?")])
    orch = build_orchestrator(model=model, root=project)
    state = _drive(orch, "hi", NanocodeUI(live=False))

    assert not state.get("todos"), "a greeting should not leave a plan in state"
    assert not state.get("session_log"), "a greeting should not record work events"
    assert "What would you like me to work on?" in str(state["messages"][-1].content)


def test_no_plan_means_nothing_is_recited(project):
    """Recitation stays silent with nothing to recite — no empty plan header."""
    model = Scripted([AIMessage("Hi there.")])
    orch = build_orchestrator(model=model, root=project)
    _drive(orch, "hi", NanocodeUI(live=False))

    only_prompt = "\n".join(str(m.content) for m in model.calls[0])
    assert "## Current plan" not in only_prompt
    assert "No todos yet" not in only_prompt


def test_a_short_real_task_still_gets_planned(project):
    """The other direction: don't over-correct into skipping real work.

    "run the tests" is three words and *is* a task — brevity is not the signal.
    """
    plan = [{"content": "run the test suite", "status": "in_progress"}]
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=plan)]),
            AIMessage(content="", tool_calls=[tool_call("ls", path=".")]),
            AIMessage("Tests pass."),
        ]
    )
    orch = build_orchestrator(model=model, root=project)
    state = _drive(orch, "run the tests", NanocodeUI(live=False))

    assert state["todos"] == plan
    assert "## Current plan" in "\n".join(str(m.content) for m in model.calls[1])


def test_task_vs_conversation_is_a_criterion_not_a_keyword_list(project):
    """The rule lives in the prompt, and stays a criterion rather than a list.

    If this fails because the decision moved into control flow, that's the
    regression worth catching: a hardcoded greeting list silently mishandles
    every phrasing it doesn't happen to contain.
    """
    from pathlib import Path

    from nanocode import cli, prompts

    # Collapse wrapping — the prompt is hard-wrapped, so match on words.
    flat = " ".join(prompts.ORCHESTRATOR_PROMPT.lower().split())
    assert "is this asking for concrete work on the project?" in flat
    assert "judge by intent, not by length or phrasing" in flat

    # No greeting list in the loop that runs before the model gets the message.
    source = Path(cli.__file__).read_text(encoding="utf-8").lower()
    for smell in ('"hi"', "'hi'", "hello", "greeting"):
        assert smell not in source, f"{smell} suggests keyword matching crept into cli.py"


def test_a_failing_verification_can_reopen_the_plan(project):
    """A red suite is new information, not a finished task.

    Nothing in the graph forces this — it's the prompt's job — so what's pinned
    here is that reopening a plan actually works: a completed todo can go back
    to in_progress, new steps can appear, and state follows.
    """
    done = [
        {"content": "add the feature", "status": "completed"},
        {"content": "run the tests", "status": "completed"},
    ]
    reopened = [
        {"content": "add the feature", "status": "completed"},
        {"content": "run the tests", "status": "completed"},
        {"content": "fix test_rate_limit_headers, which the change broke", "status": "in_progress"},
    ]
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=done)]),
            AIMessage(content="", tool_calls=[tool_call("write_todos", todos=reopened)]),
            AIMessage("Fixed the handler; the suite passes now."),
        ]
    )
    orch = build_orchestrator(model=model, root=project)
    state = _drive(orch, "add rate limiting", NanocodeUI(live=False))

    assert state["todos"] == reopened
    assert len(state["todos"]) == 3, "a failure should be able to add steps after 'completion'"


def test_the_prompt_demands_verification_and_honesty(project):
    """These five rules are the whole point of the change; pin the wording.

    They're prompt-level judgment, so a behavioural test would need a real
    model. What's cheap and worth guarding is that the rules haven't been
    quietly dropped from the prompt during an edit.
    """
    from nanocode import prompts

    flat = " ".join(prompts.ORCHESTRATOR_PROMPT.lower().split())

    # 1 — verification is a plan step, but only when it means something
    assert "end the plan with a verification step" in flat
    assert "does not earn a verification step" in flat
    # 2 — a failure reopens the plan
    assert "add todos, do not finish" in flat
    # 3 — the rule that matters most
    assert "fix the code, never the test" in flat
    # 4 — stop rather than thrash
    assert "stop after about three attempts" in flat
    # 5 — honest finish
    assert "never describe a task as done over a failing suite" in flat

    # The same rule has to hold in the other place edits happen.
    assert "never weaken the test" in " ".join(prompts.CODER_PROMPT.lower().split())


def test_the_sandbox_holds_during_a_real_run(project):
    """A model that tries to escape the root gets an error, not the file."""
    outside = project.parent / "outside_secret.txt"
    outside.write_text("do not read me", encoding="utf-8")

    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("read_file", path="../outside_secret.txt")]),
            AIMessage(content="I could not read outside the project."),
        ]
    )

    orch = build_orchestrator(model=model, root=project)
    state = _drive(orch, "read the file above the project", NanocodeUI(live=False))

    transcript = "\n".join(str(m.content) for m in state["messages"])
    assert "do not read me" not in transcript
    assert "escapes the project root" in transcript
