"""Standing constraints — the state that outlives the conversation.

A rule the user states once ("never touch the auth module") used to live only
in the message list, where it died twice: when the process exited, and again
when compaction trimmed the turn it was said in. These tests pin that it now
survives both.
"""

from __future__ import annotations

import itertools

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from nanocode import session
from nanocode.cli import _drive
from nanocode.constraints import make_constraints_tool
from nanocode.orchestrator import build_orchestrator
from nanocode.state import format_constraints
from nanocode.todo_tools import recite_context
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


# -- the tool -------------------------------------------------------------


def test_write_constraints_updates_state_and_disk(tmp_path, call):
    tool = make_constraints_tool(tmp_path)
    result = call(tool, constraints=["do not modify the auth module"])

    assert result.update["constraints"] == ["do not modify the auth module"]
    # Written immediately, not at the end of the run — a crash must not lose it.
    assert session.load_constraints(tmp_path) == ["do not modify the auth module"]


def test_write_constraints_records_an_event(tmp_path, call):
    tool = make_constraints_tool(tmp_path)
    result = call(tool, constraints=["a", "b"])

    logged = result.update["session_log"][0]
    assert logged["kind"] == "constraint"
    assert "2 standing constraints" in logged["detail"]


def test_write_constraints_overwrites_rather_than_appends(tmp_path, call):
    tool = make_constraints_tool(tmp_path)
    call(tool, constraints=["old rule", "kept rule"])
    result = call(tool, constraints=["kept rule"])

    assert result.update["constraints"] == ["kept rule"]
    assert session.load_constraints(tmp_path) == ["kept rule"]


def test_blank_entries_are_dropped(tmp_path, call):
    tool = make_constraints_tool(tmp_path)
    result = call(tool, constraints=["  real rule  ", "", "   "])
    assert result.update["constraints"] == ["real rule"]


def test_an_unwritable_path_is_an_error_not_a_crash(tmp_path, call, monkeypatch):
    """Same rule as the filesystem tools: never raise out of a tool."""
    tool = make_constraints_tool(tmp_path)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(session, "save_constraints", boom)
    result = call(tool, constraints=["something"])

    assert result.update["messages"][0].status == "error"
    assert "disk full" in result.update["messages"][0].content


# -- recitation -----------------------------------------------------------


class _Request:
    """Minimal stand-in for a ModelRequest."""

    def __init__(self, state):
        self.state = state
        self.messages = [HumanMessage("do the thing")]

    def override(self, messages):
        self.messages = messages
        return self


def test_constraints_are_recited_alongside_the_plan():
    request = _Request(
        {
            "todos": [{"content": "step one", "status": "in_progress"}],
            "constraints": ["do not modify the auth module"],
        }
    )
    seen: list = []
    recite_context.wrap_model_call(request, lambda r: seen.append(r.messages) or "ok")

    injected = "\n".join(str(m.content) for m in seen[0] if isinstance(m, SystemMessage))
    assert "do not modify the auth module" in injected
    assert "step one" in injected


def test_constraints_are_recited_even_without_a_plan():
    request = _Request({"constraints": ["targets Python 3.11"]})
    seen: list = []
    recite_context.wrap_model_call(request, lambda r: seen.append(r.messages) or "ok")

    assert "targets Python 3.11" in "\n".join(str(m.content) for m in seen[0])


def test_nothing_is_injected_when_there_is_nothing_to_recite():
    request = _Request({})
    seen: list = []
    recite_context.wrap_model_call(request, lambda r: seen.append(r.messages) or "ok")

    assert seen[0] == request.messages


def test_format_constraints_is_empty_for_an_empty_set():
    assert format_constraints([]) == ""


# -- through the real graph ----------------------------------------------


def test_a_constraint_stated_in_conversation_lands_on_disk(project):
    model = Scripted(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("write_constraints", constraints=["do not modify the auth module"])
                ],
            ),
            AIMessage("Noted."),
        ]
    )
    orch = build_orchestrator(model=model, root=project)
    state = _drive(orch, "never touch the auth module", NanocodeUI(live=False))

    assert state["constraints"] == ["do not modify the auth module"]
    assert session.load_constraints(project) == ["do not modify the auth module"]


def test_every_later_turn_sees_the_constraint(project):
    """Recitation is structural — it can't be forgotten mid-run."""
    model = Scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("write_constraints", constraints=["no auth edits"])]),
            AIMessage(content="", tool_calls=[tool_call("ls", path=".")]),
            AIMessage("done"),
        ]
    )
    orch = build_orchestrator(model=model, root=project)
    _drive(orch, "never touch auth", NanocodeUI(live=False))

    for prompt in model.calls[1:]:
        text = "\n".join(str(m.content) for m in prompt)
        assert "no auth edits" in text, "a turn went out without the constraint"


def test_a_seeded_constraint_reaches_the_model_without_being_restated(project):
    """A rule from a previous session applies without the user repeating it."""
    session.save_constraints(project, ["do not modify the auth module"])

    model = Scripted([AIMessage("understood")])
    orch = build_orchestrator(model=model, root=project)
    _drive(
        orch,
        "add a health endpoint",
        NanocodeUI(live=False),
        seed={"constraints": session.load_constraints(project)},
    )

    text = "\n".join(str(m.content) for m in model.calls[0])
    assert "do not modify the auth module" in text
