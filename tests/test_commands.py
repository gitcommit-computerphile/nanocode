"""Slash commands: /clear, /model, /help.

These act on the session rather than going through the model, so they're driven
here the same way a user drives them — by feeding console input and reading
what comes back.
"""

from __future__ import annotations

import io

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from rich.console import Console

from nanocode import commands, models, session
from nanocode.cli import _drive, _session_loop
from nanocode.commands import CommandContext
from nanocode.models import ModelListError
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


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def quiet():
    return Console(file=io.StringIO(), width=100, no_color=True)


@pytest.fixture
def ctx(project, quiet):
    orch = build_orchestrator(model=Scripted([]), root=project)
    return CommandContext(console=quiet, ui=NanocodeUI(quiet, live=False), root=project, orch=orch)


def replies(monkeypatch, *answers):
    """Queue console answers for the command's prompts."""
    queue = iter(answers)
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: next(queue))


# -- what counts as a command ---------------------------------------------


@pytest.mark.parametrize("text", ["/clear", "/model", "/model openai:x", "/help", "/CLEAR"])
def test_known_commands_are_recognised(text):
    assert commands.is_command(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "add a /health endpoint",
        "/health should return 200",  # an ask that happens to start with a slash
        "clear",
        "",
    ],
)
def test_anything_else_goes_to_the_model(text):
    assert commands.is_command(text) is False


# -- /clear ---------------------------------------------------------------


def test_clear_rotates_the_thread(ctx):
    before = ctx.orch.thread
    commands.dispatch("/clear", ctx)

    assert ctx.orch.thread != before
    assert ctx.cleared is True


def test_clear_keeps_constraints(ctx):
    """Standing rules are project state — 'clear the context' shouldn't drop them."""
    ctx.constraints = ["do not modify the auth module"]
    commands.dispatch("/clear", ctx)

    assert ctx.constraints == ["do not modify the auth module"]
    assert "1 standing constraint kept" in ctx.console.file.getvalue()


def test_clear_actually_drops_the_conversation(project):
    """The real test: after /clear the model must not see the earlier ask."""
    model = Scripted([AIMessage("first"), AIMessage("second")])
    orch = build_orchestrator(model=model, root=project)
    ui = NanocodeUI(live=False)
    console = Console(file=io.StringIO(), no_color=True)

    _drive(orch, "remember the number 41", ui)
    commands.dispatch(
        "/clear", CommandContext(console=console, ui=ui, root=project, orch=orch)
    )
    _drive(orch, "what number?", ui)

    second = "\n".join(str(m.content) for m in model.calls[-1])
    assert "41" not in second, "the cleared conversation came back"
    assert "what number?" in second


def test_clear_leaves_the_saved_record_alone(project, quiet, monkeypatch):
    """Clearing context is not deleting history."""
    model = Scripted([AIMessage("done"), AIMessage("done again")])
    orch = build_orchestrator(model=model, root=project)
    asks = iter(["/clear", "a second ask", None])
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: next(asks))

    _session_loop(orch, NanocodeUI(quiet, live=False), quiet, project, "first ask", "first ask", once=False)

    saved = [session.load(project, p.stem)["task"] for p in session.list_sessions(project)]
    assert sorted(saved) == ["a second ask", "first ask"]


def test_clear_re_seeds_constraints_into_the_new_thread(project, quiet, monkeypatch):
    model = Scripted([AIMessage("one"), AIMessage("two")])
    orch = build_orchestrator(model=model, root=project)
    asks = iter(["/clear", "a second ask", None])
    monkeypatch.setattr("nanocode.cli._prompt", lambda _c: next(asks))

    _session_loop(
        orch,
        NanocodeUI(quiet, live=False),
        quiet,
        project,
        "first ask",
        "first ask",
        once=False,
        constraints=["do not modify the auth module"],
    )

    after_clear = "\n".join(str(m.content) for m in model.calls[-1])
    assert "do not modify the auth module" in after_clear


# -- /model ---------------------------------------------------------------


def test_model_switch_keeps_the_conversation(project, monkeypatch):
    """Switching model mid-task must not cost the user their context."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    model = Scripted([AIMessage("noted")])
    orch = build_orchestrator(model=model, root=project)
    console = Console(file=io.StringIO(), no_color=True)
    ctx = CommandContext(console=console, ui=NanocodeUI(live=False), root=project, orch=orch)

    _drive(orch, "remember the number 41", NanocodeUI(live=False))
    commands.dispatch("/model openai:gpt-5.4-mini", ctx)

    assert ctx.orch is not orch, "the graph should have been rebuilt"
    assert ctx.orch.spec == "openai:gpt-5.4-mini"
    # Same checkpointer and thread — that's what carries the conversation over.
    assert ctx.orch.checkpointer is orch.checkpointer
    assert ctx.orch.thread == orch.thread
    history = ctx.orch.agent.get_state(ctx.orch.config).values["messages"]
    assert any("41" in str(m.content) for m in history)


def test_model_switch_reports_a_bad_spec_without_dying(ctx, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = ctx.orch
    commands.dispatch("/model openai:whatever", ctx)

    assert ctx.orch is before, "a failed switch must leave the session intact"
    assert "OPENAI_API_KEY" in ctx.console.file.getvalue()


def test_model_lists_what_the_key_can_reach(ctx, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        models, "list_models", lambda p, k=None: ["openai:gpt-5.4-mini", "openai:gpt-5.6-sol"]
    )
    replies(monkeypatch, "2")  # only one provider has a key, so it goes straight to the list

    commands.dispatch("/model", ctx)

    out = ctx.console.file.getvalue()
    assert "gpt-5.6-sol" in out
    assert ctx.orch.spec == "openai:gpt-5.6-sol"


def test_model_asks_for_a_missing_key_and_holds_it_for_the_process(ctx, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(models, "list_models", lambda p, k=None: ["anthropic:claude-opus-5"])
    replies(monkeypatch, "1", "sk-ant-typed-in", "1")  # provider, key, model

    commands.dispatch("/model", ctx)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-typed-in"
    assert ctx.orch.spec == "anthropic:claude-opus-5"
    out = ctx.console.file.getvalue()
    assert "never writes keys to disk" in out
    assert "sk-ant-typed-in" not in out, "a typed key must not be echoed back"


def test_a_cancelled_key_prompt_changes_nothing(ctx, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = ctx.orch
    replies(monkeypatch, "1", "")  # pick anthropic, then decline to give a key

    commands.dispatch("/model", ctx)

    assert ctx.orch is before
    assert "cancelled" in ctx.console.file.getvalue()


def test_an_unreachable_provider_is_reported_not_faked(ctx, monkeypatch):
    """A stale hardcoded list would be worse than no list — it looks authoritative."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def unreachable(provider, api_key=None):
        raise ModelListError("could not reach openai: timed out")

    monkeypatch.setattr(models, "list_models", unreachable)
    commands.dispatch("/model", ctx)

    out = ctx.console.file.getvalue()
    assert "could not reach openai" in out
    assert "/model openai:gpt-5.4-mini" in out, "should still offer the direct route"


def test_switching_to_the_current_model_is_a_no_op(ctx, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    ctx.orch.spec = "openai:gpt-5.4-mini"
    before = ctx.orch

    commands.dispatch("/model openai:gpt-5.4-mini", ctx)

    assert ctx.orch is before
    assert "already on" in ctx.console.file.getvalue()


# -- /help ----------------------------------------------------------------


def test_help_lists_every_command(ctx):
    commands.dispatch("/help", ctx)
    out = ctx.console.file.getvalue()
    for name, _ in commands.HELP:
        assert name in out
