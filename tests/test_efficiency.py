"""multi_edit, the sub-agent model split, prompt caching, structured events.

Four changes that share a purpose: make a session cost less without changing
what it can do.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from nanocode.fs_tools import VirtualFileSystem, make_fs_tools
from nanocode.orchestrator import build_orchestrator
from nanocode.state import file_edit_event
from nanocode.ui import _changed_files


class Quiet(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "quiet"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])


@pytest.fixture
def fs():
    return VirtualFileSystem(
        {"app.py": "one = 1\ntwo = 2\nthree = 3\nfour = 4\nfive = 5\n"}
    )


@pytest.fixture
def multi(fs):
    return next(t for t in make_fs_tools(fs) if t.name == "multi_edit")


def _edit(old: str, new: str) -> dict:
    return {"old_string": old, "new_string": new}


# -- multi_edit -----------------------------------------------------------


def test_several_edits_land_in_one_call(fs, multi, call):
    result = call(
        multi,
        path="app.py",
        edits=[_edit("one = 1", "one = 100"), _edit("three = 3", "three = 300"),
               _edit("five = 5", "five = 500")],
    )

    content = fs.read("app.py")
    assert "one = 100" in content and "three = 300" in content and "five = 500" in content
    assert "two = 2" in content, "untouched lines stay untouched"
    assert "applied 3 edits" in result.update["messages"][0].content


def test_one_bad_edit_leaves_the_file_completely_untouched(fs, multi, call, message):
    """All-or-nothing: a half-applied set is a state nobody asked for."""
    before = fs.read("app.py")
    result = call(
        multi,
        path="app.py",
        edits=[_edit("one = 1", "one = 100"), _edit("nonexistent", "x"),
               _edit("five = 5", "five = 500")],
    )

    assert message(result).status == "error"
    assert fs.read("app.py") == before, "the first edit was applied despite a later failure"
    assert "edit 2 of 3" in message(result).content, "says which edit failed"


def test_an_ambiguous_edit_is_refused(fs, multi, call, message):
    fs.write("dup.py", "x = 1\nx = 1\n")
    result = call(multi, path="dup.py", edits=[_edit("x = 1", "x = 2")])

    assert message(result).status == "error"
    assert "found 2 times" in message(result).content
    assert fs.read("dup.py") == "x = 1\nx = 1\n"


def test_edits_apply_in_order_against_the_evolving_text(fs, multi, call):
    """A later edit matches what earlier ones already produced."""
    result = call(
        multi,
        path="app.py",
        edits=[_edit("one = 1", "renamed = 1"), _edit("renamed = 1", "renamed = 42")],
    )

    assert "renamed = 42" in fs.read("app.py")
    assert result.update["session_log"], "a successful multi_edit is recorded"


def test_an_empty_edit_list_is_refused(fs, multi, call, message):
    before = fs.read("app.py")
    result = call(multi, path="app.py", edits=[])
    assert message(result).status == "error"
    assert fs.read("app.py") == before


def test_a_missing_file_is_an_error_not_a_crash(multi, call, message):
    result = call(multi, path="nope.py", edits=[_edit("a", "b")])
    assert message(result).status == "error"
    assert "could not read" in message(result).content


def test_multi_edit_is_absent_from_a_read_only_toolset(fs):
    names = {t.name for t in make_fs_tools(fs, writable=False)}
    assert "multi_edit" not in names and "edit_file" not in names


# -- structured events ----------------------------------------------------


def test_the_summary_reads_numbers_not_prose(tmp_path):
    """No regex anywhere: the numbers were never turned into a sentence."""
    log = [
        file_edit_event("src/app.py", added=18, removed=2),
        file_edit_event("src/app.py", added=4, removed=1),
        file_edit_event("tests/test_app.py", added=9, removed=0, created=True),
    ]
    assert _changed_files(log) == {
        "src/app.py": "+22 -3",
        "tests/test_app.py": "+9 -0  (new)",
    }


def test_the_event_still_reads_well_for_a_human():
    assert file_edit_event("a.py", added=3, removed=1)["detail"] == "a.py: +3 -1"
    assert file_edit_event("a.py", 9, 0, created=True)["detail"] == "a.py: created (9 lines)"
    assert file_edit_event("a.py", 9, 0, created=False)["detail"] == "a.py: overwrote (9 lines)"


def test_a_legacy_event_without_fields_is_skipped_not_crashed():
    """Sessions recorded before the fields existed must not break a summary."""
    from nanocode.state import event

    assert _changed_files([event("file_edit", "src/app.py: +18 -2")]) == {}


def test_edits_from_the_real_tools_feed_the_summary(fs, call):
    """End to end: what the tools record is what the summary reads.

    The bug this guards against is a producer and a consumer drifting apart —
    which is exactly what the old format-then-reparse arrangement allowed.
    """
    tools = {t.name: t for t in make_fs_tools(fs)}
    log = []
    log += call(tools["edit_file"], path="app.py", old_string="one = 1",
                new_string="one = 100").update["session_log"]
    log += call(tools["write_file"], path="new.py",
                content="a\nb\nc\n").update["session_log"]
    log += call(tools["multi_edit"], path="app.py",
                edits=[_edit("two = 2", "two = 22")]).update["session_log"]

    assert _changed_files(log) == {
        "app.py": "+2 -2",
        "new.py": "+3 -0  (new)",
    }


# -- the sub-agent model split -------------------------------------------


def test_sub_agents_share_the_model_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    orch = build_orchestrator(model="openai:gpt-5.4-mini", root=tmp_path)
    assert orch.subagent_spec == "", "no split unless one was asked for"


def test_a_cheaper_model_can_be_given_to_sub_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    orch = build_orchestrator(
        model="openai:gpt-5.6-sol", subagent_model="openai:gpt-5.4-mini", root=tmp_path
    )

    assert orch.model.model_name == "gpt-5.6-sol", "judgment stays on the good model"
    assert orch.subagent_spec == "openai:gpt-5.4-mini"


def test_switching_models_does_not_pin_sub_agents_to_the_old_one(tmp_path, monkeypatch):
    """subagent_spec must stay empty when no split was asked for, or /model
    would silently leave sub-agents on the pre-switch model."""
    import io

    from rich.console import Console

    from nanocode import commands
    from nanocode.commands import CommandContext
    from nanocode.ui import NanocodeUI

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    orch = build_orchestrator(model="openai:gpt-5.4-mini", root=tmp_path)
    ctx = CommandContext(
        console=Console(file=io.StringIO(), no_color=True),
        ui=NanocodeUI(live=False),
        root=tmp_path,
        orch=orch,
    )
    commands.dispatch("/model openai:gpt-5.6-sol", ctx)

    assert ctx.orch.model.model_name == "gpt-5.6-sol"
    assert ctx.orch.subagent_spec == "", "sub-agents should follow the new model"


def test_an_explicit_split_survives_a_model_switch(tmp_path, monkeypatch):
    import io

    from rich.console import Console

    from nanocode import commands
    from nanocode.commands import CommandContext
    from nanocode.ui import NanocodeUI

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    orch = build_orchestrator(
        model="openai:gpt-5.4-mini", subagent_model="openai:gpt-5.4-mini", root=tmp_path
    )
    ctx = CommandContext(
        console=Console(file=io.StringIO(), no_color=True),
        ui=NanocodeUI(live=False),
        root=tmp_path,
        orch=orch,
    )
    commands.dispatch("/model openai:gpt-5.6-sol", ctx)

    assert ctx.orch.subagent_spec == "openai:gpt-5.4-mini", "a cost decision shouldn't be undone"


# -- prompt caching -------------------------------------------------------


def test_caching_middleware_is_installed():
    from nanocode.orchestrator import _caching_middleware

    installed = _caching_middleware()
    assert installed, "prompt caching should be wired when the package provides it"
    assert "Caching" in type(installed[0]).__name__


def test_a_missing_caching_package_costs_a_discount_not_a_startup(monkeypatch):
    """It's a convenience. A renamed or absent middleware must not stop nanocode."""
    import builtins

    from nanocode.orchestrator import _caching_middleware

    real_import = builtins.__import__

    def no_anthropic(name, *args, **kwargs):
        if name.startswith("langchain_anthropic"):
            raise ImportError("gone")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    assert _caching_middleware() == []


def test_caching_is_silent_on_a_non_anthropic_model(tmp_path):
    """It must never raise or warn its way into a non-Anthropic run."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        orch = build_orchestrator(model=Quiet(), root=tmp_path)
    assert orch is not None