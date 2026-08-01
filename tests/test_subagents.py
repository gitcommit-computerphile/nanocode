"""Sub-agent isolation, exercised against a fake model so no key is needed."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from nanocode.fs_tools import VirtualFileSystem
from nanocode.subagents import REGISTRY, _final_text, make_delegate_tool


class ScriptedModel(BaseChatModel):
    """Replays canned AIMessages and records every prompt it is shown.

    Deliberately not a `GenericFakeChatModel` subclass: the agent may reach the
    model through several code paths, and this records all of them.
    """

    replies: list[AIMessage] = []
    seen: list[list] = []

    def __init__(self, replies: list[AIMessage]) -> None:
        super().__init__(replies=list(replies), seen=[])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # the agent binds tools; we ignore them
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if not self.replies:
            raise AssertionError("ScriptedModel ran out of scripted replies")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


@pytest.fixture
def fs():
    return VirtualFileSystem({"src/app.py": "def hello():\n    return 'hi'\n"})


def test_registry_holds_exactly_the_three_documented_roles():
    assert set(REGISTRY) == {"explorer", "coder", "test-runner"}


def test_coder_has_no_shell(fs, tmp_path):
    names = {t.name for t in REGISTRY["coder"].build_tools(fs, tmp_path)}
    assert "shell" not in names
    assert {"read_file", "edit_file", "write_file", "grep", "glob"} <= names


def test_explorer_is_read_only(fs, tmp_path):
    names = {t.name for t in REGISTRY["explorer"].build_tools(fs, tmp_path)}
    assert names.isdisjoint({"write_file", "edit_file", "shell"})


def test_test_runner_has_only_shell(fs, tmp_path):
    names = {t.name for t in REGISTRY["test-runner"].build_tools(fs, tmp_path)}
    assert names == {"shell"}


def test_delegate_returns_only_the_final_summary(fs, tmp_path, call, message):
    model = ScriptedModel([AIMessage("Found the pattern: middlewares live in src/api/.")])
    delegate = make_delegate_tool(model, fs, tmp_path)

    result = call(delegate, task="find the middleware pattern", agent_type="explorer")
    assert message(result).content == "Found the pattern: middlewares live in src/api/."
    assert result.update["session_log"][0]["kind"] == "delegate"


def test_sub_agent_sees_only_its_task_string(fs, tmp_path, call):
    """No orchestrator history crosses the boundary — just the brief."""
    model = ScriptedModel([AIMessage("done")])
    call(
        make_delegate_tool(model, fs, tmp_path),
        task="the only thing it knows",
        agent_type="explorer",
    )

    prompt = model.seen[0]
    assert len(prompt) == 2, "expected exactly a system prompt and the task"
    assert prompt[0].type == "system"
    assert prompt[1].content == "the only thing it knows"


def test_two_delegations_share_no_memory(fs, tmp_path, call):
    model = ScriptedModel([AIMessage("first"), AIMessage("second")])
    delegate = make_delegate_tool(model, fs, tmp_path)

    call(delegate, task="first task", agent_type="explorer")
    call(delegate, task="second task", agent_type="explorer")

    second_prompt = model.seen[1]
    assert len(second_prompt) == 2
    assert "first" not in str(second_prompt[1].content)


def test_unknown_agent_type_is_an_error_not_a_crash(fs, tmp_path, call, message):
    model = ScriptedModel([AIMessage("unused")])
    result = call(make_delegate_tool(model, fs, tmp_path), task="x", agent_type="researcher")
    assert message(result).status == "error"
    assert "explorer" in message(result).content


def test_a_sub_agent_failure_is_reported_not_raised(fs, tmp_path, call, message):
    class Exploding(ScriptedModel):
        def _generate(self, *a, **kw):
            raise RuntimeError("provider is down")

    result = call(make_delegate_tool(Exploding([]), fs, tmp_path), task="x", agent_type="explorer")
    assert message(result).status == "error"
    assert "provider is down" in message(result).content


def test_final_text_handles_block_content():
    messages = [AIMessage(content=[{"type": "text", "text": "the answer"}])]
    assert _final_text(messages) == "the answer"


def test_final_text_falls_back_when_nothing_was_said():
    assert "without producing a summary" in _final_text([])
