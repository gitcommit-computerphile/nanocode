from __future__ import annotations

import json

import pytest

from langchain.agents import create_agent

from nanocode import session
from nanocode.orchestrator import (
    DEFAULT_MODEL,
    ConfigError,
    build_orchestrator,
    resolve_model,
)
from nanocode.state import Todo, event
from nanocode.subagents import REGISTRY
from nanocode.ui import _changed_files


def test_save_then_load_round_trips(tmp_path):
    state = {
        "todos": [Todo(content="wire up middleware", status="in_progress")],
        "session_log": [event("file_edit", "src/api/middleware.py: +18 -2")],
    }
    path = session.save(tmp_path, state, task="add rate limiting")
    assert path == tmp_path / ".nanocode" / "session.json"

    saved = session.load(tmp_path)
    assert saved["task"] == "add rate limiting"
    assert saved["todos"] == state["todos"]
    assert saved["session_log"][0]["detail"] == "src/api/middleware.py: +18 -2"
    assert saved["cwd"] == str(tmp_path.resolve())
    assert saved["updated_at"]


def test_saved_session_never_contains_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-do-not-persist")
    session.save(tmp_path, {"todos": [], "session_log": []}, task="anything")
    raw = (tmp_path / ".nanocode" / "session.json").read_text(encoding="utf-8")
    assert "sk-ant" not in raw
    assert set(json.loads(raw)) == {"task", "todos", "session_log", "cwd", "updated_at"}


def test_load_returns_none_without_a_session(tmp_path):
    assert session.load(tmp_path) is None


def test_load_survives_a_corrupt_session_file(tmp_path):
    session.nanocode_dir(tmp_path).joinpath("session.json").write_text("{ broken", encoding="utf-8")
    assert session.load(tmp_path) is None


def test_resume_prompt_rebuilds_context_from_the_record(tmp_path):
    session.save(
        tmp_path,
        {
            "todos": [
                Todo(content="find the pattern", status="completed"),
                Todo(content="write the middleware", status="in_progress"),
                Todo(content="update the tests", status="pending"),
            ],
            "session_log": [
                event("delegate", "explorer: found 3 existing middleware, pattern is X"),
                event("file_edit", "src/api/middleware.py: +18 -2"),
            ],
        },
        task="add rate limiting middleware",
    )

    prompt = session.resume_prompt(session.load(tmp_path))
    assert "add rate limiting middleware" in prompt
    assert "[x] find the pattern" in prompt
    assert "[>] write the middleware" in prompt
    assert "explorer: found 3 existing middleware" in prompt
    assert "already on disk" in prompt


def test_resume_prompt_truncates_a_long_log(tmp_path):
    log = [event("shell", f"command {i}") for i in range(200)]
    prompt = session.resume_prompt({"task": "t", "todos": [], "session_log": log})
    assert "command 199" in prompt
    assert "command 0\n" not in prompt
    assert "of 200" in prompt


def test_gitignore_hint_only_fires_when_relevant(tmp_path):
    assert session.add_to_gitignore_hint(tmp_path) is False  # no .gitignore at all

    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__pycache__/\n", encoding="utf-8")
    assert session.add_to_gitignore_hint(tmp_path) is True

    gitignore.write_text("__pycache__/\n.nanocode/\n", encoding="utf-8")
    assert session.add_to_gitignore_hint(tmp_path) is False


# -- the summary is read out of session_log, not reconstructed ------------


def test_changed_files_folds_the_edit_record():
    log = [
        event("file_edit", "src/api/middleware.py: +18 -2"),
        event("file_edit", "src/api/middleware.py: +4 -1"),
        event("file_edit", "tests/test_middleware.py: created (9 lines)"),
        event("shell", "pytest -> exit 0"),
        event("delegate", "explorer: something"),
    ]
    assert _changed_files(log) == {
        "src/api/middleware.py": "+22 -3",
        "tests/test_middleware.py": "+9 -0  (new)",
    }


# -- model selection ------------------------------------------------------


def test_model_string_must_name_a_provider():
    with pytest.raises(ConfigError, match="provider:name"):
        resolve_model("gpt-5.4-mini")


def test_missing_api_key_is_reported_before_any_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        resolve_model("openai:gpt-5.4-mini")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        resolve_model("anthropic:claude-opus-5")


def test_the_default_model_is_usable_as_written(monkeypatch):
    """The shipped default must parse and resolve — not just look plausible."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    model = resolve_model(DEFAULT_MODEL)
    assert model.model_name == "gpt-5.4-mini"


def test_one_flag_picks_the_model_for_orchestrator_and_subagents(monkeypatch, tmp_path):
    """Sub-agents inherit the run's model — nothing mixes providers mid-task."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    orch = build_orchestrator(root=tmp_path)
    assert orch.model.model_name == "gpt-5.4-mini"

    for spec in REGISTRY.values():
        agent = create_agent(
            orch.model,
            tools=spec.build_tools(orch.fs, tmp_path),
            system_prompt=spec.prompt,
        )
        assert agent is not None
