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
from nanocode.state import Todo, event, file_edit_event
from nanocode.subagents import REGISTRY
from nanocode.ui import _changed_files


def test_save_then_load_round_trips(tmp_path):
    state = {
        "todos": [Todo(content="wire up middleware", status="in_progress")],
        "session_log": [event("file_edit", "src/api/middleware.py: +18 -2")],
    }
    path = session.save(tmp_path, state, task="add rate limiting", session_id="20260814T120000")
    assert path == tmp_path / ".nanocode" / "sessions" / "20260814T120000.json"

    saved = session.load(tmp_path)
    assert saved["task"] == "add rate limiting"
    assert saved["todos"] == state["todos"]
    assert saved["session_log"][0]["detail"] == "src/api/middleware.py: +18 -2"
    assert saved["cwd"] == str(tmp_path.resolve())
    assert saved["updated_at"]


def test_saved_session_never_contains_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-do-not-persist")
    path = session.save(tmp_path, {"todos": [], "session_log": []}, "anything", "20260814T120000")
    raw = path.read_text(encoding="utf-8")
    assert "sk-ant" not in raw
    assert set(json.loads(raw)) == {"id", "task", "todos", "session_log", "cwd", "updated_at"}


def test_load_returns_none_without_a_session(tmp_path):
    assert session.load(tmp_path) is None


def test_load_survives_a_corrupt_session_file(tmp_path):
    session.sessions_dir(tmp_path).joinpath("20260814T120000.json").write_text(
        "{ broken", encoding="utf-8"
    )
    assert session.load(tmp_path) is None


# -- one file per run; nothing is ever overwritten ------------------------


def test_each_run_gets_its_own_file(tmp_path):
    """The bug this replaced: a second run destroyed the first one's record."""
    session.save(tmp_path, {"todos": [Todo(content="a", status="pending")]}, "first", "20260813T090000")
    session.save(tmp_path, {"todos": [Todo(content="b", status="pending")]}, "second", "20260814T090000")

    assert len(session.list_sessions(tmp_path)) == 2
    # The older record is still readable, not clobbered.
    assert session.load(tmp_path, "20260813T090000")["task"] == "first"
    # And a bare load gets the most recent.
    assert session.load(tmp_path)["task"] == "second"


def test_a_legacy_single_file_session_still_resumes(tmp_path):
    """Projects written by the overwriting version must not be stranded."""
    session.nanocode_dir(tmp_path).joinpath("session.json").write_text(
        json.dumps({"task": "from the old layout", "todos": [], "session_log": []}),
        encoding="utf-8",
    )
    assert session.load(tmp_path)["task"] == "from the old layout"


def test_pruning_keeps_the_newest_sessions(tmp_path):
    for i in range(6):
        session.save(tmp_path, {}, f"run {i}", f"2026081{i}T090000")

    assert session.prune_sessions(tmp_path, keep=2) == 4
    remaining = [p.stem for p in session.list_sessions(tmp_path)]
    assert remaining == ["20260814T090000", "20260815T090000"]


# -- picking work back up -------------------------------------------------


def test_unfinished_work_is_what_triggers_a_pick_up(tmp_path):
    finished = {"todos": [Todo(content="done", status="completed")]}
    partial = {"todos": [Todo(content="done", status="completed"), Todo(content="not", status="pending")]}

    assert session.has_unfinished_work(partial) is True
    # A session that simply ended shouldn't haunt the next one.
    assert session.has_unfinished_work(finished) is False
    assert session.has_unfinished_work(None) is False
    assert session.has_unfinished_work({"todos": []}) is False


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
        session_id="20260814T120000",
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


# -- constraints: project-scoped, not session-scoped ----------------------


def test_constraints_round_trip_through_a_readable_file(tmp_path):
    rules = ["do not modify the auth module", "always run the tests before finishing"]
    path = session.save_constraints(tmp_path, rules)

    assert path == tmp_path / ".nanocode" / "constraints.md"
    assert session.load_constraints(tmp_path) == rules
    # Editable by hand, which is half the point of using markdown.
    assert "- do not modify the auth module" in path.read_text(encoding="utf-8")


def test_constraints_can_be_written_by_hand(tmp_path):
    session.nanocode_dir(tmp_path).joinpath("constraints.md").write_text(
        "# Constraints\n\n- targets Python 3.11\n\n* uses the existing retry helper\n",
        encoding="utf-8",
    )
    assert session.load_constraints(tmp_path) == [
        "targets Python 3.11",
        "uses the existing retry helper",
    ]


def test_constraints_survive_a_new_session(tmp_path):
    """The whole point: they outlive the run that recorded them."""
    session.save_constraints(tmp_path, ["do not modify the auth module"])
    session.save(tmp_path, {"todos": []}, "some later task", "20260814T120000")

    assert session.load_constraints(tmp_path) == ["do not modify the auth module"]


def test_rewriting_constraints_drops_the_old_set(tmp_path):
    """Overwrite, not append — a lifted rule has to be removable."""
    session.save_constraints(tmp_path, ["old rule", "kept rule"])
    session.save_constraints(tmp_path, ["kept rule"])
    assert session.load_constraints(tmp_path) == ["kept rule"]


def test_no_constraints_file_means_no_constraints(tmp_path):
    assert session.load_constraints(tmp_path) == []


def test_gitignore_hint_only_fires_when_relevant(tmp_path):
    assert session.add_to_gitignore_hint(tmp_path) is False  # no .gitignore at all

    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__pycache__/\n", encoding="utf-8")
    assert session.add_to_gitignore_hint(tmp_path) is True

    gitignore.write_text("__pycache__/\n.nanocode/\n", encoding="utf-8")
    assert session.add_to_gitignore_hint(tmp_path) is False


# -- the summary is read out of session_log, not reconstructed ------------


def test_changed_files_folds_the_edit_record():
    """Built with the real producer, not hand-written strings.

    The previous version of this test wrote the `detail` text itself, which
    meant it checked the parser against its own assumption rather than against
    what the filesystem tools actually record. Reword the producer and the test
    stayed green while every summary silently went blank.
    """
    log = [
        file_edit_event("src/api/middleware.py", added=18, removed=2),
        file_edit_event("src/api/middleware.py", added=4, removed=1),
        file_edit_event("tests/test_middleware.py", added=9, removed=0, created=True),
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
