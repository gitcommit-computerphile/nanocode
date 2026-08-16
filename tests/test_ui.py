from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from nanocode.state import Todo, event, file_edit_event
from nanocode.ui import FANCY, PLAIN, THINKING_WORDS, NanocodeUI, _pick_glyphs


def rendered(fn, width: int = 100) -> str:
    console = Console(file=__import__("io").StringIO(), width=width, no_color=True, markup=True)
    ui = NanocodeUI(console, live=False)
    fn(ui)
    return console.file.getvalue()


@pytest.fixture
def quiet_ui():
    console = Console(file=__import__("io").StringIO(), width=100, no_color=True)
    return NanocodeUI(console, live=False), console


def test_plain_mode_prints_the_plan_label():
    """Rich reads square brackets as markup — a literal '[plan]' vanishes."""
    out = rendered(lambda ui: ui.set_todos([Todo(content="a", status="pending")]))
    assert "plan" in out
    assert "1 step, 0 complete" in out, "singular for one step"

    two = rendered(
        lambda ui: ui.set_todos(
            [Todo(content="a", status="pending"), Todo(content="b", status="completed")]
        )
    )
    assert "2 steps, 1 complete" in two


def test_a_conversational_turn_prints_no_plan_chrome():
    """Not every message is a task — the UI shouldn't imply otherwise.

    An empty plan panel and a "0/0 todos complete" checkmark under a plain
    reply report work that never happened.
    """
    out = rendered(lambda ui: ui.set_todos([]))
    assert out.strip() == "", "an empty plan should render nothing at all"


def test_a_conversational_turn_gets_no_summary(tmp_path):
    out = rendered(lambda ui: ui.summary(todos=[], session_log=[], log_dir=Path(tmp_path)))
    assert out.strip() == ""
    assert "0/0" not in out


def test_a_summary_still_appears_once_there_is_work(tmp_path):
    """The other direction — don't silence real summaries."""
    out = rendered(
        lambda ui: ui.summary(
            todos=[Todo(content="a", status="completed")],
            session_log=[file_edit_event("src/app.py", added=3, removed=1)],
            log_dir=Path(tmp_path),
        )
    )
    assert "1/1 todos complete" in out
    assert "src/app.py" in out


def test_tool_lines_survive_bracketed_arguments():
    out = rendered(lambda ui: ui.tool("grep", "def \\[handler\\]"))
    assert "grep" in out


def test_subagent_lines_are_indented():
    out = rendered(lambda ui: ui.subagent("explorer", "find the pattern"))
    assert out.startswith("  ")
    assert "explorer" in out


def test_summary_reports_changes_and_delegations(tmp_path):
    log = [
        file_edit_event("src/app.py", added=18, removed=2),
        file_edit_event("tests/test_app.py", added=9, removed=0, created=True),
        event("delegate", "explorer: found it"),
        event("delegate", "test-runner: 3 failed"),
        event("shell", "pytest -> exit 1"),
    ]
    todos = [Todo(content="a", status="completed"), Todo(content="b", status="completed")]
    out = rendered(lambda ui: ui.summary(todos, log, Path(tmp_path)))

    assert "2/2 todos complete" in out
    assert "src/app.py" in out and "+18 -2" in out
    assert "tests/test_app.py" in out and "(new)" in out
    assert "explorer" in out and "test-runner" in out
    # A finished plan gets no pick-up notice — it isn't waiting on anything.
    assert "picked up automatically" not in out


def test_summary_is_honest_about_an_incomplete_plan(tmp_path):
    todos = [Todo(content="a", status="completed"), Todo(content="b", status="pending")]
    out = rendered(lambda ui: ui.summary(todos, [], Path(tmp_path)))
    assert "1/2 todos complete" in out
    # No flag to memorise: running nanocode here again picks it up.
    assert "picked up automatically" in out


def test_the_summary_stays_scoped_to_the_ask_just_finished(tmp_path):
    """No session-wide numbers in a per-ask block.

    Token totals used to sit here beside per-ask rows, so "51.4k in" read as
    the cost of this one task. They live above the prompt instead, where the
    scope is unambiguous.
    """
    out = rendered(
        lambda ui: ui.summary(
            todos=[Todo(content="a", status="completed")],
            session_log=[file_edit_event("src/app.py", 3, 1)],
            log_dir=Path(tmp_path),
            seconds=2.0,
        )
    )
    assert "src/app.py" in out
    assert "tokens" not in out
    assert "context" not in out


def test_summary_reports_constraints_in_force(tmp_path):
    out = rendered(
        lambda ui: ui.summary([], [], Path(tmp_path), constraints=["no auth edits", "py3.11"])
    )
    assert "2 in force" in out


# -- the header -----------------------------------------------------------


def test_the_header_shows_what_the_session_is(tmp_path):
    out = rendered(
        lambda ui: ui.header(
            model="openai:gpt-5.4-mini",
            root=tmp_path,
            branch="feature/x",
            changed=3,
            subagent_model="openai:gpt-5.4-mini",
            note="1 standing constraint loaded",
        )
    )
    assert "nanocode" in out
    assert "openai:gpt-5.4-mini" in out
    assert "feature/x" in out and "3 uncommitted" in out
    assert "1 standing constraint loaded" in out


def test_the_header_omits_git_outside_a_repository(tmp_path):
    out = rendered(lambda ui: ui.header(model="m", root=tmp_path))
    assert "branch" not in out
    assert "nanocode" in out


def test_a_clean_tree_says_clean(tmp_path):
    out = rendered(lambda ui: ui.header(model="m", root=tmp_path, branch="main", changed=0))
    assert "clean" in out and "uncommitted" not in out


# -- the waiting indicator ------------------------------------------------


def test_waiting_says_nothing_for_a_blink(quiet_ui):
    """Sub-half-second waits aren't worth naming."""
    ui, _ = quiet_ui
    ui.begin_thinking()
    assert ui._waiting_label() is None


def test_waiting_shows_a_word_and_a_clock(quiet_ui, monkeypatch):
    ui, _ = quiet_ui
    ui.begin_thinking()
    monkeypatch.setattr("nanocode.ui.time.monotonic", lambda: ui._thinking_since + 6.0)

    label = ui._waiting_label().plain
    assert "…" in label and "6s" in label
    assert label.split("…")[0] in THINKING_WORDS


def test_the_word_changes_so_a_long_wait_does_not_look_stuck(quiet_ui, monkeypatch):
    ui, _ = quiet_ui
    ui.begin_thinking()
    start = ui._thinking_since

    seen = set()
    for elapsed in (1, 5, 9, 13, 17):
        monkeypatch.setattr("nanocode.ui.time.monotonic", lambda e=elapsed: start + e)
        seen.add(ui._waiting_label().plain.split("…")[0])

    assert len(seen) > 1, "a static label starts to read as a hang"


def test_a_long_wait_mentions_how_to_stop(quiet_ui, monkeypatch):
    ui, _ = quiet_ui
    ui.begin_thinking()
    monkeypatch.setattr("nanocode.ui.time.monotonic", lambda: ui._thinking_since + 12.0)
    assert "ctrl-c" in ui._waiting_label().plain


def test_working_replaces_waiting(quiet_ui, monkeypatch):
    """Two spinners at once would be nonsense — a tool means the wait is over."""
    ui, _ = quiet_ui
    ui.begin_thinking()
    start = ui._thinking_since
    monkeypatch.setattr("nanocode.ui.time.monotonic", lambda: start + 5.0)
    assert ui._waiting_label() is not None

    ui.begin_tool("t1", "shell", "pytest")
    assert ui._waiting_label() is None, "a running tool should own the spinner"


def test_a_finished_tool_returns_to_waiting(quiet_ui):
    """The model is about to be asked again; that gap needs covering too."""
    ui, _ = quiet_ui
    ui.begin_tool("t1", "shell", "pytest")
    ui.end_tool("t1")
    assert ui._thinking_since is not None


def test_streaming_text_replaces_waiting(quiet_ui):
    ui, _ = quiet_ui
    ui.begin_thinking()
    ui.stream("the answer begins")
    assert ui._thinking_since is None


# -- compaction announces itself ------------------------------------------


def test_compaction_is_visible_rather_than_a_stall(quiet_ui):
    """An extra model call the user didn't ask for, taking seconds."""
    ui, console = quiet_ui
    ui.compacting(True)
    ui.compacting(False)

    out = console.file.getvalue()
    assert "compacting" in out
    assert "summarising earlier turns" in out


# -- tool lines -----------------------------------------------------------


def test_tool_lines_align_into_a_column():
    """Thirty calls should be scannable down the left edge."""
    out = rendered(lambda ui: (ui.tool("grep", "def handler"), ui.tool("multi_edit", "src/app.py")))
    grep, edit = [line for line in out.splitlines() if line.strip()][:2]

    assert grep.index("def handler") == edit.index("src/app.py"), "arguments should line up"


def test_a_slow_tool_reports_how_long_it_took():
    assert "8.4s" in rendered(lambda ui: ui.tool("shell", "pytest -q", seconds=8.4))


def test_a_fast_tool_does_not_clutter_the_line_with_a_duration():
    """Sub-second timings are noise, not information."""
    assert "0.0s" not in rendered(lambda ui: ui.tool("grep", "x", seconds=0.03))


def test_tool_kind_glyphs_distinguish_reading_from_changing():
    from nanocode.ui import GLYPH

    read = rendered(lambda ui: ui.tool("read_file", "a.py"))
    edit = rendered(lambda ui: ui.tool("multi_edit", "a.py"))
    run = rendered(lambda ui: ui.tool("shell", "pytest"))

    assert GLYPH["read"] in read
    assert GLYPH["edit"] in edit
    assert GLYPH["run"] in run


def test_without_a_live_region_a_tool_prints_when_it_starts(quiet_ui):
    """A piped log that only appears after a hang is no use for the hang."""
    ui, console = quiet_ui
    ui.begin_tool("call-1", "shell", "pytest -q")
    assert "shell" in console.file.getvalue(), "plain mode should not wait for completion"

    before = console.file.getvalue()
    ui.end_tool("call-1")
    assert console.file.getvalue() == before, "and should not print it twice"


def test_ending_an_unknown_tool_call_is_harmless(quiet_ui):
    ui, console = quiet_ui
    ui.end_tool("never-started")
    assert console.file.getvalue() == ""


# -- the closing block ----------------------------------------------------


def test_the_summary_reports_elapsed_time(tmp_path):
    out = rendered(
        lambda ui: ui.summary(
            todos=[Todo(content="a", status="completed")],
            session_log=[],
            log_dir=Path(tmp_path),
            seconds=94.3,
        )
    )
    assert "1m 34s" in out


def test_durations_read_at_human_scale():
    from nanocode.ui import _duration

    assert _duration(4.23) == "4.2s"
    assert _duration(94.3) == "1m 34s"
    assert _duration(None) == ""


def test_an_incomplete_plan_is_not_marked_with_a_tick(tmp_path):
    """The tick means finished. A partial run must not borrow it."""
    from nanocode.ui import GLYPH

    out = rendered(
        lambda ui: ui.summary(
            todos=[Todo(content="a", status="completed"), Todo(content="b", status="pending")],
            session_log=[],
            log_dir=Path(tmp_path),
        )
    )
    headline = next(line for line in out.splitlines() if "todos complete" in line)
    assert "1/2 todos complete" in headline
    assert GLYPH["ok"] not in headline, "the tick means finished; a partial run must not borrow it"
    assert GLYPH["warn"] in headline
    # The per-step checklist still ticks the step that genuinely is done.
    assert f"{GLYPH['ok']} a" in out
    assert f"{GLYPH['todo']} b" in out


def test_glyphs_fall_back_to_ascii_on_a_legacy_console(monkeypatch):
    """A cp1252 console must not crash on the first status line."""

    class Cp1252Stdout:
        encoding = "cp1252"

    monkeypatch.setattr("nanocode.ui.sys.stdout", Cp1252Stdout())
    assert _pick_glyphs() == PLAIN

    class Utf8Stdout:
        encoding = "utf-8"

    monkeypatch.setattr("nanocode.ui.sys.stdout", Utf8Stdout())
    assert _pick_glyphs() == FANCY


def test_every_glyph_has_an_ascii_counterpart():
    assert set(FANCY) == set(PLAIN)
    assert all(v.isascii() for v in PLAIN.values())
