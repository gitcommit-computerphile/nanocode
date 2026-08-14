from __future__ import annotations

from pathlib import Path

from rich.console import Console

from nanocode.state import Todo, event
from nanocode.ui import FANCY, PLAIN, NanocodeUI, _pick_glyphs


def rendered(fn, width: int = 100) -> str:
    console = Console(file=__import__("io").StringIO(), width=width, no_color=True, markup=True)
    ui = NanocodeUI(console, live=False)
    fn(ui)
    return console.file.getvalue()


def test_plain_mode_prints_the_plan_label():
    """Rich reads square brackets as markup — a literal '[plan]' vanishes."""
    out = rendered(lambda ui: ui.set_todos([Todo(content="a", status="pending")]))
    assert "plan" in out
    assert "1 steps, 0 complete" in out


def test_tool_lines_survive_bracketed_arguments():
    out = rendered(lambda ui: ui.tool("grep", "def \\[handler\\]"))
    assert "grep" in out


def test_subagent_lines_are_indented():
    out = rendered(lambda ui: ui.subagent("explorer", "find the pattern"))
    assert out.startswith("  ")
    assert "explorer" in out


def test_summary_reports_changes_and_delegations(tmp_path):
    log = [
        event("file_edit", "src/app.py: +18 -2"),
        event("file_edit", "tests/test_app.py: created (9 lines)"),
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


def test_summary_reports_constraints_in_force(tmp_path):
    out = rendered(
        lambda ui: ui.summary([], [], Path(tmp_path), constraints=["no auth edits", "py3.11"])
    )
    assert "2 in force" in out


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
