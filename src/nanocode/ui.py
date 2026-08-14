"""Terminal UI.

A live plan panel pinned to the bottom, tool activity streaming above it, and
sub-agent work indented so it reads as a branch. Falls back to plain lines
whenever the output is not a TTY, so piping to a file still works.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .state import Event, Todo

EDIT_DIFF = re.compile(r"^(?P<path>.+?): \+(?P<added>\d+) -(?P<removed>\d+)$")
EDIT_WRITE = re.compile(r"^(?P<path>.+?): (?P<verb>created|overwrote) \((?P<lines>\d+) lines\)$")

FANCY = {"ok": "✓", "fail": "✗", "run": "▶", "todo": "○", "branch": "↳", "warn": "!"}
PLAIN = {"ok": "+", "fail": "x", "run": ">", "todo": "-", "branch": "->", "warn": "!"}


def _prepare_stdio() -> None:
    """Prefer UTF-8 so box drawing and status glyphs survive a Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _pick_glyphs() -> dict[str, str]:
    """Fall back to ASCII when the console encoding cannot represent the glyphs.

    Without this, a legacy cp1252 console raises UnicodeEncodeError on the very
    first status line — turning a clean error message into a crash.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(FANCY.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return PLAIN
    return FANCY


_prepare_stdio()
GLYPH = _pick_glyphs()

MARKS = {
    "completed": (GLYPH["ok"], "green"),
    "in_progress": (GLYPH["run"], "yellow"),
    "pending": (GLYPH["todo"], "dim"),
}


class NanocodeUI:
    """Everything the user sees."""

    def __init__(self, console: Console | None = None, live: bool = True) -> None:
        self.console = console or Console()
        self.use_live = live and self.console.is_terminal
        self.todos: list[Todo] = []
        self._live: Live | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> NanocodeUI:
        if self.use_live:
            self._live = Live(
                self._panel(),
                console=self.console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    # -- plan panel --------------------------------------------------------

    def set_todos(self, todos: list[Todo]) -> None:
        if todos == self.todos:
            return
        self.todos = todos
        if self._live is not None:
            self._live.update(self._panel())
        else:
            done = sum(1 for t in todos if t["status"] == "completed")
            # No square brackets: rich would read them as a markup tag and eat them.
            self.console.print(f"[blue]plan[/blue] {len(todos)} steps, {done} complete")

    def _panel(self) -> Panel | Text:
        if not self.todos:
            return Panel(Text("planning…", style="dim"), title="plan", border_style="dim")
        body = Text()
        for todo in self.todos:
            mark, style = MARKS.get(todo["status"], (GLYPH["todo"], "dim"))
            line_style = "dim strike" if todo["status"] == "completed" else style
            body.append(f"{mark} ", style=style)
            body.append(f"{todo['content']}\n", style=line_style)
        done = sum(1 for t in self.todos if t["status"] == "completed")
        return Panel(
            body,
            title=f"plan · {done}/{len(self.todos)}",
            border_style="blue",
            padding=(0, 1),
        )

    # -- activity ----------------------------------------------------------

    def tool(self, name: str, summary: str) -> None:
        self._line(f"[cyan]{name}[/cyan] {_escape(summary)}")

    def edit(self, detail: str) -> None:
        self._line(f"[green]edit[/green] {_escape(detail)}")

    def subagent(self, agent: str, line: str) -> None:
        branch = GLYPH["branch"]
        self._line(f"  [magenta]{branch} {agent}[/magenta] [dim]{_escape(_one_line(line, 100))}[/dim]")

    def assistant(self, text: str) -> None:
        if text.strip():
            self._line(f"\n{_escape(text.strip())}\n")

    def warn(self, text: str) -> None:
        self._line(f"[yellow]{GLYPH['warn']}[/yellow] {_escape(text)}")

    def error(self, text: str) -> None:
        self._line(f"[red]{GLYPH['fail']}[/red] {_escape(text)}")

    def _line(self, markup: str) -> None:
        # Rich places console output above an active Live region automatically.
        self.console.print(markup)

    # -- final summary -----------------------------------------------------

    def summary(
        self,
        todos: list[Todo],
        session_log: list[Event],
        log_dir: Path,
        resumable: bool = True,
        constraints: list[str] | None = None,
    ) -> None:
        done = sum(1 for t in todos if t["status"] == "completed")
        self.console.print()
        style = "green" if todos and done == len(todos) else "yellow"
        self.console.print(f"[{style}]{GLYPH['ok']} {done}/{len(todos)} todos complete[/{style}]")

        changes = _changed_files(session_log)
        if changes:
            table = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
            table.add_column(style="cyan", no_wrap=True)
            table.add_column(style="dim")
            for path, note in changes.items():
                table.add_row(path, note)
            self.console.print("\n[bold]changed[/bold]")
            self.console.print(table)

        used = Counter(
            e["detail"].split(":", 1)[0] for e in session_log if e.get("kind") == "delegate"
        )
        rows: list[tuple[str, str]] = []
        if used:
            rows.append(("sub-agents used", ", ".join(f"{k} ×{v}" for k, v in used.items())))
        shells = sum(1 for e in session_log if e.get("kind") == "shell")
        if shells:
            rows.append(("shell commands", str(shells)))
        if constraints:
            rows.append(("constraints", f"{len(constraints)} in force"))
        if log_dir.exists():
            rows.append(("full logs", log_dir.as_posix()))
        if resumable and todos and done < len(todos):
            # No flag to remember: running `nanocode` here again picks this up.
            rows.append(("unfinished", "picked up automatically next run"))

        if rows:
            meta = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
            meta.add_column(style="dim", no_wrap=True)
            meta.add_column()
            for label, value in rows:
                meta.add_row(label, value)
            self.console.print()
            self.console.print(meta)


def _changed_files(session_log: list[Event]) -> dict[str, str]:
    """Fold the durable edit record into one line per file.

    Read straight out of session_log rather than reconstructed after the fact,
    which is why it stays accurate on a run that took forty tool calls.
    """
    added: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    created: dict[str, bool] = {}

    for entry in session_log:
        if entry.get("kind") != "file_edit":
            continue
        detail = entry.get("detail", "")
        if match := EDIT_DIFF.match(detail):
            path = match["path"]
            added[path] += int(match["added"])
            removed[path] += int(match["removed"])
            created.setdefault(path, False)
        elif match := EDIT_WRITE.match(detail):
            path = match["path"]
            added[path] += int(match["lines"])
            created[path] = created.get(path, match["verb"] == "created")

    out: dict[str, str] = {}
    for path in sorted(added):
        note = f"+{added[path]} -{removed[path]}"
        if created.get(path):
            note += "  (new)"
        out[path] = note
    return out


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _escape(text: str) -> str:
    return text.replace("[", r"\[")
