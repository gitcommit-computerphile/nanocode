"""Terminal UI.

A live plan panel pinned to the bottom, tool activity streaming above it, and
sub-agent work indented so it reads as a branch. Falls back to plain lines
whenever the output is not a TTY, so piping to a file still works.
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .state import Event, Todo
from .usage import Usage, human

# How much of an in-progress reply to show above the plan panel.
STREAM_PREVIEW_LINES = 12

# Shown while waiting on the model. The words rotate purely so a long wait
# reads as ongoing rather than stuck — a static label starts to look like a
# hang after ten seconds, whatever the spinner is doing.
THINKING_WORDS = (
    "Thinking", "Pondering", "Considering", "Reasoning", "Puzzling",
    "Mulling", "Weighing", "Untangling", "Tracing", "Deliberating",
    "Sketching", "Digesting", "Plotting", "Chewing", "Squinting",
)
WORD_SECONDS = 4.0
# Below this a wait is not worth naming; the eye barely registers it.
THINKING_AFTER = 0.4

FANCY = {
    "ok": "✓", "fail": "✗", "run": "▶", "todo": "○", "branch": "↳", "warn": "!",
    "read": "·", "edit": "✎", "bar": "▌", "dot": "·",
}
PLAIN = {
    "ok": "+", "fail": "x", "run": ">", "todo": "-", "branch": "->", "warn": "!",
    "read": ".", "edit": "*", "bar": "|", "dot": ".",
}

# One category glyph per tool, so a long run reads at a glance: what was looked
# at, what was changed, what was executed.
TOOL_KIND = {
    "ls": "read", "read_file": "read", "grep": "read", "glob": "read", "git": "read",
    "web_search": "read",
    "write_file": "edit", "edit_file": "edit", "multi_edit": "edit",
    "write_constraints": "edit",
    "shell": "run",
    "delegate": "branch",
}

# Tool names are padded to this width so their arguments line up in a column.
NAME_WIDTH = 13
# Below this, a duration is noise rather than information.
SLOW_SECONDS = 1.0


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
        self._stream: str = ""
        # tool_call_id -> (name, summary, started_at) for calls still running.
        self._running: dict[str, tuple[str, str, float]] = {}
        # Calls already logged at start (the no-live path), so end_tool knows
        # not to print them a second time.
        self._printed: set[str] = set()
        self._thinking_since: float | None = None
        self._word_seed = random.randrange(len(THINKING_WORDS))

    # -- the opening block -------------------------------------------------

    def header(
        self,
        model: str,
        root: Path,
        branch: str = "",
        changed: int = 0,
        subagent_model: str = "",
        note: str = "",
    ) -> None:
        """The startup block.

        A framed panel rather than loose lines: this is the first thing anyone
        sees, and a bordered block makes the session's setup read as one unit
        instead of stray output that happens to be at the top.
        """
        rows = [("model", model)]
        if subagent_model:
            rows.append(("sub-agents", subagent_model))
        rows.append(("project", _escape(str(root))))
        if branch:
            dirty = f"  [dim]{changed} uncommitted[/dim]" if changed else "  [dim]clean[/dim]"
            rows.append(("branch", f"{branch}{dirty}"))

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_column(style="dim", no_wrap=True, width=11)
        table.add_column(overflow="fold")
        for label, value in rows:
            table.add_row(label, value)

        body: list[Any] = [
            Text.assemble(("nanocode", "bold cyan"), ("  a small coding agent", "dim")),
            Text(""),
            table,
        ]
        if note:
            body += [Text(""), Text(note, style="yellow")]

        self.console.print()
        self.console.print(
            Panel(
                Group(*body),
                border_style="cyan",
                padding=(1, 2),
                expand=False,
            )
        )

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> NanocodeUI:
        if self.use_live:
            self._live = Live(
                self._renderable(),
                console=self.console,
                refresh_per_second=12,
                vertical_overflow="visible",
                # Cleared when the ask ends. While working it is live feedback
                # pinned below the output; afterwards it would sit between the
                # answer and the summary, restating a count the summary gives.
                # `summary` reprints the finished checklist in its proper place.
                transient=True,
            )
            self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end_stream()
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    # -- waiting on the model ----------------------------------------------

    def begin_thinking(self) -> None:
        """The model has been asked something and hasn't answered yet.

        This covers the gaps nothing else does: the pause after you hit enter,
        and every pause between a tool returning and the next call going out.
        Both used to be dead silence, which is indistinguishable from a hang.
        """
        if self._thinking_since is None:
            self._thinking_since = time.monotonic()
            self._word_seed = random.randrange(len(THINKING_WORDS))

    def end_thinking(self) -> None:
        self._thinking_since = None

    def compacting(self, active: bool) -> None:
        """Compaction runs an extra model call mid-task, taking seconds.

        It reuses the in-flight tool machinery so it gets the same spinner and
        the same closing line with a duration — the user should be able to see
        where those seconds went, and that it was the tool and not their task.
        """
        if active:
            self.begin_tool("__compaction__", "compacting", "summarising earlier turns")
        else:
            self.end_tool("__compaction__")

    # -- streaming ---------------------------------------------------------

    def stream(self, text: str) -> None:
        """Show assistant text as it arrives, without committing it.

        The tokens go into the Live region alongside the plan panel rather than
        being written straight to the console: a pinned Live region and partial
        line writes fight each other, and the result is a mangled panel. Nothing
        is printed permanently until `end_stream`.

        Outside a live terminal this deliberately does nothing visible, so
        `--plain > run.log` still gets one clean block of text per reply
        instead of a stutter of fragments.
        """
        if not text:
            return
        # Tokens arriving means the wait is over.
        self.end_thinking()
        self._stream += text
        if self._live is not None:
            self._live.update(self._renderable())

    def end_stream(self) -> bool:
        """Commit whatever was streamed. True if there was anything.

        Committing goes through `assistant()` so that streamed and unstreamed
        replies land identically — one code path for what the user finally sees.
        """
        if not (text := self._stream.strip()):
            self._stream = ""
            return False
        self._stream = ""
        if self._live is not None:
            self._live.update(self._renderable())
        self.assistant(text)
        return True

    def _renderable(self) -> Any:
        # A live-computed object rather than a fixed Group: Live re-renders
        # whatever it holds on every refresh, but only re-runs *this* method
        # when something calls update(). Returning a self-rendering frame is
        # what lets the elapsed clocks tick and the waiting word rotate
        # between updates.
        return _Frame(self)

    def _parts(self) -> list[Any]:
        """One frame of the live region. Called on every refresh."""
        parts: list[Any] = []
        if self._stream:
            # Only the tail: a long reply shouldn't push the plan off-screen.
            lines = self._stream.splitlines()[-STREAM_PREVIEW_LINES:]
            parts.append(Text("\n".join(lines)))

        for name, summary, started in self._running.values():
            # A spinner rather than a static line, because the whole point is
            # to show that something is still happening.
            elapsed = time.monotonic() - started
            label = Text.assemble(
                (f"{name} ", "cyan"),
                (_one_line(summary, 70), "dim"),
                (f"  {elapsed:.0f}s" if elapsed >= SLOW_SECONDS else "", "dim"),
            )
            parts.append(Spinner("dots", text=label, style="cyan"))

        if (waiting := self._waiting_label()) is not None:
            parts.append(Spinner("dots", text=waiting, style="magenta"))

        if self.todos:
            parts.append(self._panel())
        return parts

    def _waiting_label(self) -> Text | None:
        """The 'Thinking… 6s' line, or None when nothing is being waited on."""
        if self._thinking_since is None or self._running or self._stream:
            return None
        elapsed = time.monotonic() - self._thinking_since
        if elapsed < THINKING_AFTER:
            return None
        word = THINKING_WORDS[(self._word_seed + int(elapsed // WORD_SECONDS)) % len(THINKING_WORDS)]
        return Text.assemble(
            (f"{word}… ", "magenta"),
            (f"{elapsed:.0f}s", "dim"),
            ("  ctrl-c to stop" if elapsed >= 10 else "", "dim"),
        )

    # -- plan panel --------------------------------------------------------

    def set_todos(self, todos: list[Todo]) -> None:
        if todos == self.todos:
            return
        self.todos = todos
        if self._live is not None:
            self._live.update(self._renderable())
        elif todos:
            done = sum(1 for t in todos if t["status"] == "completed")
            step = "step" if len(todos) == 1 else "steps"
            # No square brackets: rich would read them as a markup tag and eat them.
            self.console.print(f"  [blue]plan[/blue] [dim]{len(todos)} {step}, {done} complete[/dim]")

    def _panel(self) -> Panel | Text:
        if not self.todos:
            # Nothing planned yet — and possibly nothing ever, since not every
            # message is a task. An empty "planning…" frame on a turn that was
            # only ever a reply is chrome claiming work that isn't happening.
            # Tool lines still stream above this, so activity stays visible.
            return Text("")
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

    def tool(self, name: str, summary: str, seconds: float | None = None) -> None:
        """One aligned line per tool call.

        The name is padded to a fixed column so arguments line up — a run of
        thirty calls should be scannable down the left edge rather than read
        line by line.
        """
        mark = GLYPH[TOOL_KIND.get(name, "read")]
        took = (
            f"  [dim]{seconds:.1f}s[/dim]"
            if seconds is not None and seconds >= SLOW_SECONDS
            else ""
        )
        self._line(
            f"  [dim]{mark}[/dim] [cyan]{name.ljust(NAME_WIDTH)}[/cyan]"
            f"[dim]{_escape(_one_line(summary, 96))}[/dim]{took}"
        )

    def begin_tool(self, call_id: str, name: str, summary: str) -> None:
        """A call has been issued but hasn't returned.

        In a live terminal it goes into the spinner region and is printed
        properly once it finishes, with how long it took. Piped output has no
        spinner to watch, so it prints immediately — a log that only appears
        after a hang is no use for diagnosing the hang.
        """
        # The model has answered; work is starting.
        self.end_thinking()
        # Tracked in both modes so the state machine is the same either way —
        # only *when the line prints* differs.
        self._running[call_id] = (name, summary, time.monotonic())
        if self._live is None:
            self.tool(name, summary)
            self._printed.add(call_id)
        else:
            self._live.update(self._renderable())

    def end_tool(self, call_id: str) -> None:
        entry = self._running.pop(call_id, None)
        if entry is None:
            return
        name, summary, started = entry
        if call_id in self._printed:
            self._printed.discard(call_id)  # already logged when it started
        else:
            self.tool(name, summary, time.monotonic() - started)
        # A finished tool means the model is about to be asked again.
        self.begin_thinking()
        if self._live is not None:
            self._live.update(self._renderable())

    def edit(self, detail: str) -> None:
        self._line(f"  [green]{GLYPH['edit']}[/green] {_escape(detail)}")

    def subagent(self, agent: str, line: str) -> None:
        branch = GLYPH["branch"]
        self._line(
            f"    [magenta]{branch} {agent.ljust(NAME_WIDTH - 2)}[/magenta]"
            f"[dim]{_escape(_one_line(line, 92))}[/dim]"
        )

    def assistant(self, text: str) -> None:
        if text.strip():
            self._line(f"\n{_escape(text.strip())}\n")

    def warn(self, text: str) -> None:
        self._line(f"  [yellow]{GLYPH['warn']}[/yellow] [yellow]{_escape(text)}[/yellow]")

    def error(self, text: str) -> None:
        self._line(f"  [red]{GLYPH['fail']}[/red] [red]{_escape(text)}[/red]")

    def _line(self, markup: str) -> None:
        # Rich places console output above an active Live region automatically.
        self.console.print(markup)

    # -- token accounting --------------------------------------------------

    def status(self, usage: Usage, window: int) -> None:
        """One dim line above the prompt: what's been spent, how full context is.

        Shown before asking rather than only after finishing, because the number
        that matters — how much room is left — is what you'd want to know when
        deciding what to ask next.
        """
        if not usage.calls:
            return
        percent = usage.context_percent(window)
        colour = "red" if percent >= 90 else "yellow" if percent >= 70 else "dim"
        self.console.print(
            f"[dim]{human(usage.input_tokens)} in · {human(usage.output_tokens)} out · "
            f"[/dim][{colour}]{percent:.0f}% of {human(window)} context[/{colour}]"
            + ("[dim] · compaction imminent[/dim]" if percent >= 90 else "")
        )

    # -- final summary -----------------------------------------------------

    def summary(
        self,
        todos: list[Todo],
        session_log: list[Event],
        log_dir: Path,
        resumable: bool = True,
        constraints: list[str] | None = None,
        seconds: float | None = None,
    ) -> None:
        # A turn that was only a reply has nothing to summarise. Printing
        # "0/0 todos complete" under it reports success at doing nothing, and
        # the checkmark makes it read as though something finished.
        if not todos and not session_log and not constraints:
            return

        done = sum(1 for t in todos if t["status"] == "completed")
        self.console.print()
        self.console.rule(style="dim")

        if todos:
            complete = done == len(todos)
            style = "green" if complete else "yellow"
            mark = GLYPH["ok"] if complete else GLYPH["warn"]
            took = f"  [dim]in {_duration(seconds)}[/dim]" if seconds else ""
            self.console.print(
                f"[{style}]{mark} {done}/{len(todos)} todos complete[/{style}]{took}"
            )
            # The checklist itself, now that the live panel has been cleared.
            # Which steps were done is worth keeping in scrollback; the panel
            # was only ever the in-progress view of it.
            for todo in todos:
                glyph, colour = MARKS.get(todo["status"], (GLYPH["todo"], "dim"))
                text = _escape(todo["content"])
                line = f"[dim strike]{text}[/dim strike]" if todo["status"] == "completed" else text
                self.console.print(f"  [{colour}]{glyph}[/{colour}] {line}")
        elif seconds:
            self.console.print(f"[dim]done in {_duration(seconds)}[/dim]")

        changes = _changed_files(session_log)
        if changes:
            table = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
            table.add_column(style="cyan", no_wrap=True)
            table.add_column(style="dim")
            for path, note in changes.items():
                table.add_row(path, note)
            files = f"{len(changes)} file" + ("" if len(changes) == 1 else "s")
            self.console.print(f"\n[bold]changed[/bold] [dim]({files})[/dim]")
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
        # Token totals and context deliberately absent: they are session-wide,
        # and everything else in this block is scoped to the ask just finished.
        # Putting them here made "51.4k in" read as the cost of *this* task, and
        # duplicated the status line printed moments later above the prompt.
        if constraints:
            rows.append(("constraints", f"{len(constraints)} in force"))
        if log_dir.exists():
            rows.append(("full logs", log_dir.as_posix()))
        if resumable and todos and done < len(todos):
            # No flag to remember: running `nanocode` here again picks this up.
            rows.append(("unfinished", "picked up automatically next run"))

        if rows:
            meta = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
            meta.add_column(style="dim", no_wrap=True)
            meta.add_column()
            for label, value in rows:
                meta.add_row(label, value)
            self.console.print()
            self.console.print(meta)


def _changed_files(session_log: list[Event]) -> dict[str, str]:
    """Fold the durable edit record into one line per file.

    Reads the numbers the events carry rather than parsing them back out of
    `detail`. That string is for humans; two files agreeing on its exact
    wording, with nothing checking that they still do, was a silent breakage
    waiting to happen.
    """
    added: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    created: dict[str, bool] = {}

    for entry in session_log:
        # .get on purpose: sessions recorded before these fields existed have
        # only `detail`, and should be skipped rather than crash a summary.
        if entry.get("kind") != "file_edit" or not (path := entry.get("path")):
            continue
        added[path] += int(entry.get("added") or 0)
        removed[path] += int(entry.get("removed") or 0)
        created[path] = created.get(path, False) or bool(entry.get("created"))

    out: dict[str, str] = {}
    for path in sorted(added):
        note = f"+{added[path]} -{removed[path]}"
        if created.get(path):
            note += "  (new)"
        out[path] = note
    return out


class _Frame:
    """The live region, recomputed on every refresh.

    Live re-renders the object it holds on each tick but never asks the UI for
    a fresh one, so anything time-dependent — the elapsed counters, the
    rotating word — has to be produced here rather than baked in at update().
    """

    def __init__(self, ui: NanocodeUI) -> None:
        self.ui = ui

    def __rich_console__(self, console: Console, options: Any) -> Any:
        parts = self.ui._parts()
        yield Group(*parts) if parts else Text("")


def _duration(seconds: float | None) -> str:
    """Human-scale elapsed time: 4.2s, 1m 07s."""
    if not seconds:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s"


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _escape(text: str) -> str:
    return text.replace("[", r"\[")
