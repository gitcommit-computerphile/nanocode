"""Slash commands — things you do *to* the session rather than ask it to do.

Kept out of the prompt loop and out of the model's hands on purpose. Clearing
context and swapping models are decisions about the session itself; routing them
through the agent would mean asking a model to reason about the machinery it is
running inside, and would put both behind a token cost for no benefit.

An unknown `/word` is passed through to the model rather than rejected, so
"/health should return 200" is still a task you can type.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import models, session
from .models import ModelListError
from .orchestrator import PROVIDER_KEYS, ConfigError, Orchestrator, build_orchestrator
from .ui import GLYPH, NanocodeUI, _escape

HELP = [
    ("/model", "list the models your keys can reach, and switch to one"),
    ("/clear", "forget the conversation and start fresh in this project"),
    ("/help", "this list"),
    ("exit", "leave (or 'quit', or Ctrl-D)"),
]

# Enough of a list to scan; more than that wants a filter, not a longer page.
MAX_LISTED = 40

_threads = itertools.count(1)


@dataclass
class CommandContext:
    """What a command is allowed to touch. Mutated in place by the handlers."""

    console: Console
    ui: NanocodeUI
    root: Path
    orch: Orchestrator
    constraints: list[str] = field(default_factory=list)
    # Set by /clear so the loop knows to drop its cached state and re-seed.
    cleared: bool = False


def is_command(text: str) -> bool:
    """True only for commands we actually handle — see the module docstring."""
    if not text.startswith("/"):
        return False
    return text.split(maxsplit=1)[0].lower() in {"/model", "/clear", "/help", "/?"}


def dispatch(text: str, ctx: CommandContext) -> None:
    head, _, rest = text.partition(" ")
    argument = rest.strip()

    if head.lower() == "/clear":
        _clear(ctx)
    elif head.lower() == "/model":
        _model(ctx, argument)
    else:
        _help(ctx)


# -- /help ----------------------------------------------------------------


def _help(ctx: CommandContext) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="dim")
    for name, description in HELP:
        table.add_row(name, description)
    ctx.console.print()
    ctx.console.print(table)


# -- /clear ---------------------------------------------------------------


def _clear(ctx: CommandContext) -> None:
    """Start a fresh context in the same project.

    A new thread id rather than an edit to the existing one. `session_log` has
    an `operator.add` reducer, so assigning `[]` to it appends nothing and
    clears nothing — rotating the thread is the only honest reset.

    Constraints deliberately survive. They are project state, not conversation,
    and silently dropping the user's standing rules on a context clear would be
    the most annoying possible reading of "clear".
    """
    ctx.orch.thread = f"{ctx.orch.thread.split('#')[0]}#{next(_threads)}"
    ctx.cleared = True
    ctx.ui.set_todos([])

    ctx.console.print(f"\n[green]{GLYPH['ok']}[/green] context cleared — plan, history and log")
    if ctx.constraints:
        kept = len(ctx.constraints)
        ctx.console.print(
            f"[dim]{kept} standing constraint{'' if kept == 1 else 's'} kept — "
            f"they're project rules, not conversation. Edit "
            f"{session.constraints_path(ctx.root).name} to change them.[/dim]"
        )
    ctx.console.print("[dim]files on disk are untouched, and earlier runs are still saved[/dim]")


# -- /model ---------------------------------------------------------------


def _model(ctx: CommandContext, argument: str) -> None:
    """List what the keys can reach, then switch. Asks for a key if needed."""
    if ":" in argument:
        _switch(ctx, argument)
        return

    provider = _pick_provider(ctx, argument)
    if provider is None:
        return
    if not _ensure_key(ctx, provider):
        return

    ctx.console.print(f"[dim]asking {provider} what your key can reach…[/dim]")
    try:
        available = models.list_models(provider)
    except ModelListError as exc:
        ctx.console.print(f"[red]{GLYPH['fail']}[/red] {_escape(str(exc))}")
        ctx.console.print("[dim]you can still switch directly: /model openai:gpt-5.4-mini[/dim]")
        return

    if argument:
        needle = argument.lower()
        available = [m for m in available if needle in m.lower()] or available
    if not available:
        ctx.console.print(f"[yellow]{GLYPH['warn']}[/yellow] {provider} listed no chat models")
        return

    choice = _choose(ctx, available)
    if choice:
        _switch(ctx, choice)


def _pick_provider(ctx: CommandContext, argument: str) -> str | None:
    providers = sorted(PROVIDER_KEYS)
    ready = [p for p in providers if models.api_key_for(p)]

    # Nothing to decide when exactly one provider is usable.
    if len(ready) == 1 and not argument:
        return ready[0]

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(style="dim")
    for index, provider in enumerate(providers, 1):
        has_key = bool(models.api_key_for(provider))
        table.add_row(
            str(index),
            provider,
            f"{PROVIDER_KEYS[provider]} set" if has_key else "no key yet — you'll be asked",
        )
    ctx.console.print()
    ctx.console.print(table)

    answer = _ask(ctx, "provider (number, or blank to cancel)")
    if not answer:
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(providers):
        return providers[int(answer) - 1]
    if answer.lower() in providers:
        return answer.lower()

    ctx.console.print(f"[red]{GLYPH['fail']}[/red] no such provider: {_escape(answer)}")
    return None


def _ensure_key(ctx: CommandContext, provider: str) -> bool:
    """Make sure the provider's key is in the environment, asking if it isn't.

    Held for this process only. Nanocode never writes a credential to disk, so
    a key entered here is gone when you close the terminal — the console says
    so, rather than leaving the user to assume it was saved.
    """
    var = PROVIDER_KEYS[provider]
    if os.environ.get(var):
        return True

    ctx.console.print(f"\n[dim]{var} isn't set in this terminal.[/dim]")
    key = _ask(ctx, f"{var} (paste it, or blank to cancel)", secret=True)
    if not key:
        ctx.console.print("[dim]cancelled — nothing changed[/dim]")
        return False

    os.environ[var] = key
    ctx.console.print(
        f"[dim]{GLYPH['ok']} held for this session only — nanocode never writes keys to disk.\n"
        f"  to keep it: setx {var} \"...\" (PowerShell), then reopen the terminal[/dim]"
    )
    return True


def _choose(ctx: CommandContext, available: list[str]) -> str | None:
    shown = available[:MAX_LISTED]
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="cyan", no_wrap=True, justify="right")
    table.add_column(no_wrap=True)
    for index, name in enumerate(shown, 1):
        table.add_row(str(index), name)
    ctx.console.print()
    ctx.console.print(table)
    if len(available) > len(shown):
        ctx.console.print(
            f"[dim]{len(available) - len(shown)} more — narrow it with e.g. /model gpt-5[/dim]"
        )
    ctx.console.print(f"[dim]currently: {ctx.orch.spec}[/dim]")

    answer = _ask(ctx, "model (number or full name, blank to cancel)")
    if not answer:
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(shown):
        return shown[int(answer) - 1]
    return answer if ":" in answer else f"{available[0].split(':', 1)[0]}:{answer}"


def _switch(ctx: CommandContext, spec: str) -> None:
    """Rebuild the graph around the new model, keeping the conversation.

    The replacement gets the *same* checkpointer and thread id, so switching
    model mid-task costs nothing in context — which is the point of being able
    to do it at all: reach for a bigger model on the hard step, not the hard
    session.
    """
    if spec == ctx.orch.spec:
        ctx.console.print(f"[dim]already on {_escape(spec)}[/dim]")
        return

    try:
        replacement = build_orchestrator(
            model=spec,
            # /model switches the orchestrator's model; a separate sub-agent
            # model was a cost decision and stays as it was.
            subagent_model=ctx.orch.subagent_spec or None,
            root=ctx.orch.root,
            fs=ctx.orch.fs,
            context_window=ctx.orch.context_window,
            on_trace=ctx.orch.on_trace,
            on_retry=ctx.orch.on_retry,
            on_compact=ctx.orch.on_compact,
            checkpointer=ctx.orch.checkpointer,
            # Same tally, so a mid-session swap doesn't reset what you've spent.
            usage=ctx.orch.usage,
            # Reuse the detection rather than paying for it again.
            git_context=ctx.orch.git,
        )
    except ConfigError as exc:
        ctx.console.print(f"[red]{GLYPH['fail']}[/red] {_escape(str(exc))}")
        return

    replacement.thread = ctx.orch.thread
    was = ctx.orch.spec
    ctx.orch = replacement
    ctx.console.print(
        f"[green]{GLYPH['ok']}[/green] {_escape(was)} [dim]→[/dim] {_escape(spec)}"
        "  [dim](conversation kept)[/dim]"
    )


# -- prompting ------------------------------------------------------------


def _ask(ctx: CommandContext, label: str, secret: bool = False) -> str:
    try:
        return ctx.console.input(f"[bold cyan]{label}: [/bold cyan]", password=secret).strip()
    except (EOFError, KeyboardInterrupt):
        ctx.console.print()
        return ""
