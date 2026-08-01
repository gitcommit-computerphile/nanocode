"""Entrypoint — an interactive prompt loop over one orchestrator instance.

Run it, ask for something, keep asking. Each ask runs the same loop to
completion and then prompts again; the conversation carries forward, so "now
make it return JSON" knows what "it" is. `--once` runs a single task and exits
— the same loop, minus the prompt-again step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console

from . import session
from .orchestrator import DEFAULT_MODEL, ConfigError, Orchestrator, build_orchestrator
from .shell_tool import logs_dir
from .todo_tools import DEFAULT_CONTEXT_WINDOW
from .ui import GLYPH, NanocodeUI

QUIET_TOOLS = {"write_todos"}
EXIT_WORDS = {"exit", "quit", ":q", ":wq"}

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="An interactive CLI coding agent. Run it, ask for something, keep asking.",
)


@app.command()
def run(
    task: Annotated[
        list[str] | None,
        typer.Argument(help="An opening task. Omit it to be prompted."),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="provider:name — used by the orchestrator and every sub-agent.",
        ),
    ] = DEFAULT_MODEL,
    directory: Annotated[
        Path,
        typer.Option("--dir", "-C", help="Project root. The agent is sandboxed to it."),
    ] = Path("."),
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue the interrupted run in this directory.")
    ] = False,
    once: Annotated[
        bool,
        typer.Option("--once", help="Run one task and exit, instead of prompting again."),
    ] = False,
    context_window: Annotated[
        int,
        typer.Option(
            "--context-window",
            help="Tokens before compaction kicks in. Conservative by default; raise it "
            "to match a large-window model (gpt-5.4-mini holds 400000).",
        ),
    ] = DEFAULT_CONTEXT_WINDOW,
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain streamed log — no live panel.")
    ] = False,
) -> None:
    console = Console()
    root = directory.resolve()
    if not root.is_dir():
        console.print(f"[red]{GLYPH['fail']}[/red] not a directory: {root}")
        raise typer.Exit(2)

    opening = " ".join(task).strip() if task else ""
    task_label = opening

    if resume:
        saved = session.load(root)
        if saved is None:
            console.print(
                f"[red]{GLYPH['fail']}[/red] no session to resume in {root} "
                "(.nanocode/session.json)"
            )
            raise typer.Exit(1)
        task_label = saved.get("task") or opening or "(resumed)"
        resumed = session.resume_prompt(saved)
        opening = f"{resumed}\n\n## New instruction from the user\n{opening}" if opening else resumed
    elif once and not opening:
        console.print(f"[red]{GLYPH['fail']}[/red] --once needs a task to run.")
        raise typer.Exit(2)

    try:
        ui = NanocodeUI(console, live=not plain)
        orch = build_orchestrator(
            model=model,
            root=root,
            context_window=context_window,
            on_trace=ui.subagent,
        )
    except ConfigError as exc:
        console.print(f"[red]{GLYPH['fail']}[/red] {exc}")
        raise typer.Exit(2)

    console.print(f"[dim]nanocode · {model} · {root}[/dim]")
    if resume:
        console.print("[dim]resuming from .nanocode/session.json[/dim]")
    if not once:
        console.print("[dim]ask for anything; 'exit' or Ctrl-D when you're done[/dim]")

    raise typer.Exit(_session_loop(orch, ui, console, root, opening, task_label, once))


def _session_loop(
    orch: Orchestrator,
    ui: NanocodeUI,
    console: Console,
    root: Path,
    opening: str,
    task_label: str,
    once: bool,
) -> int:
    """Prompt, run to completion, prompt again.

    One orchestrator instance for the whole session — the conversation carries
    forward between asks, so a follow-up doesn't have to re-establish context.
    """
    state: dict[str, Any] = {}
    exit_code = 0
    ask = opening

    while True:
        if not ask:
            ask = _prompt(console)
            if ask is None:  # exit, or Ctrl-D
                break
            task_label = ask

        # session_log accumulates across the whole session; the summary after
        # each ask should describe only what that ask did.
        offset = len(state.get("session_log") or [])
        try:
            with ui:
                state = _drive(orch, ask, ui)
        except KeyboardInterrupt:
            ui.warn("interrupted — progress saved" + ("" if once else "; ask again or 'exit'"))
            exit_code = 130
        except Exception as exc:  # noqa: BLE001 — surface it, then keep the session alive
            ui.error(f"{type(exc).__name__}: {exc}")
            exit_code = 1
        finally:
            if state:
                session.save(root, state, task_label)

        ui.summary(
            todos=state.get("todos") or [],
            session_log=(state.get("session_log") or [])[offset:],
            log_dir=logs_dir(root),
            resumable=not once,
        )

        if once:
            break
        ask = ""
        exit_code = 0  # a failed ask ends that ask, not the session

    if session.add_to_gitignore_hint(root):
        console.print(
            "\n[dim]tip: add `.nanocode/` to .gitignore — it's scratch state, not source.[/dim]"
        )
    return exit_code


def _prompt(console: Console) -> str | None:
    """Read the next ask. None means the user is done."""
    while True:
        try:
            console.print()
            text = console.input("[bold cyan]› [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            return None
        return text


def _drive(orch: Orchestrator, ask: str, ui: NanocodeUI) -> dict[str, Any]:
    """Stream one ask to completion, rendering each new message as it lands."""
    state: dict[str, Any] = {}
    seen = 0
    for chunk in orch.agent.stream(
        {"messages": [HumanMessage(ask)]},
        config=orch.config,
        stream_mode="values",
    ):
        state = chunk
        ui.set_todos(chunk.get("todos") or [])
        messages = chunk.get("messages") or []
        if seen == 0:
            # Continuing a session: everything already on the thread is history,
            # and was rendered when it first happened. Only show what's new.
            seen = max(len(messages) - 1, 0)
        elif len(messages) < seen:
            # Compaction rewrote the list; re-anchor rather than replay.
            seen = len(messages)
            continue
        for message in messages[seen:]:
            _render(message, ui)
        seen = len(messages)
    return state


def _render(message: Any, ui: NanocodeUI) -> None:
    if isinstance(message, AIMessage):
        for call in message.tool_calls or []:
            if call["name"] not in QUIET_TOOLS:
                ui.tool(call["name"], _describe(call))
        text = getattr(message, "text", None) or message.content
        if isinstance(text, str) and text.strip() and not message.tool_calls:
            ui.assistant(text)
    elif isinstance(message, ToolMessage) and message.status == "error":
        ui.error(_first_line(str(message.content)))


def _describe(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    name = call["name"]
    if name == "delegate":
        return f"{args.get('agent_type', '?')}: {_first_line(str(args.get('task', '')), 90)}"
    if name == "shell":
        return _first_line(str(args.get("command", "")), 90)
    for key in ("path", "pattern", "query"):
        if key in args:
            return str(args[key])
    return ", ".join(f"{k}={_first_line(str(v), 30)}" for k, v in list(args.items())[:2])


def _first_line(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
