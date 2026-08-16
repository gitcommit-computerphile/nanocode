"""Entrypoint — an interactive prompt loop over one orchestrator instance.

Run it, ask for something, keep asking. Each ask runs the same loop to
completion and then prompts again; the conversation carries forward, so "now
make it return JSON" knows what "it" is. `--once` runs a single task and exits
— the same loop, minus the prompt-again step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from rich.console import Console

from . import commands, session
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
    subagent_model: Annotated[
        str,
        typer.Option(
            "--subagent-model",
            help="provider:name for sub-agents only. They grep, read and run tests — "
            "execution, not judgment — so a cheaper model usually suffices. "
            "Defaults to --model.",
        ),
    ] = "",
    directory: Annotated[
        Path,
        typer.Option("--dir", "-C", help="Project root. The agent is sandboxed to it."),
    ] = Path("."),
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Force a pick-up of the last run, even if it finished. Usually unnecessary "
            "— unfinished work is picked up automatically.",
        ),
    ] = False,
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh",
            help="Start clean: ignore unfinished work and the project's constraints.",
        ),
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

    if once and not opening and not resume:
        console.print(f"[red]{GLYPH['fail']}[/red] --once needs a task to run.")
        raise typer.Exit(2)

    saved = None if fresh else session.load(root)
    constraints = [] if fresh else session.load_constraints(root)

    # Unfinished work is picked up without being asked for. The flag that
    # rescues you used to be one you had to know about before you needed it,
    # which is the wrong way round; `--fresh` opts out instead.
    picked_up = False
    if resume:
        if saved is None:
            console.print(
                f"[red]{GLYPH['fail']}[/red] nothing to resume in {root} "
                "(no runs recorded under .nanocode/sessions/)"
            )
            raise typer.Exit(1)
        picked_up = True
    elif not once and session.has_unfinished_work(saved):
        picked_up = True

    if picked_up:
        task_label = saved.get("task") or opening or "(resumed)"
        briefing = session.resume_prompt(saved)
        opening = (
            f"{briefing}\n\n## New instruction from the user\n{opening}" if opening else briefing
        )

    try:
        ui = NanocodeUI(console, live=not plain)
        # Resolving the model can take a moment, and git detection runs here
        # too — the one startup pause the user would otherwise stare at.
        with console.status("[dim]starting…[/dim]", spinner="dots"):
            orch = build_orchestrator(
                model=model,
                subagent_model=subagent_model or None,
                root=root,
                context_window=context_window,
                on_trace=ui.subagent,
                on_retry=ui.warn,
                on_compact=ui.compacting,
            )
    except ConfigError as exc:
        console.print(f"[red]{GLYPH['fail']}[/red] {exc}")
        raise typer.Exit(2)

    notes: list[str] = []
    if picked_up:
        todos = saved.get("todos") or []
        done = sum(1 for t in todos if t.get("status") == "completed")
        notes.append(
            f"picking up the last run — {done}/{len(todos)} steps done (--fresh to start clean)"
        )
    if constraints:
        plural = "" if len(constraints) == 1 else "s"
        notes.append(f"{len(constraints)} standing constraint{plural} loaded")

    if orch.git is not None and orch.git.enabled:
        orch.git.refresh()
    ui.header(
        model=model,
        root=root,
        branch=orch.git.branch if orch.git else "",
        changed=len(orch.git.changed) if orch.git else 0,
        subagent_model=subagent_model,
        note=" · ".join(notes),
    )
    if not once:
        console.print(
            "\n[dim]  ask for anything · /help for commands · 'exit' when you're done[/dim]"
        )

    raise typer.Exit(
        _session_loop(
            orch,
            ui,
            console,
            root,
            opening,
            task_label,
            once,
            constraints=constraints,
        )
    )


def _session_loop(
    orch: Orchestrator,
    ui: NanocodeUI,
    console: Console,
    root: Path,
    opening: str,
    task_label: str,
    once: bool,
    constraints: list[str] | None = None,
) -> int:
    """Prompt, run to completion, prompt again.

    One orchestrator instance for the whole session — the conversation carries
    forward between asks, so a follow-up doesn't have to re-establish context.
    """
    ctx = commands.CommandContext(
        console=console,
        ui=ui,
        root=root,
        orch=orch,
        constraints=list(constraints or []),
    )
    state: dict[str, Any] = {}
    exit_code = 0
    ask = opening
    session_id = session.new_session_id(root)

    # Seed the project's standing rules once. From here the checkpointer keeps
    # them, and `write_constraints` is what changes them.
    seed: dict[str, Any] | None = {"constraints": list(ctx.constraints)} if ctx.constraints else None

    while True:
        if not ask:
            ui.status(ctx.orch.usage, ctx.orch.context_window)
            ask = _prompt(console)
            if ask is None:  # exit, or Ctrl-D
                break
            task_label = ask

        if commands.is_command(ask):
            commands.dispatch(ask, ctx)
            if ctx.cleared:
                # A fresh thread means fresh state, and a fresh record to write
                # it to — the pre-clear session file stays exactly as it was.
                ctx.cleared = False
                state = {}
                session_id = session.new_session_id(root)
                seed = {"constraints": list(ctx.constraints)} if ctx.constraints else None
            if once:
                break
            ask = ""
            continue

        # One git call per ask, not per model call — the result is then read
        # from cache by every turn's recitation.
        if ctx.orch.git is not None:
            ctx.orch.git.refresh()

        # session_log accumulates across the whole session; the summary after
        # each ask should describe only what that ask did. `plan_before` serves
        # the same purpose for todos, which also survive between asks — without
        # it, a follow-up question inherits the previous ask's plan and gets
        # reported as though it had done the work.
        offset = len(state.get("session_log") or [])
        plan_before = state.get("todos") or []
        started = time.monotonic()
        try:
            with ui:
                state = _drive(ctx.orch, ask, ui, seed=seed, plan_before=plan_before)
            seed = None
        except KeyboardInterrupt:
            ui.warn("interrupted — progress saved" + ("" if once else "; ask again or 'exit'"))
            exit_code = 130
        except Exception as exc:  # noqa: BLE001 — surface it, then keep the session alive
            ui.error(f"{type(exc).__name__}: {exc}")
            exit_code = 1
        finally:
            if state:
                # Same id every ask, so one run is one file — and no run ever
                # overwrites the record of another.
                session.save(root, state, task_label, session_id)

        todos_now = state.get("todos") or []
        new_events = (state.get("session_log") or [])[offset:]
        # An ask that neither touched the plan nor recorded anything was a
        # conversational turn, whatever stale plan happens to be in state.
        if new_events or todos_now != plan_before:
            ui.summary(
                todos=todos_now,
                session_log=new_events,
                log_dir=logs_dir(root),
                resumable=not once,
                constraints=state.get("constraints") or [],
                seconds=time.monotonic() - started,
            )

        if once:
            break
        ask = ""
        exit_code = 0  # a failed ask ends that ask, not the session

    session.prune_sessions(root)
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


def _drive(
    orch: Orchestrator,
    ask: str,
    ui: NanocodeUI,
    seed: dict[str, Any] | None = None,
    plan_before: list | None = None,
) -> dict[str, Any]:
    """Stream one ask to completion, rendering each new message as it lands.

    `seed` sets state fields alongside the ask — used once at startup to load
    the project's constraints in from disk. `plan_before` is the plan as it
    stood when the ask began, so a plan left over from an earlier task isn't
    redisplayed under an answer that never touched it.
    """
    state: dict[str, Any] = {}
    plan_before = plan_before or []
    seen = 0
    # The gap between hitting enter and the first token is the longest silence
    # in the whole loop; it gets an indicator from the outset.
    ui.begin_thinking()
    for mode, payload in orch.agent.stream(
        {"messages": [HumanMessage(ask)], **(seed or {})},
        config=orch.config,
        # "values" gives whole-state snapshots (what to render and record);
        # "messages" gives token deltas as they arrive (what to stream).
        stream_mode=["values", "messages"],
    ):
        if mode == "messages":
            _stream_token(payload, ui)
            continue

        state = payload
        # Tokens for the reply just finished are committed before anything else
        # prints, so a tool line can't land in the middle of a sentence.
        streamed = ui.end_stream()
        # Only once this ask has touched the plan. Todos survive between asks,
        # so otherwise a follow-up question renders a finished plan from an
        # earlier task underneath an unrelated answer.
        todos = payload.get("todos") or []
        if todos != plan_before:
            ui.set_todos(todos)
        messages = payload.get("messages") or []
        if seen == 0:
            # Continuing a session: everything already on the thread is history,
            # and was rendered when it first happened. Only show what's new.
            seen = max(len(messages) - 1, 0)
        elif len(messages) < seen:
            # Compaction rewrote the list; re-anchor rather than replay.
            seen = len(messages)
            continue
        for message in messages[seen:]:
            _render(message, ui, orch, already_shown=streamed)
        seen = len(messages)
    ui.end_stream()
    ui.end_thinking()
    return state


def _stream_token(payload: Any, ui: NanocodeUI) -> None:
    """Render one token delta, if it belongs to the orchestrator.

    A sub-agent runs its own graph inside the delegate tool, and LangChain
    propagates the streaming callbacks into it — so its private reasoning
    arrives here too, tagged with `langgraph_node == "tools"`. Showing it would
    break the isolation the whole sub-agent design rests on: the orchestrator
    is supposed to see one summary, and so is the user.
    """
    try:
        chunk, metadata = payload
    except (TypeError, ValueError):
        return
    if (metadata or {}).get("langgraph_node") != "model":
        return
    if not isinstance(chunk, AIMessageChunk):
        return
    ui.stream(_text_of(chunk))


def _text_of(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _render(message: Any, ui: NanocodeUI, orch: Orchestrator, already_shown: bool = False) -> None:
    if isinstance(message, AIMessage):
        # Usage lands on the assembled message, not the deltas. is_context
        # marks this as an orchestrator call, so it sets the context gauge.
        orch.usage.record(message, is_context=True)
        for call in message.tool_calls or []:
            if call["name"] not in QUIET_TOOLS:
                # Opened here, closed when its result lands — that pairing is
                # what produces both the live spinner and the duration.
                ui.begin_tool(call["id"], call["name"], _describe(call))
        text = getattr(message, "text", None) or message.content
        if not already_shown and isinstance(text, str) and text.strip() and not message.tool_calls:
            ui.assistant(text)
    elif isinstance(message, ToolMessage):
        ui.end_tool(message.tool_call_id)
        if message.status == "error":
            ui.error(_first_line(str(message.content)))


def _describe(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    name = call["name"]
    if name == "delegate":
        return f"{args.get('agent_type', '?')}: {_first_line(str(args.get('task', '')), 90)}"
    if name == "shell":
        return _first_line(str(args.get("command", "")), 90)
    if name == "write_constraints":
        rules = args.get("constraints") or []
        return _first_line(rules[-1], 90) if rules else "(cleared)"
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
