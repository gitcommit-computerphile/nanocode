"""Session persistence.

This is the actual payoff of keeping state off the message list: the process
can die mid-task and resume without replaying a transcript. On resume the
orchestrator re-orients from a compact record of what happened — the same way
you'd read a commit log rather than replay every keystroke.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import Event, Todo, format_todos

SESSION_FILE = "session.json"
RESUME_LOG_LINES = 40


def nanocode_dir(root: str | Path) -> Path:
    path = Path(root).resolve() / ".nanocode"
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_path(root: str | Path) -> Path:
    return nanocode_dir(root) / SESSION_FILE


def save(root: str | Path, state: dict[str, Any], task: str) -> Path:
    """Write todos + session_log to disk. Never writes credentials."""
    path = session_path(root)
    payload = {
        "task": task,
        "todos": state.get("todos") or [],
        "session_log": state.get("session_log") or [],
        "cwd": str(Path(root).resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(root: str | Path) -> dict[str, Any] | None:
    path = session_path(root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def resume_prompt(saved: dict[str, Any]) -> str:
    """Rebuild the orchestrator's starting context from the durable record.

    Not from raw message history — from `todos` plus `session_log`, which is
    what makes resuming cheap regardless of how long the original run was.
    """
    todos: list[Todo] = saved.get("todos") or []
    log: list[Event] = saved.get("session_log") or []
    recent = log[-RESUME_LOG_LINES:]

    lines = [
        "You are resuming an interrupted run. This is the durable record of it —",
        "there is no transcript, and you should not ask for one.",
        "",
        f"## Original task\n{saved.get('task', '(not recorded)')}",
        "",
        format_todos(todos),
    ]
    if recent:
        skipped = len(log) - len(recent)
        header = f"\n## What already happened ({f'last {len(recent)} of {len(log)}' if skipped else f'{len(log)} events'})"
        lines.append(header)
        lines += [f"- [{e.get('kind')}] {e.get('detail')}" for e in recent]

    lines += [
        "",
        "Every file edit listed above is already on disk. Re-read any file whose",
        "current contents you need. Pick up at the first step that is not complete,",
        "verify it is actually still needed, and continue to the end of the plan.",
    ]
    return "\n".join(lines)


def add_to_gitignore_hint(root: str | Path) -> bool:
    """True when `.nanocode/` is not yet ignored and probably should be."""
    gitignore = Path(root).resolve() / ".gitignore"
    if not gitignore.is_file():
        return False
    try:
        return ".nanocode" not in gitignore.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
