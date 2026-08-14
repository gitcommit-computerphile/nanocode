"""Session persistence.

This is the actual payoff of keeping state off the message list: the process
can die mid-task and resume without replaying a transcript. On resume the
orchestrator re-orients from a compact record of what happened — the same way
you'd read a commit log rather than replay every keystroke.

Two different lifetimes live under `.nanocode/`, and the split matters:

    sessions/<id>.json   one file per run, never overwritten. Scoped to a task:
                         what was asked, the plan, and what actually happened.
    constraints.md       project-scoped and permanent. Rules the user gave that
                         outlive any one task. Plain text, so you can write it
                         yourself before the agent ever runs.

An earlier version wrote a single `session.json` and overwrote it on every ask,
which meant starting work without `--resume` destroyed the previous record. It
is still read as a fallback so old projects resume, but nothing writes it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import Event, Todo, format_todos

LEGACY_SESSION_FILE = "session.json"
SESSIONS_DIR = "sessions"
CONSTRAINTS_FILE = "constraints.md"
RESUME_LOG_LINES = 40
KEEP_SESSIONS = 20

CONSTRAINTS_HEADER = """\
# Constraints

Standing rules for this project. They are injected into the agent's context on
every turn and survive across sessions. Edit this file by hand if you like —
one rule per line, starting with a dash.
"""

_RULE = re.compile(r"^\s*[-*]\s+(?P<rule>.+?)\s*$")


def nanocode_dir(root: str | Path) -> Path:
    path = Path(root).resolve() / ".nanocode"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sessions_dir(root: str | Path) -> Path:
    path = nanocode_dir(root) / SESSIONS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_session_id(root: str | Path | None = None) -> str:
    """A sortable id, so the newest session is the last one alphabetically.

    Given a root, it also avoids colliding with an id already on disk — two
    runs in the same second, or a `/clear` moments after startup.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if root is None:
        return stamp
    candidate, suffix = stamp, 1
    while session_path(root, candidate).exists():
        suffix += 1
        candidate = f"{stamp}-{suffix}"
    return candidate


def session_path(root: str | Path, session_id: str) -> Path:
    return sessions_dir(root) / f"{session_id}.json"


# -- sessions -------------------------------------------------------------


def save(root: str | Path, state: dict[str, Any], task: str, session_id: str) -> Path:
    """Write one session's durable record. Never writes credentials.

    Each session owns its own file, so saving can never destroy the record of
    an earlier run — the failure mode that made `--resume` unreliable.
    """
    path = session_path(root, session_id)
    payload = {
        "id": session_id,
        "task": task,
        "todos": state.get("todos") or [],
        "session_log": state.get("session_log") or [],
        "cwd": str(Path(root).resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def list_sessions(root: str | Path) -> list[Path]:
    """Every saved session in this project, newest last."""
    try:
        return sorted(sessions_dir(root).glob("*.json"))
    except OSError:
        return []


def load(root: str | Path, session_id: str | None = None) -> dict[str, Any] | None:
    """Read a session — the most recent one unless an id is given."""
    if session_id is not None:
        return _read(session_path(root, session_id))

    for path in reversed(list_sessions(root)):
        if (saved := _read(path)) is not None:
            return saved

    # Projects written by the single-file version still resume.
    return _read(nanocode_dir(root) / LEGACY_SESSION_FILE)


def prune_sessions(root: str | Path, keep: int = KEEP_SESSIONS) -> int:
    """Drop the oldest session files. Returns how many were removed."""
    stale = list_sessions(root)[:-keep] if keep > 0 else []
    removed = 0
    for path in stale:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def has_unfinished_work(saved: dict[str, Any] | None) -> bool:
    """True when a session stopped with steps still outstanding.

    This is what decides whether starting up should pick the work back up, so
    it deliberately says no to a session that simply ended cleanly — finishing
    a task should not haunt the next one.
    """
    if not saved:
        return False
    todos: list[Todo] = saved.get("todos") or []
    return any(t.get("status") != "completed" for t in todos)


def resume_prompt(saved: dict[str, Any]) -> str:
    """Rebuild the orchestrator's starting context from the durable record.

    Not from raw message history — from `todos` plus `session_log`, which is
    what makes resuming cheap regardless of how long the original run was.
    Constraints are deliberately absent: they arrive through state and are
    recited every turn, so repeating them here would just duplicate them.
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


# -- constraints ----------------------------------------------------------


def constraints_path(root: str | Path) -> Path:
    return nanocode_dir(root) / CONSTRAINTS_FILE


def load_constraints(root: str | Path) -> list[str]:
    """Read the project's standing rules. Missing or unreadable means none."""
    path = constraints_path(root)
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [m["rule"] for line in text.splitlines() if (m := _RULE.match(line))]


def save_constraints(root: str | Path, constraints: list[str]) -> Path:
    """Write the standing rules as editable markdown, one per line."""
    path = constraints_path(root)
    body = "\n".join(f"- {c.strip()}" for c in constraints if c.strip())
    path.write_text(f"{CONSTRAINTS_HEADER}\n{body}\n" if body else CONSTRAINTS_HEADER, encoding="utf-8")
    return path


# -- misc -----------------------------------------------------------------


def add_to_gitignore_hint(root: str | Path) -> bool:
    """True when `.nanocode/` is not yet ignored and probably should be."""
    gitignore = Path(root).resolve() / ".gitignore"
    if not gitignore.is_file():
        return False
    try:
        return ".nanocode" not in gitignore.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None
