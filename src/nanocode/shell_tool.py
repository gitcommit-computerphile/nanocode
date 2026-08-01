"""The shell tool.

Test suites and build tools emit thousands of lines. The tool returns a
truncated tail to the model and writes the full capture to `.nanocode/logs/` —
the same "offload the heavy content, keep a pointer" move the filesystem tools
apply to file contents.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from . import prompts
from .state import event

TAIL_LINES = 40
MAX_TIMEOUT = 600

DESTRUCTIVE = re.compile(
    r"""
    rm\s+(-\w*\s+)*-\w*[rf]        # rm -rf and friends
  | \bgit\s+reset\s+--hard
  | \bgit\s+push\s+.*--force
  | \bgit\s+clean\s+-\w*[fdx]
  | \bmkfs\b | \bdd\s+if=
  | :\(\)\s*\{                      # fork bomb
  | \b(shutdown|reboot|halt)\b
  | >\s*/dev/(sd|nvme)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def logs_dir(root: Path) -> Path:
    path = Path(root) / ".nanocode" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_log(root: Path, prefix: str, content: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = logs_dir(root) / f"{prefix}-{stamp}.log"
    n = 1
    while path.exists():
        path = logs_dir(root) / f"{prefix}-{stamp}-{n}.log"
        n += 1
    path.write_text(content, encoding="utf-8", errors="replace")
    return path


def tail(text: str, lines: int = TAIL_LINES) -> str:
    split = text.splitlines()
    if len(split) <= lines:
        return text
    return "…\n" + "\n".join(split[-lines:])


def make_shell_tool(root: str | Path) -> BaseTool:
    """Build the shell tool bound to one project root."""
    project_root = Path(root).resolve()

    @tool("shell", description=prompts.SHELL_DESCRIPTION)
    def shell(
        command: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        timeout: int = 120,
    ) -> Command:
        if DESTRUCTIVE.search(command):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"blocked: {command!r} looks destructive and was not run. "
                            "Ask the user to run it themselves if it is genuinely needed.",
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                    ]
                }
            )

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=project_root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=min(max(timeout, 1), MAX_TIMEOUT),
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            output = f"{_as_text(partial)}\n[timed out after {timeout}s]"
            code = -1
        except OSError as exc:
            output, code = f"failed to run command: {exc}", -1

        log_path = write_log(project_root, "run", f"$ {command}\n[exit {code}]\n\n{output}")
        summary = tail(output).strip() or "(no output)"
        detail = f"{command} -> exit {code}"
        return Command(
            update={
                "session_log": [event("shell", detail)],
                "messages": [
                    ToolMessage(
                        f"[exit {code}]\n{summary}\n\n(full output: {log_path.as_posix()})",
                        tool_call_id=tool_call_id,
                        status="success" if code == 0 else "error",
                    )
                ],
            }
        )

    return shell


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
