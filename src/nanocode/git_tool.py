"""Git awareness.

nanocode could always *run* git — the shell tool allows everything except a
few destructive commands. What it lacked was any reason to. Nothing told it a
repository existed, so it rarely looked, and worked on top of your uncommitted
changes without knowing they were there.

Two halves, matching two different needs:

- **Passive**: `GitContext` refreshes once per ask and its two lines are
  injected into every model call, so the agent always knows the branch and what
  is already modified without spending a tool call to find out.
- **Active**: the `git` tool for `diff`, `log`, `blame` and `status` — the
  investigations worth making deliberately.

Everything degrades to silence. No repository, no git binary, no commits yet:
the lines are simply absent and the tool returns plain text saying so. A
project with no version control must behave exactly as it did before, and must
never be nagged about it.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from langchain_core.tools import BaseTool, tool

from . import prompts

# Long enough for a big repository, short enough that a pathological one costs
# a pause rather than a hang. Missing git context is a small loss; a stalled
# prompt is not.
TIMEOUT = 5
STATUS_TIMEOUT = 2
# Five changed paths is context. Five hundred is noise, and real tokens.
MAX_LISTED_FILES = 8
MAX_OUTPUT_CHARS = 6000

# Read-only by construction. Nothing here can rewrite history, move a branch,
# or touch the working tree — committing is the user's decision, and the
# subcommands that could do damage simply are not reachable.
SUBCOMMANDS = {
    "status": ["status", "--short", "--branch"],
    "diff": ["diff"],
    "staged": ["diff", "--staged"],
    "log": ["log", "--oneline", "--decorate", "-n", "20"],
    "blame": ["blame", "-L", "1,200", "--"],
    "show": ["show", "--stat", "--oneline"],
}


def _run(root: Path, args: list[str], timeout: int = TIMEOUT) -> tuple[bool, str]:
    """Run a git command. Never raises — see the module docstring."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return False, "git is not installed"
    except subprocess.TimeoutExpired:
        return False, f"git took longer than {timeout}s"
    except OSError as exc:
        return False, f"could not run git: {exc}"

    output = (done.stdout or "") + (done.stderr or "")
    return done.returncode == 0, output.strip()


def is_repo(root: str | Path) -> bool:
    """Checked once at startup — a non-git project should pay nothing after."""
    ok, output = _run(Path(root), ["rev-parse", "--is-inside-work-tree"], timeout=STATUS_TIMEOUT)
    return ok and output.strip() == "true"


@dataclass
class GitContext:
    """The branch and dirty-file list, refreshed once per ask.

    Deliberately not refreshed per model call: one ask can drive twenty of
    those, and twenty subprocess spawns to re-answer the same question would be
    pure waste. Nothing external changes mid-ask, and the agent's own edits are
    already tracked in `session_log`.
    """

    root: Path
    enabled: bool = False
    branch: str = ""
    changed: list[str] = field(default_factory=list)

    @classmethod
    def detect(cls, root: str | Path) -> GitContext:
        path = Path(root)
        return cls(root=path, enabled=is_repo(path))

    def refresh(self) -> None:
        if not self.enabled:
            return
        ok, output = _run(self.root, ["status", "--porcelain=v1", "--branch"], STATUS_TIMEOUT)
        if not ok:
            # A slow or broken repo turns the feature off for this ask rather
            # than failing the ask.
            self.branch, self.changed = "", []
            return

        branch, changed = "", []
        for line in output.splitlines():
            if line.startswith("## "):
                branch = _parse_branch(line[3:])
            elif line.strip():
                changed.append(line[3:].strip() if len(line) > 3 else line.strip())
        self.branch, self.changed = branch, changed

    def as_prompt(self) -> str:
        """The two lines injected into every model call. Empty when irrelevant."""
        if not self.enabled or not self.branch:
            return ""
        lines = [f"branch: {self.branch}"]
        if self.changed:
            shown = self.changed[:MAX_LISTED_FILES]
            extra = len(self.changed) - len(shown)
            listed = ", ".join(shown) + (f", …and {extra} more" if extra else "")
            lines.append(f"uncommitted changes: {listed}")
        else:
            lines.append("working tree clean")
        return "## Repository\n" + "\n".join(lines)


def _parse_branch(header: str) -> str:
    """Read the branch out of a porcelain `## ...` line.

    Git has four shapes here, and the obvious split-on-space handles only one:

        ## main...origin/main [ahead 1]   -> main
        ## main                           -> main
        ## No commits yet on main         -> main   (a fresh repo)
        ## HEAD (no branch)               -> detached HEAD

    The third is why this is a function. Splitting naively yields "No", and the
    agent is then told it is on a branch called No — which is exactly the sort
    of quietly wrong context that is worse than none at all.
    """
    header = header.strip()
    if header.startswith("No commits yet on "):
        return header[len("No commits yet on ") :].strip()
    name = header.split("...")[0].split(" ")[0].strip()
    return "detached HEAD" if name == "HEAD" else name


def make_git_tool(root: str | Path) -> BaseTool:
    """Build the read-only `git` tool, closed over the project root."""
    project_root = Path(root)

    @tool("git", description=prompts.GIT_DESCRIPTION)
    def git(command: str, path: str = "", rev: str = "") -> str:
        key = (command or "").strip().lower()
        if key not in SUBCOMMANDS:
            return f"unknown git command {command!r}. Available: {', '.join(sorted(SUBCOMMANDS))}."
        if not is_repo(project_root):
            return "not a git repository — there is no history to read here."

        args = list(SUBCOMMANDS[key])
        if key == "blame":
            if not path:
                return "blame needs a `path`."
            args.append(path)
        elif key == "show":
            args.append(rev or "HEAD")
        elif path:
            args += ["--", path]

        ok, output = _run(project_root, args)
        if not output:
            return "(no output)" if ok else "git reported nothing and failed"
        if not ok and "does not have any commits" in output:
            return "this repository has no commits yet."
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n… (truncated)"
        return output

    return git