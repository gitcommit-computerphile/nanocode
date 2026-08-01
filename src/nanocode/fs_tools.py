"""Filesystem tools.

Five signatures — ls, read_file, write_file, edit_file, grep (plus glob) — over
two interchangeable backends. Early phases run against an in-memory dict with
nothing sandboxed; the real-disk backend adds the project-root sandbox and path
checks. The agent logic is built once, against the interface.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Protocol

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from . import prompts
from .state import event

MAX_GREP_MATCHES = 60
MAX_GLOB_RESULTS = 100
MAX_LINE_LENGTH = 400
SKIP_DIRS = {".git", ".nanocode", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build", ".ruff_cache"}


class SandboxError(Exception):
    """A path resolved outside the project root."""


class FileSystem(Protocol):
    """The interface every tool is written against."""

    def ls(self, path: str) -> list[str]: ...
    def read(self, path: str) -> str: ...
    def write(self, path: str, content: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    def walk(self) -> list[str]:
        """Every file path in the project, relative and posix-style."""
        ...


def _normalize(path: str) -> PurePosixPath:
    """Reject absolute paths and traversal before they reach a backend."""
    cleaned = (path or ".").replace("\\", "/").strip()
    pure = PurePosixPath(cleaned)
    if pure.is_absolute() or (len(cleaned) > 1 and cleaned[1] == ":"):
        raise SandboxError(f"path must be relative to the project root: {path}")
    parts = [p for p in pure.parts if p not in (".", "")]
    if any(p == ".." for p in parts):
        raise SandboxError(f"path escapes the project root: {path}")
    return PurePosixPath(*parts) if parts else PurePosixPath(".")


class DiskFileSystem:
    """Real disk, hard-blocked to the project root.

    Any path that resolves outside the root raises — there is no escape hatch
    and no confirmation prompt. Symlinks are resolved before the check.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise SandboxError(f"project root is not a directory: {self.root}")

    def resolve(self, path: str) -> Path:
        rel = _normalize(path)
        candidate = (self.root / str(rel)).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise SandboxError(f"path escapes the project root: {path}")
        return candidate

    def ls(self, path: str = ".") -> list[str]:
        target = self.resolve(path)
        if not target.is_dir():
            raise FileNotFoundError(f"not a directory: {path}")
        out = []
        for child in sorted(target.iterdir(), key=lambda c: (c.is_file(), c.name)):
            out.append(f"{child.name}/" if child.is_dir() else child.name)
        return out

    def read(self, path: str) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def write(self, path: str, content: str) -> None:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")

    def exists(self, path: str) -> bool:
        try:
            return self.resolve(path).exists()
        except SandboxError:
            return False

    def walk(self) -> list[str]:
        out = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.relative_to(self.root).parts):
                continue
            out.append(p.relative_to(self.root).as_posix())
        return sorted(out)


class VirtualFileSystem:
    """An in-memory dict. Safe to run anywhere — nothing touches a real disk."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files: dict[str, str] = dict(files or {})

    def ls(self, path: str = ".") -> list[str]:
        prefix = "" if str(_normalize(path)) == "." else f"{_normalize(path)}/"
        names: set[str] = set()
        for key in self.files:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            head, _, tail = rest.partition("/")
            names.add(f"{head}/" if tail else head)
        if not names and prefix:
            raise FileNotFoundError(f"not a directory: {path}")
        return sorted(names)

    def read(self, path: str) -> str:
        key = str(_normalize(path))
        if key not in self.files:
            raise FileNotFoundError(f"no such file: {path}")
        return self.files[key]

    def write(self, path: str, content: str) -> None:
        self.files[str(_normalize(path))] = content

    def exists(self, path: str) -> bool:
        return str(_normalize(path)) in self.files

    def walk(self) -> list[str]:
        return sorted(self.files)


def _clip(line: str) -> str:
    return line if len(line) <= MAX_LINE_LENGTH else line[:MAX_LINE_LENGTH] + " …[clipped]"


# Filesystem problems the model can recover from by choosing a different path.
# They must come back as tool output, never as an exception — an exception
# would tear down the whole run over one bad guess.
RECOVERABLE = (SandboxError, FileNotFoundError, NotADirectoryError, IsADirectoryError, OSError, ValueError)


def make_fs_tools(fs: FileSystem, *, writable: bool = True) -> list[BaseTool]:
    """Build the filesystem tool set bound to one backend."""

    @tool("ls", description=prompts.LS_DESCRIPTION)
    def ls_tool(path: str = ".") -> str:
        try:
            entries = fs.ls(path)
        except RECOVERABLE as exc:
            return f"error: {exc}"
        return "\n".join(entries) if entries else "(empty directory)"

    @tool("read_file", description=prompts.READ_FILE_DESCRIPTION)
    def read_file(path: str, offset: int = 0, limit: int = 400) -> str:
        try:
            lines = fs.read(path).splitlines()
        except RECOVERABLE as exc:
            return f"error: {exc}"
        if offset >= len(lines) and lines:
            return f"{path} has {len(lines)} lines; offset {offset} is past the end."
        window = lines[offset : offset + limit]
        if not window:
            return f"{path} is empty."
        body = "\n".join(
            f"{i:>6}\t{_clip(text)}" for i, text in enumerate(window, start=offset + 1)
        )
        shown = offset + len(window)
        if shown < len(lines):
            body += (
                f"\n\n…truncated. Showing lines {offset + 1}-{shown} of {len(lines)}. "
                f"Call read_file again with offset={shown} to continue."
            )
        return body

    @tool("grep", description=prompts.GREP_DESCRIPTION)
    def grep(pattern: str, glob: str = "**/*") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"invalid regular expression: {exc}"
        hits: list[str] = []
        for path in fs.walk():
            if not _glob_match(path, glob):
                continue
            try:
                content = fs.read(path)
            except RECOVERABLE:
                continue
            for lineno, text in enumerate(content.splitlines(), start=1):
                if regex.search(text):
                    hits.append(f"{path}:{lineno}: {_clip(text.strip())}")
                    if len(hits) >= MAX_GREP_MATCHES:
                        hits.append(
                            f"…stopped at {MAX_GREP_MATCHES} matches. Narrow the pattern or the glob."
                        )
                        return "\n".join(hits)
        return "\n".join(hits) if hits else f"no matches for {pattern!r} in {glob}"

    @tool("glob", description=prompts.GLOB_DESCRIPTION)
    def glob_tool(pattern: str) -> str:
        matches = [p for p in fs.walk() if _glob_match(p, pattern)]
        if not matches:
            return f"no files match {pattern!r}"
        head = matches[:MAX_GLOB_RESULTS]
        out = "\n".join(head)
        if len(matches) > len(head):
            out += f"\n…and {len(matches) - len(head)} more"
        return out

    tools: list[BaseTool] = [ls_tool, read_file, grep, glob_tool]
    if not writable:
        return tools

    @tool("write_file", description=prompts.WRITE_FILE_DESCRIPTION)
    def write_file(
        path: str,
        content: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        try:
            existed = fs.exists(path)
            fs.write(path, content)
        except RECOVERABLE as exc:
            return _tool_error(tool_call_id, f"could not write {path}: {exc}")
        verb = "overwrote" if existed else "created"
        line_count = len(content.splitlines())
        detail = f"{path}: {verb} ({line_count} lines)"
        return Command(
            update={
                "session_log": [event("file_edit", detail)],
                "messages": [ToolMessage(f"{verb} {path} ({line_count} lines)", tool_call_id=tool_call_id)],
            }
        )

    @tool("edit_file", description=prompts.EDIT_FILE_DESCRIPTION)
    def edit_file(
        path: str,
        old_string: str,
        new_string: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        try:
            content = fs.read(path)
        except RECOVERABLE as exc:
            return _tool_error(tool_call_id, f"could not read {path}: {exc}")
        occurrences = content.count(old_string)
        if occurrences == 0:
            return _tool_error(
                tool_call_id,
                f"old_string not found in {path}. Read the file and copy the exact text, "
                "including indentation.",
            )
        if occurrences > 1:
            return _tool_error(
                tool_call_id,
                f"old_string appears {occurrences} times in {path}; it must be unique. "
                "Include more surrounding context to disambiguate.",
            )
        try:
            fs.write(path, content.replace(old_string, new_string, 1))
        except RECOVERABLE as exc:
            return _tool_error(tool_call_id, f"could not write {path}: {exc}")
        removed = len(old_string.splitlines()) or 1
        added = len(new_string.splitlines()) or 1
        detail = f"{path}: +{added} -{removed}"
        return Command(
            update={
                "session_log": [event("file_edit", detail)],
                "messages": [ToolMessage(f"edited {path} (+{added} -{removed})", tool_call_id=tool_call_id)],
            }
        )

    return [*tools, write_file, edit_file]


def _tool_error(tool_call_id: str, message: str) -> Command:
    return Command(update={"messages": [ToolMessage(message, tool_call_id=tool_call_id, status="error")]})


def _glob_match(path: str, pattern: str) -> bool:
    """Match a posix path against a glob, treating ``**`` as any depth."""
    if pattern in ("**/*", "**", "*"):
        return True
    if fnmatch.fnmatch(path, pattern):
        return True
    # fnmatch has no notion of `**`; approximate by also trying the pattern
    # with the recursive prefix stripped, so "**/*.py" matches "a.py".
    if pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]):
        return True
    return False
