from __future__ import annotations

import sys

import pytest

from nanocode.shell_tool import DESTRUCTIVE, make_shell_tool, tail


def py(code: str) -> str:
    """A python one-liner as a shell command.

    The interpreter path is quoted because it routinely contains spaces, and
    cmd.exe would otherwise split it at the first one.
    """
    return f'"{sys.executable}" -c "{code}"'


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf build",
        "RM -RF .",
        "git reset --hard HEAD~3",
        "git push origin main --force",
        "git clean -fdx",
        "sudo shutdown now",
        "dd if=/dev/zero of=/dev/sda",
    ],
)
def test_destructive_commands_are_blocked(tmp_path, call, message, command):
    result = call(make_shell_tool(tmp_path), command=command)
    msg = message(result)
    assert msg.status == "error"
    assert "blocked" in msg.content
    assert "session_log" not in result.update  # nothing ran, nothing recorded


@pytest.mark.parametrize(
    "command",
    ["pytest -q", "git status", "ls -la", "npm run build", "git push origin main"],
)
def test_ordinary_commands_are_not_blocked(command):
    assert DESTRUCTIVE.search(command) is None


def test_output_is_captured_and_the_full_log_is_written(tmp_path, call, message):
    result = call(make_shell_tool(tmp_path), command=py("print('marker')"))
    msg = message(result)
    assert "marker" in msg.content
    assert "[exit 0]" in msg.content

    logs = list((tmp_path / ".nanocode" / "logs").glob("run-*.log"))
    assert len(logs) == 1
    assert "marker" in logs[0].read_text(encoding="utf-8")


def test_long_output_is_truncated_to_a_tail_but_logged_whole(tmp_path, call, message):
    result = call(make_shell_tool(tmp_path), command=py("for i in range(500): print('line', i)"))
    content = message(result).content
    assert "line 499" in content
    assert "line 0\n" not in content  # early output stayed on disk

    log = next((tmp_path / ".nanocode" / "logs").glob("run-*.log"))
    assert "line 0" in log.read_text(encoding="utf-8")


def test_failing_command_reports_error_status_and_still_logs(tmp_path, call, message):
    result = call(make_shell_tool(tmp_path), command=py("import sys; sys.exit(3)"))
    msg = message(result)
    assert msg.status == "error"
    assert "[exit 3]" in msg.content
    assert result.update["session_log"][0]["kind"] == "shell"


def test_shell_runs_in_the_project_root(tmp_path, call, message):
    (tmp_path / "sentinel.txt").write_text("here", encoding="utf-8")
    result = call(
        make_shell_tool(tmp_path), command=py("import os; print(os.path.exists('sentinel.txt'))")
    )
    assert "True" in message(result).content


def test_tail_keeps_the_end():
    assert tail("\n".join(str(i) for i in range(100)), lines=3) == "…\n97\n98\n99"
    assert tail("short") == "short"
