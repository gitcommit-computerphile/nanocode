"""Git awareness — passive context, and the read-only tool.

These build real repositories in tmp_path rather than mocking subprocess. Git's
actual output format is the thing being parsed, so a mock would only confirm
that the parser agrees with my guess about it.

The cases that matter most are the absent ones: no repository, no git binary,
no commits. A project without version control has to behave exactly as it did
before, and must never be nagged about it.
"""

from __future__ import annotations

import subprocess

import pytest

from nanocode.git_tool import GitContext, is_repo, make_git_tool
from nanocode.orchestrator import build_orchestrator

GIT_MISSING = subprocess.run(
    ["git", "--version"], capture_output=True
).returncode != 0
pytestmark = pytest.mark.skipif(GIT_MISSING, reason="git is not installed")


def _git(root, *args):
    subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        check=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
             "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ.get("PATH", "")},
    )


@pytest.fixture
def repo(tmp_path):
    """A real repository with one commit and one uncommitted change."""
    _git(tmp_path, "init", "-b", "main")
    (tmp_path / "app.py").write_text("def handler():\n    return 1\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "add the handler")
    (tmp_path / "app.py").write_text("def handler():\n    return 2\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def bare(tmp_path_factory):
    """A plain directory — no version control at all.

    Its own root, not a subdirectory of `repo`'s: anything under a repository
    is still inside that work tree, so a nested "bare" directory would report
    itself as a repo and quietly invert the test.
    """
    path = tmp_path_factory.mktemp("plain")
    (path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return path


# -- detection ------------------------------------------------------------


def test_a_repository_is_detected(repo):
    assert is_repo(repo) is True


def test_a_plain_directory_is_not(bare):
    assert is_repo(bare) is False


def test_context_is_disabled_outside_a_repository(bare):
    git = GitContext.detect(bare)
    git.refresh()

    assert git.enabled is False
    assert git.as_prompt() == "", "a non-git project should see nothing about git"


def test_a_missing_git_binary_is_not_an_error(bare, monkeypatch):
    """Not everyone has git installed. That is not a failure condition."""
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    git = GitContext.detect(bare)
    git.refresh()

    assert git.enabled is False
    assert git.as_prompt() == ""


# -- the two injected lines ----------------------------------------------


def test_the_branch_and_dirty_files_reach_the_prompt(repo):
    git = GitContext.detect(repo)
    git.refresh()

    text = git.as_prompt()
    assert "branch: main" in text
    assert "app.py" in text
    assert "uncommitted changes" in text


def test_a_clean_tree_says_so(repo):
    _git(repo, "checkout", "--", "app.py")
    git = GitContext.detect(repo)
    git.refresh()

    assert "working tree clean" in git.as_prompt()
    assert "uncommitted changes" not in git.as_prompt()


def test_a_long_dirty_list_is_capped(repo):
    """A messy repo shouldn't flood the context with filenames."""
    for i in range(40):
        (repo / f"file{i}.py").write_text(f"X = {i}\n", encoding="utf-8")
    _git(repo, "add", "-A")

    git = GitContext.detect(repo)
    git.refresh()
    text = git.as_prompt()

    assert "…and" in text and "more" in text
    assert text.count(".py") <= 12, "the list should be trimmed, not dumped"


def test_a_fresh_repository_with_no_commits_still_reports_a_branch(tmp_path):
    """Git says "## No commits yet on main" here, not "## main".

    Splitting that naively yields a branch called "No" — quietly wrong context,
    which is worse than none.
    """
    _git(tmp_path, "init", "-b", "main")
    (tmp_path / "new.py").write_text("X = 1\n", encoding="utf-8")

    git = GitContext.detect(tmp_path)
    git.refresh()
    assert "branch: main" in git.as_prompt()


@pytest.mark.parametrize(
    "header,expected",
    [
        ("main...origin/main [ahead 1]", "main"),
        ("main", "main"),
        ("No commits yet on main", "main"),
        ("No commits yet on feature/x", "feature/x"),
        ("HEAD (no branch)", "detached HEAD"),
    ],
)
def test_every_branch_header_shape_is_parsed(header, expected):
    from nanocode.git_tool import _parse_branch

    assert _parse_branch(header) == expected


# -- the tool -------------------------------------------------------------


def test_diff_shows_the_uncommitted_change(repo):
    out = make_git_tool(repo).invoke({"command": "diff"})
    assert "return 2" in out and "return 1" in out


def test_log_shows_history(repo):
    assert "add the handler" in make_git_tool(repo).invoke({"command": "log"})


def test_blame_reports_who_touched_a_line(repo):
    out = make_git_tool(repo).invoke({"command": "blame", "path": "app.py"})
    assert "handler" in out


def test_blame_without_a_path_says_what_is_missing(repo):
    assert "needs a `path`" in make_git_tool(repo).invoke({"command": "blame"})


def test_the_tool_outside_a_repository_returns_text_not_an_exception(bare):
    """Same rule as the filesystem tools: never raise out of a tool."""
    out = make_git_tool(bare).invoke({"command": "status"})
    assert "not a git repository" in out


def test_an_unknown_subcommand_is_reported_with_the_valid_ones(repo):
    out = make_git_tool(repo).invoke({"command": "push"})
    assert "unknown git command" in out
    assert "blame" in out and "diff" in out


@pytest.mark.parametrize("command", ["commit", "push", "reset", "checkout", "clean", "rebase"])
def test_history_changing_commands_are_unreachable(repo, command):
    """Committing is the user's decision — the tool cannot make it for them."""
    out = make_git_tool(repo).invoke({"command": command})
    assert "unknown git command" in out


def test_a_slow_repository_times_out_rather_than_hanging(repo, monkeypatch):
    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=2)

    monkeypatch.setattr(subprocess, "run", slow)
    git = GitContext(root=repo, enabled=True)
    git.refresh()

    assert git.as_prompt() == "", "a stalled git call should drop out, not stall the ask"


# -- through the orchestrator --------------------------------------------


def test_the_git_tool_is_offered_only_inside_a_repository(repo, bare):
    """No repo, no tool — the model shouldn't be told about something useless."""
    assert "git" in _tool_names(repo)
    assert "git" not in _tool_names(bare)


def _tool_names(root):
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.messages import AIMessage

    class Quiet(BaseChatModel):
        @property
        def _llm_type(self):
            return "quiet"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])

    return {t.name for t in build_orchestrator(model=Quiet(), root=root).tools}