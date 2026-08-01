from __future__ import annotations

import pytest

from nanocode.fs_tools import (
    DiskFileSystem,
    SandboxError,
    VirtualFileSystem,
    make_fs_tools,
)


@pytest.fixture
def disk(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return DiskFileSystem(tmp_path)


def tools_by_name(fs, **kw):
    return {t.name: t for t in make_fs_tools(fs, **kw)}


# -- sandbox: hard block, no escape hatch ---------------------------------


@pytest.mark.parametrize(
    "path",
    ["../secrets.txt", "../../etc/passwd", "src/../../outside.py", "/etc/passwd", r"C:\Windows\win.ini"],
)
def test_paths_outside_root_are_rejected(disk, path):
    with pytest.raises(SandboxError):
        disk.resolve(path)


def test_symlink_escaping_root_is_rejected(disk, tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted in this environment")
    with pytest.raises(SandboxError):
        disk.resolve("link.txt")


def test_paths_inside_root_resolve(disk):
    assert disk.resolve("src/app.py").name == "app.py"
    assert disk.resolve(".") == disk.root


def test_root_must_be_a_directory(tmp_path):
    target = tmp_path / "nope"
    with pytest.raises(SandboxError):
        DiskFileSystem(target)


# -- read_file pagination is load-bearing ---------------------------------


def test_read_file_paginates_and_says_how_to_continue(disk):
    disk.write("big.txt", "\n".join(f"line {i}" for i in range(1, 1001)))
    read = tools_by_name(disk)["read_file"]
    out = read.invoke({"path": "big.txt", "limit": 50})
    assert "line 50" in out
    assert "line 51" not in out
    assert "offset=50" in out


def test_read_file_numbers_lines(disk):
    out = tools_by_name(disk)["read_file"].invoke({"path": "src/app.py"})
    assert "1\tdef hello():" in out


# -- edit_file forces a precise, unique match -----------------------------


def test_edit_file_replaces_a_unique_string(disk, call):
    result = call(
        tools_by_name(disk)["edit_file"],
        path="src/app.py",
        old_string="return 'hi'",
        new_string="return 'hello'",
    )
    assert "return 'hello'" in disk.read("src/app.py")
    assert result.update["session_log"][0]["kind"] == "file_edit"


def test_edit_file_refuses_an_ambiguous_match(disk, call, message):
    disk.write("dupe.py", "x = 1\nx = 1\n")
    result = call(
        tools_by_name(disk)["edit_file"], path="dupe.py", old_string="x = 1", new_string="x = 2"
    )
    assert message(result).status == "error"
    assert "2 times" in message(result).content
    assert disk.read("dupe.py") == "x = 1\nx = 1\n"  # unchanged


def test_edit_file_refuses_a_missing_match(disk, call, message):
    result = call(
        tools_by_name(disk)["edit_file"], path="src/app.py", old_string="nope", new_string="x"
    )
    assert message(result).status == "error"


# -- grep / glob ----------------------------------------------------------


def test_grep_finds_matches_with_locations(disk):
    out = tools_by_name(disk)["grep"].invoke({"pattern": r"def \w+", "glob": "**/*.py"})
    assert "src/app.py:1:" in out


def test_grep_reports_an_invalid_regex_rather_than_raising(disk):
    out = tools_by_name(disk)["grep"].invoke({"pattern": "([unclosed"})
    assert "invalid regular expression" in out


def test_grep_skips_nanocode_scratch_state(disk):
    (disk.root / ".nanocode" / "logs").mkdir(parents=True)
    (disk.root / ".nanocode" / "logs" / "run.log").write_text("needle", encoding="utf-8")
    out = tools_by_name(disk)["grep"].invoke({"pattern": "needle"})
    # The scratch dir must not appear as a hit; "no matches" mentions the
    # pattern itself, so assert on the path rather than the word.
    assert ".nanocode" not in out
    assert out.startswith("no matches")


def test_glob_matches_recursively(disk):
    out = tools_by_name(disk)["glob"].invoke({"pattern": "**/*.py"})
    assert "src/app.py" in out
    assert "README.md" not in out


# -- read-only tool sets have no write tools ------------------------------


def test_readonly_toolset_omits_writers(disk):
    names = set(tools_by_name(disk, writable=False))
    assert names == {"ls", "read_file", "grep", "glob"}


# -- the virtual backend satisfies the same interface ---------------------


def test_virtual_backend_supports_the_same_tools(call):
    fs = VirtualFileSystem({"a.py": "print('x')\n", "pkg/b.py": "y = 2\n"})
    tools = tools_by_name(fs)
    assert "a.py" in tools["ls"].invoke({"path": "."})
    assert "pkg/" in tools["ls"].invoke({"path": "."})
    assert "print" in tools["read_file"].invoke({"path": "a.py"})
    assert "pkg/b.py:1:" in tools["grep"].invoke({"pattern": "y = 2"})
    call(tools["write_file"], path="c.py", content="z = 3\n")
    assert fs.files["c.py"] == "z = 3\n"


def test_virtual_backend_also_rejects_traversal():
    fs = VirtualFileSystem({"a.py": ""})
    with pytest.raises(SandboxError):
        fs.read("../a.py")
