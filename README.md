# nanocode

A small coding agent that runs in your terminal. You point it at a project,
tell it what you want, and it goes off and does it: makes a plan, greps around,
edits files, runs your tests. Then it asks what's next.

It's a scaled-down take on how Claude Code works, built on the deep-agents
patterns (planning, filesystem offload, sub-agent isolation, prompting).

## Setup

```bash
uv venv
uv pip install -e .
```

If you want it available everywhere and not just in this repo:

```bash
uv tool install -e .
```

## Running it

```bash
export OPENAI_API_KEY=sk-...
cd ~/projects/my-app
nanocode
```

Whatever directory you're standing in is the one it works on. It can't touch
anything outside it.

```
nanocode · openai:gpt-5.4-mini · ~/projects/my-app
ask for anything; 'exit' or Ctrl-D when you're done

› add rate limiting to the handler
  ... plans, edits, runs the tests, prints what changed ...

› now cover it with a test

› exit
```

The second ask knows about the first one, so you can say "now cover it with a
test" instead of explaining what "it" is again.

You can also hand it the first task on the command line, which is handy if you
already know what you want:

```bash
nanocode "add rate limiting middleware and update the tests"
```

And `--once` does one task and quits instead of prompting again. That's the one
to use in scripts or CI.

```bash
nanocode --once "run the test suite and summarize failures"
```

## Models

Default is `openai:gpt-5.4-mini`. Everything uses it, including the sub-agents.
Swap it with `--model`:

```bash
nanocode --model openai:gpt-5.6-sol "refactor the auth layer"
nanocode --model anthropic:claude-opus-5 "build a CSV to JSON CLI tool"
```

The bit before the colon picks which API key gets read. `openai:` wants
`OPENAI_API_KEY`, `anthropic:` wants `ANTHROPIC_API_KEY`. Keys are read once
when it starts and never end up in `session.json` or the logs.

To change the default permanently, edit `DEFAULT_MODEL` in
`src/nanocode/orchestrator.py`. See `run.md` for the details.

## Things worth knowing

**It edits files for real, without asking first.** The sandbox keeps it inside
the project directory, but inside that directory it just writes. Use it on a
git repo so you can `git diff` afterwards and throw the whole thing away if it
went sideways.

**Ctrl-C is safe.** Progress gets saved as it goes, so `nanocode --resume`
picks up from the plan and the log of what already happened. You lose the
conversation itself, not the work.

**It leaves a `.nanocode/` folder behind** with the session file and full logs
from every shell command it ran. Add it to your `.gitignore`. Nanocode will
nag you about this if you forget.

## How it's put together

Longer version in `docs/architecture.md`. The short version: there's one
`create_agent` loop doing the work, and the plan and the file contents live on
disk instead of piling up in the message history. That's the whole trick, and
it's why a long session doesn't run out of context.

| Module | What it does |
| --- | --- |
| `cli.py` | the prompt loop, and starting or resuming a session |
| `state.py` | `NanocodeState`, `Todo`, `Event` |
| `orchestrator.py` | wires up `create_agent` and the middleware |
| `todo_tools.py` | `write_todos`, plus recitation and compaction |
| `fs_tools.py` | `ls`, `read_file`, `write_file`, `edit_file`, `grep`, `glob` |
| `shell_tool.py` | `shell`, with output truncation and a guard on destructive commands |
| `subagents.py` | the explorer / coder / test-runner registry and `delegate` |
| `session.py` | reading and writing `.nanocode/session.json` |
| `prompts.py` | the system prompt and every tool description |
| `ui.py` | the live todo panel, diffs, sub-agent trace |

## Tests

```bash
pytest -q
```
