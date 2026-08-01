# Running nanocode

## First time only

Install it so the `nanocode` command works from any folder:

```bash
cd "d:/genai_projects/Agentic Projects/nanocode"
uv tool install -e .
```

If your terminal still says `nanocode: command not found` after that, run
`uv tool update-shell` and open a new terminal.

You don't need to activate the venv to use nanocode. That's only for working on
nanocode's own code.

## Every time

**1. Set your API key.**

PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Git Bash:

```bash
export OPENAI_API_KEY=sk-...
```

This only lasts for that terminal window. To stop retyping it, in PowerShell:

```powershell
setx OPENAI_API_KEY "sk-..."
```

Then open a new terminal for it to take effect.

**2. Go to the project you want it to work on.**

```powershell
cd D:\some\project
```

That folder is the sandbox. Nanocode can read and write anything in it and
nothing outside it.

**3. Start it.**

```powershell
nanocode
```

You'll get a prompt:

```
nanocode · openai:gpt-5.4-mini · D:\some\project
ask for anything; 'exit' or Ctrl-D when you're done

›
```

Type what you want and hit enter. It'll show a plan panel that fills in as it
works, and print a summary of what changed when it's done. Then you get the
prompt back and can keep going. It remembers the earlier asks.

Type `exit` (or `quit`, or Ctrl-D) when you're finished.

## Flags

| Flag | What it does |
| --- | --- |
| `--model`, `-m` | Use a different model for this run |
| `--dir`, `-C` | Work on a different folder than the one you're in |
| `--once` | Do one task and quit instead of prompting again |
| `--resume` | Pick up an interrupted session in this folder |
| `--plain` | Plain log output, no live panel. Good for piping to a file |
| `--context-window` | When compaction kicks in. Default 200000 |

A few combinations that come up:

```powershell
nanocode "add a /health endpoint"              # start with a task instead of a blank prompt
nanocode --once "run the tests"                # one-shot, for scripts
nanocode --resume                              # continue after Ctrl-C or a crash
nanocode -C D:\other\project "fix the imports" # work somewhere else without cd'ing
nanocode --plain > run.log                     # capture the whole thing to a file
```

## Changing the model

**Just for one run**, pass `--model`:

```powershell
nanocode --model openai:gpt-5.6-sol "refactor the auth layer"
```

The format is always `provider:model-name`. The provider part decides which
environment variable gets read:

| Prefix | Key it reads |
| --- | --- |
| `openai:` | `OPENAI_API_KEY` |
| `anthropic:` | `ANTHROPIC_API_KEY` |

Some options:

| Model | When |
| --- | --- |
| `openai:gpt-5.4-mini` | the default. Fast and cheap, fine for most things |
| `openai:gpt-5.6-terra` | a step up when the mini one is struggling |
| `openai:gpt-5.6-sol` | big refactors, anything long-running |
| `anthropic:claude-opus-5` | if you'd rather use your Anthropic key |

Not sure what your key can actually reach? This lists it:

```powershell
python -c "from openai import OpenAI; print(sorted(m.id for m in OpenAI().models.list()))"
```

**If you're tired of typing `--model` every time**, make an alias. PowerShell:

```powershell
function nc { nanocode --model openai:gpt-5.6-terra @args }
```

Put that in your PowerShell profile (`notepad $PROFILE`) to keep it. In Git
Bash it's `alias nc='nanocode --model openai:gpt-5.6-terra'` in `~/.bashrc`.

**To actually change the default**, open
`src/nanocode/orchestrator.py` and edit this line near the top:

```python
DEFAULT_MODEL = "openai:gpt-5.4-mini"
```

Because it's installed with `-e`, editing the file is enough. No reinstall.
Check it took with `nanocode --help` — the default shows up next to `--model`.

While you're in there, `--context-window` defaults to 200000, which is
deliberately low so it's safe on any model. gpt-5.4-mini actually holds 400000,
so on long sessions you can pass `--context-window 400000` and it'll compact
less often.

## When something breaks

**`nanocode: command not found`** — `uv tool update-shell`, then restart the
terminal.

**A 401 or "incorrect API key"** — the key isn't set in the terminal you're
actually using. Check with `echo $env:OPENAI_API_KEY` in PowerShell.

**"not a directory"** — the path you gave `-C` doesn't exist.

**You hit Ctrl-C partway through** — nothing's lost. `nanocode --resume` picks
up from the plan and the log. It won't remember the conversation, but it knows
what it had done and what was left.

**Something went wrong and you want it undone** — `git checkout .` if you're in
a repo. This is the reason to use nanocode on a git repo.

**You want to see what a command actually printed** — the summary only shows
the tail. Full output for every shell command is in `.nanocode/logs/`.
