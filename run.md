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
| `--fresh` | Ignore unfinished work and the saved constraints. Clean slate |
| `--resume` | Force a pick-up even of a run that finished |
| `--plain` | Plain log output, no live panel. Good for piping to a file |
| `--context-window` | When compaction kicks in. Default 200000 |

A few combinations that come up:

```powershell
nanocode "add a /health endpoint"              # start with a task instead of a blank prompt
nanocode --once "run the tests"                # one-shot, for scripts
nanocode --fresh                               # ignore the last run, start clean
nanocode -C D:\other\project "fix the imports" # work somewhere else without cd'ing
nanocode --plain > run.log                     # capture the whole thing to a file
```

## Commands

Type these at the `›` prompt instead of a task:

| | |
| --- | --- |
| `/model` | list the models your keys can reach, and switch to one |
| `/clear` | forget the conversation and start fresh |
| `/help` | the list |

**`/model`** asks the provider what your key can actually reach, rather than
showing a list baked into nanocode that would be out of date within weeks:

```
› /model

  1  anthropic  no key yet — you'll be asked
  2  openai     OPENAI_API_KEY set
provider (number, or blank to cancel): 2
asking openai what your key can reach…

  1  openai:gpt-5.6-sol
  2  openai:gpt-5.4-mini
currently: openai:gpt-5.4-mini
model (number or full name, blank to cancel): 1
✓ openai:gpt-5.4-mini → openai:gpt-5.6-sol  (conversation kept)
```

If the provider you pick has no key set, it asks for one and holds it for that
terminal session — nanocode never writes a key to disk. Use `setx` (above) to
keep it permanently.

Switching keeps your conversation, so it's fine to start cheap and reach for a
bigger model on the one hard step:

```
› /model openai:gpt-5.6-sol    # straight to it, no menu
› /model gpt-5                 # filter the list
```

**`/clear`** drops the conversation, the plan and the log, and starts fresh in
the same folder. Your files aren't touched, earlier runs stay saved, and your
constraints are kept — those are project rules, not conversation.

## Coming back to a project

You don't need a flag. Run `nanocode` in a folder where the last run didn't
finish, and it picks up where it stopped:

```
nanocode · openai:gpt-5.4-mini · D:\some\project
picking up where the last run stopped — 3/5 steps done (--fresh to start clean)
2 standing constraints loaded from constraints.md
```

It comes back knowing the plan, which steps were done, and what it actually did
— files it edited, commands it ran. It does **not** remember the conversation.
That's deliberate: reloading yesterday's chat would eat the context budget
before you'd typed anything.

If you want a clean start, `--fresh`. If the last run finished and you want it
back anyway, `--resume`.

## Standing rules

Some things aren't tasks — they're rules that should hold forever. Just say so:

```
› never modify the auth module, it's shared with another team
```

It writes that to `.nanocode/constraints.md` and applies it in every session
from then on. You can edit the file yourself, or write it before you ever start:

```markdown
# Constraints

- do not modify the auth module — shared with another team
- always run the tests before saying you're done
- this project targets Python 3.11
```

One rule per line, starting with a dash. Unlike anything said in chat, these
can't be forgotten — they're fed back to the model on every single turn.

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

**You hit Ctrl-C partway through** — nothing's lost. Just run `nanocode` again
in that folder and it picks up from the plan and the log. It won't remember the
conversation, but it knows what it had done and what was left.

**It picked up an old task you didn't want** — `--fresh`. Nothing is deleted;
the old session file is still in `.nanocode/sessions/`.

**It keeps ignoring something you told it** — say it as a rule ("always run the
tests before finishing") so it gets written to `.nanocode/constraints.md`, or
add the line to that file yourself. Anything left in chat is forgotten when you
close the terminal.

**Something went wrong and you want it undone** — `git checkout .` if you're in
a repo. This is the reason to use nanocode on a git repo.

**You want to see what a command actually printed** — the summary only shows
the tail. Full output for every shell command is in `.nanocode/logs/`.
