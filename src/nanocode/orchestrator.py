"""The orchestrator.

A single `create_agent` loop: read state, optionally call one tool, observe,
repeat until it stops calling tools. Nothing about this loop depends on how big
the task is — a ten-step task and a two-hundred-step task run the identical
loop; only the plan and the disk grow, never the amount the model has to hold
in its head at once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from . import prompts
from .fs_tools import DiskFileSystem, FileSystem, make_fs_tools
from .search_tool import make_search_tool
from .shell_tool import make_shell_tool
from .state import NanocodeState
from .subagents import make_delegate_tool
from .todo_tools import DEFAULT_CONTEXT_WINDOW, make_compactor, recite_todos, write_todos

# One flag picks the provider and model for the whole run — the orchestrator
# and every sub-agent share it, so nothing mixes providers mid-task.
DEFAULT_MODEL = "openai:gpt-5.4-mini"

# Which environment variable each provider prefix reads its key from. The key
# is read once at startup and never written to session.json or any log.
PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class ConfigError(Exception):
    """Bad model string, or a missing API key."""


SESSION_THREAD = "nanocode-session"
RECURSION_LIMIT = 250


@dataclass
class Orchestrator:
    agent: CompiledStateGraph
    model: BaseChatModel
    fs: FileSystem
    root: Path
    tools: list[BaseTool] = field(default_factory=list)

    @property
    def config(self) -> dict:
        """The config every turn of a session shares.

        The thread id is what makes the conversation continuous: each ask
        resumes the same checkpointed state instead of starting cold.
        """
        return {
            "configurable": {"thread_id": SESSION_THREAD},
            "recursion_limit": RECURSION_LIMIT,
        }


def resolve_model(model: str | BaseChatModel) -> BaseChatModel:
    """Build the chat model from a `provider:name` string.

    Taking a model object rather than a fixed client is what makes swapping
    providers a config change instead of a code change — and lets tests hand in
    a scripted model directly.
    """
    if isinstance(model, BaseChatModel):
        return model
    if ":" not in model:
        raise ConfigError(
            f"model must be 'provider:name', e.g. {DEFAULT_MODEL!r} — got {model!r}"
        )
    provider, _, name = model.partition(":")
    provider = provider.strip().lower()
    if not name.strip():
        raise ConfigError(f"missing model name in {model!r}")

    key_var = PROVIDER_KEYS.get(provider)
    if key_var and not os.environ.get(key_var):
        raise ConfigError(
            f"{key_var} is not set, but --model {model!r} needs it.\n"
            f"  export {key_var}=..."
        )

    try:
        return init_chat_model(model)
    except Exception as exc:
        raise ConfigError(f"could not initialise model {model!r}: {exc}") from exc


def build_orchestrator(
    model: str | BaseChatModel = DEFAULT_MODEL,
    root: str | Path = ".",
    *,
    fs: FileSystem | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    on_trace: Callable[[str, str], None] | None = None,
) -> Orchestrator:
    """Wire the orchestrator: one model, one tool set, two middlewares.

    The same `model` string flows into every sub-agent built by the registry —
    one flag picks the provider for the whole run, so nothing mixes providers
    mid-task.
    """
    project_root = Path(root).resolve()
    llm = resolve_model(model)
    filesystem = fs if fs is not None else DiskFileSystem(project_root)

    tools: list[BaseTool] = [
        write_todos,
        *make_fs_tools(filesystem, writable=True),
        make_shell_tool(project_root),
        make_delegate_tool(llm, filesystem, project_root, on_trace=on_trace),
    ]
    if search := make_search_tool(project_root):
        tools.append(search)

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=prompts.ORCHESTRATOR_PROMPT,
        state_schema=NanocodeState,
        middleware=[recite_todos, make_compactor(context_window)],
        # One orchestrator instance serves the whole session. The checkpointer
        # is what carries the conversation from one ask to the next: turn two
        # sees everything turn one did, without replaying or re-explaining it.
        checkpointer=InMemorySaver(),
        name="nanocode",
    )
    return Orchestrator(agent=agent, model=llm, fs=filesystem, root=project_root, tools=tools)
