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
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from . import prompts
from .constraints import make_constraints_tool
from .fs_tools import DiskFileSystem, FileSystem, make_fs_tools
from .git_tool import GitContext, make_git_tool
from .retry import make_retrier
from .search_tool import make_search_tool
from .shell_tool import make_shell_tool
from .state import NanocodeState
from .subagents import make_delegate_tool
from .todo_tools import DEFAULT_CONTEXT_WINDOW, make_compactor, make_reciter, write_todos
from .usage import Usage

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
    # What the user asked for, kept for display and for rebuilding on a swap.
    spec: str = ""
    subagent_spec: str = ""
    context_window: int = DEFAULT_CONTEXT_WINDOW
    on_trace: Callable[[str, str], None] | None = None
    on_retry: Callable[[str], None] | None = None
    on_compact: Callable[[bool], None] | None = None
    # Token accounting for the whole session. Carried across a `/model` swap so
    # switching models doesn't reset the running total.
    usage: Usage = field(default_factory=Usage)
    # Refreshed once per ask by the CLI, then read from cache on every model
    # call — see GitContext for why those are two different rhythms.
    git: GitContext | None = None
    # Held so a mid-session model swap can hand the same one to the replacement
    # graph — that is what carries the conversation across the swap.
    checkpointer: BaseCheckpointSaver | None = None
    # Mutable: `/clear` rotates it. Everything in state hangs off the thread id,
    # including `session_log`, whose `operator.add` reducer means it cannot be
    # emptied by assignment — a new thread is the only real reset.
    thread: str = SESSION_THREAD

    @property
    def config(self) -> dict:
        """The config every turn of a session shares.

        The thread id is what makes the conversation continuous: each ask
        resumes the same checkpointed state instead of starting cold.
        """
        return {
            "configurable": {"thread_id": self.thread},
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
            f"{key_var} is not set, but {model!r} needs it.\n"
            f"  set it in this terminal (export {key_var}=...), "
            f"or run /model inside nanocode to enter it"
        )

    try:
        return init_chat_model(model)
    except Exception as exc:
        raise ConfigError(f"could not initialise model {model!r}: {exc}") from exc


def _caching_middleware() -> list:
    """Anthropic prompt caching, when the package provides it.

    Wrapped in a try/import because it is a convenience, not a requirement: a
    missing or renamed middleware should cost a discount, never a startup.
    `unsupported_model_behavior="ignore"` keeps it silent on OpenAI, which does
    its own caching server-side with no client involvement.
    """
    try:
        from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
    except ImportError:
        return []
    return [AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")]


def build_orchestrator(
    model: str | BaseChatModel = DEFAULT_MODEL,
    root: str | Path = ".",
    *,
    subagent_model: str | BaseChatModel | None = None,
    fs: FileSystem | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    on_trace: Callable[[str, str], None] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    usage: Usage | None = None,
    on_retry: Callable[[str], None] | None = None,
    on_compact: Callable[[bool], None] | None = None,
    git_context: GitContext | None = None,
) -> Orchestrator:
    """Wire the orchestrator: one model, one tool set, two middlewares.

    The same `model` string flows into every sub-agent built by the registry —
    one flag picks the provider for the whole run, so nothing mixes providers
    mid-task.

    Passing an existing `checkpointer` rebuilds the graph around a conversation
    that is already underway, which is how `/model` swaps the model without
    costing the user their context.
    """
    project_root = Path(root).resolve()
    llm = resolve_model(model)
    # Sub-agents may run on a cheaper model: explorer greps and reads,
    # test-runner runs a suite and reports failures. That is execution, not
    # judgment — and judgment is the only thing that stays in the orchestrator.
    # Defaults to the same model, so nothing changes unless asked.
    sub_llm = resolve_model(subagent_model) if subagent_model else llm
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    tally = usage if usage is not None else Usage()
    retrier = make_retrier(on_retry=on_retry)
    # Detected once. A non-git project pays one 40ms check and nothing more.
    git = git_context if git_context is not None else GitContext.detect(project_root)
    filesystem = fs if fs is not None else DiskFileSystem(project_root)

    tools: list[BaseTool] = [
        write_todos,
        make_constraints_tool(project_root),
        *make_fs_tools(filesystem, writable=True),
        make_shell_tool(project_root),
        make_delegate_tool(
            sub_llm,
            filesystem,
            project_root,
            on_trace=on_trace,
            # Not is_context: a sub-agent's context is its own, and its size
            # says nothing about how full the main conversation is.
            on_usage=lambda inp, out: tally.add(inp, out),
            # Sub-agents get the same retry, or one 529 inside a delegation
            # wastes the whole delegation and the orchestrator's turn with it.
            middleware=[retrier],
        ),
    ]
    if git.enabled:
        tools.append(make_git_tool(project_root))
    if search := make_search_tool(project_root):
        tools.append(search)

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=prompts.ORCHESTRATOR_PROMPT,
        state_schema=NanocodeState,
        # Order matters only between the two wrap_model_call hooks: recitation
        # is outermost, so the retrier sits closest to the provider and retries
        # the call itself rather than rebuilding the request each time.
        # The compactor gets the model so the span it drops is summarised into
        # notes first, rather than vanishing behind a fixed sentence.
        middleware=[
            make_reciter(git),
            retrier,
            make_compactor(context_window, model=llm, on_compact=on_compact),
            # The system prompt and the tool definitions are identical on every
            # turn of a session — exactly what caching is for. Recitation
            # appends its reminder at the *end* of the message list, so the
            # cached prefix stays byte-stable and keeps hitting.
            # OpenAI caches server-side with no client action, so this is
            # Anthropic-only and silently does nothing elsewhere.
            *_caching_middleware(),
        ],
        # One orchestrator instance serves the whole session. The checkpointer
        # is what carries the conversation from one ask to the next: turn two
        # sees everything turn one did, without replaying or re-explaining it.
        checkpointer=saver,
        name="nanocode",
    )
    return Orchestrator(
        agent=agent,
        model=llm,
        fs=filesystem,
        root=project_root,
        tools=tools,
        spec=model if isinstance(model, str) else getattr(llm, "model_name", "custom model"),
        # Empty unless a split was explicitly asked for. Recording the shared
        # model here would make `/model` pin sub-agents to the pre-switch model
        # while appearing to switch everything.
        subagent_spec=(
            ""
            if subagent_model is None
            else (
                subagent_model
                if isinstance(subagent_model, str)
                else getattr(sub_llm, "model_name", "")
            )
        ),
        context_window=context_window,
        on_trace=on_trace,
        on_retry=on_retry,
        on_compact=on_compact,
        usage=tally,
        git=git,
        checkpointer=saver,
    )
