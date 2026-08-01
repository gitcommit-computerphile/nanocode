"""Optional web search, wired the same way every other tool is.

It follows the offload shape `grep` and `shell` already use — fetch, write the
full result to disk, return a short summary — and it is read-only exploration,
which is exactly what the `explorer` sub-agent exists for. The key is read once
from the environment at startup and never written to session.json or any log.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from . import prompts
from .shell_tool import write_log
from .state import event

MAX_RESULTS = 5
SNIPPET_CHARS = 400


def search_available() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


def make_search_tool(root: str | Path) -> BaseTool | None:
    """Build the web_search tool, or None when no search key is configured."""
    if not search_available():
        return None
    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return None

    project_root = Path(root).resolve()
    client = TavilySearch(max_results=MAX_RESULTS)

    @tool("web_search", description=prompts.WEB_SEARCH_DESCRIPTION)
    def web_search(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        try:
            raw = client.invoke({"query": query})
        except Exception as exc:  # network/provider failures are expected
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"search failed: {exc}", tool_call_id=tool_call_id, status="error"
                        )
                    ]
                }
            )

        results = raw.get("results", []) if isinstance(raw, dict) else []
        log_path = write_log(project_root, "search", f"query: {query}\n\n{raw}")
        if not results:
            summary = f"no results for {query!r}"
        else:
            lines = []
            for item in results[:MAX_RESULTS]:
                title = item.get("title", "untitled")
                url = item.get("url", "")
                content = (item.get("content") or "").strip().replace("\n", " ")
                lines.append(f"- {title} ({url})\n  {content[:SNIPPET_CHARS]}")
            summary = "\n".join(lines)

        return Command(
            update={
                "session_log": [event("search", f"web_search: {query}")],
                "messages": [
                    ToolMessage(
                        f"{summary}\n\n(full results: {log_path.as_posix()})",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return web_search
