"""Standing constraints — the third durable field.

The plan says what to do and the log says what happened; both are scoped to one
task. Constraints are the rules that outlive it: "never touch the auth module",
"always run the tests before finishing". Those are the things a user says once
and expects to hold forever.

Left in the conversation, such a rule dies twice over — when the process exits,
and again when compaction trims the turn it was said in. So it gets the same
treatment as the plan: written to disk the moment it is stated, and re-injected
into every model call thereafter. The same discipline the rest of the design
applies to files and plans, applied to intent.

The file is plain markdown on purpose. Writing a rule into `.nanocode/
constraints.md` by hand before the agent starts is a supported way to use this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from . import prompts, session
from .state import event


def make_constraints_tool(root: str | Path) -> BaseTool:
    """Build `write_constraints`, closed over the project it writes to."""
    project_root = Path(root)

    @tool("write_constraints", description=prompts.WRITE_CONSTRAINTS_DESCRIPTION)
    def write_constraints(
        constraints: list[str],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        cleaned = [c.strip() for c in constraints if c and c.strip()]
        try:
            session.save_constraints(project_root, cleaned)
        except OSError as exc:
            # Same rule as the filesystem tools: a bad write is an observation,
            # not the end of the run.
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"could not write constraints: {exc}",
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                    ]
                }
            )

        note = f"{len(cleaned)} standing constraint{'' if len(cleaned) == 1 else 's'}"
        return Command(
            update={
                "constraints": cleaned,
                "session_log": [event("constraint", note)],
                "messages": [
                    ToolMessage(f"constraints updated — {note}", tool_call_id=tool_call_id)
                ],
            }
        )

    return write_constraints
