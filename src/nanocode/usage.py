"""Token accounting for a session.

Two different questions, deliberately tracked separately:

- **What has this session cost?** Cumulative input and output tokens, including
  every sub-agent, since those are billed the same as anything else.
- **How full is the context right now?** The input token count of the most
  recent *orchestrator* call. Sub-agent calls are excluded from this one on
  purpose: they run in their own context and their size says nothing about how
  close the main conversation is to compaction.

Conflating the two would make the percentage jump around whenever a sub-agent
read a large file, which is exactly when it should not move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    # Input size of the last orchestrator call — the live context measurement.
    context_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int, *, is_context: bool = False) -> None:
        if not (input_tokens or output_tokens):
            return
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        if is_context:
            self.context_tokens = input_tokens

    def record(self, message: Any, *, is_context: bool = False) -> None:
        """Take whatever a provider reported on one assistant message."""
        meta = getattr(message, "usage_metadata", None) or {}
        self.add(
            int(meta.get("input_tokens") or 0),
            int(meta.get("output_tokens") or 0),
            is_context=is_context,
        )

    def context_percent(self, window: int) -> float:
        return 100.0 * self.context_tokens / window if window else 0.0


def usage_of(messages: list[Any]) -> tuple[int, int]:
    """Total input/output across a message list — used for a sub-agent's run,
    whose messages never reach the parent and so are never counted there."""
    totals = [0, 0]
    for message in messages:
        meta = getattr(message, "usage_metadata", None) or {}
        totals[0] += int(meta.get("input_tokens") or 0)
        totals[1] += int(meta.get("output_tokens") or 0)
    return totals[0], totals[1]


def human(count: int) -> str:
    """Compact token counts: 940, 12.4k, 1.20M."""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.2f}M"