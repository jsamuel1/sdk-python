"""Marking for assistant turns that the SDK synthesized rather than a model generated.

A synthesized turn is appended to the conversation like any assistant turn, because the next model call needs the
matching ``toolUse``/``toolResult`` pair. The marking lives in ``MessageMetadata``, which is stripped before model
calls and persisted by session managers, so replays and audits can tell it apart from generation.
"""

from __future__ import annotations

from typing import Any

from ..types.content import Message

SYNTHESIZED_KEY = "strands"
"""Key under ``message["metadata"]["custom"]`` that holds the synthesized-turn marking."""

DECISION_SOURCE = "decision"
"""Source of a turn a System One decision model chose (see ``strands.experimental.decisions.FastPath``)."""


def mark_synthesized(message: Message, source: str, **fields: Any) -> None:
    """Record on ``message`` that ``source`` produced it, with any attribution ``fields``."""
    metadata = message.setdefault("metadata", {})
    custom = metadata.setdefault("custom", {})
    custom[SYNTHESIZED_KEY] = {"source": source, **fields}


def synthesized_source(message: Message) -> str | None:
    """Return the source that synthesized ``message``, or None when a model generated it."""
    marker = message.get("metadata", {}).get("custom", {}).get(SYNTHESIZED_KEY)
    if not isinstance(marker, dict):
        return None
    source = marker.get("source")
    return source if isinstance(source, str) else None
