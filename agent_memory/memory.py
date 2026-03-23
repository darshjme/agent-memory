"""
agent_memory.memory
~~~~~~~~~~~~~~~~~~~
Lightweight in-process memory for ReAct agents.
Provides short-term (sliding window) and episodic (key-event) memory.
"""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory record."""
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_message(self) -> dict[str, str]:
        """Return OpenAI-style {role, content} dict."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """
    Sliding-window buffer of the most recent N observations/actions.

    Parameters
    ----------
    max_size : int
        Maximum number of entries to retain (oldest are dropped first).
    """

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._buffer: deque[MemoryEntry] = deque(maxlen=max_size)

    def add(self, role: str, content: str, **metadata: Any) -> MemoryEntry:
        """Append a new entry; oldest entry is evicted once max_size is reached."""
        entry = MemoryEntry(role=role, content=content, metadata=metadata)
        self._buffer.append(entry)
        return entry

    def get_recent(self, n: int = 5) -> list[MemoryEntry]:
        """Return the *n* most recent entries (newest last)."""
        entries = list(self._buffer)
        return entries[-n:] if n < len(entries) else entries

    def clear(self) -> None:
        """Remove all entries."""
        self._buffer.clear()

    def to_messages(self) -> list[dict[str, str]]:
        """Return all entries as OpenAI-style message dicts."""
        return [e.to_message() for e in self._buffer]

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Key-event store for memories worth preserving across many steps.

    Parameters
    ----------
    (none — entries are added explicitly via .remember())
    """

    def __init__(self) -> None:
        self._episodes: list[MemoryEntry] = []

    def remember(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: float = 1.0,
    ) -> MemoryEntry:
        """Store a key event with optional tags and importance score."""
        meta = {"tags": list(tags or []), "importance": importance}
        entry = MemoryEntry(role="episodic", content=content, metadata=meta)
        self._episodes.append(entry)
        return entry

    def recall(
        self,
        query: Optional[str] = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Retrieve episodic memories.

        Filters by tags (any match) then by simple substring query on content,
        then sorts by importance (descending), returning up to *limit* results.
        """
        results = list(self._episodes)

        if tags:
            tag_set = set(tags)
            results = [
                e for e in results
                if tag_set & set(e.metadata.get("tags", []))
            ]

        if query:
            q = query.lower()
            results = [e for e in results if q in e.content.lower()]

        # Sort by importance descending, then by timestamp descending
        results.sort(
            key=lambda e: (e.metadata.get("importance", 1.0), e.timestamp),
            reverse=True,
        )

        return results[:limit]

    def forget(self, entry_id: str) -> bool:
        """Remove an entry by its UUID. Returns True if found and removed."""
        for i, e in enumerate(self._episodes):
            if e.id == entry_id:
                self._episodes.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all episodic memories."""
        self._episodes.clear()

    def __len__(self) -> int:
        return len(self._episodes)


# ---------------------------------------------------------------------------
# AgentMemory  (facade)
# ---------------------------------------------------------------------------

class AgentMemory:
    """
    Top-level memory facade for ReAct agents.

    Combines a ShortTermMemory (sliding window) and an EpisodicMemory (key events).

    Parameters
    ----------
    short_term_size : int
        Maximum number of short-term entries to retain.
    """

    def __init__(self, short_term_size: int = 10) -> None:
        self.short_term = ShortTermMemory(max_size=short_term_size)
        self.episodic = EpisodicMemory()

    # ------------------------------------------------------------------
    # Short-term helpers
    # ------------------------------------------------------------------

    def observe(self, content: str, **metadata: Any) -> MemoryEntry:
        """Record an observation (environment → agent) in short-term memory."""
        return self.short_term.add("observation", content, **metadata)

    def act(self, content: str, **metadata: Any) -> MemoryEntry:
        """Record an action (agent → environment) in short-term memory."""
        return self.short_term.add("action", content, **metadata)

    # ------------------------------------------------------------------
    # Episodic helpers (delegate)
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: float = 1.0,
    ) -> MemoryEntry:
        """Promote a key insight or event to episodic memory."""
        return self.episodic.remember(content, tags=tags, importance=importance)

    def recall(
        self,
        query: Optional[str] = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search episodic memory."""
        return self.episodic.recall(query=query, tags=tags, limit=limit)

    # ------------------------------------------------------------------
    # Context snapshot
    # ------------------------------------------------------------------

    def context(self) -> dict[str, Any]:
        """
        Return a serialisable snapshot useful for injecting into a prompt.

        Returns
        -------
        dict with keys:
            short_term  — list of OpenAI-style message dicts
            episodic    — list of {content, tags, importance, timestamp} dicts
        """
        episodic_summary = [
            {
                "content": e.content,
                "tags": e.metadata.get("tags", []),
                "importance": e.metadata.get("importance", 1.0),
                "timestamp": e.timestamp.isoformat(),
            }
            for e in sorted(
                self.episodic._episodes,
                key=lambda e: e.metadata.get("importance", 1.0),
                reverse=True,
            )
        ]
        return {
            "short_term": self.short_term.to_messages(),
            "episodic": episodic_summary,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear both short-term and episodic memory."""
        self.short_term.clear()
        self.episodic.clear()

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------

    def compress(
        self,
        summarize_fn: Optional[Callable[[list[MemoryEntry]], str]] = None,
    ) -> Optional[MemoryEntry]:
        """
        Compress old short-term entries into episodic memory to free capacity.

        Default behaviour (no summarize_fn):
            Moves the *oldest* short-term entry directly into episodic as a
            plain text record tagged ["compressed"].

        Custom behaviour (summarize_fn provided):
            Calls ``summarize_fn(entries)`` with all current short-term entries.
            The returned string is stored in episodic (tagged ["summary"]),
            then short-term is cleared.

        Returns the new EpisodicMemory entry, or None if short-term was empty.
        """
        if len(self.short_term) == 0:
            return None

        if summarize_fn is not None:
            entries = list(self.short_term)
            summary_text = summarize_fn(entries)
            self.short_term.clear()
            return self.episodic.remember(
                summary_text,
                tags=["summary"],
                importance=0.8,
            )
        else:
            # Move the oldest entry to episodic
            oldest = list(self.short_term)[0]
            # Remove it from the buffer by rebuilding (deque has no remove-by-index)
            remaining = list(self.short_term)[1:]
            self.short_term.clear()
            for e in remaining:
                self.short_term.add(e.role, e.content, **e.metadata)
            return self.episodic.remember(
                oldest.content,
                tags=["compressed"],
                importance=0.5,
            )
