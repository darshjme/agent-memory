"""
tests/test_memory.py
~~~~~~~~~~~~~~~~~~~~
12 tests covering ShortTermMemory, EpisodicMemory, and AgentMemory.
"""
import time
import pytest
from datetime import datetime, timezone

from agent_memory import AgentMemory, ShortTermMemory, EpisodicMemory, MemoryEntry


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

def test_short_term_sliding_window():
    """Add 15 items; only the last 10 are retained."""
    stm = ShortTermMemory(max_size=10)
    for i in range(15):
        stm.add("user", f"message {i}")
    assert len(stm) == 10
    messages = stm.to_messages()
    # The 10 retained entries should be messages 5..14
    assert messages[0]["content"] == "message 5"
    assert messages[-1]["content"] == "message 14"


def test_short_term_to_messages():
    """to_messages() returns OpenAI-style {role, content} dicts."""
    stm = ShortTermMemory(max_size=5)
    stm.add("user", "hello")
    stm.add("assistant", "world")
    msgs = stm.to_messages()
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1] == {"role": "assistant", "content": "world"}


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

def test_episodic_remember_and_recall():
    """Remembered entries appear in recall()."""
    em = EpisodicMemory()
    em.remember("The user prefers concise answers.")
    em.remember("The system crashed at step 7.")
    results = em.recall()
    assert len(results) == 2
    contents = {r.content for r in results}
    assert "The user prefers concise answers." in contents
    assert "The system crashed at step 7." in contents


def test_episodic_recall_by_tags():
    """recall(tags=...) returns only entries that match at least one tag."""
    em = EpisodicMemory()
    em.remember("API key stored", tags=["security", "config"])
    em.remember("User name is Alice", tags=["user"])
    em.remember("Tool call failed", tags=["error"])
    results = em.recall(tags=["security"])
    assert len(results) == 1
    assert results[0].content == "API key stored"

    # Multi-tag: both security and user entries should appear
    results2 = em.recall(tags=["security", "user"])
    assert len(results2) == 2


def test_episodic_recall_by_importance():
    """Higher importance entries come first in recall results."""
    em = EpisodicMemory()
    em.remember("Low importance fact", importance=0.1)
    em.remember("Critical fact", importance=0.9)
    em.remember("Medium fact", importance=0.5)
    results = em.recall()
    importances = [r.metadata["importance"] for r in results]
    assert importances == sorted(importances, reverse=True)


def test_episodic_forget():
    """forget(entry_id) removes the entry; returns True on success, False if not found."""
    em = EpisodicMemory()
    entry = em.remember("To be forgotten")
    assert len(em) == 1
    removed = em.forget(entry.id)
    assert removed is True
    assert len(em) == 0
    # Second call returns False
    assert em.forget(entry.id) is False


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------

def test_agent_memory_observe_and_act():
    """observe() and act() store entries with correct roles."""
    am = AgentMemory()
    am.observe("Environment returned: 42")
    am.act("Call tool: calculator(40+2)")
    msgs = am.short_term.to_messages()
    assert msgs[0] == {"role": "observation", "content": "Environment returned: 42"}
    assert msgs[1] == {"role": "action", "content": "Call tool: calculator(40+2)"}


def test_agent_memory_context():
    """context() returns dict with short_term and episodic keys."""
    am = AgentMemory()
    am.observe("step 1")
    am.remember("key insight", tags=["important"], importance=0.9)
    ctx = am.context()
    assert "short_term" in ctx
    assert "episodic" in ctx
    assert len(ctx["short_term"]) == 1
    assert len(ctx["episodic"]) == 1
    ep = ctx["episodic"][0]
    assert ep["content"] == "key insight"
    assert ep["importance"] == 0.9
    assert "important" in ep["tags"]


def test_agent_memory_reset():
    """reset() wipes both short-term and episodic memory."""
    am = AgentMemory()
    am.observe("data")
    am.remember("event")
    am.reset()
    assert len(am.short_term) == 0
    assert len(am.episodic) == 0


def test_agent_memory_compress_without_fn():
    """
    compress() without a summarize_fn moves the oldest short-term entry
    into episodic, tagged ['compressed'].
    """
    am = AgentMemory(short_term_size=5)
    for i in range(5):
        am.observe(f"obs {i}")

    assert len(am.short_term) == 5
    assert len(am.episodic) == 0

    new_ep = am.compress()
    assert new_ep is not None
    assert "compressed" in new_ep.metadata["tags"]
    assert new_ep.content == "obs 0"
    # Short-term now has 4 entries (oldest removed)
    assert len(am.short_term) == 4
    assert len(am.episodic) == 1


def test_agent_memory_compress_with_fn():
    """
    compress(summarize_fn=...) calls the function with all short-term entries,
    stores the summary in episodic, and clears short-term.
    """
    am = AgentMemory(short_term_size=5)
    am.observe("alpha")
    am.observe("beta")
    am.observe("gamma")

    def my_summarizer(entries):
        return "Summary: " + ", ".join(e.content for e in entries)

    ep = am.compress(summarize_fn=my_summarizer)
    assert ep is not None
    assert ep.content == "Summary: alpha, beta, gamma"
    assert "summary" in ep.metadata["tags"]
    # Short-term cleared
    assert len(am.short_term) == 0
    assert len(am.episodic) == 1


def test_memory_entry_timestamp_auto():
    """MemoryEntry auto-assigns a UTC timestamp if none provided."""
    before = datetime.now(timezone.utc)
    entry = MemoryEntry(role="user", content="hello")
    after = datetime.now(timezone.utc)
    assert before <= entry.timestamp <= after
    assert entry.timestamp.tzinfo is not None  # must be tz-aware


# ---------------------------------------------------------------------------
# Bonus: get_recent
# ---------------------------------------------------------------------------

def test_short_term_get_recent():
    """get_recent(n) returns at most n most-recent entries."""
    stm = ShortTermMemory(max_size=10)
    for i in range(8):
        stm.add("user", f"msg {i}")
    recent = stm.get_recent(3)
    assert len(recent) == 3
    assert recent[-1].content == "msg 7"
    assert recent[0].content == "msg 5"
