# agent-memory

> Lightweight in-process memory for ReAct agents — short-term context + episodic recall.

Part of the **[production-agent-toolkit](https://github.com/darshjme)**.

---

## Install

```bash
git clone https://github.com/darshjme/agent-memory.git
cd agent-memory
pip install -e .
```

---

## Quickstart

### ShortTermMemory — sliding window buffer

```python
from agent_memory import ShortTermMemory

stm = ShortTermMemory(max_size=5)
stm.add("user", "What is 2+2?")
stm.add("assistant", "It is 4.")

# OpenAI-compatible messages
messages = stm.to_messages()
# [{"role": "user", "content": "What is 2+2?"},
#  {"role": "assistant", "content": "It is 4."}]
```

### AgentMemory — full ReAct agent memory

```python
from agent_memory import AgentMemory

memory = AgentMemory(short_term_size=10)

# Record observations and actions
memory.observe("Search returned 3 results.")
memory.act("Call tool: summarize(results)")

# Promote key insight to episodic
memory.remember("User prefers bullet-point answers.", tags=["preference"], importance=0.9)

# Retrieve episodic context
hits = memory.recall(tags=["preference"])

# Get full context snapshot (inject into prompt)
ctx = memory.context()
# {"short_term": [...], "episodic": [...]}

# Compress oldest short-term into episodic
memory.compress()

# Or compress with custom summariser
memory.compress(summarize_fn=lambda entries: "Summary: " + "; ".join(e.content for e in entries))

# Reset everything
memory.reset()
```

---

## API Reference

### `MemoryEntry`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | UUID auto-assigned |
| `role` | `str` | e.g. `"user"`, `"assistant"`, `"observation"` |
| `content` | `str` | Text content |
| `metadata` | `dict` | Arbitrary extra data |
| `timestamp` | `datetime` | UTC, auto-assigned |

### `ShortTermMemory(max_size=10)`
| Method | Returns | Description |
|--------|---------|-------------|
| `.add(role, content, **meta)` | `MemoryEntry` | Append entry (evicts oldest at capacity) |
| `.get_recent(n=5)` | `list[MemoryEntry]` | N most recent entries |
| `.to_messages()` | `list[dict]` | OpenAI-style `{role, content}` list |
| `.clear()` | `None` | Wipe buffer |

### `EpisodicMemory`
| Method | Returns | Description |
|--------|---------|-------------|
| `.remember(content, tags=[], importance=1.0)` | `MemoryEntry` | Store key event |
| `.recall(query=None, tags=[], limit=10)` | `list[MemoryEntry]` | Filter + rank by importance |
| `.forget(entry_id)` | `bool` | Remove by UUID |
| `.clear()` | `None` | Wipe all episodes |

### `AgentMemory(short_term_size=10)`
| Method | Description |
|--------|-------------|
| `.observe(content, **meta)` | Add observation to short-term |
| `.act(content, **meta)` | Add action to short-term |
| `.remember(content, tags, importance)` | Add to episodic |
| `.recall(query, tags, limit)` | Search episodic |
| `.context()` | Full snapshot dict |
| `.compress(summarize_fn=None)` | Compress short-term → episodic |
| `.reset()` | Clear everything |

---

## Part of the production-agent-toolkit

→ **[github.com/darshjme](https://github.com/darshjme)**
