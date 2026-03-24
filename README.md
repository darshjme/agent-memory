<p align="center">
  <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjE2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImJnIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojMGQxMTE3Ii8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdHlsZT0ic3RvcC1jb2xvcjojMTMwZjIyIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxNjAiIGZpbGw9InVybCgjYmcpIi8+PHRleHQgeD0iNjAiIHk9Ijc4IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjQ4IiBmb250LXdlaWdodD0iYm9sZCIgZmlsbD0iI2U2ZWRmMyI+ZW5ncmFtPC90ZXh0Pjx0ZXh0IHg9IjYwIiB5PSIxMTQiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiM4Yjk0OWUiPlNob3J0LXRlcm0uIEVwaXNvZGljLiBQZXJzaXN0ZW50LjwvdGV4dD48Y2lyY2xlIGN4PSI3MzAiIGN5PSI1MCIgcj0iMTUiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2QyYThmZiIgc3Ryb2tlLXdpZHRoPSIyIi8+PGNpcmNsZSBjeD0iNzEwIiBjeT0iODAiIHI9IjEwIiBmaWxsPSJub25lIiBzdHJva2U9IiNkMmE4ZmYiIHN0cm9rZS13aWR0aD0iMiIvPjxjaXJjbGUgY3g9Ijc1MCIgY3k9IjgwIiByPSIxMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZDJhOGZmIiBzdHJva2Utd2lkdGg9IjIiLz48Y2lyY2xlIGN4PSI3MDAiIGN5PSIxMTAiIHI9IjgiIGZpbGw9IiNkMmE4ZmYiIG9wYWNpdHk9IjAuNiIvPjxjaXJjbGUgY3g9IjcyNSIgY3k9IjExNSIgcj0iOCIgZmlsbD0iI2QyYThmZiIgb3BhY2l0eT0iMC42Ii8+PGNpcmNsZSBjeD0iNzU1IiBjeT0iMTEwIiByPSI4IiBmaWxsPSIjZDJhOGZmIiBvcGFjaXR5PSIwLjYiLz48bGluZSB4MT0iNzMwIiB5MT0iNjUiIHgyPSI3MTAiIHkyPSI3MCIgc3Ryb2tlPSIjZDJhOGZmIiBzdHJva2Utd2lkdGg9IjEuNSIgb3BhY2l0eT0iMC42Ii8+PGxpbmUgeDE9IjczMCIgeTE9IjY1IiB4Mj0iNzUwIiB5Mj0iNzAiIHN0cm9rZT0iI2QyYThmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIG9wYWNpdHk9IjAuNiIvPjwvc3ZnPg==" alt="engram" width="800"/>
</p>

<p align="center">
  <a href="https://github.com/darshjme/engram/actions/workflows/ci.yml"><img src="https://github.com/darshjme/engram/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://pypi.org/project/agent-memory/"><img src="https://img.shields.io/pypi/v/agent-memory.svg" alt="PyPI version"/></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
  <img src="https://img.shields.io/badge/tests-13%20passing-brightgreen" alt="Tests"/>
  <img src="https://img.shields.io/badge/vector%20db-not%20required-purple" alt="No vector DB"/>
</p>

<p align="center"><b>Short-term working memory + episodic recall for agents — without a vector database.</b></p>

---

## What Is an Engram?

In neuroscience, an **engram** is the physical trace a memory leaves in the brain — the biological substrate of a specific experience. When you remember something, you're activating an engram.

Most agents have no engrams. They wake up blank every session, re-ask questions they've already answered, and forget what the user told them 10 minutes ago.

This library fixes that.

---

## Architecture

```mermaid
flowchart TD
    A[Agent Action / Observation] --> B[ShortTermMemory\nsliding window, fast dict]
    B -->|capacity exceeded| C[compress → EpisodicMemory]
    B -->|manual promote| C
    C[EpisodicMemory\ntimestamped events + tags + importance]
    C -->|recall query + tags| D[Relevant Episodes]
    B -->|to_messages| E[OpenAI-compatible\nmessages list]
    D --> F[context snapshot\ninjected into prompt]
    E --> F
```

Two tiers. No vector database. No external services. No configuration.

---

## Quick Start

```bash
git clone https://github.com/darshjme/engram
cd engram && pip install -e .
```

```python
from agent_memory import AgentMemory

memory = AgentMemory(short_term_size=10)

# Session 1 — record everything
memory.observe("User asked about quantum computing")
memory.act("Called search_tool('quantum computing basics')")
memory.observe("Found: superposition, entanglement, qubits")
memory.remember(
    "User prefers visual explanations with diagrams",
    tags=["preference"],
    importance=0.9
)
```

```python
# Session 2 — full context restored
hits = memory.recall(tags=["preference"])
# → [MemoryEntry: "User prefers visual explanations..."]

ctx = memory.context()
# {
#   "short_term": [...recent observations...],
#   "episodic": [...important remembered facts...]
# }

# Inject into your LLM prompt
messages = [
    {"role": "system", "content": f"Context: {ctx}"},
    {"role": "user", "content": user_input},
]
```

---

## Memory Session Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant M as engram

    U->>A: "Help me with quantum computing"
    A->>M: observe("user asked about QC")
    A->>M: act("searched Wikipedia")
    A->>M: remember("user prefers diagrams", importance=0.9)
    A-->>U: [answers with diagrams]

    Note over U,M: Session ends

    U->>A: "Continue from where we left off"
    A->>M: recall(tags=["preference"])
    M-->>A: ["user prefers visual explanations"]
    A->>M: context()
    M-->>A: {short_term: [...], episodic: [...]}
    A-->>U: [picks up with diagram-first approach]
```

---

## engram vs. Vector Database

| | engram | Vector DB (Qdrant/Pinecone) |
|--|--------|---------------------------|
| **Best for** | Session state, recent context, preferences | Semantic search across 100k+ documents |
| **Latency** | Microseconds (dict lookup) | 10–100ms (network + index) |
| **Setup** | Zero — `pip install -e .` | Database server, embeddings model, indexing pipeline |
| **Persistence** | In-process (extend to add file/DB backend) | Native |
| **Token overhead** | ~200 tokens for context snapshot | ~100 tokens per retrieved chunk |

**Rule of thumb:** If your agent needs to remember what happened 5 minutes ago — use engram. If it needs to search a knowledge base — use a vector DB.

---

## API Reference

### `ShortTermMemory(max_size=10)`

| Method | Returns | Description |
|--------|---------|-------------|
| `.add(role, content, **meta)` | `MemoryEntry` | Append (evicts oldest at capacity) |
| `.get_recent(n=5)` | `list[MemoryEntry]` | N most recent entries |
| `.to_messages()` | `list[dict]` | OpenAI-style `{role, content}` list |
| `.clear()` | `None` | Wipe buffer |

### `AgentMemory(short_term_size=10)`

| Method | Description |
|--------|-------------|
| `.observe(content, **meta)` | Add observation to short-term |
| `.act(content, **meta)` | Add action to short-term |
| `.remember(content, tags, importance)` | Promote to episodic store |
| `.recall(query=None, tags=[], limit=10)` | Search episodic by tags + importance |
| `.context()` | Full snapshot dict — inject into prompt |
| `.compress(summarize_fn=None)` | Compress short-term into episodic |
| `.reset()` | Clear everything |

---

## Part of Arsenal

```
verdict · sentinel · herald · engram · arsenal
```

| Repo | Purpose |
|------|---------|
| [verdict](https://github.com/darshjme/verdict) | Score your agents |
| [sentinel](https://github.com/darshjme/sentinel) | Stop runaway agents |
| [herald](https://github.com/darshjme/herald) | Semantic task router |
| [engram](https://github.com/darshjme/engram) | ← you are here |
| [arsenal](https://github.com/darshjme/arsenal) | The full pipeline |

---

## License

MIT © [Darshankumar Joshi](https://github.com/darshjme) · Built as part of the [Arsenal](https://github.com/darshjme/arsenal) toolkit.
