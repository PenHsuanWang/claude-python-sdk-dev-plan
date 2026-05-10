# Chapter 07 — Session Management

> **Goal:** Build the in-memory session store that persists conversation history across multiple HTTP requests from the same user.

---

## 7.1 Why Session Management?

HTTP is stateless.  Each `POST /api/v1/chat` request is independent.  But Claude needs the **full conversation history** in `messages` to answer follow-up questions like:

> "Which region had the highest revenue?" (first question)  
> "And by how much did it exceed its target?" (follow-up — requires context)

The session store bridges this gap: it maps a `session_id` to an `AgentSession` that accumulates messages across turns.

```
Request 1: session_id="u1" → get_or_create("u1") → AgentSession(messages=[])
Request 2: session_id="u1" → get_or_create("u1") → AgentSession(messages=[...turn 1 history...])
Request 3: session_id="u2" → get_or_create("u2") → AgentSession(messages=[])   # new user
```

---

## 7.2 The `InMemorySessionStore`

```python
# app/services/memory.py
"""In-memory session store (Infrastructure layer).

For MVP/development use.  Swap to Redis or PostgreSQL for production
by replacing this module only — AgentService is never touched.

Thread-safety note: a single-worker uvicorn process is effectively
single-threaded for asyncio coroutines, so a plain dict is safe here.
"""

from app.domain.models import AgentSession


class InMemorySessionStore:
    """Key-value store mapping session_id → AgentSession."""

    def __init__(self) -> None:
        self._store: dict[str, AgentSession] = {}

    def get_or_create(self, session_id: str) -> AgentSession:
        """Return an existing session or create a fresh one."""
        if session_id not in self._store:
            self._store[session_id] = AgentSession(session_id=session_id)
        return self._store[session_id]

    def save(self, session: AgentSession) -> None:
        self._store[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._store)


# Module-level singleton shared across all requests within a process lifetime
session_store = InMemorySessionStore()
```

---

## 7.3 Usage in the Chat Handler

The session store is used exclusively in the presentation layer:

```python
# app/api/v1/chat.py (excerpt)
from app.services.memory import session_store

async def chat(request: ChatRequest) -> ChatResponse:
    # 1. Load or create the session
    session = session_store.get_or_create(request.session_id)
    
    # 2. Append the user's message
    session.add_user_message(request.user_message)
    
    # 3. Run the agent loop (mutates session.messages in place)
    answer = await agent_service.run(session)
    
    # 4. Persist the updated session (with all new turns appended)
    session_store.save(session)
    
    return ChatResponse(response=answer, status="success")
```

Note that `agent_service.run(session)` mutates `session.messages` in place during the loop, so `session_store.save(session)` persists all new turns (tool calls, tool results, final answer) automatically.

---

## 7.4 Multi-Turn Conversation in Practice

```bash
# Turn 1: Ask a general question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-alice", "user_message": "What documents do you have?"}'

# {"response": "You have 2 documents: report.txt and notes.md", "status": "success"}

# Turn 2: Follow-up (session history preserved via session_id)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-alice", "user_message": "Summarize report.txt"}'

# {"response": "Here is a summary of report.txt: ...", "status": "success"}

# Turn 3: Context-dependent follow-up
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-alice", "user_message": "Which section was most important?"}'

# Claude remembers the report content from turn 2 — no need to re-read the file
```

---

## 7.5 Production Upgrade Path

The MVP uses a Python dict (in-process memory).  For production:

| Concern | Problem | Solution |
|---------|---------|---------|
| Multi-worker | Dict not shared across uvicorn workers | Redis (shared external store) |
| Restarts | Dict is lost on process restart | Redis or PostgreSQL (persistent) |
| Memory growth | Sessions accumulate forever | TTL-based expiry in Redis |
| Scale | One process can't handle all traffic | Redis or distributed session store |

Because `AgentService` depends only on `AgentSession` (domain layer), swapping the backend is surgical:

```python
# Future: app/services/memory.py — Redis-backed version
import json
import redis.asyncio as redis
from app.domain.models import AgentSession

class RedisSessionStore:
    def __init__(self, url: str, ttl_seconds: int = 3600):
        self._redis = redis.from_url(url)
        self._ttl = ttl_seconds

    async def get_or_create(self, session_id: str) -> AgentSession:
        raw = await self._redis.get(session_id)
        if raw:
            data = json.loads(raw)
            session = AgentSession(session_id=session_id)
            session.messages = data["messages"]
            return session
        return AgentSession(session_id=session_id)

    async def save(self, session: AgentSession) -> None:
        await self._redis.setex(
            session.session_id,
            self._ttl,
            json.dumps({"messages": session.messages}),
        )
```

Only `memory.py` changes. `agent.py`, `chat.py`, and all domain code remain identical.

---

## 7.6 Session ID Design Choices

The `session_id` is provided by the client — the server never generates it.  This is intentional:

- **Flexibility:** clients can use UUID v4, user ID + device ID, or any scheme
- **Determinism:** a client can reconnect to an existing session after a network failure
- **Simplicity:** no "create session" endpoint needed — sessions are lazily created

**Recommended client-side format:**
```
{user_id}-{device_id}-{conversation_id}
# e.g., "alice-web-20240310-abc123"
```

---

## 7.7 Memory Growth Considerations (MVP)

In the in-memory MVP, sessions grow indefinitely.  For a demo or internal tool, this is fine.  For a production service, add:

```python
# Simple LRU-style cleanup (add to InMemorySessionStore)
import time
from collections import OrderedDict

class BoundedSessionStore:
    def __init__(self, max_sessions: int = 1000) -> None:
        self._store: OrderedDict[str, tuple[float, AgentSession]] = OrderedDict()
        self._max = max_sessions

    def get_or_create(self, session_id: str) -> AgentSession:
        if session_id in self._store:
            # Move to end (most recently used)
            self._store.move_to_end(session_id)
            return self._store[session_id][1]
        
        session = AgentSession(session_id=session_id)
        self._store[session_id] = (time.time(), session)
        
        # Evict oldest if over limit
        if len(self._store) > self._max:
            self._store.popitem(last=False)
        
        return session
```

---

## Chapter Summary

| Concept | Key Takeaway |
|---|---|
| `session_id` | Client-provided key; server never generates it |
| `get_or_create` | Lazily creates sessions — no "create session" endpoint needed |
| Mutation in place | `agent_service.run(session)` mutates `session.messages` directly |
| Production upgrade | Swap `InMemorySessionStore` for Redis — only `memory.py` changes |
| Memory growth | Add TTL or LRU eviction for long-running services |

---

Next: [Chapter 08 → FastAPI Integration](./08_fastapi_integration.md)
