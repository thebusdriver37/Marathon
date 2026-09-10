"""Host-wide search pacing, single-flight cache, and engine cooldowns.

The advisory lock is held across a request, but acquired without blocking the
event loop. Process death releases it; SQLite transactions never span network IO.
Only callers using the same endpoint share health and cache entries.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Awaitable, Callable


class SearchUnavailable(RuntimeError):
    """Infrastructure failure: rewording the query will not fix it."""


class SearchCoordinator:
    def __init__(self, path: Path, endpoint: str, *, interval: float = 2.0,
                 cache_ttl: float = 300.0, wait_timeout: float = 45.0,
                 engines: tuple[str, ...] = ("google", "brave", "google cse")):
        self.path = path
        self.endpoint = endpoint
        self.interval = max(0.0, interval)
        self.cache_ttl = max(0.0, cache_ttl)
        self.wait_timeout = max(0.1, wait_timeout)
        self.engines = engines

    async def request(self, params: dict[str, str],
                      fetch: Callable[[dict[str, str]], Awaitable[Any]]) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + self.wait_timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise SearchUnavailable("Shared search queue is busy. Fetch a known URL or retry later.")
                    await asyncio.sleep(0.05)
            with closing(sqlite3.connect(self.path)) as db:
                os.chmod(self.path, 0o600)
                db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, expires REAL, body TEXT)")
                db.execute("CREATE TABLE IF NOT EXISTS engines (endpoint TEXT, name TEXT, failures INTEGER, ready REAL, last REAL, PRIMARY KEY(endpoint,name))")
                db.execute("DELETE FROM cache WHERE expires <= ?", (time.time(),))
                db.commit()
                return await self._request_locked(db, params, fetch)
        finally:
            os.close(fd)

    async def _request_locked(self, db, params, fetch):
        key = hashlib.sha256(json.dumps(
            [self.endpoint, params, None if params.get("engines") else self.engines],
            sort_keys=True,
        ).encode()).hexdigest()
        cached = db.execute("SELECT body FROM cache WHERE key=? AND expires>?", (key, time.time())).fetchone()
        if cached:
            body = json.loads(cached[0])
            body["marathon_cache_hit"] = True
            return body
        explicit = params.get("engines")
        candidates = explicit.split(",") if explicit else list(self.engines)
        states = {row[0]: row[1:] for row in db.execute(
            "SELECT name,failures,ready,last FROM engines WHERE endpoint=?", (self.endpoint,))}
        now = time.time()
        endpoint_ready = states.get("*transport*", (0, 0, 0))[1]
        if endpoint_ready > now:
            raise SearchUnavailable(f"Search service is cooling down after a transport failure for approximately {int(endpoint_ready-now)+1}s. Fetch known URLs or retry later.")
        available = [name for name in candidates if states.get(name, (0, 0, 0))[1] <= now]
        if not available:
            delay = max(1, int(min(states[n][1] for n in candidates) - now) + 1)
            raise SearchUnavailable(f"Selected search engines are cooling down for approximately {delay}s. Rewording will not help; fetch known sources or retry later.")
        # Rotate healthy engines so every query does not hit every provider.
        if not explicit:
            available.sort(key=lambda name: (states.get(name, (0, 0, 0))[0], states.get(name, (0, 0, 0))[2], candidates.index(name)))
        groups = [available] if explicit else [[name] for name in available]
        failures = []
        body = {"results": []}
        for names in groups:
            last = max((states.get(n, (0, 0, 0))[2] for n in names), default=0)
            await asyncio.sleep(max(0, last + self.interval - time.time()))
            started = time.time()
            # Persist dispatch before IO so a cancelled caller cannot bypass pacing.
            for name in names:
                count, ready, _ = states.get(name, (0, 0, 0))
                db.execute("INSERT OR REPLACE INTO engines VALUES (?,?,?,?,?)",
                           (self.endpoint, name, count, ready, started))
            db.commit()
            try:
                body = await fetch(dict(params, engines=",".join(names)))
                if not isinstance(body, dict) or not isinstance(body.get("results"), list):
                    raise SearchUnavailable("Search endpoint returned malformed results.")
            except Exception as exc:
                # Transport failures concern the shared endpoint, not query relevance.
                count = states.get("*transport*", (0, 0, 0))[0] + 1
                db.execute("INSERT OR REPLACE INTO engines VALUES (?,?,?,?,?)", (
                    self.endpoint, "*transport*", count,
                    time.time() + min(300, 5 * 2 ** min(count-1, 6)), started))
                db.commit()
                raise SearchUnavailable(f"Search transport failed: {exc}. Fetch a known URL or retry later.") from exc
            db.execute("DELETE FROM engines WHERE endpoint=? AND name='*transport*'", (self.endpoint,))
            raw = body.get("unresponsive_engines", [])
            if not isinstance(raw, list):
                raw = []
            bad = {str(x[0]): str(x[1]) for x in raw if isinstance(x, (list, tuple)) and len(x) >= 2}
            for name in names:
                count = states.get(name, (0, 0, 0))[0] + 1 if name in bad else 0
                reason = bad.get(name, "").lower()
                base = 300 if any(s in reason for s in ("captcha", "too many", "denied", "suspended", "429")) else 10
                ready = time.time() + min(3600, base * 2 ** min(count - 1, 6)) if count else 0
                db.execute("INSERT OR REPLACE INTO engines VALUES (?,?,?,?,?)",
                           (self.endpoint, name, count, ready, started))
            db.commit()
            failures.extend([[n, r] for n, r in bad.items()])
            if body["results"]:
                break
        body["unresponsive_engines"] = failures
        if not body["results"] and failures:
            raise SearchUnavailable("No usable results; upstream engines failed: " + "; ".join(f"{n}: {r}" for n,r in failures) + ". Fetch known sources or retry after cooldown; rewording cannot fix blocked engines.")
        # Cache successful responses briefly, including legitimate empty searches.
        ttl = self.cache_ttl if body["results"] else min(30, self.cache_ttl)
        encoded = json.dumps(body)
        # Bound disk use even when an upstream returns unexpectedly large snippets.
        if len(encoded.encode("utf-8")) <= 65536:
            db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, time.time()+ttl, encoded))
        db.execute("DELETE FROM cache WHERE key IN (SELECT key FROM cache ORDER BY expires DESC LIMIT -1 OFFSET 256)")
        db.commit()
        return body
