from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "routers"))
from marathon_search_coordinator import SearchCoordinator, SearchUnavailable


def result(name="google"):
    return {"results": [{"title": "Source", "url": "https://example.org/source", "engine": name}]}


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "search.db"
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def coordinator(self, **kw):
        return SearchCoordinator(self.path, "http://search.test", interval=0, **kw)

    async def fetch(self, params):
        self.calls.append(params)
        await asyncio.sleep(0.02)
        return result(params["engines"])

    async def test_concurrent_instances_coalesce_identical_requests(self):
        results = await asyncio.gather(*(self.coordinator().request({"q": "same"}, self.fetch) for _ in range(12)))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(sum(bool(r.get("marathon_cache_hit")) for r in results), 11)

    async def test_freshness_and_engine_filters_do_not_collide(self):
        c = self.coordinator()
        for params in ({"q": "a"}, {"q": "a", "time_range": "day"}, {"q": "a", "engines": "bing"}):
            await c.request(params, self.fetch)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.calls[-1]["engines"], "bing")

    async def test_endpoints_do_not_share_cache_or_health(self):
        await self.coordinator().request({"q": "a"}, self.fetch)
        await SearchCoordinator(self.path, "http://other", interval=0).request({"q": "a"}, self.fetch)
        self.assertEqual(len(self.calls), 2)

    async def test_custom_default_engines_do_not_reuse_other_mix(self):
        await self.coordinator(engines=("google",)).request({"q": "a"}, self.fetch)
        await self.coordinator(engines=("bing",)).request({"q": "a"}, self.fetch)
        self.assertEqual([p["engines"] for p in self.calls], ["google", "bing"])

    async def test_captcha_falls_back_and_skips_failed_engine_next_query(self):
        async def fetch(params):
            self.calls.append(params)
            if params["engines"] == "google":
                return {"results": [], "unresponsive_engines": [["google", "CAPTCHA"]]}
            return result(params["engines"])
        c = self.coordinator()
        await c.request({"q": "a"}, fetch)
        await c.request({"q": "b"}, fetch)
        self.assertEqual([p["engines"] for p in self.calls], ["google", "brave", "google cse"])

    async def test_all_blocked_fails_fast_without_new_requests(self):
        async def fetch(params):
            self.calls.append(params)
            return {"results": [], "unresponsive_engines": [[params["engines"], "too many requests"]]}
        c = self.coordinator()
        with self.assertRaises(SearchUnavailable):
            await c.request({"q": "a"}, fetch)
        with self.assertRaisesRegex(SearchUnavailable, "cooling down"):
            await c.request({"q": "reworded"}, fetch)
        self.assertEqual(len(self.calls), 3)

    async def test_cooldown_probe_recovers_and_resets_failure_count(self):
        async def failed(params):
            return {"results": [], "unresponsive_engines": [["google", "CAPTCHA"]]}
        c = self.coordinator(engines=("google",))
        with self.assertRaises(SearchUnavailable):
            await c.request({"q": "a"}, failed)
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE engines SET ready=0")
        await c.request({"q": "b"}, self.fetch)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT failures FROM engines").fetchone()[0], 0)

    async def test_pacing_applies_across_clients(self):
        c = SearchCoordinator(self.path, "http://search.test", interval=0.12, engines=("google",))
        times = []
        async def timed(params):
            times.append(time.time())
            return result()
        await c.request({"q": "a"}, timed)
        await c.request({"q": "b"}, timed)
        self.assertGreaterEqual(times[1] - times[0], 0.115)

    async def test_cancellation_releases_lock(self):
        entered = asyncio.Event()
        async def stuck(params):
            entered.set()
            await asyncio.sleep(100)
        task = asyncio.create_task(self.coordinator().request({"q": "a"}, stuck))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(self.coordinator().request({"q": "b"}, self.fetch), 1)

    async def test_queue_timeout_does_not_dispatch(self):
        entered = asyncio.Event()
        async def slow(params):
            entered.set()
            await asyncio.sleep(0.3)
            return result()
        task = asyncio.create_task(self.coordinator().request({"q": "a"}, slow))
        await entered.wait()
        with self.assertRaisesRegex(SearchUnavailable, "queue"):
            await self.coordinator(wait_timeout=0.1).request({"q": "b"}, self.fetch)
        await task
        self.assertEqual(len(self.calls), 0)

    async def test_legitimate_empty_result_is_not_an_engine_failure(self):
        async def empty(params):
            return {"results": []}
        c = self.coordinator()
        self.assertEqual((await c.request({"q": "a"}, empty))["results"], [])
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT SUM(failures) FROM engines").fetchone()[0], 0)

    async def test_expired_cache_refetches(self):
        c = self.coordinator(cache_ttl=0)
        await c.request({"q": "a"}, self.fetch)
        await c.request({"q": "a"}, self.fetch)
        self.assertEqual(len(self.calls), 2)

    async def test_transport_failure_and_malformed_result_release_lock(self):
        async def malformed(params):
            return {"error": "bad gateway"}
        with self.assertRaisesRegex(SearchUnavailable, "malformed"):
            await self.coordinator().request({"q": "a"}, malformed)
        with self.assertRaisesRegex(SearchUnavailable, "transport failure"):
            await self.coordinator().request({"q": "b"}, self.fetch)
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE engines SET ready=0")
        await self.coordinator().request({"q": "b"}, self.fetch)

    async def test_separate_processes_share_single_flight(self):
        code = '''
import asyncio, json, sys
from pathlib import Path
from marathon_search_coordinator import SearchCoordinator
async def main():
    async def fetch(params):
        with open(sys.argv[2], 'a') as f: f.write('request\\n')
        await asyncio.sleep(0.1)
        return {'results': [{'url': 'https://example.org'}]}
    await SearchCoordinator(Path(sys.argv[1]), 'http://search.test', interval=0).request({'q':'same'},fetch)
asyncio.run(main())
'''
        import os
        env = dict(os.environ, PYTHONPATH=sys.path[0])
        count = Path(self.temp.name) / "calls"
        processes = [await asyncio.create_subprocess_exec(sys.executable, "-c", code, str(self.path), str(count), env=env) for _ in range(4)]
        self.assertEqual(await asyncio.gather(*(p.wait() for p in processes)), [0]*4)
        self.assertEqual(count.read_text().splitlines(), ["request"])

    async def test_sustained_http_burst_with_duplicates_and_rate_limits(self):
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        from marathon_web_search import WebSearchExecutor, WebSearchSettings
        last = {}
        counts = {"requests": 0, "blocked": 0}

        async def handler(request):
            counts["requests"] += 1
            name = request.query.get("engines", "google")
            now = time.monotonic()
            blocked = now - last.get(name, 0) < 0.04
            last[name] = now
            if blocked:
                counts["blocked"] += 1
                return web.json_response({"results": [], "unresponsive_engines": [[name, "too many requests"]]})
            return web.json_response(result(name))

        app = web.Application()
        app.router.add_get("/search", handler)
        async with TestServer(app) as server:
            endpoint = str(server.make_url("/")).rstrip("/")
            async def run(coordinate, distinct=12, prefix="query"):
                clients = [WebSearchExecutor(WebSearchSettings(endpoint, 2, 8, 12, retries=0)) for _ in range(4)]
                if coordinate:
                    for client in clients:
                        client.coordinator = SearchCoordinator(self.path, endpoint, interval=0.06)
                try:
                    # 48 logical searches, 12 distinct keys, four callers.
                    return await asyncio.gather(*(clients[i % 4].search_with_diagnostics(f"{prefix} {i % distinct}") for i in range(48)), return_exceptions=True)
                finally:
                    await asyncio.gather(*(c.close() for c in clients))
            control = await run(False)
            self.assertTrue(any(isinstance(x, Exception) for x in control))
            self.load_metrics = {"control": {**counts, "successes": sum(not isinstance(x, Exception) for x in control)}}
            last.clear()
            counts.update(requests=0, blocked=0)
            candidate = await run(True)
            self.assertFalse(any(isinstance(x, Exception) for x in candidate))
            self.assertEqual(counts, {"requests": 12, "blocked": 0})
            self.load_metrics["candidate"] = {**counts, "successes": sum(not isinstance(x, Exception) for x in candidate)}
            counts.update(requests=0, blocked=0)
            unique = await run(True, distinct=48, prefix="unique")
            self.assertFalse(any(isinstance(x, Exception) for x in unique))
            self.assertEqual(counts, {"requests": 48, "blocked": 0})
            self.load_metrics["candidate_unique_queries"] = {**counts, "successes": 48}


if __name__ == "__main__":
    unittest.main()
