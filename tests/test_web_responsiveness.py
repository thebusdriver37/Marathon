"""Fresh downloads, exact text reuse, and responsive HTML conversion."""
import asyncio
import copy
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock
import weakref

from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/routers"))
import marathon_web_search as search


class WebResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.body = "Original fact: harbor-731. " * 40
        self.content_type = "text/plain"
        self.status = 200
        self.requests = 0

        async def handler(request):
            self.requests += 1
            return web.Response(text=self.body, content_type=self.content_type, status=self.status,
                                headers={"ETag": '"unchanged-even-if-server-is-wrong"'})

        app = web.Application()
        app.router.add_get("/{path:.*}", handler)
        self.server = TestServer(app)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.fetch = search.WebFetchExecutor(replace(search.WebFetchSettings.from_env(), allow_private_networks=True))
        self.addAsyncCleanup(self.fetch.close)
        self.url = str(self.server.make_url("/page"))

    async def test_identical_bytes_reuse_text_but_always_download_again(self):
        with mock.patch.object(search, "_extract_to_markdown", wraps=search._extract_to_markdown) as convert:
            first = await self.fetch.fetch(self.url)
            second = await self.fetch.fetch(self.url)
            self.assertIs(second, first)
            self.assertEqual(convert.await_count, 1)
            self.assertEqual(self.requests, 2)
            self.assertIs(copy.deepcopy(first), first)
            self.assertEqual(json.loads(json.dumps(second)), str(first))

    async def test_changed_same_size_bytes_invalidate_even_with_same_etag(self):
        first = await self.fetch.fetch(self.url)
        self.body = self.body.replace("harbor-731", "cobalt-592")
        second = await self.fetch.fetch(self.url)
        self.assertNotEqual(second, first)
        self.assertIn("cobalt-592", second)
        self.assertNotIn("harbor-731", second)

    async def test_url_encoding_and_output_limit_are_part_of_identity(self):
        with mock.patch.object(search, "_extract_to_markdown", wraps=search._extract_to_markdown) as convert:
            retained = [await self.fetch.fetch(self.url)]
            retained.append(await self.fetch.fetch(self.url, max_chars=500))
            retained.append(await self.fetch.fetch(str(self.server.make_url("/other"))))
            self.content_type = "application/json"
            retained.append(await self.fetch.fetch(self.url))
            self.assertEqual(convert.await_count, 4)
            self.assertIn("(truncated to 500 chars)", retained[1].lower())

    async def test_results_are_not_kept_alive_by_the_reuse_table(self):
        first = await self.fetch.fetch(self.url)
        reference = weakref.ref(first)
        del first
        gc.collect()
        self.assertIsNone(reference())
        with mock.patch.object(search, "_extract_to_markdown", wraps=search._extract_to_markdown) as convert:
            await self.fetch.fetch(self.url)
            convert.assert_awaited_once()

    async def test_metadata_is_bounded_and_errors_never_return_stale_text(self):
        retained = [await self.fetch.fetch(str(self.server.make_url(f"/{n}"))) for n in range(12)]
        self.assertEqual(len(self.fetch._extracted), 8)
        self.status = 404
        with self.assertRaisesRegex(RuntimeError, "404"):
            await self.fetch.fetch(str(self.server.make_url("/11")))
        self.assertEqual(self.requests, 13)
        self.assertEqual(len(retained), 12)

    async def test_conversion_does_not_block_other_async_requests(self):
        entered, release = threading.Event(), threading.Event()

        def convert(*args):
            entered.set()
            release.wait(2)
            return "Converted reference text"

        with mock.patch.object(search, "_html_to_markdown", side_effect=convert):
            task = asyncio.create_task(search._extract_to_markdown(b"<html>body</html>", "text/html", self.url, 1000))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                self.assertFalse(task.done(), "Conversion blocked the loop until it finished")
            finally:
                release.set()
            self.assertIn("Converted reference text", await task)


if __name__ == "__main__":
    unittest.main()
