from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlsplit
from unittest import mock

from aiohttp import ClientConnectionError
from aiohttp import web
from aiohttp.test_utils import TestServer
from dataclasses import replace


ROOT_DIR = Path(__file__).resolve().parents[1]
ROUTER_DIR = ROOT_DIR / "scripts" / "routers"
sys.path.insert(0, str(ROUTER_DIR))

import marathon_web_search as web_search


class FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        status: int = 200,
        text: str = "",
    ) -> None:
        self.payload = payload
        self.status = status
        self.response_text = text

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def json(self, *, content_type: object = None) -> object:
        del content_type
        return self.payload

    async def text(self) -> str:
        return self.response_text


class SequenceClient:
    def __init__(self, *outcomes: FakeResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.request_count = 0
        self.request_urls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> FakeResponse:
        self.request_count += 1
        self.request_urls.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def settings(*, retries: int = 1, max_results: int = 8) -> web_search.WebSearchSettings:
    return web_search.WebSearchSettings(
        base_url="http://search.test",
        timeout_s=1.0,
        max_results=max_results,
        max_iterations=5,
        retries=retries,
    )


class WebSearchExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_html_honors_document_base_for_reference_links(self):
        body = b'''<html><head><base href="https://example.org/reference/"></head>
        <body><main><h1>Reference</h1><p>This is documentation about task execution and cleanup.
        Read the <a href="tasks.html?view=full#errors">task error reference</a> for the complete details.
        The reference explains how exceptions propagate between related asynchronous tasks.</p></main></body></html>'''
        output = await web_search._extract_to_markdown(body, "text/html", "https://example.org/other/index.html", 20000)
        self.assertIn("https://example.org/reference/tasks.html?view=full#errors", output)

    async def test_fetch_resolves_relative_links_against_redirect_destination(self):
        async def redirect(request):
            raise web.HTTPFound("/docs/guide/page.html")
        async def page(request):
            return web.Response(text='''<html><body><main><h1>Task documentation</h1>
                <p>This documentation explains asynchronous tasks and error propagation.
                Read the <a href="child.html#errors">child task reference</a> for details.
                See <a href="#cancellation">cancellation on this page</a> for cleanup.
                Understanding how related tasks behave is necessary to implement reliable programs.</p>
                </main></body></html>''', content_type="text/html")
        app = web.Application()
        app.router.add_get("/start", redirect)
        app.router.add_get("/docs/guide/page.html", page)
        async with TestServer(app) as server:
            executor = web_search.WebFetchExecutor(replace(web_search.WebFetchSettings.from_env(), allow_private_networks=True))
            try:
                output = await executor.fetch(str(server.make_url('/start')))
            finally:
                await executor.close()
            self.assertIn(str(server.make_url('/docs/guide/child.html#errors')), output)
            self.assertIn(str(server.make_url('/docs/guide/page.html#cancellation')), output)

    async def test_browser_marks_truncated_content(self):
        executor = web_search.WebFetchExecutor(replace(
            web_search.WebFetchSettings.from_env(), allow_private_networks=True,
        ))
        crawler = mock.Mock()
        crawler.arun = mock.AsyncMock(return_value=mock.Mock(markdown="evidence " * 200))
        with mock.patch.object(executor, "_ensure_crawl4ai", mock.AsyncMock(return_value=crawler)):
            text = await executor.browse("https://example.test/article", max_chars=500)
        self.assertIn("truncated to 500 chars", text)
        self.assertEqual(len(text.split("\n\n", 1)[1]), 500)

    async def test_explicit_bing_preserves_full_query_and_default_mix(self):
        client = SequenceClient(FakeResponse({"results": []}), FakeResponse({"results": []}))
        executor = web_search.WebSearchExecutor(settings(), http_client=client)
        query = '"digital amnesia" cognition & memory'
        await executor.search_with_diagnostics(query, engines=["bing"])
        await executor.search_with_diagnostics(query)
        first, second = [parse_qs(urlsplit(url).query) for url in client.request_urls]
        self.assertEqual(first["q"], [query])
        self.assertEqual(first["engines"], ["bing"])
        self.assertNotIn("engines", second)

    async def test_rejects_invalid_engine_selection_before_request(self):
        client = SequenceClient()
        executor = web_search.WebSearchExecutor(settings(), http_client=client)
        for engines in ["bing", [], [""], ["bing,google"], [True], ["bing"] * 6]:
            with self.subTest(engines=engines), self.assertRaises(ValueError):
                await executor.search_with_diagnostics("query", engines=engines)
        self.assertEqual(client.request_count, 0)

    def test_research_budget_default_and_explicit_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(web_search.WebSearchSettings.from_env().max_iterations, 12)
        with mock.patch.dict(os.environ, {"MARATHON_WEB_SEARCH_MAX_ITERS": "3"}):
            self.assertEqual(web_search.WebSearchSettings.from_env().max_iterations, 3)

    def test_all_engine_failures_are_reported_even_when_google_succeeds(self):
        warnings = web_search._provider_warnings({"google cse"}, ["brave: CAPTCHA"])
        self.assertIn("brave: CAPTCHA", warnings[0])
        self.assertIn("Results supplied by: google cse", warnings[0])
        self.assertNotIn("public endpoint", warnings[0])

    async def test_pdf_extraction_rejects_missing_parser_and_empty_text(self):
        with mock.patch.object(web_search.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "pdftotext"):
                await web_search._extract_to_markdown(b"%PDF-1.4", "application/pdf", "https://example.test/paper", 1000)
        process = mock.Mock(returncode=0)
        process.communicate = mock.AsyncMock(return_value=(b"", b""))
        with (
            mock.patch.object(web_search.shutil, "which", return_value="/usr/bin/pdftotext"),
            mock.patch.object(web_search.asyncio, "create_subprocess_exec", mock.AsyncMock(return_value=process)),
        ):
            with self.assertRaisesRegex(RuntimeError, "OCR"):
                await web_search._extract_to_markdown(b"%PDF-1.4", "application/octet-stream", "https://example.test/paper", 1000)

    async def test_fetch_redirect_errors_and_size_limit_over_http(self):
        async def redirect(request):
            raise web.HTTPFound('/article')
        async def article(request):
            return web.Response(text="Research evidence " * 100, content_type="text/plain")
        app = web.Application()
        app.router.add_get('/redirect', redirect)
        app.router.add_get('/article', article)
        async with TestServer(app) as server:
            executor = web_search.WebFetchExecutor(replace(
                web_search.WebFetchSettings.from_env(), allow_private_networks=True,
            ))
            try:
                result = await executor.fetch(str(server.make_url('/redirect')), max_chars=500)
                self.assertIn(str(server.make_url('/article')), result)
                self.assertIn('truncated to 500 chars', result)
                with self.assertRaisesRegex(RuntimeError, 'HTTP 404'):
                    await executor.fetch(str(server.make_url('/missing')))
                executor.settings = replace(executor.settings, max_bytes=100)
                with self.assertRaisesRegex(RuntimeError, 'exceeds'):
                    await executor.fetch(str(server.make_url('/article')))
            finally:
                await executor.close()

    async def test_fetch_blocks_private_targets_by_default(self):
        for url in ['http://127.0.0.1/private', 'http://[::1]/', 'http://169.254.169.254/']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                await web_search._validate_fetch_url(url, allow_private_networks=False)

    def test_browser_discovery_does_not_import_the_crawler(self) -> None:
        subprocess.run([
            sys.executable, "-c",
            "import sys; import marathon_web_search as web; "
            "web.web_browse_available(); assert 'crawl4ai' not in sys.modules",
        ], env=dict(os.environ, PYTHONPATH=str(ROUTER_DIR)), check=True)

    async def test_browser_loads_on_demand_and_reuses_and_closes_the_crawler(self) -> None:
        executor = web_search.WebFetchExecutor(web_search.WebFetchSettings.from_env())
        crawler = mock.MagicMock()
        crawler.__aenter__ = mock.AsyncMock()
        crawler.__aexit__ = mock.AsyncMock()
        module = mock.Mock()
        module.AsyncWebCrawler.return_value = crawler
        with (
            mock.patch.object(web_search, "HAS_CRAWL4AI", True),
            mock.patch.dict(os.environ, {"MARATHON_WEB_BROWSE_ENABLE": "1"}),
            mock.patch.object(web_search.importlib, "import_module", return_value=module) as load,
        ):
            self.assertTrue(web_search.web_browse_available())
            load.assert_not_called()
            self.assertIs(await executor._ensure_crawl4ai(), crawler)
            self.assertIs(await executor._ensure_crawl4ai(), crawler)
            load.assert_called_once_with("crawl4ai")
            crawler.__aenter__.assert_awaited_once()
            await executor.close()
            crawler.__aexit__.assert_awaited_once()

    async def test_missing_disabled_and_broken_browser_dependencies(self) -> None:
        for installed, enabled in ((False, "1"), (True, "0"), (True, "1")):
            executor = web_search.WebFetchExecutor(web_search.WebFetchSettings.from_env())
            with (
                self.subTest(installed=installed, enabled=enabled),
                mock.patch.object(web_search, "HAS_CRAWL4AI", installed),
                mock.patch.dict(os.environ, {"MARATHON_WEB_BROWSE_ENABLE": enabled}),
                mock.patch.object(web_search.importlib, "import_module", side_effect=ImportError("dependency")) as load,
            ):
                if installed and enabled == "1":
                    with self.assertRaisesRegex(RuntimeError, "could not load Crawl4AI"):
                        await executor._ensure_crawl4ai()
                else:
                    self.assertFalse(web_search.web_browse_available())
                    self.assertIsNone(await executor._ensure_crawl4ai())
                    load.assert_not_called()

    def test_tool_schema_exposes_supported_time_ranges(self) -> None:
        parameters = web_search.web_search_function_tool()["parameters"]

        self.assertEqual(
            parameters["properties"]["time_range"]["enum"],
            ["day", "week", "month", "year"],
        )

    async def test_returns_unique_http_results_and_preserves_source_metadata(self) -> None:
        payload = {
            "results": [
                {"title": "Not a web result", "url": "ftp://example.test/file"},
                {"title": "Malformed URL", "url": "http://[invalid"},
                {
                    "title": "Official docs",
                    "url": "https://docs.example.test/guide",
                    "content": "The guide.",
                    "engines": ["google cse", "bing", "google cse"],
                    "publishedDate": "2026-08-24T00:00:00Z",
                },
                {
                    "title": "Duplicate",
                    "url": "https://docs.example.test/guide",
                    "content": "Duplicate result.",
                },
                {
                    "title": "Reference",
                    "url": "https://reference.example.test/",
                    "content": "The reference.",
                    "engine": "bing",
                },
                {
                    "title": "Over the cap",
                    "url": "https://extra.example.test/",
                },
            ]
        }
        client = SequenceClient(FakeResponse(payload))
        executor = web_search.WebSearchExecutor(
            settings(max_results=2),
            http_client=client,  # type: ignore[arg-type]
        )

        results = await executor.search("official documentation")

        self.assertEqual(
            [result.title for result in results],
            ["Official docs", "Reference"],
        )
        self.assertEqual(results[0].engine, "google cse, bing")
        self.assertEqual(results[0].published_date, "2026-08-24T00:00:00Z")
        self.assertIn("2026-08-24T00:00:00Z", results[0].to_text_block(1))

    async def test_passes_validated_time_range_to_searxng(self) -> None:
        client = SequenceClient(FakeResponse({"results": []}))
        executor = web_search.WebSearchExecutor(
            settings(),
            http_client=client,  # type: ignore[arg-type]
        )

        await executor.search("recent release", time_range="Week")

        query = parse_qs(urlsplit(client.request_urls[0]).query)
        self.assertEqual(query["time_range"], ["week"])

    async def test_rejects_an_invalid_time_range_before_network_io(self) -> None:
        client = SequenceClient(FakeResponse({"results": []}))
        executor = web_search.WebSearchExecutor(
            settings(),
            http_client=client,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(ValueError, "day, week, month, year"):
            await executor.search("recent release", time_range="all_time")

        self.assertEqual(client.request_count, 0)

    async def test_dedupes_url_variants_but_keeps_semantic_query_parameters(self) -> None:
        client = SequenceClient(
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Ranked first",
                            "url": (
                                "https://www.Example.test/docs/?b=2&utm_source=search"
                                "&a=1#install"
                            ),
                            "engines": ["google cse"],
                        },
                        {
                            "title": "Tracking duplicate",
                            "url": "http://example.test/docs?a=1&b=2",
                            "engines": ["bing"],
                        },
                        {
                            "title": "A distinct page",
                            "url": "https://example.test/docs?b=2&page=2&a=1",
                            "engines": ["google cse"],
                        },
                    ]
                }
            )
        )
        executor = web_search.WebSearchExecutor(
            settings(),
            http_client=client,  # type: ignore[arg-type]
        )

        results = await executor.search("documentation")

        self.assertEqual(
            [result.title for result in results],
            ["Ranked first", "A distinct page"],
        )
        self.assertIn("utm_source=search", results[0].url)

    async def test_does_not_assume_unconfigured_google_is_failing(self) -> None:
        client = SequenceClient(
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Bing result",
                            "url": "https://example.test/result",
                            "engines": ["bing"],
                        }
                    ]
                }
            )
        )
        executor = web_search.WebSearchExecutor(
            settings(),
            http_client=client,  # type: ignore[arg-type]
        )

        outcome = await executor.search_with_diagnostics("query")
        formatted = web_search.format_results_for_model(
            "query",
            outcome.results,
            warnings=outcome.warnings,
        )

        self.assertEqual(outcome.warnings, ())
        self.assertNotIn("WARNING:", formatted)

    async def test_surfaces_google_cse_rate_limiting_with_fallback_results(self) -> None:
        client = SequenceClient(
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Bing result",
                            "url": "https://example.test/result",
                            "engines": ["bing"],
                        }
                    ],
                    "unresponsive_engines": [
                        ["google cse", "HTTP error 429: too many requests"]
                    ],
                }
            )
        )
        executor = web_search.WebSearchExecutor(
            settings(),
            http_client=client,  # type: ignore[arg-type]
        )

        outcome = await executor.search_with_diagnostics("query")

        self.assertIn("Partial search results", outcome.warnings[0])
        self.assertIn("HTTP error 429", outcome.warnings[0])

    async def test_reports_upstream_engine_failures_instead_of_empty_results(self) -> None:
        client = SequenceClient(
            FakeResponse(
                {
                    "results": [],
                    "unresponsive_engines": [
                        ["google cse", "timeout"],
                        ["bing", "CAPTCHA"],
                    ],
                }
            )
        )
        executor = web_search.WebSearchExecutor(
            settings(),
            http_client=client,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "google cse: timeout; bing: CAPTCHA",
        ):
            await executor.search("query")

    async def test_retries_a_transient_connection_failure_once(self) -> None:
        client = SequenceClient(
            ClientConnectionError("connection reset"),
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Recovered",
                            "url": "https://example.test/recovered",
                            "content": "Search recovered.",
                        }
                    ]
                }
            ),
        )
        executor = web_search.WebSearchExecutor(
            settings(retries=1),
            http_client=client,  # type: ignore[arg-type]
        )

        with (
            mock.patch.object(asyncio, "sleep", new=mock.AsyncMock()),
            mock.patch.object(web_search.LOG, "warning") as warning,
        ):
            results = await executor.search("query")

        self.assertEqual(results[0].title, "Recovered")
        self.assertEqual(client.request_count, 2)
        warning.assert_called_once()

    async def test_does_not_retry_a_client_error_response(self) -> None:
        client = SequenceClient(
            FakeResponse(
                {"error": "forbidden"},
                status=403,
                text=json.dumps({"error": "forbidden"}),
            )
        )
        executor = web_search.WebSearchExecutor(
            settings(retries=3),
            http_client=client,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            await executor.search("query")

        self.assertEqual(client.request_count, 1)


if __name__ == "__main__":
    unittest.main()
