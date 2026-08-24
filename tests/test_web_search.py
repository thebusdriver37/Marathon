from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlsplit
from unittest import mock

from aiohttp import ClientConnectionError


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

    async def test_warns_when_results_silently_fall_back_to_bing(self) -> None:
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

        self.assertIn("Google CSE contributed no results", outcome.warnings[0])
        self.assertIn("fallback-only via bing", outcome.warnings[0])
        self.assertIn("WARNING:", formatted)

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

        self.assertIn("rate-limited or suspended", outcome.warnings[0])
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
