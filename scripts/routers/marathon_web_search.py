"""Web-search support for the Marathon Codex local router.

The router exposes a single ``web_search`` function tool to the local model.
When the model calls it, ``WebSearchExecutor`` queries a self-hosted SearXNG
instance and returns formatted results that the model can quote in its reply.

This module also handles the translation between the local view (regular
function-call / function-call-output items used by llama.cpp) and the Codex
view (``web_search_call`` ResponseItems rendered natively by the TUI).

Configuration is via environment variables so OSS users can deploy the same
router unchanged:

  MARATHON_SEARXNG_URL          base URL of the SearXNG instance
                                (default: http://127.0.0.1:18093)
  MARATHON_WEB_SEARCH_TIMEOUT   per-search timeout in seconds (default: 15)
  MARATHON_WEB_SEARCH_MAX_RESULTS  results to forward to the model (default: 8)
  MARATHON_WEB_SEARCH_MAX_ITERS    cap on tool-call iterations (default: 5)
  MARATHON_WEB_FETCH_ALLOW_PRIVATE
                                set to "1" to allow fetches to private/local IPs

The router strips the OpenAI-style ``web_search`` tool object Codex sends,
replaces it with a ``function`` tool the local model understands, and runs the
tool-call loop transparently inside a single Codex request.
"""

from __future__ import annotations

import asyncio
import copy
import ipaddress
import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.parse import urljoin
from urllib.parse import urlsplit

from aiohttp import ClientSession
from aiohttp import ClientTimeout

LOG = logging.getLogger("marathon.web_search")

WEB_SEARCH_TOOL_NAME = "web_search"
WEB_FETCH_TOOL_NAME = "web_fetch"

WEB_SEARCH_TOOL_DESCRIPTION = (
    "Search the public web via SearXNG and return ranked results with title, "
    "URL, and a short snippet. Use this whenever you need information you do "
    "not already know, current events, library changelogs, error messages, or "
    "anything else that might have changed since training. Pass a focused "
    "natural-language query. Snippets are short — when you need the full text "
    "of a page (verbatim quotes, full article, code from a docs site), follow "
    "up with web_fetch on the chosen URL."
)

WEB_FETCH_TOOL_DESCRIPTION = (
    "Fetch a single URL and return its main content as clean Markdown. Use "
    "this after web_search when you need full page content (verbatim quotes, "
    "long articles, complete docs sections). Do NOT use shell curl/wget — "
    "this tool handles encoding, redirects, content extraction, and length "
    "limits in one call."
)

WEB_SEARCH_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query in natural language.",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": "Maximum number of results to return (default: 8).",
        },
    },
    "required": ["query"],
}

WEB_FETCH_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "url": {
            "type": "string",
            "description": "Absolute http(s) URL to fetch.",
        },
        "max_chars": {
            "type": "integer",
            "minimum": 500,
            "maximum": 200000,
            "description": (
                "Maximum characters of extracted content to return "
                "(default: 20000). Increase only if the page is short."
            ),
        },
    },
    "required": ["url"],
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class WebSearchSettings:
    base_url: str
    timeout_s: float
    max_results: int
    max_iterations: int

    @classmethod
    def from_env(cls) -> "WebSearchSettings":
        return cls(
            base_url=os.getenv("MARATHON_SEARXNG_URL", "http://127.0.0.1:18093").rstrip("/"),
            timeout_s=_env_float("MARATHON_WEB_SEARCH_TIMEOUT", 15.0),
            max_results=_env_int("MARATHON_WEB_SEARCH_MAX_RESULTS", 8),
            max_iterations=_env_int("MARATHON_WEB_SEARCH_MAX_ITERS", 5),
        )


@dataclass(frozen=True)
class WebFetchSettings:
    timeout_s: float
    max_chars_default: int
    max_chars_cap: int
    max_bytes: int
    user_agent: str
    allow_private_networks: bool

    @classmethod
    def from_env(cls) -> "WebFetchSettings":
        # Mozilla-prefixed UA so sites like Wikipedia serve full content
        # (their CDN strips most of the page for non-Mozilla UAs). The
        # Marathon identifier and contact URL still satisfy Wikipedia's
        # User-Agent policy for non-browser clients.
        default_ua = (
            "Mozilla/5.0 (compatible; Marathon/1.0; "
            "+https://github.com/anthropics/marathon) local-codex-router/web_fetch"
        )
        return cls(
            timeout_s=_env_float("MARATHON_WEB_FETCH_TIMEOUT", 25.0),
            max_chars_default=_env_int("MARATHON_WEB_FETCH_MAX_CHARS", 20000),
            max_chars_cap=_env_int("MARATHON_WEB_FETCH_MAX_CHARS_CAP", 200000),
            max_bytes=_env_int("MARATHON_WEB_FETCH_MAX_BYTES", 5 * 1024 * 1024),
            user_agent=os.getenv("MARATHON_WEB_FETCH_USER_AGENT", default_ua),
            allow_private_networks=os.getenv("MARATHON_WEB_FETCH_ALLOW_PRIVATE") == "1",
        )


def web_search_function_tool() -> dict[str, Any]:
    """Return the function-tool spec injected into requests sent to llama.cpp."""

    return {
        "type": "function",
        "name": WEB_SEARCH_TOOL_NAME,
        "description": WEB_SEARCH_TOOL_DESCRIPTION,
        "parameters": copy.deepcopy(WEB_SEARCH_TOOL_PARAMETERS),
    }


def web_fetch_function_tool() -> dict[str, Any]:
    """Return the web_fetch function-tool spec."""

    return {
        "type": "function",
        "name": WEB_FETCH_TOOL_NAME,
        "description": WEB_FETCH_TOOL_DESCRIPTION,
        "parameters": copy.deepcopy(WEB_FETCH_TOOL_PARAMETERS),
    }


def request_has_web_search_tool(tools: list[Any] | None) -> bool:
    """True iff the original Codex request included a web_search tool.

    web_fetch is unconditionally paired with web_search — if the user enabled
    web search, they want fetch as well. Keeping a single trigger keeps the
    contract simple.
    """

    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "web_search":
            return True
        if tool.get("type") == "function" and tool.get("name") == WEB_SEARCH_TOOL_NAME:
            return True
    return False


def is_web_fetch_function_call(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") != "function_call":
        return False
    return item.get("name") == WEB_FETCH_TOOL_NAME


def is_managed_function_call(item: Any) -> bool:
    return is_web_search_function_call(item) or is_web_fetch_function_call(item)


def parse_function_call_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def is_web_search_function_call(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") != "function_call":
        return False
    return item.get("name") == WEB_SEARCH_TOOL_NAME


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    engine: str | None

    def to_text_block(self, index: int) -> str:
        engine_suffix = f" [{self.engine}]" if self.engine else ""
        snippet = self.snippet.strip()
        if not snippet:
            snippet = "(no snippet provided)"
        return f"{index}. {self.title}{engine_suffix}\n   {self.url}\n   {snippet}"


def format_results_for_model(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f"No results found for query: {query!r}."
    blocks = [r.to_text_block(idx + 1) for idx, r in enumerate(results)]
    header = f"SearXNG results for query: {query!r}\n"
    return header + "\n".join(blocks)


class WebSearchExecutor:
    """Async client wrapping a SearXNG instance with a small result formatter."""

    def __init__(self, settings: WebSearchSettings, http_client: ClientSession | None = None) -> None:
        self.settings = settings
        self._owned_client: ClientSession | None = None
        self._client: ClientSession | None = http_client

    async def _ensure_client(self) -> ClientSession:
        if self._client is not None:
            return self._client
        if self._owned_client is None:
            self._owned_client = ClientSession(timeout=ClientTimeout(total=self.settings.timeout_s))
            self._client = self._owned_client
        return self._client  # type: ignore[return-value]

    async def close(self) -> None:
        if self._owned_client is not None:
            await self._owned_client.close()
            self._owned_client = None
            self._client = None

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        client = await self._ensure_client()
        params = {"q": query.strip(), "format": "json", "safesearch": "0"}
        url = f"{self.settings.base_url}/search?{urlencode(params)}"
        timeout = ClientTimeout(total=self.settings.timeout_s)
        async with client.get(url, timeout=timeout) as response:
            if response.status >= 400:
                text = await response.text()
                raise RuntimeError(
                    f"SearXNG returned HTTP {response.status}: {text[:200]}"
                )
            payload = await response.json(content_type=None)

        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            return []

        cap = max_results if max_results is not None else self.settings.max_results
        cap = max(1, min(cap, 20))

        formatted: list[SearchResult] = []
        for entry in raw_results[:cap]:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title") or "").strip()
            url_str = str(entry.get("url") or "").strip()
            snippet = str(entry.get("content") or "").strip()
            engine = entry.get("engine")
            if not title and not url_str:
                continue
            formatted.append(
                SearchResult(
                    title=title or url_str,
                    url=url_str,
                    snippet=snippet,
                    engine=str(engine) if isinstance(engine, str) else None,
                )
            )
        return formatted


def make_function_call_output(call_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": text,
    }


def make_web_search_call_item(call_id: str, query: str) -> dict[str, Any]:
    """Build a Codex ``web_search_call`` ResponseItem for TUI rendering."""

    return {
        "type": "web_search_call",
        "id": call_id if call_id else f"ws_{uuid.uuid4().hex}",
        "status": "completed",
        "action": {"type": "search", "query": query},
    }


def make_web_fetch_call_item(call_id: str, url: str) -> dict[str, Any]:
    """Build a Codex ``web_search_call`` item with action.type=open_page for fetches."""

    return {
        "type": "web_search_call",
        "id": call_id if call_id else f"wf_{uuid.uuid4().hex}",
        "status": "completed",
        "action": {"type": "open_page", "url": url},
    }


class WebFetchExecutor:
    """Fetch a URL and return its main content as Markdown."""

    def __init__(
        self,
        settings: WebFetchSettings,
        http_client: ClientSession | None = None,
    ) -> None:
        self.settings = settings
        self._owned_client: ClientSession | None = None
        self._client: ClientSession | None = http_client

    async def _ensure_client(self) -> ClientSession:
        if self._client is not None:
            return self._client
        if self._owned_client is None:
            self._owned_client = ClientSession(
                timeout=ClientTimeout(total=self.settings.timeout_s),
                headers={"User-Agent": self.settings.user_agent},
            )
            self._client = self._owned_client
        return self._client  # type: ignore[return-value]

    async def close(self) -> None:
        if self._owned_client is not None:
            await self._owned_client.close()
            self._owned_client = None
            self._client = None

    async def fetch(self, url: str, max_chars: int | None = None) -> str:
        if not url or not isinstance(url, str):
            raise ValueError("web_fetch requires a non-empty url string")
        url = url.strip()

        cap = max_chars if max_chars is not None else self.settings.max_chars_default
        cap = max(500, min(cap, self.settings.max_chars_cap))

        client = await self._ensure_client()
        timeout = ClientTimeout(total=self.settings.timeout_s)
        try:
            current_url = url
            for _ in range(6):
                await _validate_fetch_url(
                    current_url,
                    allow_private_networks=self.settings.allow_private_networks,
                )
                async with client.get(
                    current_url,
                    allow_redirects=False,
                    timeout=timeout,
                ) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise RuntimeError(
                                f"HTTP {response.status} without Location for {current_url}"
                            )
                        current_url = urljoin(str(response.url), location)
                        continue
                    if response.status >= 400:
                        raise RuntimeError(f"HTTP {response.status} fetching {current_url}")
                    content_type = response.headers.get("content-type", "").lower()
                    final_url = str(response.url)
                    chunks: list[bytes] = []
                    total = 0
                    limit = self.settings.max_bytes
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        chunks.append(chunk)
                        total += len(chunk)
                        if total > limit:
                            raise RuntimeError(
                                f"web_fetch refused {current_url}: response exceeds "
                                f"{limit // (1024 * 1024)} MiB"
                            )
                    raw = b"".join(chunks)
                    break
            else:
                raise RuntimeError(f"too many redirects fetching {url}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"web_fetch failed: {exc}") from exc

        return _extract_to_markdown(raw, content_type, final_url, cap)


_BLOCKED_FETCH_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
}


async def _validate_fetch_url(url: str, *, allow_private_networks: bool) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("web_fetch only supports absolute http(s) URLs")
    if parsed.username or parsed.password:
        raise ValueError("web_fetch URLs must not include embedded credentials")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("web_fetch requires a URL with a hostname")

    if allow_private_networks:
        return

    host_key = hostname.rstrip(".").lower()
    if host_key in _BLOCKED_FETCH_HOSTNAMES:
        raise ValueError(
            "web_fetch blocked a private/local URL; set "
            "MARATHON_WEB_FETCH_ALLOW_PRIVATE=1 to allow this intentionally"
        )

    try:
        direct_ip = ipaddress.ip_address(host_key)
    except ValueError:
        direct_ip = None
    if direct_ip is not None:
        _raise_if_disallowed_fetch_ip(direct_ip)
        return

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("web_fetch URL has an invalid port") from exc

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise RuntimeError(f"web_fetch could not resolve host {hostname!r}: {exc}") from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        try:
            addresses.append(ipaddress.ip_address(str(sockaddr[0])))
        except ValueError:
            continue
    if not addresses:
        raise RuntimeError(f"web_fetch could not resolve host {hostname!r}")
    for address in addresses:
        _raise_if_disallowed_fetch_ip(address)


def _raise_if_disallowed_fetch_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> None:
    if address.is_global and not address.is_multicast:
        return
    raise ValueError(
        "web_fetch blocked a private/local URL target "
        f"({address}); set MARATHON_WEB_FETCH_ALLOW_PRIVATE=1 "
        "to allow this intentionally"
    )


def _extract_to_markdown(raw: bytes, content_type: str, url: str, cap: int) -> str:
    """Convert fetched bytes into a clean Markdown excerpt."""

    text = _decode_bytes(raw, content_type)

    if "text/plain" in content_type or "application/json" in content_type:
        body = text
    elif "text/html" in content_type or "<html" in text[:1024].lower() or "application/xhtml" in content_type:
        body = _html_to_markdown(text, url)
    else:
        body = text

    body = body.strip()
    truncated = False
    if len(body) > cap:
        body = body[:cap]
        truncated = True

    header = f"Fetched: {url}\n"
    if truncated:
        header += f"(truncated to {cap} chars)\n"
    return header + "\n" + body


def _decode_bytes(raw: bytes, content_type: str) -> str:
    encoding: str | None = None
    if "charset=" in content_type:
        encoding = content_type.split("charset=", 1)[1].split(";", 1)[0].strip().strip('"')
    for candidate in (encoding, "utf-8", "latin-1"):
        if not candidate:
            continue
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _html_to_markdown(html: str, url: str) -> str:
    try:
        import trafilatura
    except ImportError:
        return _html_fallback(html)

    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        url=url,
        include_links=True,
        include_tables=True,
        include_images=False,
        favor_recall=True,
        with_metadata=False,
    )
    if extracted and extracted.strip():
        return extracted
    return _html_fallback(html)


def _html_fallback(html: str) -> str:
    """Last-resort HTML→text when trafilatura is unavailable or returns empty."""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html

    soup = BeautifulSoup(html, "lxml") if _have_lxml() else BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    lines = [line for line in (l.strip() for l in text.splitlines()) if line]
    return "\n".join(lines)


def _have_lxml() -> bool:
    try:
        import lxml  # noqa: F401
    except ImportError:
        return False
    return True


def externalize_for_codex(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate web_search/web_fetch function_call pairs into web_search_call items.

    The TUI renders ``web_search_call`` natively (as a "Searching..." or
    "Opened URL" pill). Internal lineage keeps the real function_call /
    function_call_output items so the model retains tool-use memory across
    turns, but Codex only sees the polished view.
    """

    consumed_outputs: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            result.append(item)
            continue
        if is_web_search_function_call(item):
            call_id = str(item.get("call_id") or item.get("id") or "")
            args = parse_function_call_arguments(item.get("arguments"))
            query = str(args.get("query") or "").strip()
            if call_id:
                consumed_outputs.add(call_id)
            result.append(make_web_search_call_item(call_id, query))
            continue
        if is_web_fetch_function_call(item):
            call_id = str(item.get("call_id") or item.get("id") or "")
            args = parse_function_call_arguments(item.get("arguments"))
            url = str(args.get("url") or "").strip()
            if call_id:
                consumed_outputs.add(call_id)
            result.append(make_web_fetch_call_item(call_id, url))
            continue
        if item.get("type") == "function_call_output":
            cid = str(item.get("call_id") or "")
            if cid and cid in consumed_outputs:
                consumed_outputs.discard(cid)
                continue
        result.append(item)
    return result


def collect_managed_calls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all web_search and web_fetch function_call items in order."""

    return [item for item in items if is_managed_function_call(item)]


def collect_web_search_calls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if is_web_search_function_call(item)]


def synthesize_call_id(item: dict[str, Any], fallback_index: int) -> str:
    cid = item.get("call_id") or item.get("id")
    if isinstance(cid, str) and cid:
        return cid
    return f"ws_{int(time.time() * 1000)}_{fallback_index}"


__all__ = [
    "SearchResult",
    "WebFetchExecutor",
    "WebFetchSettings",
    "WebSearchExecutor",
    "WebSearchSettings",
    "WEB_FETCH_TOOL_NAME",
    "WEB_SEARCH_TOOL_NAME",
    "collect_managed_calls",
    "collect_web_search_calls",
    "externalize_for_codex",
    "format_results_for_model",
    "is_managed_function_call",
    "is_web_fetch_function_call",
    "is_web_search_function_call",
    "make_function_call_output",
    "make_web_fetch_call_item",
    "make_web_search_call_item",
    "parse_function_call_arguments",
    "request_has_web_search_tool",
    "synthesize_call_id",
    "web_fetch_function_tool",
    "web_search_function_tool",
]
