"""Admission control for Marathon's private, non-browser loopback API."""

import ipaddress
import os
import secrets
import urllib.request
from urllib.parse import urlsplit

from aiohttp import web


class _NoApiRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_api_request(request, *, timeout):
    """Keep inference API credentials off shell-configured proxies and redirects."""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoApiRedirects()
    )
    return opener.open(request, timeout=timeout)


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def router_security_middleware():
    token = os.environ.get("MARATHON_ROUTER_TOKEN", "")
    if not token:
        raise RuntimeError("MARATHON_ROUTER_TOKEN is required for the local router")
    expected = f"Bearer {token}".encode("utf-8")

    @web.middleware
    async def authenticate(request, handler):
        # This API has no browser frontend. Reject even opaque/null origins,
        # before upgrading WebSockets or reading potentially large bodies.
        if "Origin" in request.headers or not is_loopback(request.remote or ""):
            raise web.HTTPForbidden(text="Non-browser loopback clients only")
        try:
            host = urlsplit(f"//{request.headers.get('Host', '')}")
            socket = request.transport.get_extra_info("sockname")
            valid_host = (
                is_loopback(host.hostname or "")
                and not host.username
                and not host.password
                and not host.path
                and not host.query
                and not host.fragment
                and (host.port or 80) == socket[1]
            )
        except (ValueError, TypeError, AttributeError):
            valid_host = False
        if not valid_host:
            raise web.HTTPForbidden(text="Invalid local Host header")
        supplied = request.headers.getall("Authorization", [])
        if len(supplied) != 1 or not secrets.compare_digest(
            supplied[0].encode("utf-8", errors="surrogateescape"), expected
        ):
            raise web.HTTPUnauthorized(
                text="Local router authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await handler(request)

    return authenticate
