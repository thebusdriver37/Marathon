"""Exercise the real router's HTTP and WebSocket admission boundary."""

import sys
import asyncio
import urllib.error
import urllib.request
import unittest
from pathlib import Path
from unittest import mock

from aiohttp import WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "routers"))
import codex_local_router as router
from marathon_app.router_security import open_api_request


class RouterSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = mock.Mock()
        self.state.model_catalog.return_value = {"data": [{"id": "local-test"}]}
        with mock.patch.dict("os.environ", {"MARATHON_ROUTER_TOKEN": "test-secret"}):
            app = router.build_app(self.state)
        # Keep real routes and middleware, but never launch inference workers.
        app.on_startup.clear()
        app.on_cleanup.clear()
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.auth = {"Authorization": "Bearer test-secret"}

    async def asyncTearDown(self):
        await self.client.close()

    async def test_unauthenticated_http_cannot_read_catalog(self):
        response = await self.client.get("/v1/models")
        self.assertEqual(response.status, 401)
        self.state.model_catalog.assert_not_called()

    async def test_authenticated_http_reads_catalog(self):
        response = await self.client.get("/v1/models", headers=self.auth)
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["data"][0]["id"], "local-test")

    async def test_wrong_token_is_rejected(self):
        response = await self.client.get(
            "/v1/models", headers={"Authorization": "Bearer wrong"}
        )
        self.assertEqual(response.status, 401)

    async def test_supervisor_bypasses_environment_proxy(self):
        # A dead proxy would make the request fail if urllib honored it.
        request = urllib.request.Request(str(self.client.make_url("/v1/models")), headers=self.auth)
        with mock.patch.dict("os.environ", {
            "http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1",
            "no_proxy": "", "NO_PROXY": "",
        }):
            response = await asyncio.to_thread(open_api_request, request, timeout=2)
            with response:
                self.assertEqual(response.status, 200)

    async def test_api_redirect_cannot_forward_credentials(self):
        from aiohttp import web

        async def redirect(request):
            raise web.HTTPFound("http://127.0.0.1:1/credential-leak")

        app = web.Application()
        app.router.add_get("/redirect", redirect)
        async with TestServer(app) as server:
            request = urllib.request.Request(str(server.make_url("/redirect")), headers=self.auth)
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                await asyncio.to_thread(open_api_request, request, timeout=2)
            self.assertEqual(rejected.exception.code, 302)

    async def test_cross_origin_websocket_is_rejected_before_upgrade(self):
        with self.assertRaises(WSServerHandshakeError) as rejected:
            await self.client.ws_connect(
                "/v1/responses", headers={"Origin": "https://untrusted.example"}
            )
        self.assertEqual(rejected.exception.status, 403)

    async def test_browser_origin_is_rejected_even_with_token(self):
        response = await self.client.get(
            "/v1/models", headers={**self.auth, "Origin": "null"}
        )
        self.assertEqual(response.status, 403)

    async def test_rebinding_host_is_rejected(self):
        response = await self.client.get(
            "/v1/models", headers={**self.auth, "Host": "untrusted.example"}
        )
        self.assertEqual(response.status, 403)

    async def test_authenticated_websocket_still_works(self):
        self.state.mint_response_id.return_value = "resp-test"
        self.state.replace_active_ws_task.return_value = None
        self.state.process_websocket_create = mock.AsyncMock(return_value={
            "streamed": False, "response_id": "resp-test", "output_items": [],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        })
        async with self.client.ws_connect("/v1/responses", headers=self.auth) as ws:
            await ws.send_json({"type": "response.create", "model": "local-test", "input": []})
            self.assertEqual((await ws.receive_json())["type"], "response.created")
            self.assertEqual((await ws.receive_json())["type"], "response.completed")
        self.state.process_websocket_create.assert_awaited_once()

    async def test_health_requires_auth_without_touching_backend(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status, 401)
        self.state.backend_health.assert_not_called()

    def test_router_refuses_missing_secret(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MARATHON_ROUTER_TOKEN"):
                router.build_app(self.state)

    def test_frontend_credentials_never_reach_upstream(self):
        profile = mock.Mock(external=False)
        profile.upstream_headers.return_value = {"Authorization": "Bearer backend-key"}
        headers = router._proxy_request_headers(
            profile, {**self.auth, "Cookie": "cloud-session", "X-Request-ID": "test"}
        )
        self.assertEqual(headers, {
            "Authorization": "Bearer backend-key", "X-Request-ID": "test"
        })
