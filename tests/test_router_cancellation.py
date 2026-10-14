"""Cancellation must reach a silent backend without waiting for keepalives."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from test_router_context import router_module


class RouterCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_close_cancels_silent_generation_promptly(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        tasks = []

        async def generate(*args, **kwargs):
            tasks.append(asyncio.current_task())
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        state = SimpleNamespace(
            mint_response_id=lambda prefix: prefix + '_test',
            process_websocket_create=generate,
            replace_active_ws_task=lambda *args: None,
            telemetry=mock.Mock(),
        )
        app = web.Application()
        app['state'] = state
        app.router.add_get('/v1/responses', router_module.handle_ws_responses)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect('/v1/responses')
            await ws.send_json({'type': 'response.create', 'model': 'test', 'input': []})
            await asyncio.wait_for(started.wait(), 1)
            closing = asyncio.create_task(ws.close())
            try:
                await asyncio.wait_for(cancelled.wait(), .5)
                await asyncio.wait_for(closing, .5)
            finally:
                closing.cancel()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(closing, *tasks, return_exceptions=True)

    async def test_http_close_cancels_silent_generation_promptly(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def generate(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        state = SimpleNamespace(
            mint_response_id=lambda prefix: prefix + '_test',
            _process_websocket_create=generate,
            http_client=object(),
        )
        from test_router_context import fixture_profile
        async def handle(request):
            return await router_module._handle_http_responses(
                request, await request.json(), fixture_profile(), None)
        app = web.Application()
        app['state'] = state
        app.router.add_post('/v1/responses', handle)
        async with TestClient(TestServer(app)) as client:
            response = await client.post('/v1/responses', json={'stream': True, 'input': []})
            await asyncio.wait_for(started.wait(), 1)
            response.close()
            await asyncio.wait_for(cancelled.wait(), .5)
