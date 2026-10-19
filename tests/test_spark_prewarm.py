"""Spark selection warmups use normal admission without blocking the composer."""
import asyncio
from dataclasses import replace
from unittest import IsolatedAsyncioTestCase, mock

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from test_router_context import fixture_profile, router_module as router


class SparkPrewarmTests(IsolatedAsyncioTestCase):
    async def test_background_load_deduplicates_and_sends_interactive_header(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        seen = []

        async def backend(request):
            seen.append((dict(request.headers), await request.json()))
            entered.set()
            await finish.wait()
            return web.json_response({'choices': []})

        app = web.Application()
        app.router.add_post('/v1/chat/completions', backend)
        async with TestServer(app) as server, ClientSession() as client:
            state = self.state(client)
            profile = replace(fixture_profile(), alias='spark/Qwen3.8-Flash-Next',
                              target=str(server.make_url('')).rstrip('/'), supports_slots=False)
            result = state.schedule_spark_prewarm(profile)
            self.assertEqual(result['status'], 'scheduled')
            await asyncio.wait_for(entered.wait(), 2)
            self.assertFalse(state.spark_prewarm_task.done())
            self.assertEqual(state.schedule_spark_prewarm(profile)['status'], 'already-scheduled')
            self.assertEqual(len(seen), 1)
            headers, body = seen[0]
            self.assertEqual(headers['X-Spark-Interactive'], '1')
            self.assertEqual(body['model'], profile.alias)
            self.assertEqual(body['max_tokens'], 1)
            self.assertNotIn('tools', body)
            finish.set()
            await state.spark_prewarm_task
            self.assertEqual(state.schedule_spark_prewarm(profile)['status'], 'already-scheduled')
            self.assertEqual(state.telemetry.emit.call_args.args[0], 'router.model.prewarm_completed')

    async def test_active_reservation_failure_is_nonfatal_and_retryable(self):
        async def backend(request):
            return web.json_response({'error': {'code': 'spark_model_reserved'}}, status=429)

        app = web.Application()
        app.router.add_post('/v1/chat/completions', backend)
        async with TestServer(app) as server, ClientSession() as client:
            state = self.state(client)
            profile = replace(fixture_profile(), alias='spark/Qwen3.8-Flash-Next',
                              target=str(server.make_url('')).rstrip('/'))
            state.schedule_spark_prewarm(profile)
            await state.spark_prewarm_task
            self.assertIsNone(state.spark_prewarm_model)
            self.assertEqual(state.telemetry.emit.call_args.args[0], 'router.model.prewarm_failed')

    async def test_model_change_cancels_previous_warmup_before_new_request(self):
        state = self.state(None)
        entered, canceled = asyncio.Event(), asyncio.Event()
        first = replace(fixture_profile(), alias='spark/Qwen3.8-Flash-Next')
        second = replace(first, alias='spark/Qwen3.8-Flash-Next-LR7')

        async def request(profile, *args, **kwargs):
            if profile.alias == first.alias:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    canceled.set()
            else:
                self.assertTrue(canceled.is_set())
            return {}

        state._request_json = request
        state.schedule_spark_prewarm(first)
        await entered.wait()
        state.schedule_spark_prewarm(second)
        await state.spark_prewarm_task
        self.assertEqual(state.spark_prewarm_model, second.alias)

    @staticmethod
    def state(client):
        state = object.__new__(router.RouterState)
        state._closing = False
        state.spark_prewarm_model = None
        state.spark_prewarm_task = None
        state.http_client = client
        state.telemetry = mock.Mock()
        return state
