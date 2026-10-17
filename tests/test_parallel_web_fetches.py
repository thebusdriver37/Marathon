"""Bounded static fetches with ordered results and cancellation-safe replay."""
import asyncio
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/routers'))
import codex_local_router as router
from marathon_web_search import WebFetchExecutor, WebFetchSettings
from test_router_context import fixture_profile


def call(index, name='web_fetch', url=None):
    return {'type': 'function_call', 'name': name, 'call_id': f'call-{index}',
            'arguments': json.dumps({'url': url or f'https://example.invalid/{index}'})}


def state(limit=3):
    result = object.__new__(router.RouterState)
    result.telemetry = mock.Mock()
    result.web_fetch_semaphore = asyncio.Semaphore(limit)
    return result


class ParallelWebTests(unittest.IsolatedAsyncioTestCase):
    async def test_limit_is_shared_across_batches_and_results_keep_order(self):
        subject = state()
        active = peak = 0
        entered = asyncio.Event()
        release = asyncio.Event()

        async def execute(item, index):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 3:
                entered.set()
            try:
                await release.wait()
                # Deliberately finish in reverse order within the first batch.
                await asyncio.sleep((3 - index % 3) * .005)
                return router.make_function_call_output(item['call_id'], item['arguments'])
            finally:
                active -= 1

        subject._execute_managed_call = execute
        calls = [call(i) for i in range(9)]
        tasks = [asyncio.create_task(subject._execute_managed_calls(calls[:6], 'one')),
                 asyncio.create_task(subject._execute_managed_calls(calls[6:], 'two'))]
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(active, 3)
        release.set()
        results = await asyncio.gather(*tasks)
        self.assertEqual(peak, 3)
        self.assertEqual([r['call_id'] for group in results for r in group], [c['call_id'] for c in calls])
        self.assertEqual(active, 0)
        self.assertEqual(subject.web_tool_locks, {})

    async def test_duplicate_calls_and_reconnect_share_result_with_distinct_ids(self):
        subject = state()
        entered, release = asyncio.Event(), asyncio.Event()
        async def execute(item, index):
            entered.set()
            await release.wait()
            return router.make_function_call_output(item['call_id'], 'Exact source text')
        subject._execute_managed_call = mock.AsyncMock(side_effect=execute)
        calls = [call(i, url='https://example.invalid/same') for i in range(4)]
        tasks = [asyncio.create_task(subject._execute_managed_calls(calls[:2], 'turn')),
                 asyncio.create_task(subject._execute_managed_calls(calls[2:], 'turn'))]
        await asyncio.wait_for(entered.wait(), 1)
        release.set()
        results = await asyncio.gather(*tasks)
        self.assertEqual(subject._execute_managed_call.await_count, 1)
        self.assertEqual([r['call_id'] for group in results for r in group], [c['call_id'] for c in calls])
        results[0][0]['output'] = 'changed locally'
        replay = await subject._execute_managed_calls(calls, 'turn')
        self.assertTrue(all(r['output'] == 'Exact source text' for r in replay))
        self.assertEqual(subject._execute_managed_call.await_count, 1)
        await subject._execute_managed_calls(calls, 'next-turn')
        self.assertEqual(subject._execute_managed_call.await_count, 2)

    async def test_search_and_browser_are_ordered_barriers(self):
        subject = state()
        events = []
        async def execute(item, index):
            events.append(('start', index))
            await asyncio.sleep(.001)
            events.append(('end', index))
            return router.make_function_call_output(item['call_id'], 'source')
        subject._execute_managed_call = execute
        names = ['web_fetch', 'web_fetch', 'web_search', 'web_fetch', 'web_browse', 'web_fetch']
        await subject._execute_managed_calls([call(i, name) for i, name in enumerate(names)], 'turn')
        for barrier in (2, 4):
            self.assertTrue(all(events.index(('end', i)) < events.index(('start', barrier)) for i in range(barrier)))
            self.assertTrue(all(events.index(('start', i)) > events.index(('end', barrier)) for i in range(barrier+1, 6)))

    async def test_cancel_drains_fetches_and_preserves_completed_result_for_replay(self):
        subject = state()
        completed, blocked = asyncio.Event(), asyncio.Event()
        active = 0
        async def execute(item, index):
            nonlocal active
            active += 1
            try:
                if index == 0:
                    completed.set()
                    return router.make_function_call_output(item['call_id'], 'kept')
                blocked.set()
                await asyncio.Future()
            finally:
                active -= 1
        subject._execute_managed_call = mock.AsyncMock(side_effect=execute)
        task = asyncio.create_task(subject._execute_managed_calls([call(i) for i in range(7)], 'turn'))
        await asyncio.wait_for(completed.wait(), 1)
        await asyncio.wait_for(blocked.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(active, 0)
        self.assertEqual(subject.web_tool_locks, {})
        before = subject._execute_managed_call.await_count
        replay = await subject._execute_managed_calls([call(0)], 'turn')
        self.assertEqual(replay[0]['output'], 'kept')
        self.assertEqual(subject._execute_managed_call.await_count, before)

    async def test_unexpected_failure_drains_siblings(self):
        subject = state()
        ready = asyncio.Event()
        active = 0
        async def execute(item, index):
            nonlocal active
            active += 1
            if active == 3:
                ready.set()
            try:
                await ready.wait()
                if index == 0:
                    raise RuntimeError('synthetic failure')
                await asyncio.Future()
            finally:
                active -= 1
        subject._execute_managed_call = execute
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            await asyncio.wait_for(subject._execute_managed_calls([call(i) for i in range(3)], 'turn'), 1)
        self.assertEqual(active, 0)
        self.assertEqual(subject.web_tool_locks, {})

    async def test_cancelled_owner_does_not_cancel_duplicate_in_another_request(self):
        subject = state()
        entered = asyncio.Event()
        attempts = 0
        async def execute(item, index):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                entered.set()
                await asyncio.Future()
            return router.make_function_call_output(item['call_id'], 'retried')
        subject._execute_managed_call = execute
        first = asyncio.create_task(subject._execute_managed_calls([call(0)], 'turn'))
        await entered.wait()
        second = asyncio.create_task(subject._execute_managed_calls([call(1, url='https://example.invalid/0')], 'turn'))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        result = await asyncio.wait_for(second, 1)
        self.assertEqual(result[0]['output'], 'retried')
        self.assertEqual(result[0]['call_id'], 'call-1')
        self.assertEqual(subject.web_tool_locks, {})

    async def test_real_http_fetches_overlap_and_keep_exact_text(self):
        active = peak = 0
        requests = []
        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            requests.append(request.path)
            try:
                await asyncio.sleep(.02)
                return web.Response(text=f'Unique fresh source {request.path}. ' * 20)
            finally:
                active -= 1
        app = web.Application()
        app.router.add_get('/{page}', handler)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        subject = state()
        subject.web_fetch = WebFetchExecutor(replace(WebFetchSettings.from_env(), allow_private_networks=True))
        self.addAsyncCleanup(subject.web_fetch.close)
        calls = [call(i, url=str(server.make_url(f'/{i}'))) for i in range(6)]
        parallel = await subject._execute_managed_calls(calls, 'turn')
        self.assertEqual(peak, 3)
        subject.web_fetch_semaphore = asyncio.Semaphore(1)
        serial = await subject._execute_managed_calls(calls, 'different-turn')
        self.assertEqual(parallel, serial)
        self.assertEqual(len(requests), 12, 'A new user turn must fetch fresh content')

    async def test_entire_response_loop_matches_serial_and_replay(self):
        async def exercise(limit):
            subject = state(limit)
            subject.web_search_settings = SimpleNamespace(max_iterations=3)
            calls = [call(i) for i in range(6)]
            subject._request_json = mock.AsyncMock(side_effect=[{'output': copy.deepcopy(calls)}, {'output': [
                {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Six sources.'}]}]}])
            async def execute(item, index):
                await asyncio.sleep((3 - index % 3) * .001)
                return router.make_function_call_output(item['call_id'], f'EXACT_SOURCE_{index}')
            subject._execute_managed_call = mock.AsyncMock(side_effect=execute)
            request = {'input': [{'role': 'user', 'content': 'Compare these sources.'}],
                       'tools': [router.web_fetch_function_tool()]}
            result = await subject._run_responses_loop(profile=fixture_profile(), forward_request=request, web_search_enabled=True)
            replay = await subject._run_responses_loop(profile=fixture_profile(), forward_request=request, web_search_enabled=True)
            self.assertTrue(replay[0].pop('_marathon_web_replayed_completion'))
            self.assertEqual(result, replay)
            self.assertEqual(subject._execute_managed_call.await_count, 6)
            self.assertEqual(subject._request_json.await_count, 2)
            return result
        self.assertEqual(await exercise(3), await exercise(1))


if __name__ == '__main__':
    unittest.main()
