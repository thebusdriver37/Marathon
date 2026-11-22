import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/evals'))
import inkling_agentic_cache_probe as probe


class AgenticProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_twenty_actual_tool_calls_and_restart_continuation(self):
        calls = []

        async def models(_):
            return web.json_response({'data': [{'id': 'inkling-2-mini'}]})

        async def backend(request):
            body = await request.json()
            calls.append(body)
            index = len(calls)
            item = {'id': f'fc_{index}', 'type': 'function_call', 'name': 'read_archive_chunk',
                    'call_id': f'call_{index}', 'arguments': json.dumps({'index': index}),
                    'status': 'completed'}
            result = {'id': f'resp_{index}', 'status': 'completed', 'output': [item],
                      'usage': {'input_tokens': 100000 + 2500 * (index - 1), 'output_tokens': 8,
                                'input_tokens_details': {'cached_tokens': 100000 if index > 1 else 0}},
                      'timings': {'cache_n': 100000 if index > 1 else 0, 'prompt_n': 2500,
                                  'prompt_ms': 100, 'predicted_n': 8, 'predicted_ms': 40}}
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            for event in [
                {'type': 'response.output_item.added', 'item': {**item, 'arguments': ''}},
                {'type': 'response.function_call_arguments.delta', 'item_id': item['id'],
                 'delta': item['arguments']},
                {'type': 'response.output_item.done', 'item': item},
                {'type': 'response.completed', 'response': result},
            ]:
                await response.write(('data: ' + json.dumps(event) + '\n\n').encode())
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_get('/v1/models', models)
        app.router.add_post('/v1/responses', backend)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / 'corpus.json'
            corpus.write_text(json.dumps({'initial': 'Call index 1.',
                                         'chunks': [f'NEXT_INDEX {n+1}' for n in range(1, 21)]}))
            out = root / 'run'
            async with TestServer(app) as server:
                async def marker():
                    while not (out / 'ready-to-unload.json').exists():
                        await asyncio.sleep(.01)
                    self.assertEqual(len(calls), 20)
                    (out / 'resume-now').touch()
                mark = asyncio.create_task(marker())
                try:
                    result = await probe.run(SimpleNamespace(
                        output=out, corpus=corpus, turns=20, upstream=str(server.make_url('')).rstrip('/'),
                        cache_dir=None, engine_pid=None, restart_timeout=10))
                finally:
                    mark.cancel()
                    await asyncio.gather(mark, return_exceptions=True)
            self.assertEqual(len(calls), 21)
            self.assertEqual(len(result['turns']), 20)
            self.assertEqual(result['resumed']['input_tokens'], 150000)
            for index, call in enumerate(calls, 1):
                self.assertNotIn('previous_response_id', call)
                self.assertNotIn('id_slot', call)
                self.assertEqual(call['tools'], calls[0]['tools'])
                self.assertEqual(call['instructions'], calls[0]['instructions'])
                outputs = [x for x in call['input'] if x.get('type') == 'function_call_output']
                self.assertEqual(len(outputs), index - 1)
                if outputs:
                    self.assertEqual(outputs[-1]['call_id'], f'call_{index-1}')
            full = json.loads((out / 'resume-full-request.json').read_text())
            self.assertNotIn('previous_response_id', full)
            self.assertEqual(len([x for x in full['input'] if x.get('type') == 'function_call']), 20)
            unload = root / 'unload.json'
            unload.write_text(json.dumps({'elapsed_s': 9, 'returncode': 0, 'write_bytes_delta': 4096,
                                          'read_bytes': 999999999}))
            comparison = root / 'comparison.json'
            probe.compare(SimpleNamespace(baseline=out, candidate=out, output=comparison, unload_json=unload))
            summary = json.loads(comparison.read_text())
            self.assertEqual(summary['unload']['write_bytes_delta'], 4096)
            self.assertNotIn('read_bytes', summary['unload'])
            self.assertEqual(summary['first_use_completion_delta_s'], 0)
            self.assertNotIn('candidate_measured_write_bytes', summary)  # Missing PID counters stay unknown.


    def test_io_only_counts_writes_and_occupancy_is_separate(self):
        raw = 'rchar: 1000\nwchar: 500\nsyscr: 4\nsyscw: 5\nread_bytes: 999999\nwrite_bytes: 4096\ncancelled_write_bytes: 0\n'
        with mock.patch.object(Path, 'read_text', return_value=raw):
            self.assertEqual(probe.process_io(123), {'wchar': 500, 'write_bytes': 4096,
                                                    'cancelled_write_bytes': 0})
        self.assertEqual(probe.io_delta({'write_bytes': 0}, {'write_bytes': 4096}), {'write_bytes': 4096})
        self.assertIsNone(probe.io_delta(None, {'write_bytes': 4096}))
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / 'cache').write_bytes(b'123')
            (p / 'link').symlink_to(p / 'cache')
            result = probe.disk_snapshot(p)
            self.assertEqual(result['occupancy_bytes'], 3)
            self.assertEqual(len(result['files']), 1)
