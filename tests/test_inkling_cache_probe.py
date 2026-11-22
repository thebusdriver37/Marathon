"""Full router protocol regression; synthetic backend does not emulate native KV."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from aiohttp import web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inkling_probe', ROOT / 'scripts/evals/inkling_cache_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class InklingCacheProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_replay_and_lineage_preserve_native_cache_usage(self):
        calls = []

        async def models(_request):
            return web.json_response({'data': [{'id': 'inkling-2-mini'}]})

        async def responses(request):
            body = await request.json()
            calls.append(body)
            cached = 100000 if len(calls) > 1 else 0
            item = {'id': f'msg_{len(calls)}', 'type': 'message', 'role': 'assistant',
                    'status': 'completed', 'content': [{'type': 'output_text', 'text': 'CEDAR-731'}]}
            result = {'id': f'upstream_{len(calls)}', 'status': 'completed', 'output': [item],
                      'usage': {'input_tokens': 100010, 'output_tokens': 8,
                                'input_tokens_details': {'cached_tokens': cached}},
                      'timings': {'cache_n': cached, 'prompt_n': 100010 - cached,
                                  'prompt_ms': 100, 'predicted_n': 8, 'predicted_ms': 70}}
            stream = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await stream.prepare(request)
            for event in [
                {'type': 'response.output_item.added', 'item': item},
                {'type': 'response.output_text.delta', 'item_id': item['id'], 'delta': 'CEDAR-731'},
                {'type': 'response.output_item.done', 'item': item},
                {'type': 'response.completed', 'response': result},
            ]:
                await stream.write(('data: ' + json.dumps(event) + '\n\n').encode())
            await stream.write_eof()
            return stream

        app = web.Application()
        app.router.add_get('/v1/models', models)
        app.router.add_post('/v1/responses', responses)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'probe'
            prompt = Path(temporary) / 'synthetic.txt'
            prompt.write_text('The lighthouse access code is CEDAR-731. Return only the code.')
            async with TestServer(app) as server:
                async def allow_restart():
                    while not (output / 'cold-done.json').exists():
                        await asyncio.sleep(.05)
                    (output / 'resume-now').touch()
                marker = asyncio.create_task(allow_restart())
                try:
                    result = await probe.run(SimpleNamespace(
                        output=output, upstream=str(server.make_url('')).rstrip('/'),
                        prompt_file=prompt, restart_timeout=10))
                finally:
                    marker.cancel()
                    await asyncio.gather(marker, return_exceptions=True)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]['input'], calls[1]['input'])
        self.assertGreater(len(calls[2]['input']), len(calls[1]['input']))
        self.assertIn('CEDAR-731', json.dumps(calls[2]['input']))
        for call in calls:
            self.assertNotIn('previous_response_id', call)
            self.assertNotIn('id_slot', call)
            self.assertNotIn('cache_prompt', call)
        for phase in ['restored', 'continuation']:
            self.assertEqual(result[phase]['cache_n'], 100000)
            self.assertEqual(result[phase]['cached_tokens'], 100000)
            metrics = result[phase]['response']['usage_metadata']['marathon']
            self.assertEqual(metrics['prefill_tokens'], 10)
            self.assertEqual(metrics['context_usage']['cached_input_tokens'], 100000)
        self.assertGreater(result['reuse_fraction'], .99)
