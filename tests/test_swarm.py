"""Exercise the swarm gateway against real HTTP servers without loading GPUs."""

import asyncio
import json
import unittest
import uuid

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from marathon_app.swarm_gateway import SwarmGateway
from marathon_app.swarm_tools import agent_identity, identify_agent, flatten_request, restore_calls, translated_sse


class SwarmGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.events = []
        self.servers = []
        self.active = 0
        self.peak = 0
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        workers = []
        for index in range(3):
            async def handler(request, index=index):
                websocket = None
                if request.headers.get('Upgrade'):
                    websocket = web.WebSocketResponse()
                    await websocket.prepare(request)
                    body = json.dumps(await websocket.receive_json()).encode()
                else:
                    body = await request.read()
                self.calls.append((index, request.path, request.headers.get('Authorization'), body))
                if request.query.get('wait'):
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                    try:
                        if websocket:
                            released = asyncio.create_task(self.release.wait())
                            disconnected = asyncio.create_task(websocket.receive())
                            try:
                                done, _ = await asyncio.wait((released, disconnected), return_when=asyncio.FIRST_COMPLETED)
                                if disconnected in done:
                                    self.cancelled.set()
                                    return websocket
                            finally:
                                for pending in (released, disconnected):
                                    pending.cancel()
                                await asyncio.gather(released, disconnected, return_exceptions=True)
                        else:
                            await self.release.wait()
                    except asyncio.CancelledError:
                        self.cancelled.set()
                        raise
                    finally:
                        self.active -= 1
                if websocket:
                    await websocket.send_json({'type': 'response.completed', 'response': {'worker': index}})
                    await websocket.close()
                    return websocket
                return web.json_response({'worker': index})
            app = web.Application()
            app.router.add_route('*', '/{path:.*}', handler)
            server = TestServer(app, handler_cancellation=True)
            await server.start_server()
            self.servers.append(server)
            workers.append({'router_url': str(server.make_url('')).rstrip('/'), 'router_token': f'worker-{index}'})
        self.gateway = SwarmGateway(workers, 'gateway-secret', lambda event, data: self.events.append((event, data)))
        self.client = TestClient(TestServer(self.gateway.application(), handler_cancellation=True))
        await self.client.start_server()

    async def asyncTearDown(self):
        self.release.set()
        await self.client.close()
        for server in self.servers:
            await server.close()

    def headers(self, thread=None):
        return {'Authorization': 'Bearer gateway-secret', 'thread-id': thread or str(uuid.uuid4())}

    async def response_payload(self, response):
        if response.content_type == 'text/event-stream':
            text = await response.text()
            return json.loads(text.split('data: ', 1)[1])['response']
        return await response.json()

    async def test_sticky_threads_compaction_and_upstream_credentials(self):
        threads = [str(uuid.uuid4()) for _ in range(3)]
        for index in (0, 1, 2, 1, 0, 2):
            response = await self.client.post('/v1/responses', headers=self.headers(threads[index]), json={'input': 'hi'})
            self.assertEqual(await self.response_payload(response), {'worker': index})
        response = await self.client.post('/v1/responses/compact', headers=self.headers(threads[1]), json={})
        self.assertEqual(await response.json(), {'worker': 1})
        self.assertEqual(len(self.gateway.bindings), 3)
        for index, path, authorization, body in self.calls:
            self.assertEqual(authorization, f'Bearer worker-{index}')
            self.assertIsInstance(json.loads(body), dict)
            self.assertEqual(json.loads(body)['prompt_cache_key'], f'marathon-swarm-{threads[index]}')

    async def test_rejects_unauthorized_missing_identity_and_excess_agents(self):
        response = await self.client.post('/v1/responses')
        self.assertEqual(response.status, 401)
        response = await self.client.post('/v1/responses', headers={'Authorization': 'Bearer gateway-secret'})
        self.assertEqual(response.status, 400)
        self.assertEqual(self.calls, [])
        for _ in range(3):
            response = await self.client.post('/v1/responses', headers=self.headers())
            await response.read()
        response = await self.client.post('/v1/responses', headers=self.headers())
        self.assertEqual(response.status, 409)
        self.assertEqual(len(self.calls), 3)

    async def test_discovery_does_not_consume_an_agent_slot(self):
        response = await self.client.get('/v1/models', headers=self.headers())
        self.assertEqual(await response.json(), {'worker': 0})
        self.assertEqual(self.gateway.bindings, {})
        response = await self.client.post('/admin/stop', headers=self.headers())
        self.assertEqual(response.status, 404)

    async def test_three_threads_reach_workers_concurrently(self):
        tasks = [asyncio.create_task(self.client.post('/v1/responses?wait=1', headers=self.headers())) for _ in range(3)]
        async def wait_for_workers():
            while self.peak < 3:
                await asyncio.sleep(0.01)
        try:
            await asyncio.wait_for(wait_for_workers(), timeout=5)
            self.assertEqual(self.peak, 3)
        finally:
            self.release.set()
            responses = await asyncio.gather(*tasks)
            for response in responses:
                await response.read()

    async def test_disconnect_cancels_upstream_request(self):
        task = asyncio.create_task(self.client.post('/v1/responses?wait=1', headers=self.headers()))
        async def wait_for_request():
            while not self.active:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait_for_request(), timeout=5)
        response = await task
        response.close()
        await asyncio.wait_for(self.cancelled.wait(), timeout=5)

    async def test_namespace_translation_round_trip_and_stream_boundaries(self):
        names = {}
        call = {'type': 'function_call', 'id': 'fc1', 'call_id': 'c1',
                'namespace': 'collaboration', 'name': 'spawn_agent',
                'arguments': '{"task_name":"helper","message":"check"}'}
        request = {'tools': [{'type': 'namespace', 'name': 'collaboration', 'tools': [
            {'type': 'function', 'name': 'spawn_agent', 'parameters': {'type': 'object'}}
        ]}], 'input': [call]}
        flattened = flatten_request(request, names)
        self.assertEqual(flattened['tools'][0]['name'], 'collaboration__spawn_agent')
        self.assertNotIn('namespace', flattened['input'][0])
        self.assertEqual(restore_calls(flattened['input'][0], names), call)
        self.assertEqual(request['input'][0], call)
        event = {'type': 'response.output_item.done', 'item': flattened['input'][0]}
        raw = ('event: response.output_item.done\ndata: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()
        class Content:
            async def iter_any(self):
                for offset in range(0, len(raw), 7):
                    yield raw[offset:offset + 7]
        result = b''.join([chunk async for chunk in translated_sse(Content(), names)])
        decoded = json.loads(result.split(b'data: ', 1)[1].split(b'\n', 1)[0])
        self.assertEqual(decoded, {'type': 'response.output_item.done', 'item': call})
        self.assertTrue(result.endswith(b'data: [DONE]\n\n'))
        with self.assertRaisesRegex(ValueError, 'collide'):
            flatten_request({'tools': [*request['tools'], flattened['tools'][0]]}, {})

    async def test_native_agent_messages_preserve_sender_and_plaintext_payload(self):
        original = {'input': [{'type': 'agent_message', 'id': 'amsg1',
                              'author': '/root', 'recipient': '/root/helper', 'content': [
                                  {'type': 'input_text', 'text': 'Task:\n'},
                                  {'type': 'encrypted_content', 'encrypted_content': 'Implement words.py'},
                              ]}]}
        self.assertEqual(flatten_request(original, {}), {'input': [{
            'type': 'message', 'role': 'user', 'content': [
                {'type': 'input_text', 'text': 'Agent message from /root to /root/helper:\n'},
                {'type': 'input_text', 'text': 'Task:\n'},
                {'type': 'input_text', 'text': 'Implement words.py'},
            ],
        }]})
        self.assertEqual(original['input'][0]['type'], 'agent_message')
        identity = agent_identity(original)
        self.assertEqual(identity, '/root/helper')
        identified = identify_agent(flatten_request(original, {}), identity)
        self.assertIn('Current agent identity: /root/helper', identified['instructions'])
        self.assertIn('final answer is automatically delivered', identified['instructions'])
        self.assertEqual(agent_identity({'input': []}), '/root')
        self.assertEqual(identify_agent({'instructions': 'Lead the team'}, '/root'),
                         {'instructions': 'Lead the team'})
