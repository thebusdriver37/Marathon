"""Exercise the swarm gateway against real HTTP servers without loading GPUs."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from marathon_app.swarm_gateway import SwarmGateway
from marathon_app.swarm_tools import agent_identity, identify_agent, flatten_request, restore_calls, translated_sse


class SwarmGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.echo_backend = False
        self.summary_backend = False
        self.compaction_requests = []
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
                if websocket and self.summary_backend:
                    from marathon_app.echo_recovery import echo_marker
                    payload = json.loads(body)
                    metadata = json.loads(request.headers.get('x-codex-turn-metadata') or '{}')
                    compacting = metadata.get('request_kind') == 'compaction'
                    if compacting:
                        self.compaction_requests.append(payload)
                    number = len(self.calls)
                    await websocket.send_json({'type': 'response.created', 'response': {'id': f'resp_{number}'}})
                    if compacting and any(echo_marker(item) for item in payload.get('input', [])):
                        await websocket.send_json({'type': 'response.failed', 'response': {
                            'id': f'resp_{number}', 'status': 'failed',
                            'error': {'message': 'Test context overflow: old echo history was restored'}}})
                    else:
                        item = {'type': 'message', 'role': 'assistant', 'id': f'msg_{number}',
                                'content': [{'type': 'output_text', 'text': 'COMPACTION_OK' if compacting else 'READY'}]}
                        await websocket.send_json({'type': 'response.output_item.done', 'output_index': 0, 'item': item})
                        await websocket.send_json({'type': 'response.completed', 'response': {
                            'id': f'resp_{number}', 'status': 'completed', 'output': [item],
                            'usage': {'input_tokens': 10, 'output_tokens': 1, 'total_tokens': 11}}})
                    await websocket.close()
                    return websocket
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
                if websocket and self.echo_backend:
                    number = len(self.calls)
                    call = {'type': 'function_call', 'name': 'exec_command',
                            'id': f'fc_{number}', 'call_id': f'call_{number}',
                            'arguments': '{"cmd":"echo go"}'}
                    await websocket.send_json({'type': 'response.created', 'response': {'id': f'resp_{number}'}})
                    await websocket.send_json({'type': 'response.output_item.done', 'output_index': 0, 'item': call})
                    await websocket.send_json({'type': 'response.completed', 'response': {
                        'id': f'resp_{number}', 'status': 'completed', 'output': [call],
                        'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}}})
                    await websocket.close()
                    return websocket
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

    async def test_resumed_identity_wins_over_old_message_recipient(self):
        lead, helper = str(uuid.uuid4()), str(uuid.uuid4())
        self.gateway.agent_paths.update({lead: '/root', helper: '/root/worker'})
        for thread, recipient in ((lead, '/root/worker'), (helper, '/root')):
            response = await self.client.post('/v1/responses', headers=self.headers(thread), json={
                'instructions': 'Original instructions',
                'input': [{'type': 'agent_message', 'author': '/root',
                           'recipient': recipient, 'content': []}],
            })
            self.assertEqual(response.status, 200)
            await self.response_payload(response)
            instructions = json.loads(self.calls[-1][3])['instructions']
            if thread == lead:
                self.assertEqual(instructions, 'Original instructions')
            else:
                self.assertIn('Current agent identity: /root/worker', instructions)

    async def test_echo_recovery_lists_helpers_once_then_restores_normal_choice(self):
        from test_echo_recovery import loop
        thread = str(uuid.uuid4())
        original = {'input': [*loop(), {'role': 'user', 'content': 'Continue with helpers'}],
                    'tool_choice': 'auto',
                    'tools': [{'type': 'function', 'name': 'exec_command', 'parameters': {'type': 'object'}},
                              {'type': 'namespace', 'name': 'collaboration', 'tools': [
                        {'type': 'function', 'name': 'list_agents', 'parameters': {'type': 'object'}}]}]}
        response = await self.client.post('/v1/responses', headers=self.headers(thread), json=original)
        await self.response_payload(response)
        sent = json.loads(self.calls[-1][3])
        self.assertEqual(sent['tool_choice'], 'required')
        self.assertEqual([tool['name'] for tool in sent['tools']], ['collaboration__list_agents'])
        self.assertNotIn('echo-0', json.dumps(sent['input']))
        original['input'].extend([
            {'type': 'function_call', 'namespace': 'collaboration', 'name': 'list_agents',
             'call_id': 'inventory', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'inventory', 'output': 'helpers are idle'},
        ])
        response = await self.client.post('/v1/responses', headers=self.headers(thread), json=original)
        await self.response_payload(response)
        sent = json.loads(self.calls[-1][3])
        self.assertEqual(sent['tool_choice'], 'auto')
        self.assertEqual([tool['name'] for tool in sent['tools']], ['exec_command', 'collaboration__list_agents'])

    async def test_compaction_and_tools_disabled_requests_filter_spam_without_recovery_actions(self):
        from test_echo_recovery import loop
        summary = {'type': 'message', 'role': 'user', 'content': 'Summarize the useful work.'}
        for path, extra in (('/v1/responses/compact', {'tools': [{'type': 'function', 'name': 'exec_command'}]}),
                            ('/v1/responses', {'tools': [{'type': 'function', 'name': 'exec_command'}],
                                               'tool_choice': 'none'})):
            original = {'input': [*loop(), summary], **extra}
            response = await self.client.post(path, headers=self.headers(), json=original)
            self.assertEqual(response.status, 200)
            await self.response_payload(response)
            self.assertEqual(json.loads(self.calls[-1][3])['input'], [summary])
        response = await self.client.post('/v1/responses', headers={
            **self.headers(), 'x-codex-turn-metadata': '{"request_kind":"compaction"}'},
            json={'input': [*loop(), summary], 'tools': [{'type': 'namespace', 'name': 'collaboration',
                  'tools': [{'type': 'function', 'name': 'list_agents', 'parameters': {'type': 'object'}}]}]})
        self.assertEqual(response.status, 200)
        await self.response_payload(response)
        sent = json.loads(self.calls[-1][3])
        self.assertEqual(sent['input'], [summary])
        self.assertNotEqual(sent.get('tool_choice'), 'required')
        self.assertFalse(any(event[0] in ('swarm.echo_recovery_inventory', 'swarm.echo_loop_stopped')
                             for event in self.events))

    async def test_lead_uses_followup_but_helpers_can_still_send_messages(self):
        for identity in ('/root', '/root/helper'):
            thread = str(uuid.uuid4())
            self.gateway.agent_paths[thread] = identity
            response = await self.client.post('/v1/responses', headers=self.headers(thread), json={
                'tools': [{'type': 'namespace', 'name': 'collaboration', 'tools': [
                    {'type': 'function', 'name': name, 'parameters': {'type': 'object'}}
                    for name in ('followup_task', 'send_message')]}]})
            await self.response_payload(response)
            names = [tool['name'] for tool in json.loads(self.calls[-1][3])['tools']]
            self.assertIn('collaboration__followup_task', names)
            self.assertEqual('collaboration__send_message' in names, identity != '/root')

    async def test_new_echo_loop_stops_before_another_backend_request(self):
        from test_echo_recovery import loop
        response = await self.client.post('/v1/responses', headers=self.headers(), json={
            'input': [{'role': 'user', 'content': 'Use helpers'}, *loop()],
            'tools': [{'type': 'function', 'name': 'exec_command'}],
        })
        self.assertEqual(response.status, 400)
        self.assertIn('Marathon stopped this turn', await response.text())
        self.assertEqual(self.calls, [])

    async def test_non_object_request_is_rejected_before_history_inspection(self):
        response = await self.client.post('/v1/responses', headers=self.headers(), json=[])
        self.assertEqual(response.status, 400)
        self.assertIn('JSON objects', await response.text())
        self.assertEqual(self.calls, [])

    @unittest.skipUnless(os.environ.get('MARATHON_TEST_CODEX_BIN'), 'Requires installed Codex frontend')
    async def test_real_frontend_compacts_a_saved_echo_loop_without_restoring_spam(self):
        from test_echo_recovery import loop
        self.summary_backend = True
        with tempfile.TemporaryDirectory(prefix='marathon-echo-compaction-') as directory:
            provider = ('model_providers.marathon-local={name="Compaction test",wire_api="responses",'
                        f'base_url="{self.client.make_url("/v1")}",env_key="MARATHON_ROUTER_TOKEN",'
                        'requires_openai_auth=false,supports_websockets=false,stream_max_retries=0}')
            async def run(*arguments):
                process = await asyncio.create_subprocess_exec(
                    os.environ['MARATHON_TEST_CODEX_BIN'], '-c', provider,
                    '-c', 'model_provider="marathon-local"', '-c', 'sandbox_mode="read-only"', *arguments,
                    cwd=directory, env=dict(os.environ, CODEX_HOME=directory, CODEX_SQLITE_HOME=directory,
                                           MARATHON_LOCAL_ONLY='1', MARATHON_ROUTER_TOKEN='gateway-secret'),
                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
                    self.assertEqual(process.returncode, 0, (stdout + stderr).decode()[-3000:])
                finally:
                    if process.returncode is None:
                        process.terminate()
                        await process.wait()
            await run('exec', '--skip-git-repo-check', 'Reply READY')
            rollout = next(Path(directory).glob('sessions/**/*.jsonl'))
            recorded = [json.loads(line) for line in rollout.read_text().splitlines()]
            ordinal = max(event.get('ordinal', 0) for event in recorded) + 1
            # Seed only this disposable test session with historical call/result
            # pairs. No shell commands are executed to build the fixture.
            with rollout.open('a') as output:
                for item in loop():
                    if item['type'] != 'reasoning':
                        output.write(json.dumps({'timestamp': recorded[-1]['timestamp'], 'ordinal': ordinal,
                                                 'type': 'response_item', 'payload': item}) + '\n')
                        ordinal += 1
            await run('-c', 'model_auto_compact_token_limit=1', 'exec', 'resume', '--last',
                      '--skip-git-repo-check', 'Reply READY after compaction')
            self.assertTrue(self.compaction_requests)
            events = [json.loads(line) for line in rollout.read_text().splitlines()]
            self.assertTrue(any(event['type'] == 'compacted' and 'COMPACTION_OK' in
                                event['payload'].get('message', '') for event in events))
            self.assertTrue(any(event.get('payload', {}).get('call_id') == 'echo-0' for event in events))

    @unittest.skipUnless(os.environ.get('MARATHON_TEST_CODEX_BIN'), 'Requires installed Codex frontend')
    async def test_real_frontend_exits_a_repeating_shell_loop(self):
        from marathon_app.echo_recovery import ECHO_LIMIT
        self.echo_backend = True
        with tempfile.TemporaryDirectory(prefix='marathon-echo-guard-') as directory:
            provider = ('model_providers.marathon-local={name="Loop test",wire_api="responses",'
                        f'base_url="{self.client.make_url("/v1")}",env_key="MARATHON_ROUTER_TOKEN",'
                        'requires_openai_auth=false,supports_websockets=false}')
            process = await asyncio.create_subprocess_exec(
                os.environ['MARATHON_TEST_CODEX_BIN'], '-c', provider,
                '-c', 'model_provider="marathon-local"', '-c', 'sandbox_mode="workspace-write"',
                'exec', '--skip-git-repo-check', 'Run the test commands.',
                cwd=directory, env=dict(os.environ, CODEX_HOME=directory, CODEX_SQLITE_HOME=directory,
                                       MARATHON_LOCAL_ONLY='1', MARATHON_ROUTER_TOKEN='gateway-secret'),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(b'Marathon stopped this turn', stdout + stderr)
                self.assertEqual(len(self.calls), ECHO_LIMIT)
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
