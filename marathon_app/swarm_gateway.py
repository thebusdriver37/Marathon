"""Experimental sticky routing for Codex's native agent threads."""

import asyncio
import hmac
import json
import queue
import threading
import uuid

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from .swarm_tools import agent_identity, identify_agent, flatten_request, restore_calls, translated_sse


class SwarmGateway:
    def __init__(self, workers, token, record, *, max_agents=None):
        self.workers = workers
        self.token = token
        self.record = record
        self.max_agents = max_agents or len(workers)
        self.bindings = {}
        self.tool_names = {}
        self.agent_paths = {}
        self.stop = threading.Event()
        self.ready = queue.Queue()
        self.thread = None

    def application(self):
        app = web.Application(client_max_size=64 * 1024**2)
        app.router.add_route('*', '/{path:.*}', self.handle)
        app.cleanup_ctx.append(self.client_context)
        return app

    async def client_context(self, app):
        async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=10, sock_read=900),
                                 auto_decompress=False, trust_env=False) as client:
            self.client = client
            yield

    async def handle(self, request):
        if not hmac.compare_digest(request.headers.get('Authorization', ''), f'Bearer {self.token}'):
            raise web.HTTPUnauthorized()
        if request.path not in {'/v1/responses', '/v1/responses/compact', '/v1/models'}:
            raise web.HTTPNotFound()
        if request.headers.get('Upgrade'):
            raise web.HTTPBadRequest(text='Swarm currently uses HTTP streaming.')
        worker_index = 0
        thread_id = request.headers.get('thread-id', '')
        if request.path != '/v1/models':
            try:
                thread_id = str(uuid.UUID(thread_id))
            except ValueError:
                raise web.HTTPBadRequest(text='A valid Codex thread-id header is required.')
            if thread_id not in self.bindings:
                if len(self.bindings) == self.max_agents:
                    raise web.HTTPConflict(text='Swarm is full. Reuse the existing helper agents.')
                self.bindings[thread_id] = len(self.bindings) % len(self.workers)
                self.record('swarm.bound', {'thread': thread_id, 'worker': self.bindings[thread_id]})
            worker_index = self.bindings[thread_id]
        worker = self.workers[worker_index]
        body = await request.read()
        names = self.tool_names.setdefault(thread_id, {})
        payload = {}
        if body:
            try:
                original = json.loads(body)
                identity = self.agent_paths.setdefault(thread_id, agent_identity(original))
                payload = identify_agent(flatten_request(original, names), identity)
            except (ValueError, KeyError, TypeError) as error:
                raise web.HTTPBadRequest(text=str(error))
            # Codex can reuse the parent's cache key in children. Marathon
            # uses this key to supersede old turns, so isolate it per thread.
            payload['prompt_cache_key'] = f'marathon-swarm-{thread_id}'
            body = json.dumps(payload).encode()
        headers = {key: value for key, value in request.headers.items()
                   if key.lower() not in {'host', 'authorization', 'connection', 'transfer-encoding',
                                          'content-length', 'accept-encoding'}}
        headers['Authorization'] = f'Bearer {worker["router_token"]}'
        headers['Accept-Encoding'] = 'identity'
        request_id = uuid.uuid4().hex
        details = {'thread': thread_id, 'worker': worker_index, 'request': request_id, 'path': request.path}
        self.record('swarm.request.started', details)
        try:
            if request.path == '/v1/responses' and request.method == 'POST':
                # Use Marathon's full conversation engine, including native
                # patch conversion, tools, cancellation, and prompt caches.
                async with self.client.ws_connect(worker['router_url'] + request.path_qs,
                                                  headers=headers, max_msg_size=64 * 1024**2) as websocket:
                    await websocket.send_json({**payload, 'type': 'response.create'})
                    response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                    await response.prepare(request)
                    async for message in websocket:
                        if message.type != WSMsgType.TEXT:
                            break
                        event = restore_calls(json.loads(message.data), names)
                        try:
                            await response.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
                        except ConnectionResetError:
                            return response
                        if event.get('type') in {'response.completed', 'response.failed', 'error'}:
                            break
                    try:
                        await response.write_eof()
                    except ConnectionResetError:
                        pass
                    return response
            async with self.client.request(request.method, worker['router_url'] + request.path_qs,
                                           data=body, headers=headers,
                                           allow_redirects=False) as upstream:
                response = web.StreamResponse(status=upstream.status, headers={
                    key: value for key, value in upstream.headers.items()
                    if key.lower() not in {'connection', 'transfer-encoding', 'content-length'}
                })
                await response.prepare(request)
                try:
                    if upstream.content_type == 'text/event-stream':
                        async for chunk in translated_sse(upstream.content, names):
                            await response.write(chunk)
                    elif upstream.content_type == 'application/json':
                        await response.write(json.dumps(restore_calls(await upstream.json(), names)).encode())
                    else:
                        async for chunk in upstream.content.iter_any():
                            await response.write(chunk)
                    await response.write_eof()
                except ConnectionResetError:
                    pass  # Codex can close immediately after response.completed.
                return response
        finally:
            self.record('swarm.request.finished', details)

    async def serve(self):
        runner = web.AppRunner(self.application(), handler_cancellation=True, shutdown_timeout=2)
        try:
            await runner.setup()
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            self.ready.put(f'http://127.0.0.1:{runner.addresses[0][1]}')
            await asyncio.to_thread(self.stop.wait)
        finally:
            await runner.cleanup()

    def __enter__(self):
        def run():
            try:
                asyncio.run(self.serve())
            except Exception as error:
                self.ready.put(error)
        self.thread = threading.Thread(target=run, name='marathon-swarm-gateway', daemon=True)
        self.thread.start()
        try:
            result = self.ready.get(timeout=15)
        except queue.Empty:
            self.stop.set()
            self.thread.join(timeout=5)
            raise RuntimeError('Swarm gateway did not start within 15 seconds.')
        if isinstance(result, Exception):
            raise result
        self.url = result
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)
