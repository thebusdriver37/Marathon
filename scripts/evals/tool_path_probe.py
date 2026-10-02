#!/usr/bin/env python3
"""Bounded synthetic Spark layer probe. Saves only its own request/response data.

No generated command is executed. Fixed tool-result checkpoints keep histories
identical across arms instead of attributing conversation drift to the transport.
"""
import argparse
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/routers')]
import aiohttp
from aiohttp import web
import codex_local_router as router

TOOLS = [dict(type='function', name='exec_command', description='Run a shell command; returns output or a session ID.', parameters={
    'type': 'object', 'properties': {'cmd': {'type': 'string'}, 'workdir': {'type': 'string'},
    'yield_time_ms': {'type': 'integer'}, 'max_output_tokens': {'type': 'integer'},
    'tty': {'type': 'boolean'}, 'login': {'type': 'boolean'}}, 'required': ['cmd'], 'additionalProperties': False}),
    dict(type='function', name='write_stdin', description='Send input to an existing command session.', parameters={
    'type': 'object', 'properties': {'session_id': {'type': 'integer'}, 'chars': {'type': 'string'},
    'yield_time_ms': {'type': 'integer'}, 'max_output_tokens': {'type': 'integer'}},
    'required': ['session_id'], 'additionalProperties': False})]


def user(text):
    return {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}


def checkpoints():
    initial = [user('Synthetic coding task in /workspace. Inspect events.csv with exec_command. Then we will implement score.py and run its tests. Do the inspection now.')]
    observed = [*initial, {'type': 'function_call', 'name': 'exec_command', 'call_id': 'read_fixture',
        'arguments': '{"cmd":"cat events.csv","workdir":"/workspace"}'},
        {'type': 'function_call_output', 'call_id': 'read_fixture', 'output': 'start,end,label\n0,10,intro\n10,40,verse\n40,60,chorus\n'}]
    second = [*observed, user('Now create score.py using the available command tool. Implement boundary_prf(predicted, reference, tolerance=0.5) with one-to-one matching and return precision, recall, f1. Handle empty inputs. Use a complete Python heredoc. Do not just describe the code.')]
    third = [*second, {'type': 'function_call', 'name': 'exec_command', 'call_id': 'write_fixture',
        'arguments': '{"cmd":"python3 -c \\\"print(1)\\\""}'},
        {'type': 'function_call_output', 'call_id': 'write_fixture', 'output': 'Synthetic checkpoint: score.py exists. Tests failed: boundary_prf([], [], 0.5) returned (0,0,0), expected (1,1,1).'},
        user('Fix only the empty-empty case in score.py and run python3 -m unittest test_score.py using exec_command. Keep the rest unchanged. Make an actual tool call.')]
    return [initial, second, third]


async def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    os.environ.update(MARATHON_ROUTER_TOKEN=secrets.token_hex(24), MARATHON_TOOL_PROTOCOL_RECOVERIES='0',
        MARATHON_STALLED_RESPONSE_RECOVERIES='0', MARATHON_SLOT_SNAPSHOTS_ENABLED='0',
        MARATHON_SLOT_SAVE_ROOT=str(output/'slots'), MARATHON_RUN_LOG=str(output/'router-events.jsonl'))
    os.environ.pop('MARATHON_LAZY_POOL_BACKEND', None)
    shim_path = Path('/home/deforest/Documents/DEV/gpu-control/runtime/tf-responses-shim/responses_shim.py')
    spec = importlib.util.spec_from_file_location('probe_shim', shim_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    tunnel = subprocess.Popen(['ssh', '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
        '-L', f'127.0.0.1:{port}:127.0.0.1:8888', 'spark'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    state = router.RouterState('qwen3.8-flash-next-spark', output/'state', output/'logs')
    current = {}
    original_stream, original_json, original_events = state._request_responses_stream, state._request_json, state._iter_sse_json
    async def capture_stream(profile, payload, **kwargs):
        current.setdefault('forwarded', []).append(copy.deepcopy(payload))
        return await original_stream(profile, payload, **kwargs)
    async def capture_json(profile, method, path, payload, **kwargs):
        if path == '/v1/responses': current.setdefault('forwarded', []).append(copy.deepcopy(payload))
        return await original_json(profile, method, path, payload, **kwargs)
    async def capture_events(response):
        async for event in original_events(response):
            current.setdefault('upstream_events', []).append(copy.deepcopy(event))
            yield event
    state._request_responses_stream, state._request_json, state._iter_sse_json = capture_stream, capture_json, capture_events
    app = router.build_app(state)
    runner = web.AppRunner(app)
    summaries = []
    try:
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
        local = f'http://127.0.0.1:{runner.addresses[0][1]}'
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180), trust_env=False) as client:
            for _ in range(40):
                try:
                    async with client.get(f'http://127.0.0.1:{port}/v1/models') as r:
                        if r.status == 200: break
                except aiohttp.ClientError: pass
                await asyncio.sleep(.25)
            else: raise RuntimeError('test SSH forward did not become ready')
            for seed in (42, 43):
                for turn, history in enumerate(checkpoints()):
                    payload = {'model': 'Qwen3.8-Flash-Next', 'instructions': 'You are a coding assistant. Use the provided tool definitions. Never claim a command ran without its result.',
                        'input': history, 'tools': TOOLS, 'stream': True, 'seed': seed, 'temperature': 1.0,
                        'top_p': .95, 'max_output_tokens': 4096, 'reasoning': {'effort': 'medium'},
                        'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}}
                    order = ['raw', 'shim', 'marathon'] if seed == 42 else ['marathon', 'shim', 'raw']
                    for arm in order:
                        current = {'arm': arm, 'seed': seed, 'turn': turn}
                        body = copy.deepcopy(payload)
                        headers = {}
                        if arm == 'raw':
                            body = shim.responses_to_chat(body)
                            url = f'http://127.0.0.1:{port}/v1/chat/completions'
                        elif arm == 'shim': url = 'http://127.0.0.1:18088/v1/responses'
                        else:
                            body['model'] = state.default_model
                            url = local + '/v1/responses'
                            headers['Authorization'] = 'Bearer ' + os.environ['MARATHON_ROUTER_TOKEN']
                        current['request'] = body
                        start = time.monotonic()
                        async with client.post(url, json=body, headers=headers) as response:
                            current['status'] = response.status
                            current['raw_response'] = await response.text()
                        current['seconds'] = round(time.monotonic()-start, 3)
                        events = []
                        for line in current['raw_response'].splitlines():
                            if line.startswith('data:') and line[5:].strip() != '[DONE]':
                                try: events.append(json.loads(line[5:]))
                                except ValueError: pass
                        current['events'] = events
                        if arm == 'raw':
                            deltas = [choice.get('delta', {}) for e in events for choice in e.get('choices', [])]
                            text = ''.join(d.get('content') or '' for d in deltas)
                            names = [c.get('function', {}).get('name') for d in deltas for c in d.get('tool_calls', []) if c.get('function', {}).get('name')]
                        else:
                            text = ''.join(e.get('delta', '') for e in events if e.get('type') == 'response.output_text.delta')
                            names = [e['item'].get('name') for e in events if e.get('type') == 'response.output_item.done' and e.get('item', {}).get('type') == 'function_call']
                        summary = {k: current[k] for k in ('arm','seed','turn','status','seconds')}
                        summary.update(names=names, leaked_markup='<tool_call>' in text or '<function=' in text,
                            failed=any(e.get('type') in ('response.failed','error') for e in events),
                            upstream_requests=len(current.get('forwarded', [])))
                        current['summary'] = summary
                        (output/f'{seed}-{turn}-{arm}.json').write_text(json.dumps(current, indent=2))
                        summaries.append(summary)
                        (output/'summary.json').write_text(json.dumps(summaries, indent=2))
                        print(json.dumps(summary), flush=True)
    finally:
        await runner.cleanup()
        tunnel.terminate()
        await asyncio.to_thread(tunnel.wait, timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
