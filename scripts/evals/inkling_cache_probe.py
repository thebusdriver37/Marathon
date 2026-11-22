#!/usr/bin/env python3
"""Exercise durable native caching through an isolated, real Marathon router.

The caller holds the GPU lease and owns the worker. After cold-done.json appears,
restart that worker, wait for its API to be ready, then create resume-now. This
probe never loads, unloads, or changes a worker or reads an installed catalog.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import secrets
import sys
import time

from aiohttp import ClientSession, ClientTimeout

ROOT = Path(__file__).resolve().parents[2]


def router_environment(output, upstream):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('MARATHON_', 'CODEX_LLAMA_'))}
    catalog = output / 'catalog.toml'
    catalog.write_text('[settings]\nmodel_root = "models"\nllama_host = "127.0.0.1"\nllama_port = 18393\nrouter_host = "127.0.0.1"\nrouter_port = 18111\nhealth_timeout = 30\nstop_timeout = 10\n')
    env.update(MARATHON_ROUTER_TOKEN=secrets.token_urlsafe(32),
               MARATHON_CATALOG=str(catalog),
               MARATHON_USER_CATALOG=str(output / 'absent-user-catalog.toml'),
               MARATHON_MODEL_PATH='/synthetic/unused.gguf',
               MARATHON_MODEL_SLUG='inkling-2-mini',
               MARATHON_MODEL_EXTERNAL='1',
               MARATHON_MODEL_TARGET=upstream,
               MARATHON_MODEL_CONTEXT='262144',
               MARATHON_MODEL_REASONING_TRANSPORT='responses',
               MARATHON_BACKEND_SLOT_API='0',
               MARATHON_SLOT_SNAPSHOTS_ENABLED='0',
               MARATHON_MAX_OUTPUT_TOKENS='512',
               MARATHON_RUN_LOG=str(output / 'telemetry.jsonl'),
               XDG_STATE_HOME=str(output / 'xdg-state'),
               PYTHONPATH=str(ROOT))
    return env


async def request(client, endpoint, payload):
    started = time.monotonic()
    record = {}
    async with client.post(endpoint + '/v1/responses', json={**payload, 'stream': True}) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b'data: '):
                continue
            event = json.loads(line[6:])
            kind = event.get('type', '')
            if kind.endswith('.delta') and event.get('delta'):
                record.setdefault('ttft_s', time.monotonic() - started)
            if kind == 'response.failed':
                raise RuntimeError(event)
            if kind == 'response.completed':
                record['response'] = event['response']
    record['wall_s'] = time.monotonic() - started
    if 'response' not in record:
        raise RuntimeError('No completed response')
    result = record['response']
    record['cache_n'] = result.get('timings', {}).get('cache_n', 0)
    record['input_tokens'] = result.get('usage', {}).get('input_tokens', 0)
    record['cached_tokens'] = result.get('usage', {}).get('input_tokens_details', {}).get('cached_tokens', 0)
    return record


async def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # The file is explicitly supplied synthetic benchmark material only.
    prompt = args.prompt_file.read_text() if args.prompt_file else (
        'Synthetic record: the lighthouse access code is CEDAR-731.\n' +
        'Ordinary inventory entry: copper cable, sealed carton, north warehouse.\n' * 8500 +
        '\nWhat is the lighthouse access code? Return only the code.')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = router_environment(out, args.upstream)
    with (out / 'router.log').open('w') as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(ROOT / 'scripts/routers/codex_local_router.py'),
            '--port', str(port), '--state-dir', str(out / 'state'),
            '--log-dir', str(out / 'logs'), env=env, cwd=ROOT,
            stdout=log, stderr=log)
        try:
            endpoint = f'http://127.0.0.1:{port}'
            async with ClientSession(timeout=ClientTimeout(total=600), headers={"Authorization": "Bearer " + env["MARATHON_ROUTER_TOKEN"]}) as client:
                for _ in range(100):
                    if process.returncode is not None:
                        raise RuntimeError('Router exited: ' + (out / 'router.log').read_text()[-4000:])
                    try:
                        async with client.get(endpoint + '/health') as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    await asyncio.sleep(.1)
                else:
                    raise RuntimeError('Router failed readiness')
                payload = {'model': 'inkling-2-mini', 'input': prompt,
                           'max_output_tokens': 512, 'temperature': 0,
                           'reasoning': {'effort': 'none'},
                           'prompt_cache_key': 'synthetic-inkling-cache-probe'}
                cold = await request(client, endpoint, payload)
                (out / 'cold-done.json').write_text(json.dumps(cold, indent=2))
                print('Cold done. Restart the leased worker and create ' + str(out / 'resume-now'), flush=True)
                deadline = time.monotonic() + args.restart_timeout
                while not (out / 'resume-now').exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('No worker restart marker')
                    await asyncio.sleep(.2)
                # Full replay represents a client reconnect/session reload.
                restored = await request(client, endpoint, payload)
                continuation = await request(client, endpoint, {
                    **payload, 'previous_response_id': restored['response']['id'],
                    'input': 'Repeat your previous answer exactly.'})
                result = {'cold': cold, 'restored': restored, 'continuation': continuation}
                if cold.get('ttft_s') and restored.get('ttft_s'):
                    result['ttft_speedup'] = cold['ttft_s'] / restored['ttft_s']
                result['reuse_fraction'] = restored['cached_tokens'] / max(1, restored['input_tokens'])
                (out / 'results.json').write_text(json.dumps(result, indent=2))
                print(json.dumps({key: value for key, value in result.items()
                                  if key not in ('cold', 'restored', 'continuation')}), flush=True)
                return result
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 15)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--upstream', default='http://127.0.0.1:18393')
    parser.add_argument('--prompt-file', type=Path, help='Explicitly supplied synthetic text only')
    parser.add_argument('--restart-timeout', type=float, default=1800)
    args = parser.parse_args()
    if not args.upstream.startswith('http://127.0.0.1:'):
        parser.error('Use a standalone loopback worker')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
