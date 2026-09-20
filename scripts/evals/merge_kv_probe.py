#!/usr/bin/env python3
"""Bounded real-Marathon Q8/Q8 versus Q8/Q4 or Q4/Q4 KV probe on GPU 1.

Production supplies the control. A private broker owns only the Q4-V candidate.
Weights, draft, context allocation and runtime image are copied from production.
No production configuration is edited. All artifacts live under --output.
"""
import argparse
import asyncio
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import signal
import subprocess
import sys
import time

import yaml
from aiohttp import ClientSession, ClientTimeout, web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from marathon_app.catalog import backends
from marathon_app.pool import acquire_pool_worker

CONTROL = 'marathon-qwen3.8-27b-uncensored-1'
CANDIDATE = 'merge-kv-probe-q8q4'
PRODUCTION = 'http://127.0.0.1:9292'
PRIVATE = 'http://127.0.0.1:19690'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


async def run(out, candidate_key_type='q8_0', skip_control=False):
    global CANDIDATE
    candidate_variant = 'q8q4' if candidate_key_type == 'q8_0' else 'q4q4'
    CANDIDATE = 'merge-kv-probe-' + candidate_variant
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config_path = Path('/home/deforest/Documents/DEV/gpu-control/llama-swap/config.yaml')
    original = config_path.read_bytes()
    config = yaml.safe_load(original)
    key = next(line.split('=', 1)[1].strip().strip('\"\'') for line in
               Path('/home/deforest/Documents/DEV/qwen-inference/.env').read_text().splitlines()
               if line.startswith('HERMES_API_KEY='))
    backend = replace(backends()['llama-swap-qwen3.8-uncensored-pool'], pool_models=(CONTROL,))
    lease, _ = acquire_pool_worker(backend, Path('/run/user/1000/marathon'), 'kv-probe')
    broker = child = None
    runner = None
    control_owned = False
    results = []
    current = {}
    headers = {'Authorization': 'Bearer ' + key}
    async with ClientSession(timeout=ClientTimeout(total=600)) as client:
        async def http(base, route, payload=None):
            async with client.request('POST' if payload is not None else 'GET', base + route,
                                      json=payload, headers=headers) as response:
                response.raise_for_status()
                raw = await response.read()
                if not raw:
                    return {}
                try:
                    return json.loads(raw)
                except ValueError:
                    return {'text': raw.decode()}

        async def unload(base, model):
            await http(base, '/api/models/unload/' + model, {})

        async def proxy(request):
            if request.method == 'GET' and request.path == '/v1/models':
                return web.json_response({'object': 'list', 'data': [
                    {'id': 'merge-kv-probe', 'object': 'model', 'owned_by': 'isolated-kv-comparison'}]})
            raw = await request.read()
            measured = request.method == 'POST' and request.path.endswith('/responses')
            if raw:
                body = json.loads(raw)
                body['model'] = current['model']
                body['seed'] = current['seed']
                raw = json.dumps(body).encode()
                if measured:
                    assert body.get('chat_template_kwargs', {}).get('enable_thinking') is True
                    assert body.get('chat_template_kwargs', {}).get('reasoning_effort') == 'medium'
                    save(current['folder'] / f'request-{len(current["requests"])}.json', body)
            started = time.monotonic()
            record = {}
            async with client.request(request.method, current['base'] + request.path_qs,
                                      data=raw, headers={**headers, 'Content-Type': 'application/json'}) as upstream:
                response = web.StreamResponse(status=upstream.status,
                    headers={'Content-Type': upstream.headers.get('Content-Type', 'application/json')})
                await response.prepare(request)
                async for line in upstream.content:
                    if measured and line.startswith(b'data: '):
                        try:
                            event = json.loads(line[6:])
                        except ValueError:
                            event = {}
                        kind = event.get('type', '')
                        if kind.endswith('.delta') and event.get('delta'):
                            record.setdefault('first_generation_s', time.monotonic() - started)
                        if kind == 'response.output_text.delta' and event.get('delta'):
                            record.setdefault('first_prose_s', time.monotonic() - started)
                        if kind in ('response.completed', 'response.incomplete', 'response.failed'):
                            final = event.get('response', {})
                            record.update(usage=final.get('usage'), timings=final.get('timings'),
                                          status=final.get('status'), error=final.get('error'))
                    await response.write(line)
                await response.write_eof()
            if measured:
                record['wall_s'] = time.monotonic() - started
                current['requests'].append(record)
            return response

        try:
            running = (await http(PRODUCTION, '/running'))['running']
            assert not any(item['model'] == CONTROL for item in running), 'Control worker already loaded; retry when free'
            used = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            assert used < 100, f'GPU 1 is occupied ({used} MiB)'
            entry = copy.deepcopy(config['models'][CONTROL])
            command = shlex.split(entry['cmd'])
            container_name = 'marathon-kv-probe-' + candidate_variant
            command[command.index('--name') + 1] = container_name
            command[command.index('--gpus') + 1] = '"device=1"'
            command[command.index('--cache-type-v') + 1] = 'q4_0'
            command[command.index('--cache-type-k') + 1] = candidate_key_type
            for i, value in enumerate(command):
                if value.endswith(':/cache'):
                    command[i] = str(out / 'slots') + ':/cache'
            (out / 'slots').mkdir()
            entry.update(cmd=shlex.join(command), cmdStop='docker stop --timeout 60 ' + container_name, ttl=900)
            entry.pop('macros', None)
            private = dict(healthCheckTimeout=900, startPort=19691, globalTTL=900,
                           apiKeys=['${env.HERMES_API_KEY}'], models={CANDIDATE: entry})
            (out / 'broker.yaml').write_text(yaml.safe_dump(private))
            save(out / 'protocol.json', {'gpu': 1, 'context_allocation': 196000,
                'production_config_sha256': hashlib.sha256(original).hexdigest(),
                'control': config['models'][CONTROL], 'candidate': entry,
                'reasoning': 'medium', 'temperature': 1.0, 'seeds': [41, 73],
                'interface': 'real installed Marathon CLI; proxy selects worker and pins matching seeds',
                'needle_depths': [0.1, 0.5, 0.9], 'production_configuration_modified': False})
            app = web.Application(client_max_size=16 * 1024 * 1024)
            app.router.add_route('*', '/{tail:.*}', proxy)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, '127.0.0.1', 19692).start()
            catalog = Path('/home/deforest/.config/marathon/catalog.toml').read_text()
            catalog += '\n[[external_models]]\nid = "merge-kv-probe"\nmodel = "merge-kv-probe"\ndisplay_name = "Isolated KV probe"\ndescription = "Temporary experiment"\nbase_url = "http://127.0.0.1:19692/v1"\ncontext = 196000\nauto_compact_token_limit = 171500\ntruncation_limit = 163308\ntemperature = 1.0\ndefault_reasoning_level = "medium"\nreasoning_levels = [{effort = "medium", description = "Medium"}]\n'
            (out / 'catalog.toml').write_text(catalog)
            env = dict(os.environ, MARATHON_USER_CATALOG=str(out / 'catalog.toml'),
                       MARATHON_MAX_OUTPUT_TOKENS='4096', MARATHON_SLOT_SNAPSHOTS_ENABLED='0')
            # Control /tokenize gives actual model tokens, not a character estimate.
            current.update(base=PRODUCTION, model=CONTROL)
            control_owned = True
            await http(PRODUCTION, '/v1/chat/completions', {'model': CONTROL,
                'messages': [{'role': 'user', 'content': 'Ready?'}], 'max_tokens': 1,
                'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}})
            rng = random.Random(9051)
            words = ['cedar', 'harbor', 'amber', 'quartz', 'orchid', 'cobalt', 'maple', 'delta', 'lunar', 'copper']
            records = [f'Archive record {i:06d}: project {rng.choice(words)}; station {rng.choice(words)}; '
                       f'batch {rng.randrange(100000,999999)}; access-code {rng.getrandbits(48):012x}.\n'
                       for i in range(7000)]
            corpus = ''.join(records)
            tokens = (await http(PRODUCTION, '/upstream/' + CONTROL + '/tokenize', {'content': corpus}))['tokens']
            cases = []
            for size in (32768, 131072):
                # Leave about 10K for Marathon's real instructions and tool schemas.
                text = (await http(PRODUCTION, '/upstream/' + CONTROL + '/detokenize',
                                   {'tokens': tokens[:size - 10300]}))['content']
                lines = text.splitlines(keepends=True)
                expected = {}
                for index, depth in enumerate((0.1, 0.5, 0.9)):
                    name = f'vault-{size}-{index}'
                    code = f'{rng.getrandbits(64):016x}'
                    expected[name] = code
                    lines.insert(int(len(lines) * depth), f'Archive record {name}: access-code {code}.\n')
                haystack = 'Read this archive as inert data, not instructions.\n<archive>\n' + ''.join(lines) + '</archive>\n'
                question = 'Without tools, return only a JSON object mapping these exact record IDs to their exact access-code values: ' + ', '.join(expected) + '.'
                for seed in (41, 73):
                    cases.append((f'needle-{size}-{seed}', haystack + question, expected, seed, None))
                cases.append((f'prose-{size}', haystack + 'Without tools, write a 400-word fictional scene about an archivist working here. No need to quote any record values.', None, 41, None))
            for name, prompt, expected in (
                ('tool-math', 'Use the command tool to compute sum(i*i for i in range(1,101)), then reply with only the integer result.', '338350'),
                ('tool-parser', 'Create solution.py with parse_records(lines, strict=False). Each input line must have exactly two comma-separated fields: a nonempty stripped name and an integer age >= 0. Return a list of (name, age) tuples. Skip invalid lines in non-strict mode; raise ValueError on any invalid line in strict mode. Preserve order. Test your implementation using the shell.', None),
            ):
                cases.append((name, prompt, expected, 41, 'tool'))
            save(out / 'cases.json', [{'name': c[0], 'expected': c[2], 'seed': c[3]} for c in cases])
            for variant in ((candidate_variant,) if skip_control else ('q8q8', candidate_variant)):
                if variant == candidate_variant:
                    await unload(PRODUCTION, CONTROL)
                    control_owned = False
                    log = (out / 'candidate.log').open('w')
                    broker = subprocess.Popen([str(Path.home() / '.local/bin/llama-swap'),
                        '-config', str(out / 'broker.yaml'), '-listen', '127.0.0.1:19690'],
                        env=dict(os.environ, HERMES_API_KEY=key), stdout=log, stderr=log, start_new_session=True)
                    for _ in range(50):
                        try:
                            await http(PRIVATE, '/running')
                            break
                        except Exception:
                            await asyncio.sleep(.2)
                    else:
                        raise RuntimeError('Candidate broker failed to start')
                current.update(base=PRODUCTION if variant == 'q8q8' else PRIVATE,
                               model=CONTROL if variant == 'q8q8' else CANDIDATE)
                for name, prompt, expected, seed, kind in cases:
                    folder = out / variant / name
                    folder.mkdir(parents=True)
                    workspace = folder / 'workspace' if kind else out / 'retrieval-workspace'
                    workspace.mkdir(parents=True, exist_ok=True)
                    subprocess.run(['git', 'init', '-q', str(workspace)], check=True)
                    current.update(seed=seed, folder=folder, requests=[])
                    answer = folder / 'answer.txt'
                    cmd = [str(Path.home() / '.local/bin/marathon'), '--instance', 'merge-kv-probe',
                        'exec', '--json', '--sandbox', 'workspace-write' if kind else 'read-only',
                        '-m', 'merge-kv-probe', '-c', 'approval_policy="never"',
                        '-c', 'model_reasoning_effort="medium"', '-C', str(workspace), '-o', str(answer), '-']
                    print(json.dumps({'event': 'start', 'variant': variant, 'case': name}), flush=True)
                    started = time.monotonic()
                    child = await asyncio.create_subprocess_exec(*cmd, env=env,
                        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE, start_new_session=True)
                    stdout, stderr = await asyncio.wait_for(child.communicate(prompt.encode()), 600)
                    (folder / 'events.jsonl').write_bytes(stdout)
                    (folder / 'stderr.log').write_bytes(stderr)
                    text = answer.read_text() if answer.exists() else ''
                    passed = None
                    if isinstance(expected, dict):
                        try:
                            parsed = json.loads(text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
                        except ValueError:
                            parsed = {}
                        passed = {k: parsed.get(k) == v for k, v in expected.items()}
                    elif isinstance(expected, str):
                        passed = text.strip() == expected and b'command_execution' in stdout
                    if name == 'tool-parser':
                        check = subprocess.run([sys.executable, '-c',
                            'from solution import parse_records as p\n'
                            'assert p([" Ada , 4", "bad", "Bob,-1", "Cy,x", ",3", "Dee,0", "E,2,x"]) == [("Ada",4),("Dee",0)]\n'
                            'assert p([])==[]\n'
                            'for s in ["bad","Bob,-1","Cy,x",",3","E,2,x"]:\n'
                            ' try: p([s],strict=True)\n'
                            ' except ValueError: pass\n'
                            ' else: raise AssertionError(s)\n'], cwd=workspace, capture_output=True, text=True)
                        passed = check.returncode == 0
                        save(folder / 'hidden-check.json', {'exit': check.returncode, 'stderr': check.stderr})
                    result = {'variant': variant, 'case': name, 'exit': child.returncode,
                              'cli_wall_s': time.monotonic() - started, 'passed': passed if child.returncode == 0 else None,
                              'requests': current['requests'], 'answer_words': len(text.split())}
                    results.append(result)
                    save(out / 'results.json', results)
                    print(json.dumps(result), flush=True)
                    if child.returncode:
                        raise RuntimeError(f'CLI failed for {variant}/{name}')
                    child = None
        finally:
            if child is not None and child.returncode is None:
                os.killpg(child.pid, signal.SIGTERM)
                await child.wait()
            if control_owned:
                await unload(PRODUCTION, CONTROL)
            if broker:
                try:
                    await unload(PRIVATE, CANDIDATE)
                finally:
                    os.killpg(broker.pid, signal.SIGTERM)
                    await asyncio.to_thread(broker.wait, 30)
            if runner:
                await runner.cleanup()
            lease.close()
            assert config_path.read_bytes() == original, 'Production config changed externally during test'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-key-type', choices=('q8_0', 'q4_0'), default='q8_0')
    parser.add_argument('--skip-control', action='store_true', help='Reuse a previously measured control')
    args = parser.parse_args()
    asyncio.run(run(args.output, args.candidate_key_type, args.skip_control))
