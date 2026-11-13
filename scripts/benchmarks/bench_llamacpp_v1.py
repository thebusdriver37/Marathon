#!/usr/bin/env python3
"""Isolated synthetic comparison with the deployed Marathon command and fresh caches.

Never reads conversations, production snapshots, credentials or model content.
Only the inference server opens the explicitly mounted model files.
"""
import argparse
import base64
import io
import importlib.util
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import SimpleNamespace
import urllib.request
import urllib.error

import yaml

CONFIG = Path('/home/deforest/Documents/DEV/gpu-control/llama-swap/config.yaml')
IMAGE = 'sha256:c2a893769acc09eff58da1dfd2b42b00b4634a6d231a3dc62eac5a154f3df793'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--binary', type=Path)
    ap.add_argument('--image', default=IMAGE, help='Pinned image digest; defaults to the deployed v5 baseline')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--port', type=int, default=19873)
    ap.add_argument('--lines', type=int, nargs='+', default=[300, 1200])
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--quality', action='store_true')
    ap.add_argument('--app', action='store_true', help='Two small two-turn tasks through the real isolated Marathon frontend/router')
    ap.add_argument('--export-snapshot', type=Path, help='Export this run\'s synthetic snapshot for a migration test')
    ap.add_argument('--import-snapshot', type=Path, help='Test rejection of a synthetic snapshot from the old runtime')
    ap.add_argument('--context-cache', type=Path, help='Job-owned synthetic v1 snapshots for warm-only kernel comparisons; use a fresh RAM directory')
    ap.add_argument('--recall', action=argparse.BooleanOptionalAction, default=True, help='Check exact retrieval across each occupied context')
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--env', action='append', default=[], help='Explicit benchmark-only container environment')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.context_cache:
        if not a.binary and a.image == IMAGE:
            ap.error('--context-cache requires a v1 candidate')
        a.context_cache.mkdir(parents=True, exist_ok=True)
    slots = (a.context_cache / (a.label + '-slots')) if a.context_cache else a.output / 'slots'
    slots.mkdir()
    url = f'http://127.0.0.1:{a.port}'
    name = f'marathon-v1-{a.label}-gpu{a.gpu}'
    record = dict(label=a.label, gpu=a.gpu, rows=[], checks=[], baseline_image=IMAGE, image=a.image)

    def save():
        (a.output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')

    def http(path, data=None, timeout=1800):
        req = urllib.request.Request(url + path, data=None if data is None else json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def check(label, ok, **details):
        row = dict(check=label, passed=bool(ok), **details)
        record['checks'].append(row)
        save()
        print(json.dumps({k: v for k, v in row.items() if k not in ('response', 'result')}), flush=True)

    def complete(prompt, n=256, **kwargs):
        start = time.monotonic()
        r = http('/completion', dict(prompt=prompt, n_predict=n, temperature=0, seed=424242,
                                    cache_prompt=True, id_slot=0, **kwargs))
        r['wall_s'] = time.monotonic() - start
        return r

    def template(text):
        return http('/apply-template', dict(messages=[dict(role='user', content=text)],
                    add_generation_prompt=True, chat_template_kwargs={'enable_thinking': False}))['prompt']

    # Inspect occupancy immediately before launch; never stop another process.
    lease = None
    if a.gpu in (1, 2, 3):
        model_id = f'marathon-qwen3.8-27b-uncensored-{a.gpu}'
        pool = Path(f'/run/user/{os.getuid()}/marathon/backend-pools/llama-swap-qwen3.8-uncensored-pool')
        pool.mkdir(parents=True, exist_ok=True)
        suffix = hashlib.sha256(model_id.encode()).hexdigest()[:12]
        lease = (pool / f'{model_id}-{suffix}.lock').open('a+')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if occupied:
        raise RuntimeError(f'GPU {a.gpu} occupied: {occupied}')
    config = yaml.safe_load(CONFIG.read_text())
    original = config['models']['marathon-qwen3.8-27b-uncensored-1']['cmd']
    args = shlex.split(original)
    image_at = next(i for i, value in enumerate(args) if value.startswith('sha256:'))
    server_args = args[image_at + 1:]
    if not a.binary and a.image == IMAGE and '--spec-draft-n-ubatch' in server_args:
        at = server_args.index('--spec-draft-n-ubatch')
        del server_args[at:at + 2]
    elif (a.binary or a.image != IMAGE) and '--spec-draft-n-ubatch' not in server_args:
        server_args += ['--spec-draft-n-ubatch', '64']
    mounts = {}
    for i, value in enumerate(args[:image_at]):
        if value == '--volume':
            host, target, *mode = args[i + 1].split(':')
            mounts[target] = host
    server_args[server_args.index('--slot-save-path') + 1] = '/cache'
    record['server_args'] = server_args
    proc = None
    log_follower = None
    log = (a.output / 'server.log').open('w')
    try:
        cmd = ['docker', 'run', '-d', '--rm', '--init', '--name', name, '--pull', 'never',
               '--gpus', f'"device={a.gpu}"', '--publish', f'127.0.0.1:{a.port}:8080',
               '--ulimit', 'memlock=-1:-1', '--env', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility']
        for target, host in mounts.items():
            if target != '/cache':
                cmd += ['--volume', f'{host}:{target}:ro']
        cmd += ['--volume', f'{slots}:/cache']
        if a.binary:
            cmd += ['--volume', f'{a.binary.resolve().parent}:/candidate:ro',
                    '--entrypoint', '/candidate/' + a.binary.name,
                    '--env', 'LD_PRELOAD=', '--env', 'LD_LIBRARY_PATH=/candidate:/usr/local/cuda/lib64']
            record['binary'] = str(a.binary)
        for setting in a.env:
            cmd += ['--env', setting]
        record['extra_env'] = a.env
        cmd += [a.image, *server_args, *a.extra]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        log_follower = subprocess.Popen(['docker', 'logs', '-f', name], stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        while True:
            try:
                if http('/health', timeout=2).get('status') == 'ok':
                    break
            except Exception:
                pass
            status = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name], capture_output=True, text=True)
            if status.returncode or status.stdout.strip() != 'true':
                raise RuntimeError('Benchmark container exited before readiness; see server.log')
            if time.monotonic() - start > 600:
                raise TimeoutError('Server did not become ready')
            time.sleep(2)
        record['startup_s'] = time.monotonic() - start
        check('health_and_models', bool(http('/v1/models').get('data')))
        if a.import_snapshot:
            shutil.copyfile(a.import_snapshot, slots / 'old-synthetic.bin')
            try:
                restored = http('/slots/0?action=restore', {'filename': 'old-synthetic.bin'})
                check('old_snapshot_rejected', False, response=restored)
            except urllib.error.HTTPError as exc:
                check('old_snapshot_rejected', exc.code == 400, status=exc.code)
            http('/slots/0?action=erase', {})
            r = complete(template('What is 17 times 23? Answer only the number.'), 32)
            check('fresh_inference_after_old_snapshot', '391' in r['content'])
        record['gpu_loaded'] = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
               '--query-gpu=name,memory.used,power.limit', '--format=csv,noheader,nounits'], text=True).strip()
        if a.quality:
            tasks = [
                ('intervals', 'Implement merge_intervals(items) for closed numeric intervals. Merge touching intervals; sort the result; do not mutate inputs; reject reversed intervals with ValueError. Return a list of tuples.',
                 "x=[(5,7),(1,3),(3,5),(12,12)]; assert merge_intervals(x)==[(1,7),(12,12)]; assert x==[(5,7),(1,3),(3,5),(12,12)]; assert merge_intervals([])==[]; assert merge_intervals([(-3,-1),(-2,0)])==[(-3,0)]\ntry: merge_intervals([(4,2)])\nexcept ValueError: pass\nelse: raise AssertionError('reversed interval accepted')"),
                ('toposort', 'Implement topo(graph), where graph maps string node names to lists of dependency names. Return a deterministic topological order by choosing the alphabetically first currently ready node on each step. Raise ValueError on cycles or unknown dependencies, do not mutate inputs.',
                 "g={'c':['a'],'b':[],'a':[]}; assert topo(g)==['a','b','c']; assert g=={'c':['a'],'b':[],'a':[]}; assert topo({})==[]; assert topo({'a':[],'b':['a','a']})==['a','b']\nfor g in [{'a':['b'],'b':['a']},{'a':['missing']},{'a':['a']}]:\n try: topo(g)\n except ValueError: pass\n else: raise AssertionError('invalid graph accepted')"),
                ('csv', 'Implement totals(text) reading CSV with headers item,quantity. Sum integer quantities for each item; support quoted item names containing commas, negative quantities, blank lines, and CRLF. Return a dict; invalid noninteger quantities must raise ValueError.',
                 "assert totals('item,quantity\\r\\n\"a,b\",3\\r\\nx,2\\r\\n\"a,b\",-1\\r\\n')=={'a,b':2,'x':2}; assert totals('item,quantity\\n')=={}\ntry: totals('item,quantity\\na,nope\\n')\nexcept ValueError: pass\nelse: raise AssertionError('bad quantity accepted')"),
            ]
            for label, task, grade in tasks:
                r = http('/v1/chat/completions', dict(model='qwen3.8-27b-uncensored',
                         messages=[dict(role='user', content=task + ' Use Python standard library only. Return JSON with a single code string containing the complete implementation. No markdown.')],
                         response_format={'type': 'json_object'}, max_tokens=1800, temperature=0,
                         chat_template_kwargs={'enable_thinking': False}))
                content = r['choices'][0]['message'].get('content') or ''
                try:
                    code = json.loads(content)['code']
                    cmd = ['bwrap', '--unshare-all', '--die-with-parent', '--cap-drop', 'ALL', '--clearenv',
                           '--ro-bind', '/usr', '/usr', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
                    for path in ['/bin', '/lib', '/lib64']:
                        if Path(path).exists():
                            cmd += ['--ro-bind', path, path]
                    result = subprocess.run(cmd + ['/usr/bin/python3', '-I', '-c', code + '\n' + grade],
                                            capture_output=True, text=True, timeout=10)
                    check('code_' + label, result.returncode == 0, response=r, grading=result.stderr[-2000:])
                except Exception as exc:
                    check('code_' + label, False, response=r, grading=str(exc))
            # Exercise SSE parsing and disconnect cancellation without waiting for a full response.
            req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(dict(
                model='qwen3.8-27b-uncensored', messages=[dict(role='user', content='Count from 1 to 10000, one number per line.')],
                max_tokens=4096, stream=True, temperature=0,
                chat_template_kwargs={'enable_thinking': False})).encode(), headers={'Content-Type': 'application/json'})
            chunks = []
            with urllib.request.urlopen(req, timeout=120) as response:
                for raw in response:
                    if raw.startswith(b'data: ') and b'[DONE]' not in raw:
                        chunks.append(json.loads(raw[6:]))
                        if len(chunks) >= 5:
                            break
            deadline = time.monotonic() + 15
            idle = False
            while time.monotonic() < deadline:
                idle = not any(s.get('is_processing') for s in http('/slots'))
                if idle:
                    break
                time.sleep(.25)
            check('stream_disconnect_cancellation', len(chunks) >= 5 and idle)
        if a.smoke:
            r = http('/v1/chat/completions', dict(model='qwen3.8-27b-uncensored',
                     messages=[dict(role='user', content='What is 17 times 23? Answer only the number.')],
                     max_tokens=32, temperature=0, chat_template_kwargs={'enable_thinking': False}))
            check('arithmetic', '391' in (r['choices'][0]['message'].get('content') or ''), response=r)
            r = http('/v1/chat/completions', dict(model='qwen3.8-27b-uncensored',
                     messages=[dict(role='user', content='Get the weather in Paris using the weather tool.')],
                     tools=[dict(type='function', function=dict(name='weather', description='Get current weather',
                           parameters=dict(type='object', properties=dict(city=dict(type='string')), required=['city'])))],
                     tool_choice='required', max_tokens=96, temperature=0,
                     chat_template_kwargs={'enable_thinking': False}))
            calls = r['choices'][0]['message'].get('tool_calls', [])
            check('tool_call', bool(calls) and calls[0]['function']['name'] == 'weather'
                  and json.loads(calls[0]['function']['arguments']).get('city', '').lower() == 'paris', response=r)
            from PIL import Image, ImageDraw
            im = Image.new('RGB', (448, 224), 'white'); draw = ImageDraw.Draw(im)
            draw.rectangle((25, 50, 160, 185), fill='red'); draw.ellipse((270, 50, 405, 185), fill='blue')
            data = io.BytesIO(); im.save(data, format='PNG')
            messages = [dict(role='user', content=[dict(type='text', text='Describe the colored shapes and their left/right positions.'),
                        dict(type='image_url', image_url=dict(url='data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()))])]
            for phase in ['cold', 'warm', 'followup']:
                r = http('/v1/chat/completions', dict(model='qwen3.8-27b-uncensored', messages=messages,
                         max_tokens=96, temperature=0, chat_template_kwargs={'enable_thinking': False}))
                content = (r['choices'][0]['message'].get('content') or '').lower()
                check('image_' + phase, all(x in content for x in (['blue', 'circle'] if phase == 'followup'
                      else ['red', 'square', 'left', 'blue', 'circle', 'right'])), response=r)
                if phase == 'warm':
                    messages.extend([r['choices'][0]['message'], dict(role='user', content='Which shape is blue?')])
        for count in a.lines:
            prefix = '\n'.join(f'Ledger entry {j}: warehouse {j % 97}, pallet {j * 7919 % 100003}, inspected and sealed.' for j in range(count))
            prefix += '\nThe final marker is CORAL-7182.\n'
            prompt = template(prefix + 'Return the final marker only.')
            http('/slots/0?action=erase', {})
            cached = a.context_cache / f'context-{count}.bin' if a.context_cache else None
            reused = bool(cached and cached.exists())
            if reused:
                shutil.copyfile(cached, slots / cached.name)
                http('/slots/0?action=restore', {'filename': cached.name})
            r = complete(prompt, 24)
            check(f'needle_{count}', 'CORAL-7182' in r['content'], timings=r['timings'])
            record['rows'].append(dict(lines=count, task='restored_context' if reused else 'cold_prefill',
                                      timings=r['timings'], wall_s=r['wall_s']))
            if cached and not reused:
                http('/slots/0?action=save', {'filename': cached.name})
                shutil.copyfile(slots / cached.name, cached)
                cached.with_suffix('.json').write_text(json.dumps(dict(source_label=a.label,
                    binary=str(a.binary), lines=count, server_args=server_args)) + '\n')
            if count == a.lines[0] and a.smoke:
                saved = http('/slots/0?action=save', {'filename': 'synthetic.bin'})
                if a.export_snapshot:
                    with a.export_snapshot.open('xb') as target, (slots / 'synthetic.bin').open('rb') as source:
                        shutil.copyfileobj(source, target)
                http('/slots/0?action=erase', {})
                restored = http('/slots/0?action=restore', {'filename': 'synthetic.bin'})
                branch = complete(template(prefix + 'State the final marker. Nothing else.'), 24)
                check('snapshot_divergent_suffix', 'CORAL-7182' in branch['content'] and branch['timings'].get('cache_n', 0) > 1000,
                      save=saved, restore=restored, timings=branch['timings'])
                # Keep only synthetic results, not multi-GB checkpoint artifacts.
                for f in slots.iterdir():
                    if f.is_file():
                        f.unlink()
            for rep in range(a.repeats):
                for task, question in [('code', 'Write a Python LRU cache with TTL expiration and tests. Explain the design.'),
                                       ('prose', 'Explain in detail how warehouse inspections should handle discrepancies and traceability.'),
                                       ('copy', f'Copy ledger entries {count // 3} through {count // 3 + 12} verbatim. Nothing else.')]:
                    r = complete(template(prefix + question), 384)
                    row = dict(lines=count, repetition=rep, task=task, timings=r['timings'], wall_s=r['wall_s'], content=r['content'])
                    record['rows'].append(row); save()
                    print(json.dumps({k: v for k, v in row.items() if k != 'content'}), flush=True)
            if a.recall:
                indexes = sorted(set(i * (count - 1) // 11 for i in range(12)))
                r = complete(template(prefix + 'Copy exactly ledger entries ' + ', '.join(map(str, indexes))
                                      + ', in that order, verbatim. One entry per line; nothing else.'), 512)
                matches = [f'Ledger entry {j}: warehouse {j % 97}, pallet {j * 7919 % 100003}, inspected and sealed.'
                           in r['content'] for j in indexes]
                check('distributed_recall_' + str(count), all(matches), matched=sum(matches), total=len(matches),
                      response=r)
        if a.app:
            module_path = CONFIG.parents[1] / 'benchmarks/qwen38-short-agentic-ab.py'
            spec = importlib.util.spec_from_file_location('v1_short_app', module_path)
            app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app)
            app_root = a.output / 'app'
            app_root.mkdir()
            study = SimpleNamespace(out=app_root, base=url, http=http)
            options = SimpleNamespace(gpu=a.gpu, label=a.label, context=262144,
                       backend_model_id='qwen3.8-27b-uncensored',
                       model_path=mounts['/models'] + '/Swift-1.5-Qwen3.8-27B-Heretic-ARA-Merge-IQ4_XS.gguf',
                       cases=['cache', 'orders'], stages=2, phase_timeout=180)
            agent = app.ShortAgent(study, options)
            try:
                agent.run()
                for case in agent.results:
                    for phase in case['phases']:
                        grade = phase['grade']
                        check('app_' + case['case'] + '_' + str(phase['stage']),
                              phase['exit_code'] == 0 and bool(grade.get('checks'))
                              and grade.get('passed_checks') == grade.get('checks'), result=phase)
            finally:
                agent.close()
        record['completed'] = True
        record['passed'] = all(c['passed'] for c in record['checks'])
        save()
    except BaseException as exc:
        record['error'] = str(exc); save(); raise
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        else:
            subprocess.run(['docker', 'stop', '--time', '60', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if log_follower:
            log_follower.terminate()
            log_follower.wait(timeout=10)
        log.close()
        for f in slots.iterdir():
            if f.is_file():
                f.unlink()

    if not record.get('passed', False):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
