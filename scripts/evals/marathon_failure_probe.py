"""Bounded real-CLI regression probe; no replacement agent loop or tool executor.

Uses frozen RouteWeaver fixtures and an executable counting task.
All model interaction goes through the installed marathon exec command.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[2]
GPU = ROOT.parent / 'gpu-control'
SOURCE = Path('/home/deforest/AI/experiments/swift-uncensored')
FIXTURES = SOURCE / 'marathon-validation/templates/graph'
POOL = 'llama-swap-qwen3.8-uncensored-pool'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def http(base, route, key, post=False):
    request = urllib.request.Request(base + route, data=b'' if post else None,
                                    headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read()
        return None if post else json.loads(data)


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def check(command, cwd):
    try:
        p = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {'exit': p.returncode, 'output': p.stdout + p.stderr}
    except subprocess.TimeoutExpired:
        return {'exit': 124, 'output': 'Independent check timed out'}


def unload_idle_owned_worker(worker, key):
    """Do not unload a worker another Marathon session has leased or is using."""
    suffix = hashlib.sha256(worker.encode()).hexdigest()[:12]
    lock_path = Path('/run/user/1000/marathon/backend-pools') / POOL / f'{worker}-{suffix}.lock'
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        active = http('http://127.0.0.1:9292', '/running', key)['running']
        item = next((x for x in active if x['model'] == worker), None)
        if item:
            slots = http(item['proxy'], '/slots', key)
            if any(slot.get('is_processing') for slot in slots):
                return False
            http('http://127.0.0.1:9292', '/api/models/unload/' + worker, key, True)
        return True


def cleanup_snapshot_files(out):
    """Remove only this experiment's disposable backend snapshots after shutdown."""
    slots = out / 'slots'
    removed = []
    if slots.exists():
        for path in slots.rglob('*'):
            if path.is_file() and not path.is_symlink() and '.bin' in path.suffixes:
                assert path.resolve().is_relative_to(slots.resolve())
                removed.append({'path': str(path), 'bytes': path.stat().st_size})
        save(out / 'snapshot-cleanup.json', {'files': removed, 'status': 'planned'})
        for row in removed:
            Path(row['path']).unlink()
        save(out / 'snapshot-cleanup.json', {'files': removed, 'status': 'completed'})
    return sum(row['bytes'] for row in removed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=['baseline', 'swift'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--tasks', nargs='+', choices=['graph', 'count', 'patch', 'parser', 'diagnostics', 'research', 'research-debug'], default=['graph', 'count'])
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--frontend-bin', type=Path, help='Explicit candidate frontend for before/after native-patch checks')
    args = parser.parse_args()
    out = args.output.resolve() / args.variant
    out.mkdir(parents=True, exist_ok=False)
    gpu = 3 if args.variant == 'baseline' else 2
    worker = f'marathon-qwen3.8-27b-uncensored-{gpu}'
    config_path = GPU / 'llama-swap/config.yaml'
    original_config = config_path.read_bytes()
    config = yaml.safe_load(original_config)
    key = next(line.split('=', 1)[1].strip().strip('"\'') for line in
               (GPU.parent / 'qwen-inference/.env').read_text().splitlines()
               if line.startswith('HERMES_API_KEY='))
    broker_url = 'http://127.0.0.1:9292' if args.variant == 'baseline' else 'http://127.0.0.1:19293'
    conflicts = {worker, 'qwen3.8-27b-dflash'}
    if gpu == 2:
        conflicts.add('qwen3.8-27b')
    active = http('http://127.0.0.1:9292', '/running', key)['running']
    assert not any(item['model'] in conflicts for item in active), 'GPU worker already active'
    memory = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert int(memory.strip()) < 100, 'GPU occupied'
    catalog = Path('/home/deforest/.config/marathon/catalog.toml').read_text()
    start = catalog.index('[[backends]]\nid = "' + POOL + '"')
    end = catalog.index('[[backends]]', start + 1)
    section = catalog[start:end]
    section = re.sub(r'pool_models = \[.*?\]', 'pool_models = [' + json.dumps(worker) + ']', section, flags=re.S)
    section = section.replace('proxy = "http://127.0.0.1:9292"', 'proxy = ' + json.dumps(broker_url))
    section = re.sub(r'slot_save_root = .*', 'slot_save_root = ' + json.dumps(str(out / 'slots')), section)
    section = re.sub(r'cache_id = .*', 'cache_id = ' + json.dumps('failure-probe-' + args.variant), section)
    (out / 'catalog.toml').write_text(catalog[:start] + section + catalog[end:])
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MARATHON_', 'CODEX_'))}
    env.update(MARATHON_USER_CATALOG=str(out / 'catalog.toml'),
               MARATHON_STOCK_CODEX_HOME='/home/deforest/.codex',
               MARATHON_RUNS_DIR=str(out / 'runs'),
               MARATHON_PROXY_PORT=str(19304 if args.variant == 'baseline' else 19303),
               MARATHON_MAX_OUTPUT_TOKENS='32768',
               MARATHON_SLOT_SAVE_ROOT=str(out / 'cache-budget'),
               MARATHON_SLOT_SNAPSHOT_MAX_COUNT='0',
               MARATHON_SLOT_SNAPSHOTS_ENABLED='0',
               XDG_CONFIG_HOME=str(out / 'config'), XDG_STATE_HOME=str(out / 'state'),
               CODEX_LLAMA_DEBUG='1', PYTHONDONTWRITEBYTECODE='1')
    if args.frontend_bin:
        env['MARATHON_CODEX_BIN'] = str(args.frontend_bin.resolve(strict=True))
    instance = 'failure-probe-' + args.variant
    save(out / 'config/marathon/instances' / instance / 'selection.json',
         {'schema': 1, 'model': 'qwen3.8-27b-iq4-xs', 'profile': 'one-gpu-196k-uncensored',
          'frontend': 'codex', 'profile_policy': 'explicit'})
    entry = copy.deepcopy(config['models'][worker])
    broker = child = None
    container = 'llama-swap-' + worker
    if args.variant == 'swift':
        container = 'marathon-failure-probe-swift'
        command = shlex.split(entry['cmd'])
        def change(flag, value):
            command[command.index(flag) + 1] = value
        change('--name', container)
        change('--gpus', f'"device={gpu}"')
        change('--model', '/candidate/Swift-Qwen3.8-27B-Uncensored-Merge-IQ4_XS.gguf')
        change('--mmproj', '/candidate/Swift-Qwen3.8-27B-Uncensored-Merge-vision-F16.gguf')
        for i, value in enumerate(command):
            if value.endswith(':/cache'):
                command[i] = str(out / 'slots/qwen3.8-27b-iq4-xs') + ':/cache'
        (out / 'slots/qwen3.8-27b-iq4-xs').mkdir(parents=True)
        command[2:2] = ['--volume', str(SOURCE) + ':/candidate:ro']
        entry.update(cmd=shlex.join(command), cmdStop='docker stop --timeout 60 ' + container)
        entry.pop('macros', None)
        private = dict(healthCheckTimeout=900, logLevel='info', startPort=19400, globalTTL=900,
                       apiKeys=['${env.HERMES_API_KEY}'], models={worker: entry})
        (out / 'broker.yaml').write_text(yaml.safe_dump(private))
    save(out / 'protocol.json', {'variant': args.variant, 'gpu': gpu,
         'backend_entry': entry, 'production_config_sha256': hashlib.sha256(original_config).hexdigest(),
         'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'repeats': args.repeats, 'tasks': args.tasks, 'timeout_s': args.timeout,
         'frontend_bin_override': env.get('MARATHON_CODEX_BIN'),
         'temperature': 1.0, 'seed': 'not pinned; independent repetitions',
         'reasoning': 'medium', 'interface': 'installed marathon exec, no instruction override',
         'administrative_changes': 'isolated homes/catalog/logs; workspace-write sandbox; disk slot snapshots disabled',
         'candidate_identity_note': 'Swift uses baseline routing alias and profile; actual GGUF is pinned by private backend command.'})
    results = []
    try:
        if args.variant == 'swift':
            log = (out / 'broker.log').open('w')
            broker = subprocess.Popen(['/home/deforest/.local/bin/llama-swap', '-config', str(out / 'broker.yaml'),
                '-listen', '127.0.0.1:19293'], env=dict(env, HERMES_API_KEY=key), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(50):
                try:
                    http(broker_url, '/running', key)
                    break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError('Private broker startup failed')
        for repeat in range(args.repeats):
            for task in args.tasks:
                trial = out / f'{task}-{repeat + 1}'
                workspace = trial / 'workspace'
                if task == 'graph':
                    shutil.copytree(FIXTURES / 'workspace', workspace, ignore=shutil.ignore_patterns('.git', '__pycache__'))
                    prompt = (workspace / 'TASK.md').read_text() + '\nExplicit API contract: criticalPath() must independently reject cycles with an error containing "cycle detected", including when called directly without planWaves(). Non-strict parsing must skip semantically invalid records and retain valid ones.\n'
                    visible_before = check(['node', '--test'], workspace)
                    assert visible_before['exit'] != 0, 'Fixture unexpectedly already passes'
                    save(trial / 'before.json', visible_before)
                elif task == 'count':
                    workspace.mkdir(parents=True)
                    prompt = 'Count permutations of integers 1 through 8 where each value i appears in neither position i nor position i+1, wrapping position 9 to position 1. Positions are one-based. Write a runnable standard-library Python verification script named verify.py and answer.json with an integer count and a short explanation. Verify the count using execution and a second independent method. Do not install packages or commit. Keep all created files inside this repository. Give a concise final answer with the verified count.'
                elif task == 'patch':
                    workspace.mkdir(parents=True)
                    (workspace / 'settings.py').write_text('FIRST = 1\nKEEP = "unchanged"\nLAST = 3\n')
                    prompt = 'Using one apply_patch tool call with two small replace operations, change FIRST from 1 to 2 and LAST from 3 to 4 in settings.py. Preserve KEEP exactly. Verify the result. Do not commit or create other files.'
                elif task in {'research', 'research-debug'}:
                    workspace.mkdir(parents=True)
                    if task == 'research':
                        prompt = 'Consult the official Node.js documentation online at https://nodejs.org/api/assert.html and explain whether assert.throws(() => { throw new Error("hello") }, /^hello$/) passes. Verify your explanation against the documentation and a tiny local execution. Give a concise answer with the supporting documentation link. Do not install packages, edit files, or commit.'
                    else:
                        (workspace / 'example.py').write_text('import asyncio\n\nasync def main():\n    async with asyncio.TaskGroup() as group:\n        group.create_task(asyncio.sleep(0.01))\n\nasyncio.run(main())\n')
                        prompt = 'Review example.py and research Python asyncio.TaskGroup in the official online documentation. If a child task raises ValueError, do sibling tasks keep running, and what exception does the caller receive? Verify with a small local execution and cite the documentation supporting your answer. Do not edit files, install packages, or commit. Keep the final answer concise.'
                elif task == 'diagnostics':
                    workspace.mkdir(parents=True)
                    (workspace / 'arithmetic.test.cjs').write_text(
                        "const test = require('node:test');\n"
                        "const assert = require('node:assert/strict');\n"
                        "test('addition', () => assert.equal(2 + 2, 4));\n"
                        "test('invoice total', () => assert.equal(7 * 6, 43));\n"
                        "test('subtraction', () => assert.equal(9 - 3, 6));\n")
                    prompt = 'Run the existing Node test suite and diagnose its failure. Report the individual passing and failing test counts, failing test name, and exact actual and expected values. Do not edit files, fix the deliberately incorrect test, install packages, or commit. Finish once you have verified those facts.'
                else:
                    (workspace / 'src').mkdir(parents=True)
                    original = Path('/home/deforest/AI/experiments/marathon-failure-probe-20260919/baseline/graph-1/workspace')
                    shutil.copy2(original / 'src/planner.js', workspace / 'src/planner.js')
                    shutil.copy2(original / 'package.json', workspace / 'package.json')
                    prompt = 'Review and fix parseServices in src/planner.js against this explicit contract: default non-strict mode skips malformed JSON and semantically invalid records while retaining valid records. strict:true must throw on either kind of invalid record; the error must identify source, one-based line number, and the invalid field for semantic errors. A duration must be a positive integer, not a boolean; name must be a nonempty normalized string; deps an array of strings; critical absent or boolean. Preserve unrelated planner behavior. Reproduce a failure, fix it, and verify both modes using executable tests. Use Node built-ins only. Do not commit.'
                    save(trial / 'source.json', {'path': str(original), 'planner_sha256': hashlib.sha256((original / 'src/planner.js').read_bytes()).hexdigest(), 'scope': 'focused explicit-contract follow-up, not identical graph replay'})
                (trial / 'task.txt').write_text(prompt)
                subprocess.run(['git', 'init', '-q', str(workspace)], check=True)
                subprocess.run(['git', 'add', '.'], cwd=workspace, check=True)
                protected = {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in workspace.rglob('*') if p.is_file() and '.git' not in p.relative_to(workspace).parts
                             and ('test' in p.relative_to(workspace).parts or task == 'diagnostics')}
                current_env = dict(env, MARATHON_CODEX_HOME=str(trial / 'codex-home'),
                                   MARATHON_WEB_TRACE_FILE=str(trial / 'web-trace.jsonl'))
                command = ['/home/deforest/.local/bin/marathon', '--instance', instance, 'exec', '--json',
                    '--sandbox', 'workspace-write', '-c', 'approval_policy="never"',
                    '-c', 'model_reasoning_effort="medium"', '-C', str(workspace),
                    '-o', str(trial / 'answer.md'), prompt]
                save(trial / 'command.json', command)
                save(trial / 'code-provenance.json', {'router_sha256': hashlib.sha256((ROOT / 'scripts/routers/codex_local_router.py').read_bytes()).hexdigest()})
                started = time.monotonic()
                with (trial / 'events.jsonl').open('w') as events, (trial / 'stderr.log').open('w') as errors:
                    child = subprocess.Popen(command, cwd=workspace, env=current_env, stdout=events,
                                             stderr=errors, stdin=subprocess.DEVNULL, start_new_session=True)
                    while child.poll() is None:
                        if time.monotonic() - started > args.timeout:
                            stop(child)
                            break
                        if config_path.read_bytes() != original_config:
                            raise RuntimeError('Production configuration changed during probe')
                        if args.variant == 'swift':
                            active = http('http://127.0.0.1:9292', '/running', key)['running']
                            if any(item['model'] in conflicts for item in active):
                                raise RuntimeError('Conflicting workload requested candidate GPU')
                        time.sleep(1)
                elapsed = time.monotonic() - started
                result = {'task': task, 'repeat': repeat + 1, 'variant': args.variant,
                          'exit': child.returncode, 'wall_s': elapsed, 'timed_out': elapsed >= args.timeout}
                if task == 'graph':
                    result['visible'] = check(['node', '--test'], workspace)
                    result['tests_preserved'] = all((workspace / p).exists() and hashlib.sha256((workspace / p).read_bytes()).hexdigest() == digest for p, digest in protected.items())
                    # Install the held-out checks only after the agent has exited.
                    shutil.copy2(FIXTURES / 'hidden_tests/planner.hidden.test.js', workspace / 'test/planner.hidden.test.js')
                    result['hidden'] = check(['node', '--test', 'test/planner.hidden.test.js'], workspace)
                    if (workspace / 'AUDIT.md').exists():
                        result['audit'] = (workspace / 'AUDIT.md').read_text()
                elif task == 'count':
                    result['verification_script'] = check(['python3', 'verify.py'], workspace)
                    try:
                        answer = json.loads((workspace / 'answer.json').read_text())
                        result['answer'] = answer
                        result['correct_count'] = answer.get('count') == 4738
                    except (OSError, ValueError) as exc:
                        result['answer_error'] = str(exc)
                elif task == 'patch':
                    result['correct_file'] = (workspace / 'settings.py').read_text() == 'FIRST = 2\nKEEP = "unchanged"\nLAST = 4\n'
                elif task in {'research', 'research-debug'}:
                    result['workspace_diff'] = check(['git', 'diff'], workspace)
                    result['unexpected_files'] = subprocess.check_output(
                        ['git', 'ls-files', '--others', '--exclude-standard'], cwd=workspace, text=True).splitlines()
                    if (trial / 'answer.md').exists():
                        result['answer'] = (trial / 'answer.md').read_text()
                elif task == 'diagnostics':
                    result['independent_tests'] = check(['node', '--test', 'arithmetic.test.cjs'], workspace)
                    result['fixture_preserved'] = all((workspace / p).is_file() and
                        hashlib.sha256((workspace / p).read_bytes()).hexdigest() == digest
                        for p, digest in protected.items())
                    result['unexpected_files'] = subprocess.check_output(
                        ['git', 'ls-files', '--others', '--exclude-standard'], cwd=workspace, text=True).splitlines()
                    if (trial / 'answer.md').exists():
                        result['answer'] = (trial / 'answer.md').read_text()
                else:
                    (workspace / 'test').mkdir(exist_ok=True)
                    shutil.copy2(FIXTURES / 'hidden_tests/planner.hidden.test.js', workspace / 'test/planner.hidden.test.js')
                    result['hidden'] = check(['node', '--test', 'test/planner.hidden.test.js'], workspace)
                result['diff_check'] = check(['git', 'diff', '--check'], workspace)
                completed_items = []
                result['invalid_event_lines'] = 0
                for line in (trial / 'events.jsonl').read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        result['invalid_event_lines'] += 1
                        continue
                    if event.get('type') == 'item.completed':
                        completed_items.append(event['item'])
                result['command_calls'] = sum(item.get('type') == 'command_execution' for item in completed_items)
                save(trial / 'result.json', result)
                results.append(result)
                save(out / 'results.json', results)
                print(json.dumps({k: v for k, v in result.items() if k not in {'audit', 'visible', 'hidden', 'diff_check', 'verification_script', 'independent_tests', 'answer'}}), flush=True)
                # Archive real router diagnostics without introducing an alternate request path.
                logs = out / 'state/marathon/instances' / instance / 'logs'
                if logs.exists():
                    shutil.copytree(logs, trial / 'router-logs', dirs_exist_ok=True)
    finally:
        stop(child)
        # Only the named experimental runtime is stopped, never another session.
        subprocess.run(['/home/deforest/.local/bin/marathon', '--instance', instance, 'stop'], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        if args.variant == 'swift':
            try:
                http(broker_url, '/api/models/unload/' + worker, key, True)
            except Exception:
                pass
            stop(broker)
            cleanup_snapshot_files(out)
        elif results:
            unload_idle_owned_worker(worker, key)
        save(out / 'complete.json', {'completed_cases': len(results),
             'production_config_unchanged': config_path.read_bytes() == original_config})


if __name__ == '__main__':
    main()
