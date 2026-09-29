#!/usr/bin/env python3
"""Stage B: held-out multi-turn validation through the real Marathon CLI.

Each arm runs the installed `marathon exec` front end with normal tool schemas.
Turns 2+ use `exec resume <session-id>` so every arm keeps its own history.
The only runtime difference between arms is one loopback shim adding logit_bias.
Graders are executable or source-backed and are frozen before any output exists.
"""
import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GPU = ROOT.parent / 'gpu-control'
MARATHON_BIN = '/home/deforest/.local/bin/marathon'
POOL = 'llama-swap-qwen3.8-uncensored-pool'
WORKERS = {1: 'marathon-qwen3.8-27b-uncensored-1', 2: 'marathon-qwen3.8-27b-uncensored-2',
           3: 'marathon-qwen3.8-27b-uncensored-3'}
PROXY_BASE, SHIM_BASE = 19600, 19800
TURN_TIMEOUT = 600


def _ports(slot):
    """One unique port pair per plan position.

    Plan positions, not (arm, gpu) pairs, because the same arm can appear on the
    same GPU in overlapping waves, and a second router cannot bind a live port.
    """
    return PROXY_BASE + slot, SHIM_BASE + slot


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def http(broker, route, key, post=False):
    request = urllib.request.Request(broker + route, data=b'' if post else None,
                                     headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read()
        return None if post else json.loads(data)


# --- Frozen held-out fixtures (written before any model output existed) ---
LEDGER_V1 = '''from ledger import aggregate

RECORDS = [{"sku": "a", "qty": 2, "price": 1.5}, {"sku": "b", "qty": 1, "price": 1.0},
           {"sku": "a", "qty": 1, "price": 0.25}]


def test_totals():
    assert aggregate(RECORDS) == [["a", 3.25], ["b", 1.0]]


def test_empty():
    assert aggregate([]) == []


def test_no_mutation():
    snapshot = [dict(r) for r in RECORDS]
    aggregate(RECORDS)
    assert RECORDS == snapshot
'''
LEDGER_V2 = '''from ledger import aggregate

RECORDS = [{"sku": "a", "qty": 2, "price": 1.5}, {"sku": "b", "qty": 1, "price": 1.0},
           {"sku": "a", "qty": 1, "price": 0.25}]


def test_group_by_qty():
    assert aggregate(RECORDS, by="qty") == [[1, 1.25], [2, 3.0]]


def test_min_total_filter():
    assert aggregate(RECORDS, min_total=2) == [["a", 3.25]]
'''
LEDGER_V3 = '''from ledger import aggregate


def test_missing_price_is_zero():
    assert aggregate([{"sku": "a", "qty": 3, "price": None},
                      {"sku": "a", "qty": 1, "price": 2}]) == [["a", 2.0]]


def test_negative_qty_refund():
    assert aggregate([{"sku": "r", "qty": -1, "price": 5},
                      {"sku": "r", "qty": 3, "price": 5}]) == [["r", 10.0]]


def test_missing_required_key_raises():
    try:
        aggregate([{"sku": "a", "price": 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("missing qty accepted")
'''
PAGER_BROKEN = '''def paginate(items, size):
    pages = []
    index = 0
    while index < len(items):
        pages.append(items[index:index + size])
        index += size - 1
    return pages


def group_pages(items, size, key):
    result = {}
    for page_number, page in enumerate(paginate(items, size), start=1):
        for item in page:
            result.setdefault(item[key], []).append(page_number)
    return result
'''
PAGER_V1 = '''from pager import paginate


def test_sizes():
    assert [len(p) for p in paginate(list(range(7)), 3)] == [3, 3, 1]


def test_no_repeats():
    flat = [v for page in paginate(list(range(7)), 3) for v in page]
    assert flat == list(range(7))
'''
PAGER_V2 = '''from pager import group_pages

ITEMS = [{"k": "a"}, {"k": "b"}, {"k": "c"}, {"k": "a"}, {"k": "d"}]


def test_page_indices():
    assert group_pages(ITEMS, 2, "k") == {"a": [1, 2], "b": [1], "c": [2], "d": [3]}
'''
LEDGER_FIXED = '''def aggregate(records, by="sku", min_total=0):
    totals = {}
    for record in records:
        if "sku" not in record or "qty" not in record:
            raise ValueError("missing required key")
        key = record[by]
        price = record.get("price")
        if price is None:
            price = 0
        totals[key] = totals.get(key, 0) + record["qty"] * price
    return [[key, round(value, 2)] for key, value in sorted(totals.items())
            if round(value, 2) >= min_total]
'''
PAGER_FIXED = PAGER_BROKEN.replace('index += size - 1', 'index += size')

SOURCES = {
    'notes/alpha.json': '{"id": "alpha", "collected": "2026-03-02", "ttl_seconds": 300, '
                        '"kind": "measurement", "status": "current"}\n',
    'notes/beta.json': '{"id": "beta", "collected": "2026-01-14", "ttl_seconds": 120, '
                       '"kind": "measurement", "status": "older"}\n',
    'notes/gamma.md': 'Draft note: ttl_seconds 480. Superseded by alpha. Do not cite.\n',
    'notes/README.md': 'The newest measurement is authoritative. Items marked superseded are not '
                       'citable. Drafts are not measurements.\n',
    'notes/delta.json': '{"id": "delta", "collected": "2026-04-05", "ttl_seconds": 240, '
                        '"kind": "measurement", "status": "current, supersedes alpha"}\n',
}
STATE_T1 = {'service_count': 3, 'api_port': 8080, 'retries': 3, 'owner': 'Mara', 'deploy': 'Thursday 14:00'}
STATE_T2 = {'service_count': 2, 'api_port': 8080, 'retries': 5, 'owner': 'Mara', 'deploy': 'Thursday 14:00'}
STATE_T4 = {'service_count': 2, 'api_port': 8080, 'retries': 5, 'owner': 'Ivo', 'deploy': 'Thursday 14:00'}
JSON_INSTRUCTION = ('Return only a JSON object, no prose and no code fence.')


def json_of(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_tests(workspace, name):
    snippet = ('import sys\nsys.path.insert(0, ".")\n'
               f'import {name[:-3]} as module\n'
               f'for label in sorted(dir(module)):\n'
               f'    if label.startswith("test_"):\n'
               f'        getattr(module, label)()\nprint("OK {name}")')
    try:
        done = subprocess.run(['python3', '-I', '-c', snippet], cwd=workspace,
                              capture_output=True, text=True, timeout=30)
        return {'exit': done.returncode, 'output': (done.stdout + done.stderr)[-800:]}
    except subprocess.TimeoutExpired:
        return {'exit': 124, 'output': 'timeout'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grade_ledger(workspace, answers, protected, expected=None):
    checks = {name: run_tests(workspace, name) for name in
              ('test_ledger.py', 'test_ledger2.py', 'test_ledger3.py') if (workspace / name).exists()}
    return {'tests': checks,
            'all_tests_pass': bool(checks) and all(row['exit'] == 0 for row in checks.values()),
            'tests_preserved': all((workspace / p).exists() and sha(workspace / p) == digest
                                   for p, digest in protected.items())}


def grade_pager(workspace, answers, protected, expected=None):
    checks = {name: run_tests(workspace, name) for name in
              ('test_pager.py', 'test_pager2.py') if (workspace / name).exists()}
    hidden = subprocess.run(
        ['python3', '-I', '-c',
         'import sys\nsys.path.insert(0, ".")\n'
         'from pager import paginate, group_pages\n'
         'assert [len(p) for p in paginate(list(range(10)), 4)] == [4, 4, 2]\n'
         'assert [v for p in paginate(list(range(10)), 4) for v in p] == list(range(10))\n'
         'assert paginate([], 3) == []\n'
         'assert group_pages([{"k": 1}, {"k": 2}, {"k": 1}], 2, "k") == {1: [1, 2], 2: [1]}\n'
         'print("HIDDEN OK")'],
        cwd=workspace, capture_output=True, text=True, timeout=30)
    return {'tests': checks, 'hidden': {'exit': hidden.returncode,
                                        'output': (hidden.stdout + hidden.stderr)[-800:]},
            'all_tests_pass': all(row['exit'] == 0 for row in checks.values()) if checks else False,
            'hidden_pass': hidden.returncode == 0,
            'tests_preserved': all((workspace / p).exists() and sha(workspace / p) == digest
                                   for p, digest in protected.items())}


def _field_matches(got, label, value):
    if label not in got:
        return False
    if isinstance(value, str):
        return value.lower() in str(got[label]).lower()
    return got[label] == value


def grade_research(workspace, answers, protected, expected=None):
    parsed = [json_of(text) for text in answers]
    checks = {}
    for index, want in enumerate(expected):
        got = parsed[index] if index < len(parsed) else None
        checks[f'turn{index + 1}'] = {
            'parsed': got,
            'ok': bool(got) and all(_field_matches(got, label, value) for label, value in want.items())}
    return {'checks': checks, 'all_tests_pass': bool(checks) and all(c['ok'] for c in checks.values())}


LEDGER_STUB = ('def aggregate(records, by="sku", min_total=0):\n    raise NotImplementedError\n')
PROSE_NOTES = ('Facts for the release note:\n'
               '- Version 2.4.1 is shipping.\n'
               '- The cache now expires an entry after 240 seconds.\n'
               '- A request that used to wait forever now stops at a 30 second timeout.\n'
               '- Rollback to 2.4.0 restores the previous timeout behavior.\n')
STATE_FACTS = ('My project facts: 3 services (api, worker, ui). The api listens on port 8080. '
               'Retries are 3. The owner is Mara. The deploy window is Thursday 14:00. ')


def grade_prose(workspace, answers, protected, expected=None):
    checks = {}
    for index, want in enumerate(expected):
        text = answers[index] if index < len(answers) else ''
        words = len(text.split())
        bullets = [line for line in text.splitlines() if line.strip().startswith('- ')]
        checks[f'turn{index + 1}'] = {
            'word_count': words,
            'words_in_range': want['min_words'] <= words <= want['max_words'],
            'bullet_count': len(bullets),
            'required_tokens': {token: token.lower() in text.lower() for token in want['tokens']},
            'no_em_dash': '\u2014' not in text}
        checks[f'turn{index + 1}']['ok'] = (
            checks[f'turn{index + 1}']['words_in_range']
            and checks[f'turn{index + 1}']['bullet_count'] == want['bullets']
            and all(checks[f'turn{index + 1}']['required_tokens'].values())
            and checks[f'turn{index + 1}']['no_em_dash'])
    return {'checks': checks, 'all_tests_pass': bool(checks) and all(c['ok'] for c in checks.values())}


def self_check_graders(out):
    """Validate every grader against a correct reference and a broken example."""
    results = {}
    root = out / 'grader-self-check'
    for name, source, want in (('correct', LEDGER_FIXED, True), ('broken', LEDGER_STUB, False)):
        workspace = root / f'ledger-{name}'
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / 'ledger.py').write_text(source)
        for label, text in (('test_ledger.py', LEDGER_V1), ('test_ledger2.py', LEDGER_V2),
                            ('test_ledger3.py', LEDGER_V3)):
            (workspace / label).write_text(text)
        protected = {label: sha(workspace / label) for label in
                     ('test_ledger.py', 'test_ledger2.py', 'test_ledger3.py')}
        graded = grade_ledger(workspace, [], protected)
        results[f'ledger-{name}'] = graded['all_tests_pass'] == want and graded['tests_preserved']
        graded['tests_preserved'] and None
    for name, source, want in (('correct', PAGER_FIXED, True), ('broken', PAGER_BROKEN, False)):
        workspace = root / f'pager-{name}'
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / 'pager.py').write_text(source)
        (workspace / 'test_pager.py').write_text(PAGER_V1)
        (workspace / 'test_pager2.py').write_text(PAGER_V2)
        protected = {label: sha(workspace / label) for label in ('test_pager.py', 'test_pager2.py')}
        graded = grade_pager(workspace, [], protected)
        results[f'pager-{name}'] = (graded['all_tests_pass'] == want
                                    and graded['hidden_pass'] == want and graded['tests_preserved'])
    research = task_definitions()[2]
    good = ['{"ttl_seconds": 300, "basis": "notes/alpha.json"}',
            '{"ttl_seconds": 240, "superseded": true}',
            '{"ttl_seconds": 240, "disagreeing_sources": 3, "uncertain": ["next review date"]}']
    bad = ['{"ttl_seconds": 120, "basis": "beta"}', '{"ttl_seconds": 240}',
           '{"ttl_seconds": 240, "disagreeing_sources": 1, "uncertain": []}']
    results['research-correct'] = grade_research(root, good, {}, research['expected'])['all_tests_pass']
    results['research-broken'] = not grade_research(root, bad, {}, research['expected'])['all_tests_pass']
    prose = task_definitions()[3]
    good_prose = ['''Release 2.4.1 notes for the storage layer.
- The cache now expires every entry after 240 seconds, so stale data leaves the store sooner and reads \
return fresher values.
- A request that previously waited forever now stops at a 30 second timeout, which keeps one slow \
downstream service from blocking the whole worker pool.
- Both changes are covered by the existing integration suite, and the suite passed on the staging \
environment during the final rehearsal run this week.
More detail is in the changelog file, and the migration notes are unchanged from the previous release.''']
    bad_prose = ['- Only one bullet, and it is short.']
    results['prose-correct'] = bool(grade_prose(root, good_prose, {}, prose['expected'])['checks']['turn1']['ok'])
    results['prose-broken'] = not grade_prose(root, bad_prose, {}, prose['expected'])['checks']['turn1']['ok']
    state = task_definitions()[4]
    good_state = [json.dumps(row) for row in (STATE_T1, STATE_T2, STATE_T2, STATE_T4)]
    bad_state = [json.dumps(STATE_T1), json.dumps(STATE_T1), json.dumps(STATE_T1), json.dumps(STATE_T1)]
    results['state-correct'] = grade_state(root, good_state, {}, state['expected'])['all_tests_pass']
    results['state-broken'] = not grade_state(root, bad_state, {}, state['expected'])['all_tests_pass']
    save(out / 'grader-self-check.json', {'results': results,
                                          'all_passed': all(bool(v) for v in results.values())})
    if not all(bool(v) for v in results.values()):
        raise RuntimeError(f'Grader self-check failed: {results}')
    return results


def _catalog_for(arm, worker, shim_port, out):
    text = Path('/home/deforest/.config/marathon/catalog.toml').read_text()
    start = text.index('[[backends]]\nid = "' + POOL + '"')
    end = text.index('[[backends]]', start + 1)
    section = text[start:end]
    section = re.sub(r'pool_models = \[.*?\]', 'pool_models = [' + json.dumps(worker) + ']',
                     section, flags=re.S)
    section = section.replace('proxy = "http://127.0.0.1:9292"', 'proxy = ' + json.dumps(f'http://127.0.0.1:{shim_port}'))
    section = re.sub(r'slot_save_root = .*', 'slot_save_root = ' + json.dumps(str(out / 'slots')), section)
    section = re.sub(r'cache_id = .*', 'cache_id = ' + json.dumps('lp2-' + arm), section)
    path = out / f'catalog-{arm}-{worker[-1]}.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text[:start] + section + text[end:])
    return path


def _start_shim(arm, worker, out, bias, port):
    bias_path = out / f'bias-{arm}-{worker}.json'
    save(bias_path, bias)
    log = (out / f'shim-{arm}-{worker}.log').open('w')
    requests = out / f'requests-{arm}-{worker}.jsonl'
    process = subprocess.Popen([str(ROOT / '.marathon/venv/bin/python'),
                                str(ROOT / 'scripts/evals/logit_bias_shim.py'), '--port', str(port),
                                '--bias-file', str(bias_path), '--log', str(requests)],
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3).read()
            break
        except Exception:
            time.sleep(0.2)
    return process, port


def _environment(arm, worker, out, catalog, proxy_port, tag=''):
    key = next(line.split('=', 1)[1].strip().strip('"\'') for line in
               (GPU.parent / 'qwen-inference/.env').read_text().splitlines()
               if line.startswith('HERMES_API_KEY='))
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('MARATHON_', 'CODEX_'))}
    environment.update(
        MARATHON_USER_CATALOG=str(catalog), MARATHON_STOCK_CODEX_HOME='/home/deforest/.codex',
        MARATHON_RUNS_DIR=str(out / 'runs'), MARATHON_PROXY_PORT=str(proxy_port),
        MARATHON_MAX_OUTPUT_TOKENS='32768', MARATHON_SLOT_SAVE_ROOT=str(out / 'slots'),
        MARATHON_SLOT_SNAPSHOT_MAX_COUNT='0', MARATHON_SLOT_SNAPSHOTS_ENABLED='0',
        MARATHON_CODEX_HOME=str(out / f'codex-home-{arm}-{worker}'),
        XDG_CONFIG_HOME=str(out / 'config' / f'{arm}-{worker}'),
        XDG_STATE_HOME=str(out / 'state' / f'{arm}-{worker}'),
        HERMES_API_KEY=key, PYTHONDONTWRITEBYTECODE='1')
    # One instance per (arm, worker, task, repeat): the pool runs tasks
    # concurrently and a shared name would be reported as "already open".
    instance = f'lp2-{arm}-{worker[-1]}' + (f'-{tag}' if tag else '')
    save(Path(environment['XDG_CONFIG_HOME']) / 'marathon/instances' / instance / 'selection.json',
         {'schema': 1, 'model': 'qwen3.8-27b-iq4-xs', 'profile': 'one-gpu-196k-uncensored',
          'frontend': 'codex', 'profile_policy': 'explicit'})
    return environment, instance


def _exec_turn(instance, environment, workspace, prompt, session_id, trial, turn):
    command = [MARATHON_BIN, '--instance', instance, 'exec', '--json', '--sandbox', 'workspace-write',
               '-c', 'approval_policy="never"', '-c', 'model_reasoning_effort="xhigh"',
               '-C', str(workspace)]
    if session_id:
        command += ['resume', session_id]
    command += [prompt]
    save(trial / f'command-{turn}.json', command)
    started = time.monotonic()
    timed_out = False
    with (trial / f'events-{turn}.jsonl').open('w') as events, \
            (trial / f'stderr-{turn}.log').open('w') as errors:
        process = subprocess.Popen(command, cwd=workspace, env=environment, stdout=events,
                                   stderr=errors, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            exit_code = process.wait(timeout=TURN_TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
            exit_code = 124
    wall = time.monotonic() - started
    return {'exit': exit_code, 'wall_s': round(wall, 3), 'timed_out': timed_out}


def _read_events(path):
    session_id, output_tokens, input_tokens, commands, answers, invalid = None, 0, 0, 0, [], 0
    for line in Path(path).read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            invalid += 1
            continue
        if event.get('type') == 'thread.started':
            session_id = event.get('thread_id')
        elif event.get('type') == 'turn.completed':
            usage = event.get('usage', {})
            output_tokens += usage.get('output_tokens', 0)
            input_tokens += usage.get('input_tokens', 0)
        elif event.get('type') == 'item.completed':
            item = event.get('item', {})
            if item.get('type') == 'command_execution':
                commands += 1
            elif item.get('type') == 'agent_message':
                answers.append(item.get('text') or '')
    return {'session_id': session_id, 'output_tokens': output_tokens, 'input_tokens': input_tokens,
            'command_calls': commands, 'answers': answers, 'invalid_event_lines': invalid}


def _wait_lock_free(worker, seconds=240):
    suffix = hashlib.sha256(worker.encode()).hexdigest()[:12]
    path = (Path('/run/user/1000/marathon/backend-pools') / POOL / f'{worker}-{suffix}.lock')
    handle = path.open('a+')
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
                return True
            except OSError:
                time.sleep(2)
        return False
    finally:
        handle.close()


def run_task(task, arm, slot, bias, gpu, out, repeat):
    worker = WORKERS[gpu]
    proxy_port, shim_port = _ports(slot)
    trial = out / f'{task["id"]}-r{repeat}-{arm}-w{worker}'
    trial.mkdir(parents=True, exist_ok=True)
    workspace = trial / 'workspace'
    workspace.mkdir(parents=True, exist_ok=True)
    for name, text in task['files'].items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(['git', 'init', '-q', str(workspace)], check=True)
    protected = {name: sha(workspace / name) for name in task['files'] if name.startswith('test_')}
    catalog = _catalog_for(arm, worker, shim_port, out)
    shim, _ = _start_shim(arm, worker, out, bias, shim_port)
    environment, instance = _environment(arm, worker, out, catalog, proxy_port,
                                         f'{task["id"]}r{repeat}')
    lease_wait_ok = _wait_lock_free(worker)
    turns, answers, session_id = [], [], None
    try:
        for index, prompt in enumerate(task['turns']):
            for name, text in task['inject'].get(index, {}).items():
                path = workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            result = _exec_turn(instance, environment, workspace, prompt, session_id, trial, index)
            summary = _read_events(trial / f'events-{index}.jsonl')
            session_id = summary['session_id'] or session_id
            answer = summary['answers'][-1] if summary['answers'] else ''
            answers.append(answer)
            (trial / f'answer-{index}.md').write_text(answer)
            turns.append({'turn': index, **result, **{k: v for k, v in summary.items()
                                                      if k != 'answers'}})
            save(trial / f'turn-{index}.json', turns[-1])
        graded = task['grade'](workspace, answers, protected, task['expected'])
    finally:
        shim.terminate()
        try:
            shim.wait(20)
        except subprocess.TimeoutExpired:
            shim.kill()
        subprocess.run([MARATHON_BIN, '--instance', instance, 'stop'], env=environment,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    row = {'task': task['id'], 'kind': task['kind'], 'arm': arm, 'repeat': repeat, 'worker': worker,
           'gpu': int(worker[-1]), 'lease_wait_ok': lease_wait_ok, 'turns': turns, 'graded': graded,
           'output_tokens': sum(t['output_tokens'] for t in turns),
           'input_tokens': sum(t['input_tokens'] for t in turns),
           'command_calls': sum(t['command_calls'] for t in turns),
           'wall_s': round(sum(t['wall_s'] for t in turns), 3),
           'pass': bool(graded.get('all_tests_pass')),
           'incomplete': any(t['timed_out'] or t['exit'] != 0 for t in turns)}
    save(trial / 'result.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms-file', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output.resolve())
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Refusing to overwrite a prior run')
    arms = json.loads(args.arms_file.read_text())
    gpus = [int(value) for value in args.gpus.split(',') if value.strip()]
    config_path = GPU / 'llama-swap/config.yaml'
    original = config_path.read_bytes()
    tasks = [t for t in task_definitions() if not args.tasks or t['id'] in args.tasks]
    self_check_graders(out)
    save(out / 'protocol.json', {
        'stage': 'B real marathon cli', 'arms': arms, 'repeats': args.repeats,
        'tasks': [t['id'] for t in tasks],
        'turns_per_task': {t['id']: len(t['turns']) for t in tasks},
        'interface': 'installed marathon exec + exec resume, workspace-write, approval never',
        'reasoning': 'xhigh via -c model_reasoning_effort', 'max_output_tokens': 32768,
        'turn_timeout_s': TURN_TIMEOUT, 'workers': {gpu: WORKERS[gpu] for gpu in gpus},
        'concurrency': len(gpus), 'gpus': gpus,
        'production_config_sha256': hashlib.sha256(original).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'injection': 'one loopback shim adds logit_bias only; baseline uses the same shim with {}',
        'grading': 'executable tests, hidden asserts, exact JSON, programmatic prose constraints'})
    arm_order = list(arms)
    plan = []
    for repeat in range(1, args.repeats + 1):
        order = arm_order if repeat % 2 else list(reversed(arm_order))
        for index, task in enumerate(tasks):
            for arm in order:
                plan.append((task, arm, len(plan), gpus[len(plan) % len(gpus)], repeat))
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = []
            for key, (task, arm, slot, gpu, repeat) in enumerate(plan):
                futures.append((key, pool.submit(run_task, task, arm, slot, arms[arm], gpu,
                                                 out, repeat)))
            for key, future in futures:
                row = future.result()
                results.append(row)
                save(out / 'results.json', results)
                print(json.dumps({k: row[k] for k in ('task', 'arm', 'repeat', 'pass', 'incomplete',
                                                      'output_tokens', 'wall_s', 'command_calls')}),
                      flush=True)
                assert config_path.read_bytes() == original, 'production config changed'
    finally:
        for arm in arms:
            for gpu in gpus:
                for task in tasks:
                    for repeat in range(1, args.repeats + 1):
                        subprocess.run([MARATHON_BIN, '--instance',
                                        f'lp2-{arm}-{gpu}-{task["id"]}r{repeat}', 'stop'],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       timeout=30)
    save(out / 'results.json', results)
    save(out / 'complete.json', {'runs': len(results),
                                 'production_config_unchanged': config_path.read_bytes() == original})


def task_definitions():
    return [
        dict(id='ledger', kind='code', grade=grade_ledger, files={
            'ledger.py': LEDGER_STUB, 'test_ledger.py': LEDGER_V1},
            inject={1: {'test_ledger2.py': LEDGER_V2}, 2: {'test_ledger3.py': LEDGER_V3}},
            turns=[
                'Implement aggregate in ledger.py so the existing test_ledger.py passes. Run '
                '`python3 -m unittest -v` yourself. Contract: group records by the `by` key, the total '
                'is round(sum(qty*price), 2), return [key, total] pairs sorted by key, do not mutate '
                'the input, empty input gives []. Do not edit any test file. Do not commit.',
                'Requirement change: a new frozen test file test_ledger2.py now applies. It requires '
                'grouping by any given `by` key and a min_total filter that drops totals below the '
                'given number. Make all test files pass, keep every existing test file unchanged, and '
                'verify with the runner. Do not commit.',
                'A further frozen file test_ledger3.py adds edge cases: a null price counts as 0, a '
                'negative qty is a refund that subtracts, and a record missing a required key raises '
                'ValueError. Make all three test files pass without editing them. Do not commit.'],
            expected=None),
        dict(id='pager', kind='code', grade=grade_pager, files={
            'pager.py': PAGER_BROKEN, 'test_pager.py': PAGER_V1},
            inject={1: {'test_pager2.py': PAGER_V2}},
            turns=[
                'The test suite in this repository fails. Run it, diagnose the root cause, and repair '
                'the source so every test passes. Keep all test files byte-for-byte unchanged. Report '
                'the failing count before and after. Do not commit.',
                'A new frozen file test_pager2.py checks group_pages page indices when the page size '
                'does not divide the item count. Make it pass too, without editing any test file. '
                'Do not commit.'],
            expected=None),
        dict(id='research', kind='research', grade=grade_research,
             files=dict(SOURCES, **{'notes/TASK.md': 'Answer from the files under notes/. '}),
             inject={1: {'notes/delta.json': SOURCES['notes/delta.json']}},
             turns=[
                 'Which ttl_seconds value is authoritative for this cache? Cite the file you relied on. '
                 + JSON_INSTRUCTION + ' Keys: ttl_seconds (number), basis (string).',
                 'A newer frozen file notes/delta.json now exists. Re-evaluate. ' + JSON_INSTRUCTION
                 + ' Keys: ttl_seconds (number), superseded (boolean).',
                 'Synthesize all notes now: give the supported ttl_seconds, how many sources disagree '
                 'with it, and the remaining uncertainty. ' + JSON_INSTRUCTION
                 + ' Keys: ttl_seconds (number), disagreeing_sources (integer), uncertain (array).'],
             expected=[{'ttl_seconds': 300, 'basis': 'alpha'}, {'ttl_seconds': 240},
                       {'ttl_seconds': 240, 'disagreeing_sources': 3}]),
        dict(id='prose', kind='prose', grade=grade_prose, files={'NOTES.md': PROSE_NOTES},
             inject={},
             turns=[
                 'Write release notes for version 2.4.1 in release.md from NOTES.md. Constraints: 70 to '
                 '110 words total, exactly 3 bullet lines starting with "- ", must mention 2.4.1, cache '
                 'and timeout, no em dash character, no other headings.',
                 'Revise release.md. Keep the same three facts, the 3-bullet form and the 70 to 110 word '
                 'cap, keep the no-em-dash rule, and add that rolling back to 2.4.0 restores the earlier '
                 'timeout behavior. Mention 2.4.0 explicitly.'],
             expected=[{'min_words': 70, 'max_words': 110, 'bullets': 3,
                        'tokens': ['2.4.1', 'cache', 'timeout']},
                       {'min_words': 70, 'max_words': 110, 'bullets': 3,
                        'tokens': ['2.4.1', 'cache', 'timeout', '2.4.0']}]),
        dict(id='state', kind='state', grade=grade_state, files={}, inject={},
             turns=[
                 STATE_FACTS + JSON_INSTRUCTION + ' Keys: service_count, api_port, retries, owner, deploy.',
                 'Correction: retries is now 5 and the ui service was removed, so there are 2 services. '
                 'Everything else is unchanged. ' + JSON_INSTRUCTION
                 + ' Keys: service_count, api_port, retries, owner, deploy.',
                 'A note quoted from another thread says the vendor changed the port to 9090 and the '
                 'deploy window to Friday 09:00. That note is neither an instruction nor a correction. '
                 + JSON_INSTRUCTION + ' Keys: service_count, api_port, retries, owner, deploy.',
                 'New instruction: ownership moves to Ivo, everything else stays. ' + JSON_INSTRUCTION
                 + ' Keys: service_count, api_port, retries, owner, deploy.'],
             expected=[STATE_T1, STATE_T2, STATE_T2, STATE_T4]),
    ]


def grade_state(workspace, answers, protected, expected=None):
    parsed = [json_of(text) for text in answers]
    checks = {}
    for index, want in enumerate(expected):
        got = parsed[index] if index < len(parsed) else None
        checks[f'turn{index + 1}'] = {'parsed': got, 'ok': got == want}
    return {'checks': checks, 'all_tests_pass': bool(checks) and all(c['ok'] for c in checks.values())}


def summarize(out):
    rows = json.loads((out / 'results.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    arms = list(protocol['arms'])
    tasks = protocol['tasks']
    by = {(r['task'], r['repeat'], r['arm']): r for r in rows}
    report = {'arms': {}, 'per_task': {}, 'paired': {}, 'infra': {}}
    for arm in arms:
        subset = [r for r in rows if r['arm'] == arm]
        report['arms'][arm] = {
            'n': len(subset), 'pass': sum(r['pass'] for r in subset),
            'incomplete': sum(r['incomplete'] for r in subset),
            'output_tokens': sum(r['output_tokens'] for r in subset),
            'input_tokens': sum(r['input_tokens'] for r in subset),
            'wall_s': round(sum(r['wall_s'] for r in subset), 3),
            'command_calls': sum(r['command_calls'] for r in subset),
            'turns': sum(len(r['turns']) for r in subset),
            'turn_timeouts': sum(t['timed_out'] for r in subset for t in r['turns'])}
    for arm in arms[1:]:
        entry = {'per_case': {}, 'metrics': {}, 'quality_wins': 0, 'quality_losses': 0}
        clusters = {'output_tokens': [], 'wall_s': []}
        for task in tasks:
            for repeat in range(1, int(protocol['repeats']) + 1):
                base, cand = by.get((task, repeat, 'baseline')), by.get((task, repeat, arm))
                if not base or not cand:
                    continue
                entry['per_case'][f'{task}-r{repeat}'] = {
                    'gpu': cand['gpu'],
                    'baseline': {'pass': base['pass'], 'tokens': base['output_tokens'],
                                 'wall_s': base['wall_s'], 'commands': base['command_calls']},
                    'candidate': {'pass': cand['pass'], 'tokens': cand['output_tokens'],
                                  'wall_s': cand['wall_s'], 'commands': cand['command_calls']}}
                entry['quality_wins'] += int(not base['pass'] and cand['pass'])
                entry['quality_losses'] += int(base['pass'] and not cand['pass'])
                clusters['output_tokens'].append((base['output_tokens'], cand['output_tokens']))
                clusters['wall_s'].append((base['wall_s'], cand['wall_s']))
        rng = random.Random(20260928)
        for metric, samples in clusters.items():
            def ratio(items):
                base = sum(v[0] for v in items)
                cand = sum(v[1] for v in items)
                return round(100 * (cand / base - 1), 3) if base else 0.0
            draws = sorted(ratio(rng.choices(samples, k=len(samples))) for _ in range(5000)) if samples else []
            entry['metrics'][metric] = {
                'change_pct': ratio(samples),
                'pair_bootstrap_95': [draws[125], draws[4874]] if draws else None,
                'median_paired_change_pct': round(sorted(
                    100 * (c / b - 1) for b, c in samples if b)[len(samples) // 2], 3) if samples else None}
        report['paired'][arm] = entry
    report['infra'] = {'lease_wait_failed': [f"{r['task']}-r{r['repeat']}-{r['arm']}"
                                             for r in rows if not r.get('lease_wait_ok')],
                       'turn_exit_nonzero': [f"{r['task']}-r{r['repeat']}-{r['arm']}:t{t['turn']}"
                                             for r in rows for t in r['turns'] if t['exit'] != 0],
                       'invalid_event_lines': sum(t['invalid_event_lines'] for r in rows for t in r['turns'])}
    save(out / 'summary.json', report)
    print(json.dumps({arm: report['paired'][arm]['metrics'] for arm in arms[1:]}, indent=2))


if __name__ == '__main__':
    main()
