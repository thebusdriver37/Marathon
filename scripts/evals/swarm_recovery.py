#!/usr/bin/env python3
"""Opt-in recovery regression using a private copy of a real swarm conversation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
PROMPT = (
    'This is a recovery test only. Do not continue the app project or modify files. '
    'Ask each of the two saved helpers to reply with LOOP_RECOVERY_OK, then wait for both '
    'and report their answers. The helpers should not run any commands or modify files.'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--source-home', required=True, type=Path)
    parser.add_argument('--session', required=True, type=uuid.UUID)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error('--run-gpu is required to lease an inference worker')
    source = args.source_home.resolve()
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix='marathon-swarm-recovery-'))
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    home = output / 'codex-home'
    home.mkdir(mode=0o700)
    shutil.copytree(source / 'sessions', home / 'sessions')
    if (source / 'archived_sessions').is_dir():
        shutil.copytree(source / 'archived_sessions', home / 'archived_sessions')
    originals, lengths, expected = {}, {}, set()
    for path in home.glob('**/rollout-*.jsonl'):
        relative = path.relative_to(home)
        raw = path.read_bytes()
        originals[relative] = hashlib.sha256(raw).hexdigest()
        lengths[relative] = len(raw.splitlines())
        metadata = json.loads(raw.splitlines()[0])['payload']
        if metadata['id'] == str(args.session) or metadata.get('parent_thread_id') == str(args.session):
            expected.add(metadata['id'])
    if len(expected) != 3:
        raise ValueError('The recovery regression expects a lead with two saved helpers.')
    # SQLite backup includes committed WAL contents. Never copy live database
    # files directly or let the copied state point at original rollout paths.
    for path in source.glob('*.sqlite'):
        if path.name.startswith('logs_'):
            continue
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as original:
            with sqlite3.connect(home / path.name) as copied:
                original.backup(copied)
                tables = {row[0] for row in copied.execute("select name from sqlite_master where type='table'")}
                if 'threads' in tables:
                    copied.execute('update threads set rollout_path=replace(rollout_path,?,?)',
                                   (str(source), str(home)))
    (output / 'swarm.json').write_text(json.dumps({'version': 1, 'agents': 3, 'workers': args.workers}))
    workspace = output / 'workspace'
    workspace.mkdir()
    subprocess.run(['git', 'init', '--quiet', str(workspace)], check=True)
    with (output / 'transcript.log').open('w') as log:
        process = subprocess.Popen([
            str(ROOT / 'bin/marathon'), 'swarm', '--workers', str(args.workers), '--output-dir', str(output / 'run'),
            'exec', 'resume', str(args.session), '-c', 'sandbox_mode="read-only"', PROMPT,
        ], cwd=workspace, env=dict(os.environ, MARATHON_CODEX_HOME=str(home)),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=900)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=150)
            raise
    events = [json.loads(line) for line in (output / 'run/events.jsonl').read_text().splitlines()]
    threads = {e['data']['thread'] for e in events if e['event'] == 'swarm.bound'}
    replies, shell_calls = set(), 0
    for path in home.glob('**/rollout-*.jsonl'):
        lines = path.read_text().splitlines()
        thread = json.loads(lines[0])['payload']['id']
        for line in lines[lengths.get(path.relative_to(home), 0):]:
            event = json.loads(line)
            item = event.get('payload', {})
            if event.get('type') != 'response_item':
                continue
            if item.get('type') == 'function_call' and item.get('name') == 'exec_command':
                shell_calls += 1
            if item.get('role') == 'assistant' and any(
                'LOOP_RECOVERY_OK' in part.get('text', '') for part in item.get('content', [])
            ):
                replies.add(thread)
    unchanged = all(hashlib.sha256((source / path).read_bytes()).hexdigest() == digest
                    for path, digest in originals.items())
    passed = (code == 0 and threads == expected and replies >= expected
              and shell_calls == 0 and unchanged)
    summary = {'passed': passed, 'frontend_exit': code, 'same_threads': threads == expected,
               'threads_with_recovery_reply': sorted(replies), 'new_shell_calls': shell_calls,
               'original_history_unchanged': unchanged,
               'recovery_inventory_calls': sum(e['event'] == 'swarm.echo_recovery_inventory' for e in events),
               'loop_stops': sum(e['event'] == 'swarm.echo_loop_stopped' for e in events),
               'evidence': str(output)}
    (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
