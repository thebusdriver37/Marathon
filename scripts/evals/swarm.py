#!/usr/bin/env python3
"""Opt-in native collaboration test with independently checked coding results."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
TASK = '''Use native collaboration tools to spawn exactly two helpers before doing your own implementation.
Assign helper one sole ownership of words.py: implement word_counts(text), counting whitespace-separated words after casefold(), returning a dict. Empty input returns {}.
Assign helper two sole ownership of stats.py: implement median(values) without mutating the input; average the two middle values for even lengths; raise ValueError on empty input.
You own report.py: implement summarize(text, values), returning {"counts": word_counts(text), "median": median(values)} using imports from the two helper modules.
The helpers must not spawn more agents. Do not edit each other's files or test_task.py.
Work on report.py while the helpers work, wait for both results, then run python3 -m unittest -v and report the result.
Use no network, no packages, and no git commits. Reuse your two helpers if revisions are needed.'''
TESTS = '''import unittest
from words import word_counts
from stats import median
from report import summarize

class TaskTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(word_counts("Straße STRASSE\\tCat\\ncat"), {"strasse": 2, "cat": 2})
        self.assertEqual(word_counts("  "), {})
    def test_median(self):
        values = [9, 1, 4, 2]
        self.assertEqual(median(values), 3)
        self.assertEqual(values, [9, 1, 4, 2])
        self.assertEqual(median([8, -1, 2]), 2)
        with self.assertRaises(ValueError): median([])
    def test_integration(self):
        self.assertEqual(summarize("A a B", [3, 1, 2]), {"counts": {"a": 2, "b": 1}, "median": 2})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--check-resume', action='store_true',
                        help='Restart through ordinary resume and verify reuse of both saved helpers.')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error('--run-gpu is required to lease real inference workers')
    output = args.output_dir or Path(tempfile.mkdtemp(prefix='marathon-swarm-eval-'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / 'workspace'
    workspace.mkdir(exist_ok=False)
    subprocess.run(['git', 'init', '--quiet', str(workspace)], check=True)
    tests = workspace / 'test_task.py'
    tests.write_text(TESTS)
    started = time.monotonic()
    with (output / 'transcript.log').open('w') as log:
        process = subprocess.Popen([
            str(ROOT / 'bin/marathon'), 'swarm', '--workers', str(args.workers),
            '--output-dir', str(output / 'run'), 'exec', '--skip-git-repo-check',
            '--sandbox', 'workspace-write', '--output-last-message', str(output / 'answer.txt'), TASK,
        ], cwd=workspace, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            frontend_exit = process.wait(timeout=600)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=150)
            raise
    elapsed = time.monotonic() - started
    unchanged = tests.read_text() == TESTS
    # Reinstall the original oracle before independently checking the result.
    tests.write_text(TESTS)
    checked = subprocess.run([sys.executable, '-m', 'unittest', '-v'], cwd=workspace,
                             capture_output=True, text=True, timeout=30)
    (output / 'verification.log').write_text(checked.stdout + checked.stderr)
    events_file = output / 'run/events.jsonl'
    events = [json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
    bindings = [event['data'] for event in events if event['event'] == 'swarm.bound']
    active = {}
    peak_workers = 0
    for event in events:
        data = event['data']
        if event['event'] == 'swarm.request.started':
            active[data['request']] = data['worker']
            peak_workers = max(peak_workers, len(set(active.values())))
        elif event['event'] == 'swarm.request.finished':
            active.pop(data['request'], None)
    passed = (frontend_exit == 0 and checked.returncode == 0 and unchanged
              and len(bindings) == 3 and len({b['worker'] for b in bindings}) == args.workers
              and peak_workers == args.workers)
    summary = {'passed': passed, 'workers': args.workers, 'threads': len(bindings),
               'peak_workers_with_requests': peak_workers, 'elapsed_seconds': round(elapsed, 2),
               'frontend_exit': frontend_exit, 'tests_exit': checked.returncode,
               'original_tests_preserved': unchanged, 'evidence': str(output)}
    if args.check_resume and passed:
        session_home = output / 'run/codex-home'
        metadata = []
        for path in (session_home / 'sessions').rglob('*.jsonl'):
            with path.open() as handle:
                metadata.append(json.loads(handle.readline())['payload'])
        lead = next(item['id'] for item in metadata if isinstance(item.get('source'), str))
        helpers = [item['agent_path'] for item in metadata if item.get('parent_thread_id') == lead]
        prompt = (
            f'Resume check: reuse both existing helpers {helpers!r} via followup_task. '
            'Ask each to re-read the file they implemented and report VERIFIED. '
            'Do not spawn replacement agents. Wait for both replies, then run python3 -m unittest -v. '
            'Do not modify files, run echo commands, use network, or commit.'
        )
        before_files = {path.name: path.read_bytes() for path in workspace.glob('*.py')}
        with (output / 'resume.log').open('w') as log:
            resumed = subprocess.Popen([
                str(ROOT / 'bin/marathon'), 'exec', 'resume', lead, prompt,
            ], cwd=workspace, env=dict(os.environ, MARATHON_CODEX_HOME=str(session_home)),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                resume_exit = resumed.wait(timeout=600)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(resumed.pid, signal.SIGINT)
                try:
                    resumed.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(resumed.pid, signal.SIGTERM)
                    resumed.wait(timeout=150)
                raise
        resume_events = [json.loads(line)
                         for path in (workspace / '.marathon/swarms').glob('*/events.jsonl')
                         for line in path.read_text().splitlines()]
        resumed_bindings = [event['data'] for event in resume_events if event['event'] == 'swarm.bound']
        same_threads = {b['thread'] for b in resumed_bindings} == {b['thread'] for b in bindings}
        same_files = before_files == {path.name: path.read_bytes() for path in workspace.glob('*.py')}
        resumed_workers = len({b['worker'] for b in resumed_bindings})
        passed = (resume_exit == 0 and same_threads and same_files and resumed_workers == args.workers)
        summary.update(passed=passed, resume_exit=resume_exit, same_threads_after_resume=same_threads,
                       resume_preserved_files=same_files, resumed_workers=resumed_workers)
    (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
