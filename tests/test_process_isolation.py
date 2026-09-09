"""Opt-in checks against the installed Codex command sandbox, without GPUs."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from marathon_app.frontends import codex_command


BINARY = os.environ.get('MARATHON_TEST_CODEX_BIN')


@unittest.skipUnless(BINARY and sys.platform.startswith('linux'),
                     'Set MARATHON_TEST_CODEX_BIN to test Linux process isolation')
class ProcessIsolationTests(unittest.TestCase):
    def test_broad_pkill_cannot_reach_launcher_or_other_commands(self):
        # Even if sandbox construction regresses, assert the namespace boundary
        # before permitting pkill. Never run the destructive command on the host.
        outer_namespace = os.readlink('/proc/self/ns/pid')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = SimpleNamespace(router_url='http://127.0.0.1:1',
                                      model=SimpleNamespace(alias='test'),
                                      catalog_file=root / 'unused.json')
            script = (
                'import os, pathlib, subprocess; '
                f'assert os.readlink("/proc/self/ns/pid") != {outer_namespace!r}; '
                f'pathlib.Path({str(root / "isolated")!r}).touch(); '
                'subprocess.run(["pkill", "-f", "python3 -"]); '
                'print("isolated")'
            )
            sentinel = subprocess.Popen(['python3', '-', 'marathon-test-sentinel'],
                                        stdin=subprocess.PIPE)
            sentinel.stdin.write(b'import time; time.sleep(60)\n')
            sentinel.stdin.close()
            try:
                with mock.patch.dict(os.environ, {'MARATHON_CODEX_BIN': BINARY}):
                    command = codex_command(runtime, ['sandbox', '--', 'python3', '-c', script])
                # The standalone sandbox does not need an inference catalog.
                catalog_index = next(i for i, arg in enumerate(command)
                                     if arg.startswith('model_catalog_json='))
                del command[catalog_index - 1:catalog_index + 1]
                result = subprocess.run(command, cwd=root, capture_output=True,
                                        text=True, timeout=20,
                                        env=dict(os.environ, CODEX_HOME=str(root)))
                # pkill can kill its own sandbox command; neither a host sentinel
                # matching the pattern nor this test runner can be affected.
                self.assertNotIn('AssertionError', result.stderr)
                self.assertTrue((root / 'isolated').exists(), result.stderr)
                self.assertIn(result.returncode, (0, -15, 143), result.stderr)
                self.assertIsNone(sentinel.poll())
            finally:
                sentinel.terminate()
                sentinel.wait(timeout=5)

    def test_real_codex_resume_after_launcher_is_killed(self):
        import http.server
        import json
        import threading
        import time
        from marathon_app import child_process

        requested = threading.Event()
        release = threading.Event()

        class Backend(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                requested.set()
                release.wait(15)
                message = {'id': 'msg_recovery', 'type': 'message', 'role': 'assistant',
                           'content': [{'type': 'output_text', 'text': 'RECOVERED'}]}
                response = {'id': 'resp_recovery', 'status': 'completed', 'output': [message],
                            'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}}
                events = [
                    {'type': 'response.created', 'response': {'id': 'resp_recovery'}},
                    {'type': 'response.output_item.done', 'output_index': 0, 'item': message},
                    {'type': 'response.completed', 'response': response},
                ]
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    for event in events:
                        self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                environment = dict(os.environ, CODEX_HOME=str(root), CODEX_SQLITE_HOME=str(root),
                                   MARATHON_LOCAL_ONLY='1')
                command = [BINARY, '-c', 'model_provider="marathon-local"',
                           '-c', 'model_providers.marathon-local={name="Test",wire_api="responses",'
                           f'base_url="http://127.0.0.1:{server.server_port}/v1"}}',
                           'exec', '--skip-git-repo-check']
                supervisor = (
                    'import os, subprocess, sys; '
                    f'subprocess.run([sys.executable, {child_process.__file__!r}, '
                    f'str(os.getpid()), *{command + ["Say READY"]!r}])'
                )
                with (root / 'frontend.log').open('w') as log:
                    parent = subprocess.Popen([sys.executable, '-c', supervisor],
                                              env=environment, cwd=root, stdout=log, stderr=log)
                    try:
                        self.assertTrue(requested.wait(10), (root / 'frontend.log').read_text())
                        rollout = next((root / 'sessions').rglob('*.jsonl'))
                        session = json.loads(rollout.read_text().splitlines()[0])['payload']['id']
                        resume = [*command, 'resume', session, 'Say RECOVERED']
                        duplicate = subprocess.run(resume, env=environment, cwd=root,
                                                   capture_output=True, text=True, timeout=10)
                        self.assertIn('active writer', duplicate.stderr + duplicate.stdout)
                        parent.kill()
                        parent.wait(timeout=5)
                        release.set()
                        deadline = time.monotonic() + 10
                        while True:
                            recovered = subprocess.run(resume, env=environment, cwd=root,
                                                       capture_output=True, text=True, timeout=15)
                            if 'active writer' not in recovered.stderr + recovered.stdout:
                                break
                            self.assertLess(time.monotonic(), deadline, 'orphan kept native writer lock')
                            time.sleep(.05)
                        self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
                        self.assertIn('RECOVERED', recovered.stdout)
                    finally:
                        release.set()
                        if parent.poll() is None:
                            parent.terminate()
                            parent.wait(timeout=5)
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
