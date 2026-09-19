"""Opt-in native regression checks for patch 027.

Set MARATHON_SANDBOX_TEST_BIN to a built codex-linux-sandbox or Codex binary.
These checks execute the real Linux sandbox, without inference or networking.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == "linux" and os.getenv("MARATHON_SANDBOX_TEST_BIN"),
                     "requires an explicitly selected Linux sandbox binary")
class SandboxDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="marathon-sandbox-diagnostics-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def run_sandbox(self, command):
        profile = {
            "type": "managed",
            "file_system": {"type": "restricted", "entries": [
                {"path": {"type": "special", "value": {"kind": "root"}}, "access": "read"},
                {"path": {"type": "path", "path": str(self.workspace)}, "access": "write"},
            ]},
            "network": "restricted",
        }
        return subprocess.run(
            ["codex-linux-sandbox", "--sandbox-policy-cwd", str(self.workspace),
             "--permission-profile", json.dumps(profile), "--", *command],
            executable=os.environ["MARATHON_SANDBOX_TEST_BIN"],
            cwd=self.workspace, capture_output=True, text=True, timeout=30,
        )

    def test_node_runner_preserves_subtests_and_assertion_details(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required for the test-runner regression")
        fixture = self.workspace / "arithmetic.test.cjs"
        fixture.write_text(
            "const test = require('node:test');\n"
            "const assert = require('node:assert/strict');\n"
            "test('addition', () => assert.equal(2 + 2, 4));\n"
            "test('invoice total', () => assert.equal(7 * 6, 43));\n"
            "test('subtraction', () => assert.equal(9 - 3, 6));\n")
        result = self.run_sandbox([node, "--test", "--test-reporter=tap", fixture.name])
        self.assertEqual(result.returncode, 1, result.stderr)
        for expected in ("# pass 2", "# fail 1", "invoice total", "actual: 42", "expected: 43"):
            self.assertIn(expected, result.stdout)

    def test_child_stdout_and_stderr_are_not_silently_lost(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required for the child-output regression")
        result = self.run_sandbox([node, "-e", """
const {spawnSync} = require('node:child_process');
const result = spawnSync(process.execPath, ['-e',
    'console.log("child stdout"); console.error("child stderr")'], {encoding: 'utf8'});
process.stdout.write(JSON.stringify({status: result.status, stdout: result.stdout, stderr: result.stderr}));
"""])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            "status": 0, "stdout": "child stdout\n", "stderr": "child stderr\n"})

    def test_socket_inspection_does_not_enable_connections_or_outside_writes(self):
        outside = self.root / "outside.txt"
        outside.write_text("unchanged")
        result = self.run_sandbox([sys.executable, "-c", """
import errno, os, pathlib, socket, sys
left, right = socket.socketpair()
assert left.getsockname() == ''
assert left.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM
try:
    left.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
except PermissionError:
    pass
else:
    raise AssertionError('unrelated socket option unexpectedly allowed')
os.write(left.fileno(), b'ok')
assert right.recv(2) == b'ok'
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError:
    pass
else:
    raise AssertionError('IP socket unexpectedly allowed')
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    client.connect(str(pathlib.Path.cwd() / 'absent.sock'))
except OSError as error:
    assert error.errno == errno.EPERM, error
else:
    raise AssertionError('connect unexpectedly allowed')
try:
    pathlib.Path(sys.argv[1]).write_text('changed')
except OSError as error:
    assert error.errno in (errno.EACCES, errno.EPERM, errno.EROFS), error
else:
    raise AssertionError('outside write unexpectedly allowed')
print('inspection works; restrictions preserved')
""", str(outside)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outside.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
