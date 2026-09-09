"""Real subprocess regression tests for orphaned frontend writer locks."""

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from marathon_app import child_process


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux parent-death signals')
class FrontendLifetimeTests(unittest.TestCase):
    def test_launcher_death_releases_frontend_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / 'writer.lock'
            ready = Path(directory) / 'ready'
            target = (
                'import fcntl, pathlib, time; '
                f'f=open({str(lock)!r}, "w"); fcntl.flock(f, fcntl.LOCK_EX); '
                f'pathlib.Path({str(ready)!r}).touch(); time.sleep(60)'
            )
            supervisor = (
                'import os, subprocess, sys; '
                f'subprocess.run([sys.executable, {child_process.__file__!r}, '
                f'str(os.getpid()), sys.executable, "-c", {target!r}])'
            )
            parent = subprocess.Popen([sys.executable, '-c', supervisor])
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists(), 'frontend did not acquire its lock')
                with lock.open('r') as handle:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    parent.kill()  # Only the disposable supervisor, no cleanup handler.
                    parent.wait(timeout=5)
                    deadline = time.monotonic() + 5
                    while True:
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                self.fail('orphan frontend retained its writer lock')
                            time.sleep(.02)
            finally:
                if parent.poll() is None:
                    parent.send_signal(signal.SIGTERM)
                    parent.wait(timeout=5)

    def test_parent_exit_before_signal_setup_does_not_launch_frontend(self):
        result = subprocess.run([
            sys.executable, child_process.__file__, '0',
            sys.executable, '-c', 'raise SystemExit(99)',
        ], timeout=5)
        self.assertEqual(result.returncode, 143)
