"""Exec a frontend that cannot outlive its Linux launcher.

Use a fresh interpreter instead of calling ctypes in preexec_fn after a
multithreaded fork. The PID remains unchanged across exec, including lock ownership.
"""

import ctypes
import os
import signal
import sys


def main():
    parent = int(sys.argv[1])
    command = sys.argv[2:]
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        # Close the race where the parent exits before prctl is installed.
        if os.getppid() != parent:
            return 143
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    raise SystemExit(main())
