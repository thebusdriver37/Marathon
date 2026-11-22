"""Share one flock's open file description with a worker process.

A worker retains the exact card lease if its client dies during cold load.
The socket is owner-only and never carries prompts or credentials.
"""
import array
import hashlib
import json
import os
import socket
import struct
import threading
from pathlib import Path


def serve_lease(handle, runtime_dir: Path, model: str) -> str:
    folder = runtime_dir / 'pool-handoff'
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = hashlib.sha256(str(handle.name).encode()).hexdigest()[:24] + '.sock'
    path = folder / name
    path.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    os.chmod(path, 0o600)
    listener.listen(4)
    listener.settimeout(.2)
    inode = path.stat().st_ino

    def run():
        try:
            while not handle.closed:
                try:
                    client, _ = listener.accept()
                except socket.timeout:
                    continue
                with client:
                    client.settimeout(2)
                    try:
                        _, uid, _ = struct.unpack('3i', client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        if uid != os.getuid() or json.loads(client.recv(512)).get('model') != model:
                            continue
                        descriptor = array.array('i', [handle.fileno()])
                        client.sendmsg([b'L'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptor)])
                    except (OSError, ValueError, KeyError):
                        continue
        finally:
            listener.close()
            try:
                if path.stat().st_ino == inode:
                    path.unlink()
            except FileNotFoundError:
                pass
    threading.Thread(target=run, name='pool-lease-handoff', daemon=True).start()
    return str(path)


def receive_lease(path: str, model: str) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(path)
        client.sendall(json.dumps({'model': model}).encode())
        message, ancillary, _, _ = client.recvmsg(1, socket.CMSG_SPACE(array.array('i').itemsize))
    descriptors = array.array('i')
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors.frombytes(data[:len(data) - len(data) % descriptors.itemsize])
    if message != b'L' or len(descriptors) != 1:
        for fd in descriptors:
            os.close(fd)
        raise RuntimeError('Worker did not receive its shared card lease')
    fd = descriptors[0]
    os.set_inheritable(fd, True)
    return fd
