"""End-to-end update checks and atomic release activation."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tarfile
import tempfile
import threading
import subprocess
import unittest
from unittest import mock

from marathon_app import updates


ROOT = Path(__file__).resolve().parents[1]


def _add_file(archive: tarfile.TarFile, name: str, content: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = mode
    archive.addfile(info, io.BytesIO(content))


def _release_archive(directory: Path, version: str) -> tuple[Path, str]:
    name = f"marathon-{version}-linux-x86_64.tar.gz"
    path = directory / name
    root = f"marathon-{version}"
    launcher = b"""#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-}" == setup-deps ]]; then
  mkdir -p "$root/.marathon/venv/bin"
  printf '#!/bin/sh\\n' >"$root/.marathon/venv/bin/python3"
  chmod +x "$root/.marathon/venv/bin/python3"
  printf 'ready\\n' >"$root/.marathon/venv/.marathon-requirements"
fi
"""
    with tarfile.open(path, "w:gz") as archive:
        _add_file(
            archive,
            f"{root}/.marathon-release.json",
            json.dumps({"schema": 1, "version": version, "target": "linux-x86_64"}).encode(),
        )
        _add_file(archive, f"{root}/bin/marathon", launcher, 0o755)
        _add_file(archive, f"{root}/bin/codex", b"#!/bin/sh\n", 0o755)
        _add_file(archive, f"{root}/bin/codex.features", b"local-runtime-security\n")
        _add_file(archive, f"{root}/marathon_app/__init__.py", f'__version__ = "{version}"\n'.encode())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class _ReleaseServer(BaseHTTPRequestHandler):
    archive: bytes
    archive_name: str
    checksum: str
    version: str

    def do_GET(self) -> None:
        base = f"http://127.0.0.1:{self.server.server_port}"
        if self.path == "/latest":
            body = json.dumps(
                {
                    "tag_name": f"v{self.version}",
                    "html_url": f"{base}/release",
                    "assets": [
                        {"name": self.archive_name, "browser_download_url": f"{base}/archive"},
                        {
                            "name": f"{self.archive_name}.sha256",
                            "browser_download_url": f"{base}/checksum",
                        },
                    ],
                }
            ).encode()
            content_type = "application/json"
        elif self.path == "/archive":
            body = self.archive
            content_type = "application/gzip"
        elif self.path == "/checksum":
            body = f"{self.checksum}  {self.archive_name}\n".encode()
            content_type = "text/plain"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


class UpdateTests(unittest.TestCase):
    def test_release_download_preserves_sessions_and_atomically_activates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="marathon-update-test-") as directory:
            root = Path(directory)
            archive, checksum = _release_archive(root, "0.5.0")
            handler = type(
                "ReleaseHandler",
                (_ReleaseServer,),
                {
                    "archive": archive.read_bytes(),
                    "archive_name": archive.name,
                    "checksum": checksum,
                    "version": "0.5.0",
                },
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            source = root / "source"
            legacy_home = source / ".marathon/codex-home"
            legacy_home.mkdir(parents=True)
            session = legacy_home / "session.jsonl"
            session.write_text("preserved\n", encoding="utf-8")
            environment = {
                "HOME": str(root / "home"),
                "XDG_CONFIG_HOME": str(root / "config"),
                "XDG_CACHE_HOME": str(root / "cache"),
                "XDG_DATA_HOME": str(root / "data"),
                "MARATHON_INSTALL_BIN_DIR": str(root / "commands"),
                "MARATHON_UPDATE_API_URL": f"http://127.0.0.1:{server.server_port}/latest",
            }
            try:
                with (
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch.object(updates, "ROOT_DIR", source),
                    contextlib.redirect_stdout(io.StringIO()) as output,
                ):
                    self.assertEqual(updates.install(), 0)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            data_root = root / "data/marathon"
            installed = data_root / "releases/0.5.0"
            self.assertEqual((data_root / "current").resolve(), installed.resolve())
            self.assertEqual((root / "commands/marathon").resolve(), (installed / "bin/marathon").resolve())
            self.assertTrue(legacy_home.is_symlink())
            self.assertEqual(session.read_text(encoding="utf-8"), "preserved\n")
            self.assertTrue((installed / ".marathon/codex-home").is_symlink())
            self.assertTrue((installed / ".marathon/venv/.marathon-requirements").is_file())
            self.assertIn("New launches will use it", output.getvalue())

    def test_startup_uses_cached_notice_without_network_wait(self) -> None:
        with tempfile.TemporaryDirectory(prefix="marathon-update-cache-") as directory:
            environment = {
                "HOME": directory,
                "XDG_CONFIG_HOME": str(Path(directory) / "config"),
                "XDG_CACHE_HOME": str(Path(directory) / "cache"),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                updates._write_json(
                    updates._cache_file(),
                    {"checked_at": updates.time.time(), "latest_version": "0.5.0"},
                )
                with (
                    mock.patch.object(updates.subprocess, "Popen") as launch,
                    contextlib.redirect_stderr(io.StringIO()) as error,
                ):
                    self.assertEqual(updates.startup(), 0)
                launch.assert_not_called()
                self.assertIn("marathon update", error.getvalue())

    def test_foreground_refresh_waits_while_background_refresh_never_waits(self) -> None:
        release = updates.Release("0.5.0", "https://example.test/release", {})
        with tempfile.TemporaryDirectory(prefix="marathon-update-lock-") as directory:
            environment = {
                "HOME": directory,
                "XDG_CACHE_HOME": str(Path(directory) / "cache"),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(updates, "_fetch_release", return_value=release),
                mock.patch.object(updates.fcntl, "flock") as flock,
            ):
                self.assertEqual(updates._refresh(quiet=False), release)
                foreground_flags = flock.call_args_list[0].args[1]
                self.assertEqual(updates._refresh(quiet=True), release)
                background_flags = flock.call_args_list[1].args[1]
            self.assertEqual(foreground_flags, updates.fcntl.LOCK_EX)
            self.assertEqual(
                background_flags,
                updates.fcntl.LOCK_EX | updates.fcntl.LOCK_NB,
            )

    def test_concurrent_update_operation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="marathon-update-operation-") as directory:
            environment = {
                "HOME": directory,
                "XDG_DATA_HOME": str(Path(directory) / "data"),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                with updates._update_operation():
                    with self.assertRaisesRegex(updates.UpdateError, "already running"):
                        with updates._update_operation():
                            self.fail("the second update operation acquired the lock")

    def test_archive_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="marathon-update-archive-") as directory:
            root = Path(directory)
            archive_path = root / "bad.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                _add_file(archive, "marathon-0.5.0/../outside", b"bad")
            with self.assertRaisesRegex(updates.UpdateError, "unsafe path"):
                updates._extract(archive_path, root / "output", "marathon-0.5.0")
            self.assertFalse((root / "outside").exists())

    def test_packaged_frontend_is_preferred_over_shared_install(self) -> None:
        from marathon_app import frontends

        with tempfile.TemporaryDirectory(prefix="marathon-packaged-frontend-") as directory:
            root = Path(directory)
            packaged = root / "bin/codex"
            packaged.parent.mkdir(parents=True)
            packaged.write_text("#!/bin/sh\n", encoding="utf-8")
            packaged.chmod(0o755)
            with (
                mock.patch.object(frontends, "ROOT_DIR", root),
                mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(root / "data")}, clear=True),
            ):
                self.assertEqual(frontends._codex_binary(), str(packaged))

    def test_release_packager_emits_the_expected_verified_layout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="marathon-package-test-") as directory:
            root = Path(directory)
            frontend = root / "frontend/codex"
            frontend.parent.mkdir(parents=True)
            frontend.write_text("#!/bin/sh\n", encoding="utf-8")
            frontend.chmod(0o755)
            for marker in ("features", "source", "prompt-hash"):
                Path(f"{frontend}.{marker}").write_text(f"{marker}\n", encoding="utf-8")
            output = root / "dist"
            result = subprocess.run(
                [str(ROOT / "scripts/package_release.sh"), str(output), str(frontend)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            archive = output / "marathon-0.4.0-linux-x86_64.tar.gz"
            checksum = output / f"{archive.name}.sha256"
            self.assertTrue(archive.is_file())
            self.assertIn(hashlib.sha256(archive.read_bytes()).hexdigest(), checksum.read_text())
            with tarfile.open(archive, "r:gz") as package:
                names = set(package.getnames())
            self.assertIn("marathon-0.4.0/.marathon-release.json", names)
            self.assertIn("marathon-0.4.0/bin/codex.features", names)
            self.assertNotIn("marathon-0.4.0/codex", names)

    def test_rollback_swaps_only_complete_releases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="marathon-rollback-test-") as directory:
            root = Path(directory)
            data_root = root / "data/marathon"
            releases = data_root / "releases"
            for version in ("0.4.0", "0.5.0"):
                release = releases / version
                (release / "bin").mkdir(parents=True)
                (release / ".marathon-release.json").write_text(
                    json.dumps({"schema": 1, "version": version}), encoding="utf-8"
                )
                for executable in (release / "bin/marathon", release / "bin/codex"):
                    executable.write_text("#!/bin/sh\n", encoding="utf-8")
                    executable.chmod(0o755)
            environment = {
                "HOME": str(root / "home"),
                "XDG_DATA_HOME": str(root / "data"),
                "MARATHON_INSTALL_BIN_DIR": str(root / "commands"),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                updates._activate(releases / "0.4.0", data_root)
                updates._activate(releases / "0.5.0", data_root)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(updates.rollback(), 0)
            self.assertEqual((data_root / "current").resolve(), (releases / "0.4.0").resolve())
            self.assertEqual((data_root / "previous").resolve(), (releases / "0.5.0").resolve())


if __name__ == "__main__":
    unittest.main()
