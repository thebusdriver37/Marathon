"""Small, non-blocking updater for prebuilt Marathon releases."""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
import errno
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__


ROOT_DIR = Path(__file__).resolve().parents[1]
RELEASE_API = "https://api.github.com/repos/thebusdriver37/Marathon/releases/latest"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


class UpdateError(RuntimeError):
    """A safe, user-facing update failure."""


@dataclass(frozen=True)
class Release:
    version: str
    page_url: str
    assets: dict[str, str]


def _xdg_path(name: str, fallback: Path) -> Path:
    configured = os.environ.get(name)
    return Path(configured).expanduser() if configured else fallback


def _config_file() -> Path:
    return _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config") / "marathon" / "updates.json"


def _cache_file() -> Path:
    return _xdg_path("XDG_CACHE_HOME", Path.home() / ".cache") / "marathon" / "update.json"


def _data_root() -> Path:
    return _xdg_path("XDG_DATA_HOME", Path.home() / ".local" / "share") / "marathon"


def _install_bin_dir() -> Path:
    configured = os.environ.get("MARATHON_INSTALL_BIN_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".local" / "bin"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _version(value: str) -> tuple[int, int, int]:
    match = VERSION_PATTERN.fullmatch(value.strip())
    if not match:
        raise UpdateError(f"Unsupported Marathon release version: {value!r}")
    return tuple(int(part) for part in match.groups())


def _is_newer(candidate: str, current: str = __version__) -> bool:
    return _version(candidate) > _version(current)


def _checks_enabled() -> bool:
    environment = os.environ.get("MARATHON_UPDATE_CHECK", "").strip().lower()
    if environment in {"0", "false", "no", "off"}:
        return False
    return _read_json(_config_file()).get("enabled", True) is not False


def _set_checks_enabled(enabled: bool) -> None:
    _write_json(_config_file(), {"enabled": enabled})


def _api_url() -> str:
    return os.environ.get("MARATHON_UPDATE_API_URL", RELEASE_API)


def _request_bytes(url: str, limit: int, timeout: float = 5.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"Marathon/{__version__}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > limit:
            raise UpdateError("The update response is unexpectedly large")
        value = response.read(limit + 1)
    if len(value) > limit:
        raise UpdateError("The update response is unexpectedly large")
    return value


def _fetch_release() -> Release | None:
    try:
        raw = _request_bytes(_api_url(), MAX_METADATA_BYTES)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise UpdateError(f"Could not check for updates: HTTP {error.code}") from error
    except (OSError, urllib.error.URLError) as error:
        raise UpdateError(f"Could not check for updates: {error}") from error
    try:
        payload = json.loads(raw)
        version = payload["tag_name"].removeprefix("v")
        _version(version)
        page_url = str(payload["html_url"])
        assets = {
            str(asset["name"]): str(asset["browser_download_url"])
            for asset in payload.get("assets", [])
            if isinstance(asset, dict) and asset.get("name") and asset.get("browser_download_url")
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UpdateError) as error:
        raise UpdateError("The release service returned invalid Marathon metadata") from error
    return Release(version, page_url, assets)


def _cache_release(release: Release | None, *, error: str | None = None) -> None:
    existing = _read_json(_cache_file())
    cached: dict[str, Any] = {
        "checked_at": time.time(),
        "latest_version": release.version if release else None,
        "release_url": release.page_url if release else None,
        "dismissed_version": existing.get("dismissed_version"),
    }
    if error:
        cached["error"] = error
        cached["latest_version"] = existing.get("latest_version")
        cached["release_url"] = existing.get("release_url")
    _write_json(_cache_file(), cached)


def _refresh(*, quiet: bool) -> Release | None:
    lock_file = _cache_file().with_suffix(".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a", encoding="utf-8") as lock:
        try:
            flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if quiet else 0)
            fcntl.flock(lock, flags)
        except BlockingIOError:
            return None
        try:
            release = _fetch_release()
        except UpdateError as error:
            _cache_release(None, error=str(error))
            if quiet:
                return None
            raise
        _cache_release(release)
        return release


@contextlib.contextmanager
def _update_operation() -> Iterator[None]:
    """Prevent two installs or rollbacks from changing release pointers together."""

    lock_file = _data_root() / "update.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise UpdateError("Another Marathon update is already running") from error
        yield


def _cache_is_stale(cache: dict[str, Any]) -> bool:
    checked_at = cache.get("checked_at")
    return not isinstance(checked_at, (int, float)) or checked_at < time.time() - CHECK_INTERVAL_SECONDS


def _print_notice(cache: dict[str, Any]) -> None:
    latest = cache.get("latest_version")
    if not isinstance(latest, str) or cache.get("dismissed_version") == latest:
        return
    try:
        newer = _is_newer(latest)
    except UpdateError:
        return
    if newer:
        print(
            f"Update available: Marathon {__version__} -> {latest}. "
            "Run 'marathon update' or 'marathon update --skip'.",
            file=sys.stderr,
        )


def startup() -> int:
    """Print cached information and refresh it without delaying startup."""

    if not _checks_enabled():
        return 0
    cache = _read_json(_cache_file())
    _print_notice(cache)
    if _cache_is_stale(cache):
        subprocess.Popen(
            [sys.executable, "-m", "marathon_app.updates", "_refresh"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
            start_new_session=True,
            close_fds=True,
        )
    return 0


def _target_name(version: str) -> str:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise UpdateError("Prebuilt Marathon updates currently support Linux x86_64")
    return f"marathon-{version}-linux-x86_64.tar.gz"


def _asset_urls(release: Release) -> tuple[str, str]:
    archive = _target_name(release.version)
    try:
        return release.assets[archive], release.assets[f"{archive}.sha256"]
    except KeyError as error:
        raise UpdateError(f"Marathon {release.version} has no prebuilt package for this machine") from error


def _download(url: str, destination: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": f"Marathon/{__version__}"})
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=30) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise UpdateError("The Marathon update package is unexpectedly large")
                digest.update(chunk)
                output.write(chunk)
    except (OSError, urllib.error.URLError) as error:
        raise UpdateError(f"Could not download the Marathon update: {error}") from error
    return digest.hexdigest()


def _expected_checksum(url: str, archive_name: str) -> str:
    try:
        text = _request_bytes(url, 4096, timeout=10).decode("ascii")
    except (UnicodeDecodeError, OSError, urllib.error.URLError) as error:
        raise UpdateError(f"Could not download the update checksum: {error}") from error
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].lstrip("*") == archive_name and re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            return fields[0].lower()
    raise UpdateError("The Marathon update checksum is invalid")


def _extract(archive_path: Path, destination: Path, expected_root: str) -> Path:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            total = 0
            for member in members:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                    raise UpdateError("The Marathon update package contains an unsafe path")
                if not relative.parts or relative.parts[0] != expected_root:
                    raise UpdateError("The Marathon update package has an unexpected layout")
                total += member.size
                if total > MAX_EXTRACTED_BYTES:
                    raise UpdateError("The extracted Marathon update is unexpectedly large")
            for member in members:
                target = destination.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(member.mode & 0o777)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise UpdateError("The Marathon update package contains an unreadable file")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
    except (OSError, tarfile.TarError) as error:
        raise UpdateError(f"Could not unpack the Marathon update: {error}") from error
    return destination / expected_root


def _validate_release(path: Path, version: str) -> None:
    metadata = _read_json(path / ".marathon-release.json")
    required = (path / "bin" / "marathon", path / "bin" / "codex")
    if metadata.get("version") != version or any(not item.is_file() for item in required):
        raise UpdateError("The Marathon update package is incomplete")
    if any(not os.access(item, os.X_OK) for item in required):
        raise UpdateError("The Marathon update package contains a non-executable launcher")


def _link_session_home(release_root: Path, data_root: Path) -> None:
    shared_home = data_root / "codex-home"
    legacy_home = ROOT_DIR / ".marathon" / "codex-home"
    if not shared_home.exists() and not shared_home.is_symlink():
        if legacy_home.exists() and legacy_home.resolve() != shared_home.resolve():
            try:
                legacy_home.rename(shared_home)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise UpdateError(f"Could not preserve the Marathon session home: {error}") from error
                shared_home.symlink_to(legacy_home.resolve(), target_is_directory=True)
            else:
                _atomic_symlink(legacy_home, shared_home)
        else:
            shared_home.mkdir(parents=True, exist_ok=True)
    release_state = release_root / ".marathon"
    release_state.mkdir(exist_ok=True)
    release_home = release_state / "codex-home"
    if release_home.exists() or release_home.is_symlink():
        if release_home.resolve() != shared_home.resolve():
            raise UpdateError("The release contains an unexpected Codex session directory")
    else:
        release_home.symlink_to(shared_home, target_is_directory=True)


def _prepare_python(release_root: Path) -> None:
    python = release_root / ".marathon" / "venv" / "bin" / "python3"
    marker = release_root / ".marathon" / "venv" / ".marathon-requirements"
    if python.is_file() and marker.is_file():
        return
    print("Preparing the updated Marathon environment...")
    result = subprocess.run(
        [str(release_root / "bin" / "marathon"), "setup-deps"],
        cwd=release_root,
        env=dict(os.environ, MARATHON_CONFIGURE_SHELL="0", MARATHON_UPDATE_CHECK="0"),
        check=False,
    )
    if result.returncode:
        raise UpdateError("The updated Marathon environment could not be prepared")


def _atomic_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target, target_is_directory=True)
    try:
        temporary.replace(link)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _activate(release_root: Path, data_root: Path) -> None:
    current = data_root / "current"
    previous = data_root / "previous"
    launcher = _install_bin_dir() / "marathon"
    if current.exists() and not current.is_symlink():
        raise UpdateError(f"Cannot replace non-symlink install pointer: {current}")
    if launcher.exists() and not launcher.is_symlink():
        raise UpdateError(f"Cannot replace user-owned command: {launcher}")
    if current.is_symlink():
        old = current.resolve()
        releases = (data_root / "releases").resolve()
        if old.parent == releases and old != release_root.resolve():
            _atomic_symlink(previous, old)
    _atomic_symlink(current, release_root.resolve())

    _atomic_symlink(launcher, current / "bin" / "marathon")


def _install_locked() -> int:
    release = _refresh(quiet=False)
    if release is None:
        print("No published Marathon release was found.")
        return 0
    if not _is_newer(release.version):
        print(f"Marathon {__version__} is current.")
        return 0

    data_root = _data_root()
    releases = data_root / "releases"
    destination = releases / release.version
    releases.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_release(destination, release.version)
    else:
        archive_url, checksum_url = _asset_urls(release)
        archive_name = _target_name(release.version)
        print(f"Downloading Marathon {release.version}...")
        with tempfile.TemporaryDirectory(prefix=".update-", dir=data_root) as temporary_name:
            temporary = Path(temporary_name)
            archive_path = temporary / archive_name
            actual = _download(archive_url, archive_path)
            expected = _expected_checksum(checksum_url, archive_name)
            if actual != expected:
                raise UpdateError("The Marathon update checksum did not match")
            extracted = _extract(archive_path, temporary, f"marathon-{release.version}")
            _validate_release(extracted, release.version)
            extracted.replace(destination)

    _link_session_home(destination, data_root)
    _prepare_python(destination)
    _activate(destination, data_root)
    print(f"Marathon {release.version} is installed. New launches will use it.")
    return 0


def install() -> int:
    with _update_operation():
        return _install_locked()


def _rollback_locked() -> int:
    data_root = _data_root()
    current = data_root / "current"
    previous = data_root / "previous"
    if not current.is_symlink() or not previous.is_symlink():
        raise UpdateError("No previous Marathon release is available")
    old_current = current.resolve()
    old_previous = previous.resolve()
    _validate_release(old_previous, old_previous.name)
    _atomic_symlink(current, old_previous)
    _atomic_symlink(previous, old_current)
    print(f"Marathon {old_previous.name} will be used for new launches.")
    return 0


def rollback() -> int:
    with _update_operation():
        return _rollback_locked()


def check() -> int:
    release = _refresh(quiet=False)
    if release is None:
        print("No published Marathon release was found.")
    elif _is_newer(release.version):
        print(f"Marathon {release.version} is available: {release.page_url}")
    else:
        print(f"Marathon {__version__} is current.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marathon update", description="Check for or install Marathon updates")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="check now without installing")
    action.add_argument("--off", action="store_true", help="disable automatic update checks")
    action.add_argument("--on", action="store_true", help="enable automatic update checks")
    action.add_argument("--skip", action="store_true", help="hide the currently offered version")
    action.add_argument("--rollback", action="store_true", help="use the previous installed release")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["startup"]:
        return startup()
    if arguments == ["_refresh"]:
        _refresh(quiet=True)
        return 0
    args = build_parser().parse_args(arguments)
    try:
        if args.off:
            _set_checks_enabled(False)
            print("Automatic Marathon update checks are off.")
            return 0
        if args.on:
            _set_checks_enabled(True)
            print("Automatic Marathon update checks are on.")
            return 0
        if args.skip:
            cache = _read_json(_cache_file())
            latest = cache.get("latest_version")
            if not isinstance(latest, str):
                raise UpdateError("There is no cached Marathon update to skip")
            cache["dismissed_version"] = latest
            _write_json(_cache_file(), cache)
            print(f"Marathon {latest} will no longer be announced.")
            return 0
        if args.rollback:
            return rollback()
        if args.check:
            return check()
        return install()
    except UpdateError as error:
        print(f"Marathon update failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
