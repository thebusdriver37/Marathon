"""Bounded, content-free metadata for rolling llama.cpp slot checkpoints."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


# Schema 7 checkpoints track speculative-decoder snapshot sidecars.
# Schema 6 preserved chronological developer and system context, but its slot
# files only contained the target model state.
# Schema 5 hoisted those messages ahead of the conversation, invalidating KV
# state whenever Codex refreshed resume-time context.
# Schema 4 could fingerprint reasoning that Codex omitted during persistence.
# Older schemas cannot safely validate the reusable prompt prefix.
CHECKPOINT_SCHEMA = 7
CHECKPOINT_PREFIX = "conversation__"
# Optional recurrent rewind state is tracked in schema 7 like the draft sidecars.
# Older schema 7 bundles without it remain loadable, but cannot rewind after restore.
SNAPSHOT_SIDECAR_SUFFIXES = (".draft", ".spec", ".checkpoints")
MAX_CHECKPOINT_CACHE_BYTES = 32 * 1024**3
PENDING_MAX_AGE_SECONDS = 60 * 60


def conversation_key_hash(prompt_cache_key: str) -> str:
    """Return a filesystem-safe identity without persisting the raw session key."""

    return hashlib.sha256(prompt_cache_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointMetadata:
    """Compatibility and retention metadata, never conversation content."""

    schema: int
    key_hash: str
    profile_slug: str
    profile_alias: str
    backend_cache_id: str
    scaffold_fingerprint: str
    response_id_hash: str
    context_tokens: int
    conversation_item_count: int
    conversation_prefix_hash: str
    sidecar_suffixes: tuple[str, ...]
    created_at: float
    updated_at: float

    @classmethod
    def from_path(cls, path: Path) -> CheckpointMetadata | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
            return None
        try:
            conversation_item_count = max(
                0,
                int(payload["conversation_item_count"]),
            )
            conversation_prefix_hash = str(payload["conversation_prefix_hash"])
            raw_sidecar_suffixes = payload["sidecar_suffixes"]
            if (
                conversation_item_count <= 0
                or len(conversation_prefix_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in conversation_prefix_hash
                )
                or not isinstance(raw_sidecar_suffixes, list)
                or len(raw_sidecar_suffixes) > len(SNAPSHOT_SIDECAR_SUFFIXES)
            ):
                return None
            sidecar_suffixes = tuple(str(value) for value in raw_sidecar_suffixes)
            if (
                len(set(sidecar_suffixes)) != len(sidecar_suffixes)
                or any(
                    suffix not in SNAPSHOT_SIDECAR_SUFFIXES
                    for suffix in sidecar_suffixes
                )
            ):
                return None
            return cls(
                schema=CHECKPOINT_SCHEMA,
                key_hash=str(payload["key_hash"]),
                profile_slug=str(payload["profile_slug"]),
                profile_alias=str(payload["profile_alias"]),
                backend_cache_id=str(payload["backend_cache_id"]),
                scaffold_fingerprint=str(payload["scaffold_fingerprint"]),
                response_id_hash=str(payload["response_id_hash"]),
                context_tokens=max(0, int(payload["context_tokens"])),
                conversation_item_count=conversation_item_count,
                conversation_prefix_hash=conversation_prefix_hash,
                sidecar_suffixes=sidecar_suffixes,
                created_at=float(payload["created_at"]),
                updated_at=float(payload["updated_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class CheckpointRecord:
    snapshot_path: Path
    sidecar_paths: tuple[Path, ...]
    metadata_path: Path
    metadata: CheckpointMetadata | None
    size_bytes: int
    last_used_at: float


class RollingCheckpointStore:
    """Manage atomic rolling checkpoints under local and shared disk budgets."""

    def __init__(
        self,
        local_root: Path,
        budget_root: Path,
        *,
        max_count: int,
        max_bytes: int,
        ttl_seconds: int,
    ) -> None:
        self.local_root = local_root
        self.budget_root = budget_root
        self.max_count = max(0, max_count)
        self.max_bytes = min(
            MAX_CHECKPOINT_CACHE_BYTES,
            max(1, max_bytes),
        )
        self.ttl_seconds = max(1, ttl_seconds)

    @staticmethod
    def response_id_hash(response_id: str) -> str:
        return hashlib.sha256(response_id.encode("utf-8")).hexdigest()

    @staticmethod
    def snapshot_filename(key_hash: str) -> str:
        return f"{CHECKPOINT_PREFIX}{key_hash}.bin"

    @staticmethod
    def metadata_filename(key_hash: str) -> str:
        return f"{CHECKPOINT_PREFIX}{key_hash}.json"

    @staticmethod
    def pending_filename(key_hash: str, response_id: str) -> str:
        response_hash = hashlib.sha256(response_id.encode("utf-8")).hexdigest()[:16]
        return f"{CHECKPOINT_PREFIX}{key_hash}__{response_hash}.pending.bin"

    def profile_dir(self, profile_slug: str) -> Path:
        return self.local_root / profile_slug

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.budget_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.budget_root / ".conversation-checkpoints.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            os.chmod(lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _metadata_path(snapshot_path: Path) -> Path:
        return snapshot_path.with_suffix(".json")

    @staticmethod
    def _sidecar_path(snapshot_path: Path, suffix: str) -> Path:
        return Path(f"{snapshot_path}{suffix}")

    @staticmethod
    def _write_metadata(path: Path, metadata: CheckpointMetadata) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(metadata), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    @staticmethod
    def _unlink_record(record: CheckpointRecord) -> int:
        removed_bytes = 0
        for path in (record.snapshot_path, *record.sidecar_paths):
            try:
                removed_bytes += path.stat().st_size
                path.unlink()
            except OSError:
                pass
        try:
            record.metadata_path.unlink()
        except OSError:
            pass
        return removed_bytes

    @classmethod
    def _record(cls, snapshot_path: Path) -> CheckpointRecord | None:
        try:
            stat = snapshot_path.stat()
        except OSError:
            return None
        if not snapshot_path.is_file() or stat.st_size <= 0:
            return None
        metadata_path = cls._metadata_path(snapshot_path)
        metadata = CheckpointMetadata.from_path(metadata_path)
        sidecar_paths = tuple(
            cls._sidecar_path(snapshot_path, suffix)
            for suffix in SNAPSHOT_SIDECAR_SUFFIXES
            if cls._sidecar_path(snapshot_path, suffix).is_file()
        )
        if metadata is not None:
            expected_sidecars = tuple(
                cls._sidecar_path(snapshot_path, suffix)
                for suffix in metadata.sidecar_suffixes
            )
            try:
                sidecars_valid = all(
                    path.is_file() and path.stat().st_size > 0
                    for path in expected_sidecars
                )
            except OSError:
                sidecars_valid = False
            if not sidecars_valid:
                metadata = None
        last_used_at = max(
            stat.st_mtime,
            metadata.updated_at if metadata is not None else 0.0,
        )
        return CheckpointRecord(
            snapshot_path=snapshot_path,
            sidecar_paths=sidecar_paths,
            metadata_path=metadata_path,
            metadata=metadata,
            size_bytes=stat.st_size + sum(
                path.stat().st_size for path in sidecar_paths
            ),
            last_used_at=last_used_at,
        )

    def _local_records(self) -> list[CheckpointRecord]:
        records: list[CheckpointRecord] = []
        if not self.local_root.is_dir():
            return records
        for snapshot_path in self.local_root.glob(
            f"*/{CHECKPOINT_PREFIX}*.bin"
        ):
            if snapshot_path.name.endswith(".pending.bin"):
                continue
            record = self._record(snapshot_path)
            if record is not None:
                records.append(record)
        return records

    def _budget_records(self) -> list[CheckpointRecord]:
        records: list[CheckpointRecord] = []
        if not self.budget_root.is_dir():
            return records
        for snapshot_path in self.budget_root.rglob(
            f"{CHECKPOINT_PREFIX}*.bin"
        ):
            if snapshot_path.name.endswith(".pending.bin"):
                continue
            record = self._record(snapshot_path)
            if record is not None:
                records.append(record)
        return records

    def _clean_pending_locked(self, now: float) -> list[str]:
        deleted: list[str] = []
        cutoff = now - PENDING_MAX_AGE_SECONDS
        if not self.budget_root.is_dir():
            return deleted
        for path in self.budget_root.rglob(
            f"{CHECKPOINT_PREFIX}*.pending.bin"
        ):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    for member in (
                        path,
                        *(self._sidecar_path(path, suffix) for suffix in SNAPSHOT_SIDECAR_SUFFIXES),
                    ):
                        member.unlink(missing_ok=True)
                    deleted.append(str(path))
            except OSError:
                continue
        return deleted

    def _prune_locked(
        self,
        *,
        protected_path: Path | None,
        now: float,
    ) -> dict[str, object]:
        deleted: list[str] = self._clean_pending_locked(now)
        deleted_bytes = 0
        cutoff = now - self.ttl_seconds

        for record in self._budget_records():
            if record.snapshot_path == protected_path:
                continue
            if record.last_used_at < cutoff or record.metadata is None:
                deleted_bytes += self._unlink_record(record)
                deleted.append(str(record.snapshot_path))

        local_records = sorted(
            self._local_records(),
            key=lambda record: (
                record.snapshot_path == protected_path,
                record.last_used_at,
                str(record.snapshot_path),
            ),
            reverse=True,
        )
        if self.max_count > 0:
            kept = 0
            for record in local_records:
                if record.snapshot_path == protected_path:
                    kept += 1
                    continue
                if kept < self.max_count:
                    kept += 1
                    continue
                deleted_bytes += self._unlink_record(record)
                deleted.append(str(record.snapshot_path))

        budget_records = sorted(
            self._budget_records(),
            key=lambda record: (
                record.snapshot_path == protected_path,
                record.last_used_at,
                str(record.snapshot_path),
            ),
            reverse=True,
        )
        kept_bytes = 0
        for record in budget_records:
            if record.snapshot_path == protected_path:
                kept_bytes += record.size_bytes
                continue
            if kept_bytes + record.size_bytes <= self.max_bytes:
                kept_bytes += record.size_bytes
                continue
            deleted_bytes += self._unlink_record(record)
            deleted.append(str(record.snapshot_path))

        return {
            "deleted": deleted,
            "deleted_count": len(deleted),
            "deleted_bytes": deleted_bytes,
            "kept_bytes": kept_bytes,
            "max_bytes": self.max_bytes,
            "max_count": self.max_count,
            "ttl_seconds": self.ttl_seconds,
        }

    def prune(self, protected_path: Path | None = None) -> dict[str, object]:
        with self._locked():
            return self._prune_locked(
                protected_path=protected_path,
                now=time.time(),
            )

    def find(
        self,
        *,
        profile_slug: str,
        profile_alias: str,
        prompt_cache_key: str,
        backend_cache_id: str,
        scaffold_fingerprint: str,
    ) -> CheckpointRecord | None:
        key_hash = conversation_key_hash(prompt_cache_key)
        snapshot_path = self.profile_dir(profile_slug) / self.snapshot_filename(key_hash)
        record = self._record(snapshot_path)
        if record is None or record.metadata is None:
            return None
        metadata = record.metadata
        if (
            metadata.key_hash != key_hash
            or metadata.profile_slug != profile_slug
            or metadata.profile_alias != profile_alias
            or metadata.backend_cache_id != backend_cache_id
            or metadata.scaffold_fingerprint != scaffold_fingerprint
        ):
            return None
        return record

    def find_any(
        self,
        *,
        profile_slug: str,
        prompt_cache_key: str,
    ) -> CheckpointRecord | None:
        key_hash = conversation_key_hash(prompt_cache_key)
        snapshot_path = self.profile_dir(profile_slug) / self.snapshot_filename(key_hash)
        return self._record(snapshot_path)

    def mark_used(self, record: CheckpointRecord) -> None:
        now = time.time()
        with self._locked():
            for path in (
                record.snapshot_path,
                *record.sidecar_paths,
                record.metadata_path,
            ):
                try:
                    os.utime(path, (now, now))
                except OSError:
                    pass

    def discard(self, record: CheckpointRecord) -> None:
        with self._locked():
            self._unlink_record(record)

    def discard_pending(self, profile_slug: str, pending_filename: str) -> None:
        path = self.profile_dir(profile_slug) / pending_filename
        for member in (
            path,
            *(self._sidecar_path(path, suffix) for suffix in SNAPSHOT_SIDECAR_SUFFIXES),
        ):
            try:
                member.unlink()
            except OSError:
                pass

    def commit(
        self,
        *,
        profile_slug: str,
        profile_alias: str,
        prompt_cache_key: str,
        backend_cache_id: str,
        scaffold_fingerprint: str,
        response_id: str,
        context_tokens: int,
        conversation_item_count: int,
        conversation_prefix_hash: str,
        pending_filename: str,
    ) -> dict[str, object]:
        key_hash = conversation_key_hash(prompt_cache_key)
        directory = self.profile_dir(profile_slug)
        pending_path = directory / pending_filename
        final_path = directory / self.snapshot_filename(key_hash)
        metadata_path = directory / self.metadata_filename(key_hash)
        try:
            pending_paths = (
                pending_path,
                *(
                    self._sidecar_path(pending_path, suffix)
                    for suffix in SNAPSHOT_SIDECAR_SUFFIXES
                    if self._sidecar_path(pending_path, suffix).is_file()
                ),
            )
            pending_size = sum(path.stat().st_size for path in pending_paths)
        except OSError:
            return {"status": "error", "reason": "slot save produced no checkpoint"}
        if pending_size <= 0:
            self.discard_pending(profile_slug, pending_filename)
            return {"status": "error", "reason": "slot save produced an empty checkpoint"}
        if pending_size > self.max_bytes:
            self.discard_pending(profile_slug, pending_filename)
            return {
                "status": "skipped",
                "reason": "checkpoint exceeds the 32 GiB rolling-cache ceiling",
                "size_bytes": pending_size,
                "max_bytes": self.max_bytes,
            }

        now = time.time()
        existing = self._record(final_path)
        created_at = (
            existing.metadata.created_at
            if existing is not None and existing.metadata is not None
            else now
        )
        metadata = CheckpointMetadata(
            schema=CHECKPOINT_SCHEMA,
            key_hash=key_hash,
            profile_slug=profile_slug,
            profile_alias=profile_alias,
            backend_cache_id=backend_cache_id,
            scaffold_fingerprint=scaffold_fingerprint,
            response_id_hash=self.response_id_hash(response_id),
            context_tokens=max(0, context_tokens),
            conversation_item_count=max(0, conversation_item_count),
            conversation_prefix_hash=conversation_prefix_hash,
            sidecar_suffixes=tuple(
                str(path).removeprefix(str(pending_path))
                for path in pending_paths[1:]
            ),
            created_at=created_at,
            updated_at=now,
        )

        with self._locked():
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)
            for path in pending_paths:
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    # Container-backed llama.cpp workers can create snapshots as
                    # root. The mode is protected by the 0700 parent directory.
                    pass
            replaced = False
            replaced_sidecars: list[Path] = []
            try:
                for suffix in metadata.sidecar_suffixes:
                    pending_sidecar = self._sidecar_path(pending_path, suffix)
                    final_sidecar = self._sidecar_path(final_path, suffix)
                    os.replace(pending_sidecar, final_sidecar)
                    replaced_sidecars.append(final_sidecar)
                os.replace(pending_path, final_path)
                replaced = True
                for suffix in SNAPSHOT_SIDECAR_SUFFIXES:
                    if suffix not in metadata.sidecar_suffixes:
                        self._sidecar_path(final_path, suffix).unlink(missing_ok=True)
                self._write_metadata(metadata_path, metadata)
            except BaseException:
                if replaced:
                    final_path.unlink(missing_ok=True)
                for sidecar in replaced_sidecars:
                    sidecar.unlink(missing_ok=True)
                if replaced:
                    metadata_path.unlink(missing_ok=True)
                metadata_path.with_suffix(".json.tmp").unlink(missing_ok=True)
                raise
            pruned = self._prune_locked(protected_path=final_path, now=now)

        return {
            "status": "saved",
            "snapshot_filename": final_path.name,
            "size_bytes": pending_size,
            "context_tokens": metadata.context_tokens,
            "prune_result": pruned,
        }

    def delete_local(self) -> dict[str, object]:
        deleted: list[str] = []
        deleted_bytes = 0
        with self._locked():
            for record in self._local_records():
                deleted_bytes += self._unlink_record(record)
                deleted.append(str(record.snapshot_path))
            if self.local_root.is_dir():
                for pending in self.local_root.glob(
                    f"*/{CHECKPOINT_PREFIX}*.pending.bin"
                ):
                    self.discard_pending(pending.parent.name, pending.name)
                    deleted.append(str(pending))
        return {
            "deleted": deleted,
            "deleted_count": len(deleted),
            "deleted_bytes": deleted_bytes,
        }
