from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from marathon_app import checkpoints


class RollingCheckpointStoreTests(unittest.TestCase):
    def make_store(
        self,
        root: Path,
        *,
        local_name: str = "default",
        max_count: int = 2,
        max_bytes: int = 1024,
        ttl_seconds: int = 172_800,
    ) -> checkpoints.RollingCheckpointStore:
        return checkpoints.RollingCheckpointStore(
            root / local_name,
            root,
            max_count=max_count,
            max_bytes=max_bytes,
            ttl_seconds=ttl_seconds,
        )

    def commit(
        self,
        store: checkpoints.RollingCheckpointStore,
        *,
        key: str,
        response_id: str,
        content: bytes,
        profile: str = "model",
        context_tokens: int = 20_000,
    ) -> dict[str, object]:
        key_hash = checkpoints.conversation_key_hash(key)
        pending = store.pending_filename(key_hash, response_id)
        directory = store.profile_dir(profile)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / pending).write_bytes(content)
        return store.commit(
            profile_slug=profile,
            profile_alias=profile,
            prompt_cache_key=key,
            backend_cache_id="backend-v1",
            scaffold_fingerprint="scaffold-v1",
            response_id=response_id,
            context_tokens=context_tokens,
            conversation_item_count=2,
            conversation_prefix_hash="a" * 64,
            pending_filename=pending,
        )

    def test_commit_atomically_replaces_one_rolling_file_without_content_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.make_store(root)

            first = self.commit(
                store,
                key="synthetic-session",
                response_id="response-one",
                content=b"first checkpoint",
            )
            second = self.commit(
                store,
                key="synthetic-session",
                response_id="response-two",
                content=b"second checkpoint",
                context_tokens=24_096,
            )

            snapshots = list(store.local_root.glob("*/conversation__*.bin"))
            metadata_files = list(store.local_root.glob("*/conversation__*.json"))
            self.assertEqual(first["status"], "saved")
            self.assertEqual(second["status"], "saved")
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0].read_bytes(), b"second checkpoint")
            self.assertEqual(len(metadata_files), 1)
            metadata_text = metadata_files[0].read_text(encoding="utf-8")
            self.assertNotIn("synthetic-session", metadata_text)
            self.assertNotIn("response-one", metadata_text)
            self.assertNotIn("response-two", metadata_text)
            self.assertNotIn("question", metadata_text)
            metadata = json.loads(metadata_text)
            self.assertEqual(metadata["schema"], checkpoints.CHECKPOINT_SCHEMA)
            self.assertEqual(metadata["context_tokens"], 24_096)
            self.assertEqual(metadata["conversation_item_count"], 2)
            self.assertEqual(metadata["conversation_prefix_hash"], "a" * 64)

    def test_profile_slug_selects_slot_directory_when_alias_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(Path(directory))
            profile_slug = "marathon-profile"
            profile_alias = "backend-model"
            prompt_cache_key = "synthetic-session"
            key_hash = checkpoints.conversation_key_hash(prompt_cache_key)
            pending = store.pending_filename(key_hash, "response-one")
            profile_dir = store.profile_dir(profile_slug)
            profile_dir.mkdir(parents=True)
            (profile_dir / pending).write_bytes(b"checkpoint")

            result = store.commit(
                profile_slug=profile_slug,
                profile_alias=profile_alias,
                prompt_cache_key=prompt_cache_key,
                backend_cache_id="backend-v1",
                scaffold_fingerprint="scaffold-v1",
                response_id="response-one",
                context_tokens=20_000,
                conversation_item_count=2,
                conversation_prefix_hash="a" * 64,
                pending_filename=pending,
            )
            restored = store.find(
                profile_slug=profile_slug,
                profile_alias=profile_alias,
                prompt_cache_key=prompt_cache_key,
                backend_cache_id="backend-v1",
                scaffold_fingerprint="scaffold-v1",
            )

            self.assertEqual(result["status"], "saved")
            self.assertIsNotNone(restored)
            self.assertFalse(store.profile_dir(profile_alias).exists())

    def test_commit_accepts_container_owned_snapshot_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(Path(directory))
            real_chmod = checkpoints.os.chmod

            def chmod(path: object, mode: int) -> None:
                if str(path).endswith(".pending.bin"):
                    raise PermissionError("container-owned snapshot")
                real_chmod(path, mode)

            with mock.patch.object(checkpoints.os, "chmod", side_effect=chmod):
                result = self.commit(
                    store,
                    key="synthetic-session",
                    response_id="response-one",
                    content=b"checkpoint",
                )

            self.assertEqual(result["status"], "saved")

    def test_speculative_sidecars_are_managed_as_one_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.make_store(root)
            key = "speculative-session"
            key_hash = checkpoints.conversation_key_hash(key)
            pending_name = store.pending_filename(key_hash, "response-one")
            profile_dir = store.profile_dir("model")
            profile_dir.mkdir(parents=True)
            pending = profile_dir / pending_name
            pending.write_bytes(b"target-state")
            Path(f"{pending}.draft").write_bytes(b"draft-state")
            Path(f"{pending}.spec").write_bytes(b"spec-state")

            result = store.commit(
                profile_slug="model",
                profile_alias="model",
                prompt_cache_key=key,
                backend_cache_id="backend-v1",
                scaffold_fingerprint="scaffold-v1",
                response_id="response-one",
                context_tokens=20_000,
                conversation_item_count=2,
                conversation_prefix_hash="a" * 64,
                pending_filename=pending_name,
            )
            record = store.find(
                profile_slug="model",
                profile_alias="model",
                prompt_cache_key=key,
                backend_cache_id="backend-v1",
                scaffold_fingerprint="scaffold-v1",
            )

            self.assertEqual(result["status"], "saved")
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.size_bytes, 33)
            self.assertEqual(
                {path.suffix for path in record.sidecar_paths},
                {".draft", ".spec"},
            )
            self.assertEqual(
                record.metadata.sidecar_suffixes if record.metadata else (),
                (".draft", ".spec"),
            )

            store.discard(record)
            self.assertFalse(record.snapshot_path.exists())
            self.assertTrue(all(not path.exists() for path in record.sidecar_paths))

    def test_missing_recorded_sidecar_invalidates_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.make_store(root)
            key = "speculative-session"
            key_hash = checkpoints.conversation_key_hash(key)
            pending_name = store.pending_filename(key_hash, "response-one")
            profile_dir = store.profile_dir("model")
            profile_dir.mkdir(parents=True)
            pending = profile_dir / pending_name
            pending.write_bytes(b"target-state")
            Path(f"{pending}.draft").write_bytes(b"draft-state")
            self.commit(
                store,
                key="unrelated-session",
                response_id="unrelated-response",
                content=b"unrelated",
            )
            store.commit(
                profile_slug="model",
                profile_alias="model",
                prompt_cache_key=key,
                backend_cache_id="backend-v1",
                scaffold_fingerprint="scaffold-v1",
                response_id="response-one",
                context_tokens=20_000,
                conversation_item_count=2,
                conversation_prefix_hash="a" * 64,
                pending_filename=pending_name,
            )
            final = profile_dir / store.snapshot_filename(key_hash)
            Path(f"{final}.draft").unlink()

            self.assertIsNone(
                store.find(
                    profile_slug="model",
                    profile_alias="model",
                    prompt_cache_key=key,
                    backend_cache_id="backend-v1",
                    scaffold_fingerprint="scaffold-v1",
                )
            )

    def test_older_checkpoint_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "conversation.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema": 4,
                        "key_hash": "a" * 64,
                        "profile_slug": "model",
                        "profile_alias": "model",
                        "backend_cache_id": "backend-v1",
                        "scaffold_fingerprint": "b" * 64,
                        "response_id_hash": "c" * 64,
                        "context_tokens": 100_000,
                        "created_at": 1.0,
                        "updated_at": 2.0,
                    }
                ),
                encoding="utf-8",
            )

            self.assertIsNone(checkpoints.CheckpointMetadata.from_path(metadata_path))

    def test_retention_keeps_only_two_recent_conversations_per_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(Path(directory), max_count=2)

            self.commit(store, key="session-one", response_id="one", content=b"1")
            self.commit(store, key="session-two", response_id="two", content=b"2")
            self.commit(store, key="session-three", response_id="three", content=b"3")

            snapshots = list(store.local_root.glob("*/conversation__*.bin"))
            self.assertEqual(len(snapshots), 2)
            names = {path.name for path in snapshots}
            self.assertNotIn(
                store.snapshot_filename(checkpoints.conversation_key_hash("session-one")),
                names,
            )

    def test_expired_inactive_checkpoint_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(Path(directory), ttl_seconds=2)
            self.commit(store, key="old-session", response_id="one", content=b"old")
            future = time.time() + 10

            with mock.patch.object(checkpoints.time, "time", return_value=future):
                result = store.prune()

            self.assertEqual(result["deleted_count"], 1)
            self.assertEqual(list(store.local_root.glob("*/conversation__*.bin")), [])

    def test_shared_budget_prunes_across_named_instance_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = self.make_store(root, local_name="default", max_bytes=10)
            named = self.make_store(root, local_name="instances/gpu23", max_bytes=10)

            self.commit(default, key="default-session", response_id="one", content=b"123456")
            self.commit(named, key="named-session", response_id="two", content=b"abcdef")

            snapshots = list(root.rglob("conversation__*.bin"))
            self.assertEqual(len(snapshots), 1)
            self.assertLessEqual(sum(path.stat().st_size for path in snapshots), 10)

    def test_oversized_new_checkpoint_preserves_previous_good_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(Path(directory), max_bytes=8)
            self.commit(
                store,
                key="synthetic-session",
                response_id="one",
                content=b"good",
            )
            result = self.commit(
                store,
                key="synthetic-session",
                response_id="two",
                content=b"too-large",
            )

            snapshot = next(store.local_root.glob("*/conversation__*.bin"))
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(snapshot.read_bytes(), b"good")

    def test_configured_budget_cannot_exceed_32_gib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(
                Path(directory),
                max_bytes=checkpoints.MAX_CHECKPOINT_CACHE_BYTES * 2,
            )

            self.assertEqual(
                store.max_bytes,
                checkpoints.MAX_CHECKPOINT_CACHE_BYTES,
            )


if __name__ == "__main__":
    unittest.main()
