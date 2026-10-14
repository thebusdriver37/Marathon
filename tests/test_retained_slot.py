"""Clean worker handoffs reuse cached tokens without trusting stale processes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from marathon_app.pool import retain_slot, take_retained_slot
from marathon_app.checkpoints import conversation_key_hash
from test_router_context import fixture_profile, router_module as router


class RetainedMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "worker.lock"
        self.handle = self.path.open("w+")
        self.addCleanup(self.handle.close)
        self.lease([[100, "12345"]])

    def lease(self, identity):
        self.handle.seek(0)
        self.handle.truncate()
        json.dump({"worker_processes": identity}, self.handle)
        self.handle.flush()

    def test_metadata_is_private_and_consumed_once(self):
        retain_slot(self.handle, {"scaffold_fingerprint": "abc"})
        path = Path(str(self.path) + ".slot.json")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(take_retained_slot(self.handle)["scaffold_fingerprint"], "abc")
        self.assertFalse(path.exists())
        self.assertIsNone(take_retained_slot(self.handle))

    def test_restarted_worker_and_reused_pid_are_rejected(self):
        for identity in ([[200, "12345"]], [[100, "67890"]], []):
            with self.subTest(identity=identity):
                self.lease([[100, "12345"]])
                retain_slot(self.handle, {"scaffold_fingerprint": "abc"})
                self.lease(identity)
                self.assertIsNone(take_retained_slot(self.handle))

    def test_unknown_worker_does_not_publish_and_invalid_metadata_is_ignored(self):
        self.lease([])
        retain_slot(self.handle, {"scaffold_fingerprint": "abc"})
        path = Path(str(self.path) + ".slot.json")
        self.assertFalse(path.exists())
        for contents in ("{", "[]", "x" * 4097):
            path.write_text(contents)
            self.assertIsNone(take_retained_slot(self.handle))
            self.assertFalse(path.exists())

    def test_unsafe_handoff_removes_previous_metadata(self):
        retain_slot(self.handle, {"scaffold_fingerprint": "abc"})
        retain_slot(self.handle, None)
        self.assertIsNone(take_retained_slot(self.handle))


class RetainedRouterTests(unittest.IsolatedAsyncioTestCase):
    def state(self):
        state = object.__new__(router.RouterState)
        state.experimental_delta_only = False
        state.backend_cache_id = "backend-version"
        state.slot_id = 0
        state.live_slot_by_model = {}
        state.live_prompt_cache_key_by_model = {}
        state.retained_slot_scaffolds = {}
        state.retained_slot_key_hashes = {}
        state.lineage = {}
        state._slot_has_cached_prompt = mock.AsyncMock(return_value=True)
        state.telemetry = mock.Mock()
        profile = fixture_profile()
        state.retained_slot_hint = {
            "profile_slug": profile.slug, "backend_cache_id": "backend-version",
            "slot_id": 0, "scaffold_fingerprint": "fingerprint",
            "key_hash": conversation_key_hash("original"), "starter": False,
        }
        return state, profile

    async def test_matching_live_slot_adopts_without_a_disk_restore(self):
        state, profile = self.state()
        self.assertTrue(await state._adopt_retained_slot(profile, "fingerprint", "original"))
        self.assertEqual(state.live_prompt_cache_key_by_model[profile.slug], "original")
        self.assertTrue(state._live_slot_scaffold_matches(profile.slug, "fingerprint", "", ""))
        self.assertFalse(state._live_slot_scaffold_matches(profile.slug, "changed", "", ""))
        self.assertEqual(state.cache_handoff_profile, profile.slug)
        self.assertIsNone(state.retained_slot_hint)

    async def test_different_conversation_keeps_checkpoint_preference(self):
        state, profile = self.state()
        self.assertTrue(await state._adopt_retained_slot(profile, "fingerprint", "new"))
        self.assertNotEqual(state.live_prompt_cache_key_by_model[profile.slug], "new")
        self.assertEqual(state.retained_slot_key_hashes[profile.slug], conversation_key_hash("original"))

    async def test_incompatible_or_lost_slot_is_never_adopted(self):
        for field, value in (("profile_slug", "other"), ("backend_cache_id", "other"),
                             ("slot_id", 1), ("scaffold_fingerprint", "other"),
                             ("slot_valid", False), ("slot_valid", None), ("delta_only", True)):
            with self.subTest(field=field, value=value):
                state, profile = self.state()
                if field == "slot_valid":
                    state._slot_has_cached_prompt.return_value = value
                elif field == "delta_only":
                    state.experimental_delta_only = value
                else:
                    state.retained_slot_hint[field] = value
                self.assertFalse(await state._adopt_retained_slot(profile, "fingerprint", "original"))
                self.assertFalse(state.live_slot_by_model)
                self.assertIsNone(state.retained_slot_hint)

    async def test_advancing_or_clearing_slot_discards_retained_metadata(self):
        state, profile = self.state()
        await state._adopt_retained_slot(profile, "fingerprint", "new")
        state._set_live_slot(profile.slug, "resp_new", "new", starter=False)
        self.assertFalse(state.retained_slot_scaffolds)
        self.assertFalse(state.retained_slot_key_hashes)
        self.assertFalse(state._live_slot_scaffold_matches(profile.slug, "fingerprint", "", ""))

    async def test_release_hands_off_only_the_last_successful_profile(self):
        for safe in (True, False):
            with self.subTest(safe=safe):
                state, profile = self.state()
                await state._adopt_retained_slot(profile, "fingerprint", "original")
                state.pool_slugs = ["other-alias", profile.slug]
                state.live_slot_by_model["other-alias"] = "old-response"
                state.retained_slot_scaffolds["other-alias"] = "old-fingerprint"
                state.pool_handle = mock.Mock()
                state.pool_model = "worker"
                state._refresh_profiles = mock.Mock()
                if not safe:
                    state.cache_handoff_profile = None
                with mock.patch.object(router, "retain_slot") as retain, \
                     mock.patch.object(router, "release_pool_worker") as release:
                    state._release_pool_worker()
                hint = retain.call_args.args[1]
                if safe:
                    self.assertEqual(hint["profile_slug"], profile.slug)
                    self.assertEqual(hint["scaffold_fingerprint"], "fingerprint")
                else:
                    self.assertIsNone(hint)
                release.assert_called_once()
                self.assertIsNone(state.pool_handle)
                self.assertIsNone(state.cache_handoff_profile)

    async def test_metadata_write_failure_still_releases_the_worker(self):
        state, _ = self.state()
        state.pool_handle = mock.Mock()
        state.pool_model = "worker"
        state._refresh_profiles = mock.Mock()
        with mock.patch.object(router, "retain_slot", side_effect=OSError("disk full")), \
             mock.patch.object(router, "release_pool_worker") as release:
            state._release_pool_worker()
        release.assert_called_once()
        self.assertIsNone(state.pool_handle)

