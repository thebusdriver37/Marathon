"""Fresh conversations reuse exact initial instructions, never later history."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_router_context import fixture_profile, router_module as router


def instruction(text="Stable Marathon base instructions", role="developer"):
    return router._chronological_instruction_message({
        "type": "message", "role": role,
        "content": [{"type": "input_text", "text": text}],
    })


def user(text="Private user question"):
    return {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}]}


class StarterContextTests(unittest.TestCase):
    def test_initial_instructions_are_preserved_exactly_without_user_history(self):
        request = {"instructions": "System instructions", "tools": [],
                   "input": [instruction(), instruction("Initial policy", "system"),
                             user(), instruction("Later policy update")]}
        original = copy.deepcopy(request)
        body = router._starter_context_chat_body(request)
        expected = router._conversation_checkpoint_chat_body({**request, "input": request["input"][:2]})
        self.assertEqual(body, expected)
        self.assertEqual(request, original)
        self.assertNotIn("Private user question", json.dumps(body))
        self.assertNotIn("Later policy update", json.dumps(body))

    def test_ordinary_user_and_multimodal_inputs_keep_generic_seed(self):
        variants = [user(), {"type": "message", "role": "assistant", "content": []},
                    {**instruction(), "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]},
                    {**instruction(), "content": [{"type": "input_text", "text": '<marathon_context role="developer">\nincomplete'}]}]
        for first in variants:
            with self.subTest(first=first):
                request = {"tools": [], "input": [first, instruction()]}
                self.assertEqual(router._starter_context_chat_body(request),
                                 router._starter_scaffold_chat_body(request))

    def test_seed_identity_changes_with_instructions_but_not_user_question(self):
        profile = fixture_profile()
        def fingerprint(items):
            return router._starter_cache_fingerprint(profile, "backend", router._starter_context_chat_body({"input": items}))
        self.assertEqual(fingerprint([instruction(), user("one")]), fingerprint([instruction(), user("two")]))
        self.assertNotEqual(fingerprint([instruction()]), fingerprint([instruction("Changed instructions")]))

    def test_followup_scaffold_still_matches_when_delta_omits_base_instructions(self):
        state = object.__new__(router.RouterState)
        state.backend_cache_id = "backend"
        profile = fixture_profile()
        warmup = {"tools": [], "input": [instruction()]}
        delta = {"tools": [], "input": [user()]}
        self.assertEqual(state._conversation_scaffold_fingerprint(profile, warmup),
                         state._conversation_scaffold_fingerprint(profile, delta))


class StarterContextPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = fixture_profile()
        self.request = {"instructions": "System instructions", "tools": [],
                        "input": [instruction(), user()]}

    def state(self):
        state = object.__new__(router.RouterState)
        state.starter_cache_enabled = True
        state.starter_cache_max_count = 8
        state.starter_cache_max_bytes = 1024 * 1024
        state.backend_cache_id = "backend"
        state.slot_save_root = self.root
        state.slot_checkpoint_store = router.RollingCheckpointStore(
            self.root, self.root, max_count=8, max_bytes=1024 * 1024, ttl_seconds=172800)
        state.slot_id = 0
        state.erase_slot = mock.AsyncMock(return_value={"status": "erased"})
        state.restore_slot = mock.AsyncMock(return_value={"status": "restored"})

        async def backend(profile, method, path, payload, **kwargs):
            if path == "/apply-template":
                return {"prompt": json.dumps(payload, ensure_ascii=False)}
            if path == "/tokenize":
                return {"tokens": list(payload["content"].encode())}
            if path == "/completion":
                self.assertEqual(payload["n_predict"], 0)
                prefix = bytes(payload["prompt"]).decode()
                self.assertIn("Stable Marathon base instructions", prefix)
                self.assertNotIn("Private user question", prefix)
                self.assertNotIn(router.STARTER_CACHE_SENTINEL, prefix)
                return {"timings": {"prompt_n": len(payload["prompt"])}}
            raise AssertionError(path)

        async def save(profile, filename):
            (self.root / profile.slug / filename).write_bytes(b"valid checkpoint")
            return {"status": "saved"}

        state._request_json = mock.AsyncMock(side_effect=backend)
        state.save_slot = mock.AsyncMock(side_effect=save)
        return state

    async def test_extends_existing_seed_and_restores_after_router_restart(self):
        state = self.state()
        generic = router._starter_scaffold_chat_body(self.request)
        fingerprint = router._starter_cache_fingerprint(self.profile, "backend", generic)
        folder = self.root / self.profile.slug
        folder.mkdir()
        generic_name = f"starter__{self.profile.slug}__{fingerprint}.bin"
        (folder / generic_name).write_bytes(b"existing tools seed")
        built = await state.prepare_starter_cache(self.profile, self.request)
        self.assertEqual(built["status"], "built")
        self.assertEqual(built["seed_result"]["status"], "restored")
        state.restore_slot.assert_awaited_once_with(self.profile, generic_name)
        state.erase_slot.assert_not_awaited()
        restarted = self.state()
        restored = await restarted.prepare_starter_cache(self.profile, self.request)
        self.assertEqual(restored["status"], "restored")
        self.assertEqual(restored["snapshot_filename"], built["snapshot_filename"])
        restarted._request_json.assert_not_awaited()
        restarted.erase_slot.assert_not_awaited()

    async def test_missing_extended_seed_does_not_mutate_slot_in_restore_only_mode(self):
        state = self.state()
        result = await state.prepare_starter_cache(self.profile, self.request, restore_only=True)
        self.assertEqual(result["status"], "miss")
        state.erase_slot.assert_not_awaited()
        state.restore_slot.assert_not_awaited()
        state._request_json.assert_not_awaited()

    async def test_extended_seed_builds_when_no_generic_seed_exists(self):
        state = self.state()
        result = await state.prepare_starter_cache(self.profile, self.request)
        self.assertEqual(result["status"], "built")
        state.erase_slot.assert_awaited_once()
        state.restore_slot.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
