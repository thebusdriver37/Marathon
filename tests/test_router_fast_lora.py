"""Fast mode: a llama.cpp LoRA enabled only at reasoning effort "none"."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts" / "routers"))

import codex_local_router as router_module
from marathon_app import catalog
from marathon_app.checkpoints import FAST_LORA_SIDECAR_SUFFIX, SNAPSHOT_SIDECAR_SUFFIXES


def fast_profile(lora_id: int | None = 0) -> router_module.ModelProfile:
    return router_module.ModelProfile(
        slug="dynamic-model",
        alias="dynamic-model",
        display_name="Dynamic model",
        description="Test model",
        model_paths=("/tmp/model.gguf",),
        target="http://127.0.0.1:18000",
        context_window=253_952,
        auto_compact_token_limit=222_208,
        truncation_limit=214_016,
        default_reasoning_level="xhigh",
        supported_reasoning_levels=(("none", "Fast"), ("medium", "Mid"), ("xhigh", "Deep")),
        fast_lora_id=lora_id,
    )


def slots(scale: float | None) -> list[dict]:
    lora = [] if scale is None else [{"id": 0, "scale": scale}]
    return [{"id": 0, "params": {"lora": lora}}]


class FastLoraRequestTests(unittest.TestCase):
    def test_effort_none_enables_adapter_and_other_efforts_disable_it(self) -> None:
        profile = fast_profile()
        for effort, scale in (("none", 1.0), ("medium", 0.0), ("xhigh", 0.0)):
            normalized = router_module.normalize_responses_request(
                {"input": [], "reasoning": {"effort": effort}}, profile
            )
            self.assertEqual(normalized["lora"], [{"id": 0, "scale": scale}], effort)
        default = router_module.normalize_responses_request({"input": []}, profile)
        self.assertEqual(default["lora"], [{"id": 0, "scale": 0.0}])

    def test_fast_service_tier_is_fast_mode_regardless_of_effort(self) -> None:
        profile = fast_profile()
        self.assertEqual(router_module._fast_service_tiers(profile)[0]["id"], "priority")
        for tier in ("priority", "fast"):
            normalized = router_module.normalize_responses_request(
                {"input": [], "reasoning": {"effort": "xhigh"}, "service_tier": tier}, profile
            )
            self.assertEqual(normalized["lora"], [{"id": 0, "scale": 1.0}])
            self.assertFalse(normalized["chat_template_kwargs"]["enable_thinking"])
            self.assertNotIn("service_tier", normalized)
        normalized = router_module.normalize_responses_request(
            {"input": [], "reasoning": {"effort": "xhigh"}, "service_tier": "default"}, profile
        )
        self.assertEqual(normalized["lora"], [{"id": 0, "scale": 0.0}])
        self.assertEqual(router_module._fast_service_tiers(fast_profile(None)), [])
        no_none = replace(profile, supported_reasoning_levels=(("xhigh", "Deep"),))
        self.assertEqual(router_module._fast_service_tiers(no_none), [])

    def test_profiles_without_fast_mode_send_no_adapter_field(self) -> None:
        normalized = router_module.normalize_responses_request(
            {"input": [], "reasoning": {"effort": "none"}}, fast_profile(None)
        )
        self.assertNotIn("lora", normalized)

    def test_render_body_carries_adapter_so_snapshot_fingerprints_split_by_mode(self) -> None:
        profile = fast_profile()
        bodies = []
        for effort in ("none", "xhigh"):
            request = router_module.normalize_responses_request(
                {"input": [], "instructions": "sys", "reasoning": {"effort": effort}}, profile
            )
            bodies.append(router_module._starter_scaffold_chat_body(request))
        self.assertEqual(bodies[0]["lora"], [{"id": 0, "scale": 1.0}])
        self.assertNotEqual(
            router_module._starter_cache_fingerprint(profile, "cache", bodies[0]),
            router_module._starter_cache_fingerprint(profile, "cache", bodies[1]),
        )

    def test_env_and_catalog_plumbing(self) -> None:
        with mock.patch.dict("os.environ", {"MARATHON_MODEL_PATH": "/tmp/m.gguf",
                                            "MARATHON_MODEL_FAST_LORA_ID": "0"}):
            self.assertEqual(router_module._custom_model_profile(ROOT_DIR).fast_lora_id, 0)
        with mock.patch.dict("os.environ", {"MARATHON_MODEL_PATH": "/tmp/m.gguf"}):
            self.assertIsNone(router_module._custom_model_profile(ROOT_DIR).fast_lora_id)
        self.assertEqual(catalog._fast_mode_lora_id({"id": "b", "fast_mode_lora_id": 0}), 0)
        self.assertIsNone(catalog._fast_mode_lora_id({"id": "b"}))
        for bad in (-1, True, "0"):
            with self.assertRaises(ValueError):
                catalog._fast_mode_lora_id({"id": "b", "fast_mode_lora_id": bad})
        self.assertIn(FAST_LORA_SIDECAR_SUFFIX, SNAPSHOT_SIDECAR_SUFFIXES)


class FastLoraSlotTests(unittest.TestCase):
    def state(self, root: Path, profile: router_module.ModelProfile):
        state = object.__new__(router_module.RouterState)
        state.slot_id = 0
        state.slot_save_root = root
        state._slot_action = mock.AsyncMock(return_value={"status": "ok"})
        return state

    def test_save_records_the_adapter_state_that_built_the_snapshot(self) -> None:
        profile = fast_profile()
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state(Path(tmp), profile)
            (Path(tmp) / profile.slug).mkdir()
            state._request_json = mock.AsyncMock(return_value=slots(1.0))
            asyncio.run(state.save_slot(profile, "snap.bin"))
            sidecar = Path(tmp) / profile.slug / ("snap.bin" + FAST_LORA_SIDECAR_SUFFIX)
            self.assertEqual(json.loads(sidecar.read_text()), {"id": 0, "scale": 1.0})
            state._request_json = mock.AsyncMock(return_value=slots(None))
            asyncio.run(state.save_slot(profile, "snap.bin"))
            self.assertEqual(json.loads(sidecar.read_text())["scale"], 0.0)

    def test_save_without_known_adapter_state_drops_the_snapshot(self) -> None:
        profile = fast_profile()
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state(Path(tmp), profile)
            snapshot = Path(tmp) / profile.slug / "snap.bin"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"kv")
            state._request_json = mock.AsyncMock(return_value=[])
            with self.assertRaises(RuntimeError):
                asyncio.run(state.save_slot(profile, "snap.bin"))
            self.assertFalse(snapshot.exists())

    def test_restore_primes_slot_to_the_snapshot_mode_only_when_it_differs(self) -> None:
        profile = fast_profile()
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state(Path(tmp), profile)
            directory = Path(tmp) / profile.slug
            directory.mkdir()
            (directory / ("fast.bin" + FAST_LORA_SIDECAR_SUFFIX)).write_text('{"id": 0, "scale": 1.0}')

            state._request_json = mock.AsyncMock(side_effect=[slots(0.0), {}])
            asyncio.run(state.restore_slot(profile, "fast.bin"))
            prime = state._request_json.await_args_list[1]
            self.assertEqual(prime.args[1:3], ("POST", "/completion"))
            self.assertEqual(prime.args[3]["lora"], [{"id": 0, "scale": 1.0}])
            self.assertEqual(prime.args[3]["n_predict"], 0)
            state._slot_action.assert_awaited_with(profile, "restore", "fast.bin")

            # Already in the right mode: no priming request.
            state._request_json = mock.AsyncMock(return_value=slots(1.0))
            asyncio.run(state.restore_slot(profile, "fast.bin"))
            self.assertEqual(state._request_json.await_count, 1)

            # Snapshots from before fast mode count as adapter-off.
            state._request_json = mock.AsyncMock(side_effect=[slots(1.0), {}])
            asyncio.run(state.restore_slot(profile, "legacy.bin"))
            self.assertEqual(state._request_json.await_args_list[1].args[3]["lora"], [{"id": 0, "scale": 0.0}])

    def test_profiles_without_fast_mode_restore_without_extra_requests(self) -> None:
        profile = fast_profile(None)
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state(Path(tmp), profile)
            state._request_json = mock.AsyncMock()
            asyncio.run(state.restore_slot(profile, "snap.bin"))
            asyncio.run(state.save_slot(profile, "snap.bin"))
            state._request_json.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
