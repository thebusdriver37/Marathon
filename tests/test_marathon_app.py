from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marathon_app import catalog
from marathon_app.frontends import _codex_binary, _stream_chat, codex_command
from marathon_app import runtime as runtime_module
from marathon_app.runtime import (
    Runtime,
    _loaded_model_context,
    _model_is_loaded,
    _props_context_window,
)
from marathon_app.ui import Selection, _home


def fixture_model(family_id: str = "qwen3.6-27b") -> catalog.Model:
    family = next(item for item in catalog.families() if item.id == family_id)
    quant = "UD-Q4_K_XL" if family_id == "qwen3.6-27b" else "Q4_K_M"
    return catalog.Model(
        id=f"{family_id}-{quant.lower()}",
        display_name=f"{family.display_name} {quant}",
        path=Path("/tmp/marathon-test-model.gguf"),
        size_bytes=17 * 1024**3,
        family=family,
        quant=quant,
    )


class CatalogTests(unittest.TestCase):
    def test_discovers_first_shard_and_sums_all_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "DeepSeek-V4-Flash-Q2_K-XL-00001-of-00002.gguf"
            second = root / "DeepSeek-V4-Flash-Q2_K-XL-00002-of-00002.gguf"
            first.write_bytes(b"a" * 11)
            second.write_bytes(b"b" * 13)
            models = catalog.discover_models(root)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].size_bytes, 24)
        self.assertEqual(models[0].quant, "Q2_K-XL")

    def test_qwen_quant_and_family_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
            path.write_bytes(b"model")
            model = catalog.discover_models(root)[0]
        self.assertEqual(model.family.id, "qwen3.6-27b")
        self.assertEqual(model.quant, "UD-Q4_K_XL")

    def test_quick_profile_rejects_codex(self) -> None:
        model = fixture_model()
        with self.assertRaisesRegex(ValueError, "not compatible with codex"):
            catalog.find_profile(model, "quick", "codex")

    def test_server_command_uses_selected_profile(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        backend = catalog.Backend("test", "Test backend", Path("/bin/true"))
        command = catalog.server_command(model, profile, backend)
        self.assertEqual(command[command.index("--ctx-size") + 1], "65536")
        self.assertEqual(command[command.index("--tensor-split") + 1], "1,1,1,1")
        self.assertIn("--flash-attn", command)

    def test_portable_ai_root_resolves_catalog_paths(self) -> None:
        loaded = catalog.load_catalog()
        with mock.patch.dict(os.environ, {"HOME": "/tmp/marathon-home"}, clear=True):
            configured = catalog.settings(loaded)
            upstream = catalog.backends(loaded)["upstream"]

        self.assertEqual(configured.ai_root, Path("/tmp/marathon-home/AI"))
        self.assertEqual(
            configured.model_root, Path("/tmp/marathon-home/AI/models/gguf")
        )
        self.assertEqual(
            upstream.server,
            Path("/tmp/marathon-home/AI/backends/llama.cpp-current/build/bin/llama-server"),
        )

    def test_ai_root_and_specific_model_override_precedence(self) -> None:
        loaded = catalog.load_catalog()
        environment = {
            "HOME": "/tmp/marathon-home",
            "MARATHON_AI_ROOT": "/srv/marathon-ai",
            "MARATHON_MODELS_DIR": "/models/gguf",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            configured = catalog.settings(loaded)
            upstream = catalog.backends(loaded)["upstream"]

        self.assertEqual(configured.ai_root, Path("/srv/marathon-ai"))
        self.assertEqual(configured.model_root, Path("/models/gguf"))
        self.assertEqual(
            upstream.server,
            Path("/srv/marathon-ai/backends/llama.cpp-current/build/bin/llama-server"),
        )


class RuntimeTests(unittest.TestCase):
    def test_loaded_context_uses_backend_runtime_value(self) -> None:
        payload = {
            "data": [
                {
                    "id": "local-model",
                    "meta": {"n_ctx": 262_144, "n_ctx_train": 1_048_576},
                }
            ]
        }

        self.assertEqual(_loaded_model_context(payload, "local-model"), 262_144)

    def test_loaded_context_can_fall_back_to_backend_props(self) -> None:
        payload = {"default_generation_settings": {"n_ctx": 131_072}}

        self.assertEqual(_props_context_window(payload), 131_072)

    def test_model_readiness_accepts_llama_models_shape(self) -> None:
        payload = {"models": [{"name": "local-model"}]}

        self.assertTrue(_model_is_loaded(payload, "local-model"))

    def test_context_limits_follow_loaded_model_context(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, None)
        runtime = Runtime(model, profile)
        runtime._context_window = 131_072

        self.assertEqual(runtime.context_window, 131_072)
        self.assertEqual(runtime.auto_compact_token_limit, 117_964)
        self.assertEqual(runtime.truncation_limit, 111_411)

    def test_cleanup_terminates_owned_process_group(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, None)
        runtime = Runtime(model, profile)
        process = subprocess.Popen(["sleep", "60"], start_new_session=True)
        runtime.llama = process
        runtime.cleanup()
        self.assertIsNotNone(process.poll())

    def test_cleanup_is_idempotent(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, None)
        runtime = Runtime(model, profile)
        runtime.cleanup()
        runtime.cleanup()

    def test_failed_second_runtime_does_not_remove_owner_session(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_dir = root / "runtime"
            session_file = runtime_dir / "session.json"
            lock_file = runtime_dir / "runtime.lock"
            patches = (
                mock.patch.object(runtime_module, "CONFIG_DIR", root / "config"),
                mock.patch.object(runtime_module, "USER_STATE_DIR", root / "state"),
                mock.patch.object(runtime_module, "RUNTIME_DIR", runtime_dir),
                mock.patch.object(runtime_module, "ROUTER_STATE_DIR", root / "router"),
                mock.patch.object(runtime_module, "SLOT_ROOT", root / "slots"),
                mock.patch.object(runtime_module, "SESSION_FILE", session_file),
                mock.patch.object(runtime_module, "LOCK_FILE", lock_file),
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
            ):
                owner = Runtime(model, profile)
                contender = Runtime(model, profile)
                owner.acquire()
                session_file.write_text('{"supervisor_pid": 1}\n', encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "already open"):
                    contender.acquire()
                contender.cleanup()
                self.assertTrue(session_file.exists())
                owner.cleanup()
                self.assertFalse(session_file.exists())


class FrontendTests(unittest.TestCase):
    def test_patched_codex_is_preferred_when_installed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "marathon" / "bin" / "codex"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            with mock.patch.dict(
                os.environ,
                {"XDG_DATA_HOME": directory},
                clear=True,
            ):
                selected = _codex_binary()

        self.assertEqual(selected, str(binary))

    def test_codex_uses_runtime_overrides_without_ignoring_user_config(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        runtime = Runtime(model, profile)
        runtime._context_window = 262_144
        command = codex_command(runtime)
        joined = " ".join(command)
        self.assertIn("marathon_local", joined)
        self.assertIn("model_catalog_json", joined)
        self.assertIn("model_context_window=262144", command)
        self.assertIn("model_auto_compact_token_limit=235929", command)
        self.assertNotIn("--ignore-user-config", command)

    def test_direct_chat_sends_no_tools_or_agent_instructions(self) -> None:
        model = fixture_model()
        runtime = Runtime(model, catalog.find_profile(model, "balanced", "direct"))

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
                yield b"data: [DONE]\n"

        captured = {}

        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            self.assertEqual(timeout, 3600)
            return Response()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            answer = _stream_chat(runtime, [{"role": "user", "content": "hi"}])

        self.assertEqual(answer, "hello")
        self.assertNotIn("tools", captured)
        self.assertNotIn("instructions", captured)


class UiTests(unittest.TestCase):
    def test_warm_model_change_returns_to_runtime_supervisor(self) -> None:
        model = fixture_model()
        models = [model]
        current = Selection(model, catalog.find_profile(model, "balanced"), "codex")
        changed = Selection(model, catalog.find_profile(model, "quick"), "direct")
        console = mock.Mock()

        with (
            mock.patch("marathon_app.ui._arrow_menu", return_value=2),
            mock.patch("marathon_app.ui._choose_model_profile", return_value=changed),
        ):
            action, result = _home(console, models, current, warm=True)

        self.assertEqual(action, "change")
        self.assertEqual(result.profile.id, "quick")

    def test_cold_model_change_stays_on_home_screen(self) -> None:
        model = fixture_model()
        models = [model]
        current = Selection(model, catalog.find_profile(model, "balanced"), "codex")
        changed = Selection(model, catalog.find_profile(model, "quick"), "direct")
        console = mock.Mock()

        with (
            mock.patch("marathon_app.ui._arrow_menu", side_effect=[2, 2]),
            mock.patch("marathon_app.ui._choose_model_profile", return_value=changed),
        ):
            action, result = _home(console, models, current, warm=False)

        self.assertEqual(action, "quit")
        self.assertEqual(result.profile.id, "quick")


if __name__ == "__main__":
    unittest.main()
