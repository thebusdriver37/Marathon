from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from marathon_app import __main__ as main_module
from marathon_app import catalog
from marathon_app import model_library
from marathon_app.codex_home import SHARED_PROFILE_FILE, prepare_codex_home
from marathon_app.frontends import (
    _codex_binary,
    _stream_chat,
    codex_command,
    hermes_command,
    run_codex,
    run_hermes,
)
from marathon_app.codex_telemetry import snapshot_sessions, summarize_session_changes
from marathon_app import runtime as runtime_module
from marathon_app.runtime import (
    Runtime,
    _loaded_model_context,
    _model_is_loaded,
    _props_context_window,
)
from marathon_app.ui import (
    Selection,
    _ensure_local_tools,
    _home,
    _home_items,
    _initial_selection,
    run_codex_default,
)
from marathon_app.telemetry import EventWriter, read_events, summarize_run


TEST_EXECUTABLE = Path(sys.executable)


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
    def test_catalog_reparses_only_when_a_source_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.toml"
            path.write_text("[settings]\nvalue = 1\n", encoding="utf-8")
            catalog._load_catalog_cached.cache_clear()
            with mock.patch.object(
                catalog.tomllib, "load", wraps=catalog.tomllib.load
            ) as parse:
                first = catalog.load_catalog(path)
                second = catalog.load_catalog(path)
                path.write_text("[settings]\nvalue = 200\n", encoding="utf-8")
                third = catalog.load_catalog(path)

        self.assertEqual(first, second)
        self.assertEqual(first["settings"]["value"], 1)
        self.assertEqual(third["settings"]["value"], 200)
        self.assertEqual(parse.call_count, 2)

    def test_user_catalog_merges_profiles_over_base(self) -> None:
        base = {
            "settings": {"ai_root": "~/AI"},
            "backends": [{"id": "upstream", "server": "backends/llama-server"}],
            "families": [
                {
                    "id": "qwen",
                    "display_name": "Qwen",
                    "patterns": ["qwen"],
                    "backend": "upstream",
                    "default_profile": "base",
                    "profiles": [
                        {
                            "id": "base",
                            "display_name": "Base",
                            "context": 1024,
                            "split_mode": "layer",
                            "tensor_split": "1,1,1,1",
                        }
                    ],
                }
            ],
        }
        override = {
            "families": [
                {
                    "id": "qwen",
                    "profiles": [
                        {
                            "id": "two-gpu",
                            "display_name": "Two GPU",
                            "context": 262144,
                            "split_mode": "layer",
                            "tensor_split": "1,1",
                            "gpus": [0, 1],
                        }
                    ],
                }
            ]
        }

        merged = catalog._merge_catalog(base, override)
        family = merged["families"][0]

        self.assertEqual(
            [profile["id"] for profile in family["profiles"]],
            ["base", "two-gpu"],
        )
        two_gpu = family["profiles"][1]
        self.assertEqual(two_gpu["tensor_split"], "1,1")
        self.assertEqual(two_gpu["gpus"], [0, 1])
        # Base profile and base backend survive the merge untouched.
        self.assertEqual(family["profiles"][0]["tensor_split"], "1,1,1,1")
        self.assertEqual(merged["backends"], base["backends"])
        self.assertEqual(merged["settings"], base["settings"])

    def test_reasoning_catalog_requires_structured_levels(self) -> None:
        loaded = catalog.load_catalog()
        loaded["families"][0]["reasoning_levels"] = "low"

        with self.assertRaisesRegex(ValueError, "list of tables"):
            catalog.families(loaded)

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

    def test_speculative_sidecars_are_not_offered_as_full_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "DeepSeek-V4-Flash-IQ2_XXS.gguf").write_bytes(b"model")
            for filename in (
                "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
                "Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
                "Qwen3.8-27B-DSpark-Q4_K_M.gguf",
                "Qwen3.8-27B-Eagle3-Q4_K_M.gguf",
            ):
                (root / filename).write_bytes(b"draft")

            models = catalog.discover_models(root)

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].path.name, "DeepSeek-V4-Flash-IQ2_XXS.gguf")

    def test_multimodal_projector_is_attached_to_model_and_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Qwen3.8-27B-Q8_0.gguf").write_bytes(b"model")
            projector = root / "mmproj-F16.gguf"
            projector.write_bytes(b"projector")

            model = catalog.discover_models(root)[0]
            profile = catalog.find_profile(model, "native-256k", "codex")
            backend = catalog.Backend("test", "Test backend", TEST_EXECUTABLE)
            command = catalog.server_command(model, profile, backend)

        self.assertEqual(model.multimodal_projector, projector)
        self.assertEqual(command[command.index("--mmproj") + 1], str(projector))

    def test_named_vision_sidecar_is_not_offered_as_full_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Qwen3.8-27B-Uncensored-Q5_K_M.gguf").write_bytes(b"model")
            projector = root / "Qwen3.8-27B-Uncensored-vision-f16.gguf"
            projector.write_bytes(b"projector")

            models = catalog.discover_models(root)

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].multimodal_projector, projector)

    def test_qwen_quant_and_family_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
            path.write_bytes(b"model")
            model = catalog.discover_models(root)[0]
        self.assertEqual(model.family.id, "qwen3.6-27b")
        self.assertEqual(model.quant, "UD-Q4_K_XL")

    def test_embedded_gguf_metadata_identifies_a_renamed_model(self) -> None:
        from gguf import GGUFWriter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mystery-Q8_0.gguf"
            writer = GGUFWriter(path, "qwen3")
            writer.add_name("Qwen3.8-27B")
            writer.add_context_length(262_144)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            model = catalog.discover_models(Path(directory))[0]

        self.assertEqual(model.family.id, "qwen3.8-27b")
        self.assertEqual(model.architecture, "qwen3")
        self.assertEqual(model.native_context, 262_144)

    def test_qwen38_profile_has_bounded_post_tool_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "Qwen3.8-27B-Q8_0.gguf"
            path.write_bytes(b"model")
            model = catalog.discover_models(root)[0]

        profile = catalog.find_profile(model, None, "codex")
        self.assertEqual(model.family.id, "qwen3.8-27b")
        self.assertEqual(model.quant, "Q8_0")
        self.assertEqual(model.family.default_reasoning_level, "xhigh")
        self.assertEqual(
            [level.effort for level in model.family.reasoning_levels],
            ["none", "low", "medium", "xhigh"],
        )
        self.assertEqual(profile.id, "auto")
        self.assertEqual(profile.context, 262_144)
        self.assertEqual(profile.tool_thinking_budget, 2_048)

        backend = catalog.Backend("test", "Test backend", TEST_EXECUTABLE)
        command = catalog.server_command(model, profile, backend)
        self.assertEqual(command[command.index("--ctx-size") + 1], "262144")
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "auto")
        self.assertNotIn("--tensor-split", command)
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q8_0")
        self.assertIn("--cache-prompt", command)
        self.assertIn("--cache-idle-slots", command)
        self.assertEqual(command[command.index("--cache-ram") + 1], "8192")
        self.assertEqual(command[command.index("--fit") + 1], "on")
        self.assertEqual(command[command.index("--fit-ctx") + 1], "32768")

        exact = catalog.find_profile(model, "native-256k", "codex")
        exact_command = catalog.server_command(model, exact, backend)
        self.assertEqual(
            exact_command[exact_command.index("--tensor-split") + 1], "1,1,1,1"
        )
        self.assertEqual(exact_command[exact_command.index("--fit") + 1], "off")

    def test_qwen38_two_gpu_profile_pins_gpus_and_split(self) -> None:
        base = catalog.load_catalog(catalog.CATALOG_PATH)
        override = {
            "families": [
                {
                    "id": "qwen3.8-27b",
                    "profiles": [
                        {
                            "id": "two-gpu-256k",
                            "display_name": "Two GPU 256K",
                            "context": 262_144,
                            "gpu_layers": "999",
                            "split_mode": "layer",
                            "tensor_split": "1,1",
                            "cache_k": "q8_0",
                            "cache_v": "q8_0",
                            "extra_args": ["--fit", "off"],
                            "frontends": ["codex"],
                            "gpus": [0, 1],
                        }
                    ],
                }
            ]
        }
        merged = catalog._merge_catalog(base, override)
        family = next(
            item for item in catalog.families(merged) if item.id == "qwen3.8-27b"
        )
        model = catalog.Model(
            id="qwen3.8-27b-q8-0",
            display_name="Qwen 3.8 27B Q8_0",
            path=Path("/tmp/Qwen3.8-27B-Q8_0.gguf"),
            size_bytes=27 * 1024**3,
            family=family,
            quant="Q8_0",
        )

        profile = catalog.find_profile(model, "two-gpu-256k", "codex")
        self.assertEqual(model.family.id, "qwen3.8-27b")
        self.assertEqual(profile.gpus, (0, 1))
        self.assertEqual(profile.tensor_split, "1,1")
        self.assertEqual(profile.gpu_layers, "999")
        self.assertEqual(profile.context, 262_144)
        self.assertEqual(profile.cache_k, "q8_0")
        self.assertEqual(profile.cache_v, "q8_0")

        backend = catalog.Backend("test", "Test backend", TEST_EXECUTABLE)
        command = catalog.server_command(model, profile, backend)
        self.assertEqual(command[command.index("--ctx-size") + 1], "262144")
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "999")
        self.assertEqual(command[command.index("--tensor-split") + 1], "1,1")
        self.assertEqual(command[command.index("--fit") + 1], "off")

    def test_registered_model_roots_are_discovered_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            first = home / "existing-a"
            second = home / "existing-b"
            first.mkdir()
            second.mkdir()
            (first / "Qwen3.8-27B-Q4_K_M.gguf").write_bytes(b"first")
            (second / "my-model-Q8_0.gguf").write_bytes(b"second")
            library_file = home / "config" / "models.json"
            with mock.patch.dict(
                os.environ,
                {
                    "HOME": str(home),
                    "MARATHON_AI_ROOT": str(home / "AI"),
                    "MARATHON_MODEL_LIBRARY_FILE": str(library_file),
                },
                clear=False,
            ):
                model_library.register_model_root(first)
                model_library.register_model_root(second)
                models = catalog.discover_models()
                registered = model_library.load_registered_model_roots(library_file)

        self.assertEqual(
            {model.path.parent for model in models},
            {first.resolve(), second.resolve()},
        )
        self.assertEqual(registered, (first.resolve(), second.resolve()))

    def test_generic_automatic_profile_uses_conservative_context(self) -> None:
        model = fixture_model("generic")

        profile = catalog.find_profile(model, None, "codex")

        self.assertEqual(profile.id, "auto")
        self.assertEqual(profile.context, 32_768)
        self.assertEqual(profile.extra_args, ("--fit", "on", "--fit-ctx", "8192"))

    def test_quick_profile_rejects_codex(self) -> None:
        model = fixture_model()
        with self.assertRaisesRegex(ValueError, "not compatible with codex"):
            catalog.find_profile(model, "quick", "codex")

    def test_server_command_uses_selected_profile(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        backend = catalog.Backend("test", "Test backend", TEST_EXECUTABLE)
        command = catalog.server_command(model, profile, backend)
        self.assertEqual(command[command.index("--ctx-size") + 1], "65536")
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "999")
        self.assertEqual(command[command.index("--tensor-split") + 1], "1,1,1,1")
        self.assertEqual(command[command.index("--flash-attn") + 1], "on")

    def test_prompt_cache_ram_has_environment_override(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        backend = catalog.Backend("test", "Test backend", TEST_EXECUTABLE)

        with mock.patch.dict(
            os.environ,
            {"MARATHON_PROMPT_CACHE_RAM_MIB": "4096"},
            clear=False,
        ):
            command = catalog.server_command(model, profile, backend)

        self.assertEqual(command[command.index("--cache-ram") + 1], "4096")

    def test_slot_snapshot_settings_have_environment_overrides(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "MARATHON_SLOT_SNAPSHOTS_ENABLED": "1",
                "MARATHON_SLOT_SNAPSHOT_MAX_COUNT": "7",
                "MARATHON_SLOT_SNAPSHOT_MAX_BYTES": "12345",
                "MARATHON_SLOT_SNAPSHOT_CLEAN_STARTUP": "true",
            },
            clear=False,
        ):
            configured = catalog.settings()

        self.assertTrue(configured.slot_snapshots_enabled)
        self.assertEqual(configured.slot_snapshot_max_count, 7)
        self.assertEqual(configured.slot_snapshot_max_bytes, 12_345)
        self.assertTrue(configured.slot_snapshot_clean_startup)

    def test_deepseek_profiles_keep_optimized_paths_separate(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        self.assertEqual(model.family.backend, "ds4-distributed")
        safe_profile = catalog.find_profile(model, "safe", "direct")
        long_profile = catalog.find_profile(model, "long-64k", "codex")
        mtp_profile = catalog.find_profile(model, "experimental-mtp-64k", "codex")
        mtp_128k_profile = catalog.find_profile(
            model, "experimental-mtp-128k", "codex"
        )
        self.assertEqual(
            safe_profile.extra_args,
            ("--dist-prefill-chunk", "512", "--dist-prefill-window", "4"),
        )
        self.assertEqual(long_profile.backend, "deepseek-v4-longctx")
        self.assertNotIn("dsv4-mtp", long_profile.extra_args)
        self.assertEqual(long_profile.temperature, 0.0)
        long_environment = dict(catalog.backends()["deepseek-v4-longctx"].environment)
        mtp_environment = dict(catalog.backends()["deepseek-v4-longctx-mtp"].environment)
        for environment in (long_environment, mtp_environment):
            self.assertEqual(environment["DSV4_SPARSE_FA"], "1")
            self.assertNotIn("DSV4_FA_UNION", environment)
        self.assertEqual(mtp_environment["DSV4_MTP_DEV"], "CUDA3")
        self.assertEqual(mtp_environment["DSV4_MTP_EMBD_DEV"], "CUDA3")
        self.assertEqual(mtp_profile.backend, "deepseek-v4-longctx-mtp")
        self.assertIn("dsv4-mtp", mtp_profile.extra_args)
        self.assertEqual(mtp_128k_profile.backend, "deepseek-v4-longctx-mtp")
        self.assertEqual(mtp_128k_profile.context, 131_072)
        self.assertEqual(mtp_128k_profile.tensor_split, "1,1,1,0.85")
        self.assertEqual(mtp_128k_profile.cache_k, "f16")
        self.assertEqual(mtp_128k_profile.cache_v, "f16")
        self.assertIn("dsv4-mtp", mtp_128k_profile.extra_args)
        self.assertEqual(mtp_128k_profile.confidence, "verified")
        self.assertIsNone(long_profile.tool_thinking_budget)
        self.assertFalse(long_profile.parallel_tool_calls)

    def test_profile_backend_selection_is_generic(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        profile = catalog.find_profile(model, "experimental-mtp-64k", "codex")
        selected = catalog.Backend("deepseek-v4-longctx-mtp", "MTP", TEST_EXECUTABLE)
        fallback = catalog.Backend("ds4-distributed", "DS4", TEST_EXECUTABLE)
        with mock.patch.object(
            catalog,
            "backends",
            return_value={selected.id: selected, fallback.id: fallback},
        ):
            self.assertEqual(catalog.backend_for(model, profile), selected)
            self.assertEqual(catalog.backend_for(model), fallback)

    def test_backend_environment_expands_model_directory_and_user_override(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        backend = catalog.Backend(
            "test",
            "Test",
            TEST_EXECUTABLE,
            environment=(
                ("DSV4_MTP_GGUF", "{model_dir}/mtp.gguf"),
                ("DSV4_MOE_TILE", "1"),
            ),
        )
        with mock.patch.dict(os.environ, {"DSV4_MOE_TILE": "0"}, clear=False):
            environment = catalog.backend_environment(model, backend)
        self.assertEqual(environment["DSV4_MTP_GGUF"], "/tmp/mtp.gguf")
        self.assertEqual(environment["DSV4_MOE_TILE"], "0")

    def test_missing_backend_sidecar_fails_before_server_launch(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        profile = catalog.find_profile(model, "experimental-mtp-64k", "codex")
        backend = catalog.Backend(
            profile.backend or "test",
            "Test",
            TEST_EXECUTABLE,
            environment=(("DSV4_MTP_GGUF", "{model_dir}/missing-mtp.gguf"),),
        )
        with mock.patch.object(catalog, "backends", return_value={backend.id: backend}):
            with self.assertRaisesRegex(ValueError, "DSV4_MTP_GGUF"):
                catalog.backend_for(model, profile)

    def test_portable_ai_root_resolves_catalog_paths(self) -> None:
        loaded = catalog.load_catalog()
        with mock.patch.dict(os.environ, {"HOME": "/tmp/marathon-home"}, clear=True):
            configured = catalog.settings(loaded)
            resolved_backends = catalog.backends(loaded)
            upstream = resolved_backends["upstream"]
            ds4 = resolved_backends["ds4-distributed"]
            longctx = resolved_backends["deepseek-v4-longctx-mtp"]

        self.assertEqual(configured.ai_root, Path("/tmp/marathon-home/AI"))
        self.assertEqual(
            configured.model_root, Path("/tmp/marathon-home/AI/models/gguf")
        )
        self.assertEqual(
            upstream.server,
            Path("/tmp/marathon-home/AI/backends/llama.cpp-current/build/bin/llama-server"),
        )
        self.assertEqual(
            ds4.server,
            Path("/tmp/marathon-home/AI/backends/ds4-tp/ds4-server"),
        )
        self.assertEqual(
            ds4.worker,
            Path("/tmp/marathon-home/AI/backends/ds4-tp/ds4"),
        )
        self.assertEqual(
            longctx.server,
            Path(
                "/tmp/marathon-home/AI/backends/llama.cpp-ds4-longctx/"
                "build-cuda/bin/llama-server"
            ),
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
    def test_port_pid_finds_an_actual_listening_process(self) -> None:
        if not (
            runtime_module.shutil.which("ss")
            or runtime_module.shutil.which("lsof")
        ):
            self.skipTest("port ownership tools are unavailable")
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen()

        self.assertEqual(
            runtime_module._port_pid(listener.getsockname()[1]),
            os.getpid(),
        )

    def test_gpu_processes_include_physical_gpu_index(self) -> None:
        gpu_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, GPU-a\n1, GPU-b\n",
            stderr="",
        )
        process_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="GPU-b, 123, llama-server, 4096\n",
            stderr="",
        )

        with mock.patch(
            "marathon_app.runtime.subprocess.run",
            side_effect=[gpu_result, process_result],
        ):
            processes = runtime_module._gpu_processes()

        self.assertEqual(processes[0]["gpu_index"], "1")
        self.assertEqual(processes[0]["pid"], "123")

    def test_pinned_profile_ignores_processes_on_other_gpus(self) -> None:
        model = fixture_model("qwen3.8-27b")
        profile = replace(
            catalog.find_profile(model, "native-256k", "codex"),
            gpus=(0, 1),
        )
        runtime = Runtime(model, profile)
        processes = [
            {
                "gpu_uuid": "GPU-c",
                "gpu_index": "2",
                "pid": "123",
                "name": "llama-server",
                "memory_mib": "16000",
            }
        ]

        with (
            mock.patch("marathon_app.runtime._port_pid", return_value=None),
            mock.patch("marathon_app.runtime._gpu_processes", return_value=processes),
        ):
            runtime._check_conflicts()

    def test_pinned_profile_blocks_process_on_selected_gpu(self) -> None:
        model = fixture_model("qwen3.8-27b")
        profile = replace(
            catalog.find_profile(model, "native-256k", "codex"),
            gpus=(0, 1),
        )
        runtime = Runtime(model, profile)
        processes = [
            {
                "gpu_uuid": "GPU-b",
                "gpu_index": "1",
                "pid": "123",
                "name": "llama-server",
                "memory_mib": "16000",
            }
        ]

        with (
            mock.patch("marathon_app.runtime._port_pid", return_value=None),
            mock.patch("marathon_app.runtime._gpu_processes", return_value=processes),
            self.assertRaisesRegex(RuntimeError, "GPU 1 PID 123"),
        ):
            runtime._check_conflicts()

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

    def test_loaded_context_accepts_ds4_model_shape(self) -> None:
        payload = {
            "data": [
                {"id": "deepseek-v4-flash", "context_length": 65_536}
            ]
        }

        self.assertEqual(
            _loaded_model_context(payload, "deepseek-v4-flash"), 65_536
        )

    def test_ds4_backend_builds_three_workers_and_one_coordinator(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        profile = catalog.find_profile(model, "safe", "direct")
        runtime = Runtime(model, profile)
        backend = catalog.Backend(
            "ds4",
            "DS4",
            TEST_EXECUTABLE,
            kind="ds4_distributed",
            worker=TEST_EXECUTABLE,
            model_alias="deepseek-v4-flash",
            layer_slices=("0:9", "10:20", "21:31", "32:output"),
            gpu_ids=(0, 1, 2, 3),
        )

        with mock.patch.dict(
            os.environ,
            {"MARATHON_DS4_CONTROL_PORT": "19300"},
            clear=False,
        ):
            specs = runtime._backend_specs(backend, Path("/tmp/slots"))

        self.assertEqual(
            [spec.name for spec in specs],
            ["ds4-worker-1", "ds4-worker-2", "ds4-worker-3", "ds4-coordinator"],
        )
        self.assertEqual(
            dict(specs[0].environment)["CUDA_VISIBLE_DEVICES"], "1"
        )
        coordinator = list(specs[-1].command)
        self.assertIn("--dist-prefill-chunk", coordinator)
        self.assertEqual(coordinator[coordinator.index("--listen") + 2], "19300")

    def test_llama_backend_receives_catalog_environment(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        profile = catalog.find_profile(model, "long-64k", "codex")
        runtime = Runtime(model, profile)
        backend = catalog.Backend(
            "test",
            "Test",
            TEST_EXECUTABLE,
            environment=(("DSV4_MTP_GGUF", "{model_dir}/mtp.gguf"),),
        )

        specs = runtime._backend_specs(backend, Path("/tmp/slots"))

        self.assertEqual(len(specs), 1)
        self.assertEqual(
            dict(specs[0].environment)["DSV4_MTP_GGUF"], "/tmp/mtp.gguf"
        )
        self.assertEqual(
            list(specs[0].command)[-2:],
            ["--slot-save-path", "/tmp/slots"],
        )

    def test_multimodal_llama_backend_enables_slot_api(self) -> None:
        model = replace(
            fixture_model("qwen3.8-27b"),
            multimodal_projector=Path("/tmp/mmproj-F16.gguf"),
        )
        profile = catalog.find_profile(model, "native-256k", "codex")
        runtime = Runtime(model, profile)
        backend = catalog.Backend("test", "Test", TEST_EXECUTABLE)

        specs = runtime._backend_specs(backend, Path("/tmp/slots"))

        self.assertTrue(runtime_module._slot_api_enabled(model, backend))
        self.assertEqual(
            list(specs[0].command)[-2:],
            ["--slot-save-path", "/tmp/slots"],
        )

    def test_backend_can_explicitly_disable_slot_api_for_text_model(self) -> None:
        model = fixture_model("qwen3.8-27b")
        backend = catalog.Backend(
            "test",
            "Test",
            TEST_EXECUTABLE,
            supports_slots=False,
        )

        self.assertFalse(runtime_module._slot_api_enabled(model, backend))

    def test_llama_backend_pins_profile_gpus(self) -> None:
        model = fixture_model("qwen3.8-27b")
        base_profile = catalog.find_profile(model, "native-256k", "codex")
        profile = replace(base_profile, gpus=(0, 1), tensor_split="1,1")
        runtime = Runtime(model, profile)
        backend = catalog.Backend("test", "Test", TEST_EXECUTABLE)

        specs = runtime._backend_specs(backend, Path("/tmp/slots"))

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "llama")
        self.assertEqual(
            dict(specs[0].environment)["CUDA_VISIBLE_DEVICES"], "0,1"
        )

    def test_llama_backend_without_gpus_leaves_devices_unpinned(self) -> None:
        model = fixture_model("qwen3.8-27b")
        profile = catalog.find_profile(model, "native-256k", "codex")
        runtime = Runtime(model, profile)
        backend = catalog.Backend("test", "Test", TEST_EXECUTABLE)

        specs = runtime._backend_specs(backend, Path("/tmp/slots"))

        self.assertNotIn("CUDA_VISIBLE_DEVICES", dict(specs[0].environment))

    def test_model_readiness_accepts_llama_models_shape(self) -> None:
        payload = {"models": [{"name": "local-model"}]}

        self.assertTrue(_model_is_loaded(payload, "local-model"))

    def test_context_limits_follow_loaded_model_context(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, None)
        runtime = Runtime(model, profile)
        runtime._context_window = 131_072

        self.assertEqual(runtime.context_window, 131_072)
        self.assertEqual(runtime.context_reserve_tokens, 16_384)
        self.assertEqual(runtime.auto_compact_token_limit, 114_688)
        self.assertEqual(runtime.truncation_limit, 108_135)

    def test_context_headroom_scales_without_model_specific_constants(self) -> None:
        model = fixture_model()
        runtime = Runtime(model, catalog.find_profile(model, None))
        expected = {
            65_536: (12_288, 53_248, 49_972),
            131_072: (16_384, 114_688, 108_135),
            262_144: (32_768, 229_376, 221_184),
        }
        for context, limits in expected.items():
            with self.subTest(context=context):
                runtime._context_window = context
                self.assertEqual(
                    (
                        runtime.context_reserve_tokens,
                        runtime.auto_compact_token_limit,
                        runtime.truncation_limit,
                    ),
                    limits,
                )

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


class CodexHomeTests(unittest.TestCase):
    def test_isolated_home_shares_tools_without_sharing_writable_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock = root / "stock"
            isolated = root / "isolated"
            stock.mkdir()
            (stock / "config.toml").write_text(
                'model = "gpt-stock"\n'
                'model_reasoning_effort = "medium"\n'
                'sqlite_home = "/tmp/stock-sqlite"\n'
                'log_dir = "/tmp/stock-logs"\n'
                'personality = "pragmatic"\n'
                "\n[mcp_servers.example]\n"
                'command = "example-mcp"\n',
                encoding="utf-8",
            )
            (stock / "AGENTS.md").write_text("shared instructions\n", encoding="utf-8")
            (stock / "skills").mkdir()
            (stock / "skills" / "local-skill").mkdir()
            isolated.mkdir()
            (isolated / "config.toml").write_text(
                'model = "marathon-local-model"\n', encoding="utf-8"
            )

            home, profile = prepare_codex_home(
                {
                    "CODEX_HOME": str(stock),
                    "MARATHON_CODEX_HOME": str(isolated),
                }
            )

            shared_config = (isolated / SHARED_PROFILE_FILE).read_text(
                encoding="utf-8"
            )
            self.assertEqual(home, isolated.resolve())
            self.assertEqual(profile, "marathon-shared")
            self.assertEqual(
                (isolated / "config.toml").read_text(encoding="utf-8"),
                'model = "marathon-local-model"\n',
            )
            self.assertNotIn('model = "gpt-stock"', shared_config)
            self.assertNotIn("model_reasoning_effort", shared_config)
            self.assertNotIn("sqlite_home", shared_config)
            self.assertNotIn("log_dir", shared_config)
            self.assertIn('personality = "pragmatic"', shared_config)
            self.assertIn("[mcp_servers.example]", shared_config)
            self.assertEqual((isolated / SHARED_PROFILE_FILE).stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                (isolated / "AGENTS.md").resolve(), (stock / "AGENTS.md").resolve()
            )
            self.assertEqual(
                (isolated / "skills" / "local-skill").resolve(),
                (stock / "skills" / "local-skill").resolve(),
            )

    def test_existing_marathon_sessions_move_out_of_stock_codex_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock = root / "stock"
            isolated = root / "isolated"
            sessions = stock / "sessions" / "2026" / "08" / "19"
            sessions.mkdir(parents=True)
            marathon_session = sessions / "marathon.jsonl"
            legacy_session = sessions / "legacy-marathon.jsonl"
            stock_session = sessions / "stock.jsonl"
            marathon_session.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "marathon-session",
                            "model_provider": "marathon-local",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            legacy_session.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "legacy-marathon-session",
                            "model_provider": "marathon_local",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stock_session.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "stock-session",
                            "model_provider": "openai",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            prepare_codex_home(
                {
                    "CODEX_HOME": str(stock),
                    "MARATHON_CODEX_HOME": str(isolated),
                }
            )

            migrated = isolated / "sessions" / "2026" / "08" / "19" / "marathon.jsonl"
            migrated_legacy = (
                isolated
                / "sessions"
                / "2026"
                / "08"
                / "19"
                / "legacy-marathon.jsonl"
            )
            self.assertTrue(migrated.is_file())
            self.assertFalse(marathon_session.exists())
            self.assertTrue(migrated_legacy.is_file())
            self.assertFalse(legacy_session.exists())
            legacy_meta = json.loads(
                migrated_legacy.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                legacy_meta["payload"]["model_provider"], "marathon-local"
            )
            self.assertTrue(stock_session.is_file())

    def test_codex_binary_override_is_consistent_across_entry_points(self) -> None:
        paths = (
            Path("marathon_app/frontends.py"),
            Path("scripts/build_codex.sh"),
            Path("scripts/ops/update_codex.sh"),
            Path("scripts/ops/doctor.sh"),
        )
        for path in paths:
            with self.subTest(path=path):
                content = path.read_text(encoding="utf-8")
                self.assertIn("MARATHON_CODEX_BIN", content)
                self.assertNotIn("MARATHON_CODEX_BIN_PATH", content)
                self.assertNotIn("MARATHON_PATCHED_CODEX_BIN", content)


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

    def test_codex_uses_per_model_catalog_without_ignoring_user_config(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        runtime = Runtime(model, profile)
        runtime._context_window = 262_144
        command = codex_command(runtime)
        joined = " ".join(command)
        self.assertIn("marathon-local", joined)
        self.assertIn("model_catalog_json", joined)
        self.assertNotIn("model_context_window=262144", command)
        self.assertNotIn("model_auto_compact_token_limit=229376", command)
        self.assertNotIn("--ignore-user-config", command)

    def test_codex_cli_arguments_reach_the_supervised_frontend(self) -> None:
        with mock.patch.object(
            main_module, "run_codex_default", return_value=0
        ) as run:
            result = main_module.main(
                ["codex", "--sandbox", "read-only", "--cd", "/tmp/project"]
            )

        self.assertEqual(result, 0)
        run.assert_called_once_with(
            ["--sandbox", "read-only", "--cd", "/tmp/project"]
        )

    def test_exec_uses_the_supervised_codex_runtime(self) -> None:
        with mock.patch.object(
            main_module, "run_codex_default", return_value=7
        ) as run:
            result = main_module.main(["exec", "--json", "check the project"])

        self.assertEqual(result, 7)
        run.assert_called_once_with(["exec", "--json", "check the project"])

    def test_codex_child_uses_marathon_resume_command(self) -> None:
        model = fixture_model()
        runtime = Runtime(model, catalog.find_profile(model, "balanced", "codex"))
        completed = subprocess.CompletedProcess([], 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_home = root / "stock-codex"
            marathon_home = root / "marathon-codex"
            stock_home.mkdir()
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "CODEX_HOME": str(stock_home),
                        "CODEX_SQLITE_HOME": str(root / "stock-sqlite"),
                        "MARATHON_CODEX_HOME": str(marathon_home),
                    },
                    clear=False,
                ),
                mock.patch(
                    "marathon_app.frontends.subprocess.run", return_value=completed
                ) as run,
                mock.patch(
                    "marathon_app.frontends.snapshot_sessions", return_value={}
                ),
                mock.patch(
                    "marathon_app.frontends.summarize_session_changes", return_value=[]
                ) as summarize,
                mock.patch(
                    "marathon_app.frontends.shutil.which",
                    return_value="/usr/bin/marathon",
                ),
            ):
                code = run_codex(runtime, ["resume", "session-id"])

        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.kwargs["env"]["CODEX_CLI_NAME"], "marathon")
        self.assertEqual(
            run.call_args.kwargs["env"]["CODEX_HOME"],
            str(marathon_home.resolve()),
        )
        self.assertEqual(
            run.call_args.kwargs["env"]["CODEX_SQLITE_HOME"],
            str(marathon_home.resolve()),
        )
        self.assertEqual(run.call_args.args[0][-2:], ["resume", "session-id"])
        self.assertEqual(summarize.call_args.kwargs["provider"], "marathon-local")

    def test_project_cannot_enable_patched_features_for_stock_codex(self) -> None:
        model = fixture_model()
        runtime = Runtime(model, catalog.find_profile(model, "balanced", "codex"))

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "codex.features").write_text(
                "tokens-per-second\n", encoding="utf-8"
            )
            previous = Path.cwd()
            try:
                os.chdir(project)
                with mock.patch(
                    "marathon_app.frontends._codex_binary", return_value="codex"
                ):
                    command = codex_command(runtime)
            finally:
                os.chdir(previous)

        self.assertFalse(any("tokens-per-second" in item for item in command))

    def test_patched_codex_enables_turn_throughput_status_item(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        runtime = Runtime(model, profile)

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "marathon" / "bin" / "codex"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            Path(f"{binary}.features").write_text(
                "tokens-per-second\n", encoding="utf-8"
            )
            with mock.patch.dict(
                os.environ,
                {"XDG_DATA_HOME": directory},
                clear=True,
            ):
                command = codex_command(runtime)

        self.assertIn(
            'tui.status_line=["model-with-reasoning", "tokens-per-second", '
            '"context-remaining", "context-window-size", "context-tokens"]',
            command,
        )

    def test_hermes_uses_selected_model_without_ignoring_user_config(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "hermes")
        runtime = Runtime(model, profile)

        command = hermes_command(runtime)

        self.assertEqual(
            command,
            ["hermes", "chat", "--model", model.alias, "--provider", "custom"],
        )
        self.assertNotIn("--ignore-user-config", command)
        self.assertNotIn("--ignore-rules", command)

    def test_hermes_child_targets_marathon_router_only(self) -> None:
        model = fixture_model()
        runtime = Runtime(model, catalog.find_profile(model, "balanced", "hermes"))
        completed = subprocess.CompletedProcess([], 0)

        with mock.patch("marathon_app.frontends.subprocess.run", return_value=completed) as run:
            code = run_hermes(runtime)

        self.assertEqual(code, 0)
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["CUSTOM_BASE_URL"], f"{runtime.router_url}/v1")
        self.assertEqual(environment["HERMES_INFERENCE_MODEL"], model.alias)
        self.assertEqual(environment["HERMES_INFERENCE_PROVIDER"], "custom")

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
        self.assertEqual(captured["temperature"], 0.7)

    def test_direct_chat_uses_model_profile_temperature(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        runtime = Runtime(model, catalog.find_profile(model, "long-64k", "direct"))

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __iter__(self):
                yield b"data: [DONE]\n"

        captured = {}

        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return Response()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _stream_chat(runtime, [{"role": "user", "content": "hi"}])

        self.assertEqual(captured["temperature"], 0.0)


class TelemetryTests(unittest.TestCase):
    def test_event_writer_appends_redacted_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            writer = EventWriter(path, "run-test", "test")
            writer.emit("test.event", {"header": "Authorization: Bearer secret-value"})
            events = list(read_events(path))

        self.assertEqual(events[0]["run_id"], "run-test")
        self.assertEqual(events[0]["event"], "test.event")
        self.assertNotIn("secret-value", json.dumps(events[0]))

    def test_run_summary_calculates_throughput_and_energy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            writer = EventWriter(path, "run-test", "runtime")
            writer.emit(
                "run.started",
                {"model": {"id": "test-model"}, "profile": {"id": "fast", "requested_context": 65536}},
            )
            router = EventWriter(path, "run-test", "router")
            router.emit(
                "router.response.completed",
                {
                    "backend": {
                        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                        "timings": {"prompt_n": 100, "prompt_ms": 500, "predicted_n": 20, "predicted_ms": 1000},
                        "latency_ms": 1500,
                    }
                },
            )
            writer.emit(
                "hardware.gpu.sample",
                {"gpus": [{"power_w": 100, "utilization_pct": 80, "memory_used_mib": 1000, "temperature_c": 60}]},
            )
            writer.emit(
                "codex.session.completed",
                {"tool_metrics": [{"duration_ms": 49}]},
            )
            writer.emit("run.completed", {"duration_s": 2, "dropped_events": 0})
            summary = summarize_run(path)

        self.assertEqual(summary["model"], "test-model")
        self.assertEqual(summary["usage"]["output_tokens"], 20)
        self.assertEqual(summary["prompt_tps"], 200)
        self.assertEqual(summary["decode_tps"], 20)
        self.assertEqual(summary["duration_s"], 2)

    def test_run_summary_counts_hermes_and_chat_api_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            writer = EventWriter(path, "run-test", "runtime")
            writer.emit(
                "run.started",
                {"model": {"id": "test-model"}, "profile": {"id": "agent"}},
            )
            writer.emit("frontend.started", {"frontend": "hermes"})
            writer.emit(
                "router.http.completed",
                {"path": "/v1/chat/completions", "status": 200},
            )

            summary = summarize_run(path)

        self.assertEqual(summary["hermes_sessions"], 1)
        self.assertEqual(summary["chat_completion_requests"], 1)

    def test_run_summary_falls_back_to_llama_process_timings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            writer = EventWriter(path, "run-test", "runtime")
            writer.emit(
                "run.started",
                {"model": {"id": "test-model"}, "profile": {"id": "fast"}},
            )
            writer.emit(
                "process.output",
                {
                    "process": "llama",
                    "message": "I slot print_timing: id 0 | prompt eval time = 500.00 ms / 100 tokens",
                },
            )
            writer.emit(
                "process.output",
                {
                    "process": "llama",
                    "message": "       eval time = 1000.00 ms / 40 tokens (25.00 ms per token)",
                },
            )
            summary = summarize_run(path)

        self.assertEqual(summary["prompt_tps"], 200)
        self.assertEqual(summary["decode_tps"], 40)

    def test_run_summary_reads_ds4_coordinator_timings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            writer = EventWriter(path, "run-test", "runtime")
            writer.emit(
                "run.started",
                {"model": {"id": "deepseek"}, "profile": {"id": "long"}},
            )
            writer.emit(
                "process.output",
                {
                    "process": "ds4-coordinator",
                    "message": "ds4-server: chat ctx=0..1000:1000 prompt done 5.000s",
                },
            )
            writer.emit(
                "process.output",
                {
                    "process": "ds4-coordinator",
                    "message": "ds4-server: chat ctx=0..1000:1000 gen=100 finish=stop 10.000s",
                },
            )
            summary = summarize_run(path)

        self.assertEqual(summary["prompt_tps"], 200)
        self.assertEqual(summary["decode_tps"], 20)

    def test_active_codex_session_reports_tool_failures_before_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            run_path = root / "run.jsonl"
            writer = EventWriter(run_path, "run-test", "runtime")
            writer.emit(
                "run.started",
                {"model": {"id": "test-model"}, "profile": {"id": "fast"}},
            )
            writer.emit(
                "frontend.started",
                {
                    "frontend": "codex",
                    "cwd": str(workspace),
                    "codex_home": str(root),
                },
            )
            sessions = root / "sessions" / "2026" / "01" / "01"
            sessions.mkdir(parents=True)
            rollout = sessions / "rollout.jsonl"
            records = [
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "session-1",
                        "cwd": str(workspace),
                        "model_provider": "marathon-local",
                    },
                },
                {
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "name": "apply_patch",
                        "call_id": "call-1",
                    },
                },
                {
                    "timestamp": "2026-01-01T00:00:00.250Z",
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "call_id": "call-1",
                        "output": "apply_patch verification failed: invalid patch",
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {"input_tokens": 30, "output_tokens": 5},
                            "last_token_usage": {"input_tokens": 30, "output_tokens": 5},
                        },
                    },
                },
            ]
            rollout.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"CODEX_HOME": str(root / "stock-codex")},
                clear=False,
            ):
                summary = summarize_run(run_path, live=True)

        self.assertEqual(summary["codex_sessions"], 1)
        self.assertEqual(summary["active_codex_sessions"], 1)
        self.assertEqual(summary["tool_calls"], {"apply_patch": 1})
        self.assertEqual(summary["tool_failures"], 1)
        self.assertEqual(summary["failed_tools"], {"apply_patch": 1})
        self.assertEqual(summary["codex_usage"]["output_tokens"], 5)

    def test_gpu_sampler_does_not_report_shutdown_signal_as_hardware_error(self) -> None:
        model = fixture_model()
        runtime = Runtime(model, catalog.find_profile(model, "balanced", "codex"))
        interrupted = mock.Mock(returncode=-signal.SIGINT, stdout="", stderr="")
        with mock.patch("marathon_app.runtime.subprocess.run", return_value=interrupted):
            with mock.patch.object(runtime, "record") as record:
                runtime._sample_gpus()

        record.assert_not_called()

    def test_codex_import_reads_new_complete_line_and_no_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            sessions = Path(directory) / "sessions" / "2026" / "01" / "01"
            sessions.mkdir(parents=True)
            path = sessions / "rollout.jsonl"
            stock_path = sessions / "stock-rollout.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "session-1",
                            "cwd": str(workspace),
                            "model_provider": "marathon-local",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stock_path.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "stock-session",
                            "cwd": str(workspace),
                            "model_provider": "openai",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"CODEX_HOME": directory}, clear=False):
                before = snapshot_sessions()
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "type": "turn_context",
                        "payload": {"model": "local", "effort": "high", "developer_instructions": "private"},
                    }) + "\n")
                    handle.write(json.dumps({
                        "type": "event_msg",
                        "payload": {"type": "token_count", "info": {
                            "total_token_usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
                            "last_token_usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
                            "model_context_window": 65536,
                        }},
                    }) + "\n")
                    handle.write(json.dumps({
                        "timestamp": "2026-01-01T00:00:00.000Z",
                        "type": "response_item",
                        "payload": {"type": "function_call", "name": "exec_command", "call_id": "call-1", "arguments": "private"},
                    }) + "\n")
                    handle.write(json.dumps({
                        "timestamp": "2026-01-01T00:00:00.250Z",
                        "type": "response_item",
                        "payload": {"type": "function_call_output", "call_id": "call-1", "output": "private"},
                    }) + "\n")
                with stock_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "type": "turn_context",
                                "payload": {
                                    "cwd": str(workspace),
                                    "model": "gpt-stock",
                                },
                            }
                        )
                        + "\n"
                    )
                summaries = summarize_session_changes(
                    before,
                    cwd=workspace,
                    provider="marathon-local",
                )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["session_id"], "session-1")
        self.assertEqual(summaries[0]["provider"], "marathon-local")
        self.assertEqual(summaries[0]["reasoning_efforts"], ["high"])
        self.assertEqual(summaries[0]["token_delta"]["total_tokens"], 35)
        self.assertEqual(summaries[0]["tool_calls"], {"exec_command": 1})
        self.assertEqual(summaries[0]["tool_metrics"][0]["duration_ms"], 250)
        self.assertNotIn("private", json.dumps(summaries[0]))

class UiTests(unittest.TestCase):
    def test_direct_frontend_does_not_require_codex(self) -> None:
        model = fixture_model("qwen3.8-27b")
        console = mock.Mock()
        with (
            mock.patch("marathon_app.ui.backend_for"),
            mock.patch("marathon_app.ui._codex_binary") as codex_binary,
        ):
            ready = _ensure_local_tools(
                console, Selection(model, model.family.profiles[0], "direct"), "direct"
            )

        self.assertTrue(ready)
        codex_binary.assert_not_called()

    def test_initial_selection_prefers_qwen38(self) -> None:
        older = fixture_model("qwen3.6-27b")
        current = fixture_model("qwen3.8-27b")

        selection = _initial_selection([older, current], remembered={})

        self.assertEqual(selection.model.family.id, "qwen3.8-27b")
        self.assertEqual(selection.profile.id, "auto")

    def test_default_launch_opens_codex_without_home_menu(self) -> None:
        model = fixture_model("qwen3.8-27b")
        runtime = mock.Mock()
        console = mock.MagicMock()
        console.status.return_value.__enter__.return_value = mock.Mock()
        with (
            mock.patch("marathon_app.ui.Console", return_value=console),
            mock.patch("marathon_app.ui.discover_models", return_value=[model]),
            mock.patch("marathon_app.ui.load_selection", return_value={}),
            mock.patch("marathon_app.ui._ensure_local_tools", return_value=True),
            mock.patch("marathon_app.ui.save_selection") as save,
            mock.patch("marathon_app.ui.Runtime", return_value=runtime),
            mock.patch("marathon_app.ui._launch_frontend", return_value=0) as launch,
            mock.patch("marathon_app.ui._home") as home,
        ):
            result = run_codex_default()

        self.assertEqual(result, 0)
        home.assert_not_called()
        runtime.start.assert_called_once()
        launch.assert_called_once_with(console, runtime, "codex", None)
        runtime.cleanup.assert_called_once()
        self.assertEqual(save.call_args.args[2], "codex")

    def test_default_launch_returns_the_codex_exit_status(self) -> None:
        model = fixture_model("qwen3.8-27b")
        runtime = mock.Mock()
        console = mock.MagicMock()
        console.status.return_value.__enter__.return_value = mock.Mock()
        with (
            mock.patch("marathon_app.ui.Console", return_value=console),
            mock.patch("marathon_app.ui.discover_models", return_value=[model]),
            mock.patch("marathon_app.ui.load_selection", return_value={}),
            mock.patch("marathon_app.ui._ensure_local_tools", return_value=True),
            mock.patch("marathon_app.ui.save_selection"),
            mock.patch("marathon_app.ui.Runtime", return_value=runtime),
            mock.patch("marathon_app.ui._launch_frontend", return_value=7),
        ):
            result = run_codex_default(["exec", "check the project"])

        self.assertEqual(result, 7)
        runtime.cleanup.assert_called_once()

    def test_warm_model_change_returns_to_runtime_supervisor(self) -> None:
        model = fixture_model()
        models = [model]
        current = Selection(model, catalog.find_profile(model, "balanced"), "codex")
        changed = Selection(model, catalog.find_profile(model, "quick"), "direct")
        console = mock.Mock()

        with (
            mock.patch("marathon_app.ui._arrow_menu", return_value=3),
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
            mock.patch("marathon_app.ui._arrow_menu", side_effect=[3, 3]),
            mock.patch("marathon_app.ui._choose_model_profile", return_value=changed),
        ):
            action, result = _home(console, models, current, warm=False)

        self.assertEqual(action, "quit")
        self.assertEqual(result.profile.id, "quick")

    def test_dyno_is_one_cold_menu_entry_and_not_shown_while_warm(self) -> None:
        model = fixture_model()
        selection = Selection(model, catalog.find_profile(model, "balanced"), "codex")

        cold = [item.value for item in _home_items(selection, warm=False)]
        warm = [item.value for item in _home_items(selection, warm=True)]

        self.assertEqual(cold.count("tune"), 1)
        self.assertNotIn("tune", warm)

    def test_hermes_is_offered_only_by_agent_context_profiles(self) -> None:
        model = fixture_model()
        balanced = Selection(
            model, catalog.find_profile(model, "balanced"), "hermes"
        )
        quick = Selection(model, catalog.find_profile(model, "quick"), "direct")

        self.assertIn(
            "hermes", [item.value for item in _home_items(balanced, warm=False)]
        )
        self.assertNotIn(
            "hermes", [item.value for item in _home_items(quick, warm=False)]
        )

    def test_dyno_is_hidden_for_architecture_specific_backend(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        selection = Selection(model, catalog.find_profile(model, "long-64k"), "codex")

        items = [item.value for item in _home_items(selection, warm=False)]

        self.assertNotIn("tune", items)

    def test_remote_dashboard_does_not_offer_local_gpu_tuning(self) -> None:
        model = fixture_model()
        selection = Selection(model, catalog.find_profile(model, "balanced"), "codex")

        remote_items = [
            item.value
            for item in _home_items(selection, warm=False, allow_tune=False)
        ]

        self.assertNotIn("tune", remote_items)


if __name__ == "__main__":
    unittest.main()
