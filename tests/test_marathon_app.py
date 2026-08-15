from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marathon_app import catalog
from marathon_app.frontends import (
    _codex_binary,
    _stream_chat,
    codex_command,
    hermes_command,
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
from marathon_app.ui import Selection, _home, _home_items
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

    def test_mtp_sidecar_is_not_offered_as_a_full_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "DeepSeek-V4-Flash-IQ2_XXS.gguf").write_bytes(b"model")
            (root / "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf").write_bytes(
                b"draft"
            )

            models = catalog.discover_models(root)

        self.assertEqual(len(models), 1)
        self.assertNotIn("MTP", models[0].path.name)

    def test_qwen_quant_and_family_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
            path.write_bytes(b"model")
            model = catalog.discover_models(root)[0]
        self.assertEqual(model.family.id, "qwen3.6-27b")
        self.assertEqual(model.quant, "UD-Q4_K_XL")

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
        self.assertEqual(profile.id, "native-256k")
        self.assertEqual(profile.context, 262_144)
        self.assertEqual(profile.tool_thinking_budget, 2_048)

        backend = catalog.Backend("test", "Test backend", TEST_EXECUTABLE)
        command = catalog.server_command(model, profile, backend)
        self.assertEqual(command[command.index("--ctx-size") + 1], "262144")
        self.assertEqual(command[command.index("--tensor-split") + 1], "1,1,1,1")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q8_0")
        self.assertEqual(command[command.index("--fit") + 1], "off")

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

    def test_deepseek_profiles_keep_optimized_and_legacy_paths_separate(self) -> None:
        model = fixture_model("deepseek-v4-flash")
        self.assertEqual(model.family.backend, "ds4-distributed")
        safe_profile = catalog.find_profile(model, "safe", "direct")
        long_profile = catalog.find_profile(model, "long-64k", "codex")
        mtp_profile = catalog.find_profile(model, "experimental-mtp-64k", "codex")
        mtp_128k_profile = catalog.find_profile(
            model, "experimental-mtp-128k", "codex"
        )
        legacy_profile = catalog.find_profile(model, "legacy-ds4-64k", "codex")
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
        self.assertEqual(legacy_profile.backend, "ds4-distributed")
        self.assertIn("--dist-prefill-chunk", legacy_profile.extra_args)
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
        profile = catalog.find_profile(model, "legacy-ds4-64k", "codex")
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
        self.assertIn("marathon-local", joined)
        self.assertIn("model_catalog_json", joined)
        self.assertIn("model_context_window=262144", command)
        self.assertIn("model_auto_compact_token_limit=229376", command)
        self.assertNotIn("--ignore-user-config", command)

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
                {"frontend": "codex", "cwd": str(workspace)},
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
            with mock.patch.dict(os.environ, {"CODEX_HOME": directory}, clear=False):
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
            sessions = Path(directory) / "sessions" / "2026" / "01" / "01"
            sessions.mkdir(parents=True)
            path = sessions / "rollout.jsonl"
            path.write_text('{"type":"session_meta","payload":{"id":"session-1"}}\n', encoding="utf-8")
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
                summaries = summarize_session_changes(before)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["reasoning_efforts"], ["high"])
        self.assertEqual(summaries[0]["token_delta"]["total_tokens"], 35)
        self.assertEqual(summaries[0]["tool_calls"], {"exec_command": 1})
        self.assertEqual(summaries[0]["tool_metrics"][0]["duration_ms"], 250)
        self.assertNotIn("private", json.dumps(summaries[0]))


class UiTests(unittest.TestCase):
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
