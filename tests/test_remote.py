from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from marathon_app import catalog
from marathon_app import remote
from marathon_app.frontends import codex_command


def fixture_model() -> catalog.Model:
    family = next(item for item in catalog.families() if item.id == "qwen3.8-27b")
    return catalog.Model(
        id="qwen3.8-27b-q8-0",
        display_name="Qwen 3.8 27B Dense Q8_0",
        path=Path("/srv/ai/qwen.gguf"),
        size_bytes=28 * 1024**3,
        family=family,
        quant="Q8_0",
        architecture="qwen3",
        native_context=262_144,
    )


class RemoteCatalogTests(unittest.TestCase):
    def test_remote_catalog_round_trip_preserves_profiles(self) -> None:
        model = fixture_model()
        with mock.patch("marathon_app.remote.discover_models", return_value=[model]):
            payload = remote.remote_catalog_payload()

        decoded = remote._models_from_payload(payload)

        self.assertEqual(decoded[0].id, model.id)
        self.assertEqual(decoded[0].size_bytes, model.size_bytes)
        self.assertEqual(decoded[0].architecture, model.architecture)
        self.assertEqual(decoded[0].native_context, model.native_context)
        self.assertEqual(decoded[0].family.default_profile, model.family.default_profile)
        self.assertEqual(
            decoded[0].family.default_reasoning_level,
            model.family.default_reasoning_level,
        )
        self.assertEqual(
            decoded[0].family.reasoning_levels,
            model.family.reasoning_levels,
        )
        self.assertEqual(
            [item.id for item in decoded[0].family.profiles],
            [item.id for item in catalog.profiles_for_model(model)],
        )

    def test_fetch_remote_catalog_uses_noninteractive_ssh(self) -> None:
        model = fixture_model()
        with mock.patch("marathon_app.remote.discover_models", return_value=[model]):
            payload = remote.remote_catalog_payload()
        output = remote.CATALOG_PREFIX + json.dumps(payload) + "\n"
        completed = subprocess.CompletedProcess([], 0, output, "")
        with (
            mock.patch("marathon_app.remote._ssh_binary", return_value="/usr/bin/ssh"),
            mock.patch("marathon_app.remote.subprocess.run", return_value=completed) as run,
        ):
            result = remote.fetch_remote_catalog("gpu-rig")

        command = run.call_args.args[0]
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(command[-2], "gpu-rig")
        self.assertEqual(result.models[0].id, model.id)

    def test_fetch_remote_catalog_rejects_ssh_option_instead_of_host(self) -> None:
        with self.assertRaisesRegex(ValueError, "SSH host"):
            remote.fetch_remote_catalog("-oProxyCommand=bad")

    def test_remote_host_refuses_exposed_inference_bindings(self) -> None:
        exposed = replace(catalog.settings(), router_host="0.0.0.0")
        with mock.patch("marathon_app.remote.settings", return_value=exposed):
            with self.assertRaisesRegex(RuntimeError, "loopback-only"):
                remote.remote_catalog_payload()

    def test_remote_selections_are_isolated_by_host(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selections.json"
            with mock.patch.object(remote, "REMOTE_SELECTION_FILE", path):
                remote.save_remote_selection("rig-a", model, profile, "codex")
                remote.save_remote_selection("rig-b", model, profile, "direct")
                rig_a = remote.load_remote_selection("rig-a")
                rig_b = remote.load_remote_selection("rig-b")

        self.assertEqual(rig_a["frontend"], "codex")
        self.assertEqual(rig_b["frontend"], "direct")


class RemoteRuntimeTests(unittest.TestCase):
    def test_ssh_tunnel_binds_only_to_client_and_host_loopback(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        with (
            mock.patch("marathon_app.remote._available_port", return_value=31415),
            mock.patch("marathon_app.remote._ssh_binary", return_value="/usr/bin/ssh"),
        ):
            runtime = remote.RemoteRuntime("gpu-rig", 18111, model, profile)
            command = runtime._command()

        self.assertIn("127.0.0.1:31415:127.0.0.1:18111", command)
        self.assertNotIn("0.0.0.0", command)
        self.assertEqual(runtime.router_url, "http://127.0.0.1:31415")

    def test_frontend_events_are_sent_to_host_supervisor(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        runtime = remote.RemoteRuntime("gpu-rig", 18111, model, profile)

        class Process:
            stdin = io.StringIO()

            @staticmethod
            def poll():
                return None

        runtime.process = Process()  # type: ignore[assignment]
        runtime.record("frontend.started", {"frontend": "codex"})
        payload = json.loads(Process.stdin.getvalue())

        self.assertEqual(payload["op"], "event")
        self.assertEqual(payload["event"], "frontend.started")

    def test_local_codex_command_targets_tunnel_and_per_model_catalog(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        with mock.patch("marathon_app.remote._available_port", return_value=31415):
            runtime = remote.RemoteRuntime("gpu-rig", 18111, model, profile)
        runtime._context_window = 131_072
        runtime._auto_compact_token_limit = 114_688

        command = codex_command(runtime)  # type: ignore[arg-type]
        joined = " ".join(command)

        self.assertIn('base_url = "http://127.0.0.1:31415/v1"', joined)
        self.assertIn(
            f"model_catalog_json={json.dumps(str(runtime.catalog_file))}",
            command,
        )
        self.assertNotIn("model_context_window=131072", command)
        self.assertNotIn("model_auto_compact_token_limit=114688", command)

    def test_client_cleanup_requests_remote_stop(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        runtime = remote.RemoteRuntime("gpu-rig", 18111, model, profile)

        class Input:
            def __init__(self) -> None:
                self.value = ""
                self.closed = False

            def write(self, value: str) -> None:
                self.value += value

            def flush(self) -> None:
                pass

            def close(self) -> None:
                self.closed = True

        class Process:
            def __init__(self) -> None:
                self.stdin = Input()

            @staticmethod
            def poll():
                return None

            @staticmethod
            def wait(timeout):
                return 0

        process = Process()
        runtime.process = process  # type: ignore[assignment]
        runtime.cleanup()

        self.assertEqual(json.loads(process.stdin.value), {"op": "stop"})
        self.assertTrue(process.stdin.closed)

    def test_host_protocol_records_client_events_and_always_cleans_up(self) -> None:
        model = fixture_model()
        profile = catalog.find_profile(model, "balanced", "codex")
        runtime_instance = mock.Mock()
        runtime_instance.context_window = 65_536
        runtime_instance.auto_compact_token_limit = 53_248
        runtime_instance.truncation_limit = 49_972
        runtime_instance.run_id = "run-1"
        runtime_instance.run_log = Path("/tmp/run-1.jsonl")
        input_stream = io.StringIO(
            json.dumps(
                {
                    "op": "event",
                    "event": "frontend.started",
                    "data": {"frontend": "codex"},
                    "level": "info",
                }
            )
            + "\n"
            + json.dumps({"op": "stop"})
            + "\n"
        )
        with (
            mock.patch("marathon_app.remote.discover_models", return_value=[model]),
            mock.patch("marathon_app.remote.find_model", return_value=model),
            mock.patch("marathon_app.remote.find_profile", return_value=profile),
            mock.patch("marathon_app.remote.Runtime", return_value=runtime_instance),
            mock.patch("marathon_app.remote.sys.stdin", input_stream),
            mock.patch("marathon_app.remote._protocol_print"),
        ):
            code = remote.run_remote_host(model.id, profile.id)

        self.assertEqual(code, 0)
        runtime_instance.record.assert_called_with(
            "frontend.started",
            {"frontend": "codex", "client": "ssh"},
            level="info",
        )
        runtime_instance.cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
