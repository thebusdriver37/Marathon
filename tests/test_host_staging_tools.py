import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("host_staging_gpu", ROOT / "scripts/evals/host_staging_gpu.py")
summary = load("summarize_host_staging", ROOT / "scripts/evals/summarize_host_staging.py")


class HostStagingGuards(unittest.TestCase):
    def test_default_plan_never_inspects_hardware_or_configuration(self):
        with mock.patch.object(runtime.sys, "argv", ["host_staging_gpu", "--output", "/unused",
                                                     "--build", "/unused"]), \
             mock.patch.object(runtime, "acquire_pool_worker") as acquire, \
             mock.patch.object(runtime.subprocess, "check_output") as query, \
             mock.patch.object(runtime.Path, "read_bytes") as read, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            runtime.main()
        self.assertEqual(json.loads(output.getvalue())["status"], "planned")
        acquire.assert_not_called()
        query.assert_not_called()
        read.assert_not_called()

    def test_occupied_gpu_is_rejected_before_any_container_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = runtime.Worker(Path(folder) / "worker", 3, b"", "candidate", Path(folder))
            with mock.patch.object(runtime.subprocess, "check_output", return_value="24000\n"), \
                 mock.patch.object(runtime.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "occupied"):
                    worker.launch()
                worker.close()
            run.assert_not_called()

    def test_existing_worker_is_not_replaced_even_if_gpu_looks_idle(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = runtime.Worker(Path(folder) / "worker", 3, b"", "candidate", Path(folder))
            with mock.patch.object(runtime.subprocess, "check_output", side_effect=["1\n", worker.name + "\n"]), \
                 mock.patch.object(runtime.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "refusing duplicate"):
                    worker.launch()
            run.assert_not_called()

    def test_request_overrides_preserve_explicit_cache_and_output_settings(self):
        worker = mock.Mock()
        runtime.complete(worker, [1, 2, 3], n_predict=32, cache_prompt=False)
        body = worker.http.call_args.args[1]
        self.assertEqual(body["n_predict"], 32)
        self.assertFalse(body["cache_prompt"])
        self.assertTrue(body["return_tokens"])

    def test_timing_exclusion_keeps_output_validation_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for variant, speed in (("control", 50), ("candidate", 100)):
                window = root / ("window-0-" + variant)
                window.mkdir()
                row = {"depth": 8192, "cache": "cold", "repeat": -1, "tokens_sha256": "same",
                       "wall_s": 2, "timings": {"predicted_per_second": speed}}
                (window / "screen.json").write_text(json.dumps([row]))
            (root / "timing_exclusions.json").write_text(json.dumps([
                {"window": "window-0-candidate", "depth": 8192, "cache": "cold", "repeat": -1,
                 "reason": "fixture interference"}]))
            result = summary.summarize(root)
            cold = next(c for c in result["comparisons"] if c["cache"] == "cold")
            self.assertTrue(cold["tokens_identical"])
            self.assertIsNone(cold["decode_change_pct"])
            self.assertEqual(len(result["windows"][0]["screen"]), 1)

    def test_frozen_source_is_independent_of_later_workspace_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "project"
            router = root / "scripts/routers/codex_local_router.py"
            router.parent.mkdir(parents=True)
            router.write_text("original router\n")
            prompt = root / "codex/codex-rs/models-manager/prompt.md"
            prompt.parent.mkdir(parents=True)
            prompt.write_text("public fixture prompt\n")
            with mock.patch.object(runtime, "ROOT", root), \
                 mock.patch.object(runtime.subprocess, "check_output", return_value="scripts/routers/codex_local_router.py\n"):
                frozen = runtime.freeze_source(Path(folder) / "frozen")
            router.write_text("concurrent edit\n")
            self.assertEqual((frozen / "scripts/routers/codex_local_router.py").read_text(), "original router\n")
            self.assertEqual((frozen / "codex/codex-rs/models-manager/prompt.md").read_text(), "public fixture prompt\n")
            manifest = json.loads((frozen / "source-manifest.json").read_text())
            self.assertIn("scripts/routers/codex_local_router.py", manifest["sha256"])


if __name__ == "__main__":
    unittest.main()
