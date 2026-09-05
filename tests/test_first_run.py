"""First-run regressions: exercise menu construction and real setup scripts."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from rich.console import Console

from marathon_app import catalog, frontends, runtime, runtime_setup, ui
from marathon_app.model_library import HuggingFaceGguf

ROOT = Path(__file__).resolve().parents[1]


class FirstRunTests(unittest.TestCase):
    def test_cancelling_first_run_exits_without_a_traceback(self):
        with (
            mock.patch.object(ui, "run_setup_dashboard", side_effect=KeyboardInterrupt),
            mock.patch.object(sys, "argv", ["marathon", "setup"]),
            self.assertRaises(SystemExit) as stopped,
        ):
            entrypoint = ROOT / "marathon_app/__main__.py"
            exec(compile(entrypoint.read_text(), str(entrypoint), "exec"),
                 {"__name__": "__main__", "__package__": "marathon_app"})
        self.assertEqual(stopped.exception.code, 130)

    def test_download_menu_constructs_choices_and_downloads_selected_quant(self):
        files = [HuggingFaceGguf("example/model", "revision", f"Model-{q}.gguf", 123, q)
                 for q in ("Q4_K_M", "Q8_0")]
        with (
            mock.patch.object(ui, "list_huggingface_ggufs", return_value=files),
            mock.patch.object(ui, "_arrow_menu", return_value=1) as menu,
            mock.patch.object(ui, "download_huggingface_gguf", return_value=Path("/models/selected.gguf")) as download,
        ):
            result = ui._download_gguf(Console(file=io.StringIO()), "example/model")
        self.assertEqual([(item.label, item.value) for item in menu.call_args.args[3]],
                         [("Q4_K_M", "0"), ("Q8_0", "1")])
        self.assertEqual(download.call_args.args[0], files[1])
        self.assertEqual(result, Path("/models/selected.gguf"))

    def test_cancel_download_menu_does_not_download(self):
        file = HuggingFaceGguf("example/model", "revision", "Model-Q4_K_M.gguf", 123, "Q4_K_M")
        with (
            mock.patch.object(ui, "list_huggingface_ggufs", return_value=[file]),
            mock.patch.object(ui, "_arrow_menu", return_value=None),
            mock.patch.object(ui, "download_huggingface_gguf") as download,
        ):
            self.assertIsNone(ui._download_gguf(Console(file=io.StringIO()), "example/model"))
        download.assert_not_called()

    def test_doctor_checks_selected_backend_not_every_optional_profile(self):
        # Exercise the exact inventory program embedded in the doctor command.
        script = (ROOT / "scripts/ops/doctor.sh").read_text()
        inventory = script.split('backend_inventory="$', 1)[1].split("\n", 1)[1].split("\nPY\n", 1)[0]
        family = next(f for f in catalog.families() if f.id == "qwen3.8-27b")
        model = catalog.Model("test", "Test model", Path("/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"), 1, family, "IQ4_XS")
        for remembered, expected in [({}, "upstream"), ({"model": "test", "profile": "qwen38-iq4-xs-196k"}, "qwen38")]:
            output = io.StringIO()
            with (
                mock.patch.object(catalog, "discover_models", return_value=[model]),
                mock.patch.object(runtime, "load_selection", return_value=remembered),
                contextlib.redirect_stdout(output),
            ):
                exec(compile(inventory, "doctor-backend-inventory", "exec"), {})
            backends = [line.split("|")[1] for line in output.getvalue().splitlines() if line.startswith("backend|")]
            self.assertEqual(backends, [expected])

    def test_general_download_checks_prerequisites_before_using_network(self):
        with (
            mock.patch.object(ui, "discover_models", return_value=[]),
            mock.patch.object(ui, "eligible_gpus", return_value=()),
            mock.patch.object(ui, "_arrow_menu", side_effect=[0, 3]),
            mock.patch.object(ui, "_setup_prerequisites", return_value=False),
            mock.patch.object(ui, "_download_gguf") as download,
        ):
            self.assertIsNone(ui._setup_model_selection(Console(file=io.StringIO())))
        download.assert_not_called()

    def test_stock_codex_is_not_treated_as_an_installed_hardened_frontend(self):
        with (
            mock.patch.object(frontends, "_codex_binary", return_value="codex"),
            mock.patch.object(frontends.shutil, "which", return_value="/bin/codex"),
            mock.patch.object(frontends, "_codex_features", return_value=set()),
        ):
            self.assertFalse(frontends.hardened_codex_available())

    def test_build_preflight_reports_system_libraries_but_not_developer_test_tools(self):
        tools = {"git", "cargo", "cmake", "c++", "pkg-config", "nvcc"}
        with (
            mock.patch.object(runtime_setup.shutil, "which", side_effect=lambda name: f"/bin/{name}" if name in tools else None),
            mock.patch.object(runtime_setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)),
        ):
            missing = runtime_setup.missing_build_tools(runtime_installed=False, frontend_installed=False, cuda=True)
        self.assertEqual(missing, ("OpenSSL development libraries (libssl-dev on Ubuntu)",))

    def test_linux_sandbox_is_required_even_with_binaries_already_installed(self):
        with (
            mock.patch.object(runtime_setup.sys, "platform", "linux"),
            mock.patch.object(runtime_setup.shutil, "which", return_value=None),
        ):
            self.assertEqual(runtime_setup.missing_runtime_tools("codex"), ("bubblewrap (Linux sandbox)",))
            self.assertEqual(runtime_setup.missing_runtime_tools("direct"), ())
            self.assertEqual(runtime_setup.missing_runtime_tools("hermes"), ())
        with mock.patch.object(runtime_setup.sys, "platform", "darwin"):
            self.assertEqual(runtime_setup.missing_runtime_tools("codex"), ())

    def test_launcher_install_opt_out_preserves_an_existing_file(self):
        with tempfile.TemporaryDirectory(prefix="marathon-install-test-") as directory:
            target = Path(directory) / "marathon"
            target.write_text("user-owned command\n")
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/ops/install_cli.sh")], capture_output=True, text=True,
                env={**os.environ, "MARATHON_INSTALL_BIN_DIR": directory, "MARATHON_CONFIGURE_SHELL": "0"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_text(), "user-owned command\n")

    @unittest.skipUnless(os.environ.get("MARATHON_NETWORK_TESTS") == "1", "opt-in network bootstrap test")
    def test_incomplete_python_environment_recovers_and_retry_preserves_files(self):
        with tempfile.TemporaryDirectory(prefix="marathon-bootstrap-test-") as directory:
            root = Path(directory)
            scripts = root / "scripts/ops"
            scripts.mkdir(parents=True)
            shutil.copy2(ROOT / "scripts/ops/setup_python_env.sh", scripts)
            (root / "scripts/requirements.txt").write_text("")
            venv = root / ".marathon/venv"
            venv.mkdir(parents=True)
            sentinel = venv / "preserve-me"
            sentinel.write_text("user state\n")
            command = ["bash", str(scripts / "setup_python_env.sh")]
            first = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            second = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(sentinel.read_text(), "user state\n")
            self.assertTrue((venv / ".marathon-requirements").is_file())


if __name__ == "__main__":
    unittest.main()
