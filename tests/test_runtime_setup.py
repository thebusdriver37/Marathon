from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rich.console import Console

from marathon_app import catalog, runtime_setup, ui


class RuntimeSetupTests(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog.load_catalog(catalog.ROOT_DIR / "config/runtime_catalog.toml")
        self.family = next(f for f in catalog.families(self.catalog) if f.id == "qwen3.8-27b")
        self.profile = next(p for p in self.family.profiles if p.bundle)
        self.bundle = runtime_setup.model_bundle(self.profile.bundle)
        self.target = next(a for a in self.bundle["files"] if a["role"] == "model")
        self.gpu = runtime_setup.AvailableGpu(2, "GPU-test", "NVIDIA GeForce RTX 3090", 24000)

    def test_general_setup_remains_the_default_on_other_hardware(self):
        with mock.patch.object(catalog, "load_catalog", return_value=self.catalog):
            model = catalog.Model("other", "Other model", Path("/models/ordinary.gguf"), 0, self.family, "Q4_K_M")
            profiles = catalog.profiles_for_model(model)
            self.assertEqual(catalog.find_profile(model, None).id, "auto")
            self.assertNotIn(self.profile, profiles)
            self.assertIs(runtime_setup.prepare_bundle_profile(model, profiles[0]), profiles[0])

    def test_every_shipped_profile_references_a_defined_backend(self):
        names = {b["id"] for b in self.catalog["backends"]}
        for family in self.catalog["families"]:
            self.assertIn(family["backend"], names)
            for profile in family["profiles"]:
                self.assertIn(profile.get("backend", family["backend"]), names)

    def test_gpu_selection_rejects_other_cards_busy_cards_and_visibility_exclusions(self):
        output = "\n".join([
            "0, GPU-a, NVIDIA GeForce RTX 4090, 24000",
            "1, GPU-b, NVIDIA GeForce RTX 3090, 22000",
            "2, GPU-c, NVIDIA GeForce RTX 3090 Ti, 24000",
            "3, GPU-d, NVIDIA GeForce RTX 3090, 24000",
            "4, GPU-e, NVIDIA GeForce RTX 3090, N/A",
        ])
        with (
            mock.patch.object(runtime_setup.sys, "platform", "linux"),
            mock.patch.object(runtime_setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output)),
            mock.patch.dict(runtime_setup.os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-c"}),
        ):
            self.assertEqual(runtime_setup.eligible_gpus(self.profile.bundle), (
                runtime_setup.AvailableGpu(2, "GPU-c", "NVIDIA GeForce RTX 3090 Ti", 24000),
            ))

    def test_absent_driver_does_not_break_general_setup(self):
        with mock.patch.object(runtime_setup.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(runtime_setup.eligible_gpus(self.profile.bundle), ())

    def test_download_uses_all_exact_revisions_and_verifies_real_bytes(self):
        bundle = {**self.bundle, "files": [
            {**asset, "size_bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            for asset in self.bundle["files"]
        ]}
        with tempfile.TemporaryDirectory() as folder:
            def download(**kwargs):
                path = kwargs["local_dir"] / kwargs["filename"]
                path.write_bytes(b"abc")
                return str(path)

            downloader = mock.Mock(side_effect=download)
            with (
                mock.patch.object(runtime_setup, "model_bundle", return_value=bundle),
                mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=downloader)}),
            ):
                target = runtime_setup.download_bundle(self.profile.bundle, Path(folder))
            self.assertEqual(target.name, self.target["filename"])
            self.assertTrue(target.with_suffix(".gguf.marathon.json").exists())
            self.assertEqual(
                [(call.kwargs["repo_id"], call.kwargs["revision"]) for call in downloader.call_args_list],
                [(a["repository"], a["revision"]) for a in bundle["files"]],
            )

    def test_corrupt_download_stops_before_other_assets_or_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / self.target["filename"]
            bad.write_bytes(b"wrong model")
            downloader = mock.Mock(return_value=str(bad))
            with mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=downloader)}):
                with self.assertRaisesRegex(ValueError, "Checksum verification failed"):
                    runtime_setup.download_bundle(self.profile.bundle, Path(folder))
            downloader.assert_called_once()
            self.assertFalse(bad.with_suffix(".gguf.marathon.json").exists())
            self.assertEqual(bad.read_bytes(), b"wrong model")

    def test_insufficient_disk_space_prevents_download(self):
        with tempfile.TemporaryDirectory() as folder:
            downloader = mock.Mock()
            with (
                mock.patch.object(runtime_setup.shutil, "disk_usage", return_value=types.SimpleNamespace(free=1)),
                mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=downloader)}),
            ):
                with self.assertRaisesRegex(ValueError, "GiB free"):
                    runtime_setup.download_bundle(self.profile.bundle, Path(folder))
            downloader.assert_not_called()

    def test_missing_bundle_file_is_rejected_before_gpu_selection(self):
        model = catalog.Model("test", "Test", Path("/missing") / self.target["filename"], 0, self.family, "IQ4_XS")
        with mock.patch.object(runtime_setup, "eligible_gpus") as detect:
            with self.assertRaisesRegex(ValueError, "missing or incomplete"):
                runtime_setup.prepare_bundle_profile(model, self.profile)
            detect.assert_not_called()

    def test_bundle_launch_uses_relocated_paths_and_one_available_gpu(self):
        bundle = {**self.bundle, "files": [{**a, "size_bytes": 3} for a in self.bundle["files"]]}
        with tempfile.TemporaryDirectory(prefix="marathon path with spaces ") as folder:
            root = Path(folder)
            for asset in bundle["files"]:
                (root / asset["filename"]).write_bytes(b"abc")
            projector = root / next(a["filename"] for a in bundle["files"] if a["role"] == "projector")
            model = catalog.Model("test", "Test", root / self.target["filename"], 3, self.family, "IQ4_XS", projector)
            with (
                mock.patch.object(runtime_setup, "model_bundle", return_value=bundle),
                mock.patch.object(runtime_setup, "eligible_gpus", return_value=(self.gpu,)),
            ):
                profile = runtime_setup.prepare_bundle_profile(model, self.profile)
                self.assertEqual(profile.gpus, (2,))
                with self.assertRaisesRegex(ValueError, "exactly one GPU"):
                    runtime_setup.prepare_bundle_profile(model, replace(self.profile, gpus=(1, 2)))
                with self.assertRaisesRegex(ValueError, "supported, available GPU"):
                    runtime_setup.prepare_bundle_profile(model, replace(self.profile, gpus=(0,)))
            command = catalog.server_command(model, profile, catalog.Backend("fixture", "Fixture", Path("/portable/bin/llama-server")))
            draft = command[command.index("--spec-draft-model") + 1]
            self.assertEqual(draft, str(root / "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"))
            self.assertIn("--no-mmproj-offload", command)
            self.assertNotIn("{model_dir}", " ".join(command))
            self.assertNotIn("/home/deforest", " ".join(command))

    def test_first_run_bundle_flow_with_no_personal_catalog(self):
        model = catalog.Model("test", "Test", Path("/models") / self.target["filename"], 0, self.family, "IQ4_XS")
        console = Console(file=io.StringIO())
        with (
            mock.patch.object(catalog, "load_catalog", return_value=self.catalog),
            mock.patch.object(ui, "discover_models", side_effect=[[], [model]]),
            mock.patch.object(ui, "eligible_gpus", return_value=(self.gpu,)),
            mock.patch.object(ui, "missing_build_tools", return_value=()),
            mock.patch.object(ui, "missing_runtime_tools", return_value=()),
            mock.patch.object(ui, "_arrow_menu", return_value=0),
            mock.patch.object(ui, "_confirm_install", return_value=True),
            mock.patch.object(ui, "download_bundle", return_value=model.path) as download,
        ):
            selection = ui._setup_model_selection(console)
        self.assertEqual(selection.profile.id, self.profile.id)
        self.assertEqual(selection.model.path, model.path)
        self.assertTrue(selection.install_confirmed)
        download.assert_called_once()

    def test_tuned_backend_missing_binary_offers_its_own_installer(self):
        model = catalog.Model("test", "Test", Path("/models") / self.target["filename"], 0, self.family, "IQ4_XS")
        with (
            mock.patch.object(ui, "_setup_prerequisites", return_value=True),
            mock.patch.object(ui, "backend_for", side_effect=[ValueError("missing binary"), mock.Mock()]),
            mock.patch.object(ui, "_confirm_install", return_value=True),
            mock.patch.object(ui, "_run_install_command", return_value=True) as install,
        ):
            self.assertTrue(ui._ensure_local_tools(Console(file=io.StringIO()), ui.Selection(model, self.profile, "direct"), "direct"))
        self.assertEqual(install.call_args.args[2][-2:], ["setup-llama", "qwen38"])

    def test_missing_build_tools_are_reported_before_downloading(self):
        with (
            mock.patch.object(ui, "discover_models", return_value=[]),
            mock.patch.object(ui, "eligible_gpus", return_value=(self.gpu,)),
            mock.patch.object(ui, "_arrow_menu", side_effect=[0, 4]),
            mock.patch.object(ui, "missing_build_tools", return_value=("nvcc",)),
            mock.patch.object(ui.Prompt, "ask", return_value=""),
            mock.patch.object(ui, "download_bundle") as download,
        ):
            self.assertIsNone(ui._setup_model_selection(Console(file=io.StringIO())))
        download.assert_not_called()

    def test_bundle_approval_also_covers_missing_backend_and_frontend_builds(self):
        model = catalog.Model("test", "Test", Path("/models") / self.target["filename"], 0, self.family, "IQ4_XS")
        selection = ui.Selection(model, self.profile, "codex", install_confirmed=True)
        with (
            mock.patch.object(ui, "backend_for", side_effect=[ValueError("missing binary"), mock.Mock()]),
            mock.patch.object(ui, "hardened_codex_available", return_value=False),
            mock.patch.object(ui, "_setup_prerequisites", return_value=True),
            mock.patch.object(ui, "_confirm_install") as confirm,
            mock.patch.object(ui, "_run_install_command", return_value=True) as install,
        ):
            self.assertTrue(ui._ensure_local_tools(Console(file=io.StringIO()), selection))
        confirm.assert_not_called()
        self.assertEqual([call.args[2][-1] for call in install.call_args_list], ["qwen38", "build-codex"])


if __name__ == "__main__":
    unittest.main()
