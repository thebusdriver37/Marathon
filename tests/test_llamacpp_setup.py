from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LlamaPatchSetupTests(unittest.TestCase):
    def test_codex_patch_setup_is_repeatable_and_preserves_upstream(self):
        with tempfile.TemporaryDirectory(prefix="marathon-codex-patch-test-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "value").write_text("original\n")
            subprocess.run(["git", "-C", str(source), "add", "value"], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Fixture", "-c",
                            "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
            patches = root / "patches"
            patches.mkdir()
            (patches / "001.patch").write_text(
                "diff --git a/value b/value\n--- a/value\n+++ b/value\n"
                "@@ -1 +1 @@\n-original\n+patched\n"
            )
            target = root / "patched"
            env = {key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "MARATHON_"))}
            env.update({"MARATHON_CODEX_DIR": str(source), "MARATHON_PATCH_DIR": str(patches),
                        "MARATHON_PATCHED_CODEX_DIR": str(target), "GIT_CONFIG_GLOBAL": os.devnull,
                        "GIT_CONFIG_NOSYSTEM": "1"})
            command = ["bash", str(ROOT / "scripts/apply_codex_patches.sh")]
            for _ in range(2):
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((source / "value").read_text(), "original\n")
            self.assertEqual((target / "value").read_text(), "patched\n")
            (target / "value").write_text("user edit\n")
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((target / "value").read_text(), "user edit\n")

    def test_real_setup_is_repeatable_and_preserves_local_edits(self):
        # Run the actual patch command against an isolated repository, with two
        # overlapping patches. No user source, backend, or network is involved.
        with tempfile.TemporaryDirectory(prefix="marathon-patch-test-") as folder:
            root = Path(folder)
            source = root / "upstream"
            source.mkdir()
            env = {key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "MARATHON_"))}
            env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(source), *args], env=env,
                    text=True, capture_output=True, check=True,
                )

            git("init", "-q")
            (source / "value").write_text("original\n")
            git("add", "value")
            git("-c", "user.name=Test Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
            patch_dir = root / "patches"
            patch_dir.mkdir()
            for name, old, new in [("001", "original", "first"), ("002", "first", "second")]:
                (patch_dir / f"{name}.patch").write_text(
                    "diff --git a/value b/value\n--- a/value\n+++ b/value\n"
                    f"@@ -1 +1 @@\n-{old}\n+{new}\n"
                )
            patched = root / "patched"
            env.update({
                "MARATHON_LLAMACPP_DIR": str(source),
                "MARATHON_LLAMACPP_PATCH_DIR": str(patch_dir),
                "MARATHON_PATCHED_LLAMACPP_DIR": str(patched),
                "MARATHON_LLAMACPP_VARIANT": "upstream",
            })
            command = ["bash", str(ROOT / "scripts/apply_llamacpp_patches.sh")]
            first = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual((patched / "value").read_text(), "second\n")
            self.assertEqual((source / "value").read_text(), "original\n")
            second = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already applied", second.stdout)
            (patched / "value").write_text("user changes\n")
            third = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertNotEqual(third.returncode, 0)
            self.assertIn("preserving modified worktree", third.stderr)
            self.assertEqual((patched / "value").read_text(), "user changes\n")
            self.assertEqual(git("status", "--porcelain").stdout, "")

    def test_upstream_does_not_implicitly_apply_optional_qwen_patches(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "MARATHON_"))}
        env["MARATHON_LLAMACPP_VARIANT"] = "upstream"
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/apply_llamacpp_patches.sh")],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no llama.cpp patches selected", result.stdout)

    def test_unknown_runtime_fails_before_fetch_or_build(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/ops/setup_llamacpp.sh"), "not-a-runtime"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown runtime", result.stderr)


if __name__ == "__main__":
    unittest.main()
