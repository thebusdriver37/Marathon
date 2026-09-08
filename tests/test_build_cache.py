"""Exercise managed patch updates with real Git worktrees and Cargo caching."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BuildCacheTests(unittest.TestCase):
    def test_patch_edits_keep_unchanged_crates_fresh_and_preserve_user_edits(self):
        with tempfile.TemporaryDirectory(prefix="marathon-build-cache-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "MARATHON_", "CARGO_"))}
            env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", MARATHON_PATCH_UPDATE="1")

            def git(*args, cwd=source):
                return subprocess.run(["git", "-C", str(cwd), *args], env=env,
                                      check=True, capture_output=True, text=True).stdout

            git("init", "-q")
            (source / "Cargo.toml").write_text('[package]\nname="cache-app"\nversion="0.1.0"\nedition="2021"\n[dependencies]\ncache-dep={path="dep"}\n')
            (source / "src").mkdir()
            (source / "src/main.rs").write_text('fn main() { println!("{}", cache_dep::value() + 1); }\n')
            (source / "dep/src").mkdir(parents=True)
            (source / "dep/Cargo.toml").write_text('[package]\nname="cache-dep"\nversion="0.1.0"\nedition="2021"\n')
            (source / "dep/src/lib.rs").write_text('pub fn value() -> u32 { 1 }\n')
            (source / ".gitignore").write_text('Cargo.lock\ntarget/\n')
            git("add", ".")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
            patch = root / "change.patch"
            target = root / "build"

            def set_patch(value):
                patch.write_text('diff --git a/src/main.rs b/src/main.rs\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-fn main() { println!("{}", cache_dep::value() + 1); }\n+fn main() { println!("{}", cache_dep::value() + ' + str(value) + '); }\n')

            def apply():
                return subprocess.run(["bash", str(ROOT / "scripts/lib/apply_patch_stack.sh"),
                                       str(source), str(target), str(patch)], env=env,
                                      capture_output=True, text=True)

            def build():
                result = subprocess.run(["cargo", "build", "--offline", "--message-format=json"],
                                        cwd=target, env=env, capture_output=True, text=True, check=True)
                return {e["target"]["name"]: e["fresh"] for line in result.stdout.splitlines()
                        if (e := json.loads(line)).get("reason") == "compiler-artifact"}

            set_patch(2)
            self.assertEqual(apply().returncode, 0)
            unchanged = target / "dep/src/lib.rs"
            initial_stat = unchanged.stat()
            if shutil.which("cargo"):
                self.assertEqual(build(), {"cache_dep": False, "cache-app": False})
                self.assertEqual(build(), {"cache_dep": True, "cache-app": True})
            set_patch(3)
            result = apply()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(unchanged.stat().st_mtime_ns, initial_stat.st_mtime_ns)
            self.assertEqual(unchanged.stat().st_ino, initial_stat.st_ino)
            if shutil.which("cargo"):
                self.assertEqual(build(), {"cache_dep": True, "cache-app": False})
            self.assertIn('+ 1', (source / "src/main.rs").read_text())
            self.assertIn('+ 3', (target / "src/main.rs").read_text())
            set_patch(4)
            for staged in (False, True):
                with self.subTest(staged=staged):
                    (target / "src/main.rs").write_text("user edit\n")
                    if staged:
                        git("add", "src/main.rs", cwd=target)
                    result = apply()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("preserving modified worktree", result.stderr)
                    self.assertEqual((target / "src/main.rs").read_text(), "user edit\n")


if __name__ == "__main__":
    unittest.main()
