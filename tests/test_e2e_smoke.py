import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LiveSmokeGuardTests(unittest.TestCase):
    def test_launcher_requires_explicit_live_inference(self):
        result = subprocess.run(
            [str(ROOT / "bin/marathon"), "eval", "smoke"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("pass --run-gpu", result.stderr)
        self.assertNotIn("Evidence:", result.stdout)

    def test_help_exposes_capacity_and_evidence_options(self):
        result = subprocess.run(
            [str(ROOT / "bin/marathon"), "eval", "smoke", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--workers", result.stdout)
        self.assertIn("--output-dir", result.stdout)
