"""Check the evaluator's ability to reject wrong work and invalid comparisons."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "scripts/evals"
sys.path.insert(0, str(EVALS))
import reliability
from reliability_contracts import cases as additional_cases
from reliability_lifecycle import fixture


def trusted_fixture_oracle(project, source, timeout=30):
    # Only source fixtures written by these tests run here. Real model-generated
    # projects use reliability.oracle's separate filesystem/network namespace.
    result = subprocess.run([sys.executable, "-I", "-c",
                             "import sys; sys.path.insert(0, '.')\n" + source],
                            cwd=project, capture_output=True, text=True, timeout=timeout)
    return {"passed": result.returncode == 0, "exit_code": result.returncode,
            "output": result.stdout + result.stderr}


class ReliabilityGradingTests(unittest.TestCase):
    def test_old_ready_title_does_not_override_current_work(self):
        ready = "\x1b]0;Marathon | Ready | fixture\x07"
        working = "\x1b]0;Marathon | Working | fixture\x07"
        self.assertFalse(reliability.terminal_ready(ready + working))
        self.assertTrue(reliability.terminal_ready(ready + working + ready))

    def test_live_inference_requires_opt_in(self):
        result = subprocess.run([str(ROOT / "bin/marathon"), "eval", "reliability"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("pass --run-gpu", result.stderr)
        self.assertNotIn("Evidence:", result.stdout)

    def test_seeded_bugs_fail_hidden_contracts(self):
        cases = reliability.coding_cases() | additional_cases()
        with tempfile.TemporaryDirectory() as folder:
            for name, case in cases.items():
                if not case["oracle"]:
                    continue
                with self.subTest(case=name):
                    project = Path(folder) / name
                    reliability.prepare(project, case["files"])
                    self.assertFalse(trusted_fixture_oracle(project, case["oracle"])["passed"])

    def test_passing_claim_and_modified_public_test_cannot_hide_wrong_code(self):
        case = reliability.coding_cases()["retry"]
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            reliability.prepare(project, case["files"])
            (project / "test_retry.py").write_text("# all tests pass now\n")
            with patch.object(reliability, "oracle", trusted_fixture_oracle):
                checks, _ = reliability.check_case(project, case, "All tests passed", [])
            self.assertFalse(checks["independent_contract"])
            self.assertFalse(checks["preserved:test_retry.py"])

    def test_correct_fix_passes_and_protected_edit_still_fails(self):
        case = reliability.coding_cases()["recovery"]
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            reliability.prepare(project, case["files"])
            (project / "words.py").write_text(
                "from collections import Counter\ndef counts(text):\n    return dict(Counter(text.casefold().split()))\n")
            with patch.object(reliability, "oracle", trusted_fixture_oracle):
                checks, _ = reliability.check_case(project, case, "Fixed", [])
                self.assertTrue(all(checks.values()))
                (project / "project.json").write_text("{}")
                checks, _ = reliability.check_case(project, case, "Fixed", [])
                self.assertFalse(all(checks.values()))

    def test_protected_symlink_outside_project_is_not_read(self):
        case = {"files": {"protected.txt": "keep"}, "protected": ["protected.txt"], "oracle": ""}
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            reliability.prepare(project, case["files"])
            outside = Path(folder) / "canary"
            outside.write_text("keep")
            (project / "protected.txt").unlink()
            (project / "protected.txt").symlink_to(outside)
            with patch.object(reliability, "oracle", return_value={"passed": True}):
                checks, _ = reliability.check_case(project, case, "", [])
            self.assertFalse(checks["preserved:protected.txt"])

    def test_lifecycle_checker_requires_correction_and_cli_agreement(self):
        case = fixture("compaction", 0)
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            reliability.prepare(project, case["files"])
            # Extract the independent expected literal from the checker only in
            # this evaluator test; it is never copied into live model projects.
            import ast
            expected = ast.literal_eval(next(n.value for n in ast.parse(case["oracle"]).body
                                            if isinstance(n, ast.Assign) and n.targets[0].id == "expected"))
            (project / "release.py").write_text("def manifest():\n    return " + repr(expected) + "\n")
            (project / "cli.py").write_text("import json\nfrom release import manifest\nprint(json.dumps(manifest()))\n")
            self.assertTrue(trusted_fixture_oracle(project, case["oracle"])["passed"])
            expected["owner"] = "Ivo"
            (project / "release.py").write_text("def manifest():\n    return " + repr(expected) + "\n")
            self.assertFalse(trusted_fixture_oracle(project, case["oracle"])["passed"])

    def test_hidden_checker_sandbox_cannot_read_outside_canary(self):
        import shutil
        if not shutil.which("bwrap"):
            self.skipTest("bubblewrap not installed")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = root / "project"
            project.mkdir()
            canary = root / "outside-canary"
            canary.write_text("TEST CANARY")
            probe = reliability.oracle(project, "print('ready')")
            if not probe["passed"]:
                self.skipTest("bubblewrap unavailable: " + probe["output"][:200])
            result = reliability.oracle(project, "from pathlib import Path\nassert not Path(" + repr(str(canary)) + ").exists()\n")
            self.assertTrue(result["passed"], result["output"])
            early_exit = reliability.oracle(project, "raise SystemExit(0)")
            self.assertFalse(early_exit["passed"])
            self.assertIn("did not reach all assertions", early_exit["output"])
            mutation = reliability.oracle(project, "from pathlib import Path\nPath('checker-side-effect').write_text('test')")
            self.assertTrue(mutation["passed"], mutation["output"])
            self.assertFalse((project / "checker-side-effect").exists())

    def test_release_archive_provenance_needs_no_git_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "marathon_app").mkdir()
            source = root / "marathon_app/example.py"
            source.write_text("before\n")
            revision, before = reliability.source_provenance(root)
            self.assertIsNone(revision)
            source.write_text("after\n")
            _, after = reliability.source_provenance(root)
            self.assertNotEqual(before, after)


class ReliabilityComparisonTests(unittest.TestCase):
    def report(self):
        return {"manifest": {"model": "fixture", "profile": "test", "cases": ["a", "b"],
                             "repeats": 1, "timeout": 30, "evaluation_sha256": "fixed"},
                "results": [{"case": name, "repeat": 0, "passed": True, "status": "passed",
                             "interventions": 0, "task_seconds": 10, "elapsed_seconds": 11}
                            for name in ("a", "b")]}

    def test_faster_failure_remains_a_regression(self):
        baseline, candidate = self.report(), self.report()
        candidate["results"][0].update(passed=False, status="timeout", task_seconds=1)
        candidate["results"][1]["task_seconds"] = 5
        result = reliability.compare_reports(baseline, candidate)
        self.assertEqual(result["regressions"], [["a", 0]])
        self.assertEqual(result["candidate_success_rate"], .5)
        self.assertEqual(result["matched_successes"], 1)
        self.assertEqual(result["median_paired_seconds_saved"], 5)

    def test_incomplete_duplicate_and_incompatible_reports_rejected(self):
        baseline = self.report()
        mutations = [lambda x: x["results"].pop(),
                     lambda x: x["results"].append(copy.deepcopy(x["results"][0])),
                     lambda x: x["manifest"].update(evaluation_sha256="different"),
                     lambda x: x["manifest"].update(model="different"),
                     lambda x: x["results"][0].update(status="running")]
        for mutate in mutations:
            candidate = self.report()
            mutate(candidate)
            with self.assertRaises(ValueError):
                reliability.compare_reports(baseline, candidate)
        for report in (baseline, candidate):
            report["manifest"].pop("evaluation_sha256", None)
        with self.assertRaises(ValueError):
            reliability.compare_reports(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
