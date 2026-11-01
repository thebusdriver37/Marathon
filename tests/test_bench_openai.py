import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmarks/bench_openai.py"
SPEC = importlib.util.spec_from_file_location("bench_openai", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def completion(text="answer", finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 11},
            "timings": {"prompt_n": 100, "prompt_ms": 500,
                        "predicted_n": 11, "predicted_ms": 200,
                        "cache_n": 0, "draft_n": 20, "draft_n_accepted": 8}}


class BenchmarkTests(unittest.TestCase):
    def invoke(self, arguments, result=None, side_effect=None):
        output = io.StringIO()
        with mock.patch.object(bench, "post_json", return_value=result or completion(),
                               side_effect=side_effect) as post, contextlib.redirect_stdout(output):
            code = bench.main(["--prompt", "Synthetic fixture", *arguments])
        return code, json.loads(output.getvalue()), post

    def test_plan_never_sends_requests_or_reads_credentials_or_writes_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.json"
            code, report, post = self.invoke(["--api-key-env", "ABSENT_FIXTURE_KEY",
                                              "--save-json", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "planned")
            post.assert_not_called()
            self.assertFalse(path.exists())
            self.assertNotIn("Synthetic fixture", json.dumps(report))

    def test_phase_metrics_use_native_counts_without_request_latency(self):
        row = bench.metrics(completion(), 2)
        self.assertEqual(row["request_tps"], 5.5)
        self.assertEqual(row["backend_decode_tps"], 50)
        self.assertEqual(row["backend_prefill_tps"], 200)
        self.assertEqual(row["draft_acceptance"], .4)

    def test_absent_counters_are_unknown_not_word_counts(self):
        response = completion("many words do not represent measured tokens")
        response.pop("timings")
        response.pop("usage")
        row = bench.metrics(response, 2)
        for key in ("request_tps", "backend_decode_tps", "completion_tokens", "draft_acceptance"):
            self.assertIsNone(row[key])

    def test_invalid_native_timing_and_acceptance_are_not_measurements(self):
        for value in (None, -1, 0, True, float("nan"), float("inf"), "200"):
            with self.subTest(value=value):
                response = completion()
                response["timings"].update(predicted_ms=value, draft_n_accepted=21)
                row = bench.metrics(response, 2)
                self.assertIsNone(row["backend_decode_tps"])
                self.assertIsNone(row["draft_acceptance"])

    def test_first_token_only_has_no_decode_rate_without_decode_time(self):
        response = completion()
        response["timings"].update(predicted_n=1, predicted_ms=0)
        self.assertIsNone(bench.metrics(response, 1)["backend_decode_tps"])

    def test_pair_alternates_order_and_preserves_request_parameters(self):
        code, report, post = self.invoke([
            "--run-inference", "--url", "http://baseline.invalid/v1/chat/completions",
            "--candidate-url", "http://candidate.invalid/v1/chat/completions",
            "--baseline-id", "build-a", "--candidate-id", "build-b",
            "--candidate-model", "candidate-alias", "--runs", "3", "--warmup-runs", "0",
            "--require-identical"])
        self.assertEqual(code, 0)
        self.assertEqual([c.args[0].split("//")[1].split(".")[0] for c in post.call_args_list],
                         ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"])
        bodies = [{k: v for k, v in c.args[1].items() if k != "model"} for c in post.call_args_list]
        self.assertTrue(all(body == bodies[0] for body in bodies))
        self.assertFalse(bodies[0]["cache_prompt"])
        self.assertTrue(report["comparison"]["all_outputs_identical"])
        self.assertNotIn("answer", json.dumps(report))

    def test_failures_and_truncation_remain_in_evidence_and_fail_gate(self):
        failure = urllib.error.HTTPError("http://fixture.invalid", 503, "sensitive detail", {}, None)
        code, report, _ = self.invoke(["--run-inference", "--runs", "3", "--warmup-runs", "0"],
                                     side_effect=[completion(), failure, completion(finish="length")])
        self.assertEqual(code, 1)
        self.assertEqual(len(report["rows"]), 3)
        self.assertEqual(report["rows"][1]["http_status"], 503)
        self.assertFalse(report["rows"][2]["complete"])
        self.assertEqual(report["summary"]["baseline"]["successful_requests"], 2)
        self.assertIsNone(report["summary"]["baseline"]["latency_s"]["mean"])
        self.assertNotIn("sensitive detail", json.dumps(report))

    def test_changed_output_fails_exact_gate_and_is_not_a_speed_win(self):
        code, report, _ = self.invoke([
            "--run-inference", "--candidate-url", "http://candidate.invalid",
            "--baseline-id", "a", "--candidate-id", "b",
            "--runs", "1", "--warmup-runs", "0", "--require-identical"],
            side_effect=[completion(), completion("changed")])
        self.assertEqual(code, 1)
        self.assertIsNone(report["comparison"]["paired_latency_delta_s_median"])

    def test_different_prompt_or_cache_counts_are_not_a_matched_speed_comparison(self):
        for field in ("prompt_tokens", "cached_tokens"):
            response = completion()
            if field == "prompt_tokens":
                response["usage"][field] = 200
            else:
                response["timings"]["cache_n"] = 50
            with self.subTest(field=field):
                code, report, _ = self.invoke([
                    "--run-inference", "--candidate-url", "http://candidate.invalid",
                    "--baseline-id", "a", "--candidate-id", "b",
                    "--runs", "1", "--warmup-runs", "0"], side_effect=[completion(), response])
                self.assertEqual(code, 1)
                self.assertIsNone(report["comparison"]["paired_latency_delta_s_median"])

    def test_explicit_long_prompt_file_is_hashed_without_printing_content(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.txt"
            path.write_text("Public synthetic archive.\n" * 10000)
            output = io.StringIO()
            with mock.patch.object(bench, "post_json") as post, contextlib.redirect_stdout(output):
                self.assertEqual(bench.main(["--prompt-file", str(path)]), 0)
            post.assert_not_called()
            self.assertNotIn("Public synthetic", output.getvalue())

    def test_tool_ids_are_ignored_but_arguments_and_reasoning_are_not(self):
        response = completion()
        response["choices"][0]["message"].update(reasoning_content="thought",
            tool_calls=[{"id": "random-a", "type": "function",
                         "function": {"name": "inspect", "arguments": '{"x":1}'}}])
        other = copy.deepcopy(response)
        other["choices"][0]["message"]["tool_calls"][0]["id"] = "random-b"
        self.assertEqual(bench.metrics(response, 1)["output_sha256"], bench.metrics(other, 1)["output_sha256"])
        other["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"x":2}'
        self.assertNotEqual(bench.metrics(response, 1)["output_sha256"], bench.metrics(other, 1)["output_sha256"])

    def test_partial_timing_coverage_does_not_average_only_the_fast_rows(self):
        missing = completion()
        missing.pop("timings")
        code, report, _ = self.invoke(["--run-inference", "--runs", "2", "--warmup-runs", "0"],
                                     side_effect=[completion(), missing])
        self.assertEqual(code, 0)
        summary = report["summary"]["baseline"]["backend_decode_tps"]
        self.assertEqual(summary, {"coverage": 1, "mean": None, "median": None})

    def test_evidence_file_is_not_overwritten_and_key_is_not_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.json"
            with mock.patch.dict(bench.os.environ, {"FIXTURE_KEY": "synthetic-secret"}):
                code, report, post = self.invoke(["--run-inference", "--runs", "1", "--warmup-runs", "0",
                                                   "--api-key-env", "FIXTURE_KEY", "--save-json", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(post.call_args.args[3], "synthetic-secret")
            self.assertNotIn("synthetic-secret", path.read_text())
            with mock.patch.object(bench, "post_json") as network, self.assertRaises(FileExistsError):
                bench.main(["--prompt", "fixture", "--run-inference", "--save-json", str(path)])
            network.assert_not_called()

    def test_warm_mode_priming_is_excluded_from_measured_rows(self):
        code, report, post = self.invoke(["--run-inference", "--runs", "1", "--cache-mode", "warm"])
        self.assertEqual(code, 0)
        self.assertEqual(len(report["warmups"]), 1)
        self.assertEqual(len(report["rows"]), 1)
        self.assertTrue(all(c.args[1]["cache_prompt"] for c in post.call_args_list))

    def test_invalid_run_configuration_fails_before_network(self):
        for args in (["--runs", "0"], ["--timeout", "nan"], ["--temperature", "inf"],
                     ["--cache-mode", "warm", "--warmup-runs", "0"], ["--require-identical"],
                     ["--url", "http://user:secret@fixture.invalid"],
                     ["--url", "http://fixture.invalid/?api_key=secret"]):
            with self.subTest(args=args), mock.patch.object(bench, "post_json") as post:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    bench.main(["--prompt", "fixture", "--run-inference", *args])
                post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
