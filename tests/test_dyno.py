from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from marathon_app import catalog, dyno


def fixture_model(path: Path | None = None) -> catalog.Model:
    family = next(item for item in catalog.families() if item.id == "deepseek-v4-flash")
    return catalog.Model(
        id="deepseek-v4-flash-ud-iq2-m",
        display_name="DeepSeek V4 Flash UD-IQ2_M",
        path=path or Path("/tmp/dyno-test-model.gguf"),
        size_bytes=85 * 1024**3,
        family=family,
        quant="UD-IQ2_M",
    )


def trial(
    candidate_id: str,
    *,
    prompt: float,
    decode: float,
    context: int,
    power: float,
    cache: str = "q8_0",
) -> dyno.TrialResult:
    model = fixture_model()
    base = catalog.find_profile(model, "long-64k")
    profile = dyno._variant(base, candidate_id, candidate_id, cache=cache).profile
    candidate = dyno.Candidate(candidate_id, candidate_id, profile)
    return dyno.TrialResult(
        candidate=candidate,
        success=True,
        loaded_context=context,
        prompt_tps=prompt,
        decode_tps=decode,
        average_power_w=power,
        quality_pass=True,
    )


class CandidateTests(unittest.TestCase):
    def test_old_deepseek_tuned_profile_cannot_disable_context_checkpoints(self) -> None:
        model = fixture_model()
        base = catalog.find_profile(model, "long-64k")
        stale = replace(
            dyno._variant(
                base,
                "stale",
                "Stale",
                add_args=("--ctx-checkpoints", "0", "--poll", "0"),
            ).profile,
            flash_attention="on",
            tool_thinking_budget=None,
        )

        sanitized = dyno._sanitize_tuned_profile(model, stale)

        self.assertNotIn("--ctx-checkpoints", sanitized.extra_args)
        self.assertNotIn("--swa-full", sanitized.extra_args)
        self.assertEqual(sanitized.cache_k, "f16")
        self.assertEqual(sanitized.cache_v, "f16")
        self.assertEqual(sanitized.flash_attention, "off")
        self.assertIsNone(sanitized.tool_thinking_budget)
        self.assertIn("--poll", sanitized.extra_args)

    def test_context_objective_scales_context_without_unbounded_search(self) -> None:
        model = fixture_model()
        base = catalog.find_profile(model, "long-64k")

        candidates = dyno.candidate_profiles(model, base, "context")

        self.assertEqual(
            [item.profile.context for item in candidates],
            [65_536, 131_072, 262_144],
        )
        self.assertTrue(all(item.profile.cache_k == "q8_0" for item in candidates))

    def test_quality_objective_keeps_full_precision_cache(self) -> None:
        model = fixture_model()
        base = catalog.find_profile(model, "long-64k")

        candidates = dyno.candidate_profiles(model, base, "quality")

        self.assertEqual(len(candidates), 3)
        self.assertTrue(
            all(
                item.profile.cache_k == "f16" and item.profile.cache_v == "f16"
                for item in candidates
            )
        )

    def test_speed_objective_includes_bounded_speculative_trial(self) -> None:
        model = fixture_model()
        base = catalog.find_profile(model, "long-64k")

        candidates = dyno.candidate_profiles(model, base, "speed")

        self.assertEqual(len(candidates), 4)
        speculative = next(item for item in candidates if item.id == "ngram-spec")
        self.assertIn("ngram-mod", speculative.profile.extra_args)

    def test_near_vram_capacity_keeps_known_good_micro_batch(self) -> None:
        model = fixture_model()
        base = catalog.find_profile(model, "long-64k")
        hardware = {
            "gpus": [{"memory_mib": "24576"} for _ in range(4)],
        }

        with mock.patch("marathon_app.dyno.machine_identity", return_value=hardware):
            candidates = dyno.candidate_profiles(model, base, "balanced")

        self.assertTrue(all(item.profile.ubatch == base.ubatch for item in candidates))

    def test_backend_assertion_is_detected_from_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(
                "startup\nGGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed\n",
                encoding="utf-8",
            )

            message = dyno._fatal_log_message(path)

        self.assertIn("GGML_ASSERT", message or "")


class SelectionTests(unittest.TestCase):
    def test_pareto_frontier_removes_strictly_dominated_trial(self) -> None:
        strong = trial("strong", prompt=100, decode=30, context=65_536, power=400)
        weak = trial("weak", prompt=80, decode=20, context=65_536, power=500)

        frontier = dyno.pareto_frontier([strong, weak])

        self.assertEqual([item.candidate.id for item in frontier], ["strong"])

    def test_objective_changes_winner_deterministically(self) -> None:
        fast = trial("fast", prompt=120, decode=35, context=65_536, power=650)
        long = trial("long", prompt=70, decode=22, context=262_144, power=500)
        efficient = trial("efficient", prompt=80, decode=25, context=65_536, power=250)

        self.assertEqual(dyno.select_winner([fast, long, efficient], "speed").candidate.id, "fast")
        self.assertEqual(dyno.select_winner([fast, long, efficient], "context").candidate.id, "long")
        self.assertEqual(
            dyno.select_winner([fast, long, efficient], "efficiency").candidate.id,
            "efficient",
        )


class PersistenceTests(unittest.TestCase):
    def test_published_profile_is_local_and_invalidates_with_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"model")
            model = fixture_model(model_path)
            base = catalog.find_profile(model, "long-64k")
            candidate = dyno._variant(base, "winner", "winner", batch=1024, ubatch=256)
            winner = dyno.TrialResult(
                candidate=candidate,
                success=True,
                loaded_context=65_536,
                prompt_tps=100,
                decode_tps=25,
                average_power_w=400,
                quality_pass=True,
            )
            environment = {"machine": "one"}
            with (
                mock.patch.dict(
                    "os.environ",
                    {"MARATHON_DYNO_CONFIG_DIR": str(root / "config")},
                    clear=False,
                ),
                mock.patch("marathon_app.dyno.machine_identity", return_value=environment),
                mock.patch("marathon_app.dyno._backend_identity", return_value={"backend": "one"}),
            ):
                path = dyno._publish_profile(model, "balanced", winner)
                loaded = dyno.load_tuned_profiles(model)
                self.assertEqual(loaded[0].id, "dyno-balanced")
                self.assertEqual(loaded[0].batch, 1024)
                self.assertTrue(path.is_file())

                with mock.patch(
                    "marathon_app.dyno.machine_identity", return_value={"machine": "two"}
                ):
                    self.assertEqual(dyno.load_tuned_profiles(model), ())


if __name__ == "__main__":
    unittest.main()
