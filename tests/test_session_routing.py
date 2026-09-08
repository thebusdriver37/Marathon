"""Conversation lookup must be independent of the allocated runtime worker."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from marathon_app import __main__ as main
from marathon_app.codex_home import session_home_for_id
from marathon_app.runtime import RuntimeBusyError

ID = "123e4567-e89b-12d3-a456-426614174000"


class SessionRoutingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "marathon"
        environment = mock.patch.dict(os.environ, {"MARATHON_CODEX_HOME": str(self.root)}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def save(self, home, category="sessions"):
        path = home / category / "2026/09/08" / f"rollout-date-{ID}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')

    def test_finds_legacy_named_home_and_archived_session(self):
        home = self.root / "instances/second"
        self.save(home, "archived_sessions")
        self.assertEqual(session_home_for_id(ID), home)

    def test_does_not_search_stock_codex(self):
        stock = self.root.parent / "stock"
        self.save(stock)
        with mock.patch.dict(os.environ, {"CODEX_HOME": str(stock)}):
            with self.assertRaisesRegex(ValueError, "was not found"):
                session_home_for_id(ID)

    def test_duplicate_uuid_is_not_silently_routed_to_wrong_home(self):
        self.save(self.root)
        self.save(self.root / "instances/second")
        with self.assertRaisesRegex(ValueError, "multiple Marathon homes"):
            session_home_for_id(ID)

    def test_unknown_id_does_not_start_a_worker(self):
        with mock.patch.object(main, "automatic_launch_instance") as allocate:
            self.assertEqual(main.main(["resume", ID]), 2)
        allocate.assert_not_called()

    def test_resume_routes_saved_home_to_a_different_free_runtime(self):
        home = self.root / "instances/second"
        self.save(home)
        with (
            mock.patch.object(main, "automatic_launch_instance", return_value="third"),
            mock.patch.object(main, "_relaunch_as_instance", return_value=0) as relaunch,
        ):
            self.assertEqual(main.main(["resume", ID]), 0)
        relaunch.assert_called_once_with("third", ["resume", ID], home)

    def test_child_consumes_home_override_and_keeps_worker_separate(self):
        home = self.root / "instances/second"
        with (
            mock.patch.dict(os.environ, {"_MARATHON_SESSION_HOME": str(home)}),
            mock.patch.object(main, "run_codex_default", return_value=0) as run,
        ):
            self.assertEqual(main.main(["--instance", "third", "resume", ID]), 0)
            self.assertNotIn("_MARATHON_SESSION_HOME", os.environ)
        run.assert_called_once_with(["resume", ID], "third", session_home=home)

    def test_launch_race_retries_without_stacking_instance_arguments(self):
        home = self.root / "instances/second"
        with (
            mock.patch.dict(os.environ, {"_MARATHON_SESSION_HOME": str(home)}),
            mock.patch.object(main, "run_codex_default", side_effect=RuntimeBusyError("busy")),
            mock.patch.object(main, "automatic_launch_instance", return_value="third"),
            mock.patch.object(main, "_relaunch_as_instance", return_value=0) as relaunch,
        ):
            self.assertEqual(main.main(["--instance", "second", "resume", ID]), 0)
        relaunch.assert_called_once_with("third", ["resume", ID], home)

    def test_no_capacity_does_not_launch_or_wait(self):
        with (
            mock.patch.object(main, "automatic_launch_instance", side_effect=RuntimeError("No free worker.")),
            mock.patch.object(main, "run_codex_default") as run,
        ):
            self.assertEqual(main.main([]), 2)
        run.assert_not_called()
