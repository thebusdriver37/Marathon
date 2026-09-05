"""Local-home migration and frontend authentication regression coverage."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marathon_app.codex_home import SHARED_PROFILE_FILE, codex_environment, tomllib
from marathon_app.frontends import run_codex


class CodexSecurityTests(unittest.TestCase):
    def test_migration_keeps_stock_resources_and_sessions_but_not_cloud_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock, local = root / "stock", root / "local"
            stock.mkdir()
            local.mkdir()
            (stock / "config.toml").write_text('''
personality = "pragmatic"
model = "cloud-model"
notify = ["cloud-notifier"]
[analytics]
enabled = true
[mcp_servers.implicit-cloud]
command = "cloud-mcp"
[features]
plugins = true
[projects."/tmp/test"]
trust_level = "trusted"
''')
            for name in ("auth.json", "hooks.json", "plugins"):
                (stock / name).write_text("private stock data")
                (local / name).symlink_to(stock / name)
            (local / "sessions").mkdir()
            session = local / "sessions" / "existing.jsonl"
            session.write_text("existing conversation")
            environment = {
                "CODEX_HOME": str(stock), "MARATHON_CODEX_HOME": str(local),
                "OPENAI_API_KEY": "do-not-inherit", "SANCTIONED_TOOL_KEY": "keep",
            }
            child, home, _ = codex_environment(environment)
            config = tomllib.loads((home / SHARED_PROFILE_FILE).read_text())
            self.assertNotIn("mcp_servers", config)
            self.assertNotIn("notify", config)
            self.assertNotIn("model", config)
            self.assertEqual(config["analytics"], {"enabled": False})
            self.assertEqual(config["projects"], {"/tmp/test": {"trust_level": "trusted"}})
            self.assertEqual(config["personality"], "pragmatic")
            self.assertEqual(child["MARATHON_LOCAL_ONLY"], "1")
            self.assertNotIn("OPENAI_API_KEY", child)
            self.assertEqual(child["SANCTIONED_TOOL_KEY"], "keep")
            for name in ("auth.json", "hooks.json", "plugins"):
                self.assertFalse((local / name).is_symlink())
                self.assertFalse((local / name).exists())
                self.assertEqual((stock / name).read_text(), "private stock data")
            self.assertEqual(session.read_text(), "existing conversation")
            self.assertEqual(home.stat().st_mode & 0o777, 0o700)
            self.assertEqual((home / SHARED_PROFILE_FILE).stat().st_mode & 0o777, 0o600)
            self.assertEqual(codex_environment(environment)[0], child)

    def test_locally_owned_credentials_are_recoverable_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            local.mkdir()
            (local / "auth.json").write_text("owned credentials")
            codex_environment({"CODEX_HOME": str(root / "stock"), "MARATHON_CODEX_HOME": str(local)})
            backups = list(local.glob("auth.json.disabled-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), "owned credentials")

    def test_global_home_escape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no longer supported"):
            codex_environment({"MARATHON_USE_USER_CONFIG": "1"})

    def test_unhardened_binary_fails_before_execution(self):
        with (
            mock.patch("marathon_app.frontends.codex_environment", return_value=({}, Path("/unused"), None)),
            mock.patch("marathon_app.frontends.codex_command", return_value=["codex"]),
            mock.patch("marathon_app.frontends._codex_features", return_value=set()),
            mock.patch("marathon_app.frontends.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "hardened frontend"):
                run_codex(mock.Mock())
            run.assert_not_called()
