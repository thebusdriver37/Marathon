"""Opt-in real frontend resume tests with an isolated home and local Responses server."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

from marathon_app.codex_home import codex_environment
from marathon_app.frontends import _restore_conversation_model, codex_command
from scripts.evals.e2e_smoke import Terminal

ROOT = Path(__file__).resolve().parents[1]
FLASH = "qwen3.8-flash-next-spark"
DEFAULT = "qwen3.6-27b-test"


@unittest.skipUnless(os.environ.get("MARATHON_TEST_CODEX_BIN"),
                     "Set MARATHON_TEST_CODEX_BIN to test native conversation resume")
class ConversationModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="marathon-resume-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.requests = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.send_error(404)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Encoding") == "zstd":
                    import zstandard
                    body = zstandard.ZstdDecompressor().decompress(body)
                request = json.loads(body)
                requests.append(request)
                identifier = f"resp_{len(requests)}"
                answer = f"RESUME_VERIFIED_{len(requests)}"
                item = {"type": "message", "role": "assistant", "id": identifier + "_msg",
                        "status": "completed", "content": [
                            {"type": "output_text", "text": answer, "annotations": []}]}
                response = {"id": identifier, "object": "response", "status": "completed",
                            "output": [item], "usage": {"input_tokens": 100,
                            "output_tokens": 5, "total_tokens": 105}}
                events = [
                    {"type": "response.created", "response": dict(response, output=[], status="in_progress")},
                    {"type": "response.output_item.added", "output_index": 0,
                     "item": dict(item, content=[], status="in_progress")},
                    {"type": "response.output_text.delta", "item_id": item["id"],
                     "output_index": 0, "content_index": 0, "delta": answer},
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                    {"type": "response.completed", "response": response},
                ]
                wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        bundled = json.loads((ROOT / "codex/codex-rs/models-manager/models.json").read_text())
        template = bundled["models"][0]
        models = [dict(template, slug=slug, display_name=slug, prefer_websockets=False,
                       default_reasoning_level="low") for slug in (FLASH, DEFAULT)]
        self.catalog = self.root / "models.json"
        self.catalog.write_text(json.dumps({"models": models}))
        self.environment = dict(os.environ,
            MARATHON_CODEX_BIN=os.environ["MARATHON_TEST_CODEX_BIN"],
            MARATHON_STOCK_CODEX_HOME=str(self.root / "stock"),
            MARATHON_CODEX_HOME=str(self.root / "home"),
            MARATHON_INSTRUCTIONS_FILE=str(self.root / "MARATHON.md"),
            MARATHON_ROUTER_TOKEN="isolated-test-token", TERM="xterm-256color")

    def launch(self, default, arguments):
        environment, home, profile = codex_environment(self.environment)
        environment["MARATHON_RESTORE_THREAD_MODEL"] = (
            "1" if _restore_conversation_model(arguments) else "0")
        runtime = SimpleNamespace(model=SimpleNamespace(alias=default),
            router_url=f"http://127.0.0.1:{self.server.server_port}", catalog_file=self.catalog)
        with mock.patch.dict(os.environ, self.environment):
            command = codex_command(runtime, [
                "-c", "model_providers.marathon-local.supports_websockets=false",
                "-c", 'tui.status_line=["model-with-reasoning","run-state","context-tokens"]',
                *arguments], shared_profile=profile)
        return command, environment, home

    def execute(self, default, arguments):
        command, environment, home = self.launch(default, arguments)
        result = subprocess.run(command, cwd=self.root, env=environment,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return home

    def assert_request(self, model, effort):
        self.assertTrue(self.requests)
        self.assertEqual(self.requests[-1]["model"], model)
        self.assertEqual(self.requests[-1].get("reasoning", {}).get("effort"), effort)

    def close_terminal(self, terminal):
        if terminal.process.poll() is None:
            terminal.prompt("/quit")
        terminal.close()

    def wait_for_answer(self, terminal, previous_requests=0):
        answer = f"RESUME_VERIFIED_{previous_requests + 1}"
        self.wait_terminal(terminal, lambda: len(self.requests) > previous_requests
                           and answer in terminal.clean()
                           and "Ready" in terminal.clean().rsplit(answer, 1)[-1])

    def wait_terminal(self, terminal, predicate):
        try:
            terminal.wait(predicate, timeout=30)
        except TimeoutError:
            self.fail("Frontend did not reach the expected state:\n" + terminal.clean()[-2500:])

    def seed(self):
        home = self.execute(DEFAULT, ["-m", FLASH, "-c", 'model_reasoning_effort="high"',
                                     "exec", "--skip-git-repo-check", "hello"])
        self.assert_request(FLASH, "high")
        rollout = next((home / "sessions").rglob("*.jsonl"))
        metadata = json.loads(rollout.read_text().splitlines()[0])
        return metadata["payload"]["id"]

    def test_exec_resume_restores_both_settings_and_persists_explicit_changes(self):
        session = self.seed()
        for target in ([session], ["--last"]):
            self.execute(DEFAULT, ["exec", "resume", *target, "--skip-git-repo-check", "continue"])
            self.assert_request(FLASH, "high")
        self.execute(FLASH, ["-m", DEFAULT, "-c", 'model_reasoning_effort="medium"',
                            "exec", "resume", session, "--skip-git-repo-check", "change model"])
        self.assert_request(DEFAULT, "medium")
        self.execute(FLASH, ["exec", "resume", session, "--skip-git-repo-check", "continue again"])
        self.assert_request(DEFAULT, "medium")
        self.execute(FLASH, ["exec", "--skip-git-repo-check", "a separate conversation"])
        self.assert_request(FLASH, "low")
        self.execute(FLASH, ["exec", "resume", session, "--skip-git-repo-check", "return to the earlier conversation"])
        self.assert_request(DEFAULT, "medium")

    def test_interactive_resume_restores_model_effort_and_sends_them(self):
        command, environment, home = self.launch(DEFAULT, [
            "-m", FLASH, "-c", 'model_reasoning_effort="high"'])
        terminal = Terminal(command, self.root, environment, self.root / "seed.log")
        try:
            self.wait_terminal(terminal, lambda: FLASH in terminal.clean() and "high" in terminal.clean())
            terminal.prompt("hello from the interactive conversation")
            self.wait_for_answer(terminal)
            self.assert_request(FLASH, "high")
        finally:
            self.close_terminal(terminal)
        rollout = next((home / "sessions").rglob("*.jsonl"))
        session = json.loads(rollout.read_text().splitlines()[0])["payload"]["id"]
        for target in ([session], ["--last"], []):
            with self.subTest(target=target):
                command, environment, _ = self.launch(DEFAULT, ["resume", *target])
                terminal = Terminal(command, self.root, environment, self.root / "terminal.log")
                try:
                    if not target:
                        self.wait_terminal(terminal, lambda: "hellofrom" in "".join(terminal.clean().split()))
                        terminal.send("\r")
                    self.wait_terminal(terminal, lambda: FLASH in terminal.clean() and "high" in terminal.clean())
                    before = len(self.requests)
                    terminal.prompt("continue from the saved conversation")
                    self.wait_for_answer(terminal, before)
                    self.assert_request(FLASH, "high")
                finally:
                    self.close_terminal(terminal)
