#!/usr/bin/env python3
"""Opt-in live terminal and headless regression suite; no mocked inference.

All project files, transcripts, sessions, and traces stay in the printed output
folder. Marathon owns worker allocation; the suite never evicts other workloads.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import selectors
import shlex
import struct
import subprocess
import tempfile
import termios
import time

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "bin/marathon"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1bM")
PHRASE = "copper-otter-731"


def read_events(path: Path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # Live logs can end in a partial write.
            continue
    return events


class Terminal:
    def __init__(self, argv, cwd, environment, transcript):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 35, 120, 0, 0))

        def controlling_terminal():
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        self.process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=slave,
                                        stdout=slave, stderr=slave,
                                        preexec_fn=controlling_terminal)
        os.close(slave)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.master, selectors.EVENT_READ)
        self.log = transcript.open("wb")
        self.text = ""
        self.trusted = False
        self.closed = False

    def send(self, text):
        os.write(self.master, text.encode())

    def pump(self):
        for _key, _mask in self.selector.select(0.2):
            try:
                raw = os.read(self.master, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    return
                raise
            self.log.write(raw)
            self.log.flush()
            chunk = raw.decode(errors="replace")
            self.text += chunk
            if "\x1b[6n" in chunk:
                self.send("\x1b[1;1R")
            if "Press enter to continue" in self.clean() and not self.trusted:
                time.sleep(0.25)
                self.send("\r")
                self.trusted = True

    def prompt(self, text):
        self.send(text)
        # Let the terminal's paste detector settle before Enter.
        time.sleep(0.25)
        self.send("\r")

    def wait(self, predicate, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump()
            result = predicate()
            if result:
                return result
            if self.process.poll() is not None:
                raise RuntimeError("Marathon exited early: " + self.clean()[-1500:])
        raise TimeoutError("Marathon did not complete the expected action; see transcript")

    def clean(self):
        return ANSI.sub("", self.text)

    def close(self):
        if self.closed:
            return
        if self.process.poll() is None:
            self.send("\x03")
            deadline = time.monotonic() + 30
            while self.process.poll() is None and time.monotonic() < deadline:
                self.pump()
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=30)
        for _ in range(3):
            self.pump()
        self.selector.close()
        os.close(self.master)
        self.log.close()
        self.closed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true", help="run real inference on the remembered model")
    parser.add_argument("--output-dir", type=Path, help="new directory to retain test evidence")
    parser.add_argument("--workers", type=int, choices=(1, 3), default=1,
                        help="use three workers to additionally test pool exhaustion")
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("live inference is opt-in; pass --run-gpu")
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="marathon-smoke-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / "adder.py").write_text("def add(a, b):\n    return a - b\n")
    (project / "test_adder.py").write_text(
        "import unittest\nfrom adder import add\nclass Tests(unittest.TestCase):\n"
        "    def test_sum(self): self.assertEqual(add(2, 3), 5)\n"
        "    def test_negative(self): self.assertEqual(add(-3, -4), -7)\n")
    (project / "slow.py").write_text(
        "import time\nfrom pathlib import Path\n"
        "for tick in range(1200):\n"
        "    Path('heartbeat.txt').write_text(str(tick))\n    time.sleep(0.1)\n"
        "Path('late.txt').write_text('FINISHED')\n")
    instance = f"smoke-{os.getpid()}"
    environment = dict(os.environ, TERM="xterm-256color", CODEX_CLI_NAME=str(LAUNCHER),
                       MARATHON_CODEX_HOME=str(output / "codex-home"), MARATHON_RUNS_DIR=str(output / "runs"))
    sessions = output / "codex-home/instances" / instance / "sessions"
    terminals = []
    results = []

    def record(name, **details):
        row = {"test": name, "passed": True, **details}
        results.append(row)
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)

    def launch(name, prompt):
        terminal = Terminal([str(LAUNCHER), "--instance", name, "codex", "--no-alt-screen",
                             "-s", "workspace-write", "-c", 'model_reasoning_effort="low"', prompt],
                            project, environment, output / f"{name}.terminal.txt")
        terminals.append(terminal)
        return terminal

    def completed(session):
        return [e["payload"] for e in read_events(session)
                if e.get("type") == "event_msg" and e.get("payload", {}).get("type") == "task_complete"]

    def next_turn(terminal, session, prompt):
        count = len(completed(session))
        terminal.prompt(prompt)
        return terminal.wait(lambda: completed(session)[count:] or None)[0]

    print(f"Evidence: {output}", flush=True)
    try:
        terminal = launch(instance, f"Remember {PHRASE}. Reply exactly READY. Do not use tools.")
        session = terminal.wait(lambda: next(iter(sessions.rglob("*.jsonl")), None))
        first = terminal.wait(lambda: completed(session) or None)[0]
        assert first["last_agent_message"].strip() == "READY", first
        thread_id = next(e["payload"]["id"] for e in read_events(session) if e.get("type") == "session_meta")
        latencies = [first.get("time_to_first_token_ms")]
        for _ in range(3):
            turn = next_turn(terminal, session, "Reply only with the phrase I asked you to remember. Do not use tools.")
            assert turn["last_agent_message"].strip() == PHRASE, turn
            latencies.append(turn.get("time_to_first_token_ms"))
        # No hidden naming inference should displace the active prompt prefix.
        traces = [e for p in (output / "runs").rglob("*.jsonl") for e in read_events(p)]
        requests = [e["data"] for e in traces if e.get("event") == "router.request.normalized"
                    and e["data"].get("normalized", {}).get("input_items", 0) > 0]
        assert {r["prompt_cache_key"] for r in requests} == {thread_id}, requests
        completions = [e["data"] for e in traces if e.get("event") == "router.response.completed"
                       and e["data"].get("backend_ms", 0) > 0]
        assert all(e["slot"]["prepare_mode"] == "reuse-live-parent" for e in completions[1:]), completions
        record("chat-memory-and-live-cache", first_activity_ms=latencies,
               router_timings=[{k: e.get(k) for k in ("request_ms", "queue_wait_ms", "first_activity_ms", "backend_ms")}
                               for e in completions])

        next_turn(terminal, session, "Reproduce the failing unittest tests, fix adder.py, then rerun them. "
                  "Work only in this project. Do not use the network or delegate.")
        checked = subprocess.run(["python3", "-m", "unittest", "-v"], cwd=project,
                                 capture_output=True, text=True)
        (output / "coding-tests.txt").write_text(checked.stdout + checked.stderr)
        assert checked.returncode == 0, checked.stderr
        record("file-edit-and-real-tests")

        next_turn(terminal, session, 'Call exec_command exactly with {"cmd":"python3 slow.py","yield_time_ms":1000}. '
                  "Do not add &, nohup, redirection, or any other command. "
                  "The tool will return a running session ID; leave that session running and finish your reply. "
                  "Do not poll it or wait for completion. I will use /stop to cancel it.")
        heartbeat = project / "heartbeat.txt"
        terminal.wait(heartbeat.exists)
        previous = heartbeat.stat().st_mtime_ns
        terminal.wait(lambda: heartbeat.stat().st_mtime_ns != previous, timeout=5)
        terminal.prompt("/stop")
        previous = heartbeat.stat().st_mtime_ns
        stable_since = time.monotonic()

        def stopped():
            nonlocal previous, stable_since
            current = heartbeat.stat().st_mtime_ns
            if current != previous:
                previous, stable_since = current, time.monotonic()
            return time.monotonic() - stable_since >= 2

        terminal.wait(stopped, timeout=15)
        assert not (project / "late.txt").exists()
        turn = next_turn(terminal, session, "Reply exactly RECOVERED. Do not use tools.")
        assert turn["last_agent_message"].strip() == "RECOVERED", turn
        record("cancel-background-tool-and-continue")

        if args.workers == 3:
            for index in (2, 3):
                name = f"{instance}-{index}"
                extra = launch(name, "Reply exactly CAPACITY_OK. Do not use tools.")
                extra_sessions = output / "codex-home/instances" / name / "sessions"
                path = extra.wait(lambda: next(iter(extra_sessions.rglob("*.jsonl")), None))
                extra.wait(lambda: completed(path) or None)
            overflow = subprocess.run([str(LAUNCHER), "--instance", f"{instance}-overflow", "exec", "Reply READY"],
                                      cwd=project, env=environment, capture_output=True, text=True, timeout=30)
            (output / "capacity.txt").write_text(overflow.stdout + overflow.stderr)
            assert overflow.returncode == 2 and "already assigned" in overflow.stdout, overflow
            record("three-workers-and-exhaustion")
            for extra in terminals[1:]:
                extra.close()

        terminal.close()
        match = re.search(r"To continue this session, run:\s*\r?\n\s*([^\r\n]+)", terminal.clean())
        assert match, terminal.clean()[-1500:]
        command = match.group(1).strip()
        assert shlex.split(command) == [str(LAUNCHER), "--instance", instance, "resume", thread_id], command
        resumed = Terminal(["bash", "-lc", command], project, environment, output / "resume.terminal.txt")
        terminals.append(resumed)
        resumed.wait(lambda: PHRASE in resumed.clean())
        turn = next_turn(resumed, session, "Reply only with the original remembered phrase. Do not use tools.")
        assert turn["last_agent_message"].strip() == PHRASE, turn
        resumed.close()
        record("paste-printed-resume-command", command=command)

        headless = subprocess.run([str(LAUNCHER), "--instance", instance, "exec", "--json", "-s", "workspace-write",
                                   "--config=model_reasoning_effort=\"low\"", "-c", 'web_search="disabled"',
                                   "Run python3 -m unittest and report the result. Do not modify files or delegate."],
                                  cwd=project, env=environment, capture_output=True, text=True, timeout=180)
        (output / "headless.jsonl").write_text(headless.stdout)
        (output / "headless.stderr.txt").write_text(headless.stderr)
        assert headless.returncode == 0, headless.stderr
        events = [json.loads(line) for line in headless.stdout.splitlines() if line.startswith("{")]
        assert any(e.get("item", {}).get("type") == "command_execution"
                   and e["item"].get("exit_code") == 0 for e in events), events
        assert any(e.get("type") == "turn.completed" for e in events), events
        record("headless-overrides-and-shell-tools")
    except Exception as error:
        results.append({"test": "suite", "passed": False, "error": str(error)})
        (output / "results.json").write_text(json.dumps(results, indent=2))
        raise
    finally:
        for terminal in reversed(terminals):
            terminal.close()
    print(f"PASS: {len(results)} workflows; evidence retained at {output}", flush=True)


if __name__ == "__main__":
    main()
