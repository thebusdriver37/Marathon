#!/usr/bin/env python3
"""Repeated, independently graded tasks through the real Marathon terminal.

Only synthetic projects and isolated conversation homes are used. GPU leases
remain global; existing workers and personal conversations are never evicted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import statistics
import subprocess
import sys
import tempfile
import time

from e2e_smoke import ROOT, Terminal, read_events
from prompt_audit import cases as coding_cases
from reliability_contracts import cases as additional_cases
from reliability_lifecycle import NAMES as LIFECYCLE_NAMES, fixture as lifecycle_fixture, run as run_lifecycle

sys.path.insert(0, str(ROOT))

SUFFIX = ("\nWork only in this project. Do not use the network, delegate, install packages, "
          "or inspect conversation/session/log files. Use the available tools to complete "
          "the task. Respect the requested scope.")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def child(spec):
    from dataclasses import replace
    from marathon_app.catalog import discover_models, find_profile
    from marathon_app.frontends import run_codex
    from marathon_app.runtime import Runtime

    models = [m for m in discover_models(Path(spec["model_path"]).parent) if str(m.path) == spec["model_path"]]
    if len(models) != 1:
        raise ValueError("--model must identify exactly one installed Marathon model")
    model = replace(models[0], id=spec["model_alias"])
    runtime = Runtime(model, find_profile(model, spec["profile"], "codex"), spec["instance"])
    try:
        runtime.start(lazy_pool=True)
        return run_codex(runtime, spec["argv"])
    finally:
        runtime.cleanup()


def sandbox_config(project):
    from marathon_app.frontends import _codex_binary

    # workspace-write permits reading the user's home. This narrower native
    # profile exposes only OS dependencies, the fixture, and the sandbox binary.
    files = {":minimal": "read", str(project): "write", str(Path(_codex_binary()).resolve()): "read"}
    filesystem = "{" + ", ".join(json.dumps(k) + "=" + json.dumps(v) for k, v in files.items()) + "}"
    settings = [
        'default_permissions="reliability"', "permissions.reliability.filesystem=" + filesystem,
        "permissions.reliability.network.enabled=false", 'approval_policy="never"',
        "allow_login_shell=false", 'web_search="disabled"', "features.multi_agent=false",
        "project_doc_max_bytes=0", 'model_reasoning_effort="medium"',
    ]
    return [argument for setting in settings for argument in ("-c", setting)]


def isolated_environment(trial, instance):
    home = trial / "codex-home" / "instances" / instance
    (home / "skills/.system").mkdir(parents=True, exist_ok=True)
    stock = trial / "empty-stock"
    (stock / "skills/.system").mkdir(parents=True, exist_ok=True)
    (stock / "config.toml").write_text("[projects." + json.dumps(str(trial / "project")) + "]\ntrust_level = \"trusted\"\n")
    instructions = trial / "instructions.md"
    instructions.write_text("Use only the synthetic project provided for this evaluation.\n")
    environment = dict(
        os.environ, TERM="xterm-256color", PYTHONPATH=str(ROOT),
        MARATHON_CODEX_HOME=str(trial / "codex-home"), MARATHON_STOCK_CODEX_HOME=str(stock),
        MARATHON_INSTRUCTIONS_FILE=str(instructions), MARATHON_RUNS_DIR=str(trial / "runs"),
        MARATHON_AUTO_INSTALL_CLI="0", MARATHON_WEB_SEARCH_MODE="disabled",
        MARATHON_STARTER_CACHE_ENABLED="0", MARATHON_SLOT_SNAPSHOTS_ENABLED="0",
        XDG_STATE_HOME=str(trial / "state"), XDG_CACHE_HOME=str(trial / "cache"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    # Keep XDG_RUNTIME_DIR unchanged: isolating it would bypass global GPU leases.
    environment.pop("_MARATHON_SESSION_HOME", None)
    return environment, home / "sessions"


def oracle(project, source, timeout=30):
    """Run the hidden checker without giving candidate code access to private data."""
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--ro-bind", "/usr", "/usr", "--dev", "/dev", "--proc", "/proc",
               "--tmpfs", "/tmp", "--dir", "/etc"]
    for path in ("/lib", "/lib64", "/etc/ld.so.cache"):
        if Path(path).exists():
            command += ["--ro-bind", path, path]
    marker = "MARATHON_CHECK_COMPLETE_" + secrets.token_hex(16)
    command += ["--ro-bind", str(project), "/source", "--tmpfs", "/work", "--chdir", "/work",
                "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--",
                "/usr/bin/python3", "-I", "-B", "-c",
                "import shutil, sys\n"
                "shutil.copytree('/source', '/work', dirs_exist_ok=True, "
                "ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))\n"
                "sys.path.insert(0, '/work')\n" +
                source + "\nprint(" + repr(marker) + ")\n"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "exit_code": 124, "output": "Independent checker timed out"}
    complete = result.stdout.endswith(marker + "\n")
    stdout = result.stdout[:-(len(marker) + 1)] if complete else result.stdout
    return {"passed": result.returncode == 0 and complete, "exit_code": result.returncode,
            "output": (stdout + result.stderr + ("" if complete else "\nChecker did not reach all assertions"))[-12000:]}


def terminal_ready(text):
    titles = re.findall(r"\x1b\](?:0|2);([^\x07\x1b]*)(?:\x07|\x1b\\)", text)
    return bool(titles) and "| Ready |" in titles[-1]


class Session:
    def __init__(self, trial, args, instance):
        self.trial, self.args = trial, args
        self.project = trial / "project"
        environment, self.sessions = isolated_environment(trial, instance)
        if args.router_source:
            wrapper = trial / "router-python"
            wrapper.write_text(f"#!{sys.executable}\n" + f'''
import sys
from pathlib import Path
sys.argv = sys.argv[1:]
__file__ = sys.argv[0]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
__name__ = "evaluation_router"
sys.modules[__name__] = sys.modules["__main__"]
exec(compile(Path({str(args.router_source.resolve())!r}).read_text(), __file__, "exec"), globals())
main()
''')
            wrapper.chmod(0o700)
            environment["MARATHON_ROUTER_PYTHON"] = str(wrapper)
        config = sandbox_config(self.project)
        if args.instructions_file:
            config += ["-c", "model_instructions_file=" + json.dumps(str(args.instructions_file.resolve()))]
        spec = {"instance": instance, "model": args.model, "profile": args.profile,
                "model_path": args.model_path, "model_alias": args.model_alias,
                "argv": ["--no-alt-screen", *config]}
        write_json(trial / "launch.json", spec)
        self.spec, self.environment = spec, environment
        self.started_ns = time.monotonic_ns()
        self.terminal = Terminal([sys.executable, str(Path(__file__).resolve()), "--child",
                                  str(trial / "launch.json")], self.project, environment, trial / "terminal.txt")
        self.path = None
        self.turns = []

    def events(self):
        return read_events(self.path) if self.path else []

    def completed(self):
        return [e["payload"] for e in self.events() if e.get("type") == "event_msg"
                and e.get("payload", {}).get("type") == "task_complete"]

    def ready(self):
        def connected():
            return terminal_ready(self.terminal.text) and any(
                e.get("event") == "router.response.completed" and e.get("mono_ns", 0) >= self.started_ns
                for p in (self.trial / "runs").rglob("*.jsonl") for e in read_events(p))
        self.terminal.wait(connected, self.args.timeout)

    def turn(self, prompt):
        before = len(self.completed())
        started = time.monotonic()
        deadline = started + self.args.timeout

        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("turn exceeded its deadline")
            return seconds

        self.terminal.wait(lambda: terminal_ready(self.terminal.text), remaining())
        if prompt.startswith("/") and "\n" not in prompt:
            self.terminal.prompt(prompt)
        else:
            # Real terminals bracket multiline pastes. Raw byte injection can
            # let the Enter key become part of a paste burst and never submit.
            self.terminal.send("\x1b[200~" + prompt + "\x1b[201~")
            paste_deadline = time.monotonic() + .3
            while time.monotonic() < paste_deadline:
                self.terminal.pump()
            self.terminal.send("\r")
        if self.path is None:
            self.path = self.terminal.wait(lambda: next(iter(self.sessions.rglob("*.jsonl")), None), remaining())
        answer = self.terminal.wait(lambda: self.completed()[before:] or None, remaining())[0]
        # Journal completion precedes the TUI's Ready transition. Slash commands
        # submitted during that gap can be dropped, especially after long turns.
        self.terminal.wait(lambda: terminal_ready(self.terminal.text), remaining())
        row = {"prompt": prompt, "answer": answer.get("last_agent_message") or "",
               "seconds": time.monotonic() - started}
        self.turns.append(row)
        write_json(self.trial / "turns.json", self.turns)
        return row["answer"]

    def compact(self):
        before = sum(e.get("type") == "compacted" for e in self.events())
        self.turn("/compact")
        if sum(e.get("type") == "compacted" for e in self.events()) != before + 1:
            raise AssertionError("native compaction did not occur")

    def resume(self):
        thread_id = next(e["payload"]["id"] for e in self.events() if e.get("type") == "session_meta")
        self.close()
        write_json(self.trial / "launch-first.json", self.spec)
        self.spec["argv"] += ["resume", thread_id]
        write_json(self.trial / "launch.json", self.spec)
        self.started_ns = time.monotonic_ns()
        self.terminal = Terminal([sys.executable, str(Path(__file__).resolve()), "--child",
                                  str(self.trial / "launch.json")], self.project, self.environment,
                                 self.trial / "terminal-resume.txt")
        self.ready()
        resumed = list(self.sessions.rglob("*.jsonl"))
        if resumed != [self.path]:
            raise AssertionError("resume created or selected a different conversation")

    def close(self):
        self.terminal.close()


def prepare(project, files):
    project.mkdir(parents=True)
    for name, content in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q", str(project)], check=True)


def check_case(project, case, answer, events):
    checks = {}
    for name in case.get("protected", []):
        path = project / name
        content = path.read_text() if path.is_file() and path.resolve().is_relative_to(project) else None
        checks["preserved:" + name] = content is not None and (
            content == case["files"][name] or
            (name in case.get("allow_append", []) and content.startswith(case["files"][name])))
    for needle in case.get("answer_contains", []):
        checks["answer:" + needle] = needle.casefold() in answer.casefold()
    if case.get("readonly"):
        checks["no_extra_files"] = {p.relative_to(project).as_posix() for p in project.rglob("*")
                                    if p.is_file() and ".git" not in p.parts} == set(case["files"])
    if case.get("require_patch"):
        checks["patch_used"] = any(e.get("type") == "response_item" and
                                    "apply_patch" in e.get("payload", {}).get("name", "") for e in events)
    graded = oracle(project, case.get("oracle", ""))
    checks["independent_contract"] = graded["passed"]
    return checks, graded


def run_trial(output, args, case_name, repeat):
    trial = output / f"{case_name}-r{repeat}"
    project = trial / "project"
    case = (lifecycle_fixture(case_name, repeat) if case_name in LIFECYCLE_NAMES
            else (coding_cases() | additional_cases())[case_name])
    if case_name == "precedence":
        case["allow_append"] = ["test_limits.py"]
    if case_name == "deep_trace":
        case["protected"] += [name for name in case["files"] if name.startswith("pipeline/stage_")
                              and name != "pipeline/stage_21.py"]
    prepare(project, case["files"])
    instance = f"reliability-{os.getpid()}-{hashlib.sha256(trial.name.encode()).hexdigest()[:8]}"
    started = time.monotonic()
    session = None
    result = {"case": case_name, "repeat": repeat, "passed": False, "status": "running",
              "interventions": 0}
    write_json(trial / "result.json", result)
    try:
        session = Session(trial, args, instance)
        session.ready()
        ready = time.monotonic()
        if case_name in LIFECYCLE_NAMES:
            answer, lifecycle_checks = run_lifecycle(case_name, session, case, SUFFIX, oracle)
        else:
            answer, lifecycle_checks = session.turn(case["prompt"] + SUFFIX), {}
        checks, grade = check_case(project, case, answer, session.events())
        checks.update(lifecycle_checks)
        if case_name == "recovery":
            outputs = [str(e.get("payload", {}).get("output", "")) for e in session.events()
                       if e.get("type") == "response_item" and e.get("payload", {}).get("type") == "function_call_output"]
            checks["failed_runner_exercised"] = any("Process exited with code 127" in output and
                                                    "Optional runner unavailable" in output for output in outputs)
        result.update(checks=checks, grade=grade, passed=all(checks.values()),
                      startup_seconds=ready - started, task_seconds=time.monotonic() - ready,
                      status="passed" if all(checks.values()) else "failed")
    except Exception as error:
        result.update(status="timeout" if isinstance(error, TimeoutError) else "error", error=str(error))
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                result.update(status="error", passed=False, cleanup_error=str(error))
            events = session.events()
            result["tool_calls"] = sum(e.get("type") == "response_item" and
                                       e.get("payload", {}).get("type") in ("function_call", "custom_tool_call")
                                       for e in events)
            result["compactions"] = sum(e.get("type") == "compacted" for e in events)
            traces = [e for p in (trial / "runs").rglob("*.jsonl") for e in read_events(p)]
            result["automatic_recoveries"] = sum(e.get("event") in {
                "router.response.stalled_recovery", "router.response.tool_protocol_recovery",
                "router.response.compaction_recovery"} for e in traces)
            result["workers"] = sorted({e["data"]["worker"] for e in traces
                                        if e.get("event") == "router.pool.acquired"})
            completed = [e["data"] for e in traces if e.get("event") == "router.response.completed"]
            result["backend_output_tokens"] = sum(
                (e.get("usage") or e.get("backend", {}).get("usage") or {}).get("output_tokens", 0)
                for e in completed)
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(trial / "result.json", result)
    return result


def summarize(rows):
    passed = [r for r in rows if r["passed"]]
    times = [r["task_seconds"] for r in passed]
    return {"runs": len(rows), "unassisted_passes": len(passed),
            "unassisted_success_rate": len(passed) / len(rows) if rows else 0,
            "timeouts": sum(r["status"] == "timeout" for r in rows),
            "errors": sum(r["status"] == "error" for r in rows),
            "interventions": sum(r["interventions"] for r in rows),
            "median_success_seconds": statistics.median(times) if times else None,
            "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows)}


def compare_reports(baseline, candidate):
    """Compare matched runs; incompatible or incomplete experiments cannot pass."""
    for field in ("model", "profile", "cases", "repeats", "timeout", "evaluation_sha256"):
        if (baseline["manifest"].get(field) is None or
                baseline["manifest"].get(field) != candidate["manifest"].get(field)):
            raise ValueError(f"comparison requires matching {field}")
    expected = {(case, repeat) for case in candidate["manifest"]["cases"]
                for repeat in range(candidate["manifest"]["repeats"])}
    if not expected:
        raise ValueError("comparison requires at least one planned run")
    indexed = []
    for report in (baseline, candidate):
        rows = report["results"]
        if any(type(row["passed"]) is not bool or row["status"] not in {"passed", "failed", "timeout", "error"}
               or row["passed"] != (row["status"] == "passed") for row in rows):
            raise ValueError("comparison requires consistent, completed outcomes")
        by_key = {(row["case"], row["repeat"]): row for row in rows}
        if len(by_key) != len(rows) or set(by_key) != expected:
            raise ValueError("comparison requires all planned runs, without duplicates")
        indexed.append(by_key)
    before, after = indexed
    regressions = [list(key) for key in sorted(expected) if before[key]["passed"] and not after[key]["passed"]]
    improvements = [list(key) for key in sorted(expected) if not before[key]["passed"] and after[key]["passed"]]
    paired = [(before[k]["task_seconds"], after[k]["task_seconds"]) for k in expected
              if before[k]["passed"] and after[k]["passed"]]
    return {"regressions": regressions, "improvements": improvements,
            "baseline_success_rate": summarize(baseline["results"])["unassisted_success_rate"],
            "candidate_success_rate": summarize(candidate["results"])["unassisted_success_rate"],
            "matched_successes": len(paired),
            "median_paired_seconds_saved": statistics.median(a - b for a, b in paired) if paired else None}


def source_provenance(root):
    """Release archives do not contain .git; still fingerprint their runtime."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                         capture_output=True, text=True)
    checkout = top.returncode == 0 and Path(top.stdout.strip()).resolve() == root.resolve()
    if checkout:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        names = subprocess.check_output(
            ["git", "ls-files", "marathon_app", "scripts/routers", "patches", "config", "bin"],
            cwd=root, text=True).splitlines()
    else:
        revision = None
        names = [str(p.relative_to(root)) for directory, pattern in (
            ("marathon_app", "*.py"), ("marathon_app", "*.md"), ("scripts/routers", "*.py"),
            ("patches", "*.patch"), ("config", "*.toml"), ("config", "*.ref"))
                 for p in (root / directory).rglob(pattern) if p.is_file()]
        if (root / "bin/marathon").is_file():
            names.append("bin/marathon")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)}
    return revision, hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--model", help="installed model alias, ID, or GGUF basename")
    parser.add_argument("--profile", help="explicit Marathon runtime profile")
    all_cases = list(coding_cases()) + list(additional_cases()) + list(LIFECYCLE_NAMES)
    parser.add_argument("--cases", nargs="+", choices=all_cases, default=all_cases)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240, help="seconds allowed per turn")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--router-source", type=Path, help="isolated baseline/candidate router override")
    parser.add_argument("--candidate-router", type=Path,
                        help="run an interleaved A/B comparison; reverse arm order on alternate repeats")
    parser.add_argument("--instructions-file", type=Path, help="isolated prompt ablation")
    parser.add_argument("--compare", type=Path, help="baseline results.json; fail on any matched pass-to-fail regression")
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child(json.loads(args.child.read_text()))
    if not args.run_gpu:
        parser.error("live inference is opt-in; pass --run-gpu")
    if not args.model or not args.profile:
        parser.error("--model and --profile are required for a reproducible run")
    if min(args.repeats, args.timeout) < 1 or not 0 <= args.min_success_rate <= 1:
        parser.error("repeats/timeout must be positive and min-success-rate between 0 and 1")
    if len(set(args.cases)) != len(args.cases):
        parser.error("case names must be unique")
    if args.compare and args.candidate_router:
        parser.error("use either --compare or --candidate-router")
    from marathon_app.catalog import discover_models
    models = [m for m in discover_models() if args.model in (m.alias, m.id, m.path.name)]
    if len(models) != 1:
        parser.error("--model must identify exactly one installed Marathon model")
    args.model_path, args.model_alias = str(models[0].path), models[0].alias
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="marathon-reliability-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    revision, runtime_hash = source_provenance(ROOT)
    manifest = {"schema": 1, "model": args.model, "profile": args.profile,
                "cases": args.cases, "repeats": args.repeats, "timeout": args.timeout,
                "revision": revision,
                "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), ROOT / "scripts/evals/prompt_audit.py",
                                               ROOT / "scripts/evals/reliability_lifecycle.py",
                                               ROOT / "scripts/evals/reliability_contracts.py",
                                               ROOT / "scripts/evals/long_context_project.py",
                                               ROOT / "scripts/routers/codex_local_router.py"]}}
    manifest["evaluation_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in manifest["source_hashes"].items() if k.startswith("scripts/evals/")},
        sort_keys=True).encode()).hexdigest()
    from marathon_app.frontends import _codex_binary
    binary = Path(_codex_binary()).resolve()
    model_stat = models[0].path.stat()
    manifest["frontend"] = {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    manifest["model_file"] = {"name": models[0].path.name, "size": model_stat.st_size,
                              "mtime_ns": model_stat.st_mtime_ns}
    manifest["runtime_sources_sha256"] = runtime_hash
    if args.candidate_router and not args.router_source:
        args.router_source = ROOT / "scripts/routers/codex_local_router.py"
    for key in ("router_source", "instructions_file", "candidate_router"):
        path = getattr(args, key)
        if path:
            content = path.read_bytes()
            snapshot = output / (key + path.suffix)
            snapshot.write_bytes(content)
            manifest[key] = {"path": str(path.resolve()), "snapshot": str(snapshot),
                             "sha256": hashlib.sha256(content).hexdigest()}
            setattr(args, key, snapshot)
    write_json(output / "manifest.json", manifest)
    baseline_report = None
    if args.compare:
        try:
            baseline_report = json.loads(args.compare.read_text())
            # Reuse the baseline rows to validate compatibility/completeness
            # before allocating any model worker for the new experiment.
            compare_reports(baseline_report, {"manifest": manifest, "results": baseline_report["results"]})
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
            parser.error("invalid baseline comparison: " + str(error))
        write_json(output / "baseline-report.json", baseline_report)
    print(f"Evidence: {output}", flush=True)
    if args.candidate_router:
        reports = {arm: {"manifest": copy.deepcopy(manifest), "results": []}
                   for arm in ("baseline", "candidate")}
        for arm, report in reports.items():
            report["manifest"]["comparison_arm"] = arm
            if arm == "candidate":
                report["manifest"]["router_source"] = manifest["candidate_router"]
        for repeat in range(args.repeats):
            arms = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for case in args.cases:
                for arm in arms:
                    arm_args = copy.copy(args)
                    if arm == "candidate":
                        arm_args.router_source = args.candidate_router
                    row = run_trial(output / arm, arm_args, case, repeat)
                    reports[arm]["results"].append(row)
                    reports[arm]["summary"] = summarize(reports[arm]["results"])
                    write_json(output / arm / "results.json", reports[arm])
                    print(json.dumps({"arm": arm, **{k: row[k] for k in (
                        "case", "repeat", "passed", "status", "elapsed_seconds")}}), flush=True)
        comparison = compare_reports(reports["baseline"], reports["candidate"])
        write_json(output / "comparison.json", comparison)
        print(json.dumps(comparison), flush=True)
        return 0 if (not comparison["regressions"] and
                     comparison["candidate_success_rate"] >= args.min_success_rate) else 1
    rows = []
    for repeat in range(args.repeats):
        for case in args.cases:
            row = run_trial(output, args, case, repeat)
            rows.append(row)
            report = {"manifest": manifest, "summary": summarize(rows), "results": rows}
            write_json(output / "results.json", report)
            print(json.dumps({k: row[k] for k in ("case", "repeat", "passed", "status", "elapsed_seconds")}), flush=True)
    print(json.dumps(report["summary"]), flush=True)
    passed = report["summary"]["unassisted_success_rate"] >= args.min_success_rate
    if args.compare:
        try:
            report["comparison"] = compare_reports(baseline_report, report)
            passed = passed and not report["comparison"]["regressions"]
        except (ValueError, KeyError, OSError) as error:
            report["comparison"] = {"error": str(error)}
            passed = False
        write_json(output / "results.json", report)
        print(json.dumps(report["comparison"]), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
