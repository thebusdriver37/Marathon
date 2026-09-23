#!/usr/bin/env python3
"""Trace the real frontend in an offline namespace with a loopback model fixture.

Usage: .marathon/venv/bin/python3 scripts/test_local_security.py /path/to/codex
No GPU workers or production sessions are touched.
Synthetic evidence is retained in the printed temporary directory for review.
"""

import asyncio
import fcntl
import ipaddress
import json
import os
import pty
import re
import secrets
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/routers"))
from codex_local_router import ModelProfile, RouterState
from marathon_app.codex_home import codex_environment
from marathon_app.frontends import codex_command
from marathon_app.router_security import router_security_middleware


async def inside(binary, output):
    subprocess.run(["ip", "link", "set", "lo", "up"], check=True)
    os.umask(0o077)
    stock = output / "stock"
    stock.mkdir()
    # Never inspect the operator's configuration, credentials, or conversations.
    (stock / "config.toml").write_text('[analytics]\nenabled = true\n')
    environment, home, profile = codex_environment({
        **os.environ, "CODEX_HOME": str(stock), "MARATHON_CODEX_HOME": str(output / "home"),
    })
    # A shipped build must enforce isolation even if the runtime flag is disabled.
    environment["MARATHON_LOCAL_ONLY"] = "0"
    token = secrets.token_urlsafe(32)
    environment.update({"MARATHON_ROUTER_TOKEN": token, "TERM": "xterm-256color"})
    os.environ["MARATHON_ROUTER_TOKEN"] = token
    os.environ["MARATHON_CODEX_BIN"] = binary
    requests = []
    tool_hits = []
    send_tool = False

    async def tool_check(request):
        tool_hits.append(True)
        return web.Response(text="SANCTIONED_TOOL_OK")

    def events(payload):
        nonlocal send_tool
        requests.append(payload)
        rid = f"resp_security_{len(requests)}"
        if payload.get("generate") is False:
            return [
                {"type": "response.created", "response": {"id": rid}},
                {"type": "response.completed", "response": {"id": rid, "output": [],
                 "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}},
            ]
        if send_tool:
            send_tool = False
            offered = {tool.get("name") for tool in payload.get("tools", [])}
            if "exec_command" not in offered:
                raise AssertionError(f"Fixture requires exec_command, offered: {offered}")
            item = {
                "type": "function_call", "name": "exec_command", "call_id": "security-tool",
                "arguments": json.dumps({"cmd": f"curl -fsS {url}/tool-check"}),
            }
        else:
            item = {"type": "message", "id": f"msg_{rid}", "role": "assistant",
                    "content": [{"type": "output_text", "text": "LOCAL_SECURITY_OK"}]}
        return [
            {"type": "response.created", "response": {"id": rid}},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {
                "id": rid, "status": "completed", "output": [item],
                "usage": {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105},
            }},
        ]

    async def responses(request):
        if request.method == "GET":
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                for event in events(json.loads(message.data)):
                    await ws.send_json(event)
            return ws
        payload = await request.json()
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events(payload))
        return web.Response(text=body, content_type="text/event-stream")

    # The tool endpoint is intentionally separate from the authenticated API.
    # A model-sanctioned shell tool can reach it; no browser accesses inference.
    protected = web.Application(middlewares=[router_security_middleware()])
    protected.router.add_route("*", "/v1/responses", responses)
    app = web.Application()
    app.router.add_get("/tool-check", tool_check)
    app.add_subapp("/api/", protected)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    catalog = output / "models.json"
    # Use Marathon's real catalog serializer with a synthetic profile, without
    # loading the machine's catalog, model files, or starting inference workers.
    alias = "marathon-security-fixture"
    display_name = "Marathon security fixture"
    profile_fixture = ModelProfile(
        slug=alias, alias=alias, display_name=display_name, description="Synthetic local test",
        model_paths=(), target=url, context_window=32768,
        auto_compact_token_limit=29000, truncation_limit=24000,
    )
    os.environ["MARATHON_PROMPT_FILE"] = str(ROOT / "codex/codex-rs/models-manager/prompt.md")
    catalog_state = SimpleNamespace(
        available_profiles={alias: profile_fixture}, _refresh_profiles=lambda: None,
    )
    catalog.write_text(json.dumps(RouterState.model_catalog(catalog_state)))
    runtime = SimpleNamespace(router_url=url + "/api", model=SimpleNamespace(alias=alias), catalog_file=catalog)
    command = codex_command(runtime, shared_profile=profile)
    # These deliberately hostile overrides must lose to the compiled local boundary.
    command += ["-c", "analytics.enabled=true", "-c", "check_for_update_on_startup=true",
                "-c", "features.plugins=true", "-c", 'otel.metrics_exporter="statsig"',
                "-c", f'projects.{json.dumps(str(output))}.trust_level="trusted"',
                "-c", "tui.animations=false"]

    async def launch(label, args, idle_seconds=0, expect_failure=False):
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))
        os.set_blocking(master, False)

        def terminal_session():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        process = await asyncio.create_subprocess_exec(
            "strace", "-f", "-qq", "-e", "trace=connect,sendto,sendmsg", "-o", str(output / f"{label}-connect.log"),
            *command, *args, env=environment, cwd=output,
            stdin=slave, stdout=slave, stderr=slave, preexec_fn=terminal_session,
        )
        os.close(slave)
        captured = bytearray()
        started = time.monotonic()
        stopped = False
        trusted = False
        try:
            while process.returncode is None:
                try:
                    chunk = os.read(master, 65536)
                    captured.extend(chunk)
                    if b"\x1b[6n" in chunk:
                        os.write(master, b"\x1b[1;1R")
                except (BlockingIOError, OSError):
                    pass
                elapsed = time.monotonic() - started
                if idle_seconds and not trusted:
                    screen = re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]", b"", captured)
                    if (output.name.encode() in screen and (
                            b"Trust and continue" in screen
                            or (b"Press enter to continue" in screen and b"trust" in screen.lower()))):
                        # Accept only the fixture directory's ordinary first-run trust prompt.
                        # Upstream quarantines pending input for up to one second
                        # when entering a protected screen; do not race that drain.
                        await asyncio.sleep(1.1)
                        os.write(master, b"\r")
                        trusted = True
                if idle_seconds and elapsed >= idle_seconds and not stopped:
                    os.write(master, b"\x03\x03")
                    stopped = True
                if elapsed > max(90, idle_seconds + 15):
                    raise TimeoutError(f"{label} frontend did not exit")
                await asyncio.sleep(0.02)
            await process.wait()
            # Fast rejection commands can exit between the last poll and their
            # final terminal write. Drain those bytes before closing the PTY.
            while True:
                try:
                    chunk = os.read(master, 65536)
                except (BlockingIOError, OSError):
                    break
                if not chunk:
                    break
                captured.extend(chunk)
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            os.close(master)
            (output / f"{label}-terminal.log").write_bytes(captured)
        if (process.returncode != 0) != expect_failure:
            raise AssertionError(f"{label} exit {process.returncode}; inspect synthetic terminal log")
        return captured

    try:
        denied = await launch("cloud-login", ["login"], expect_failure=True)
        assert b"does not support Codex cloud" in denied, "Cloud login did not fail at the local boundary"
        denied = await launch("remote-archive", [
            "archive", "00000000-0000-4000-8000-000000000001",
            "--remote", "ws://127.0.0.1:9",
        ], expect_failure=True)
        assert (b"cannot attach to an external Codex server" in denied
                or b"--no-daemon cannot be used with --remote" in denied), "Nested remote flag escaped isolation"
        idle_seconds = float(os.environ.get("MARATHON_SECURITY_IDLE_SECONDS", "75"))
        startup = await launch("startup", ["--no-alt-screen"], idle_seconds=idle_seconds)
        assert (alias.encode() in startup or display_name.encode() in startup), "Idle test never reached the ready local model screen"
        assert all(item.get("generate") is False for item in requests), "Idle startup generated model output"
        send_tool = True
        await launch("tool", ["--dangerously-bypass-approvals-and-sandbox", "exec",
                              "--skip-git-repo-check", "Run the requested tool and report the result."])
        assert tool_hits, "Model-sanctioned network tool did not run"
        assert any("SANCTIONED_TOOL_OK" in json.dumps(item.get("input")) for item in requests)
        rollouts = list((home / "sessions").rglob("*.jsonl"))
        assert rollouts, "No resumable session was saved"
        session_id = None
        for rollout in rollouts:
            for line in rollout.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("type") == "session_meta":
                    session_id = entry["payload"]["id"]
        before = len(requests)
        await launch("resume", ["--dangerously-bypass-approvals-and-sandbox", "exec", "resume",
                                "--skip-git-repo-check", session_id, "Continue the local test."])
        assert len(requests) > before, "Resume did not reach the local model"
        assert "LOCAL_SECURITY_OK" in json.dumps(requests[-1].get("input")), "Resume lost history"
        bad = []
        for trace in output.glob("*-connect.log"):
            for line in trace.read_text().splitlines():
                if "AF_INET" not in line:
                    continue
                addresses = re.findall(r'(?:inet_addr\(|inet_pton\(AF_INET6, )"([^"]+)"', line)
                ports = re.findall(r'sin6?_port=htons\((\d+)\)', line)
                fixture_port = str(site._server.sockets[0].getsockname()[1])
                if (not addresses or any(not ipaddress.ip_address(address).is_loopback for address in addresses)
                        or not ports or any(port != fixture_port for port in ports)):
                    bad.append(line)
        assert not bad, f"Network attempts outside the loopback fixture detected: {len(bad)}"
        result = {"idle_seconds": idle_seconds, "model_requests": len(requests),
                  "sanctioned_tool_hits": len(tool_hits), "resume": "passed", "non_loopback_connects": 0}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--inside":
        asyncio.run(inside(sys.argv[2], Path(sys.argv[3])))
    else:
        binary = str(Path(sys.argv[1]).resolve())
        output = Path(tempfile.mkdtemp(prefix="marathon-local-security-"))
        (output / "user-home").mkdir()
        print(f"Synthetic evidence: {output}", flush=True)
        result = subprocess.run([
            "bwrap", "--die-with-parent", "--unshare-user", "--uid", "0", "--gid", "0",
            "--unshare-net", "--cap-add", "CAP_NET_ADMIN", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--ro-bind", "/", "/", "--tmpfs", "/tmp", "--tmpfs", "/run", "--proc", "/proc", "--dev", "/dev",
            "--bind", str(output), str(output),
            "--setenv", "HOME", str(output / "user-home"),
            sys.executable, str(Path(__file__).resolve()), "--inside", binary, str(output),
        ], check=False)
        raise SystemExit(result.returncode)
