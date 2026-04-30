#!/usr/bin/env python3
"""Server-based config sweep for Qwen3.6 27B 128K on llama.cpp.

This launches the real llama.cpp server with focused 128K configs, then scores
each config using:
1. A short coding prompt through /v1/chat/completions for decode + prompt speed.
2. A long Codex-like /v1/responses replay for stability and long-prefill speed.

The goal is to choose a practical daily-driver config for Codex-style local use:
stable on long prompts, strong prompt throughput, and solid decode speed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAUNCHER = ROOT / "scripts" / "launchers" / "server_27b_128k.sh"
DEFAULT_MODEL = "qwen3.6-27b-q4-128k"
DEFAULT_PORT = 18091
DEFAULT_TARGET = f"http://127.0.0.1:{DEFAULT_PORT}"
DEFAULT_CAPTURED_REQUEST = ROOT / "logs" / "codex_local_router_request.json"

SHORT_PROMPTS = [
    "Write a concise Python function that deduplicates a list while preserving order, then add two short tests.",
    "Write a Python function that groups file paths by extension and include two assert-based tests.",
]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    split_mode: str
    tensor_split: str
    batch: int
    ubatch: int
    cache_type: str
    threads: int = 24


FOCUSED_CONFIGS: list[SweepConfig] = [
    SweepConfig("layer_q8_b224_u56", "layer", "1,1", 224, 56, "q8_0"),
    SweepConfig("layer_q8_b256_u64", "layer", "1,1", 256, 64, "q8_0"),
    SweepConfig("layer_q8_b288_u72", "layer", "1,1", 288, 72, "q8_0"),
    SweepConfig("layer_q8_b320_u80", "layer", "1,1", 320, 80, "q8_0"),
    SweepConfig("row_q8_b224_u56", "row", "1,1", 224, 56, "q8_0"),
    SweepConfig("row_q8_b256_u64", "row", "1,1", 256, 64, "q8_0"),
    SweepConfig("row_q8_b288_u72", "row", "1,1", 288, 72, "q8_0"),
    SweepConfig("row_q8_b320_u80", "row", "1,1", 320, 80, "q8_0"),
    SweepConfig("layer_q5_b256_u64", "layer", "1,1", 256, 64, "q5_1"),
    SweepConfig("layer_q5_b320_u80", "layer", "1,1", 320, 80, "q5_1"),
    SweepConfig("row_q5_b256_u64", "row", "1,1", 256, 64, "q5_1"),
    SweepConfig("row_q5_b320_u80", "row", "1,1", 320, 80, "q5_1"),
]


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def port_owner_pid(port: int) -> int | None:
    proc = run_cmd(
        [
            "bash",
            "-lc",
            f"ss -ltnp '( sport = :{port} )' 2>/dev/null | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p' | head -n1",
        ],
        timeout=10,
    )
    text = proc.stdout.strip()
    return int(text) if text.isdigit() else None


def pid_cmdline(pid: int) -> str:
    proc = run_cmd(["ps", "-p", str(pid), "-o", "cmd="], timeout=10)
    return proc.stdout.strip()


def stop_backend(port: int) -> None:
    pid = port_owner_pid(port)
    if pid is None:
        return
    cmd = pid_cmdline(pid)
    if "llama-server" not in cmd:
        raise RuntimeError(f"port {port} owned by unexpected process: {cmd}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        time.sleep(1)
        if port_owner_pid(port) is None:
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_model(base_url: str, expected_model: str, timeout: float = 240.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            payload = get_json(f"{base_url}/v1/models", timeout=3)
        except Exception:
            time.sleep(1)
            continue
        models = payload.get("data", [])
        if isinstance(models, list):
            for item in models:
                if isinstance(item, dict) and item.get("id") == expected_model:
                    return
        time.sleep(1)
    raise RuntimeError(f"backend did not become ready for model {expected_model}")


def start_backend(
    launcher: Path,
    config: SweepConfig,
    port: int,
    log_path: Path,
    model_alias: str,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(port)
    env["MODEL_ALIAS"] = model_alias
    env["CTX_SIZE"] = "131072"
    env["SPLIT_MODE"] = config.split_mode
    env["TENSOR_SPLIT"] = config.tensor_split
    env["THREADS"] = str(config.threads)
    env["BATCH"] = str(config.batch)
    env["UBATCH"] = str(config.ubatch)
    env["CACHE_TYPE_K"] = config.cache_type
    env["CACHE_TYPE_V"] = config.cache_type
    with log_path.open("ab") as handle:
        proc = subprocess.Popen(
            [str(launcher)],
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc


def benchmark_short_chat(base_url: str, model: str, runs: int, warmups: int) -> dict[str, Any]:
    latencies: list[float] = []
    prompt_tps: list[float] = []
    decode_tps: list[float] = []
    completion_tokens: list[int] = []
    prompt_tokens: list[int] = []
    structure_passes = 0

    for index in range(warmups + runs):
        prompt = SHORT_PROMPTS[index % len(SHORT_PROMPTS)]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 220,
            "temperature": 0.0,
            "stream": False,
        }
        started = time.perf_counter()
        result = post_json(f"{base_url}/v1/chat/completions", payload, timeout=300)
        elapsed = time.perf_counter() - started
        if index < warmups:
            continue
        usage = result.get("usage", {})
        timings = result.get("timings", {})
        message = ""
        choices = result.get("choices", [])
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = str(choice.get("message", {}).get("content", ""))
        if "def " in message and "assert" in message:
            structure_passes += 1
        latencies.append(elapsed)
        prompt_tps.append(float(timings.get("prompt_per_second", 0.0)))
        decode_tps.append(float(timings.get("predicted_per_second", 0.0)))
        prompt_tokens.append(int(usage.get("prompt_tokens", 0)))
        completion_tokens.append(int(usage.get("completion_tokens", 0)))

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "chat_ok": True,
        "chat_runs": runs,
        "chat_latency_s_avg": avg(latencies),
        "chat_prompt_tok_s_avg": avg(prompt_tps),
        "chat_decode_tok_s_avg": avg(decode_tps),
        "chat_prompt_tokens_avg": avg([float(v) for v in prompt_tokens]),
        "chat_completion_tokens_avg": avg([float(v) for v in completion_tokens]),
        "chat_structure_pass_rate": round(structure_passes / runs, 3) if runs else 0.0,
    }


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(part for part in parts if part)


def normalize_responses_request(data: dict[str, Any], model_alias: str) -> dict[str, Any]:
    payload = json.loads(json.dumps(data))
    payload["model"] = model_alias

    tools = payload.get("tools")
    if isinstance(tools, list):
        payload["tools"] = [tool for tool in tools if tool.get("type") == "function"]

    input_items = payload.get("input")
    if isinstance(input_items, list):
        lifted_messages: list[str] = []
        normalized_input: list[Any] = []
        for item in input_items:
            if not isinstance(item, dict):
                normalized_input.append(item)
                continue
            role = item.get("role")
            if item.get("type") == "message" and role in {"developer", "system"}:
                text = content_text(item.get("content"))
                if text:
                    lifted_messages.append(text)
                continue
            normalized_input.append(item)
        if lifted_messages:
            existing = payload.get("instructions")
            instructions = existing if isinstance(existing, str) else ""
            payload["instructions"] = "\n\n".join(
                part for part in [instructions, *lifted_messages] if part
            )
            payload["input"] = normalized_input

    payload["stream"] = False
    return payload


def extract_response_text(result: dict[str, Any]) -> str:
    output = result.get("output", [])
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    parts.append(content["text"])
        if parts:
            return "\n".join(parts)
    return ""


def benchmark_long_responses(base_url: str, model: str, captured_request: Path) -> dict[str, Any]:
    if not captured_request.exists():
        raise FileNotFoundError(f"captured Codex request not found: {captured_request}")
    data = json.loads(captured_request.read_text(encoding="utf-8"))
    payload = normalize_responses_request(data, model)

    started = time.perf_counter()
    result = post_json(f"{base_url}/v1/responses", payload, timeout=1800)
    elapsed = time.perf_counter() - started
    status = str(result.get("status", ""))
    usage = result.get("usage", {})
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    text = extract_response_text(result).strip().lower()
    success = status == "completed" and output_tokens > 0 and bool(text)
    prefill_tps = round(input_tokens / elapsed, 3) if elapsed > 0 and input_tokens > 0 else 0.0
    return {
        "codex_ok": success,
        "codex_status": status,
        "codex_latency_s": round(elapsed, 3),
        "codex_input_tokens": input_tokens,
        "codex_output_tokens": output_tokens,
        "codex_prefill_tok_s": prefill_tps,
        "codex_output_text": text[:200],
    }


def quality_rank(cache_type: str) -> int:
    if cache_type == "q8_0":
        return 3
    if cache_type == "q5_1":
        return 2
    return 1


def score_result(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        1 if result.get("codex_ok") else 0,
        1 if result.get("chat_ok") else 0,
        result.get("chat_structure_pass_rate", 0.0),
        quality_rank(result.get("cache_type", "")),
        result.get("codex_prefill_tok_s", 0.0),
        result.get("chat_decode_tok_s_avg", 0.0),
        result.get("chat_prompt_tok_s_avg", 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", default=str(DEFAULT_LAUNCHER))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--captured-request", default=str(DEFAULT_CAPTURED_REQUEST))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    launcher = Path(args.launcher).expanduser().resolve()
    captured_request = Path(args.captured_request).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else ROOT / "logs" / f"qwen36_128k_server_sweep_{now_stamp()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = FOCUSED_CONFIGS[: args.limit] if args.limit else FOCUSED_CONFIGS
    base_url = f"http://127.0.0.1:{args.port}"
    results: list[dict[str, Any]] = []

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "launcher": str(launcher),
                "model": args.model,
                "port": args.port,
                "runs": args.runs,
                "warmups": args.warmups,
                "captured_request": str(captured_request),
                "configs": [asdict(cfg) for cfg in configs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for config in configs:
        row: dict[str, Any] = {
            "name": config.name,
            "split_mode": config.split_mode,
            "tensor_split": config.tensor_split,
            "batch": config.batch,
            "ubatch": config.ubatch,
            "cache_type": config.cache_type,
            "threads": config.threads,
            "status": "failed",
        }
        log_path = out_dir / f"{config.name}.log"
        try:
            stop_backend(args.port)
            start_backend(launcher, config, args.port, log_path, args.model)
            wait_for_model(base_url, args.model, timeout=240)
            row.update(benchmark_short_chat(base_url, args.model, args.runs, args.warmups))
            row.update(benchmark_long_responses(base_url, args.model, captured_request))
            row["status"] = "ok" if row.get("codex_ok") and row.get("chat_ok") else "degraded"
        except Exception as exc:
            row["error"] = str(exc)
        finally:
            try:
                stop_backend(args.port)
            except Exception as exc:
                row.setdefault("stop_error", str(exc))
        results.append(row)
        print(
            f"{config.name}: {row['status']} "
            f"prefill={row.get('codex_prefill_tok_s', 0.0):.2f} "
            f"decode={row.get('chat_decode_tok_s_avg', 0.0):.2f} "
            f"cache={config.cache_type}",
            flush=True,
        )

    results_sorted = sorted(results, key=score_result, reverse=True)
    summary = {
        "model": args.model,
        "port": args.port,
        "launcher": str(launcher),
        "captured_request": str(captured_request),
        "results": results_sorted,
        "winner": results_sorted[0] if results_sorted else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = out_dir / "results.csv"
    fieldnames: list[str] = []
    for row in results_sorted:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)

    lines = [
        f"model={args.model}",
        f"launcher={launcher}",
        f"captured_request={captured_request}",
        "",
        "Top configs:",
    ]
    for row in results_sorted[:8]:
        lines.append(
            "- "
            + f"{row['name']}: status={row['status']}, "
            + f"prefill={row.get('codex_prefill_tok_s', 0.0):.2f} tok/s, "
            + f"decode={row.get('chat_decode_tok_s_avg', 0.0):.2f} tok/s, "
            + f"prompt={row.get('chat_prompt_tok_s_avg', 0.0):.2f} tok/s, "
            + f"cache={row.get('cache_type', '')}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {out_dir / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
