#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict


def post_json(url: str, payload: Dict[str, Any], timeout: float, api_key: str = "") -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple OpenAI-compatible benchmark client")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="local-qwen")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--save-json", default="")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }

    latencies = []
    output_tok_per_sec = []
    prompt_tokens = []
    completion_tokens = []

    total_runs = args.warmup_runs + args.runs
    for i in range(total_runs):
        started = time.perf_counter()
        try:
            result = post_json(args.url, payload, args.timeout, args.api_key)
        except urllib.error.URLError as exc:
            print(f"request failed on run {i + 1}: {exc}", file=sys.stderr)
            return 2
        elapsed = time.perf_counter() - started

        usage = result.get("usage", {})
        p_tok = int(usage.get("prompt_tokens", 0))
        c_tok = int(usage.get("completion_tokens", 0))

        if c_tok <= 0:
            content = ""
            choices = result.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = str(msg.get("content", ""))
            c_tok = max(1, len(content.split()))

        if i >= args.warmup_runs:
            latencies.append(elapsed)
            completion_tokens.append(c_tok)
            prompt_tokens.append(p_tok)
            output_tok_per_sec.append(c_tok / elapsed if elapsed > 0 else 0.0)

    summary = {
        "url": args.url,
        "model": args.model,
        "runs": args.runs,
        "prompt_tokens_avg": statistics.mean(prompt_tokens) if prompt_tokens else 0.0,
        "completion_tokens_avg": statistics.mean(completion_tokens) if completion_tokens else 0.0,
        "latency_s_avg": statistics.mean(latencies) if latencies else 0.0,
        "latency_s_p50": statistics.median(latencies) if latencies else 0.0,
        "tok_s_avg": statistics.mean(output_tok_per_sec) if output_tok_per_sec else 0.0,
        "tok_s_p50": statistics.median(output_tok_per_sec) if output_tok_per_sec else 0.0,
        "tok_s_max": max(output_tok_per_sec) if output_tok_per_sec else 0.0,
    }

    print(json.dumps(summary, indent=2))
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
