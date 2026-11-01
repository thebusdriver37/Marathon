#!/usr/bin/env python3
"""Matched completion benchmarks. Planning is offline; --run-inference sends requests.

Use an independently reserved scratch endpoint. This client does not acquire a
GPU lease, start workers, inspect local configuration, or change power settings.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit


def post_json(url: str, payload: Dict[str, Any], timeout: float, api_key: str = "") -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def duration(value: Any) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def metrics(response: dict, elapsed: float) -> dict:
    """Keep request throughput separate from native llama.cpp decode timing."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("Missing completion message")
    message = dict(choice["message"])
    # Request-specific call IDs are not model output equivalence evidence.
    if "tool_calls" in message:
        message["tool_calls"] = [
            {k: v for k, v in call.items() if k != "id"}
            for call in message["tool_calls"]
        ]
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    if not isinstance(usage, dict) or not isinstance(timings, dict):
        raise ValueError("Malformed usage or timings")
    output = count(usage.get("completion_tokens"))
    predicted = count(timings.get("predicted_n"))
    decode_ms = duration(timings.get("predicted_ms"))
    prefill_ms = duration(timings.get("prompt_ms"))
    prefill = count(timings.get("prompt_n"))
    drafted = count(timings.get("draft_n"))
    accepted = count(timings.get("draft_n_accepted"))
    details = usage.get("prompt_tokens_details") or {}
    cached = count(details.get("cached_tokens")) if isinstance(details, dict) else None
    if cached is None:
        cached = count(timings.get("cache_n"))
    finish = choice.get("finish_reason")
    # Native llama.cpp includes the first generated token in the prefill phase.
    decode_tokens = max(0, predicted - 1) if predicted is not None else None
    return {
        "latency_s": elapsed,
        "prompt_tokens": count(usage.get("prompt_tokens")),
        "completion_tokens": output,
        "cached_tokens": cached,
        "request_tps": output / elapsed if output is not None and elapsed > 0 else None,
        "backend_decode_tps": decode_tokens * 1000 / decode_ms
            if decode_tokens is not None and decode_ms else None,
        "backend_prefill_tps": prefill * 1000 / prefill_ms
            if prefill is not None and prefill_ms else None,
        "draft_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "draft_acceptance": accepted / drafted
            if drafted and accepted is not None and accepted <= drafted else None,
        "finish_reason": finish,
        "complete": finish in {"stop", "tool_calls"},
        "output_sha256": digest({"message": message, "finish_reason": finish}),
    }


MEASURES = ("latency_s", "request_tps", "backend_decode_tps", "backend_prefill_tps",
            "prompt_tokens", "completion_tokens", "cached_tokens", "draft_acceptance")


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["status"] == "ok"]
    result = {"runs": len(rows), "successful_requests": len(successful),
              "complete_requests": sum(row["complete"] for row in successful)}
    for key in MEASURES:
        values = [row[key] for row in successful if row[key] is not None]
        # Missing timings or failed requests must not silently improve the average.
        complete = bool(rows) and len(values) == len(rows)
        result[key] = {"coverage": len(values),
                       "mean": statistics.mean(values) if complete else None,
                       "median": statistics.median(values) if complete else None}
    return result


def compare(rows: list[dict], runs: int) -> dict:
    pairs = []
    for index in range(runs):
        arms = {row["arm"]: row for row in rows if row["run"] == index}
        left, right = arms["baseline"], arms["candidate"]
        ok = left["status"] == right["status"] == "ok"
        identical = left["output_sha256"] == right["output_sha256"] if ok else None
        complete = ok and left["complete"] and right["complete"]
        matched = ok and all(left[key] is not None and left[key] == right[key]
                             for key in ("prompt_tokens", "cached_tokens"))
        pairs.append({"run": index, "outputs_identical": identical, "complete": complete,
                      "input_and_cache_counts_matched": matched,
                      "latency_delta_s": right["latency_s"] - left["latency_s"]
                      if complete and identical and matched else None})
    deltas = [pair["latency_delta_s"] for pair in pairs if pair["latency_delta_s"] is not None]
    return {"pairs": pairs, "all_outputs_identical": all(p["outputs_identical"] is True for p in pairs),
            "all_complete": all(p["complete"] for p in pairs),
            "all_input_and_cache_counts_matched": all(p["input_and_cache_counts_matched"] for p in pairs),
            "paired_latency_delta_s_median": statistics.median(deltas) if len(deltas) == runs else None}


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:19998/v1/chat/completions")
    parser.add_argument("--candidate-url")
    parser.add_argument("--model", default="local-qwen")
    parser.add_argument("--candidate-model")
    parser.add_argument("--baseline-id", default="unspecified")
    parser.add_argument("--candidate-id", default="unspecified")
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt")
    prompt_source.add_argument("--prompt-file", type=Path, help="Explicit public or synthetic UTF-8 fixture")
    parser.add_argument("--max-tokens", type=positive_int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cache-mode", choices=("cold", "warm", "server-default"), default="cold")
    parser.add_argument("--require-identical", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--save-json")
    parser.add_argument("--api-key-env", default="")
    args = parser.parse_args(argv)
    if args.warmup_runs < 0 or not duration(args.timeout):
        parser.error("warmup-runs must be nonnegative and timeout must be finite and positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    if args.cache_mode == "warm" and args.warmup_runs < 1:
        parser.error("warm cache mode needs at least one warmup request per endpoint")
    if args.require_identical and not args.candidate_url:
        parser.error("--require-identical needs --candidate-url")
    if args.candidate_model and not args.candidate_url:
        parser.error("--candidate-model needs --candidate-url")
    if args.candidate_url and (args.baseline_id == "unspecified" or args.candidate_id == "unspecified"):
        parser.error("paired runs need --baseline-id and --candidate-id (build/patch identities)")
    for url in (args.url, args.candidate_url):
        if url is None:
            continue
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            parser.error("endpoints must be HTTP(S) URLs")
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            parser.error("endpoint URLs cannot contain credentials, query strings, or fragments; use --api-key-env")

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": args.max_tokens, "temperature": args.temperature,
               "seed": args.seed, "stream": False}
    if args.cache_mode != "server-default":
        payload["cache_prompt"] = args.cache_mode == "warm"
    arms = [("baseline", args.url, args.model, args.baseline_id)]
    if args.candidate_url:
        arms.append(("candidate", args.candidate_url, args.candidate_model or args.model, args.candidate_id))
    report = {"schema_version": 2, "status": "planned", "request_sha256": digest(payload),
              "arms": [{"arm": name, "url": url, "model": model, "identity": identity}
                       for name, url, model, identity in arms],
              "runs": args.runs, "warmup_runs": args.warmup_runs, "cache_mode": args.cache_mode,
              "max_tokens": args.max_tokens, "temperature": args.temperature, "seed": args.seed,
              "timing_scope": "nonstreaming HTTP request; backend phase rates only when reported",
              "quality_scope": "output equality and finish reasons, not task correctness",
              "backend_timing_contract": "llama.cpp predicted_n includes first token in prefill"}
    if not args.run_inference:
        print(json.dumps(report, indent=2))
        return 0

    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    if args.api_key_env and not api_key:
        parser.error("the named API key environment variable is empty")
    # Reserve the evidence path before inference; never overwrite an earlier run.
    output = open(args.save_json, "x", encoding="utf-8") if args.save_json else None
    rows = []
    warmups = []
    try:
        for index in range(-args.warmup_runs, args.runs):
            for name, url, model, _ in (arms if index % 2 == 0 else arms[::-1]):
                started = time.perf_counter()
                row = {"arm": name, "run": index}
                try:
                    response = post_json(url, {**payload, "model": model}, args.timeout, api_key)
                    row.update(metrics(response, time.perf_counter() - started), status="ok")
                except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    row.update(status="error", error_type=type(exc).__name__,
                               http_status=exc.code if isinstance(exc, urllib.error.HTTPError) else None)
                (rows if index >= 0 else warmups).append(row)
        report.update(status="measured", rows=rows, warmups=warmups,
                      summary={name: summarize([row for row in rows if row["arm"] == name])
                               for name, _, _, _ in arms})
        failed = any(row["status"] != "ok" or not row["complete"] for row in rows + warmups)
        if len(arms) == 2:
            report["comparison"] = compare(rows, args.runs)
            failed |= args.require_identical and not report["comparison"]["all_outputs_identical"]
            failed |= not report["comparison"]["all_input_and_cache_counts_matched"]
        report["checks_passed"] = not failed
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if output:
            output.write(encoded)
        print(encoded, end="")
        return 1 if failed else 0
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    sys.exit(main())
