#!/usr/bin/env python3
"""Summarize an explicit newly generated host_staging_gpu evidence directory."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    exclusions = json.loads((root / "timing_exclusions.json").read_text()) if (root / "timing_exclusions.json").exists() else []
    def excluded(window, row):
        return any(e["window"] == window and all(e[k] == row[k] for k in ("depth", "cache", "repeat"))
                   for e in exclusions)
    windows = []
    for folder in sorted(root.glob("window-*")):
        screen = folder / "screen.json"
        if not screen.exists():
            continue
        row = {"name": folder.name, "variant": folder.name.split("-")[-1],
               "screen": json.loads(screen.read_text())}
        for source, key in (("lifecycle.json", "lifecycle"), ("agentic-summary.json", "app")):
            if (folder / source).exists():
                row[key] = json.loads((folder / source).read_text())
        timings = []
        if (folder / "router-events.jsonl").exists():
            for line in (folder / "router-events.jsonl").read_text().splitlines():
                event = json.loads(line)
                if event.get("event") == "router.response.completed":
                    timings.append(event["data"])
        if timings:
            native = [t["backend_timings"] for t in timings if isinstance(t.get("backend_timings"), dict)]
            decode_ms = sum(t.get("predicted_ms", 0) for t in native)
            decode_tokens = sum(max(0, t.get("predicted_n", 0) - 1) for t in native)
            row["app_timing"] = {"requests": len(timings), "timed_requests": len(native),
                "decode_tokens": decode_tokens, "decode_ms": decode_ms,
                "prefill_ms": sum(t.get("prompt_ms", 0) for t in native),
                "decode_tps": decode_tokens * 1000 / decode_ms if decode_ms and len(native) == len(timings) else None,
                "request_ms": sum(t.get("request_ms", 0) for t in timings),
                "queue_wait_ms": sum(t.get("queue_wait_ms", 0) for t in timings),
                "slot_prepare_ms": sum((t.get("slot") or {}).get("prepare_ms", 0) for t in timings),
                "max_input_tokens": max((t.get("backend", {}).get("usage", {}).get("input_tokens", 0)
                                         for t in timings), default=0),
                "first_activity_ms_median": statistics.median(t["first_activity_ms"] for t in timings
                    if t.get("first_activity_ms") is not None)
                    if any(t.get("first_activity_ms") is not None for t in timings) else None}
        windows.append(row)
    comparisons = []
    depths = sorted({r["depth"] for w in windows for r in w["screen"] if w["variant"] != "production"})
    for depth in depths:
        for cache in ("cold", "warm"):
            arms = {}
            all_rows = []
            for variant in ("control", "candidate"):
                selected = [(w, [r for r in w["screen"] if r["depth"] == depth and r["cache"] == cache])
                            for w in windows if w["variant"] == variant]
                all_rows.extend(r for _, rows in selected for r in rows)
                groups = [[r for r in rows if not excluded(w["name"], r)] for w, rows in selected]
                groups = [g for g in groups if g]
                rows = [r for g in groups for r in g]
                medians = [statistics.median(r["timings"]["predicted_per_second"] for r in g) for g in groups]
                arms[variant] = {"windows": len(groups), "requests": len(rows),
                    "decode_tps_window_medians": medians,
                    "decode_tps": statistics.mean(medians) if medians else None,
                    "wall_s": statistics.mean(statistics.median(r["wall_s"] for r in g) for g in groups) if groups else None}
            identical = bool(all_rows) and len({r["tokens_sha256"] for r in all_rows}) == 1
            left, right = arms["control"]["decode_tps"], arms["candidate"]["decode_tps"]
            comparisons.append({"depth": depth, "cache": cache, "arms": arms,
                "tokens_identical": identical,
                "decode_change_pct": 100 * (right / left - 1) if left and right and identical else None})
    lifecycle = [r for w in windows for r in w.get("lifecycle", [])]
    phases = [p for w in windows for case in w.get("app", []) for p in case["phases"]]
    return {"windows": windows, "comparisons": comparisons, "timing_exclusions": exclusions,
        "lifecycle_checks": len(lifecycle), "lifecycle_passed": sum(bool(r["passed"]) for r in lifecycle),
        "app_stages": len(phases),
        "app_passed_stages": sum(p["exit_code"] == 0 and p["grade"].get("checks") is not None
            and p["grade"]["checks"] == p["grade"]["passed_checks"] for p in phases),
        "limitation": "Fixed-work token equality and bounded synthetic task checks; few process windows, no general speed or quality claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.evidence)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as target:
            target.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
