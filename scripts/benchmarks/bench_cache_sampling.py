#!/usr/bin/env python3
"""Exercise disk-cache reuse or grammar overhead on an isolated llama-server.

Run through gpu-control's guarded benchmark lifecycle with BENCH_HOOK set to
["python3", ".../bench_cache_sampling.py", MODE, TOKEN_FIXTURE].
Never point this at a production worker: cache mode erases scratch slot zero.
"""

import hashlib
import asyncio
import json
import sys
import time
import urllib.error
from pathlib import Path

from bench_openai import post_json


def main() -> None:
    mode, fixture, base, output = sys.argv[1:]
    if base not in ("http://127.0.0.1:19998", "http://127.0.0.1:19999"):
        raise ValueError("Only the reserved scratch ports are supported")
    body = json.loads(Path(fixture).read_text())
    prompt = body["prompt"]
    if not isinstance(prompt, list):
        prompt = post_json(base + "/tokenize", {"content": prompt, "add_special": True}, 60)["tokens"]
    rows = []

    def call(label, endpoint, payload):
        start = time.monotonic()
        result = post_json(base + endpoint, payload, 1800)
        row = {"label": label, "wall_ms": (time.monotonic() - start) * 1000,
               "timings": result.get("timings"),
               "tokens_hash": hashlib.sha256(json.dumps(result.get("tokens")).encode()).hexdigest(),
               "result": result if endpoint != "/completion" else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
        Path(output, "diagnostic.json").write_text(json.dumps(rows, indent=2) + "\n")
        return result

    def complete(label, tokens, **overrides):
        return call(label, "/completion", {"prompt": tokens, "id_slot": 0,
                    "n_predict": 128, "temperature": 0, "seed": 424242,
                    "cache_prompt": True, "return_tokens": True, "ignore_eos": True,
                    **overrides})

    if mode in {"cache", "cache-verify"}:
        if len(prompt) < 12100:
            raise ValueError("Cache diagnostic needs at least 12,100 prompt tokens")
        prefix = prompt[:12000]
        append = prompt[:12100]
        branch = prompt[:11760] + prompt[-100:]
        # The saved endpoint is later than the branch point, as happens when
        # persisted client history differs from the live backend tool history.
        complete("prefix-cold", prefix, cache_prompt=False, n_predict=0)
        call("save", "/slots/0?action=save", {"filename": "branch.bin"})
        live = complete("live-branch", branch)
        call("erase", "/slots/0?action=erase", {})
        call("restore-for-append", "/slots/0?action=restore", {"filename": "branch.bin"})
        complete("disk-append", append)
        call("erase", "/slots/0?action=erase", {})
        call("restore-for-branch", "/slots/0?action=restore", {"filename": "branch.bin"})
        disk = complete("disk-branch", branch)
        cold = complete("cold-branch-control", branch, cache_prompt=False)
        if not live.get("tokens") or not live["tokens"] == disk.get("tokens") == cold.get("tokens"):
            raise RuntimeError("Cached and cold greedy continuations differ")
        if mode == "cache-verify":
            if live["timings"]["cache_n"] <= 0 or disk["timings"]["cache_n"] != live["timings"]["cache_n"]:
                raise RuntimeError("Disk restore lost a reusable live checkpoint")
            sidecar = Path(output, "slots", "branch.bin.checkpoints")
            original = sidecar.read_bytes()
            held = sidecar.with_suffix(".checkpoints.held")
            if held.exists():
                raise RuntimeError("Refusing to overwrite a held scratch snapshot")
            # Container-created files can be root-owned. Rename the intact
            # control and write separate test fixtures in our scratch directory.
            sidecar.rename(held)
            try:
                for label, invalid in (
                    ("truncated", original[:20]),
                    ("checksum", original[:-1] + bytes([original[-1] ^ 1])),
                    ("prompt-mismatch", original[:28] + bytes([original[28] ^ 1]) + original[29:]),
                ):
                    sidecar.write_bytes(invalid)
                    try:
                        call(label, "/slots/0?action=restore", {"filename": "branch.bin"})
                    except urllib.error.HTTPError as error:
                        detail = error.read().decode()
                        if error.code != 400:
                            raise
                        row = {"label": label, "expected_status": error.code, "error": detail}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                    else:
                        raise RuntimeError(f"Accepted corrupt checkpoint: {label}")
            finally:
                held.replace(sidecar)
            # A missing optional sidecar represents the previous on-disk format.
            sidecar.rename(held)
            try:
                call("legacy-restore", "/slots/0?action=restore", {"filename": "branch.bin"})
                legacy = complete("legacy-append", append)
            finally:
                held.rename(sidecar)
            call("validated-restore", "/slots/0?action=restore", {"filename": "branch.bin"})
            restored = complete("validated-append", append)
            if legacy.get("tokens") != restored.get("tokens") or restored["timings"]["cache_n"] != 12000:
                raise RuntimeError("Exact append compatibility failed")
            # Exercise Marathon's real bundle rename and metadata accounting,
            # not just llama-server's slot API with a fixed filename.
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from marathon_app.checkpoints import RollingCheckpointStore

            store = RollingCheckpointStore(Path(output), Path(output), max_count=2,
                                           max_bytes=32 * 1024**3, ttl_seconds=172800)
            committed = store.commit(
                profile_slug="slots", profile_alias="scratch",
                prompt_cache_key="scratch-cache-verification", backend_cache_id="scratch",
                scaffold_fingerprint="scratch", response_id="scratch-response",
                context_tokens=12000, conversation_item_count=2,
                conversation_prefix_hash="a" * 64, pending_filename="branch.bin")
            if committed["status"] != "saved":
                raise RuntimeError(f"Marathon bundle commit failed: {committed}")
            call("marathon-bundle-restore", "/slots/0?action=restore",
                 {"filename": committed["snapshot_filename"]})
            bundled = complete("marathon-bundle-branch", branch)
            if bundled.get("tokens") != disk.get("tokens") or bundled["timings"]["cache_n"] != disk["timings"]["cache_n"]:
                raise RuntimeError("Marathon bundle lost rewind state")
    elif mode == "cache-restore":
        snapshots = list(Path(output, "slots").glob("conversation__*.bin"))
        if len(snapshots) != 1 or len(prompt) < 12100:
            raise ValueError("Portable restore requires one verified bundle and its token fixture")
        call("new-worker-restore", "/slots/0?action=restore", {"filename": snapshots[0].name})
        branch = prompt[:11760] + prompt[-100:]
        restored = complete("new-worker-branch", branch)
        cold = complete("new-worker-cold-control", branch, cache_prompt=False)
        if restored["timings"]["cache_n"] != 11740 or restored.get("tokens") != cold.get("tokens"):
            raise RuntimeError("Fresh-worker restore lost reusable state or changed greedy output")
    elif mode == "starter":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "routers"))
        import codex_local_router as router

        profile = router.ModelProfile("slots", "scratch", "Scratch", "Diagnostic", (),
                                      base, 196000, 170000, 160000)
        # Use the actual starter-cache implementation with isolated paths and
        # scratch HTTP transport, without initializing production routing.
        state = object.__new__(router.RouterState)
        state.slot_id = 0
        state.slot_save_root = Path(output)
        state.backend_cache_id = "scratch"
        state.starter_cache_enabled = True
        state.starter_cache_max_count = 2
        state.starter_cache_max_bytes = 1024**3

        async def request_json(_profile, method, endpoint, payload=None, **_kwargs):
            if method != "POST":
                raise ValueError("Starter diagnostic supports POST only")
            return post_json(base + endpoint, payload or {}, 1800)

        state._request_json = request_json
        request = {"instructions": router._base_instructions(), "tools": [{
            "type": "function", "name": "lookup", "description": "Look up a local item.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                           "required": ["name"], "additionalProperties": False}}]}
        built = asyncio.run(state.prepare_starter_cache(profile, request))
        if built["status"] != "built":
            raise RuntimeError(f"Starter build failed: {built}")
        chat = router._starter_scaffold_chat_body(request, "Explain how a hash table works.")
        chat["add_generation_prompt"] = True
        rendered = post_json(base + "/apply-template", chat, 60)["prompt"]
        tokens = post_json(base + "/tokenize", {"content": rendered, "add_special": True,
                                               "parse_special": True}, 60)["tokens"]
        cold = complete("starter-cold-control", tokens, cache_prompt=False)
        call("erase", "/slots/0?action=erase", {})
        restored = asyncio.run(state.prepare_starter_cache(profile, request, restore_only=True))
        if restored["status"] != "restored":
            raise RuntimeError(f"Starter restore failed: {restored}")
        cached = complete("starter-restored", tokens)
        if cold.get("tokens") != cached.get("tokens") or cached["timings"]["cache_n"] <= 0:
            raise RuntimeError("Starter cache did not preserve output and reuse tokens")
    elif mode == "sampling":
        complete("warmup", prompt, n_predict=16)
        # Excludes only NUL; output identity verifies the constraint was inert
        # for this fixture. Production tool constraints are never disabled.
        variants = [("backend", {}), ("cpu", {"backend_sampling": False}),
                    ("cpu-grammar", {"grammar": 'root ::= [^\\x00]*'})]
        for repeat in range(4):
            for name, overrides in variants[::1 if repeat % 2 == 0 else -1]:
                complete(f"{name}-{repeat}", prompt, n_predict=512,
                         temperature=body.get("temperature", 0), **overrides)
    else:
        raise ValueError("Expected cache, cache-verify, cache-restore, starter, or sampling mode")


if __name__ == "__main__":
    main()
