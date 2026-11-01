#!/usr/bin/env python3
"""Guarded real-runtime/Marathon checks for the experimental DFlash host patch.

Uses a leased idle GPU, the pinned production image, synthetic inputs, and new
caches. Production configuration, existing conversations and caches are unused.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from marathon_app.catalog import Backend
from marathon_app.pool import acquire_pool_worker, release_pool_worker
import yaml

CONFIG = ROOT.parent / "gpu-control/llama-swap/config.yaml"
IMAGE = "sha256:8ac2e9bcd61aae5769c9f87ffe25ff7edbfefb2a8a0157e112d8da944775f4ff"
MODEL = "qwen3.8-27b-uncensored"
FRONTEND = Path("/home/deforest/.local/share/marathon/bin/codex")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Worker:
    def __init__(self, root, gpu, original, variant, build):
        self.out, self.gpu, self.original = root, gpu, original
        self.out.mkdir()
        (self.out / "slots").mkdir()
        self.variant, self.build = variant, build
        self.name = "marathon-host-staging-test-" + str(gpu)
        self.base = "http://127.0.0.1:19979"
        self.owned = False

    def http(self, path, body=None):
        if CONFIG.read_bytes() != self.original:
            raise RuntimeError("Production configuration changed during the run")
        request = urllib.request.Request(self.base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)

    def launch(self):
        memory = int(subprocess.check_output(["nvidia-smi", "-i", str(self.gpu),
            "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if memory > 100:
            raise RuntimeError("Selected GPU is occupied")
        names = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).splitlines()
        if self.name in names or "llama-swap-marathon-qwen3.8-27b-uncensored-" + str(self.gpu) in names:
            raise RuntimeError("Existing worker: refusing duplicate")
        entry = yaml.safe_load(self.original)["models"]["marathon-qwen3.8-27b-uncensored-" + str(self.gpu)]
        command = entry["cmd"].replace("${MODEL_ID}", "host-staging-test").replace("${PORT}", "19979")
        for key, value in entry.get("macros", {}).items():
            command = command.replace("${" + key + "}", str(value))
        argv = shlex.split(command)
        assert argv[:2] == ["docker", "run"] and IMAGE in argv
        argv[argv.index("--name") + 1] = self.name
        argv[argv.index("--alias") + 1] = MODEL
        for index, value in enumerate(argv):
            if value.endswith(":/cache"):
                argv[index] = str(self.out / "slots") + ":/cache"
        argv.remove("--rm")
        argv[2:2] = ["--detach"]
        if self.variant != "production":
            library = self.build / ("libdflash-host-" + self.variant + ".so")
            argv[2:2] = ["--volume", str(library) + ":/app/libdflash-host.so:ro", "--env",
                "LD_PRELOAD=/app/libdflash-host.so:/app/libfast-sampling.so:/app/libdraft-ubatch.so:/app/libq8-compact.so"]
        dump(self.out / "launch.json", argv)
        subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)
        self.owned = True
        for _ in range(180):
            state = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Running}}", self.name], text=True).strip()
            if state != "true":
                raise RuntimeError("Scratch worker exited; see its server.log")
            try:
                if self.http("/health").get("status") == "ok":
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("Scratch worker startup timeout")
        logs = subprocess.check_output(["docker", "logs", self.name], stderr=subprocess.STDOUT, text=True)
        if self.variant != "production":
            assert logs.count("dflash-host-staging: " + self.variant + " constructor active") == 1
        assert "draft-ubatch: dflash microbatch 256 -> 64; target unchanged" in logs
        inspection = json.loads(subprocess.check_output(["docker", "inspect", self.name], text=True))[0]
        assert inspection["Image"] == IMAGE
        assert inspection["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] == [str(self.gpu)]
        print(json.dumps({"window": self.out.name, "variant": self.variant, "status": "ready"}), flush=True)

    def close(self):
        if self.owned:
            with (self.out / "server.log").open("w") as log:
                subprocess.run(["docker", "logs", self.name], stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(["docker", "stop", "--time", "30", self.name], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["docker", "rm", self.name], check=True, stdout=subprocess.DEVNULL)
            self.owned = False


def complete(worker, prompt, **overrides):
    payload = dict(prompt=prompt, n_predict=128, temperature=0,
        seed=424242, cache_prompt=True, return_tokens=True, id_slot=0)
    payload.update(overrides)
    return worker.http("/completion", payload)


def screen(worker, depths, repeats):
    archive = "\n".join(f"Synthetic record {i:05d}: region=amber; revision={i % 37}; checksum=fixture-{i:05d}." for i in range(18000))
    text = archive + "\nEnd of archive. Explain how a bounded work queue should handle cancellation, timeouts and shutdown. Use concrete examples."
    rendered = worker.http("/apply-template", {"messages": [{"role": "user", "content": text}],
        "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    tokens = worker.http("/tokenize", {"content": rendered, "add_special": False})["tokens"]
    rows = []
    for depth in depths:
        assert 1024 <= depth < len(tokens)
        prompt = tokens[:depth - 512] + tokens[-512:]
        for repeat in range(-1, repeats):
            started = time.monotonic()
            response = complete(worker, prompt, cache_prompt=repeat >= 0, ignore_eos=True)
            elapsed = time.monotonic() - started
            row = {"depth": depth, "repeat": repeat, "cache": "cold" if repeat < 0 else "warm",
                   "wall_s": elapsed, "timings": response.get("timings"),
                   "tokens_sha256": sha(response.get("tokens")), "token_count": len(response.get("tokens", []))}
            assert row["token_count"] == 128
            rows.append(row)
            dump(worker.out / "screen.json", rows)
            print(json.dumps({"window": worker.out.name, **row}), flush=True)
    return rows


def lifecycle(worker, donor=None):
    results = []
    prompt = worker.http("/apply-template", {"messages": [{"role": "user", "content":
        "\n".join(f"Synthetic record {i}: fixture-only, region amber." for i in range(350)) +
        "\nFinal marker: CORAL-7182. Repeat only the final marker."}],
        "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    response = complete(worker, prompt, n_predict=32, cache_prompt=False)
    assert "CORAL-7182" in response["content"]
    saved = worker.http("/slots/0?action=save", {"filename": "synthetic.bin"})
    worker.http("/slots/0?action=erase", {})
    restored = worker.http("/slots/0?action=restore", {"filename": "synthetic.bin"})
    repeat = complete(worker, prompt, n_predict=32)
    assert "CORAL-7182" in repeat["content"] and repeat["timings"]["cache_n"] > 1000
    results.append({"check": "own_snapshot", "passed": True, "saved": saved, "restored": restored})
    if donor:
        for path in (donor / "slots").glob("synthetic.bin*"):
            shutil.copy2(path, worker.out / "slots" / ("donor.bin" + path.name[len("synthetic.bin"):]))
        worker.http("/slots/0?action=erase", {})
        worker.http("/slots/0?action=restore", {"filename": "donor.bin"})
        response = complete(worker, prompt, n_predict=32)
        assert "CORAL-7182" in response["content"] and response["timings"]["cache_n"] > 1000
        results.append({"check": "cross_runtime_snapshot", "passed": True})
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (448, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 50, 160, 185), fill="red")
    draw.ellipse((270, 50, 405, 185), fill="blue")
    buffer = io.BytesIO(); image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [{"role": "user", "content": [{"type": "text", "text":
        "Describe the two colored shapes and their left/right positions in one sentence."},
        {"type": "image_url", "image_url": {"url": url}}]}]
    for cache in (False, True):
        response = worker.http("/v1/chat/completions", {"model": MODEL, "messages": messages,
            "max_tokens": 96, "temperature": 0, "cache_prompt": cache,
            "chat_template_kwargs": {"enable_thinking": False}})
        content = response["choices"][0]["message"]["content"].lower()
        assert all(word in content for word in ("red", "square", "left", "blue", "circle", "right")), content
        results.append({"check": "image_warm" if cache else "image_cold", "passed": True,
                        "timings": response.get("timings")})
    messages += [response["choices"][0]["message"], {"role": "user", "content": "Which shape is blue? Reply briefly."}]
    response = worker.http("/v1/chat/completions", {"model": MODEL, "messages": messages,
        "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    assert "circle" in response["choices"][0]["message"]["content"].lower()
    results.append({"check": "image_followup", "passed": True})
    dump(worker.out / "lifecycle.json", results)
    return results


def freeze_source(destination, revision=None):
    destination.mkdir()
    prefixes = ["marathon_app", "config", "scripts/routers"]
    command = ["git", "ls-tree", "-r", "--name-only", revision, "--", *prefixes] if revision else ["git", "ls-files", "--", *prefixes]
    files = subprocess.check_output(command, cwd=ROOT, text=True).splitlines()
    hashes = {}
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = subprocess.check_output(["git", "show", revision + ":" + relative], cwd=ROOT) if revision else (ROOT / relative).read_bytes()
        target.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    prompt = "codex/codex-rs/models-manager/prompt.md"
    if revision:
        codex_revision = subprocess.check_output(["git", "ls-tree", revision, "codex"], cwd=ROOT, text=True).split()[2]
        content = subprocess.check_output(["git", "show", codex_revision + ":codex-rs/models-manager/prompt.md"], cwd=ROOT / "codex")
    else:
        content = (ROOT / prompt).read_bytes()
    target = destination / prompt
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    hashes[prompt] = hashlib.sha256(content).hexdigest()
    (destination / ".marathon").mkdir()
    (destination / ".marathon/venv").symlink_to(ROOT / ".marathon/venv", target_is_directory=True)
    dump(destination / "source-manifest.json", {"revision": revision, "sha256": hashes})
    return destination


def app(worker, label, cases, source):
    path = ROOT.parent / "gpu-control/benchmarks/qwen38-short-agentic-ab.py"
    spec = importlib.util.spec_from_file_location("host_staging_short_agent", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.base.MARATHON = source
    agent = module.ShortAgent(worker, argparse.Namespace(gpu=worker.gpu, label=label,
        cases=cases, stages=2, phase_timeout=180, context=196000, max_output_tokens=8192,
        backend_model_id=MODEL))
    try:
        agent.run()
        return agent.results
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--gpu", type=int, choices=(1, 3), default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[8192, 32768, 131072, 186000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--windows", nargs="+", choices=("production", "control", "candidate"),
                        default=["production", "control", "candidate", "candidate", "control"])
    parser.add_argument("--skip-app", action="store_true")
    parser.add_argument("--app-cases", nargs="+", choices=("cache", "orders"), default=["cache", "orders"])
    parser.add_argument("--marathon-revision", help="Freeze this source revision; otherwise snapshot current project code")
    args = parser.parse_args()
    if not args.run_gpu:
        print(json.dumps({"status": "planned", "gpu": args.gpu, "windows": args.windows,
                          "depths": args.depths, "app_cases": args.app_cases})); return
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT):
        parser.error("choose new evidence directory outside Marathon")
    args.output.mkdir(parents=True)
    # Keep library/client helpers from consulting personal caches or Docker auth.
    for variable, directory in (("HOME", "controller-home"), ("XDG_CONFIG_HOME", "controller-config"),
                                ("XDG_CACHE_HOME", "controller-cache"), ("XDG_STATE_HOME", "controller-state"),
                                ("DOCKER_CONFIG", "docker-config")):
        path = args.output.resolve() / directory
        path.mkdir()
        os.environ[variable] = str(path)
    original = CONFIG.read_bytes()
    source = freeze_source(args.output.resolve() / "marathon-source", args.marathon_revision)
    frontend_sha256 = hashlib.sha256(FRONTEND.read_bytes()).hexdigest()
    backend = Backend(id="llama-swap-qwen3.8-uncensored-pool", display_name="Host staging validation",
        server=None, pool_models=("marathon-qwen3.8-27b-uncensored-" + str(args.gpu),),
        identity_config=CONFIG, pool_min_free_vram_mib=23500)
    lease, _ = acquire_pool_worker(backend, Path(f"/run/user/{os.getuid()}/marathon"), "host-staging-validation")
    results = []
    donor = None
    done = set()
    def interrupted(*unused):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        dump(args.output / "protocol.json", {"image": IMAGE, "config_sha256": hashlib.sha256(original).hexdigest(),
            "gpu": args.gpu, "depths": args.depths, "windows": args.windows, "repeats": args.repeats,
            "build": json.loads((args.build / "manifest.json").read_text()),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "marathon_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "frozen_source_revision": args.marathon_revision, "frontend_sha256": frontend_sha256,
            "scope": "fresh synthetic inputs, isolated real Marathon frontend/router, unchanged production image and GPU kernels"})
        for index, variant in enumerate(args.windows):
            worker = Worker(args.output / f"window-{index}-{variant}", args.gpu, original, variant, args.build)
            row = {"window": index, "variant": variant}
            results.append(row)
            try:
                worker.launch()
                row["screen"] = screen(worker, [8192] if variant == "production" else args.depths, args.repeats)
                if variant not in done:
                    row["lifecycle"] = lifecycle(worker, donor if variant == "candidate" else None)
                    if variant == "production": donor = worker.out
                    if variant in {"control", "candidate"} and not args.skip_app:
                        if hashlib.sha256(FRONTEND.read_bytes()).hexdigest() != frontend_sha256:
                            raise RuntimeError("Frontend changed during the comparison")
                        row["app"] = app(worker, "host-staging-" + variant, args.app_cases, source)
                    done.add(variant)
                row["status"] = "complete"
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                worker.close()
                dump(args.output / "results.json", results)
    finally:
        release_pool_worker(lease)
        assert CONFIG.read_bytes() == original


if __name__ == "__main__":
    main()
