#!/usr/bin/env python3
"""Live public-page fetches and model-picker latency through the real router.

The temporary authenticated evaluation route invokes Marathon's actual managed
tool path. It is installed only in the evaluator's child router process.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from aiohttp import ClientSession

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
URL = "https://en.wikipedia.org/wiki/List_of_Unicode_characters"


async def exercise(runtime, output, repeats):
    rows = []
    async with ClientSession(headers=runtime.router_headers) as client:
        for repeat in range(repeats):
            marker = output / "extracting"
            marker.unlink(missing_ok=True)
            started = time.perf_counter()
            async def fetch():
                async with client.post(runtime.router_url + "/_eval/fetch", json={"url": URL, "scope": str(repeat)}) as r:
                    r.raise_for_status()
                    return await r.json()
            task = asyncio.create_task(fetch())
            while not marker.exists() and not task.done():
                await asyncio.sleep(0.005)
            probe = time.perf_counter()
            async with client.get(runtime.router_url + "/v1/models") as r:
                r.raise_for_status()
                catalog = await r.json()
                assert catalog.get("models"), "Model picker must remain populated"
            picker_ms = (time.perf_counter() - probe) * 1000
            result = await task
            text = result["output"]
            assert "Unicode" in text and not text.startswith("web_fetch failed"), text[:200]
            row = {"repeat": repeat, "passed": True, "fetch_ms": (time.perf_counter() - started) * 1000,
                   "model_picker_ms": picker_ms, "converted": marker.exists(),
                   "output_sha256": hashlib.sha256(text.encode()).hexdigest()}
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-network", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--web-source", type=Path, help="alternate module for an isolated A/B child")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.run_network or args.repeats < 2:
        parser.error("pass --run-network and use at least two repeats")
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="marathon-web-response-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    source = args.web_source.resolve() if args.web_source else ROOT / "scripts/routers/marathon_web_search.py"
    wrapper = output / "router-python"
    wrapper.write_text(f"#!{sys.executable}\n" + f'''
import sys, json, importlib.util
from pathlib import Path
sys.argv = sys.argv[1:]
__file__ = sys.argv[0]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, {str(ROOT)!r})
spec = importlib.util.spec_from_file_location("marathon_web_search", {str(source)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
original_html = module._html_to_markdown
def measured_html(*args):
    Path({str(output / "extracting")!r}).touch()
    return original_html(*args)
module._html_to_markdown = measured_html
__name__ = "evaluation_router"
sys.modules[__name__] = sys.modules["__main__"]
exec(compile(Path(__file__).read_text(), __file__, "exec"), globals())
original_app = build_app
def evaluation_app(state):
    app = original_app(state)
    async def fetch(request):
        data = await request.json()
        call = {{"type":"function_call", "name":"web_fetch", "call_id":"eval",
                 "arguments":json.dumps({{"url":data["url"]}})}}
        result = await state._execute_managed_call_cached(call, 0, data["scope"])
        return web.json_response(result)
    app.router.add_post("/_eval/fetch", fetch)
    return app
build_app = evaluation_app
main()
''')
    wrapper.chmod(0o700)
    os.environ["MARATHON_ROUTER_PYTHON"] = str(wrapper)
    os.environ["MARATHON_RUNS_DIR"] = str(output / "runs")
    from marathon_app.catalog import discover_models
    from marathon_app.runtime import Runtime, automatic_launch_instance, load_selection
    from marathon_app.ui import _initial_selection
    selection = _initial_selection(discover_models(), load_selection())
    runtime = Runtime(selection.model, selection.profile, automatic_launch_instance())
    print(f"Evidence: {output}", flush=True)
    try:
        runtime.start(lazy_pool=True)
        asyncio.run(exercise(runtime, output, args.repeats))
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    main()
