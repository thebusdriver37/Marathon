#!/usr/bin/env python3
"""CPU-only differential for dflash-host-staging.patch against a source checkout.

Pass the pinned llama.cpp source with Marathon's 14 default patches applied.
Only source is read. The candidate and test executable are built in a temporary
directory; no model, CUDA library, worker, or local runtime config is loaded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "scripts/experiments/dflash-host-staging.patch"
FIXTURE = ROOT / "tests/fixtures/dflash_host_staging.cpp"


def methods(source: str) -> str:
    struct = source.split("struct common_speculative_impl_draft_dflash :", 1)[1]
    process = struct.split("    bool process(const llama_batch & batch_in) override {", 1)[1]
    process = process.split("\n    void draft(", 1)[0]
    draft = struct.split("    void draft(common_speculative_draft_params_vec & dparams) override {", 1)[1]
    draft = draft.split("        // decode all sequence's noise block", 1)[0]
    return ("bool process(const llama_batch & batch_in) {" + process
            + "\nvoid build_draft(common_speculative_draft_params_vec & dparams) {" + draft + "}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    before = (source / "common/speculative.cpp").read_text()
    with tempfile.TemporaryDirectory(prefix="marathon-dflash-cpu-") as folder:
        work = Path(folder)
        (work / "common").mkdir()
        (work / "src").symlink_to(source / "src", target_is_directory=True)
        candidate = work / "common/speculative.cpp"
        candidate.write_text(before)
        subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=work, check=True)
        subprocess.run(["git", "apply", str(PATCH)], cwd=work, check=True)
        after = candidate.read_text()
        includes = ["-I" + str(source / path) for path in ("include", "common", "ggml/include", "src")]
        subprocess.run([args.cxx, "-std=c++17", "-fsyntax-only", *includes, str(candidate)], check=True)
        fixture = FIXTURE.read_text().replace("// BEFORE_METHODS", methods(before)).replace(
            "// AFTER_METHODS", methods(after))
        driver = work / "check.cpp"
        driver.write_text(fixture)
        executable = work / "check"
        subprocess.run([args.cxx, "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
                        "-fno-omit-frame-pointer", str(driver), "-o", str(executable)], check=True)
        result = subprocess.run([str(executable)], text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"CPU differential failed ({result.returncode}):\n{result.stderr}")
        print(json.dumps({"scope": "CPU host staging only; no model or GPU execution",
                          "baseline_sha256": hashlib.sha256(before.encode()).hexdigest(),
                          "candidate_sha256": hashlib.sha256(after.encode()).hexdigest(),
                          "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
                          "result": result.stdout.strip()}, indent=2))


if __name__ == "__main__":
    main()
