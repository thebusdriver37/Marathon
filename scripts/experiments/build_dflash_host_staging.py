#!/usr/bin/env python3
"""Build pinned control/candidate DFlash constructor preloads without CUDA changes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


EXPECTED = {
    "include/llama.h": "3d1b18eda626c1b9ecf5bda0798e65b974a8f70f30d56726610633a980cb4160",
    "common/common.h": "2d189832492ca9899e2035ad7e7aaef8e2893726ae5ebf880462ed249c23f87f",
    "common/speculative.cpp": "fe17c69dd68e94926899093d79ac112d46880fc4a8a668f8e7e8c47be73af4e9",
}
LIBRARY_SHA256 = "babcd972837a139e1a0e848a2cb56e32dccab168a20583396c834fe7f74fb5c6"
SYMBOL = "_Z23common_speculative_initR25common_params_speculativej"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--runtime-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("choose a new build directory outside Marathon")
    for relative, expected in EXPECTED.items():
        if sha(args.runtime_source / relative) != expected:
            parser.error("unqualified source: " + relative)
    if sha(args.runtime_library) != LIBRARY_SHA256:
        parser.error("unqualified production common library")
    args.output.mkdir(parents=True)
    (args.output / "common").mkdir()
    (args.output / "src").symlink_to(args.runtime_source / "src", target_is_directory=True)
    generated = args.output / "common/speculative.cpp"
    original = (args.runtime_source / "common/speculative.cpp").read_text()
    # The deployed file differs from the portable source only in these three
    # formatting/cast spellings. Pin its hash above before normalizing for apply.
    portable = original
    for old, new in (
        ("};\n                    const int32_t rc = llama_encode(ctx_dft, enc_batch);",
         "};\n\n                    const int32_t rc = llama_encode(ctx_dft, enc_batch);"),
        ("                    }\n                    const float * inp_g = llama_get_embeddings_nextn(ctx_dft);",
         "                    }\n\n                    const float * inp_g = llama_get_embeddings_nextn(ctx_dft);"),
        ("std::memcpy(inject_batch.embd, inp_g, size_t(n_chunk) * n_embd_dec * sizeof(float));",
         "std::memcpy(inject_batch.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));"),
    ):
        assert portable.count(old) == 1
        portable = portable.replace(old, new)
    generated.write_text(portable)
    patch = root / "scripts/experiments/dflash-host-staging.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=args.output, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=args.output, check=True)
    candidate = generated.read_text()
    versions = args.output / "exports.map"
    versions.write_text("{ global: " + SYMBOL + "; local: *; };\n")
    signature = "common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {"
    manifest = {"source_hashes": EXPECTED, "production_common_sha256": LIBRARY_SHA256,
                "patch_sha256": sha(patch), "export": SYMBOL, "variants": {}}
    for label, source in (("control", original), ("candidate", candidate)):
        assert source.count(signature) == 1
        marker = f"dflash-host-staging: {label} constructor active"
        generated.write_text("#include <cstdio>\n" + source.replace(signature, signature +
            f'\n    std::fputs("{marker}\\n", stderr);'))
        output = args.output / ("libdflash-host-" + label + ".so")
        command = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-ffunction-sections",
                   "-fdata-sections", "-Wall", "-Wextra",
                   *["-I" + str(args.runtime_source / p) for p in ("include", "ggml/include", "common")],
                   str(generated), "-Wl,--gc-sections", "-Wl,--version-script=" + str(versions),
                   "-o", str(output)]
        subprocess.run(command, check=True)
        exports = subprocess.check_output(["nm", "-D", "--defined-only", str(output)], text=True)
        assert len(exports.splitlines()) == 1 and SYMBOL in exports, exports
        manifest["variants"][label] = {"sha256": sha(output), "command": command, "marker": marker}
        (args.output / (label + "-speculative.cpp")).write_text(generated.read_text())
    generated.write_text(candidate)
    manifest["compiler"] = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
