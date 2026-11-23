# Status

## In flight

- **Single-3090 decode.** Same target weights and quantization, 262K context. Goals: 200K over 100 prose / 150 code tok/s; owner asked for 50K prose >100, code >200. Baseline 50K 84.2/141.6, 200K 62.7/129.8. See Speed200.
- **Prose acceptance.** Temp 1.0/top-p 0.95/top-k 20, C3 drafter, seven nodes. Drafter agrees with the target only ~69% at the first token on prose (1-TV, measured), so lossless tricks are capped; coupled Gumbel drafting gave no gain. [Research](docs/spec-decoding-research-20261116.md).

- **Speed200 (2026-11-21, live without the head copy).** Image `cf9b4306` + original drafter, ~3-4% shorter rounds, exact output. The earlier 12:25 crash loop was the headQ3K drafter (+521 MiB vs ~400 MiB headroom with GPU vision); its files are deleted. Qualify runtime changes with `~/AI/experiments/oom-20261121/scripts/repro.py` (production command + image request). Lossy relax declined; keep `GGML_CUDA_MARATHON_Q4_OPT=0`. Next lever for speed: a better drafter.

## Runtime

- The four-worker Marathon pool config pins llama.cpp v1.0.0 speed450b + lorafast + slot cache + speed200 (`cf9b4306`, gpu-control `runtime/llamacpp-v1-speed200`; previous `1580ad9a`), 262144 context, Q4_0 KV, seven-node DFlash2 C3 at tree temperature 2.0. Env: CPU top-k sampling prefilter and three verify-kernel switches. [Image, bench, incident and rollback](../gpu-control/runtime/llamacpp-v1-speed200/README.txt).
- Qualify runtime changes in isolation with byte-identical checks at 50K and a separate 200K run. Record rollback and ship requested or agreed changes when the workers are idle.

## Next

- [Secret scanning alert #1](https://github.com/thebusdriver37/Marathon/security/secret-scanning/1) is open. Verified upstream Codex redaction test data. Safe to close as `Used in tests`.

- **Slot-file cache (2026-11-21, live).** Image `1580ad9a` (gpu-control `6f71fd8`, `runtime/llamacpp-v1-slotcas`): lazy checkpoint reads, thinning, and a content-addressed chunk store (`LLAMA_SLOT_CAS=1`). 44K synthetic: 9.08 -> 2.84 GB, restore 0.5 s, first token 1.4 s; outputs identical. Lossy rounding not used. Only open item: watch the numbers-only `prompt rollback` log line for avoidable full recomputes (3 of 27 past deep rewinds). Details: archive and README.

- **Majesty drives Marathon (2026-11-20).** `marathon codex app-server` now works: app-server refuses `--profile`, so the shared profile's settings (privacy hardening included) go in as `-c` overrides (`codex_home.shared_profile_overrides`). Majesty runs it as its local-only Marathon engine and lets it go after 10 idle minutes, which stops the runtime. Majesty resumes a thread while he types; a warmup for a slotless model (Inkling) now loads its worker (`6685235`).

- **Inkling 2 Mini (2026-11-22, live in Marathon and Majesty).** Selectable Maya 262K / 344K on demand, sharing Qwen card leases; Qwen stays default (switch in [INKLING](docs/INKLING.md)). Agentic 150K+ 2/2 vs Qwen 1/2; gate 60/63. Live image d2e2cece since 11-22 (unified cache, above; before: ff4b4cd0 = 0f0cc401 + 344K scratch-fit, gpu-control `32cd8a1`): tok/s in Majesty, 344K first answer -23%, copy speculation 1.65x on edits, durable cache (saves only on unload/switch; 149K resume 72s->4.7s; ~/AI/cache/inkling 8 GiB LRU). Verified live incl. unload->restore. Lacks LoRA, vision. **344K fix (11-22):** ~200 MiB freed with identical tokens; fits 3090 Ti, cache back on, pool GPUs 2/3/0, verified via `marathon codex exec` on GPU 2; rollback to 0f0cc401 in the runtime README. Earlier rollback: image 0f778b24, `git checkout 51741e5 -- runtime/inkling-maya-c520fd14-attn80`, config backup in parity-cache/combined-qual. Evidence `~/AI/experiments/inkling2-mini-20261121/` (parity-speed, parity-cache).

- **Cloud harnesses.** Installed Gemini CLI v0.70.0 (npm, node 24) and Antigravity CLI v1.5.13 (`agy`, `~/.local/bin`) as standalone harnesses for cloud-model comparisons; first-run auth for both is pending.

- **Unified cache (2026-11-22, live for Inkling).** Marathon owns cache policy for every model, with per-engine adapters (`scripts/routers/marathon_cache_backends.py`, [conversation-cache](docs/conversation-cache.md)); the Qwen path is byte-identical. Inkling image d2e2cece (= ff4b4cd0 scratch-fit + Marathon cache API; gpu-control `9821a6f`), with `cache_backend = "maya"` on both Inkling catalog entries (backup `catalog.toml.bak-before-unified-cache-20261122`). New chat, 21.8K scaffold: 262K 8.4s -> 1.25s, 344K 11.1s -> 1.24s (1.0s with the worker loaded); the warmup builds the same starter as the first turn (`6a4fcfd`). Tokens byte-identical vs ff4b4cd0 at 262K/344K; GPU 2 344K peak 23846 MiB. Running routers get it when they restart. Rollback: drop the catalog lines (lazy cache only) and/or pin ff4b4cd0. Report `~/AI/experiments/unified-cache-20261122/qual-sf/REPORT.md`. DeepSeek: no save/restore API; owner OK'd the SSD prefix cache, now live on the Studio (launch.sh `--prefix-cache-disk 100GB`, backup launch.sh.bak-20261122): 28K prompt after restart 14.6s->0.19s.

- **Web fetch fallback (minor, not broken).** Live router has trafilatura so links are kept; only the no-trafilatura fallback drops link URLs. Tests pass in the router venv (650); system python lacks trafilatura/gguf.

## Log

- 2026-11-21: Separate research skills shipped in Majesty commit `a5e3e04`: `arxiv`, `hugging-face`, `research-feeds`, model-routed beside existing `x`/`social`; checks and 111 tests pass, live sources verified. Helper `cmwq4anpj` completed a manual sweep. Removed its six-hour Frontier sweep routine `rmwq4hd0o`; no recurring sweep remains pending owner-approved cadence.
