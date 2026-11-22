# Status

## In flight

- **Single-3090 decode.** Same target weights and quantization, 262K context. Goals: 200K over 100 prose / 150 code tok/s; owner asked for 50K prose >100, code >200. Baseline 50K 84.2/141.6, 200K 62.7/129.8. See Speed200.
- **Prose acceptance.** Temp 1.0/top-p 0.95/top-k 20, C3 drafter, seven nodes. Drafter agrees with the target only ~69% at the first token on prose (1-TV, measured), so lossless tricks are capped; coupled Gumbel drafting gave no gain. [Research](docs/spec-decoding-research-20261116.md).

- **Speed200 (2026-11-21, live without the head copy).** Image `cf9b4306` + original drafter, ~3-4% shorter rounds, exact output. The earlier 12:25 crash loop was the headQ3K drafter (+521 MiB vs ~400 MiB headroom with GPU vision); its files are deleted. Qualify runtime changes with `~/AI/experiments/oom-20261121/scripts/repro.py` (production command + image request). Lossy relax declined; keep `GGML_CUDA_MARATHON_Q4_OPT=0`. Next lever for speed: a better drafter.

## Runtime

- The four-worker Marathon pool config pins llama.cpp v1.0.0 speed450b + lorafast + slot cache + speed200 (`cf9b4306`, gpu-control `runtime/llamacpp-v1-speed200`; previous `1580ad9a`), 262144 context, Q4_0 KV, seven-node DFlash2 C3 at tree temperature 2.0. Env: CPU top-k sampling prefilter and three verify-kernel switches. [Image, bench, incident and rollback](../gpu-control/runtime/llamacpp-v1-speed200/README.txt).
- Qualify runtime changes in isolation with byte-identical checks at 50K and a separate 200K run. Record rollback and ship requested or agreed changes when the workers are idle.

## Next


- **Slot-file cache (2026-11-21, live).** Image `1580ad9a` (gpu-control `6f71fd8`, `runtime/llamacpp-v1-slotcas`): lazy checkpoint reads, thinning, and a content-addressed chunk store (`LLAMA_SLOT_CAS=1`). 44K synthetic: 9.08 -> 2.84 GB, restore 0.5 s, first token 1.4 s; outputs identical. Lossy rounding not used. Only open item: watch the numbers-only `prompt rollback` log line for avoidable full recomputes (3 of 27 past deep rewinds). Details: archive and README.

- **Majesty drives Marathon (2026-11-20).** `marathon codex app-server` now works: app-server refuses `--profile`, so the shared profile's settings (privacy hardening included) go in as `-c` overrides (`codex_home.shared_profile_overrides`). Majesty runs it as its local-only Marathon engine and lets it go after 10 idle minutes, which stops the runtime.

- **Inkling 2 Mini (2026-11-22, live in Marathon).** Selectable Maya 262K / 344K on demand; Qwen stays default. Shared Qwen card leases, FP8 KV, attn split 80, DSpark w4. 150K+ agentic 2/2 vs Qwen 1/2; short 8/10 vs 9/10 at 0.68x time. Lacks slot-file cache/snapshot resume, fast-mode LoRA and GPU vision. Setup/one-line default change: [INKLING](docs/INKLING.md). Main `97da50e` pushed; Majesty idle refresh and cleanup pending. Evidence `~/AI/experiments/inkling2-mini-20261121/integration/go-live`.

- **Cloud harnesses.** Installed Gemini CLI v0.70.0 (npm, node 24) and Antigravity CLI v1.5.13 (`agy`, `~/.local/bin`) as standalone harnesses for cloud-model comparisons; first-run auth for both is pending.

- **Web fetch fallback.** The current router preserves links when trafilatura is available. A fresh synthetic check with trafilatura disabled kept the anchor text and dropped the destination. Preserve the resolved URL in the BeautifulSoup fallback and add a regression check before closing.

## Log

- 2026-11-21: Social sentiment skill shipped in Majesty commit `fed6d16`: HN Algolia, Google News RSS and Reddit app-only OAuth reads, paced/cached, read-only. `npm run check` and 99 tests pass; live HN/news verified. Reddit needs an approved script-app token and otherwise gives a clean setup error. Prod helper `cmwq2dgn3` completed Anthropic IPO sentiment: HN 111 comments 14+/52-/45 neutral; news 93 headlines 18+/31-/44 neutral; report `/pages/2ccbb153fffb41ea/index.html` (HTTP 200). Next: Reddit app approval if sustained Reddit reads are needed; Lemmy is a working no-auth fallback. X skill remains shipped (`03bdf36`, `6b370ef`).
- 2026-11-21: Separate research skills shipped in Majesty commit `a5e3e04`: `arxiv`, `hugging-face`, `research-feeds`, model-routed beside existing `x`/`social`; checks and 111 tests pass, live sources verified. Helper `cmwq4anpj` completed a manual sweep. Removed its six-hour Frontier sweep routine `rmwq4hd0o`; no recurring sweep remains pending owner-approved cadence.
- 2026-11-18: GPU 3 verified at +150 MHz live and in the enabled boot policy. [Power policy and rollback](../gpu-control/runtime/gpu-power-policy-20261114.md).
- 2026-11-17: Reconciled this file with the code, benchmark evidence and installed settings. Closed investigations and superseded deployment notes are in the [archive](docs/status/ARCHIVE.md). The eight-node candidate remains in the isolated `bench/tree8-real-20261117` worktree.
