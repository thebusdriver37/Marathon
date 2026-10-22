# Marathon performance and Rust audit

## Implemented fixes

The router now uses incremental, bounded SSE buffering for both decoded Responses and raw HTTP proxy streams. Repetition detection now finds candidate periods with Python's native substring search before checking complete spans, preserving the previous guard's character-based behavior. This Python implementation improved on the original Rust prototype without adding a native runtime dependency.

Worker acquisition now runs outside the event loop and releases leases acquired after cancellation, including cancellation between thread completion and ownership transfer. Request fingerprinting is skipped when both telemetry and debug tracing are disabled. Active tracing retains its existing hashes and behavior.

Measured medians from seven interleaved runs of the frozen original code and the actual updated router:

| Workload | Before | Implemented | Speedup |
| --- | ---: | ---: | ---: |
| 8,192 small events | 53.20 ms | 51.33 ms | 1.0x |
| One 1 MiB event | 41.13 ms | 5.57 ms | 7.4x |
| One 8 MiB event | 2472.74 ms | 51.37 ms | 48.1x |
| 24 KiB argument guard | 256.57 ms | 3.50 ms | 73.4x |

The guard workload is one non-repeating 24 KiB argument delivered in 8-character deltas. Small-event parsing changed little; the large-frame results are stress tests, not token-generation speedups. These numbers exclude model inference and do not establish a total task speedup.

Validation: **582 tests run, 568 passed, 14 optional tests skipped**. The suite ran with isolated HOME, XDG, and Codex directories and no private conversation data. Optional network/bootstrap and native frontend checks were not enabled. Python compile checks, shell syntax checks, and whitespace checks passed. Regression coverage includes every split point in mixed-delimiter/Unicode streams, immediate streaming and cancellation, linear delimiter-scanning work, frame limits, all candidate repetition periods and boundary lengths, and both worker-acquisition cancellation races.

SSE frames now have a **64 MiB limit** (excluding the complete delimiter). Oversized complete or unfinished frames fail explicitly. Ordinary non-SSE HTTP forwarding keeps its existing behavior. No history-index rewrite, model change, or Rust production extension was introduced. Existing sessions were not restarted; exit and relaunch Marathon from this checkout to load the updated router.

Run the comparison yourself, without starting a model or accessing conversations:

```bash
python3 scripts/benchmarks/bench_stream_buffer.py
```

Use `--repeats 7 --output /tmp/marathon-performance-fixes.json` to save timings and source hashes. The before implementation is frozen under `scripts/benchmarks/fixtures/`; the after implementation is extracted from the actual current router. Both parser arms use the async iterator.

[Implemented before/after timing samples and source hashes](performance-fixes-20261022.json)

## Original audit scope

The initial source audit and Rust experiments below were measured before these production changes, against revision `19daa5f1a5c6ed2e16f637fb2c40e2a964a87ab0`. Their original results and source hashes are retained. Subsequent Python repetition optimization supersedes the original recommendation to port that function to Rust. History scanning remains an unmeasured rewrite proposal, and the slower JSON fingerprint bridge was not adopted.

No private conversations, telemetry, model configuration, or user data were opened. No GPU workers were started or modified. Source line references in the original findings refer to the baseline revision.

## Original measured results

Medians of 11 interleaved samples, in milliseconds. Each implementation receives the same generated workload. These independent cases cannot be summed into a total application speedup. The native bridge includes conversion costs unless explicitly labeled otherwise.

| Workload | Current Python | Python algorithm prototype | Rust prototype | Decision |
| --- | ---: | ---: | ---: | --- |
| One 8 MiB SSE event, delivered in 8 KiB chunks | 2468.30 | **39.09** | 61.84 | Fix Python buffering first: **63.1x**, 98.4% less time |
| One 1 MiB SSE event, delivered in 8 KiB chunks | 37.89 | **2.76** | 3.15 | Same fix: **13.7x** |
| Repetition checks across 24,576 argument characters, 8 characters/delta | 259.68 | Not measured | **12.65** | Rust candidate: **20.5x**, 95.1% less CPU time |
| Same guard, 64 characters/delta | 31.95 | Not measured | **1.25** | Rust candidate: **25.5x**, saves about 31 ms |
| 8,192 small SSE events, 8 KiB chunks | 56.90 | 46.11 | 17.36 batched; **63.30 per complete frame** | Batch throughput does not establish a live-stream benefit |
| Fingerprint 8,192 items, 10.73 MiB UTF-8 wire payload | **91.60** | Not measured | **117.78 from Python objects**; 52.81 from existing wire bytes | Do not add a JSON bridge: 28.6% slower from objects |
| Repeated history search, 20,000 records / 43.80 MiB journal | **227.52** | Not measured | Not implemented | Native scan is worth investigating; savings unmeasured |
| First index plus search of that journal, filesystem cache warm | **490.46** | Not measured | Not implemented | Avoid parsing eligible entries twice if correctness permits |
| Direct indexed read from that journal | **0.165** | Not measured | Not implemented | Already cheap; leave it alone |

The 8 MiB frame is a stress case for a large completion event, not an ordinary token delta. Its Python prototype ranged from 37.47 to 47.47 ms, versus 2440.81 to 2495.51 ms for the current implementation. At 2,000 history records, repeated search took 22.29 ms and initial index plus search took 48.44 ms, showing approximately linear scaling across these two fixtures.

CPU time closely tracked wall time: the large-frame baseline used 2468.11 CPU ms versus 39.04 CPU ms for the Python prototype; the 8-character repetition workload used 259.67 CPU ms versus 12.64 CPU ms in Rust. These are measured CPU savings. GPU throughput, power consumption, total application memory, cold-disk latency, and end-to-end task speed were not measured.

## Initial audit findings

### 1 Streaming buffer repeatedly scans and copies incomplete frames

Evidence: [`_pop_sse_frame`](../scripts/routers/codex_local_router.py#L1569) and [`_iter_sse_json`](../scripts/routers/codex_local_router.py#L3390).

Every new chunk appends to an immutable byte buffer. The frame finder searches the accumulated buffer for both delimiters from the beginning; extracting frames also copies the remaining suffix. For a single growing frame, repeated searches and copies produce quadratic work in frame size. This executes in the async router and can stall unrelated tasks in that process while parsing.

The benchmark's Python alternative uses a bytearray, tracks a delimiter-search cursor with three bytes of overlap, and removes consumed bytes once per chunk. It preserves the production frame parser and JSON decoder. This is the highest-priority implementation candidate. A production change still needs cancellation/transport regression tests and an explicit buffer-bound policy that accommodates legitimate large completion events.

The Rust prototype uses the same improved scanning strategy. Its large gain over baseline is therefore largely algorithmic, not a benefit that requires changing languages. For small events, batching all 8,192 events across one native call gives a 3.3x throughput improvement; crossing the ctypes/JSON boundary for each complete event is about 11.3% slower than current Python. A native implementation would need a persistent incremental parser or ownership of more of the transport path. Do not batch until completion in production: that would destroy streaming responsiveness.

### 2 Repetition detection is the clearest narrow Rust candidate

Evidence: [`_has_runaway_repetition`](../scripts/routers/codex_local_router.py#L1783) and the apply-patch argument accumulation in [`_request_responses_stream`](../scripts/routers/codex_local_router.py#L3610).

The guard checks up to 128 possible repeating suffix widths for every streamed apply-patch argument delta. Python creates repeated strings and slices for those comparisons. The Rust prototype compares slices, retaining Unicode codepoint semantics rather than accidentally using byte widths. The benchmark includes UTF-8 encoding and a native call on every check.

For a non-repeating, maximum-default-size 24,576-character argument, 3,072 small deltas cost 260 ms of CPU in Python and 13 ms in Rust. At 384 larger deltas, the absolute saving falls to about 31 ms. These are totals across the argument stream; they are not a single 260 ms pause and do not include model generation or argument accumulation.

If introducing native packaging is worthwhile, this small pure function is the first Rust port to try. Keep the Python fallback, preserve character limits and detection behavior, and measure actual delta sizes with generated end-to-end tool calls before shipping. Do not weaken or skip the guard merely to improve a benchmark.

### 3 Full-input fingerprinting pays serialization costs even with a fingerprint cap

Evidence: [`_input_fingerprint_summary`](../scripts/routers/codex_local_router.py#L1618), [`trace_request`](../scripts/routers/codex_local_router.py#L3028), and its unconditional invocation in the Responses path at [line 6115](../scripts/routers/codex_local_router.py#L6115).

The 256-item cap limits individual fingerprints, not the aggregate hash: every item is still canonically serialized, UTF-8 encoded, and hashed. The synthetic 8,192-item case cost 92 ms, chiefly serialization in the Python profile. One standalone deepcopy of the same object graph took another 45 ms; that is an isolated copy cost, not a measured count or total for a live request.

Rust processing already-encoded bytes took 53 ms, but the router already holds normalized Python objects. Serializing those objects for the bridge increased total time to 118 ms. Prefer avoiding unnecessary tracing preparation when no consumer needs it, reusing immutable scaffold encodings, and reducing redundant copies before adding native code. Any cached representation must invalidate on mutations and preserve current hash behavior.

The prototype serializer is not a drop-in replacement: Python and serde_json differ on float spelling, non-finite numbers, and some invalid Unicode inputs. Hash changes can affect cache identity and must not be silently introduced. A future native request pipeline could change the boundary economics, but the standalone function port does not pay off here.

### 4 History search redecodes all active records on every query

Evidence: [`HistoryArchive._refresh`, `entries`, and `search`](../marathon_app/history_archive.py#L170).

The offset index supports cheap direct reads, but search seeks, reads, decodes JSON, extracts text, and casefolds every eligible entry each time. Initial search first indexes and decodes records, then decodes indexed records again. Profiling confirmed JSON decoding and entry extraction are substantial costs; this is not simply waiting for disk in these warm-cache runs.

A Rust reader/search kernel over validated records could reduce CPU cost, but this audit does not claim a Rust speedup for it. Preserve the current offset-only design and no-duplicate-transcript-store contract. Any implementation must retain exact substring and Unicode casefold behavior, original IDs and roles, rollback handling, retrieval-output exclusion, oversized-record coverage, partial appends, file identity checks, and symlink protection. The existing 24 history tests pass; they are an initial regression gate, not proof of a future native rewrite.

### 5 Blocking worker acquisition can pause the async router

Evidence: [`pool_request`](../scripts/routers/codex_local_router.py#L3134) calls [`acquire_pool_worker`](../marathon_app/pool.py#L122) directly; [`_worker_gpu_blockers`](../marathon_app/pool.py#L25) waits on subprocess inventories and per-running-worker `docker top` calls.

The three initial inventories already run concurrently in a thread pool, but their caller synchronously waits for them from the async request handler. Per-worker Docker probes then run sequentially. Individual commands have five-second timeouts; that is a configured bound, not observed latency.

Move the whole acquisition operation away from the event loop, with cancellation-safe ownership and lease cleanup. A naive cancelled `to_thread` can leave a worker thread acquiring a lease after its caller exits. Rust cannot eliminate driver, Docker, or process startup latency. This is a potentially serious responsiveness issue inferred from source; no live pool probes were performed.

### 6 Inference queues and external waits need architecture changes

Evidence: the [`backend_lock`](../scripts/routers/codex_local_router.py#L6248) spans slot preparation and the full response/tool loop. [`SearchCoordinator`](../scripts/routers/marathon_search_coordinator.py#L64) holds its cross-process coordination lock across pacing and provider fetches.

Single-slot inference serialization protects shared KV state. Search pacing and backoff protect provider availability. Rewriting the same locks in Rust will not change queueing time. Evaluate independent worker/slot routing where safe, and measure queue wait, slot preparation, model first activity, backend generation, and web waits separately. Do not simply remove these locks.

HTML extraction already runs via `asyncio.to_thread`, PDF extraction uses an async subprocess, and model-readiness checks already run outside the event loop. History lineage is already stored as deltas. These existing optimizations should be preserved rather than broadly rewritten. The frontend is already Rust and inference is already native llama.cpp; Python rewrites do not directly accelerate GPU kernels or model loading.

## Initial rollout recommendations

1. Promote and regression-test the incremental Python SSE buffer. This removes roughly **2.43 seconds** of CPU work on the measured 8 MiB stress event without a native dependency.
2. Fix blocking pool acquisition with explicit cancellation and cleanup tests. Measure event-loop delay using a fake slow inventory before touching live workers.
3. Try the Rust repetition guard as an optional small extension. The measured saving is **247 ms** per maximum-sized argument with tiny deltas, or **31 ms** with larger deltas.
4. Reduce avoidable serialization and copying in Python. Reject the measured standalone fingerprint JSON bridge.
5. Prototype history scanning only if generated long-history retrieval is frequent enough to justify a larger correctness-sensitive port.

A 20x faster guard is not a 20x faster agent. If a 10-second turn contained this exact tiny-delta workload and all saved CPU time lay on its critical path, the guard change would reduce total turn time by about **2.5%**. Savings can be smaller when work overlaps backend generation. There is no measured end-to-end percentage for this audit.

## Original audit reproduction and validation

Machine: AMD EPYC 7402, 24 physical cores / 48 logical CPUs; Python 3.12.3; Rust 1.84.1, release optimization. CPU affinity covered all 48 CPUs; no dedicated-core or idle-machine guarantee. Baseline revision: `19daa5f1a5c6ed2e16f637fb2c40e2a964a87ab0`.

The harness extracts selected production definitions via AST, avoiding router initialization. History files are generated in temporary directories. Timings use one warmup per arm and 11 samples with rotated arm order; all samples and separate process CPU times are retained. The Python parser prototype is a synchronous harness loop; production integration must preserve async I/O. Neither prototype measures sockets or backpressure. The profiles are separate runs with instrumentation and are not used as latency measurements. First-index measurements use a warm filesystem cache. The generated large history/request cases are stress fixtures, not a claimed distribution of real user activity.

```bash
CARGO_TARGET_DIR=/tmp/marathon-rust-audit-target cargo build --release --locked \
  --manifest-path scripts/benchmarks/rust_audit/Cargo.toml
python3 scripts/benchmarks/bench_rust_audit.py \
  --library /tmp/marathon-rust-audit-target/release/libmarathon_rust_audit.so \
  --output /tmp/marathon-rust-audit.json
```

The native artifact path above is for Linux. Cargo dependencies are pinned in the manifest and lockfile; the first build may download them. Build artifacts stay outside the checkout.

Validation: **287 fixture parity checks passed**, including mixed-prefix Unicode repetition inputs, split delimiters, malformed frames, invalid UTF-8, and randomized chunk boundaries. **24 existing history tests passed** with an isolated temporary HOME and XDG directories. Python syntax and Rust formatting checks passed. The native FFI is a benchmark prototype, not a production extension, and its limited fixture parity does not establish general JSON or transport compatibility.

- [Raw timing samples, CPU samples, source hashes, and profiles](performance-rust-audit-20261022.json)
- [Reproducible benchmark and Python buffering prototype](../scripts/benchmarks/bench_rust_audit.py)
- [Rust prototypes](../scripts/benchmarks/rust_audit/src/lib.rs)
