# Spark Flash Next quick audit

Follow-up: the [FP8/BF16 comparison](SPARK_KV_COMPARISON_2026-09-20.md) passed 4/4 checks on both settings and found no demonstrated benefit from changing KV precision.

Read-only audit of the running service and public serving recipes on September 20, 2026.
No production restart, setting change, model download, benchmark request, or private conversation access was performed.
The existing central deployment document describes an older configuration; the findings below use live container arguments and remote recipe files.

## Verified live configuration

| Item | Observed value |
| --- | --- |
| Hardware | Single NVIDIA GB10, 121 GiB unified host memory |
| Service/container | `qwen38-flash-next-nvfp4.service` / `spark-qwen38-nvfp4` |
| Model | `iSkye/Qwen3.8-Flash-Next-NVFP4-ablit-a070` |
| Model revision | `91c3e3d4daf14f8e9389b95f43112410f06ed3d5` |
| Engine | Patched vLLM `0.1.dev20073+g8e685d198` |
| Recipe commit | `6b5086458023474a7809ea30e1bcf42f03dcd75f`, matching upstream main at audit time |
| Context limit | 262144, YaRN disabled |
| Cache | FP8 KV; BF16 recurrent SSM state |
| Speculation | MTP3; reduced 47172-token English/code vocabulary |
| Graphs | V2 model runner; FULL_DECODE_ONLY; capture sizes 4/8/12/16 |
| Compilation | Mode 0, which does not mean CUDA graphs are disabled |
| Scheduling | Four maximum sequences; 2048-token prefill chunks |
| Memory settings | GPU utilization budget 0.786; configured 26 GiB host reserve |
| Marathon | Medium reasoning default; temperature 1.0 |
| Model generation defaults | Temperature 1.0, top-p 0.95, top-k 20 |
| Template | Custom template with reasoning effort, thinking enabled by default, history reasoning retention, and qwen3_xml tool parser |

The remote recipe also has local changes, including a custom QSA operations override and a mounted deterministic-top-k library.
An unchanged recipe commit does not prove the live overlays match upstream; their speed effect was not measured in this audit.
The installed recipe resides at `/home/deforest/Ai/experiments/mia-spark-nvfp4` on `spark`.

Live cumulative counters recorded 357966 proposed draft tokens and 195043 accepted, approximately 54.5% acceptance across prior traffic.
This is not a controlled benchmark and does not prove that a different draft depth would win.
There were no running or queued requests at both inspection points.
Available host memory was about 17 GiB; approximately 513 MiB of swap was occupied.
Swap-out counters did not increase between observations; these idle snapshots cannot rule out pressure during long inference.

## Findings and ranked next checks

1. **Quality first: compare BF16 KV with the current FP8 KV, preserving the weights, medium reasoning, and other settings.**
   The recipe explicitly treats FP8 quality as incompletely established: its small suite passed, but it also cites contrary long-reasoning evidence from another implementation.
   This makes cache precision a hypothesis to test, not a diagnosis of the user's complaints.
   Keep the same memory budget and verify the actual BF16 pool can still accommodate a 262K request; do not promise the current four-way capacity.
   If inconclusive, test float32 recurrent state separately rather than changing both precisions at once.
2. **Speed: a bounded MTP2 versus MTP3 comparison on realistic code and prose.**
   Keep weights and cache precision fixed for this comparison, and update graph capture sizes consistently with draft depth.
   Fewer draft steps may save work but also reduce accepted tokens per target pass; current cumulative acceptance alone cannot pick the winner.
   The current reduced draft vocabulary and graph optimizations are already enabled, so their published gains are not unclaimed gains here.
3. **Model modification: compare the same-quant stock parent on neutral tasks if cache precision does not explain quality.**
   The [model card](https://huggingface.co/iSkye/Qwen3.8-Flash-Next-NVFP4-ablit-a070) documents a 0.7-strength modification of nine attention output projections, with the MTP head unchanged.
   It reports severe coherence problems with the full-strength parent modification, which motivated this milder version.
   This does not establish that the deployed milder version is defective, but it makes stock-versus-modified a useful diagnostic control.
   The card specifically warns against transplanting its modified shards into differently quantized checkpoints.

## Speed claims and backend choice

The [same serving recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) reports about 48.7 tok/s for single-stream prose and 67.7 per-stream tok/s for predictable counting in distinct tests.
It reports larger aggregate numbers with concurrent streams, which are not individual conversation speed.
No exact source for the user's 80 tok/s claim was supplied, so its comparability remains unresolved.
The earlier local September 5 comparison measured about 48 tok/s prose and 61 tok/s repetitive code on the stock Mia checkpoint, not today's abliterated model and overlays.
Those historical results are context, not a fresh measurement of today's service.

An alternative [single-Spark recipe](https://github.com/Radar105/qwen38-flash-next-nvfp4-spark) demonstrates BF16 KV with 262K context and MTP2 using NVIDIA weights, but its reported production-window rates do not establish a speed advantage over this setup.
The [official model's thinking sampling recommendations](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) agree with the local model defaults; no obvious sampling mismatch was found from configuration inspection.
Actual per-request overrides were not captured in this read-only audit.

Changing engines is possible in principle, but this checkpoint and its overlays are tailored to patched vLLM.
A llama.cpp experiment would require compatible weights and support rather than changing one launch flag; the old GGUF rollback is a different checkpoint, not the same uncensored weights.
The existing local whole-stack comparison favored vLLM, so an engine migration is lower priority than the small precision and MTP checks above.

The recommendation is to keep vLLM and the current uncensored checkpoint while testing one variable at a time in an agreed Spark maintenance window.
No speed improvement or quality defect was proven by this audit.
