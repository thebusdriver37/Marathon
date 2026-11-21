# Active work

- Branch `exp/inkling-2-mini`, isolated worktree `Marathon-inkling`.
- Commits `8ffa7ec`, `8f80a13`, `0b4a493`: experimental Inkling 32K/262K profiles, model discovery, native template-effort mapping and recovery. Qwen defaults unchanged.
- Validation: 136 router-context tests, 16 transport (1 existing skip), 17 runtime setup, 110 app/catalog, 4 Inkling regressions, 9 fast-LoRA regressions. Full frontend smoke passed for both models.
- Agentic A/B: 50 evaluable trials complete, both 23/25. Inkling route tasks time out; Qwen's route failures concern an ambiguous pattern-validation requirement. One recovered Inkling invalid-JSON tool call.
- Two later SQL trials received external SIGTERM, with originals retained as infrastructure interruptions. GPU 2 is now occupied by another leased NInfer process; reported to the evaluation lead. Do not interrupt it.
- Evidence/report and unapplied rollout: `~/AI/experiments/inkling2-mini-20261121/integration/REPORT.md` and `rollout/`.
- Next: coordinate GPU ownership, finish remaining trials and administrative replacements, qualify latest runtime, run two prepared 150K+ context tasks. Nothing pushed or deployed.
