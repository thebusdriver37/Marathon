# Active work

- Branch `exp/inkling-2-mini`, isolated worktree `Marathon-inkling`.
- Commits `8ffa7ec`, `8f80a13`, `0b4a493`: experimental Inkling 32K/262K profiles, model discovery, native template-effort mapping and recovery. Qwen defaults unchanged.
- Validation: 136 router-context tests, 16 transport (1 existing skip), 17 runtime setup, 110 app/catalog, 4 Inkling regressions, 9 fast-LoRA regressions. Full frontend smoke passed for both models.
- Agentic A/B: 50 evaluable trials complete, both 23/25. Inkling route tasks time out; Qwen's route failures concern an ambiguous pattern-validation requirement. One recovered Inkling invalid-JSON tool call.
- The lead confirmed both later SQL stops were intentional: the owner ordered SPEED first, quality later. Originals remain infrastructure interruptions, excluded from scores and speed. GPU 2 belongs to the speed member; all C jobs stopped and the lease released. No C services/timers remain.
- Evidence/report and unapplied rollout: `~/AI/experiments/inkling2-mini-20261121/integration/REPORT.md` and `rollout/`.
- Partial report finalized with per-task/seed evidence, failure analysis, runtime/weight pairing, effort cache caveat and exact manual resume commands. Two large synthetic tasks have 18 qualified reference checks; model token counts and long-context results are still pending.
- Next: wait for the lead to restart quality work after speed lands. Then qualify the chosen faster runtime, resume seed 3 as separate evidence and run the long tasks. Nothing pushed or deployed; rollout remains an unapplied draft.
