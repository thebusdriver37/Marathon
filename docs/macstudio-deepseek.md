# Mac Studio DeepSeek integration

Validated on AiGuy on 2026-11-11, after owner approval of the SSH tunnel.
In a newly launched Marathon session, select **DeepSeek V4.1 Flash 1017
(Mac Studio)** in `/model` (catalog ID `macstudio-deepseek`). Existing sessions
may retain their startup catalog. No default model was changed.

## Connection and registration

The dedicated enabled user service `marathon-studio-tunnel.service` forwards
AiGuy `127.0.0.1:11235` to Studio `127.0.0.1:11234` over LAN SSH. Both endpoints
remain loopback-only. The service source is
`config/examples/marathon-studio-tunnel.service`, installed at
`~/.config/systemd/user/marathon-studio-tunnel.service`. It reconnects on failure
with a ten-second delay. Check it with:

```sh
systemctl --user status marathon-studio-tunnel.service
curl --max-time 15 -fsS http://127.0.0.1:11235/v1/models
```

`config/examples/macstudio-deepseek.toml` was appended to the machine-local
`~/.config/marathon/catalog.toml`. The previous catalog was saved as
`catalog.toml.before-macstudio-20261111` beside it. A parsed comparison confirmed
all pre-existing entries/settings were preserved. No credentials are in the
tracked templates. No model bundle or GPU worker registration is needed:
Marathon's existing external Responses route handles this service.

Context is 262144, compaction threshold 229376, and truncation limit 221184.

## Thinking and DeepSeek tuning (2026-11-13)

mlx-serve renders DeepSeek V4.1 in non-thinking chat mode unless the request
carries `reasoning.effort`; with the original unset settings every Marathon
request arrived as `thinking=false`. The entry now sets `temperature = 1.0` and
reasoning levels `none`, `high` (default) and `xhigh`. The server's template maps
low/medium/high to effort hints 50/62/75 and xhigh/max to 100; low, medium and
high measured about the same depth, xhigh about twice as deep.

Opt-in keys used only by this entry:
- `native_instruction_roles = true` keeps developer/system messages in their
  roles, so the instructions land in DeepSeek's real system block instead of a
  tagged user turn (the default conversion other models rely on).
- `max_output_tokens = 32768`: thinking counts toward the cap; 8192 truncated
  max-effort turns. Cut-off recovery retries now send effort `none` on Responses
  backends that declare it, matching the chat-template path.
- `disabled_tools` hides `multi_agent_v1__*` and `view_image` (about half the
  tool schema text per request on a text-only, single-server model).
- `instructions_file = "config/prompts/deepseek-v4-agent.md"` replaces the
  21K-character upstream Codex prompt with a 4K DeepSeek prompt; Marathon's
  runtime rules are still appended.

Single A/B pass at `high` through the real router and frontend (calc
interpreter with hidden tests, multi-module pipeline bug, bug reports, coding
gauntlet): every setup passed every task. The full package was faster on calc
(9.0 vs 12.5 min) and bugs (75 vs 98 s) and slower on the gauntlet (153 vs
115 s). One run each, so treat speed differences as indicative only.
Parallel tool calls are enabled, the Marathon default since 2026-11-12; a direct
probe returned three independent `read_file` calls in one response. Server
concurrency remains 1. Text-only is
conservative pending vision validation. No server tuning, restart, GPU worker
change, or power-setting change was performed.

## Evidence

Read-only discovery confirmed the exact model ID, ready state, 262144 context,
tool-use capability and advertised output budget 65536. Real requests through
an isolated Marathon router and the approved tunnel passed:

- `17 plus 25`, integer-only response: `42`.
- Forced synthetic `lookup_test_code` function: arguments `{"item":"blue"}`.
- Tool result continued via `previous_response_id`: exact reply `STUDIO-7391`.
- Frontend catalog exposes the Studio with `visibility=list` and expected limits.

The driver was `.marathon/venv/bin/python /tmp/check-marathon-studio.py`.
Synthetic evidence is at `/tmp/marathon-studio-e2e-ko2tq1q_/` (models, arithmetic,
tool call, tool-result response, isolated router logs/state). The driver used a
fresh temporary HOME, catalog, router token and loopback port; it stopped only
its own test router afterward. Two initial test-launch attempts lacked import
path/router authentication setup; both were corrected before live validation.
No existing conversations or private sessions were opened. Interactive terminal
menu clicking was not exercised; the actual catalog consumed by that menu was
verified. Full-context retrieval and a coding gauntlet were not run for this
configuration-only integration.

Full suite, including `tests/test_macstudio_catalog.py`: 627 tests, 614 passed,
13 skipped in 17.226 seconds. Ran the repository virtualenv's
`python -m unittest discover -s tests -v` with temporary HOME/XDG directories,
`RUSTUP_HOME=/home/deforest/.rustup`, and `MARATHON_NETWORK_TESTS=1`.
Log: `/tmp/marathon-studio-complete-tests.log`.

Final health check: service active/running, zero restarts, model ready after
several minutes without generation. This does not establish overnight sleep
reliability. SSH keepalives detect loss but do not guarantee wakefulness. If
sleep blocks use, obtain owner approval before changing Studio power settings.

## Rollback

Stop/disable only the new service with
`systemctl --user disable --now marathon-studio-tunnel.service`, and remove only
the `macstudio-deepseek` table from the local catalog. Preserve any subsequent
catalog edits instead of blindly restoring the backup. No Studio rollback is
needed because its server configuration was not changed.
