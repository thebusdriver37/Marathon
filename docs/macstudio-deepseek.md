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
Sampling and reasoning effort remain unset, preserving server defaults.
Parallel tool calls are disabled; server concurrency remains 1. Text-only is
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
