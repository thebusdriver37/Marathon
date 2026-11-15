# AiGuy CPU fan configuration

Saved on November 14, 2026 to reduce the Dynatron A38's repeated loud speed changes. The final configuration uses a temperature-based curve stored in the motherboard's BMC. No browser, script, or agent needs to remain running.

## Hardware and access

- Host: AiGuy
- Motherboard: ASRock Rack ROMED8-2T
- CPU: AMD EPYC 7402
- Cooler: Dynatron A38, connected to FAN2
- BMC firmware: 2.08.00; BIOS: P4.10
- Dashboard: https://192.168.1.86/ on the local network
- Settings location: **Settings → FAN Settings**

Safari on the MacBook could reach the dashboard. Chrome automation needed a temporary SSH tunnel through AiGuy. If needed again, run on the MacBook:

```bash
ssh -N -L 127.0.0.1:18443:192.168.1.86:443 aiguy
```

Then open https://127.0.0.1:18443/ while that SSH command remains running. This tunnel is only for dashboard access; fan control does not depend on it. No credentials are stored in these notes.

## Saved settings

**Open Loop Control Table → Table 1 → Customized Setting:**

| Row | CPU temperature °C | Fan duty % |
|---|---:|---:|
| 1 | 30 | 35 |
| 2 | 40 | 36 |
| 3 | 45 | 37 |
| 4 | 50 | 38 |
| 5 | 55 | 40 |
| 6 | 60 | 50 |
| 7 | 65 | 60 |
| 8 | 70 | 70 |
| 9 | 75 | 85 |
| 10 | 80 | 100 |

Rows 11–24 are blank. Save the table.

**Temperature Sensor and Corresponding Fan Table → CPU Temp → Customized Setting:**

- Sensor: CPU Temp, sensor number 48
- Open-loop table: Table 1
- Closed-loop table: Disable
- Selected fan: FAN2 only
- Click **Set**.

**FAN Mode:** set FAN2 to **Customized**, then **Save control mode**. Other fan headers remain **Default**. Manual duty settings are unused.

## Important firmware behavior

Both temperatures and duty percentages must **strictly increase** between populated rows. Equal adjacent percentages are rejected by firmware 2.08 even though the web form allows them. The error was HTTP 500, code 1008, `Set duty cycle Error`; native firmware validation returns IPMI completion code `0xcc` for non-increasing entries. A failed save can leave temperatures written without the duties, so always reload and verify both columns before activating a custom table.

The attempted closed-loop configuration ran FAN2 at approximately 10,700–10,800 RPM despite a CPU temperature of 37–42°C. It was reverted and its custom table cleared. The final working configuration uses open-loop control only.

## Verification and persistence

The original curve jumped from 40% at 45°C to 60% at 50°C; an earlier reading was 6,600 RPM at 51°C. With the saved curve, the fan settled around 4,000 RPM at 45–47°C. The user reported that the noise was less annoying.

A 45-second, 16-worker CPU-only check peaked at 63.5°C, with the fan increasing to about 5,500 RPM. Afterward it returned to 4,100 RPM at 49°C. This was a limited check, not a sustained all-core or combined CPU/GPU stress test.

Saved values and active mode were read back from the BMC. Settings should survive normal restarts and power cycles, but reboot persistence was not tested. Firmware updates or factory resets may reset the configuration. No firmware was flashed and no reboot was performed.

## Rollback

Open **Settings → FAN Settings → FAN Mode**, select **Default for FAN2**, and click **Save control mode**. This restores the factory automatic fan behavior.

## Evidence

Detailed results and firmware validation evidence are on AiGuy in `/home/deforest/.local/state/a38-fan-tuning/`, particularly `result-20261114.json` and `firmware-validation-evidence.json`. The downloaded firmware was inspected offline only.
