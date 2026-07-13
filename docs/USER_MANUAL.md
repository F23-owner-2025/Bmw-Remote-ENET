# User Manual

## What you see in the GUI

| Indicator | Meaning |
|-----------|---------|
| Gateway Running | Desktop service / tunnel process is up |
| Laptop Connected | Agent peer is online |
| Vehicle Connected | ENET link reported up |
| Vehicle Awake | Recent ENET activity |
| ENET / Tunnel | Tunnel state is `Connected` |
| Packet rate / RTT / Loss / CPU | Live telemetry |
| Flash Safety | Whether programming is advisable |

Buttons: **Start**, **Stop**, **Restart**, **Settings**, **Diagnostics**, **Export Logs**.

## Daily workflow

1. Park near the car; connect ENET cable to the laptop.
2. Confirm laptop and desktop are on the same LAN.
3. Start (or auto-start) `enet-gateway` on the desktop and `enet-agent` on the laptop.
4. Open **BMW ENET Gateway** GUI on the desktop.
5. Wait until Laptop Connected + Vehicle Awake.
6. Launch ISTA+ / E-Sys / BimmerUtility on the desktop against the `BMW-ENET` adapter.
7. For coding: proceed when loss/RTT look healthy.
8. For **flashing**: only when Flash Safety shows **SAFE**.

## Settings

| Setting | Guidance |
|---------|----------|
| Tunnel port | Default 47900; must match both sides |
| Password | Optional PSK; enable `require_crypto` on both |
| Reconnect delay | Base backoff when the peer drops |
| Timeouts | Peer timeout; increase slightly on flaky Wi-Fi |
| Allowed CIDRs | Keep tight to your LAN |
| Auto start | Install as Windows service / scheduled task |
| Logging level | `info` normal; `debug` for troubleshooting |

## Recovery behavior

| Event | Expected behavior |
|-------|-------------------|
| Cable unplug | Vehicle Connected clears; tunnel stays up |
| Vehicle sleep | Awake clears; rediscovery on wake |
| Ignition cycle | Tool TCP sessions drop; reconnect in the tool |
| Laptop sleep/disconnect | Gateway shows reconnecting; agent backs off |
| Desktop reboot | Service starts; agent reconnects automatically |

## Troubleshooting

| Symptom | Checks |
|---------|--------|
| Laptop never connects | Firewall UDP 47900; `peer_addr`; same password/crypto flags |
| Vehicle never awake | ENET cable, ignition, Npcap on correct NIC, activation line |
| ISTA cannot find car | Virtual NIC IP `169.254.1.1/16`; L2 tunnel connected; try discovery again |
| High loss / NOT SAFE | Prefer Ethernet over Wi-Fi; close bandwidth hogs; shorten path |
| GUI API unreachable | Gateway running; API on `127.0.0.1:47901` |

## Logging

Logs rotate daily under `logs/enet-gateway.log.*`. Use **Export Logs** to dump a bundle for support.

## Safety rules

- Never flash when the GUI warns.
- Never leave a programming session unattended on Wi-Fi.
- This software does not modify vehicle data by itself — tools do. Use trusted tooling only.
