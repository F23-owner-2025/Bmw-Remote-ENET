# BMW ENET Remote Gateway

Transparent Layer-2 Ethernet tunnel that lets a **desktop PC** run ISTA+, E-Sys, BimmerUtility, Tool32, and related BMW F-Series tools while the **ENET cable** stays plugged into a **laptop** near the vehicle.

**Target vehicle:** 2017 BMW M240i Convertible (F23 / B58)

## Recommended architecture

**Custom L2 Ethernet-over-UDP tunnel** (not Layer-3 port forwarding).

BMW ENET discovery depends on ARP and UDP broadcasts (`169.254.0.0/16`, HSFZ UDP 6811, DoIP 13400). Only a Layer-2 tunnel preserves tool behavior as if the ENET cable were local.

```
Vehicle ──ENET──► Laptop Agent ══UDP :47900══► Desktop Gateway ──TAP/Wintun──► ISTA / E-Sys
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full options analysis.

## Components

| Binary | Role |
|--------|------|
| `enet-agent` | Laptop: captures ENET NIC, tunnels frames to desktop |
| `enet-gateway` | Desktop Windows service + control API |
| `enet-gui` | Status / settings / flash-safety UI |
| `enet-sim` | Simulated BMW traffic for CI / lab |

## Quick start (lab / CI)

```bash
# Requires Rust 1.85+
cargo test --workspace --exclude enet-gui
cargo run -p enet-sim -- lab --seconds 5 --flaps --burst 200
cargo run -p enet-gateway -- --simulate --run-seconds 3
```

## Production setup (Windows LAN)

1. Install on **desktop**: `enet-gateway`, `enet-gui`, Wintun (or TAP-Windows), Npcap optional on desktop.
2. Install on **laptop**: `enet-agent`, Npcap (required for raw ENET capture/inject).
3. Copy `config/gateway.toml` / `config/agent.toml` and set `peer_addr` on the agent to the desktop LAN IP.
4. Start gateway service, then agent. Open GUI on the desktop.
5. Configure ISTA/E-Sys to use the virtual `BMW-ENET` adapter (`169.254.1.1/16`).

Full steps: [docs/INSTALL.md](docs/INSTALL.md) · User manual: [docs/USER_MANUAL.md](docs/USER_MANUAL.md)

## Safety

The gateway **never writes** to the vehicle. The flash-safety gate warns when latency, loss, or CPU exceed thresholds. Do not flash ECUs until the GUI shows **SAFE**.

## License

MIT
