# Installation Guide

## Prerequisites

### Desktop (runs ISTA / E-Sys)

- Windows 10/11 x64
- Admin rights (service install + firewall)
- [Wintun](https://www.wintun.net/) **or** TAP-Windows
- Optional: WireGuard if using Internet stretch mode

### Laptop (ENET cable)

- Windows 10/11 x64 (Linux agent works in lab/sim mode)
- Admin rights
- [Npcap](https://npcap.com/) (WinPcap API compatible) for raw Ethernet capture/inject
- Physical ENET cable to the vehicle OBD port

### Network

- Prefer wired Ethernet between laptop and desktop
- Wi-Fi works but increases jitter — verify flash-safety before programming
- Default tunnel UDP **47900**, control API TCP **47901** (localhost)

## Build from source

```bash
rustup default stable   # 1.85+
cargo build --release -p enet-agent -p enet-gateway -p enet-sim
# GUI (needs display / Windows):
cargo build --release -p enet-gui
```

Binaries land in `target/release/`.

## Configure

1. Copy `config/gateway.toml` to the desktop install directory.
2. Copy `config/agent.toml` to the laptop; set `peer_addr` to the desktop LAN IP.
3. Optional shared password: set the same `password` on both and `require_crypto = true`.
4. Restrict `allowed_cidrs` to your LAN.

## Windows service (desktop)

```powershell
# After copying enet-gateway.exe + gateway.toml
sc.exe create BmwEnetGateway binPath= "C:\Program Files\BMW-ENET-Gateway\enet-gateway.exe --config `"C:\Program Files\BMW-ENET-Gateway\config\gateway.toml`"" start= auto
sc.exe description BmwEnetGateway "BMW ENET L2 tunnel gateway"
sc.exe start BmwEnetGateway
```

Laptop agent can be installed similarly as `BmwEnetAgent`, or started at logon via Task Scheduler.

## Firewall

Allow **inbound UDP 47900** on the desktop from your LAN only:

```powershell
New-NetFirewallRule -DisplayName "BMW ENET Tunnel" -Direction Inbound -Protocol UDP -LocalPort 47900 -RemoteAddress 192.168.0.0/16,10.0.0.0/8,172.16.0.0/12 -Action Allow
```

Keep TCP 47901 bound to `127.0.0.1` (default) so the GUI talks locally only.

## Virtual adapter (desktop)

1. Install Wintun; the gateway expects a virtual interface named `BMW-ENET` (configurable).
2. Assign:
   - IP: `169.254.1.1`
   - Mask: `255.255.0.0`
   - No gateway / DNS required
3. Point ISTA / E-Sys at ENET / that adapter.

> Raw Wintun IO hooks are stubbed behind `--simulate` in this release for CI. On Windows, replace the `SimulatedEthernet` / `VirtualNic` path with the Wintun session API (documented in `docs/WINDOWS_DRIVERS.md`).

## Laptop ENET NIC

1. Plug ENET cable; note the new Ethernet adapter in Device Manager.
2. Prefer DHCP-off with APIPA or set `169.254.1.2/16` if tools on the laptop also need direct access (normally tools run only on the desktop).
3. Ensure Npcap can open that adapter; set `enet_interface` in `agent.toml` if auto-detect picks the wrong NIC.

## Verify

1. Start gateway → GUI shows Gateway Running.
2. Start agent → Laptop Connected.
3. Ignition ON / wake vehicle → Vehicle Connected / Awake after discovery traffic.
4. Run `enet-sim lab` on a single machine for a dry run without hardware.
5. Open ISTA connection test / E-Sys gateway detection.

## Uninstall

```powershell
sc.exe stop BmwEnetGateway
sc.exe delete BmwEnetGateway
# Remove install directory and firewall rule
Remove-NetFirewallRule -DisplayName "BMW ENET Tunnel"
```

Installer scripts: `installer/`.
