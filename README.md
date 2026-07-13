# BMW ENET Remote Gateway

Connect your **desktop** (ISTA / E-Sys) to your BMW while the **ENET cable** stays on a **laptop** near the car.

Works on the **same Wi‑Fi** *or* on **different networks** (relay / WireGuard).

## 5-minute setup

- Same network → **[docs/QUICKSTART.md](docs/QUICKSTART.md)**  
- Different networks → **[docs/REMOTE.md](docs/REMOTE.md)**

| Situation | What to run |
|-----------|-------------|
| Same home Wi‑Fi | `Install-Desktop.bat` + `Install-Laptop.bat` |
| Different networks | `enet-relay` on a VPS + `enet-setup … --remote-relay` |
| Best remote quality | `enet-setup wireguard` then import WireGuard configs |

Dashboard: **http://127.0.0.1:47901/**

```bash
# Same LAN
enet-setup gateway --yes && enet-gateway
enet-setup agent && enet-agent

# Different networks (relay)
enet-relay --listen 0.0.0.0:47910
enet-setup gateway --remote-relay vps:47910 --yes && enet-gateway
enet-setup agent --remote-relay vps:47910 --pair-code BMW-XXXX --yes && enet-agent
```

## How it works

Transparent **Layer-2** tunnel (required for BMW ARP / HSFZ / DoIP discovery).

```
Vehicle ──ENET──► Laptop agent ══ LAN or Relay/VPN ══► Desktop gateway ──► ISTA / E-Sys
```

## Safety

Never auto-writes vehicle data. Flash only when the UI says SAFE — especially on remote links.

## License

MIT
