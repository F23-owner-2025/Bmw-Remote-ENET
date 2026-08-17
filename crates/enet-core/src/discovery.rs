//! Interface discovery helpers for ENET and LAN adapters.

use enet_protocol::magic::LINK_LOCAL_PREFIX;
use serde::{Deserialize, Serialize};
use std::net::IpAddr;

/// Snapshot of a network interface useful for auto-detection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InterfaceInfo {
    /// OS interface name (e.g. "eth0", "Ethernet 2").
    pub name: String,
    /// Optional friendly description.
    pub description: String,
    /// MAC address string.
    pub mac: String,
    /// Assigned IPv4 addresses.
    pub ipv4: Vec<IpAddr>,
    /// Whether the link is reported up.
    pub is_up: bool,
    /// Whether any address is in 169.254.0.0/16.
    pub has_link_local: bool,
}

/// Return true if `ip` is in 169.254.0.0/16.
pub fn looks_like_enet_subnet(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            o[0] == LINK_LOCAL_PREFIX[0] && o[1] == LINK_LOCAL_PREFIX[1]
        }
        IpAddr::V6(_) => false,
    }
}

/// Detect candidate interfaces using `sysinfo` / OS enumeration.
///
/// On CI/Linux without real ENET hardware this still returns available NICs so
/// auto-detection logic and the simulator can be exercised.
pub fn detect_candidate_interfaces() -> Vec<InterfaceInfo> {
    // Per-interface IPv4 addresses (if_addrs) — the 169.254.x link-local
    // signal is the strongest ENET hint, so this must be populated.
    let mut addrs_by_name: std::collections::HashMap<String, Vec<IpAddr>> =
        std::collections::HashMap::new();
    if let Ok(ifaces) = if_addrs::get_if_addrs() {
        for iface in ifaces {
            if let if_addrs::IfAddr::V4(v4) = &iface.addr {
                addrs_by_name
                    .entry(iface.name.clone())
                    .or_default()
                    .push(IpAddr::V4(v4.ip));
            }
        }
    }

    // sysinfo 0.33 Networks API for MAC + traffic counters.
    let networks = sysinfo::Networks::new_with_refreshed_list();
    let mut out = Vec::new();
    for (name, data) in networks.list() {
        let mac = format!("{}", data.mac_address());
        let ipv4 = addrs_by_name.remove(name).unwrap_or_default();
        let has_link_local = ipv4.iter().copied().any(looks_like_enet_subnet);
        out.push(InterfaceInfo {
            name: name.clone(),
            description: String::new(),
            mac,
            ipv4,
            is_up: data.total_received() > 0 || data.total_transmitted() > 0,
            has_link_local,
        });
    }
    // Interfaces if_addrs saw but sysinfo missed. Skip 127.0.0.1-only loopback.
    for (name, ipv4) in addrs_by_name {
        if ipv4.iter().all(|ip| match ip {
            IpAddr::V4(v4) => v4.is_loopback(),
            IpAddr::V6(v6) => v6.is_loopback(),
        }) {
            continue;
        }
        let has_link_local = ipv4.iter().copied().any(looks_like_enet_subnet);
        out.push(InterfaceInfo {
            name,
            description: String::new(),
            mac: String::new(),
            ipv4,
            is_up: false,
            has_link_local,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Windows software loopback / Npcap localhost capture — never the car ENET cable.
pub fn is_software_loopback(name: &str, description: &str) -> bool {
    let blob = format!("{name} {description}").to_lowercase();
    let compact = blob.replace([' ', '_'], "-");
    blob.contains("loopback pseudo-interface")
        || compact.contains("loopback-interface")
        || compact.contains("npf-loopback")
        || blob.contains("adapter for loopback traffic")
        || blob.contains("npcap loopback")
        || blob.contains("teredo tunneling")
        || blob.contains("isatap")
}

/// Host-only virtual NIC (`BMW-ENET` / KM-TEST). The laptop Client must not capture this.
pub fn is_host_virtual_enet(name: &str, description: &str) -> bool {
    let blob = format!("{name} {description}").to_lowercase();
    blob.contains("bmw-enet") || blob.contains("km-test")
}

/// Score an interface for likelihood of being the BMW ENET NIC (car cable).
pub fn score_enet_candidate(iface: &InterfaceInfo, preferred_name: &str) -> i32 {
    let desc = iface.description.to_lowercase();
    let name = iface.name.to_lowercase();
    if is_software_loopback(&iface.name, &iface.description) {
        return -1000;
    }
    if is_host_virtual_enet(&iface.name, &iface.description) {
        return -1000;
    }

    let mut score = 0;
    if !preferred_name.is_empty() && iface.name.eq_ignore_ascii_case(preferred_name) {
        score += 100;
    }
    if iface.has_link_local {
        score += 50;
    }
    for needle in ["enet", "realtek", "ax88179", "rtl815", "usb", "ethernet"] {
        if desc.contains(needle) || name.contains(needle) {
            score += 5;
        }
    }
    // Deprioritize known virtual / tunnel / wifi adapters (Wi‑Fi is not BMW ENET).
    for needle in [
        "wintun",
        "tap",
        "vpn",
        "hyper-v",
        "vethernet",
        "docker",
        "wsll",
        "wi-fi",
        "wifi",
        "wlan",
        "wireless",
        "bluetooth",
    ] {
        if desc.contains(needle) || name.contains(needle) {
            score -= 80;
        }
    }
    if iface.is_up {
        score += 10;
    }
    score
}

/// Best-effort OS check: is this adapter's link/carrier up?
///
/// Used by the laptop Client to show “ENET cable plugged” before Npcap capture exists.
pub fn adapter_link_up(name: &str) -> bool {
    if name.is_empty() || name == "pending-enet" {
        return false;
    }
    // Cache probes — never spam the OS every UI tick.
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    struct Cache {
        name: String,
        up: bool,
        at: Instant,
    }
    static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
    if let Ok(guard) = CACHE.lock() {
        if let Some(c) = guard.as_ref() {
            if c.name.eq_ignore_ascii_case(name) && c.at.elapsed() < Duration::from_secs(2) {
                return c.up;
            }
        }
    }
    let up = adapter_link_up_uncached(name);
    if let Ok(mut guard) = CACHE.lock() {
        *guard = Some(Cache {
            name: name.to_string(),
            up,
            at: Instant::now(),
        });
    }
    up
}

fn adapter_link_up_uncached(name: &str) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use std::process::Command;
        /// Hide console windows — visible PowerShell flashing every few seconds is unusable.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;

        // Primary: MediaConnectionState — Status=Up is often true for USB-ENET
        // dongles even when no Ethernet cable is in the car.
        let script = format!(
            "$a = Get-NetAdapter -Name '{}' -ErrorAction SilentlyContinue; \
             if ($null -eq $a) {{ exit 2 }}; \
             if ($a.MediaConnectionState -eq 'Connected') {{ exit 0 }}; \
             exit 1",
            name.replace('\'', "''")
        );
        match Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-Command",
                &script,
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .status()
        {
            Ok(s) if s.success() => return true,
            Ok(s) if s.code() == Some(1) => return false,
            Ok(s) if s.code() == Some(2) => return false,
            _ => {}
        }

        // Fallback: netsh connect state (weaker — admin "connected", not media).
        let name_arg = format!("name=\"{name}\"");
        if let Ok(out) = Command::new("netsh")
            .args(["interface", "show", "interface", &name_arg])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
        {
            let text = String::from_utf8_lossy(&out.stdout).to_lowercase();
            if let Some(line) = text.lines().find(|l| l.contains("connect state")) {
                // Exact token check — "disconnected" also contains "connected".
                let state = line.split(':').nth(1).unwrap_or("").trim();
                return state == "connected";
            }
        }

        // Do NOT fall back to traffic counters / default-true is_up — that made
        // Vehicle ENET show green whenever the USB dongle was present.
        return false;
    }
    #[cfg(unix)]
    {
        let carrier = std::path::Path::new("/sys/class/net")
            .join(name)
            .join("carrier");
        if let Ok(v) = std::fs::read_to_string(&carrier) {
            return v.trim() == "1";
        }
        let oper = std::path::Path::new("/sys/class/net")
            .join(name)
            .join("operstate");
        if let Ok(v) = std::fs::read_to_string(&oper) {
            return v.trim() == "up";
        }
        false
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = name;
        false
    }
}

/// Pick the best ENET candidate, if any.
///
/// Never returns Windows/Npcap loopback or the Host `BMW-ENET` virtual NIC.
pub fn pick_enet_interface(preferred: &str) -> Option<InterfaceInfo> {
    let mut ifaces = detect_candidate_interfaces();
    ifaces.sort_by_key(|i| std::cmp::Reverse(score_enet_candidate(i, preferred)));
    ifaces.into_iter().find(|i| {
        let s = score_enet_candidate(i, preferred);
        s > 0
            && !is_software_loopback(&i.name, &i.description)
            && !is_host_virtual_enet(&i.name, &i.description)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn link_local_detection() {
        assert!(looks_like_enet_subnet(IpAddr::V4(Ipv4Addr::new(169, 254, 1, 1))));
        assert!(!looks_like_enet_subnet(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));
    }

    #[test]
    fn scoring_prefers_name() {
        let a = InterfaceInfo {
            name: "Ethernet 3".into(),
            description: "USB ENET".into(),
            mac: "00:11:22:33:44:55".into(),
            ipv4: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 5, 77))],
            is_up: true,
            has_link_local: true,
        };
        let b = InterfaceInfo {
            name: "Wi-Fi".into(),
            description: "Wireless".into(),
            mac: "aa:bb:cc:dd:ee:ff".into(),
            ipv4: vec![IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10))],
            is_up: true,
            has_link_local: false,
        };
        assert!(score_enet_candidate(&a, "Ethernet 3") > score_enet_candidate(&b, "Ethernet 3"));
    }

    #[test]
    fn scoring_rejects_windows_loopback() {
        let lo = InterfaceInfo {
            name: "Loopback Pseudo-Interface 1".into(),
            description: String::new(),
            mac: String::new(),
            ipv4: vec![IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))],
            is_up: true,
            has_link_local: false,
        };
        assert!(score_enet_candidate(&lo, "") < 0);
        assert!(is_software_loopback("loopback-interface 1", ""));
        assert!(is_software_loopback(r"\Device\NPF_Loopback", "Adapter for loopback traffic capture"));
        let host = InterfaceInfo {
            name: "BMW-ENET".into(),
            description: "Microsoft KM-TEST Loopback Adapter".into(),
            mac: String::new(),
            ipv4: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 1, 1))],
            is_up: true,
            has_link_local: true,
        };
        assert!(score_enet_candidate(&host, "") < 0);
    }
}
