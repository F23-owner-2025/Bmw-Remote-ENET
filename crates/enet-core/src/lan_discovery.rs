//! LAN auto-discovery for gateway ↔ agent pairing (no manual IP required).
//!
//! Works with DHCP / changing IPs: Host beacons advertise all LAN IPs; Client
//! finds the Host by pair code and re-discovers when the address changes.

use enet_protocol::magic::DEFAULT_DISCOVERY_PORT;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use tokio::net::UdpSocket;
use tracing::{debug, info, warn};

/// Discovery beacon / query magic.
pub const DISCOVERY_MAGIC: &str = "BMWENET1";

/// Message exchanged on the discovery UDP port.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DiscoveryMessage {
    /// Agent looking for a gateway.
    Query {
        /// Optional pair code filter (empty = accept any).
        pair_code: String,
    },
    /// Gateway advertising itself.
    Announce {
        /// Human hostname.
        hostname: String,
        /// Software version.
        version: String,
        /// Tunnel UDP port.
        tunnel_port: u16,
        /// HTTP dashboard / API port.
        api_port: u16,
        /// Pair code shown in the desktop UI.
        pair_code: String,
        /// Whether a password is configured.
        password_required: bool,
        /// All usable LAN IPv4 addresses on the Host (DHCP-friendly).
        #[serde(default)]
        lan_ips: Vec<String>,
    },
}

impl DiscoveryMessage {
    /// Encode as UDP payload: magic + JSON.
    pub fn encode(&self) -> anyhow::Result<Vec<u8>> {
        let mut out = DISCOVERY_MAGIC.as_bytes().to_vec();
        out.extend_from_slice(&serde_json::to_vec(self)?);
        Ok(out)
    }

    /// Decode from UDP payload.
    pub fn decode(data: &[u8]) -> anyhow::Result<Self> {
        let magic = DISCOVERY_MAGIC.as_bytes();
        if data.len() < magic.len() || &data[..magic.len()] != magic {
            anyhow::bail!("bad discovery magic");
        }
        Ok(serde_json::from_slice(&data[magic.len()..])?)
    }
}

/// Generate a short human-friendly pair code (e.g. `BMW-7K2Q`).
pub fn generate_pair_code() -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let mut rng_bytes = [0u8; 4];
    // Prefer OS randomness; fall back to time-based if unavailable.
    if getrandom_fill(&mut rng_bytes).is_err() {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        rng_bytes = t.to_le_bytes()[..4].try_into().unwrap_or([1, 2, 3, 4]);
    }
    let mut code = String::from("BMW-");
    for b in rng_bytes {
        code.push(ALPHABET[(b as usize) % ALPHABET.len()] as char);
    }
    code
}

fn getrandom_fill(buf: &mut [u8]) -> Result<(), ()> {
    #[cfg(unix)]
    {
        use std::fs::File;
        use std::io::Read;
        let mut f = File::open("/dev/urandom").map_err(|_| ())?;
        f.read_exact(buf).map_err(|_| ())
    }
    #[cfg(windows)]
    {
        // RtlGenRandom — CSPRNG available on every supported Windows.
        #[link(name = "advapi32")]
        extern "system" {
            #[link_name = "SystemFunction036"]
            fn rtl_gen_random(buf: *mut u8, len: u32) -> u8;
        }
        let ok = unsafe { rtl_gen_random(buf.as_mut_ptr(), buf.len() as u32) };
        if ok != 0 {
            Ok(())
        } else {
            Err(())
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = buf;
        Err(())
    }
}

/// Resolved gateway from LAN discovery.
#[derive(Debug, Clone)]
pub struct DiscoveredGateway {
    /// Best reachable IP for the tunnel (may differ from UDP source on multi-homed Host).
    pub addr: IpAddr,
    /// Tunnel port.
    pub tunnel_port: u16,
    /// API port.
    pub api_port: u16,
    /// Hostname.
    pub hostname: String,
    /// Pair code.
    pub pair_code: String,
    /// Password required flag.
    pub password_required: bool,
    /// All LAN IPs advertised by the Host.
    pub lan_ips: Vec<IpAddr>,
}

fn is_usable_lan_v4(v4: Ipv4Addr) -> bool {
    !v4.is_loopback()
        && !v4.is_unspecified()
        && !v4.is_multicast()
        && !v4.is_broadcast()
        && !v4.is_link_local()
}

fn same_slash24(a: Ipv4Addr, b: Ipv4Addr) -> bool {
    let ao = a.octets();
    let bo = b.octets();
    ao[0] == bo[0] && ao[1] == bo[1] && ao[2] == bo[2]
}

fn slash24_broadcast(v4: Ipv4Addr) -> Ipv4Addr {
    let o = v4.octets();
    Ipv4Addr::new(o[0], o[1], o[2], 255)
}

/// Enumerate local IPv4 addresses suitable for LAN discovery (excludes loopback / link-local).
pub fn list_lan_ipv4s() -> Vec<Ipv4Addr> {
    let mut out = Vec::new();
    if let Ok(ifaces) = if_addrs::get_if_addrs() {
        for iface in ifaces {
            if iface.is_loopback() {
                continue;
            }
            if let if_addrs::IfAddr::V4(v4) = iface.addr {
                if is_usable_lan_v4(v4.ip) {
                    out.push(v4.ip);
                }
            }
        }
    }
    // Fallback: primary outbound IP (works even if if_addrs is empty).
    if out.is_empty() {
        if let Ok(sock) = std::net::UdpSocket::bind("0.0.0.0:0") {
            if sock.connect("8.8.8.8:80").is_ok() {
                if let Ok(local) = sock.local_addr() {
                    if let IpAddr::V4(v4) = local.ip() {
                        if is_usable_lan_v4(v4) {
                            out.push(v4);
                        }
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Local IPv4 + matching /24 broadcast for each interface.
fn local_iface_broadcasts() -> Vec<(Ipv4Addr, Ipv4Addr)> {
    let mut out = Vec::new();
    for ip in list_lan_ipv4s() {
        out.push((ip, slash24_broadcast(ip)));
    }
    out.sort_by_key(|(ip, _)| *ip);
    out.dedup();
    out
}

/// Broadcast targets: global + every local /24.
fn local_subnet_broadcasts() -> Vec<Ipv4Addr> {
    let mut out = vec![Ipv4Addr::BROADCAST];
    for (_, bcast) in local_iface_broadcasts() {
        out.push(bcast);
    }
    out.sort();
    out.dedup();
    out
}

/// Pick the Host IP the Client should dial (prefer same /24 as a local NIC).
pub fn pick_reachable_host_ip(src: IpAddr, advertised: &[IpAddr], local: &[Ipv4Addr]) -> IpAddr {
    let mut candidates: Vec<Ipv4Addr> = Vec::new();
    if let IpAddr::V4(v4) = src {
        if is_usable_lan_v4(v4) {
            candidates.push(v4);
        }
    }
    for ip in advertised {
        if let IpAddr::V4(v4) = ip {
            if is_usable_lan_v4(*v4) {
                candidates.push(*v4);
            }
        }
    }
    candidates.sort();
    candidates.dedup();

    for local_ip in local {
        for cand in &candidates {
            if same_slash24(*local_ip, *cand) {
                return IpAddr::V4(*cand);
            }
        }
    }
    candidates
        .first()
        .copied()
        .map(IpAddr::V4)
        .unwrap_or(src)
}

/// Broadcast a discovery query and wait for announces.
pub async fn discover_gateways(
    discovery_port: u16,
    pair_code: &str,
    timeout: Duration,
) -> anyhow::Result<Vec<DiscoveredGateway>> {
    let sock = UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).await?;
    sock.set_broadcast(true)?;

    // Also listen on the discovery port so we hear Host beacons (not only query replies).
    // Binding may fail if another process already owns 47902 — that's fine.
    let passive = UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, discovery_port)))
        .await
        .ok();
    if let Some(ref p) = passive {
        let _ = p.set_broadcast(true);
    }

    let query = DiscoveryMessage::Query {
        pair_code: pair_code.to_string(),
    };
    let payload = query.encode()?;
    let local_ips = list_lan_ipv4s();

    // Send from the unbound socket to every broadcast target.
    for bcast in local_subnet_broadcasts() {
        let dest = SocketAddr::from((bcast, discovery_port));
        match sock.send_to(&payload, dest).await {
            Ok(_) => debug!(%dest, "sent discovery query"),
            Err(e) => debug!(%dest, error = %e, "discovery send failed"),
        }
    }

    // Also send from each local interface so Wi‑Fi↔wired replies use the right path.
    for (local, bcast) in local_iface_broadcasts() {
        if let Ok(s) = UdpSocket::bind(SocketAddr::from((local, 0))).await {
            let _ = s.set_broadcast(true);
            let dest = SocketAddr::from((bcast, discovery_port));
            let _ = s.send_to(&payload, dest).await;
            let _ = s
                .send_to(&payload, SocketAddr::from((Ipv4Addr::BROADCAST, discovery_port)))
                .await;
        }
    }

    let mut found = Vec::new();
    let deadline = tokio::time::Instant::now() + timeout;
    let mut buf_a = vec![0u8; 2048];
    let mut buf_b = vec![0u8; 2048];
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let recv = async {
            if let Some(ref p) = passive {
                tokio::select! {
                    r = sock.recv_from(&mut buf_a) => r.map(|(n, src)| (n, src, true)),
                    r = p.recv_from(&mut buf_b) => r.map(|(n, src)| (n, src, false)),
                }
            } else {
                sock.recv_from(&mut buf_a)
                    .await
                    .map(|(n, src)| (n, src, true))
            }
        };
        match tokio::time::timeout(remaining, recv).await {
            Ok(Ok((n, src, from_query_sock))) => {
                let data = if from_query_sock {
                    &buf_a[..n]
                } else {
                    &buf_b[..n]
                };
                match DiscoveryMessage::decode(data) {
                    Ok(DiscoveryMessage::Announce {
                        hostname,
                        tunnel_port,
                        api_port,
                        pair_code: announced_code,
                        password_required,
                        lan_ips,
                        ..
                    }) => {
                        if !pair_code.is_empty()
                            && !announced_code.is_empty()
                            && !pair_code.eq_ignore_ascii_case(&announced_code)
                        {
                            debug!(
                                %src,
                                want = %pair_code,
                                got = %announced_code,
                                "ignoring gateway with different pair code"
                            );
                            continue;
                        }
                        let advertised: Vec<IpAddr> = lan_ips
                            .iter()
                            .filter_map(|s| s.parse().ok())
                            .collect();
                        let addr = pick_reachable_host_ip(src.ip(), &advertised, &local_ips);
                        if found.iter().any(|g: &DiscoveredGateway| g.addr == addr) {
                            continue;
                        }
                        info!(%addr, from = %src, %hostname, "discovered gateway");
                        found.push(DiscoveredGateway {
                            addr,
                            tunnel_port,
                            api_port,
                            hostname,
                            pair_code: announced_code,
                            password_required,
                            lan_ips: advertised,
                        });
                    }
                    Ok(_) => {}
                    Err(e) => debug!(error = %e, "ignore discovery packet"),
                }
            }
            Ok(Err(e)) => warn!(error = %e, "discovery recv error"),
            Err(_) => break,
        }
    }
    Ok(found)
}

async fn send_announce_everywhere(payload: &[u8], discovery_port: u16) {
    // Unbound socket → global + each /24 broadcast.
    if let Ok(sock) = UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).await {
        let _ = sock.set_broadcast(true);
        for bcast in local_subnet_broadcasts() {
            let _ = sock
                .send_to(payload, SocketAddr::from((bcast, discovery_port)))
                .await;
        }
    }
    // Per-interface send so multi-homed Hosts advertise from the wired NIC too.
    for (local, bcast) in local_iface_broadcasts() {
        if let Ok(s) = UdpSocket::bind(SocketAddr::from((local, 0))).await {
            let _ = s.set_broadcast(true);
            let _ = s
                .send_to(payload, SocketAddr::from((bcast, discovery_port)))
                .await;
            let _ = s
                .send_to(
                    payload,
                    SocketAddr::from((Ipv4Addr::BROADCAST, discovery_port)),
                )
                .await;
        }
    }
}

/// Run gateway discovery responder / beacon loop until cancelled.
pub async fn run_gateway_beacon(
    discovery_port: u16,
    tunnel_port: u16,
    api_port: u16,
    pair_code: String,
    password_required: bool,
    version: String,
) -> anyhow::Result<()> {
    let sock = UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, discovery_port))).await?;
    sock.set_broadcast(true)?;
    info!(port = discovery_port, %pair_code, "discovery beacon listening");

    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "desktop".into());

    let mut buf = vec![0u8; 2048];
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    loop {
        tokio::select! {
            _ = interval.tick() => {
                let lan_ips: Vec<String> = list_lan_ipv4s().into_iter().map(|ip| ip.to_string()).collect();
                let announce = DiscoveryMessage::Announce {
                    hostname: hostname.clone(),
                    version: version.clone(),
                    tunnel_port,
                    api_port,
                    pair_code: pair_code.clone(),
                    password_required,
                    lan_ips,
                };
                if let Ok(payload) = announce.encode() {
                    send_announce_everywhere(&payload, discovery_port).await;
                }
            }
            res = sock.recv_from(&mut buf) => {
                if let Ok((n, src)) = res {
                    if let Ok(DiscoveryMessage::Query { pair_code: want }) = DiscoveryMessage::decode(&buf[..n]) {
                        if !want.is_empty() && !pair_code.is_empty() && !want.eq_ignore_ascii_case(&pair_code) {
                            continue;
                        }
                        let lan_ips: Vec<String> = list_lan_ipv4s().into_iter().map(|ip| ip.to_string()).collect();
                        let announce = DiscoveryMessage::Announce {
                            hostname: hostname.clone(),
                            version: version.clone(),
                            tunnel_port,
                            api_port,
                            pair_code: pair_code.clone(),
                            password_required,
                            lan_ips,
                        };
                        if let Ok(payload) = announce.encode() {
                            let _ = sock.send_to(&payload, src).await;
                            debug!(%src, "answered discovery query");
                        }
                    }
                }
            }
        }
    }
}

/// Default discovery port helper.
pub fn default_discovery_port() -> u16 {
    DEFAULT_DISCOVERY_PORT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_message() {
        let m = DiscoveryMessage::Announce {
            hostname: "desk".into(),
            version: "0.1.0".into(),
            tunnel_port: 47900,
            api_port: 47901,
            pair_code: "BMW-ABCD".into(),
            password_required: false,
            lan_ips: vec!["192.168.1.10".into()],
        };
        let enc = m.encode().unwrap();
        let dec = DiscoveryMessage::decode(&enc).unwrap();
        assert_eq!(m, dec);
    }

    #[test]
    fn announce_without_lan_ips_still_decodes() {
        let json = br#"{"type":"announce","hostname":"desk","version":"0.1.0","tunnel_port":47900,"api_port":47901,"pair_code":"BMW-ABCD","password_required":false}"#;
        let mut data = DISCOVERY_MAGIC.as_bytes().to_vec();
        data.extend_from_slice(json);
        let m = DiscoveryMessage::decode(&data).unwrap();
        match m {
            DiscoveryMessage::Announce { lan_ips, .. } => assert!(lan_ips.is_empty()),
            _ => panic!("expected announce"),
        }
    }

    #[test]
    fn pair_code_format() {
        let c = generate_pair_code();
        assert!(c.starts_with("BMW-"));
        assert_eq!(c.len(), 8);
    }

    #[test]
    fn pick_same_subnet_ip() {
        let local = vec![Ipv4Addr::new(192, 168, 1, 50)];
        let advertised = [
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 187)),
        ];
        let src = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let picked = pick_reachable_host_ip(src, &advertised, &local);
        assert_eq!(picked, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 187)));
    }

    #[tokio::test]
    async fn discover_local_beacon() {
        let code = "BMW-TEST".to_string();
        let port = 47992u16;
        let beacon = tokio::spawn(run_gateway_beacon(
            port,
            47900,
            47901,
            code.clone(),
            false,
            "test".into(),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        let found = discover_gateways(port, &code, Duration::from_millis(800))
            .await
            .unwrap();
        assert!(!found.is_empty());
        assert_eq!(found[0].pair_code, code);
        beacon.abort();
    }
}
