//! LAN auto-discovery for gateway ↔ agent pairing (no manual IP required).

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
        // Simple time-based fill when /dev/urandom is unavailable.
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((t >> (i * 8)) as u8).wrapping_add(i as u8).wrapping_mul(31);
        }
        Ok(())
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
    /// Source IP of the announce packet.
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
}

/// Guess /24 subnet broadcast addresses for local IPv4 interfaces.
async fn local_subnet_broadcasts() -> Vec<Ipv4Addr> {
    let mut out = vec![Ipv4Addr::BROADCAST];
    // Learn primary outbound IPv4 (works on Windows/macOS/Linux).
    if let Ok(sock) = UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).await {
        if sock
            .connect(SocketAddr::from((Ipv4Addr::new(8, 8, 8, 8), 80)))
            .await
            .is_ok()
        {
            if let Ok(local) = sock.local_addr() {
                if let IpAddr::V4(v4) = local.ip() {
                    let o = v4.octets();
                    if o[0] != 127 {
                        out.push(Ipv4Addr::new(o[0], o[1], o[2], 255));
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
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

    for bcast in local_subnet_broadcasts().await {
        let dest = SocketAddr::from((bcast, discovery_port));
        match sock.send_to(&payload, dest).await {
            Ok(_) => debug!(%dest, "sent discovery query"),
            Err(e) => debug!(%dest, error = %e, "discovery send failed"),
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
                    if found.iter().any(|g: &DiscoveredGateway| g.addr == src.ip()) {
                        continue;
                    }
                    info!(%src, %hostname, "discovered gateway");
                    found.push(DiscoveredGateway {
                        addr: src.ip(),
                        tunnel_port,
                        api_port,
                        hostname,
                        pair_code: announced_code,
                        password_required,
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

    let announce = DiscoveryMessage::Announce {
        hostname: hostname.clone(),
        version,
        tunnel_port,
        api_port,
        pair_code: pair_code.clone(),
        password_required,
    };
    let payload = announce.encode()?;

    let mut buf = vec![0u8; 2048];
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    loop {
        tokio::select! {
            _ = interval.tick() => {
                for bcast in local_subnet_broadcasts().await {
                    let _ = sock.send_to(&payload, SocketAddr::from((bcast, discovery_port))).await;
                }
            }
            res = sock.recv_from(&mut buf) => {
                if let Ok((n, src)) = res {
                    if let Ok(DiscoveryMessage::Query { pair_code: want }) = DiscoveryMessage::decode(&buf[..n]) {
                        if !want.is_empty() && !pair_code.is_empty() && !want.eq_ignore_ascii_case(&pair_code) {
                            continue;
                        }
                        let _ = sock.send_to(&payload, src).await;
                        debug!(%src, "answered discovery query");
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
        };
        let enc = m.encode().unwrap();
        let dec = DiscoveryMessage::decode(&enc).unwrap();
        assert_eq!(m, dec);
    }

    #[test]
    fn pair_code_format() {
        let c = generate_pair_code();
        assert!(c.starts_with("BMW-"));
        assert_eq!(c.len(), 8);
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
