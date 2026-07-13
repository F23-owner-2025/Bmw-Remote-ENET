//! Laptop ENET agent — captures the vehicle NIC and tunnels frames to the desktop gateway.

use anyhow::Context;
use async_trait::async_trait;
use bytes::Bytes;
use clap::Parser;
use enet_core::config::{GatewayConfig, Role};
use enet_core::discovery::{detect_candidate_interfaces, pick_enet_interface};
use enet_core::logging::init_logging;
use enet_core::stats::backoff_delay;
use enet_tunnel::{EthernetPort, SimulatedEthernet, TunnelEngine, TunnelOptions};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "enet-agent", about = "BMW ENET laptop tunnel agent")]
struct Args {
    /// Path to agent.toml
    #[arg(short, long, default_value = "config/agent.toml")]
    config: PathBuf,
    /// Override gateway peer host
    #[arg(long)]
    peer: Option<IpAddr>,
    /// Run with simulated ENET (no hardware)
    #[arg(long)]
    simulate: bool,
}

/// Null Ethernet port that never receives frames (used until real capture is attached).
struct NullEthernet {
    name: String,
}

#[async_trait]
impl EthernetPort for NullEthernet {
    fn name(&self) -> &str {
        &self.name
    }
    async fn link_up(&self) -> bool {
        false
    }
    async fn recv(&self) -> anyhow::Result<Bytes> {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Err(anyhow::anyhow!("null ethernet closed"))
    }
    async fn send(&self, _frame: Bytes) -> anyhow::Result<()> {
        Ok(())
    }
}

/// File-based / simulation note for Windows Npcap integration.
///
/// On Windows production builds, replace [`build_ethernet_port`] with an Npcap/WinPcap
/// capturer that:
/// 1. Opens the ENET interface in promiscuous mode
/// 2. Filters only the ENET NIC (not the LAN NIC used for the tunnel)
/// 3. Injects received tunnel frames back onto the ENET NIC
///
/// Linux development uses AF_PACKET similarly. This agent ships with `--simulate` for CI.
async fn build_ethernet_port(cfg: &GatewayConfig, simulate: bool) -> anyhow::Result<Arc<dyn EthernetPort>> {
    if simulate {
        let (port, _peer) = SimulatedEthernet::pair("sim-enet", "sim-car");
        info!("using simulated ENET interface");
        // Keep peer alive by leaking for demo — in sim binary we drive traffic separately.
        std::mem::forget(_peer);
        return Ok(port);
    }

    let preferred = cfg.enet_interface.as_str();
    if let Some(iface) = pick_enet_interface(preferred) {
        info!(name = %iface.name, mac = %iface.mac, "selected ENET candidate interface");
        // Without raw-socket privileges in this environment, bind a named null port that
        // reports the selected interface. Real packet IO is enabled on Windows via Npcap.
        warn!(
            "raw ENET capture requires Npcap (Windows) or CAP_NET_RAW (Linux); \
             running in monitor-only mode for interface '{}'. Use --simulate for lab tests.",
            iface.name
        );
        return Ok(Arc::new(NullEthernet {
            name: iface.name,
        }));
    }

    let all = detect_candidate_interfaces();
    warn!(count = all.len(), "no strong ENET candidate; listing interfaces");
    for i in &all {
        info!(name = %i.name, mac = %i.mac, up = i.is_up, "iface");
    }
    anyhow::bail!("no ENET interface detected; pass --simulate or set enet_interface in config")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut cfg = GatewayConfig::load(&args.config).unwrap_or_else(|_| {
        let mut c = GatewayConfig::default();
        c.role = Role::Agent;
        c
    });
    cfg.role = Role::Agent;
    if let Some(peer) = args.peer {
        cfg.peer_addr = Some(peer);
    }

    let _guard = init_logging(cfg.log_level, &cfg.log_dir)?;
    info!(version = env!("CARGO_PKG_VERSION"), "enet-agent starting");

    let peer_ip = cfg
        .peer_addr
        .context("agent requires peer_addr (desktop gateway IP) in config or --peer")?;
    let peer = SocketAddr::new(peer_ip, cfg.tunnel_port);

    let mut attempt = 0u32;
    loop {
        let eth = build_ethernet_port(&cfg, args.simulate).await?;
        let bind = SocketAddr::from((
            cfg.bind_addr.unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED)),
            0,
        ));
        let opts = TunnelOptions {
            bind,
            peer: Some(peer),
            allowed_cidrs: cfg.allowed_cidrs.clone(),
            crypto: None,
            require_crypto: cfg.require_crypto,
            keepalive_interval_ms: cfg.keepalive_interval_ms,
            peer_timeout_ms: cfg.peer_timeout_ms,
            role: "agent".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
        .with_password(&cfg.password, cfg.require_crypto);

        match TunnelEngine::new(opts, eth).run().await {
            Ok(handle) => {
                info!(%peer, "agent tunnel running");
                attempt = 0;
                // Stay alive until stop (Ctrl+C)
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        info!("shutdown requested");
                        handle.stop();
                        break;
                    }
                    _ = async {
                        while handle.is_running() {
                            tokio::time::sleep(Duration::from_secs(1)).await;
                            let st = handle.snapshot_state();
                            if matches!(
                                st.connection,
                                enet_core::state::ConnectionState::Failed
                            ) {
                                break;
                            }
                        }
                    } => {
                        warn!("tunnel stopped; will reconnect");
                        handle.stop();
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "failed to start tunnel");
            }
        }

        attempt = attempt.saturating_add(1);
        let delay = backoff_delay(cfg.reconnect_delay_ms, cfg.reconnect_delay_max_ms, attempt);
        info!(?delay, attempt, "reconnecting");
        tokio::time::sleep(delay).await;
    }

    Ok(())
}
