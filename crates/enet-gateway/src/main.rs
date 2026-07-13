//! Desktop ENET gateway — Windows service-compatible tunnel server + status API.

use anyhow::Context;
use async_trait::async_trait;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use clap::Parser;
use enet_core::config::{GatewayConfig, Role};
use enet_core::health::HealthMonitor;
use enet_core::logging::init_logging;
use enet_core::safety::{FlashSafetyChecker, SafetyThresholds};
use enet_core::state::{ConnectionState, GatewayState};
use enet_tunnel::{EthernetPort, SimulatedEthernet, TunnelEngine, TunnelHandle, TunnelOptions};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "enet-gateway", about = "BMW ENET desktop gateway")]
struct Args {
    /// Path to gateway.toml
    #[arg(short, long, default_value = "config/gateway.toml")]
    config: PathBuf,
    /// Use simulated TAP (no Wintun)
    #[arg(long)]
    simulate: bool,
    /// Run once and exit after N seconds (for tests)
    #[arg(long)]
    run_seconds: Option<u64>,
}

/// Virtual NIC placeholder until Wintun/TAP is installed.
struct VirtualNic {
    name: String,
    inner: Arc<SimulatedEthernet>,
}

#[async_trait]
impl EthernetPort for VirtualNic {
    fn name(&self) -> &str {
        &self.name
    }
    async fn link_up(&self) -> bool {
        self.inner.link_up().await
    }
    async fn recv(&self) -> anyhow::Result<Bytes> {
        self.inner.recv().await
    }
    async fn send(&self, frame: Bytes) -> anyhow::Result<()> {
        self.inner.send(frame).await
    }
}

#[derive(Clone)]
struct AppState {
    cfg: Arc<RwLock<GatewayConfig>>,
    handle: Arc<RwLock<Option<TunnelHandle>>>,
    health: Arc<RwLock<HealthMonitor>>,
}

#[derive(Serialize)]
struct StatusResponse {
    state: GatewayState,
    stats: enet_core::stats::StatsSnapshot,
    cpu_pct: f64,
    memory_used: u64,
    memory_total: u64,
    flash_safety: enet_core::safety::FlashSafetyReport,
}

#[derive(Deserialize)]
struct SettingsUpdate {
    tunnel_port: Option<u16>,
    password: Option<String>,
    log_level: Option<String>,
    reconnect_delay_ms: Option<u64>,
    peer_timeout_ms: Option<u64>,
    require_crypto: Option<bool>,
    auto_start: Option<bool>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut cfg = GatewayConfig::load(&args.config).unwrap_or_default();
    cfg.role = Role::Gateway;
    let _guard = init_logging(cfg.log_level, &cfg.log_dir)?;
    info!(version = env!("CARGO_PKG_VERSION"), "enet-gateway starting");

    if cfg.manage_firewall {
        info!(
            "firewall: ensure UDP {} and TCP {} are allowed from LAN only",
            cfg.tunnel_port, cfg.api_port
        );
    }

    let (tap, _tool_peer) = SimulatedEthernet::pair(&cfg.virtual_interface, "tool-stack");
    tap.set_link(true);
    // Keep the tool-side endpoint alive for future packet-injection diagnostics.
    std::mem::forget(_tool_peer);

    let eth: Arc<dyn EthernetPort> = Arc::new(VirtualNic {
        name: cfg.virtual_interface.clone(),
        inner: tap,
    });

    if !args.simulate {
        info!(
            iface = %cfg.virtual_interface,
            ip = %cfg.tester_ip,
            mask = %cfg.tester_mask,
            "Wintun/TAP integration: assign tester IP on virtual NIC (see docs/INSTALL.md)"
        );
    }

    let bind = SocketAddr::from((
        cfg.bind_addr.unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED)),
        cfg.tunnel_port,
    ));
    let opts = TunnelOptions {
        bind,
        peer: cfg.peer_addr.map(|ip| SocketAddr::new(ip, 0)),
        allowed_cidrs: cfg.allowed_cidrs.clone(),
        crypto: None,
        require_crypto: cfg.require_crypto,
        keepalive_interval_ms: cfg.keepalive_interval_ms,
        peer_timeout_ms: cfg.peer_timeout_ms,
        role: "gateway".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    }
    .with_password(&cfg.password, cfg.require_crypto);

    let engine = TunnelEngine::new(opts, eth);
    let handle = engine
        .run()
        .await
        .context("failed to bind gateway tunnel")?;
    info!(%bind, "gateway tunnel listening");

    let app_state = AppState {
        cfg: Arc::new(RwLock::new(cfg.clone())),
        handle: Arc::new(RwLock::new(Some(handle.clone()))),
        health: Arc::new(RwLock::new(HealthMonitor::new())),
    };

    let api = Router::new()
        .route("/api/status", get(api_status))
        .route("/api/start", post(api_start))
        .route("/api/stop", post(api_stop))
        .route("/api/restart", post(api_restart))
        .route("/api/settings", get(api_get_settings).post(api_set_settings))
        .route("/api/safety", get(api_safety))
        .route("/api/export-logs", post(api_export_logs))
        .layer(CorsLayer::permissive())
        .with_state(app_state.clone());

    let api_addr = SocketAddr::from((Ipv4Addr::LOCALHOST, cfg.api_port));
    info!(%api_addr, "control API listening");
    let server = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(api_addr).await.expect("api bind");
        axum::serve(listener, api).await.expect("api serve");
    });

    if let Some(secs) = args.run_seconds {
        tokio::time::sleep(Duration::from_secs(secs)).await;
        handle.stop();
        server.abort();
        return Ok(());
    }

    tokio::signal::ctrl_c().await.ok();
    info!("shutdown");
    handle.stop();
    server.abort();
    Ok(())
}

async fn api_status(State(state): State<AppState>) -> Json<StatusResponse> {
    let cfg = state.cfg.read().clone();
    let (mut gateway_state, stats) = {
        let guard = state.handle.read();
        if let Some(h) = guard.as_ref() {
            (h.snapshot_state(), h.stats.snapshot())
        } else {
            (
                GatewayState {
                    connection: ConnectionState::Stopped,
                    gateway_running: false,
                    status_message: "Stopped".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                    ..Default::default()
                },
                enet_core::stats::PacketStats::new().snapshot(),
            )
        }
    };
    gateway_state.gateway_running = state.handle.read().is_some();

    let (cpu_pct, memory_used, memory_total) = {
        let mut health = state.health.write();
        (
            health.cpu_pct(),
            health.memory_used_bytes(),
            health.memory_total_bytes(),
        )
    };

    let checker = FlashSafetyChecker::new(SafetyThresholds::from(&cfg));
    let flash_safety = checker.evaluate(
        &stats,
        &gateway_state.vehicle,
        cpu_pct,
        gateway_state.laptop_connected
            || matches!(gateway_state.connection, ConnectionState::Connected),
    );

    Json(StatusResponse {
        state: gateway_state,
        stats,
        cpu_pct,
        memory_used,
        memory_total,
        flash_safety,
    })
}

async fn api_start(State(state): State<AppState>) -> Json<serde_json::Value> {
    if state.handle.read().is_some() {
        return Json(serde_json::json!({"ok": true, "message": "already running"}));
    }
    Json(serde_json::json!({
        "ok": false,
        "message": "restart the enet-gateway process to start (service mode)"
    }))
}

async fn api_stop(State(state): State<AppState>) -> Json<serde_json::Value> {
    if let Some(h) = state.handle.write().take() {
        h.stop();
        Json(serde_json::json!({"ok": true}))
    } else {
        Json(serde_json::json!({"ok": true, "message": "already stopped"}))
    }
}

async fn api_restart(State(state): State<AppState>) -> Json<serde_json::Value> {
    if let Some(h) = state.handle.read().as_ref() {
        h.stop();
    }
    Json(serde_json::json!({
        "ok": true,
        "message": "stop signaled; service manager should restart the process"
    }))
}

async fn api_get_settings(State(state): State<AppState>) -> Json<GatewayConfig> {
    Json(state.cfg.read().clone())
}

async fn api_set_settings(
    State(state): State<AppState>,
    Json(update): Json<SettingsUpdate>,
) -> Json<serde_json::Value> {
    let mut cfg = state.cfg.write();
    if let Some(p) = update.tunnel_port {
        cfg.tunnel_port = p;
    }
    if let Some(p) = update.password {
        cfg.password = p;
    }
    if let Some(ms) = update.reconnect_delay_ms {
        cfg.reconnect_delay_ms = ms;
    }
    if let Some(ms) = update.peer_timeout_ms {
        cfg.peer_timeout_ms = ms;
    }
    if let Some(v) = update.require_crypto {
        cfg.require_crypto = v;
    }
    if let Some(v) = update.auto_start {
        cfg.auto_start = v;
    }
    if let Some(level) = update.log_level {
        cfg.log_level = match level.to_lowercase().as_str() {
            "error" => enet_core::config::LogLevel::Error,
            "warn" => enet_core::config::LogLevel::Warn,
            "debug" => enet_core::config::LogLevel::Debug,
            "trace" => enet_core::config::LogLevel::Trace,
            _ => enet_core::config::LogLevel::Info,
        };
    }
    let _ = cfg.save(GatewayConfig::default_path_for(Role::Gateway));
    Json(serde_json::json!({"ok": true}))
}

async fn api_safety(State(state): State<AppState>) -> Json<enet_core::safety::FlashSafetyReport> {
    let status = api_status(State(state)).await;
    Json(status.0.flash_safety)
}

async fn api_export_logs(State(state): State<AppState>) -> Json<serde_json::Value> {
    let cfg = state.cfg.read().clone();
    let src = cfg.log_dir.clone();
    let dest = std::env::temp_dir().join(format!(
        "enet-logs-{}.zip.txt",
        chrono_lite_timestamp()
    ));
    // Simple export: concatenate log files into one text bundle (zip optional later).
    let mut bundle = String::new();
    if src.exists() {
        if let Ok(entries) = std::fs::read_dir(&src) {
            for entry in entries.flatten() {
                if let Ok(text) = std::fs::read_to_string(entry.path()) {
                    bundle.push_str(&format!("===== {} =====\n", entry.path().display()));
                    bundle.push_str(&text);
                    bundle.push('\n');
                }
            }
        }
    }
    match std::fs::write(&dest, bundle) {
        Ok(()) => Json(serde_json::json!({"ok": true, "path": dest})),
        Err(e) => {
            warn!(error = %e, "export failed");
            Json(serde_json::json!({"ok": false, "error": e.to_string()}))
        }
    }
}

fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".into())
}
