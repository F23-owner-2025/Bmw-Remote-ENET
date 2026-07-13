//! BMW ENET Gateway GUI (egui).

use chrono::Local;
use clap::Parser;
use eframe::egui;
use enet_core::safety::FlashSafetyReport;
use enet_core::state::GatewayState;
use enet_core::stats::StatsSnapshot;
use serde::Deserialize;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "enet-gui")]
struct Args {
    /// Gateway API base URL
    #[arg(long, default_value = "http://127.0.0.1:47901")]
    api: String,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct StatusResponse {
    state: GatewayState,
    stats: StatsSnapshot,
    cpu_pct: f64,
    memory_used: u64,
    memory_total: u64,
    flash_safety: FlashSafetyReport,
}

struct GatewayApp {
    api: String,
    status: StatusResponse,
    last_fetch: Instant,
    log_lines: Vec<String>,
    settings_open: bool,
    password: String,
    tunnel_port: String,
    error: Option<String>,
    client: reqwest::blocking::Client,
}

impl GatewayApp {
    fn new(api: String) -> Self {
        Self {
            api,
            status: StatusResponse::default(),
            last_fetch: Instant::now() - Duration::from_secs(10),
            log_lines: vec![format!("{}  GUI started", Local::now().format("%H:%M:%S"))],
            settings_open: false,
            password: String::new(),
            tunnel_port: "47900".into(),
            error: None,
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_millis(800))
                .build()
                .expect("http client"),
        }
    }

    fn push_log(&mut self, msg: impl Into<String>) {
        self.log_lines
            .push(format!("{}  {}", Local::now().format("%H:%M:%S"), msg.into()));
        if self.log_lines.len() > 500 {
            self.log_lines.drain(0..self.log_lines.len() - 500);
        }
    }

    fn refresh(&mut self) {
        match self.client.get(format!("{}/api/status", self.api)).send() {
            Ok(resp) => match resp.json::<StatusResponse>() {
                Ok(s) => {
                    self.status = s;
                    self.error = None;
                }
                Err(e) => self.error = Some(format!("parse status: {e}")),
            },
            Err(e) => self.error = Some(format!("API unreachable: {e}")),
        }
        self.last_fetch = Instant::now();
    }

    fn post(&mut self, path: &str) {
        match self.client.post(format!("{}{path}", self.api)).send() {
            Ok(_) => self.push_log(format!("OK {path}")),
            Err(e) => {
                self.push_log(format!("FAIL {path}: {e}"));
                self.error = Some(e.to_string());
            }
        }
    }
}

impl eframe::App for GatewayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.last_fetch.elapsed() > Duration::from_millis(500) {
            self.refresh();
        }
        ctx.request_repaint_after(Duration::from_millis(250));

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(18, 22, 28);
        visuals.window_fill = egui::Color32::from_rgb(24, 30, 38);
        visuals.override_text_color = Some(egui::Color32::from_rgb(230, 234, 240));
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(36, 46, 58);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(48, 62, 78);
        visuals.selection.bg_fill = egui::Color32::from_rgb(0, 140, 160);
        ctx.set_visuals(visuals);

        egui::TopBottomPanel::top("brand").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.heading(
                    egui::RichText::new("BMW ENET Gateway")
                        .size(28.0)
                        .color(egui::Color32::from_rgb(220, 230, 240)),
                );
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("F-Series remote diagnostics bridge")
                        .size(14.0)
                        .color(egui::Color32::from_rgb(140, 160, 175)),
                );
            });
            ui.add_space(6.0);
        });

        egui::TopBottomPanel::bottom("actions").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Start").clicked() {
                    self.post("/api/start");
                }
                if ui.button("Stop").clicked() {
                    self.post("/api/stop");
                }
                if ui.button("Restart").clicked() {
                    self.post("/api/restart");
                }
                if ui.button("Settings").clicked() {
                    self.settings_open = true;
                }
                if ui.button("Diagnostics").clicked() {
                    self.refresh();
                    self.push_log("Diagnostics refresh");
                }
                if ui.button("Export Logs").clicked() {
                    self.post("/api/export-logs");
                }
            });
            ui.add_space(6.0);
        });

        egui::SidePanel::left("status_panel")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Status");
                ui.separator();
                status_row(ui, "Gateway Running", self.status.state.gateway_running);
                status_row(ui, "Laptop Connected", self.status.state.laptop_connected);
                status_row(ui, "Vehicle Connected", self.status.state.vehicle.link_up);
                status_row(ui, "Vehicle Awake", self.status.state.vehicle.awake);
                status_row(
                    ui,
                    "ENET / Tunnel",
                    matches!(
                        self.status.state.connection,
                        enet_core::state::ConnectionState::Connected
                    ),
                );
                ui.separator();
                ui.label(format!("State: {:?}", self.status.state.connection));
                ui.label(&self.status.state.status_message);
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(220, 90, 70), err);
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Telemetry");
            ui.separator();
            let s = &self.status.stats;
            ui.columns(3, |cols| {
                cols[0].label(format!("TX pps: {:.1}", s.tx_pps));
                cols[0].label(format!("RX pps: {:.1}", s.rx_pps));
                cols[1].label(format!("RTT: {:.2} ms", s.rtt_ms));
                cols[1].label(format!("RTT p99: {:.2} ms", s.rtt_p99_ms));
                cols[2].label(format!("Loss: {:.4}%", s.loss_rate * 100.0));
                cols[2].label(format!("CPU: {:.1}%", self.status.cpu_pct));
            });
            ui.label(format!(
                "Bandwidth ~ TX {:.0} B/s  RX {:.0} B/s",
                s.tx_bps / 8.0,
                s.rx_bps / 8.0
            ));
            ui.label(format!(
                "Errors: {}  Dropped: {}  Reconnects: {}",
                s.errors, s.dropped, s.reconnects
            ));
            ui.label(format!(
                "Memory: {:.1} / {:.1} MB",
                self.status.memory_used as f64 / 1_048_576.0,
                self.status.memory_total as f64 / 1_048_576.0
            ));

            ui.add_space(12.0);
            ui.heading("Flash Safety");
            ui.separator();
            let safe = self.status.flash_safety.safe;
            let color = if safe {
                egui::Color32::from_rgb(60, 180, 120)
            } else {
                egui::Color32::from_rgb(220, 120, 60)
            };
            ui.colored_label(
                color,
                if safe {
                    "SAFE — thresholds met"
                } else {
                    "NOT SAFE — do not flash"
                },
            );
            ui.label(&self.status.flash_safety.warning);
            for r in &self.status.flash_safety.reasons {
                ui.label(format!("• {r}"));
            }

            ui.add_space(12.0);
            ui.heading("Log");
            ui.separator();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .max_height(220.0)
                .show(ui, |ui| {
                    for line in &self.log_lines {
                        ui.monospace(line);
                    }
                });
        });

        if self.settings_open {
            egui::Window::new("Settings")
                .collapsible(false)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.label("Tunnel port");
                    ui.text_edit_singleline(&mut self.tunnel_port);
                    ui.label("Password (PSK)");
                    ui.text_edit_singleline(&mut self.password);
                    if ui.button("Save").clicked() {
                        let port: u16 = self.tunnel_port.parse().unwrap_or(47900);
                        let body = serde_json::json!({
                            "tunnel_port": port,
                            "password": self.password,
                        });
                        let _ = self
                            .client
                            .post(format!("{}/api/settings", self.api))
                            .json(&body)
                            .send();
                        self.push_log("Settings saved");
                        self.settings_open = false;
                    }
                    if ui.button("Close").clicked() {
                        self.settings_open = false;
                    }
                });
        }
    }
}

fn status_row(ui: &mut egui::Ui, label: &str, ok: bool) {
    ui.horizontal(|ui| {
        let (dot, color) = if ok {
            ("●", egui::Color32::from_rgb(60, 180, 120))
        } else {
            ("●", egui::Color32::from_rgb(120, 130, 140))
        };
        ui.colored_label(color, dot);
        ui.label(label);
    });
}

fn main() -> eframe::Result<()> {
    let args = Args::parse();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 640.0])
            .with_title("BMW ENET Gateway"),
        ..Default::default()
    };
    eframe::run_native(
        "BMW ENET Gateway",
        options,
        Box::new(move |_cc| Ok(Box::new(GatewayApp::new(args.api)))),
    )
}
