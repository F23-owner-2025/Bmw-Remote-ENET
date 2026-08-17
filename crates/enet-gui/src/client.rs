//! Native BMW ENET Client window (laptop at the car).

use eframe::egui;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ClientStatus {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub pair_code: String,
    #[serde(default)]
    pub desktop_connected: bool,
    #[serde(default)]
    pub desktop_peer: String,
    #[serde(default)]
    pub configured_peer: Option<String>,
    #[serde(default)]
    pub enet_interface: String,
    #[serde(default)]
    pub enet_link: bool,
    #[serde(default)]
    pub l2_active: bool,
    #[serde(default)]
    pub l2_label: String,
    #[serde(default)]
    pub vehicle_awake: bool,
    #[serde(default)]
    pub rtt_ms: f64,
    #[serde(default)]
    pub loss_rate: f64,
    #[serde(default)]
    pub friendly: String,
    #[serde(default)]
    pub update_available: Option<String>,
    #[serde(default = "default_true")]
    pub auto_update: bool,
    #[serde(default)]
    pub password_set: bool,
    #[serde(default)]
    pub preferred_enet: String,
    #[serde(default)]
    pub adapters: Vec<AdapterChoice>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct AdapterChoice {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub usable: bool,
}

#[derive(Debug, Deserialize)]
struct ApiMsg {
    #[serde(default)]
    ok: bool,
    #[serde(default)]
    message: String,
    #[serde(default)]
    update_available: Option<String>,
}

pub struct ClientApp {
    api: String,
    status: ClientStatus,
    last_fetch: Instant,
    error: Option<String>,
    pair_code: String,
    password: String,
    show_password: bool,
    settings_open: bool,
    connect_msg: String,
    connect_ok: bool,
    auto_update: bool,
    selected_adapter: String,
    client: reqwest::blocking::Client,
    starting_agent: bool,
}

impl ClientApp {
    pub fn new(api: String) -> Self {
        Self {
            api,
            status: ClientStatus::default(),
            last_fetch: Instant::now() - Duration::from_secs(10),
            error: None,
            pair_code: String::new(),
            password: String::new(),
            show_password: false,
            settings_open: false,
            connect_msg: String::new(),
            connect_ok: false,
            auto_update: true,
            selected_adapter: String::new(),
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_millis(800))
                .build()
                .expect("http client"),
            starting_agent: false,
        }
    }

    fn refresh(&mut self) {
        match self.client.get(format!("{}/api/status", self.api)).send() {
            Ok(resp) => match resp.json::<ClientStatus>() {
                Ok(s) => {
                    if self.pair_code.is_empty() && !s.pair_code.is_empty() {
                        self.pair_code = s.pair_code.clone();
                    }
                    self.auto_update = s.auto_update;
                    if self.selected_adapter.is_empty() {
                        self.selected_adapter = s.preferred_enet.clone();
                    }
                    self.status = s;
                    self.error = None;
                    self.starting_agent = false;
                }
                Err(e) => {
                    self.error = Some(format!("Could not read Client status: {e}"));
                }
            },
            Err(_) => {
                self.error = Some(
                    "Could not reach http://127.0.0.1:47903/api/status. \
                     Re-run Setup so it can stop leftover Host/Client processes, then try again."
                        .into(),
                );
            }
        }
        self.last_fetch = Instant::now();
    }

    fn post_json(&mut self, path: &str, body: serde_json::Value) {
        let slow = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_else(|_| self.client.clone());
        match slow
            .post(format!("{}{path}", self.api))
            .json(&body)
            .send()
        {
            Ok(resp) => match resp.json::<ApiMsg>() {
                Ok(j) => {
                    self.connect_msg = j.message;
                    self.connect_ok = j.ok;
                    if let Some(v) = j.update_available {
                        self.status.update_available = Some(v);
                    }
                }
                Err(e) => {
                    self.connect_msg = format!("Bad response: {e}");
                    self.connect_ok = false;
                }
            },
            Err(e) => {
                self.connect_msg = format!("Could not reach Client service: {e}");
                self.connect_ok = false;
            }
        }
        self.refresh();
    }

    fn post(&mut self, path: &str) {
        self.post_json(path, serde_json::json!({}));
    }

    fn start_local_agent(&mut self) {
        self.starting_agent = true;
        self.connect_msg = "Starting Client service…".into();
        self.connect_ok = true;
        let dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(PathBuf::from));
        let Some(dir) = dir else {
            self.connect_msg = "Could not find the Client install folder.".into();
            self.connect_ok = false;
            self.starting_agent = false;
            return;
        };
        let exe = dir.join("enet-agent.exe");
        if !exe.is_file() {
            self.connect_msg =
                "enet-agent.exe is missing. Re-run BMW-ENET-Setup.exe and choose Client.".into();
            self.connect_ok = false;
            self.starting_agent = false;
            return;
        }
        let cfg = dir.join("config").join("agent.toml");
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("--config").arg(&cfg).current_dir(&dir);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            const DETACHED_PROCESS: u32 = 0x0000_0008;
            cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
        }
        match cmd.spawn() {
            Ok(_) => {
                self.connect_msg = "Client service launching — status should appear in a few seconds."
                    .into();
            }
            Err(e) => {
                self.connect_msg = format!("Failed to start enet-agent: {e}");
                self.connect_ok = false;
                self.starting_agent = false;
            }
        }
    }

    fn open_setup_download(&self) {
        let repo = enet_core::default_github_repo();
        let url = if repo.is_empty() {
            "https://github.com/F23-owner-2025/Bmw-Remote-ENET/releases/latest".into()
        } else {
            format!("https://github.com/{repo}/releases/latest")
        };
        let _ = enet_core::open_http_url(&url);
    }
}

fn row(ui: &mut egui::Ui, label: &str, detail: &str, on: bool) {
    ui.horizontal(|ui| {
        let (mark, color) = if on {
            ("■", egui::Color32::from_rgb(61, 204, 134))
        } else {
            ("□", egui::Color32::from_rgb(120, 132, 148))
        };
        ui.colored_label(color, egui::RichText::new(mark).size(16.0));
        ui.vertical(|ui| {
            ui.label(egui::RichText::new(label).strong());
            if !detail.is_empty() {
                ui.colored_label(
                    egui::Color32::from_rgb(140, 160, 175),
                    egui::RichText::new(detail).size(12.0),
                );
            }
        });
    });
    ui.add_space(6.0);
}

impl eframe::App for ClientApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.last_fetch.elapsed() > Duration::from_millis(500) {
            self.refresh();
        }
        ctx.request_repaint_after(Duration::from_millis(250));

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(11, 15, 20);
        visuals.window_fill = egui::Color32::from_rgb(20, 27, 35);
        visuals.override_text_color = Some(egui::Color32::from_rgb(238, 243, 248));
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(36, 46, 58);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(48, 62, 78);
        visuals.selection.bg_fill = egui::Color32::from_rgb(43, 125, 153);
        ctx.set_visuals(visuals);

        let ver = if self.status.version.is_empty() {
            env!("CARGO_PKG_VERSION").to_string()
        } else {
            self.status.version.clone()
        };

        egui::TopBottomPanel::top("brand").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading(
                        egui::RichText::new(format!("BMW ENET Client (v{ver})"))
                            .size(22.0)
                            .color(egui::Color32::from_rgb(220, 230, 240)),
                    );
                    ui.colored_label(
                        egui::Color32::from_rgb(140, 160, 175),
                        "Laptop at the car",
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Settings").clicked() {
                        self.settings_open = !self.settings_open;
                    }
                });
            });
            ui.add_space(6.0);
            let friendly = if !self.status.friendly.is_empty() {
                self.status.friendly.clone()
            } else if self.error.is_some() {
                "Waiting for the Client service…".into()
            } else {
                "Waiting for the desktop — paste pair code + password".into()
            };
            ui.colored_label(
                egui::Color32::from_rgb(77, 179, 212),
                egui::RichText::new(friendly).size(14.0),
            );
            ui.add_space(6.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            {
                let s = &self.status;
                let desk_detail = if s.desktop_connected {
                    if s.desktop_peer.is_empty() {
                        "Tunnel up".into()
                    } else {
                        s.desktop_peer.clone()
                    }
                } else if let Some(p) = &s.configured_peer {
                    format!("{p}:47900")
                } else {
                    "Auto-finding / dialing desktop…".into()
                };
                row(ui, "Desktop", &desk_detail, s.desktop_connected);
                row(
                    ui,
                    "Car bridge (L2)",
                    if s.l2_label.is_empty() {
                        "Npcap capture on the ENET adapter"
                    } else {
                        s.l2_label.as_str()
                    },
                    s.l2_active,
                );
                let enet_detail = if s.enet_interface.is_empty() {
                    "No ENET adapter detected yet".into()
                } else {
                    s.enet_interface.clone()
                };
                row(ui, "ENET cable", &enet_detail, s.enet_link);
                row(
                    ui,
                    "Vehicle awake",
                    if s.vehicle_awake {
                        "Car answering / traffic seen"
                    } else {
                        "Ignition ON after ENET is plugged"
                    },
                    s.vehicle_awake,
                );

                ui.add_space(4.0);
                ui.colored_label(
                    egui::Color32::from_rgb(140, 160, 175),
                    format!(
                        "RTT {:.1} ms · loss {:.1}% · v{ver}",
                        s.rtt_ms,
                        s.loss_rate * 100.0,
                    ),
                );
            }

            if let Some(err) = &self.error {
                ui.add_space(10.0);
                ui.colored_label(
                    egui::Color32::from_rgb(224, 93, 82),
                    egui::RichText::new(err).size(14.0),
                );
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !self.starting_agent,
                            egui::Button::new("Start Client service").fill(
                                egui::Color32::from_rgb(20, 120, 80),
                            ),
                        )
                        .clicked()
                    {
                        self.start_local_agent();
                    }
                    if ui.button("Retry").clicked() {
                        self.refresh();
                    }
                });
            }

            if let Some(v) = &self.status.update_available {
                ui.add_space(10.0);
                ui.colored_label(
                    egui::Color32::from_rgb(224, 177, 74),
                    format!("Update available: v{v}"),
                );
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::new("Install update").fill(
                            egui::Color32::from_rgb(43, 125, 153),
                        ))
                        .clicked()
                    {
                        if self.error.is_some() {
                            self.open_setup_download();
                            self.connect_msg =
                                "Opened the GitHub release page — run BMW-ENET-Setup.exe on this laptop."
                                    .into();
                            self.connect_ok = true;
                        } else {
                            self.post("/api/update");
                        }
                    }
                });
            }

            ui.add_space(14.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Pairing").strong().size(15.0));
            ui.label("Must match the desktop Host. Auto-find uses the pair code on your LAN.");
            ui.add_space(6.0);
            ui.label("Pair code");
            ui.add(
                egui::TextEdit::singleline(&mut self.pair_code)
                    .desired_width(f32::INFINITY)
                    .hint_text("BMW-XXXX"),
            );
            ui.add_space(6.0);
            ui.label("Tunnel password");
            ui.horizontal(|ui| {
                let mut pw = egui::TextEdit::singleline(&mut self.password)
                    .desired_width(280.0)
                    .hint_text(if self.status.password_set {
                        "Leave blank to keep current"
                    } else {
                        "Optional — same as Host"
                    });
                if !self.show_password {
                    pw = pw.password(true);
                }
                ui.add(pw);
                if ui
                    .button(if self.show_password { "Hide" } else { "Show" })
                    .clicked()
                {
                    self.show_password = !self.show_password;
                }
            });
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Option 1 — Auto-find (recommended)").strong());
            if ui
                .add(
                    egui::Button::new("Save & auto-find desktop")
                        .fill(egui::Color32::from_rgb(20, 120, 80)),
                )
                .clicked()
            {
                if !self.pair_code.trim().is_empty() || !self.password.is_empty() {
                    let mut body = serde_json::json!({});
                    if let Some(obj) = body.as_object_mut() {
                        if !self.pair_code.trim().is_empty() {
                            obj.insert(
                                "pair_code".into(),
                                serde_json::json!(self.pair_code.trim()),
                            );
                        }
                        if !self.password.is_empty() {
                            obj.insert("password".into(), serde_json::json!(self.password));
                        }
                    }
                    self.post_json("/api/settings", body);
                    self.password.clear();
                }
                self.post("/api/discover");
            }
            if !self.connect_msg.is_empty() {
                ui.add_space(6.0);
                ui.colored_label(
                    if self.connect_ok {
                        egui::Color32::from_rgb(61, 204, 134)
                    } else {
                        egui::Color32::from_rgb(224, 93, 82)
                    },
                    &self.connect_msg,
                );
            }
        });

        if self.settings_open {
            egui::Window::new("Settings")
                .collapsible(false)
                .resizable(true)
                .default_width(440.0)
                .show(ctx, |ui| {
                    ui.heading("Car ENET adapter");
                    ui.label(
                        "This laptop must capture the USB/Ethernet cable to the car — never Windows Loopback. BMW-ENET exists only on the desktop for ISTA.",
                    );
                    ui.add_space(6.0);
                    let mut adapter = self.selected_adapter.clone();
                    egui::ComboBox::from_id_salt("enet_adapter")
                        .selected_text(if adapter.is_empty() {
                            "Auto (USB Ethernet / ENET cable)".to_string()
                        } else {
                            adapter.clone()
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut adapter, String::new(), "Auto (USB Ethernet / ENET cable)");
                            for a in &self.status.adapters {
                                let label = if a.usable {
                                    a.name.clone()
                                } else {
                                    format!("{} — not car ENET", a.name)
                                };
                                ui.add_enabled_ui(a.usable || a.name == self.selected_adapter, |ui| {
                                    ui.selectable_value(&mut adapter, a.name.clone(), label);
                                });
                            }
                        });
                    self.selected_adapter = adapter;
                    if ui.button("Save adapter").clicked() {
                        self.post_json(
                            "/api/settings",
                            serde_json::json!({ "enet_interface": self.selected_adapter }),
                        );
                    }

                    ui.add_space(12.0);
                    ui.heading("Updates");
                    ui.label(format!("Installed version: v{ver}"));
                    ui.checkbox(
                        &mut self.auto_update,
                        "Automatically install updates when idle",
                    );
                    ui.horizontal(|ui| {
                        if ui.button("Check for updates").clicked() {
                            self.post("/api/check-update");
                        }
                        if ui.button("Save").clicked() {
                            self.post_json(
                                "/api/settings",
                                serde_json::json!({ "auto_update": self.auto_update }),
                            );
                        }
                    });
                    ui.add_space(8.0);
                    if ui.button("Close").clicked() {
                        self.settings_open = false;
                    }
                });
        }
    }
}
