//! Flash-safety evaluation — never auto-modify vehicle data.

use crate::config::GatewayConfig;
use crate::stats::StatsSnapshot;
use crate::state::VehicleState;
use serde::{Deserialize, Serialize};

/// Thresholds for declaring a connection flash-safe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyThresholds {
    /// Maximum acceptable RTT p99 in milliseconds.
    pub max_rtt_p99_ms: f64,
    /// Maximum acceptable loss rate (0.0–1.0).
    pub max_loss_rate: f64,
    /// Maximum host CPU percent.
    pub max_cpu_pct: f64,
    /// Minimum samples before trusting metrics.
    pub min_rtt_samples_hint: u32,
}

impl From<&GatewayConfig> for SafetyThresholds {
    fn from(cfg: &GatewayConfig) -> Self {
        Self {
            max_rtt_p99_ms: cfg.safety_rtt_p99_ms,
            max_loss_rate: cfg.safety_max_loss_rate,
            max_cpu_pct: cfg.safety_max_cpu_pct,
            min_rtt_samples_hint: 20,
        }
    }
}

/// Result of a flash-safety check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FlashSafetyReport {
    /// True only when all checks pass.
    pub safe: bool,
    /// Human-readable failure reasons (empty when safe).
    pub reasons: Vec<String>,
    /// Echo of measured RTT p99.
    pub rtt_p99_ms: f64,
    /// Echo of measured loss rate.
    pub loss_rate: f64,
    /// Echo of CPU percent.
    pub cpu_pct: f64,
    /// Whether vehicle link is up.
    pub vehicle_link: bool,
    /// Whether vehicle appears awake.
    pub vehicle_awake: bool,
    /// Strong warning text for UI.
    pub warning: String,
}

impl Default for FlashSafetyReport {
    fn default() -> Self {
        Self {
            safe: false,
            reasons: vec!["No status yet".into()],
            rtt_p99_ms: 0.0,
            loss_rate: 0.0,
            cpu_pct: 0.0,
            vehicle_link: false,
            vehicle_awake: false,
            warning: "Waiting for gateway API".into(),
        }
    }
}

/// Evaluates whether ECU flashing should be allowed.
#[derive(Debug, Clone)]
pub struct FlashSafetyChecker {
    thresholds: SafetyThresholds,
}

impl FlashSafetyChecker {
    /// Create with explicit thresholds.
    pub fn new(thresholds: SafetyThresholds) -> Self {
        Self { thresholds }
    }

    /// Evaluate current stats + vehicle state + host CPU.
    pub fn evaluate(
        &self,
        stats: &StatsSnapshot,
        vehicle: &VehicleState,
        cpu_pct: f64,
        peer_connected: bool,
    ) -> FlashSafetyReport {
        let mut reasons = Vec::new();

        if !peer_connected {
            reasons.push("Tunnel peer is not connected".into());
        }

        let vehicle_ready = vehicle.link_up && vehicle.awake;
        if !vehicle.link_up {
            reasons.push("Vehicle ENET link is down — plug ENET into the car + laptop".into());
        } else if !vehicle.awake {
            reasons.push("Vehicle appears asleep — turn ignition ON".into());
        }

        // Until the car is online, idle Wi‑Fi keepalive spikes (3↔100ms) are normal and
        // must not dominate the flash-safety message. Quality is certified after ENET is up.
        if vehicle_ready {
            // Prefer p50 when p99 is a Wi‑Fi sleep outlier (p99 >> p50).
            let rtt_for_gate = effective_rtt_ms(stats);
            if rtt_for_gate > self.thresholds.max_rtt_p99_ms && rtt_for_gate > 0.0 {
                reasons.push(format!(
                    "RTT {:.2} ms exceeds limit {:.2} ms",
                    rtt_for_gate, self.thresholds.max_rtt_p99_ms
                ));
            }
            if stats.loss_rate > self.thresholds.max_loss_rate && stats.rx_pps + stats.tx_pps > 0.5
            {
                reasons.push(format!(
                    "Packet loss {:.4}% exceeds limit {:.4}%",
                    stats.loss_rate * 100.0,
                    self.thresholds.max_loss_rate * 100.0
                ));
            }
            if cpu_pct > self.thresholds.max_cpu_pct {
                reasons.push(format!(
                    "Host CPU {:.1}% exceeds limit {:.1}%",
                    cpu_pct, self.thresholds.max_cpu_pct
                ));
            }
            if stats.rx_packets < self.thresholds.min_rtt_samples_hint as u64 {
                reasons.push("Insufficient traffic samples to certify link quality".into());
            }
        }

        let safe = reasons.is_empty();
        let warning = if safe {
            "Connection quality is within flash-safe thresholds. Proceed only if you accept the risk of remote flashing.".into()
        } else if !vehicle_ready && peer_connected {
            format!(
                "Not ready to flash yet — {}. Tunnel quality is checked after the vehicle is online. Do not start ECU programming.",
                reasons.join("; ")
            )
        } else {
            format!(
                "FLASHING NOT RECOMMENDED: {}. Do not start ECU programming until these are resolved.",
                reasons.join("; ")
            )
        };

        FlashSafetyReport {
            safe,
            reasons,
            rtt_p99_ms: stats.rtt_p99_ms,
            loss_rate: stats.loss_rate,
            cpu_pct,
            vehicle_link: vehicle.link_up,
            vehicle_awake: vehicle.awake,
            warning,
        }
    }
}

/// When Wi‑Fi power-save creates rare ~100ms spikes, p99 stays ugly while p50 tracks the
/// real working latency (and matches continuous ping / ISTA load).
fn effective_rtt_ms(stats: &StatsSnapshot) -> f64 {
    let p50 = stats.rtt_p50_ms;
    let p99 = stats.rtt_p99_ms;
    let last = stats.rtt_ms;
    if p50 > 0.0 && p99 > p50 * 3.0 && p99 > 40.0 {
        // Spiky idle link — gate on p50/last, not the sleep outlier.
        p50.max(last.min(p50 * 2.0))
    } else {
        p99.max(last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::StatsSnapshot;

    fn good_stats() -> StatsSnapshot {
        StatsSnapshot {
            tx_packets: 1000,
            rx_packets: 1000,
            tx_bytes: 100_000,
            rx_bytes: 100_000,
            dropped: 0,
            reconnects: 0,
            errors: 0,
            seq_gaps: 0,
            tx_pps: 100.0,
            rx_pps: 100.0,
            tx_bps: 80_000.0,
            rx_bps: 80_000.0,
            rtt_ms: 1.0,
            rtt_p50_ms: 1.0,
            rtt_p99_ms: 2.0,
            loss_rate: 0.0,
        }
    }

    #[test]
    fn safe_when_healthy() {
        let checker = FlashSafetyChecker::new(SafetyThresholds {
            max_rtt_p99_ms: 20.0,
            max_loss_rate: 0.001,
            max_cpu_pct: 80.0,
            min_rtt_samples_hint: 20,
        });
        let vehicle = VehicleState {
            link_up: true,
            awake: true,
            last_activity_ms: 0,
            discovered_ip: Some("169.254.5.77".into()),
            vin: None,
        };
        let report = checker.evaluate(&good_stats(), &vehicle, 10.0, true);
        assert!(report.safe);
        assert!(report.reasons.is_empty());
    }

    #[test]
    fn unsafe_on_loss() {
        let checker = FlashSafetyChecker::new(SafetyThresholds {
            max_rtt_p99_ms: 20.0,
            max_loss_rate: 0.001,
            max_cpu_pct: 80.0,
            min_rtt_samples_hint: 20,
        });
        let mut stats = good_stats();
        stats.loss_rate = 0.05;
        let vehicle = VehicleState {
            link_up: true,
            awake: true,
            last_activity_ms: 0,
            discovered_ip: None,
            vin: None,
        };
        let report = checker.evaluate(&stats, &vehicle, 10.0, true);
        assert!(!report.safe);
        assert!(!report.reasons.is_empty());
    }

    #[test]
    fn vehicle_down_skips_rtt_noise() {
        let checker = FlashSafetyChecker::new(SafetyThresholds {
            max_rtt_p99_ms: 20.0,
            max_loss_rate: 0.001,
            max_cpu_pct: 80.0,
            min_rtt_samples_hint: 20,
        });
        let mut stats = good_stats();
        stats.rtt_p99_ms = 92.0;
        stats.rtt_ms = 3.0;
        let vehicle = VehicleState {
            link_up: false,
            awake: false,
            last_activity_ms: 0,
            discovered_ip: None,
            vin: None,
        };
        let report = checker.evaluate(&stats, &vehicle, 10.0, true);
        assert!(!report.safe);
        assert!(report.reasons.iter().all(|r| r.contains("ENET") || r.contains("asleep") || r.contains("ignition") || r.contains("plug")));
        assert!(!report.warning.contains("exceeds limit"));
        assert!(report.warning.contains("vehicle is online") || report.warning.contains("Not ready"));
    }
}
