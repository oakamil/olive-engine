// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

//! Wireless motorized telescope mount control and closed-loop astrometric GoTo.
//!
//! Runs a single blocking `std::thread` controller (`MountController`) over a
//! small synchronous `MountDriver` trait. Keeping the mount loop off Tokio
//! avoids cancelling half-duplex serial-over-WiFi TCP reads mid-flight and lets
//! `:Q#` aborts fire immediately on the already-open socket.

mod celestron_aux;
mod ioptron;
mod kinematic;
mod sky_model;
mod skywatcher;

use std::io::{Error, ErrorKind, Result as IoResult};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::{Builder, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use chrono::Utc;
use clap::ValueEnum;
use log::{debug, info, warn};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use starfield::time::Timescale;

use crate::engine::calculate_alt_az;
use crate::telescope_server::{AtomicOptionF64, precess_coordinates};
use celestron_aux::CelestronAuxMount;
use ioptron::IoptronMount;
use kinematic::KinematicMount;
use skywatcher::SkyWatcherMount;

const CENTERING_TOLERANCE_DEG: f64 = 0.05;
const HORIZON_SAFETY_MIN_ALT_DEG: f64 = 5.0;
const CALIBRATION_MIN_ALT_DEG: f64 = 15.0;
const CALIBRATION_MAX_ALT_DEG: f64 = 75.0;
const CALIBRATION_STEP_DEG: f64 = 5.0;
const CALIBRATION_POLE_NEAR_DEG: f64 = 6.0;
const CALIBRATION_POLE_MID_DEG: f64 = 9.0;
const CALIBRATION_POLE_NEAR_AX2_STEP_DEG: f64 = 9.0;
const CALIBRATION_POLE_MID_AX2_STEP_DEG: f64 = 3.0;
const CALIBRATION_MAX_AX1_STEP_DEG: f64 = 20.0;
const CALIBRATION_RETURN_MAX_DEC_DEG: f64 = 85.0;
const PASSIVE_CAL_MIN_INTERVAL: Duration = Duration::from_secs(1);
const PASSIVE_CAL_MIN_SKY_SEP_DEG: f64 = 2.0;
const POLE_SINGULARITY_SKIP_DEG: f64 = 88.0;
const FRAME_FENCE_MARGIN: Duration = Duration::from_millis(100);
const TCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const TCP_IO_TIMEOUT: Duration = Duration::from_millis(350);

/// Optional platform TCP socket factory (e.g. Android `ConnectivityManager` `TRANSPORT_WIFI` socket).
pub type TcpConnector = Arc<dyn Fn(&str, u16) -> IoResult<TcpStream> + Send + Sync>;

/// Supported wireless mount protocols.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, ValueEnum, Serialize, Deserialize)]
pub enum MountProtocol {
    /// Class A: iOptron V2.5 & V3.10 over Wi-Fi TCP (default port 8899).
    #[default]
    Ioptron,
    /// Class B: Sky-Watcher SynScan Wi-Fi Motor Controller over UDP (default port 11880).
    #[value(name = "skywatcher", alias = "synscan")]
    SkyWatcher,
    /// Class B: Celestron NexStar AUX Wi-Fi Motor Controller over TCP (default port 2000).
    #[value(name = "celestron", alias = "celestron-aux", alias = "nexstar")]
    CelestronAux,
}

impl MountProtocol {
    /// Returns the default network port for this mount protocol.
    pub fn default_port(self) -> u16 {
        match self {
            MountProtocol::Ioptron => 8899,
            MountProtocol::SkyWatcher => 11880,
            MountProtocol::CelestronAux => 2000,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Runtime configuration for motorized mount control.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MountConfig {
    pub protocol: MountProtocol,
    pub host: String,
    pub port: u16,
    /// When `true`, performs one post-slew plate solve + sync + fine re-GoTo pass.
    pub closed_loop_goto: bool,
    /// When `true` (default), enforces counterweight-down pier-side meridian flips on GoTo
    /// and a `+10°` past-meridian tracking safety halt for Class B German Equatorial mounts.
    #[serde(default = "default_true")]
    pub gem_meridian_flip: bool,
}

impl MountConfig {
    /// Convenience constructor for an iOptron Wi-Fi mount (`TCP 8899`).
    pub fn ioptron(host: impl Into<String>, port: u16) -> Self {
        Self::new(MountProtocol::Ioptron, host, port)
    }

    /// Convenience constructor for a Sky-Watcher SynScan Wi-Fi mount (`UDP 11880`).
    pub fn skywatcher(host: impl Into<String>, port: u16) -> Self {
        Self::new(MountProtocol::SkyWatcher, host, port)
    }

    /// Convenience constructor for a Celestron NexStar AUX Wi-Fi mount (`TCP 2000`).
    pub fn celestron_aux(host: impl Into<String>, port: u16) -> Self {
        Self::new(MountProtocol::CelestronAux, host, port)
    }

    /// Constructs a mount config with closed-loop GoTo and GEM meridian safety enabled.
    pub fn new(protocol: MountProtocol, host: impl Into<String>, port: u16) -> Self {
        Self {
            protocol,
            host: host.into(),
            port,
            closed_loop_goto: true,
            gem_meridian_flip: true,
        }
    }
}

/// High-level mount state published in [`crate::engine::EngineStateData`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MountUiState {
    #[default]
    Disabled,
    Connecting,
    Connected,
    NeedsCalibration,
    Calibrating,
    AtZero,
    Parked,
    Slewing,
    Centering,
    Tracking,
    TrackingNoSolve,
    Reconnecting,
}

/// Snapshot of mount telemetry exposed to the UI and engine state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MountTelemetry {
    pub state: MountUiState,
    pub model: Option<String>,
    pub notice: Option<String>,
    /// `true` when the mount is ready to accept GoTo commands (always `true` when connected
    /// for Class A mounts; `true` after the first stationary plate solve for Class B mounts).
    pub goto_ready: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MountRawStatus {
    is_slewing: bool,
    is_tracking: bool,
    is_parked: bool,
    at_zero: bool,
    gps_locked: bool,
    ra_jnow_deg: f64,
    dec_jnow_deg: f64,
    safety_notice: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StartTrackingOutcome {
    Started,
    Refused,
    SafetyHalt(&'static str),
}

impl MountRawStatus {
    fn ui_state(&self) -> MountUiState {
        if self.is_slewing {
            MountUiState::Slewing
        } else if self.is_parked {
            MountUiState::Parked
        } else if self.at_zero {
            MountUiState::AtZero
        } else if self.is_tracking {
            MountUiState::Tracking
        } else {
            MountUiState::Connected
        }
    }

    fn effective_ui_state(&self, goto_ready: bool) -> MountUiState {
        if !goto_ready && !self.is_slewing {
            MountUiState::NeedsCalibration
        } else {
            self.ui_state()
        }
    }
}

trait MountDriver: Send {
    fn model_name(&self) -> &str;
    fn is_alt_az(&self) -> bool;
    /// Returns `true` when the mount has a complete alignment/sky model.
    /// Class A mounts return `true` unconditionally; Class B mounts return `true` once fitted.
    fn is_calibrated(&self) -> bool {
        true
    }
    /// Returns `true` when the mount can accept a GoTo command.
    /// Class A mounts return `true` unconditionally; Class B mounts return `true` once at least
    /// one stationary plate solve has been recorded (auto-calibrating on the first GoTo).
    fn is_goto_ready(&self) -> bool {
        true
    }
    /// Returns `true` for Class B mounts whose continuous tracking rates must be halted on disconnect.
    fn requires_host_tracking(&self) -> bool {
        false
    }
    /// Enables or disables counterweight-down GEM meridian safety for Class B EQ mounts.
    fn set_gem_meridian_flip(&mut self, _enabled: bool) {}
    /// Clears any internal Class B sky-model samples prior to a fresh calibration run.
    fn reset_calibration(&mut self) {}
    fn poll(&mut self) -> IoResult<MountRawStatus>;
    fn goto_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<bool>;
    /// Syncs the mount to `(ra_deg, dec_deg)` JNow and verifies readback.
    /// Returns `Ok(Ok(()))` on verified convergence, or `Ok(Err(offset_deg))` if rejected/divergent.
    fn sync_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<Result<(), f64>>;
    fn start_tracking(&mut self) -> IoResult<StartTrackingOutcome>;
    fn stop_tracking(&mut self) -> IoResult<bool>;
    fn abort(&mut self) -> IoResult<()>;
    fn unpark(&mut self) -> IoResult<bool>;
    fn sync_site_time(&mut self, lat_deg: f64, lon_deg: f64, utc_ms: i64) -> IoResult<()>;
    /// Relative axis slew in radians for Class B mini-calibration.
    fn jog_axes_rad(&mut self, _d_ax1: f64, _d_ax2: f64) -> IoResult<bool> {
        Ok(false)
    }
}

fn invalid_data(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

fn connect_tcp_stream(
    host: &str,
    port: u16,
    connector: Option<&TcpConnector>,
) -> IoResult<TcpStream> {
    let stream = if let Some(connect_fn) = connector {
        connect_fn(host, port)?
    } else {
        let mut last_err = None;
        let mut connected = None;
        for addr in (host, port).to_socket_addrs()?.take(2) {
            match TcpStream::connect_timeout(&addr, TCP_CONNECT_TIMEOUT) {
                Ok(s) => {
                    connected = Some(s);
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        connected.ok_or_else(|| {
            last_err.unwrap_or_else(|| {
                Error::new(ErrorKind::AddrNotAvailable, "No socket address resolved")
            })
        })?
    };
    let _ = stream.set_nodelay(true);
    stream.set_read_timeout(Some(TCP_IO_TIMEOUT))?;
    stream.set_write_timeout(Some(TCP_IO_TIMEOUT))?;
    Ok(stream)
}

/// Exact 3D great-circle angular separation in degrees (immune to 0h/24h wrap & poles).
fn angular_separation_deg(ra1_deg: f64, dec1_deg: f64, ra2_deg: f64, dec2_deg: f64) -> f64 {
    let (r1, d1) = (ra1_deg.to_radians(), dec1_deg.to_radians());
    let (r2, d2) = (ra2_deg.to_radians(), dec2_deg.to_radians());
    let v1 = Vector3::new(d1.cos() * r1.cos(), d1.cos() * r1.sin(), d1.sin());
    let v2 = Vector3::new(d2.cos() * r2.cos(), d2.cos() * r2.sin(), d2.sin());
    v1.cross(&v2)
        .norm()
        .atan2(v1.dot(&v2).clamp(-1.0, 1.0))
        .to_degrees()
}

fn current_julian_epoch(ts: &Timescale) -> f64 {
    let t = ts.from_datetime(Utc::now());
    2000.0 + (t.tdb() - 2451545.0) / 365.25
}

fn j2000_to_jnow(ra_deg: f64, dec_deg: f64, ts: &Timescale) -> (f64, f64) {
    precess_coordinates(ra_deg, dec_deg, 2000.0, current_julian_epoch(ts), ts)
}

/// Chooses the Axis-2 calibration jog step so that regardless of Axis-2 sign, the
/// post-jog declination always lands at least 3° away from the celestial pole.
fn calibration_ax2_step_deg(dec_jnow_deg: f64) -> f64 {
    let d_pole = 90.0 - dec_jnow_deg.abs();
    if d_pole < CALIBRATION_POLE_NEAR_DEG {
        CALIBRATION_POLE_NEAR_AX2_STEP_DEG
    } else if d_pole < CALIBRATION_POLE_MID_DEG {
        CALIBRATION_POLE_MID_AX2_STEP_DEG
    } else {
        CALIBRATION_STEP_DEG
    }
}

/// Scales the Axis-1 calibration jog step by `1 / cos(dec)` (capped at 20°) so EQ mounts
/// near a celestial pole still produce a clean sky arc after stepping off the pole.
fn calibration_ax1_step_deg(dec_jnow_deg: f64) -> f64 {
    let cos_dec = dec_jnow_deg.to_radians().cos().abs().max(0.25);
    (CALIBRATION_STEP_DEG / cos_dec).clamp(CALIBRATION_STEP_DEG, CALIBRATION_MAX_AX1_STEP_DEG)
}

enum MountCmd {
    Configure(Option<MountConfig>),
    Reconnect,
    GoTo {
        ra_j2000: f64,
        dec_j2000: f64,
        seq: u64,
    },
    Calibrate {
        seq: u64,
    },
    SetTracking {
        enabled: bool,
        seq: u64,
    },
    Wake,
    Solved {
        ra_j2000: f64,
        dec_j2000: f64,
        epoch: u64,
    },
    TargetPixelChanged,
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlewPass {
    Coarse,
    Fine,
}

#[derive(Clone, Copy, Debug)]
enum CalStep {
    WaitInitialSolve {
        epoch: u64,
        deadline: Instant,
    },
    SlewAx1 {
        started_at: Instant,
        slew_seen: bool,
    },
    WaitAx1Solve {
        epoch: u64,
        deadline: Instant,
    },
    SlewAx2 {
        started_at: Instant,
        slew_seen: bool,
    },
    WaitAx2Solve {
        epoch: u64,
        deadline: Instant,
    },
}

#[derive(Clone, Copy, Debug)]
enum CalPostAction {
    TrackInPlace,
    ReturnToInitialAndTrack,
    GoTo(f64, f64),
}

impl CalPostAction {
    fn is_goto_active(self) -> bool {
        !matches!(self, CalPostAction::TrackInPlace)
    }
}

enum ControllerState {
    Disconnected,
    Idle,
    Calibrating {
        step: CalStep,
        post_action: CalPostAction,
    },
    Slewing {
        target_j2000: (f64, f64),
        pass: SlewPass,
        started_at: Instant,
        slew_seen: bool,
    },
    WaitingPostStopSolve {
        target_j2000: (f64, f64),
        epoch: u64,
        deadline: Instant,
    },
}

#[derive(Clone, Copy, Debug)]
struct ControllerTimings {
    poll_interval_slewing: Duration,
    poll_interval_idle: Duration,
    coarse_start_grace: Duration,
    fine_start_grace: Duration,
    reconnect_backoff: Duration,
    min_solve_timeout: Duration,
    max_solve_timeout: Duration,
}

impl Default for ControllerTimings {
    fn default() -> Self {
        Self {
            poll_interval_slewing: Duration::from_millis(200),
            poll_interval_idle: Duration::from_millis(500),
            coarse_start_grace: Duration::from_millis(400),
            fine_start_grace: Duration::from_millis(400),
            reconnect_backoff: Duration::from_secs(2),
            min_solve_timeout: Duration::from_secs(8),
            max_solve_timeout: Duration::from_secs(20),
        }
    }
}

type DriverFactory = Box<
    dyn Fn(&MountConfig, Option<&TcpConnector>, &Arc<Timescale>) -> IoResult<Box<dyn MountDriver>>
        + Send,
>;

fn default_driver_factory(
    cfg: &MountConfig,
    connector: Option<&TcpConnector>,
    timescale: &Arc<Timescale>,
) -> IoResult<Box<dyn MountDriver>> {
    match cfg.protocol {
        MountProtocol::Ioptron => Ok(Box::new(IoptronMount::connect(
            &cfg.host, cfg.port, connector,
        )?)),
        MountProtocol::SkyWatcher => {
            let raw = SkyWatcherMount::connect(&cfg.host, cfg.port)?;
            Ok(Box::new(KinematicMount::new(
                raw,
                timescale.clone(),
                cfg.gem_meridian_flip,
            )))
        }
        MountProtocol::CelestronAux => {
            let raw = CelestronAuxMount::connect(&cfg.host, cfg.port, connector)?;
            Ok(Box::new(KinematicMount::new(
                raw,
                timescale.clone(),
                cfg.gem_meridian_flip,
            )))
        }
    }
}

#[derive(Default)]
struct FenceState {
    physically_slewing: bool,
    slew_epoch: u64,
    last_slew_stop: Option<SystemTime>,
}

/// Thread-safe handle shared across `Engine` (`t1`, `t2`, `polling_task`) and external callers/FFI.
pub struct MountHandle {
    cmd_tx: Sender<MountCmd>,
    enabled: AtomicBool,
    /// Monotonic command sequence bumped on GoTo, Abort, disable, and shutdown.
    /// Kept separate from `FenceState.slew_epoch` so polls/reconnects never drop valid GoTos.
    cmd_seq: AtomicU64,
    abort_flag: AtomicBool,
    goto_active: AtomicBool,
    wants_solve: AtomicBool,
    fence: Mutex<FenceState>,
    telemetry: Mutex<(MountTelemetry, u64)>,
    site_lat: AtomicOptionF64,
    site_lon: AtomicOptionF64,
    expected_solve_window_ms: AtomicU64,
    connector: Mutex<Option<TcpConnector>>,
}

impl MountHandle {
    /// Spawns the background `MountController` thread (initially disabled until [`Self::configure`]).
    pub fn start(timescale: Arc<Timescale>, connector: Option<TcpConnector>) -> Arc<Self> {
        let (handle, _join) = Self::start_internal(
            timescale,
            connector,
            Box::new(default_driver_factory),
            ControllerTimings::default(),
        );
        handle
    }

    fn start_internal(
        timescale: Arc<Timescale>,
        connector: Option<TcpConnector>,
        factory: DriverFactory,
        timings: ControllerTimings,
    ) -> (Arc<Self>, JoinHandle<()>) {
        let (cmd_tx, cmd_rx) = channel();
        let handle = Arc::new(Self {
            cmd_tx,
            enabled: AtomicBool::new(false),
            cmd_seq: AtomicU64::new(0),
            abort_flag: AtomicBool::new(false),
            goto_active: AtomicBool::new(false),
            wants_solve: AtomicBool::new(false),
            fence: Mutex::new(FenceState::default()),
            telemetry: Mutex::new((MountTelemetry::default(), 0)),
            site_lat: AtomicOptionF64::new(None),
            site_lon: AtomicOptionF64::new(None),
            expected_solve_window_ms: AtomicU64::new(2_000),
            connector: Mutex::new(connector),
        });

        let controller_handle = handle.clone();
        let join = Builder::new()
            .name("olive-mount".to_string())
            .spawn(move || {
                let mut controller =
                    MountController::new(controller_handle, cmd_rx, timescale, factory, timings);
                controller.run();
            })
            .expect("Failed to spawn olive-mount thread");

        (handle, join)
    }

    /// Installs or clears an optional platform TCP socket connector (e.g. Android Wi-Fi socket).
    pub fn set_tcp_connector(&self, connector: Option<TcpConnector>) {
        *self.connector.lock().unwrap() = connector;
        if self.enabled.load(Ordering::Acquire) {
            let _ = self.cmd_tx.send(MountCmd::Reconnect);
        }
    }

    /// Enables/updates motorized mount control, or disables and disconnects when `None`.
    pub fn configure(&self, config: Option<MountConfig>) {
        if config.is_some() {
            self.enabled.store(true, Ordering::SeqCst);
        } else {
            self.reset_disabled_state();
        }
        let _ = self.cmd_tx.send(MountCmd::Configure(config));
    }

    /// Returns `true` when motorized mount control is configured and enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Returns `true` strictly while the mount's physical motors are slewing.
    pub fn is_physically_slewing(&self) -> bool {
        self.enabled.load(Ordering::Acquire) && self.fence.lock().unwrap().physically_slewing
    }

    /// Returns `true` across the entire motorized GoTo sequence (Coarse -> Centering -> Fine).
    pub fn is_goto_active(&self) -> bool {
        self.enabled.load(Ordering::Acquire) && self.goto_active.load(Ordering::Acquire)
    }

    /// Returns `true` while an abort request is queued and awaiting controller processing.
    pub fn is_abort_pending(&self) -> bool {
        self.abort_flag.load(Ordering::Acquire)
    }

    /// Atomically evaluates the `t1` frame fence, returning `(should_drop, slew_epoch)`.
    ///
    /// Drops frames while motors are physically slewing or when the frame's estimated
    /// exposure start (`img_timestamp - exposure/2 - 100ms`) precedes `last_slew_stop`.
    pub fn frame_fence(&self, img_timestamp: SystemTime, exposure_us: i32) -> (bool, u64) {
        if !self.enabled.load(Ordering::Acquire) {
            return (false, 0);
        }
        let f = self.fence.lock().unwrap();
        if f.physically_slewing {
            return (true, f.slew_epoch);
        }
        if let Some(t_stop) = f.last_slew_stop {
            let half_exp = Duration::from_micros((exposure_us.max(0) as u64) / 2);
            let lookback = half_exp + FRAME_FENCE_MARGIN;
            let overlapped = img_timestamp
                .checked_sub(lookback)
                .is_none_or(|t_start| t_start < t_stop);
            if overlapped {
                return (true, f.slew_epoch);
            }
        }
        (false, f.slew_epoch)
    }

    /// Initiates a motorized GoTo to J2000 `(ra_j2000, dec_j2000)` in degrees.
    pub fn goto(&self, ra_j2000: f64, dec_j2000: f64) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        let seq = self.cmd_seq.fetch_add(1, Ordering::SeqCst) + 1;
        self.mark_slew_started();
        self.goto_active.store(true, Ordering::Release);
        let _ = self.cmd_tx.send(MountCmd::GoTo {
            ra_j2000,
            dec_j2000,
            seq,
        });
    }

    /// Initiates a safe 2-step mini-calibration (Axis 2 first, then adaptive Axis 1) for Class B mounts.
    pub fn calibrate(&self) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        let seq = self.cmd_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.cmd_tx.send(MountCmd::Calibrate { seq });
    }

    /// Starts or stops sidereal tracking on the mount.
    ///
    /// For an uncalibrated Class B mount with at least one plate solve, enabling tracking
    /// automatically runs the 2-step alignment probe and returns to track the aimed position.
    pub fn set_tracking(&self, enabled: bool) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        let seq = self.cmd_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.cmd_tx.send(MountCmd::SetTracking { enabled, seq });
    }

    /// Requests immediate slew abort (`:Q#`). Safe to call from any thread (UI, FFI, `polling_task`).
    pub fn abort(&self) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        self.cmd_seq.fetch_add(1, Ordering::SeqCst);
        self.goto_active.store(false, Ordering::Release);
        self.abort_flag.store(true, Ordering::SeqCst);
        let _ = self.cmd_tx.send(MountCmd::Wake);
    }

    /// Aborts only if the mount is enabled, actively slewing or running a GoTo, and no abort is pending.
    pub fn abort_if_active(&self) {
        if self.is_enabled()
            && !self.is_abort_pending()
            && (self.is_goto_active() || self.is_physically_slewing())
        {
            self.abort();
        }
    }

    /// Delivers a verified `t2` plate solve (`SolveStatus::MatchFound`) with its `slew_epoch`.
    /// Fast-paths out without queuing when the controller does not currently need a solve.
    pub fn on_solved(&self, ra_j2000: f64, dec_j2000: f64, epoch: u64) {
        if !self.enabled.load(Ordering::Acquire) || !self.wants_solve.load(Ordering::Acquire) {
            return;
        }
        let _ = self.cmd_tx.send(MountCmd::Solved {
            ra_j2000,
            dec_j2000,
            epoch,
        });
    }

    /// Re-arms the background sync latch after `target_pixel` eyepiece alignment changes.
    pub fn notify_target_pixel_changed(&self) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        let _ = self.cmd_tx.send(MountCmd::TargetPixelChanged);
    }

    /// Updates observer latitude/longitude (degrees, East positive) used for horizon floor and GPS-unlocked seeding.
    pub fn set_site_location(&self, lat: Option<f64>, lon: Option<f64>) {
        if lat.is_some() {
            self.site_lat.store(lat, Ordering::Relaxed);
        }
        if lon.is_some() {
            self.site_lon.store(lon, Ordering::Relaxed);
        }
    }

    /// Updates the expected single-solve integration window (ms) so centering timeout adapts to exposure/stacking.
    pub fn set_expected_solve_window_ms(&self, ms: u64) {
        self.expected_solve_window_ms.store(ms, Ordering::Relaxed);
    }

    /// Atomically returns a snapshot of mount UI telemetry and its monotonic revision number.
    pub fn telemetry_with_seq(&self) -> (MountTelemetry, u64) {
        let guard = self.telemetry.lock().unwrap();
        (guard.0.clone(), guard.1)
    }

    /// Signals the background mount thread to abort any active slew and exit without blocking the caller.
    pub fn shutdown(&self) {
        self.reset_disabled_state();
        let _ = self.cmd_tx.send(MountCmd::Shutdown);
    }

    fn reset_disabled_state(&self) {
        self.enabled.store(false, Ordering::SeqCst);
        self.cmd_seq.fetch_add(1, Ordering::SeqCst);
        self.goto_active.store(false, Ordering::SeqCst);
        self.wants_solve.store(false, Ordering::SeqCst);
        self.abort_flag.store(false, Ordering::SeqCst);
        self.clear_slewing(false);
        self.reset_telemetry_disabled();
    }

    fn slew_epoch(&self) -> u64 {
        self.fence.lock().unwrap().slew_epoch
    }

    fn telemetry(&self) -> MountTelemetry {
        self.telemetry.lock().unwrap().0.clone()
    }

    fn site_location(&self) -> Option<(f64, f64)> {
        Some((
            self.site_lat.load(Ordering::Relaxed)?,
            self.site_lon.load(Ordering::Relaxed)?,
        ))
    }

    fn reset_telemetry_disabled(&self) {
        let mut guard = self.telemetry.lock().unwrap();
        guard.0 = MountTelemetry::default();
        guard.1 += 1;
    }

    fn set_connected_model(&self, model: Option<String>, goto_ready: bool) {
        let mut guard = self.telemetry.lock().unwrap();
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        guard.0.model = model;
        guard.0.goto_ready = goto_ready;
        guard.1 += 1;
    }

    fn set_goto_ready(&self, goto_ready: bool) {
        let mut guard = self.telemetry.lock().unwrap();
        if !self.enabled.load(Ordering::Acquire) || guard.0.goto_ready == goto_ready {
            return;
        }
        guard.0.goto_ready = goto_ready;
        guard.1 += 1;
    }

    fn set_state_and_notice(&self, state: MountUiState, notice: Option<String>) {
        let mut guard = self.telemetry.lock().unwrap();
        if state != MountUiState::Disabled && !self.enabled.load(Ordering::Acquire) {
            return;
        }
        if guard.0.state == state && guard.0.notice == notice {
            return;
        }
        guard.0.state = state;
        guard.0.notice = notice;
        guard.1 += 1;
    }

    fn mark_slew_started(&self) -> u64 {
        let mut f = self.fence.lock().unwrap();
        f.slew_epoch += 1;
        f.physically_slewing = true;
        f.slew_epoch
    }

    fn clear_slewing(&self, record_stop: bool) -> u64 {
        let mut f = self.fence.lock().unwrap();
        if record_stop {
            f.last_slew_stop = Some(SystemTime::now());
        }
        f.slew_epoch += 1;
        f.physically_slewing = false;
        f.slew_epoch
    }
}

struct MountController {
    handle: Arc<MountHandle>,
    cmd_rx: Receiver<MountCmd>,
    timescale: Arc<Timescale>,
    factory: DriverFactory,
    timings: ControllerTimings,

    config: Option<MountConfig>,
    driver: Option<Box<dyn MountDriver>>,
    state: ControllerState,
    last_status: Option<MountRawStatus>,
    hc_slewing: bool,
    site_synced: bool,
    needs_bg_sync: bool,
    sync_rejected: bool,
    tracking_paused: bool,
    last_passive_sample: Option<(Instant, f64, f64)>,
    next_poll_at: Instant,
    next_reconnect_at: Option<Instant>,
}

impl MountController {
    fn new(
        handle: Arc<MountHandle>,
        cmd_rx: Receiver<MountCmd>,
        timescale: Arc<Timescale>,
        factory: DriverFactory,
        timings: ControllerTimings,
    ) -> Self {
        Self {
            handle,
            cmd_rx,
            timescale,
            factory,
            timings,
            config: None,
            driver: None,
            state: ControllerState::Disconnected,
            last_status: None,
            hc_slewing: false,
            site_synced: false,
            needs_bg_sync: false,
            sync_rejected: false,
            tracking_paused: false,
            last_passive_sample: None,
            next_poll_at: Instant::now(),
            next_reconnect_at: None,
        }
    }

    fn is_motor_slewing(&self) -> bool {
        matches!(
            self.state,
            ControllerState::Slewing { .. }
                | ControllerState::Calibrating {
                    step: CalStep::SlewAx1 { .. } | CalStep::SlewAx2 { .. },
                    ..
                }
        ) || self.hc_slewing
            || self.last_status.is_some_and(|s| s.is_slewing)
    }

    fn is_actively_moving(&self) -> bool {
        matches!(
            self.state,
            ControllerState::Slewing { .. }
                | ControllerState::WaitingPostStopSolve { .. }
                | ControllerState::Calibrating { .. }
        ) || self.hc_slewing
            || self.last_status.is_some_and(|s| s.is_slewing)
    }

    fn run(&mut self) {
        loop {
            if self.handle.abort_flag.swap(false, Ordering::SeqCst) {
                self.handle_abort();
            }

            let wait = self.next_wait_duration();
            match self.cmd_rx.recv_timeout(wait) {
                Ok(cmd) => {
                    if !self.process_command(cmd) {
                        return;
                    }
                    while let Ok(next) = self.cmd_rx.try_recv() {
                        if !self.process_command(next) {
                            return;
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }

            if self.handle.abort_flag.swap(false, Ordering::SeqCst) {
                self.handle_abort();
            }

            self.tick();
        }
    }

    fn next_wait_duration(&self) -> Duration {
        if self.config.is_none() {
            return Duration::from_secs(60);
        }
        let now = Instant::now();
        match self.state {
            ControllerState::Disconnected => self
                .next_reconnect_at
                .map_or(Duration::ZERO, |at| at.saturating_duration_since(now)),
            ControllerState::WaitingPostStopSolve { deadline, .. }
            | ControllerState::Calibrating {
                step:
                    CalStep::WaitInitialSolve { deadline, .. }
                    | CalStep::WaitAx1Solve { deadline, .. }
                    | CalStep::WaitAx2Solve { deadline, .. },
                ..
            } => self
                .next_poll_at
                .min(deadline)
                .saturating_duration_since(now),
            ControllerState::Slewing { .. }
            | ControllerState::Calibrating {
                step: CalStep::SlewAx1 { .. } | CalStep::SlewAx2 { .. },
                ..
            }
            | ControllerState::Idle => self.next_poll_at.saturating_duration_since(now),
        }
    }

    fn process_command(&mut self, cmd: MountCmd) -> bool {
        match cmd {
            MountCmd::Shutdown => {
                self.disconnect(false);
                return false;
            }
            MountCmd::Configure(new_cfg) => self.handle_configure(new_cfg),
            MountCmd::Reconnect => {
                if self.config.is_some() {
                    self.disconnect(false);
                    self.next_reconnect_at = Some(Instant::now());
                    self.try_connect();
                }
            }
            MountCmd::GoTo {
                ra_j2000,
                dec_j2000,
                seq,
            } => self.handle_goto(ra_j2000, dec_j2000, seq),
            MountCmd::Calibrate { seq } => self.handle_calibrate(seq),
            MountCmd::SetTracking { enabled, seq } => self.handle_set_tracking(enabled, seq),
            MountCmd::Wake => {
                if self.handle.abort_flag.swap(false, Ordering::SeqCst) {
                    self.handle_abort();
                }
            }
            MountCmd::Solved {
                ra_j2000,
                dec_j2000,
                epoch,
            } => self.handle_solved(ra_j2000, dec_j2000, epoch),
            MountCmd::TargetPixelChanged => {
                self.needs_bg_sync = true;
                self.sync_rejected = false;
                self.update_wants_solve();
            }
        }
        true
    }

    fn handle_configure(&mut self, new_cfg: Option<MountConfig>) {
        match (&self.config, &new_cfg) {
            (Some(old), Some(new))
                if old.protocol == new.protocol
                    && old.host == new.host
                    && old.port == new.port
                    && self.driver.is_some() =>
            {
                if let Some(driver) = self.driver.as_mut() {
                    driver.set_gem_meridian_flip(new.gem_meridian_flip);
                }
                self.config = new_cfg;
                return;
            }
            _ => {}
        }

        self.disconnect(false);
        self.config = new_cfg;
        if self.config.is_some() {
            self.handle.set_connected_model(None, false);
            self.handle
                .set_state_and_notice(MountUiState::Connecting, None);
            self.next_reconnect_at = Some(Instant::now());
            self.try_connect();
        } else {
            self.handle.reset_telemetry_disabled();
        }
    }

    fn tick(&mut self) {
        if self.config.is_none() {
            return;
        }

        if self.driver.is_none() {
            if self.next_reconnect_at.is_none_or(|at| Instant::now() >= at) {
                self.try_connect();
            }
            return;
        }

        let now = Instant::now();
        match self.state {
            ControllerState::WaitingPostStopSolve { deadline, .. } if now >= deadline => {
                warn!("Mount centering timed out waiting for plate solve; tracking open-loop");
                self.finish_goto_tracking(
                    MountUiState::TrackingNoSolve,
                    Some("Centering timed out (no plate solve) — tracking open-loop".to_string()),
                );
                return;
            }
            ControllerState::Calibrating {
                step:
                    CalStep::WaitInitialSolve { deadline, .. }
                    | CalStep::WaitAx1Solve { deadline, .. }
                    | CalStep::WaitAx2Solve { deadline, .. },
                ..
            } if now >= deadline => {
                warn!("Mount calibration timed out waiting for plate solve");
                let ready = self.driver.as_ref().is_some_and(|d| d.is_goto_ready());
                self.handle.set_goto_ready(ready);
                let ui_state = self.ready_idle_ui_state();
                self.finish_goto_idle(
                    true,
                    false,
                    ui_state,
                    Some("Calibration timed out waiting for plate solve".to_string()),
                );
                return;
            }
            _ => {}
        }

        if now < self.next_poll_at {
            return;
        }
        let interval = match self.state {
            ControllerState::Slewing { .. }
            | ControllerState::WaitingPostStopSolve { .. }
            | ControllerState::Calibrating { .. } => self.timings.poll_interval_slewing,
            ControllerState::Idle | ControllerState::Disconnected => {
                self.timings.poll_interval_idle
            }
        };
        self.next_poll_at = now + interval;

        let Some(driver) = self.driver.as_mut() else {
            return;
        };
        let cmd_seq_before_poll = self.handle.cmd_seq.load(Ordering::Acquire);
        let poll_res = driver.poll();

        if self.handle.abort_flag.swap(false, Ordering::SeqCst) {
            self.handle_abort();
            return;
        }

        let raw = match poll_res {
            Ok(s) => s,
            Err(e) => {
                warn!("Mount poll error: {}", e);
                self.disconnect(true);
                return;
            }
        };
        self.last_status = Some(raw);

        if let Some(msg) = raw.safety_notice {
            if matches!(self.state, ControllerState::Idle) {
                self.tracking_paused = true;
                self.update_wants_solve();
                let goto_ready = self.driver.as_ref().is_some_and(|d| d.is_goto_ready());
                let ui_state = raw.effective_ui_state(goto_ready);
                self.handle
                    .set_state_and_notice(ui_state, Some(msg.to_string()));
            }
        }

        if let Err(e) = self.sync_site_if_needed(raw.gps_locked) {
            warn!("Mount site/time sync error: {}", e);
            self.disconnect(true);
            return;
        }

        // If a new GoTo/Calibrate/Abort was queued while `driver.poll()` blocked on I/O,
        // defer state transitions so the queued command is handled cleanly next loop.
        if self.handle.cmd_seq.load(Ordering::Acquire) != cmd_seq_before_poll {
            return;
        }

        match self.state {
            ControllerState::Calibrating {
                step: ref mut step @ (CalStep::SlewAx1 { .. } | CalStep::SlewAx2 { .. }),
                post_action,
            } => {
                let (is_ax1, started_at, slew_seen) = match step {
                    CalStep::SlewAx1 {
                        started_at,
                        slew_seen,
                    } => (true, *started_at, slew_seen),
                    CalStep::SlewAx2 {
                        started_at,
                        slew_seen,
                    } => (false, *started_at, slew_seen),
                    _ => unreachable!(),
                };
                if raw.is_slewing {
                    *slew_seen = true;
                } else if *slew_seen || started_at.elapsed() >= self.timings.coarse_start_grace {
                    let make_wait: fn(u64, Instant) -> CalStep = if is_ax1 {
                        |epoch, deadline| CalStep::WaitAx1Solve { epoch, deadline }
                    } else {
                        |epoch, deadline| CalStep::WaitAx2Solve { epoch, deadline }
                    };
                    self.advance_cal_slew_to_wait(post_action, make_wait);
                }
            }
            ControllerState::Calibrating { .. } => {
                if raw.is_slewing {
                    self.handle_external_slew_interrupt(
                        "Manual/external slew detected during calibration; cancelling calibration",
                    );
                }
            }
            ControllerState::Slewing {
                target_j2000,
                pass,
                started_at,
                ref mut slew_seen,
            } => {
                if raw.is_slewing {
                    *slew_seen = true;
                    return;
                }
                let grace = match pass {
                    SlewPass::Coarse => self.timings.coarse_start_grace,
                    SlewPass::Fine => self.timings.fine_start_grace,
                };
                if *slew_seen || started_at.elapsed() >= grace {
                    let post_stop_epoch = self.handle.clear_slewing(true);
                    let closed_loop = self.config.as_ref().is_some_and(|c| c.closed_loop_goto);

                    if pass == SlewPass::Coarse && closed_loop {
                        let timeout = self.compute_solve_timeout();
                        debug!(
                            "Coarse slew stopped (epoch={}); waiting up to {:?} for post-stop solve",
                            post_stop_epoch, timeout
                        );
                        self.state = ControllerState::WaitingPostStopSolve {
                            target_j2000,
                            epoch: post_stop_epoch,
                            deadline: Instant::now() + timeout,
                        };
                        self.update_wants_solve();
                        self.handle
                            .set_state_and_notice(MountUiState::Centering, None);
                    } else if let Some(msg) = raw.safety_notice {
                        self.tracking_paused = true;
                        let fallback = self.ready_idle_ui_state();
                        self.finish_goto_idle(false, false, fallback, Some(msg.to_string()));
                    } else {
                        self.finish_goto_tracking(MountUiState::Tracking, None);
                    }
                }
            }
            ControllerState::WaitingPostStopSolve { .. } => {
                if raw.is_slewing {
                    self.handle_external_slew_interrupt(
                        "Manual/external slew detected during centering; cancelling centering",
                    );
                }
            }
            ControllerState::Idle => {
                let goto_ready = self.driver.as_ref().is_some_and(|d| d.is_goto_ready());
                self.handle.set_goto_ready(goto_ready);

                let ui_state = raw.effective_ui_state(goto_ready);
                if raw.is_slewing && !self.hc_slewing {
                    self.hc_slewing = true;
                    self.handle.mark_slew_started();
                } else if !raw.is_slewing && self.hc_slewing {
                    self.hc_slewing = false;
                    self.handle.clear_slewing(true);
                }

                let prev = self.handle.telemetry();
                let leaving_no_solve = prev.state == MountUiState::TrackingNoSolve
                    && ui_state != MountUiState::Tracking;
                let notice = if raw.is_slewing || leaving_no_solve {
                    None
                } else {
                    prev.notice
                };
                if prev.state != MountUiState::TrackingNoSolve || ui_state != MountUiState::Tracking
                {
                    self.handle.set_state_and_notice(ui_state, notice);
                }
            }
            ControllerState::Disconnected => {}
        }
    }

    fn try_connect(&mut self) {
        let Some(cfg) = self.config.clone() else {
            return;
        };
        let connector = self.handle.connector.lock().unwrap().clone();
        match (self.factory)(&cfg, connector.as_ref(), &self.timescale) {
            Ok(mut driver) => {
                let model = driver.model_name().to_string();
                let goto_ready = driver.is_goto_ready();
                let init_res = (|| -> IoResult<(MountRawStatus, bool)> {
                    let mut status = driver.poll()?;
                    if status.is_parked {
                        info!("Mount {} is parked; sending unpark", model);
                        if driver.unpark()? {
                            status = driver.poll()?;
                        }
                    }
                    let mut site_synced = status.gps_locked;
                    if !status.gps_locked {
                        if let Some((lat, lon)) = self.handle.site_location() {
                            driver.sync_site_time(lat, lon, Utc::now().timestamp_millis())?;
                            site_synced = true;
                        }
                    }
                    Ok((status, site_synced))
                })();

                match init_res {
                    Ok((status, site_synced)) => {
                        let ui_state = status.effective_ui_state(goto_ready);
                        self.driver = Some(driver);
                        self.last_status = Some(status);
                        self.hc_slewing = status.is_slewing;
                        self.site_synced = site_synced;
                        self.needs_bg_sync = true;
                        self.sync_rejected = false;
                        self.tracking_paused = false;
                        self.state = ControllerState::Idle;
                        self.next_reconnect_at = None;
                        self.next_poll_at = Instant::now() + self.timings.poll_interval_idle;
                        // Discard any stale abort requested while disconnected/connecting.
                        self.handle.abort_flag.store(false, Ordering::SeqCst);
                        self.update_wants_solve();
                        self.handle.set_connected_model(Some(model), goto_ready);
                        self.handle.set_state_and_notice(ui_state, None);
                    }
                    Err(e) => {
                        warn!("Mount initialization failed: {}", e);
                        self.schedule_reconnect(format!("Mount connection error: {}", e));
                    }
                }
            }
            Err(e) => {
                debug!("Mount connect to {}:{} failed: {}", cfg.host, cfg.port, e);
                self.schedule_reconnect(format!("Connecting to {}:{}...", cfg.host, cfg.port));
            }
        }
    }

    fn sync_site_if_needed(&mut self, gps_locked: bool) -> IoResult<()> {
        if self.site_synced || gps_locked {
            self.site_synced = true;
            return Ok(());
        }
        if let (Some(driver), Some((lat, lon))) =
            (self.driver.as_mut(), self.handle.site_location())
        {
            driver.sync_site_time(lat, lon, Utc::now().timestamp_millis())?;
            self.site_synced = true;
        }
        Ok(())
    }

    fn to_jnow(&self, ra_j2000: f64, dec_j2000: f64) -> (f64, f64) {
        j2000_to_jnow(ra_j2000, dec_j2000, &self.timescale)
    }

    fn alt_az_now(&self, ra_jnow: f64, dec_jnow: f64) -> Option<(f64, f64)> {
        let (lat, lon) = self.handle.site_location()?;
        Some(calculate_alt_az(
            ra_jnow,
            dec_jnow,
            lat,
            lon,
            &Utc::now(),
            &self.timescale,
        ))
    }

    fn finish_goto_idle(
        &mut self,
        clear_fence: bool,
        record_stop: bool,
        ui_state: MountUiState,
        notice: Option<String>,
    ) {
        self.handle.goto_active.store(false, Ordering::Release);
        if clear_fence {
            self.handle.clear_slewing(record_stop);
        }
        self.hc_slewing = false;
        self.state = ControllerState::Idle;
        self.update_wants_solve();
        self.handle.set_state_and_notice(ui_state, notice);
    }

    fn finish_goto_tracking(&mut self, ui_state: MountUiState, notice: Option<String>) {
        let clear_fence = self.handle.is_physically_slewing();
        match self.ensure_tracking() {
            Ok(StartTrackingOutcome::Started) => {
                self.finish_goto_idle(clear_fence, false, ui_state, notice);
            }
            Ok(StartTrackingOutcome::Refused) => {
                let fallback = self.ready_idle_ui_state();
                self.finish_goto_idle(
                    clear_fence,
                    false,
                    fallback,
                    Some("Mount refused to start tracking".to_string()),
                );
            }
            Ok(StartTrackingOutcome::SafetyHalt(msg)) => {
                let fallback = self.ready_idle_ui_state();
                self.finish_goto_idle(clear_fence, false, fallback, Some(msg.to_string()));
            }
            Err(e) => {
                warn!("Start tracking after GoTo failed: {}", e);
                self.disconnect(true);
            }
        }
    }

    fn ensure_connected_or_notify(&mut self) -> bool {
        if self.driver.is_some() {
            return true;
        }
        let cur = self.handle.telemetry().state;
        let state = if cur == MountUiState::Connecting {
            MountUiState::Connecting
        } else {
            MountUiState::Reconnecting
        };
        self.handle.goto_active.store(false, Ordering::Release);
        self.handle.clear_slewing(false);
        self.handle
            .set_state_and_notice(state, Some("Mount not connected".to_string()));
        false
    }

    fn unpark_if_needed(&mut self, aborted_prior_motion: bool) -> bool {
        if !self.last_status.is_some_and(|s| s.is_parked) {
            return true;
        }
        let Some(driver) = self.driver.as_mut() else {
            return false;
        };
        match driver.unpark() {
            Ok(true) => {
                if let Some(st) = self.last_status.as_mut() {
                    st.is_parked = false;
                }
                true
            }
            Ok(false) => {
                self.finish_goto_idle(
                    true,
                    aborted_prior_motion,
                    MountUiState::Parked,
                    Some("Mount refused to unpark".to_string()),
                );
                false
            }
            Err(e) => {
                warn!("Mount unpark failed: {}", e);
                self.disconnect(true);
                false
            }
        }
    }

    fn ready_idle_ui_state(&self) -> MountUiState {
        if self.driver.as_ref().is_some_and(|d| d.is_goto_ready()) {
            MountUiState::Connected
        } else {
            MountUiState::NeedsCalibration
        }
    }

    fn handle_external_slew_interrupt(&mut self, reason: &str) {
        info!("{}", reason);
        self.hc_slewing = true;
        self.handle.goto_active.store(false, Ordering::Release);
        self.handle.mark_slew_started();
        self.state = ControllerState::Idle;
        self.update_wants_solve();
        self.handle
            .set_state_and_notice(MountUiState::Slewing, None);
    }

    fn refresh_cal_wait_fence(&mut self, new_deadline: Option<Instant>) {
        if self.is_motor_slewing() {
            return;
        }
        let new_epoch = self.handle.clear_slewing(false);
        if let ControllerState::Calibrating { ref mut step, .. } = self.state {
            match step {
                CalStep::WaitInitialSolve { epoch, deadline }
                | CalStep::WaitAx1Solve { epoch, deadline }
                | CalStep::WaitAx2Solve { epoch, deadline } => {
                    *epoch = new_epoch;
                    if let Some(d) = new_deadline {
                        *deadline = d;
                    }
                }
                CalStep::SlewAx1 { .. } | CalStep::SlewAx2 { .. } => {}
            }
        }
    }

    fn reject_goto_preserving_active_motion(&mut self, idle_state: MountUiState, notice: String) {
        match self.state {
            ControllerState::Calibrating { post_action, .. } => {
                self.handle
                    .goto_active
                    .store(post_action.is_goto_active(), Ordering::Release);
                self.refresh_cal_wait_fence(None);
                self.handle
                    .set_state_and_notice(MountUiState::Calibrating, Some(notice));
            }
            ControllerState::Slewing { .. } => {
                self.handle.goto_active.store(true, Ordering::Release);
                self.handle
                    .set_state_and_notice(MountUiState::Slewing, Some(notice));
            }
            ControllerState::WaitingPostStopSolve { ref mut epoch, .. } => {
                *epoch = self.handle.clear_slewing(false);
                self.handle.goto_active.store(true, Ordering::Release);
                self.handle
                    .set_state_and_notice(MountUiState::Centering, Some(notice));
            }
            ControllerState::Idle if self.is_motor_slewing() => {
                self.handle.goto_active.store(false, Ordering::Release);
                self.handle
                    .set_state_and_notice(MountUiState::Slewing, Some(notice));
            }
            ControllerState::Idle | ControllerState::Disconnected => {
                self.finish_goto_idle(true, false, idle_state, Some(notice));
            }
        }
    }

    fn start_coarse_goto(
        &mut self,
        ra_j2000: f64,
        dec_j2000: f64,
        aborted_prior_motion: bool,
        rejected_state: MountUiState,
    ) {
        let (ra_jnow, dec_jnow) = self.to_jnow(ra_j2000, dec_j2000);
        // Re-assert fence and active flags right before commanding motors so a concurrent
        // `tick()` slew-stop transition cannot clear them between `MountHandle::goto()` and here.
        self.tracking_paused = false;
        self.handle.mark_slew_started();
        self.handle.goto_active.store(true, Ordering::Release);
        let Some(driver) = self.driver.as_mut() else {
            return;
        };
        match driver.goto_jnow(ra_jnow, dec_jnow) {
            Ok(true) => {
                info!(
                    "Mount coarse GoTo started -> J2000 ({:.4}, {:.4}) / JNow ({:.4}, {:.4})",
                    ra_j2000, dec_j2000, ra_jnow, dec_jnow
                );
                self.state = ControllerState::Slewing {
                    target_j2000: (ra_j2000, dec_j2000),
                    pass: SlewPass::Coarse,
                    started_at: Instant::now(),
                    slew_seen: false,
                };
                self.next_poll_at = Instant::now() + self.timings.poll_interval_slewing;
                self.update_wants_solve();
                self.handle
                    .set_state_and_notice(MountUiState::Slewing, None);
            }
            Ok(false) => {
                warn!("Mount rejected GoTo command");
                let mut final_state = rejected_state;
                let mut notice = "Mount rejected GoTo command";
                if rejected_state == MountUiState::Tracking {
                    match self.ensure_tracking() {
                        Ok(StartTrackingOutcome::Started) => {}
                        Ok(StartTrackingOutcome::Refused) => {
                            final_state = self.ready_idle_ui_state();
                        }
                        Ok(StartTrackingOutcome::SafetyHalt(msg)) => {
                            final_state = self.ready_idle_ui_state();
                            notice = msg;
                        }
                        Err(e) => {
                            warn!("Resume tracking after rejected GoTo failed: {}", e);
                            self.disconnect(true);
                            return;
                        }
                    }
                }
                self.finish_goto_idle(
                    true,
                    aborted_prior_motion,
                    final_state,
                    Some(notice.to_string()),
                );
            }
            Err(e) => {
                warn!("Mount GoTo I/O error: {}", e);
                self.disconnect(true);
            }
        }
    }

    /// Aborts active driver motion and clears local motion flags.
    /// Returns `Some(was_motor_slewing)` on success, or `None` after disconnecting on I/O error.
    fn abort_driver_motion(&mut self, err_context: &str) -> Option<bool> {
        let was_slewing = self.is_motor_slewing();
        let driver = self.driver.as_mut()?;
        if let Err(e) = driver.abort() {
            warn!("{}: {}", err_context, e);
            self.disconnect(true);
            return None;
        }
        if let Some(st) = self.last_status.as_mut() {
            st.is_slewing = false;
            st.is_tracking = false;
        }
        self.hc_slewing = false;
        Some(was_slewing)
    }

    fn start_calibration(&mut self, post_action: CalPostAction) {
        self.tracking_paused = false;
        if self.is_actively_moving() {
            let Some(motors_moving) = self.abort_driver_motion("Abort before calibration failed")
            else {
                return;
            };
            self.handle.clear_slewing(motors_moving);
        } else if self.handle.is_physically_slewing() {
            self.handle.clear_slewing(false);
        }

        self.handle
            .goto_active
            .store(post_action.is_goto_active(), Ordering::Release);
        let epoch = self.handle.slew_epoch();
        let deadline = Instant::now() + self.compute_solve_timeout();
        self.state = ControllerState::Calibrating {
            step: CalStep::WaitInitialSolve { epoch, deadline },
            post_action,
        };
        self.next_poll_at = Instant::now() + self.timings.poll_interval_slewing;
        self.update_wants_solve();
        let notice = if post_action.is_goto_active() {
            "Detecting mount alignment..."
        } else {
            "Calibrating mount: waiting for initial plate solve..."
        };
        self.handle
            .set_state_and_notice(MountUiState::Calibrating, Some(notice.to_string()));
    }

    /// Reconciles optimistic `physically_slewing` / `goto_active` flags (e.g. from a superseded
    /// queued `GoTo`) with actual controller state so wait states never starve `frame_fence`.
    fn reconcile_fence_with_state(&mut self) {
        if !self.is_motor_slewing() && self.handle.is_physically_slewing() {
            match self.state {
                ControllerState::Calibrating { .. } => self.refresh_cal_wait_fence(None),
                ControllerState::WaitingPostStopSolve { ref mut epoch, .. } => {
                    *epoch = self.handle.clear_slewing(false);
                }
                ControllerState::Idle | ControllerState::Disconnected => {
                    self.handle.clear_slewing(false);
                }
                ControllerState::Slewing { .. } => unreachable!(),
            }
        }
        let goto_active = match self.state {
            ControllerState::Slewing { .. } | ControllerState::WaitingPostStopSolve { .. } => true,
            ControllerState::Calibrating { post_action, .. } => post_action.is_goto_active(),
            ControllerState::Idle | ControllerState::Disconnected => false,
        };
        self.handle
            .goto_active
            .store(goto_active, Ordering::Release);
    }

    fn is_stale_command(&self, cmd_seq: u64) -> bool {
        !self.handle.is_enabled()
            || self.handle.abort_flag.load(Ordering::Acquire)
            || cmd_seq != self.handle.cmd_seq.load(Ordering::Acquire)
    }

    fn handle_goto(&mut self, ra_j2000: f64, dec_j2000: f64, cmd_seq: u64) {
        // Skip GoTos superseded by a newer GoTo, Abort, disable, or shutdown.
        if self.is_stale_command(cmd_seq) {
            return;
        }
        self.sync_rejected = false;

        if !self.ensure_connected_or_notify() {
            return;
        }

        if self.driver.as_ref().is_some_and(|d| !d.is_goto_ready()) {
            warn!("Rejected GoTo: initial plate solve required before moving Class B mount");
            self.reject_goto_preserving_active_motion(
                MountUiState::NeedsCalibration,
                "Initial plate solve required before GoTo".to_string(),
            );
            return;
        }

        let (ra_jnow, dec_jnow) = self.to_jnow(ra_j2000, dec_j2000);
        let idle_state = self
            .last_status
            .map_or(MountUiState::Connected, |s| s.ui_state());

        if let Some((alt, _)) = self.alt_az_now(ra_jnow, dec_jnow) {
            if alt < HORIZON_SAFETY_MIN_ALT_DEG {
                warn!(
                    "Rejected GoTo to ({:.3}, {:.3}): target altitude {:.1}° < {:.1}°",
                    ra_j2000, dec_j2000, alt, HORIZON_SAFETY_MIN_ALT_DEG
                );
                let notice = format!(
                    "Target below horizon limit ({:.1}° < {:.0}°)",
                    alt, HORIZON_SAFETY_MIN_ALT_DEG
                );
                self.reject_goto_preserving_active_motion(idle_state, notice);
                return;
            }
        }

        // Uncalibrated Class B mount with >= 1 plate solve: automatically probe Alt-Az vs EQ
        // and build the sky model, then chain straight into slewing to `(ra_j2000, dec_j2000)`.
        if self.driver.as_ref().is_some_and(|d| !d.is_calibrated()) {
            if let ControllerState::Calibrating {
                ref mut post_action,
                ..
            } = self.state
            {
                *post_action = CalPostAction::GoTo(ra_j2000, dec_j2000);
                let new_deadline = Instant::now() + self.compute_solve_timeout();
                self.refresh_cal_wait_fence(Some(new_deadline));
                self.handle.goto_active.store(true, Ordering::Release);
            } else {
                self.start_calibration(CalPostAction::GoTo(ra_j2000, dec_j2000));
            }
            return;
        }

        let mut aborted_prior_motion = false;
        if self.is_actively_moving() {
            let Some(was_slewing) = self.abort_driver_motion("Abort before new GoTo failed") else {
                return;
            };
            aborted_prior_motion = was_slewing;
        }

        if !self.unpark_if_needed(aborted_prior_motion) {
            return;
        }

        // Recompute post-abort/post-unpark state so a rejected GoTo never reports Slewing/Parked.
        let rejected_state = self
            .last_status
            .map_or_else(|| self.ready_idle_ui_state(), |s| s.ui_state());
        self.start_coarse_goto(ra_j2000, dec_j2000, aborted_prior_motion, rejected_state);
    }

    fn advance_cal_slew_to_wait(
        &mut self,
        post_action: CalPostAction,
        make_wait: fn(u64, Instant) -> CalStep,
    ) {
        let post_stop_epoch = self.handle.clear_slewing(true);
        let deadline = Instant::now() + self.compute_solve_timeout();
        self.state = ControllerState::Calibrating {
            step: make_wait(post_stop_epoch, deadline),
            post_action,
        };
        self.update_wants_solve();
    }

    fn record_cal_sample_and_jog(
        &mut self,
        ra_jnow: f64,
        dec_jnow: f64,
        d1: f64,
        d2: f64,
        post_action: CalPostAction,
        make_slew: fn(Instant) -> CalStep,
        step_notice: &str,
    ) {
        let Some(driver) = self.driver.as_mut() else {
            return;
        };
        match driver.sync_jnow(ra_jnow, dec_jnow) {
            Ok(Ok(())) => {}
            Ok(Err(offset)) => {
                warn!("Calibration sample rejected (offset {:.2}°)", offset);
                let ui_state = self.ready_idle_ui_state();
                self.finish_goto_idle(
                    false,
                    false,
                    ui_state,
                    Some(format!(
                        "Calibration sample rejected (offset {:.2}°)",
                        offset
                    )),
                );
                return;
            }
            Err(e) => {
                warn!("Calibration sample failed: {}", e);
                self.disconnect(true);
                return;
            }
        }
        self.handle.mark_slew_started();
        match driver.jog_axes_rad(d1, d2) {
            Ok(true) => {
                self.state = ControllerState::Calibrating {
                    step: make_slew(Instant::now()),
                    post_action,
                };
                self.next_poll_at = Instant::now() + self.timings.poll_interval_slewing;
                self.update_wants_solve();
                self.handle
                    .set_state_and_notice(MountUiState::Calibrating, Some(step_notice.to_string()));
            }
            Ok(false) => {
                self.finish_goto_idle(
                    true,
                    false,
                    MountUiState::NeedsCalibration,
                    Some("Mount does not support calibration jog".to_string()),
                );
            }
            Err(e) => {
                warn!("Calibration jog failed: {}", e);
                self.disconnect(true);
            }
        }
    }

    fn handle_calibrate(&mut self, cmd_seq: u64) {
        if self.is_stale_command(cmd_seq) {
            return;
        }
        if !self.ensure_connected_or_notify() {
            return;
        }
        self.reconcile_fence_with_state();
        if self.driver.as_ref().is_some_and(|d| d.is_calibrated())
            || matches!(self.state, ControllerState::Calibrating { .. })
        {
            return;
        }
        self.start_calibration(CalPostAction::TrackInPlace);
    }

    fn handle_set_tracking(&mut self, enabled: bool, cmd_seq: u64) {
        if self.is_stale_command(cmd_seq) {
            return;
        }
        if !self.ensure_connected_or_notify() {
            return;
        }
        self.reconcile_fence_with_state();
        if self.is_actively_moving() {
            return;
        }

        if !enabled {
            let Some(driver) = self.driver.as_mut() else {
                return;
            };
            match driver.stop_tracking() {
                Ok(true) => {
                    if let Some(st) = self.last_status.as_mut() {
                        st.is_tracking = false;
                    }
                    self.tracking_paused = true;
                    self.update_wants_solve();
                    let ui_state = self.ready_idle_ui_state();
                    self.handle.set_state_and_notice(ui_state, None);
                }
                Ok(false) => {
                    let cur = self.handle.telemetry().state;
                    self.handle.set_state_and_notice(
                        cur,
                        Some("Mount refused to stop tracking".to_string()),
                    );
                }
                Err(e) => {
                    warn!("Mount stop tracking failed: {}", e);
                    self.disconnect(true);
                }
            }
            return;
        }

        self.tracking_paused = false;
        self.update_wants_solve();

        if !self.unpark_if_needed(false) {
            return;
        }

        let is_calibrated = self.driver.as_ref().is_some_and(|d| d.is_calibrated());
        if !is_calibrated {
            if self.driver.as_ref().is_some_and(|d| !d.is_goto_ready()) {
                self.handle.set_state_and_notice(
                    MountUiState::NeedsCalibration,
                    Some("Initial plate solve required before tracking".to_string()),
                );
                return;
            }
            self.start_calibration(CalPostAction::ReturnToInitialAndTrack);
            return;
        }

        match self.ensure_tracking() {
            Ok(StartTrackingOutcome::Started) => {
                self.handle
                    .set_state_and_notice(MountUiState::Tracking, None);
            }
            Ok(StartTrackingOutcome::Refused) => {
                let ui_state = self.ready_idle_ui_state();
                self.handle.set_state_and_notice(
                    ui_state,
                    Some("Mount refused to start tracking".to_string()),
                );
            }
            Ok(StartTrackingOutcome::SafetyHalt(msg)) => {
                let ui_state = self.ready_idle_ui_state();
                self.handle
                    .set_state_and_notice(ui_state, Some(msg.to_string()));
            }
            Err(e) => {
                warn!("Mount start tracking failed: {}", e);
                self.disconnect(true);
            }
        }
    }

    fn handle_calibration_solve(&mut self, ra_j2000: f64, dec_j2000: f64, solve_epoch: u64) {
        let ControllerState::Calibrating { step, post_action } = self.state else {
            return;
        };
        let (ra_jnow, dec_jnow) = self.to_jnow(ra_j2000, dec_j2000);

        match step {
            CalStep::WaitInitialSolve { epoch, .. } => {
                if solve_epoch != epoch {
                    return;
                }
                if let Some((alt, _)) = self.alt_az_now(ra_jnow, dec_jnow) {
                    if !(CALIBRATION_MIN_ALT_DEG..=CALIBRATION_MAX_ALT_DEG).contains(&alt) {
                        let ui_state = self.ready_idle_ui_state();
                        self.finish_goto_idle(
                            false,
                            false,
                            ui_state,
                            Some(format!(
                                "Aim between {:.0}°–{:.0}° altitude before calibrating (current {:.1}°)",
                                CALIBRATION_MIN_ALT_DEG, CALIBRATION_MAX_ALT_DEG, alt
                            )),
                        );
                        return;
                    }
                }
                let next_action = match post_action {
                    CalPostAction::ReturnToInitialAndTrack
                        if dec_jnow.abs() <= CALIBRATION_RETURN_MAX_DEC_DEG =>
                    {
                        CalPostAction::GoTo(ra_j2000, dec_j2000)
                    }
                    CalPostAction::ReturnToInitialAndTrack => CalPostAction::TrackInPlace,
                    other => other,
                };
                self.handle
                    .goto_active
                    .store(next_action.is_goto_active(), Ordering::Release);
                if let Some(driver) = self.driver.as_mut() {
                    driver.reset_calibration();
                }
                self.last_passive_sample = None;
                // Jog Axis 2 first so an EQ mount starting at Polaris steps cleanly off the pole.
                let ax2_step_rad = calibration_ax2_step_deg(dec_jnow).to_radians();
                self.record_cal_sample_and_jog(
                    ra_jnow,
                    dec_jnow,
                    0.0,
                    ax2_step_rad,
                    next_action,
                    |started_at| CalStep::SlewAx2 {
                        started_at,
                        slew_seen: false,
                    },
                    "Calibrating mount (step 1/2)...",
                );
            }
            CalStep::WaitAx2Solve { epoch, .. } => {
                if solve_epoch != epoch {
                    return;
                }
                let ax1_step_rad = calibration_ax1_step_deg(dec_jnow).to_radians();
                self.record_cal_sample_and_jog(
                    ra_jnow,
                    dec_jnow,
                    ax1_step_rad,
                    0.0,
                    post_action,
                    |started_at| CalStep::SlewAx1 {
                        started_at,
                        slew_seen: false,
                    },
                    "Calibrating mount (step 2/2)...",
                );
            }
            CalStep::WaitAx1Solve { epoch, .. } => {
                if solve_epoch != epoch {
                    return;
                }
                let Some(driver) = self.driver.as_mut() else {
                    return;
                };
                let sync_ok = match driver.sync_jnow(ra_jnow, dec_jnow) {
                    Ok(Ok(())) => true,
                    Ok(Err(offset)) => {
                        warn!("Calibration final sample rejected (offset {:.2}°)", offset);
                        false
                    }
                    Err(e) => {
                        warn!("Calibration final sample failed: {}", e);
                        self.disconnect(true);
                        return;
                    }
                };
                let calibrated = sync_ok && driver.is_calibrated();
                if calibrated {
                    self.handle.set_goto_ready(driver.is_goto_ready());
                    self.needs_bg_sync = false;
                    match post_action {
                        CalPostAction::GoTo(tgt_ra, tgt_dec) => {
                            self.start_coarse_goto(tgt_ra, tgt_dec, false, MountUiState::Tracking);
                        }
                        CalPostAction::TrackInPlace | CalPostAction::ReturnToInitialAndTrack => {
                            self.finish_goto_tracking(MountUiState::Tracking, None);
                        }
                    }
                } else {
                    driver.reset_calibration();
                    self.last_passive_sample = None;
                    self.handle.set_goto_ready(false);
                    self.finish_goto_idle(
                        false,
                        false,
                        MountUiState::NeedsCalibration,
                        Some("Calibration failed — try another sky region".to_string()),
                    );
                }
            }
            CalStep::SlewAx1 { .. } | CalStep::SlewAx2 { .. } => {}
        }
    }

    fn handle_abort(&mut self) {
        if !self.is_actively_moving() {
            self.handle.goto_active.store(false, Ordering::Release);
            // Clear optimistic fence if a queued GoTo was superseded before running.
            if self.handle.is_physically_slewing() {
                self.handle.clear_slewing(false);
            }
            return;
        }

        let ready = self.driver.as_ref().is_some_and(|d| d.is_goto_ready());
        self.handle.set_goto_ready(ready);
        let idle_state = self.ready_idle_ui_state();
        let notice = if matches!(self.state, ControllerState::Calibrating { .. }) {
            "Calibration aborted"
        } else {
            "Slew aborted"
        };

        let Some(motors_moving) = self.abort_driver_motion("Mount abort I/O error") else {
            return;
        };
        self.finish_goto_idle(true, motors_moving, idle_state, Some(notice.to_string()));
    }

    fn handle_solved(&mut self, ra_j2000: f64, dec_j2000: f64, solve_epoch: u64) {
        match self.state {
            ControllerState::WaitingPostStopSolve {
                target_j2000,
                epoch,
                ..
            } => {
                if solve_epoch != epoch {
                    debug!(
                        "Ignoring stale solve (epoch={}, expected={})",
                        solve_epoch, epoch
                    );
                    return;
                }
                self.handle_post_stop_solve(ra_j2000, dec_j2000, target_j2000);
            }
            ControllerState::Calibrating { .. } => {
                self.handle_calibration_solve(ra_j2000, dec_j2000, solve_epoch);
            }
            ControllerState::Idle => {
                let needs_cal = self.driver.as_ref().is_some_and(|d| !d.is_calibrated());
                if !needs_cal && (self.tracking_paused || !self.needs_bg_sync || self.sync_rejected)
                {
                    return;
                }
                if solve_epoch != self.handle.slew_epoch() {
                    return;
                }
                self.handle_background_sync(ra_j2000, dec_j2000);
            }
            ControllerState::Slewing { .. } | ControllerState::Disconnected => {}
        }
    }

    fn ensure_tracking(&mut self) -> IoResult<StartTrackingOutcome> {
        if self.last_status.is_some_and(|s| s.is_tracking) {
            if self.tracking_paused {
                self.tracking_paused = false;
                self.update_wants_solve();
            }
            return Ok(StartTrackingOutcome::Started);
        }
        let Some(driver) = self.driver.as_mut() else {
            return Ok(StartTrackingOutcome::Refused);
        };
        let outcome = driver.start_tracking()?;
        match outcome {
            StartTrackingOutcome::Started => {
                if let Some(st) = self.last_status.as_mut() {
                    st.is_tracking = true;
                }
                if self.tracking_paused {
                    self.tracking_paused = false;
                    self.update_wants_solve();
                }
            }
            StartTrackingOutcome::SafetyHalt(_) => {
                if let Some(st) = self.last_status.as_mut() {
                    st.is_tracking = false;
                }
                self.tracking_paused = true;
                self.update_wants_solve();
            }
            StartTrackingOutcome::Refused => {}
        }
        Ok(outcome)
    }

    fn handle_post_stop_solve(
        &mut self,
        solved_ra_j2000: f64,
        solved_dec_j2000: f64,
        target_j2000: (f64, f64),
    ) {
        if self.driver.is_none() {
            return;
        }
        let err_deg = angular_separation_deg(
            solved_ra_j2000,
            solved_dec_j2000,
            target_j2000.0,
            target_j2000.1,
        );

        if err_deg <= CENTERING_TOLERANCE_DEG {
            info!("Mount centered within {:.3}°; GoTo complete", err_deg);
            self.needs_bg_sync = false;
            self.finish_goto_tracking(MountUiState::Tracking, None);
            return;
        }

        info!(
            "Post-slew error {:.3}° > {:.3}°; syncing mount and issuing fine re-GoTo",
            err_deg, CENTERING_TOLERANCE_DEG
        );
        let (solved_ra_jnow, solved_dec_jnow) = self.to_jnow(solved_ra_j2000, solved_dec_j2000);

        // Class A mounts require tracking active before accepting a sky sync; Class B mounts
        // update their internal sky model directly without moving motors (so a coarse overshoot
        // into a safety cone still syncs and fine-slews back out cleanly).
        if !self
            .driver
            .as_ref()
            .is_some_and(|d| d.requires_host_tracking())
        {
            match self.ensure_tracking() {
                Ok(StartTrackingOutcome::Started) => {}
                Ok(StartTrackingOutcome::Refused) => {
                    warn!("Mount rejected start tracking before post-slew sync; skipping re-GoTo");
                    let fallback = self.ready_idle_ui_state();
                    self.finish_goto_idle(
                        false,
                        false,
                        fallback,
                        Some("Mount refused to enable tracking".to_string()),
                    );
                    return;
                }
                Ok(StartTrackingOutcome::SafetyHalt(msg)) => {
                    warn!("Mount safety halt before post-slew sync: {}", msg);
                    let fallback = self.ready_idle_ui_state();
                    self.finish_goto_idle(false, false, fallback, Some(msg.to_string()));
                    return;
                }
                Err(e) => {
                    warn!("Mount start tracking before sync failed: {}", e);
                    self.disconnect(true);
                    return;
                }
            }
        }

        let Some(driver) = self.driver.as_mut() else {
            return;
        };
        match driver.sync_jnow(solved_ra_jnow, solved_dec_jnow) {
            Ok(Ok(())) => {
                self.needs_bg_sync = false;
                let (target_ra_jnow, target_dec_jnow) =
                    self.to_jnow(target_j2000.0, target_j2000.1);
                self.handle.mark_slew_started();

                let Some(driver) = self.driver.as_mut() else {
                    return;
                };
                match driver.goto_jnow(target_ra_jnow, target_dec_jnow) {
                    Ok(true) => {
                        self.state = ControllerState::Slewing {
                            target_j2000,
                            pass: SlewPass::Fine,
                            started_at: Instant::now(),
                            slew_seen: false,
                        };
                        self.next_poll_at = Instant::now() + self.timings.poll_interval_slewing;
                        self.update_wants_solve();
                        self.handle
                            .set_state_and_notice(MountUiState::Centering, None);
                    }
                    Ok(false) => {
                        warn!("Mount rejected fine re-GoTo command");
                        self.finish_goto_tracking(
                            MountUiState::Tracking,
                            Some("Mount rejected centering slew".to_string()),
                        );
                    }
                    Err(e) => {
                        warn!("Mount fine re-GoTo I/O error: {}", e);
                        self.disconnect(true);
                    }
                }
            }
            Ok(Err(offset_deg)) => {
                warn!(
                    "Mount rejected sync (readback offset {:.2}°); skipping fine re-GoTo",
                    offset_deg
                );
                self.sync_rejected = true;
                self.finish_goto_tracking(
                    MountUiState::Tracking,
                    Some(format!("Mount rejected sync (offset {:.2}°)", offset_deg)),
                );
            }
            Err(e) => {
                warn!("Mount sync I/O error: {}", e);
                self.disconnect(true);
            }
        }
    }

    fn handle_background_sync(&mut self, solved_ra_j2000: f64, solved_dec_j2000: f64) {
        let Some(status) = self.last_status else {
            return;
        };
        if status.is_slewing || status.is_parked || status.at_zero {
            return;
        }
        let Some((is_alt_az, was_calibrated)) = self
            .driver
            .as_ref()
            .map(|d| (d.is_alt_az(), d.is_calibrated()))
        else {
            return;
        };

        let (solved_ra_jnow, solved_dec_jnow) = self.to_jnow(solved_ra_j2000, solved_dec_j2000);

        // Throttled passive Class B sky-model collection: accept the first solve immediately
        // (unlocking GoTo), then only sample when time >= 1s AND sky has moved >= 2°.
        if !was_calibrated {
            let now = Instant::now();
            if let Some((prev_at, prev_ra, prev_dec)) = self.last_passive_sample {
                let sky_sep =
                    angular_separation_deg(solved_ra_jnow, solved_dec_jnow, prev_ra, prev_dec);
                if now.duration_since(prev_at) < PASSIVE_CAL_MIN_INTERVAL
                    || sky_sep < PASSIVE_CAL_MIN_SKY_SEP_DEG
                {
                    return;
                }
            }
            let Some(driver) = self.driver.as_mut() else {
                return;
            };
            self.last_passive_sample = Some((now, solved_ra_jnow, solved_dec_jnow));
            match driver.sync_jnow(solved_ra_jnow, solved_dec_jnow) {
                Ok(_) => {
                    let goto_ready = driver.is_goto_ready();
                    let calibrated = driver.is_calibrated();
                    self.handle.set_goto_ready(goto_ready);
                    if calibrated {
                        self.needs_bg_sync = false;
                        self.update_wants_solve();
                        if !self.tracking_paused {
                            match self.ensure_tracking() {
                                Ok(StartTrackingOutcome::Started) => {
                                    self.handle
                                        .set_state_and_notice(MountUiState::Tracking, None);
                                }
                                Ok(StartTrackingOutcome::Refused) => {
                                    self.handle
                                        .set_state_and_notice(MountUiState::Connected, None);
                                }
                                Ok(StartTrackingOutcome::SafetyHalt(msg)) => {
                                    self.handle.set_state_and_notice(
                                        MountUiState::Connected,
                                        Some(msg.to_string()),
                                    );
                                }
                                Err(e) => {
                                    warn!("Start tracking after passive calibration failed: {}", e);
                                    self.disconnect(true);
                                    return;
                                }
                            }
                        } else {
                            self.handle
                                .set_state_and_notice(MountUiState::Connected, None);
                        }
                    } else if goto_ready {
                        self.handle
                            .set_state_and_notice(MountUiState::Connected, None);
                    }
                }
                Err(e) => {
                    warn!("Passive Class B calibration sample I/O error: {}", e);
                    self.disconnect(true);
                }
            }
            return;
        }

        // Silent exact-axis-pole singularity guard (no user-facing banner).
        if is_alt_az {
            if let Some((alt, _)) = self.alt_az_now(solved_ra_jnow, solved_dec_jnow) {
                if alt > POLE_SINGULARITY_SKIP_DEG {
                    return;
                }
            }
        } else if solved_dec_jnow.abs() > POLE_SINGULARITY_SKIP_DEG {
            return;
        }

        let mount_err = angular_separation_deg(
            solved_ra_jnow,
            solved_dec_jnow,
            status.ra_jnow_deg,
            status.dec_jnow_deg,
        );
        if mount_err < CENTERING_TOLERANCE_DEG {
            self.needs_bg_sync = false;
            self.update_wants_solve();
            if self.handle.telemetry().state == MountUiState::TrackingNoSolve {
                self.handle
                    .set_state_and_notice(MountUiState::Tracking, None);
            }
            return;
        }

        match self.ensure_tracking() {
            Ok(StartTrackingOutcome::Started) => {}
            Ok(StartTrackingOutcome::Refused) => {
                self.sync_rejected = true;
                self.update_wants_solve();
                return;
            }
            Ok(StartTrackingOutcome::SafetyHalt(msg)) => {
                let ui_state = self.ready_idle_ui_state();
                self.handle
                    .set_state_and_notice(ui_state, Some(msg.to_string()));
                return;
            }
            Err(e) => {
                warn!("Mount start tracking before background sync failed: {}", e);
                self.disconnect(true);
                return;
            }
        }

        let Some(driver) = self.driver.as_mut() else {
            return;
        };
        match driver.sync_jnow(solved_ra_jnow, solved_dec_jnow) {
            Ok(Ok(())) => {
                info!(
                    "Background mount sync succeeded (corrected {:.3}° offset)",
                    mount_err
                );
                self.needs_bg_sync = false;
                self.update_wants_solve();
                self.handle
                    .set_state_and_notice(MountUiState::Tracking, None);
            }
            Ok(Err(offset_deg)) => {
                warn!(
                    "Background mount sync rejected (readback offset {:.2}°)",
                    offset_deg
                );
                self.sync_rejected = true;
                self.update_wants_solve();
                let cur_state = self.handle.telemetry().state;
                self.handle.set_state_and_notice(
                    cur_state,
                    Some(format!("Mount rejected sync (offset {:.2}°)", offset_deg)),
                );
            }
            Err(e) => {
                warn!("Background mount sync I/O error: {}", e);
                self.disconnect(true);
            }
        }
    }

    fn update_wants_solve(&self) {
        let wants = match self.state {
            ControllerState::WaitingPostStopSolve { .. }
            | ControllerState::Calibrating {
                step:
                    CalStep::WaitInitialSolve { .. }
                    | CalStep::WaitAx1Solve { .. }
                    | CalStep::WaitAx2Solve { .. },
                ..
            } => true,
            ControllerState::Idle => {
                let needs_cal = self.driver.as_ref().is_some_and(|d| !d.is_calibrated());
                needs_cal || (!self.tracking_paused && self.needs_bg_sync && !self.sync_rejected)
            }
            ControllerState::Disconnected
            | ControllerState::Slewing { .. }
            | ControllerState::Calibrating {
                step: CalStep::SlewAx1 { .. } | CalStep::SlewAx2 { .. },
                ..
            } => false,
        };
        self.handle.wants_solve.store(wants, Ordering::Release);
    }

    fn compute_solve_timeout(&self) -> Duration {
        let win_ms = self
            .handle
            .expected_solve_window_ms
            .load(Ordering::Relaxed)
            .max(1_000);
        Duration::from_millis(win_ms.saturating_mul(3)).clamp(
            self.timings.min_solve_timeout,
            self.timings.max_solve_timeout,
        )
    }

    fn disconnect(&mut self, schedule_reconnect: bool) {
        let was_slewing = self.is_motor_slewing() || self.handle.is_physically_slewing();
        let needs_track_halt = self
            .driver
            .as_ref()
            .is_some_and(|d| d.requires_host_tracking());

        if !schedule_reconnect && (was_slewing || needs_track_halt) {
            if let Some(driver) = self.driver.as_mut() {
                let _ = driver.abort();
            }
        }

        self.driver = None;
        self.last_status = None;
        self.hc_slewing = false;
        self.site_synced = false;
        self.tracking_paused = false;
        self.last_passive_sample = None;
        self.state = ControllerState::Disconnected;
        self.update_wants_solve();
        self.handle.goto_active.store(false, Ordering::Release);
        self.handle.clear_slewing(was_slewing);
        if schedule_reconnect && self.config.is_some() {
            self.schedule_reconnect("Mount disconnected — reconnecting...".to_string());
        } else {
            self.handle.set_goto_ready(false);
            self.next_reconnect_at = None;
        }
    }

    fn schedule_reconnect(&mut self, notice: String) {
        self.next_reconnect_at = Some(Instant::now() + self.timings.reconnect_backoff);
        self.handle.set_connected_model(None, false);
        self.handle
            .set_state_and_notice(MountUiState::Reconnecting, Some(notice));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    use kinematic::RawAxisMount;
    use nalgebra::{Matrix3, Rotation3};
    use sky_model::earth_vec_to_jnow;

    #[derive(Clone, Debug)]
    struct MockMountState {
        is_alt_az: bool,
        status: MountRawStatus,
        goto_accepts: bool,
        sync_accepts: bool,
        sync_reject_offset: f64,
        slew_polls_remaining: usize,
        poll_calls: usize,
        goto_calls: Vec<(f64, f64)>,
        sync_calls: Vec<(f64, f64)>,
        abort_calls: usize,
        unpark_calls: usize,
        start_tracking_calls: usize,
        stop_tracking_calls: usize,
        site_sync_calls: usize,
    }

    impl MockMountState {
        fn idle_tracking(is_alt_az: bool, ra_jnow_deg: f64, dec_jnow_deg: f64) -> Arc<Mutex<Self>> {
            Arc::new(Mutex::new(Self {
                is_alt_az,
                status: MountRawStatus {
                    is_slewing: false,
                    is_tracking: true,
                    is_parked: false,
                    at_zero: false,
                    gps_locked: true,
                    ra_jnow_deg,
                    dec_jnow_deg,
                    safety_notice: None,
                },
                goto_accepts: true,
                sync_accepts: true,
                sync_reject_offset: 0.0,
                slew_polls_remaining: 0,
                poll_calls: 0,
                goto_calls: Vec::new(),
                sync_calls: Vec::new(),
                abort_calls: 0,
                unpark_calls: 0,
                start_tracking_calls: 0,
                stop_tracking_calls: 0,
                site_sync_calls: 0,
            }))
        }
    }

    struct MockMountDriver {
        shared: Arc<Mutex<MockMountState>>,
    }

    impl MountDriver for MockMountDriver {
        fn model_name(&self) -> &str {
            "MockMount"
        }

        fn is_alt_az(&self) -> bool {
            self.shared.lock().unwrap().is_alt_az
        }

        fn poll(&mut self) -> IoResult<MountRawStatus> {
            let mut st = self.shared.lock().unwrap();
            st.poll_calls += 1;
            if st.slew_polls_remaining > 0 {
                st.slew_polls_remaining -= 1;
                st.status.is_slewing = true;
            } else {
                st.status.is_slewing = false;
            }
            Ok(st.status)
        }

        fn goto_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<bool> {
            let mut st = self.shared.lock().unwrap();
            st.goto_calls.push((ra_deg, dec_deg));
            if !st.goto_accepts {
                return Ok(false);
            }
            st.status.at_zero = false;
            st.status.is_tracking = true;
            st.status.is_slewing = st.slew_polls_remaining > 0;
            Ok(true)
        }

        fn sync_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<Result<(), f64>> {
            let mut st = self.shared.lock().unwrap();
            st.sync_calls.push((ra_deg, dec_deg));
            if st.sync_accepts {
                st.status.ra_jnow_deg = ra_deg;
                st.status.dec_jnow_deg = dec_deg;
                Ok(Ok(()))
            } else {
                Ok(Err(st.sync_reject_offset))
            }
        }

        fn start_tracking(&mut self) -> IoResult<StartTrackingOutcome> {
            let mut st = self.shared.lock().unwrap();
            st.start_tracking_calls += 1;
            st.status.is_tracking = true;
            Ok(StartTrackingOutcome::Started)
        }

        fn stop_tracking(&mut self) -> IoResult<bool> {
            let mut st = self.shared.lock().unwrap();
            st.stop_tracking_calls += 1;
            st.status.is_tracking = false;
            Ok(true)
        }

        fn abort(&mut self) -> IoResult<()> {
            let mut st = self.shared.lock().unwrap();
            st.abort_calls += 1;
            st.slew_polls_remaining = 0;
            st.status.is_slewing = false;
            Ok(())
        }

        fn unpark(&mut self) -> IoResult<bool> {
            let mut st = self.shared.lock().unwrap();
            st.unpark_calls += 1;
            st.status.is_parked = false;
            Ok(true)
        }

        fn sync_site_time(&mut self, _lat: f64, _lon: f64, _utc: i64) -> IoResult<()> {
            self.shared.lock().unwrap().site_sync_calls += 1;
            Ok(())
        }
    }

    struct TestMount {
        handle: Arc<MountHandle>,
        join: Option<JoinHandle<()>>,
    }

    impl TestMount {
        fn stop(self) {
            drop(self);
        }
    }

    impl Drop for TestMount {
        fn drop(&mut self) {
            self.handle.shutdown();
            if let Some(j) = self.join.take() {
                let _ = j.join();
            }
        }
    }

    fn fast_timings() -> ControllerTimings {
        ControllerTimings {
            poll_interval_slewing: Duration::from_millis(5),
            poll_interval_idle: Duration::from_millis(15),
            coarse_start_grace: Duration::from_millis(10),
            fine_start_grace: Duration::from_millis(0),
            reconnect_backoff: Duration::from_millis(20),
            min_solve_timeout: Duration::from_millis(80),
            max_solve_timeout: Duration::from_millis(120),
        }
    }

    fn start_mock_controller(
        mock: Arc<Mutex<MockMountState>>,
        timings: ControllerTimings,
    ) -> TestMount {
        let ts = Arc::new(Timescale::default());
        let mock_factory = mock.clone();
        let factory: DriverFactory = Box::new(move |_cfg, _conn, _ts| {
            Ok(Box::new(MockMountDriver {
                shared: mock_factory.clone(),
            }))
        });
        let (handle, join) = MountHandle::start_internal(ts, None, factory, timings);
        handle.configure(Some(MountConfig::ioptron("127.0.0.1", 8899)));
        wait_until(
            || handle.telemetry().model.is_some(),
            Duration::from_secs(2),
        );
        TestMount {
            handle,
            join: Some(join),
        }
    }

    fn wait_until<F: Fn() -> bool>(cond: F, timeout: Duration) {
        let start = Instant::now();
        while !cond() {
            assert!(start.elapsed() < timeout, "Timed out waiting for condition");
            sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn test_angular_separation_wrap_and_poles() {
        assert!((angular_separation_deg(359.99, 0.0, 0.01, 0.0) - 0.02).abs() < 1e-6);
        assert!((angular_separation_deg(0.0, 89.9, 180.0, 89.9) - 0.2).abs() < 1e-6);
        assert!(angular_separation_deg(123.456, -45.678, 123.456, -45.678) < 1e-9);
    }

    #[test]
    fn test_closed_loop_goto_stale_epoch_and_single_fine_pass() {
        let mock = MockMountState::idle_tracking(true, 100.0, 20.0);
        let tm = start_mock_controller(mock.clone(), fast_timings());
        let handle = &tm.handle;
        let pre_goto_epoch = handle.slew_epoch();

        mock.lock().unwrap().slew_polls_remaining = 2;
        handle.goto(150.0, 30.0);
        assert!(handle.is_goto_active());
        assert!(handle.is_physically_slewing());

        wait_until(
            || handle.telemetry().state == MountUiState::Centering,
            Duration::from_secs(2),
        );
        assert!(!handle.is_physically_slewing());
        assert!(handle.is_goto_active());

        // Stale pre-GoTo solve must be ignored.
        handle.on_solved(100.0, 20.0, pre_goto_epoch);
        sleep(Duration::from_millis(20));
        assert_eq!(mock.lock().unwrap().sync_calls.len(), 0);
        assert_eq!(mock.lock().unwrap().goto_calls.len(), 1);

        // Fresh post-stop solve with 1.0 deg error triggers sync + single fine re-GoTo.
        let post_stop_epoch = handle.slew_epoch();
        handle.on_solved(151.0, 30.0, post_stop_epoch);

        wait_until(
            || !handle.is_goto_active() && handle.telemetry().state == MountUiState::Tracking,
            Duration::from_secs(2),
        );
        assert_eq!(mock.lock().unwrap().sync_calls.len(), 1);
        assert_eq!(mock.lock().unwrap().goto_calls.len(), 2);

        tm.stop();
    }

    #[test]
    fn test_cloud_timeout_fallback_and_no_surprise_slew() {
        let mock = MockMountState::idle_tracking(true, 50.0, 10.0);
        let tm = start_mock_controller(mock.clone(), fast_timings());
        let handle = &tm.handle;

        mock.lock().unwrap().slew_polls_remaining = 1;
        handle.goto(60.0, 15.0);

        wait_until(
            || handle.telemetry().state == MountUiState::TrackingNoSolve,
            Duration::from_secs(2),
        );
        assert!(!handle.is_goto_active());
        assert_eq!(mock.lock().unwrap().goto_calls.len(), 1);

        let ep = handle.slew_epoch();
        handle.on_solved(61.5, 15.0, ep);
        sleep(Duration::from_millis(30));
        assert_eq!(mock.lock().unwrap().goto_calls.len(), 1);

        tm.stop();
    }

    #[test]
    fn test_sync_rejection_latch_and_start_tracking_before_sync() {
        let mock = MockMountState::idle_tracking(true, 10.0, 10.0);
        {
            let mut st = mock.lock().unwrap();
            st.status.is_tracking = false;
            st.sync_accepts = false;
            st.sync_reject_offset = 12.5;
        }

        let tm = start_mock_controller(mock.clone(), fast_timings());
        let handle = &tm.handle;
        let ep = handle.slew_epoch();
        handle.on_solved(20.0, 20.0, ep);
        wait_until(
            || handle.telemetry().notice.is_some(),
            Duration::from_secs(2),
        );
        assert_eq!(mock.lock().unwrap().start_tracking_calls, 1);
        assert_eq!(mock.lock().unwrap().sync_calls.len(), 1);

        for _ in 0..5 {
            handle.on_solved(20.0, 20.0, ep);
        }
        sleep(Duration::from_millis(30));
        assert_eq!(mock.lock().unwrap().sync_calls.len(), 1);

        handle.notify_target_pixel_changed();
        sleep(Duration::from_millis(15));
        handle.on_solved(20.0, 20.0, ep);
        wait_until(
            || mock.lock().unwrap().sync_calls.len() == 2,
            Duration::from_secs(2),
        );

        tm.stop();
    }

    #[test]
    fn test_at_zero_skips_bg_sync_and_allows_first_goto() {
        let mock = MockMountState::idle_tracking(true, 0.0, 37.0);
        {
            let mut st = mock.lock().unwrap();
            st.status.is_tracking = false;
            st.status.at_zero = true;
        }

        let tm = start_mock_controller(mock.clone(), fast_timings());
        let handle = &tm.handle;
        assert_eq!(handle.telemetry().state, MountUiState::AtZero);

        handle.on_solved(10.0, 37.0, handle.slew_epoch());
        sleep(Duration::from_millis(30));
        assert_eq!(mock.lock().unwrap().sync_calls.len(), 0);

        mock.lock().unwrap().slew_polls_remaining = 1;
        handle.goto(80.0, 25.0);
        wait_until(
            || handle.telemetry().state == MountUiState::Centering,
            Duration::from_secs(2),
        );
        let ep = handle.slew_epoch();
        handle.on_solved(82.0, 25.0, ep);
        wait_until(
            || !handle.is_goto_active() && handle.telemetry().state == MountUiState::Tracking,
            Duration::from_secs(2),
        );
        assert_eq!(mock.lock().unwrap().sync_calls.len(), 1);
        assert_eq!(mock.lock().unwrap().goto_calls.len(), 2);

        tm.stop();
    }

    #[test]
    fn test_fast_abort_single_fire_idle_abort_noop_and_frame_fence() {
        let mock = MockMountState::idle_tracking(true, 0.0, 0.0);
        let tm = start_mock_controller(mock.clone(), fast_timings());
        let handle = &tm.handle;

        // Abort while idle must NOT send spurious :Q# or drop stationary frames.
        handle.abort();
        sleep(Duration::from_millis(25));
        assert_eq!(mock.lock().unwrap().abort_calls, 0);
        assert_eq!(handle.telemetry().state, MountUiState::Tracking);
        assert!(!handle.frame_fence(SystemTime::now(), 500_000).0);

        mock.lock().unwrap().slew_polls_remaining = 100;
        handle.goto(45.0, 45.0);
        wait_until(
            || handle.telemetry().state == MountUiState::Slewing,
            Duration::from_secs(2),
        );

        assert!(handle.frame_fence(SystemTime::now(), 33_333).0);

        let t0 = Instant::now();
        handle.abort();
        assert!(!handle.is_goto_active());
        wait_until(
            || mock.lock().unwrap().abort_calls == 1 && !handle.is_physically_slewing(),
            Duration::from_secs(1),
        );
        assert!(t0.elapsed() < Duration::from_millis(200));

        sleep(Duration::from_millis(25));
        assert_eq!(mock.lock().unwrap().abort_calls, 1);

        assert!(handle.frame_fence(SystemTime::now(), 500_000).0);
        assert!(
            !handle
                .frame_fence(SystemTime::now() + Duration::from_millis(500), 33_333)
                .0
        );

        tm.stop();
    }

    #[test]
    fn test_configure_then_immediate_goto_and_goto_then_immediate_abort() {
        let mock = MockMountState::idle_tracking(true, 10.0, 10.0);
        let ts = Arc::new(Timescale::default());
        let mock_factory = mock.clone();
        let factory: DriverFactory = Box::new(move |_cfg, _conn, _ts| {
            Ok(Box::new(MockMountDriver {
                shared: mock_factory.clone(),
            }))
        });
        let (handle, join) = MountHandle::start_internal(ts, None, factory, fast_timings());
        let tm = TestMount {
            handle,
            join: Some(join),
        };

        // configure(Some) immediately followed by goto() must NOT drop the GoTo.
        mock.lock().unwrap().slew_polls_remaining = 2;
        tm.handle
            .configure(Some(MountConfig::ioptron("127.0.0.1", 8899)));
        tm.handle.goto(70.0, 20.0);
        wait_until(
            || mock.lock().unwrap().goto_calls.len() == 1,
            Duration::from_secs(2),
        );

        // Wait until settled, then issue goto() immediately followed by abort():
        // cmd_seq bumps in abort(), so the superseded GoTo never commands motors.
        tm.handle.abort();
        wait_until(
            || !tm.handle.is_physically_slewing() && !tm.handle.is_abort_pending(),
            Duration::from_secs(2),
        );
        let gotos_before = mock.lock().unwrap().goto_calls.len();
        tm.handle.goto(90.0, 30.0);
        tm.handle.abort();
        wait_until(
            || !tm.handle.is_abort_pending() && !tm.handle.is_physically_slewing(),
            Duration::from_secs(2),
        );
        sleep(Duration::from_millis(20));
        assert_eq!(mock.lock().unwrap().goto_calls.len(), gotos_before);
        assert!(!tm.handle.is_goto_active());
        assert!(!tm.handle.is_physically_slewing());

        tm.stop();
    }

    #[test]
    fn test_horizon_safety_floor_unpark_gps_respect_and_no_false_frame_drop() {
        let mock = MockMountState::idle_tracking(true, 0.0, 0.0);
        mock.lock().unwrap().status.is_parked = true;

        let tm = start_mock_controller(mock.clone(), fast_timings());
        let handle = &tm.handle;
        assert_eq!(mock.lock().unwrap().unpark_calls, 1);
        assert_eq!(mock.lock().unwrap().site_sync_calls, 0);

        handle.set_site_location(Some(89.9), Some(0.0));
        handle.goto(0.0, -45.0);
        wait_until(
            || !handle.is_goto_active() && handle.telemetry().notice.is_some(),
            Duration::from_secs(2),
        );
        assert_eq!(mock.lock().unwrap().goto_calls.len(), 0);
        assert!(
            handle
                .telemetry()
                .notice
                .unwrap()
                .contains("below horizon limit")
        );
        assert!(!handle.frame_fence(SystemTime::now(), 500_000).0);

        tm.stop();
    }

    #[test]
    fn test_goto_while_slewing_goto_rejected_and_hc_cancels_centering() {
        let mock = MockMountState::idle_tracking(false, 0.0, 0.0);
        let tm = start_mock_controller(mock.clone(), fast_timings());
        let handle = &tm.handle;

        mock.lock().unwrap().slew_polls_remaining = 50;
        handle.goto(30.0, 20.0);
        wait_until(
            || handle.telemetry().state == MountUiState::Slewing,
            Duration::from_secs(2),
        );

        handle.goto(40.0, 25.0);
        wait_until(
            || mock.lock().unwrap().goto_calls.len() == 2,
            Duration::from_secs(2),
        );
        assert_eq!(mock.lock().unwrap().abort_calls, 1);

        mock.lock().unwrap().goto_accepts = false;
        handle.goto(50.0, 30.0);
        wait_until(
            || !handle.is_goto_active() && handle.telemetry().notice.is_some(),
            Duration::from_secs(2),
        );
        assert!(handle.telemetry().notice.unwrap().contains("rejected GoTo"));

        {
            let mut st = mock.lock().unwrap();
            st.goto_accepts = true;
            st.slew_polls_remaining = 1;
        }
        handle.goto(60.0, 30.0);
        wait_until(
            || handle.telemetry().state == MountUiState::Centering,
            Duration::from_secs(2),
        );
        mock.lock().unwrap().slew_polls_remaining = 10;
        wait_until(
            || !handle.is_goto_active() && handle.telemetry().state == MountUiState::Slewing,
            Duration::from_secs(2),
        );

        let aborts_before = mock.lock().unwrap().abort_calls;
        handle.configure(None);
        wait_until(
            || mock.lock().unwrap().abort_calls == aborts_before + 1,
            Duration::from_secs(2),
        );
        assert_eq!(handle.telemetry().state, MountUiState::Disabled);

        tm.stop();
    }

    #[test]
    fn test_silent_axis_pole_guard_altaz_vs_eq() {
        let mock_aa = MockMountState::idle_tracking(true, 0.0, 80.0);
        let tm_aa = start_mock_controller(mock_aa.clone(), fast_timings());
        tm_aa.handle.set_site_location(Some(45.0), Some(0.0));
        tm_aa
            .handle
            .on_solved(37.95, 89.26, tm_aa.handle.slew_epoch());
        wait_until(
            || mock_aa.lock().unwrap().sync_calls.len() == 1,
            Duration::from_secs(2),
        );
        assert!(tm_aa.handle.telemetry().notice.is_none());
        tm_aa.stop();

        let mock_eq = MockMountState::idle_tracking(false, 0.0, 80.0);
        let tm_eq = start_mock_controller(mock_eq.clone(), fast_timings());
        tm_eq
            .handle
            .on_solved(37.95, 89.26, tm_eq.handle.slew_epoch());
        sleep(Duration::from_millis(30));
        assert_eq!(mock_eq.lock().unwrap().sync_calls.len(), 0);
        assert!(tm_eq.handle.telemetry().notice.is_none());
        tm_eq.stop();
    }

    #[test]
    fn test_solves_do_not_flood_tcp_polling() {
        let mock = MockMountState::idle_tracking(true, 10.0, 10.0);
        let mut timings = fast_timings();
        timings.poll_interval_idle = Duration::from_millis(200);

        let tm = start_mock_controller(mock.clone(), timings);
        let polls_after_connect = mock.lock().unwrap().poll_calls;
        let ep = tm.handle.slew_epoch();

        for _ in 0..50 {
            tm.handle.on_solved(10.0, 10.0, ep);
        }
        sleep(Duration::from_millis(50));
        assert_eq!(mock.lock().unwrap().poll_calls, polls_after_connect);

        tm.stop();
    }

    #[test]
    fn test_disabled_push_to_passthrough() {
        let ts = Arc::new(Timescale::default());
        let handle = MountHandle::start(ts, None);
        assert!(!handle.is_enabled());
        assert!(!handle.is_goto_active());
        assert!(!handle.is_physically_slewing());
        assert_eq!(handle.telemetry().state, MountUiState::Disabled);
        assert!(!handle.telemetry().goto_ready);
        assert_eq!(handle.frame_fence(SystemTime::now(), 1_000_000), (false, 0));
        handle.goto(10.0, 20.0);
        assert!(!handle.is_goto_active());
        handle.shutdown();
    }

    struct SimRawState {
        ax1: f64,
        ax2: f64,
        rot: Matrix3<f64>,
        slew_polls_per_goto: usize,
        slew_polls_remaining: usize,
        goto_count: usize,
        rate_updates: usize,
    }

    struct SimRawMount(Arc<Mutex<SimRawState>>);

    impl RawAxisMount for SimRawMount {
        fn model_name(&self) -> &str {
            "SimClassB"
        }
        fn is_alt_az(&self) -> bool {
            true
        }
        fn read_axes_rad(&mut self) -> IoResult<(f64, f64)> {
            let s = self.0.lock().unwrap();
            Ok((s.ax1, s.ax2))
        }
        fn is_slewing(&mut self) -> IoResult<bool> {
            let mut s = self.0.lock().unwrap();
            if s.slew_polls_remaining > 0 {
                s.slew_polls_remaining -= 1;
                Ok(true)
            } else {
                Ok(false)
            }
        }
        fn goto_axes_rad(&mut self, ax1_rad: f64, ax2_rad: f64) -> IoResult<()> {
            let mut s = self.0.lock().unwrap();
            s.ax1 = ax1_rad;
            s.ax2 = ax2_rad;
            s.slew_polls_remaining = s.slew_polls_per_goto;
            s.goto_count += 1;
            Ok(())
        }
        fn set_rates_rad_per_sec(&mut self, _r1: f64, _r2: f64) -> IoResult<()> {
            self.0.lock().unwrap().rate_updates += 1;
            Ok(())
        }
        fn abort(&mut self) -> IoResult<()> {
            self.0.lock().unwrap().slew_polls_remaining = 0;
            Ok(())
        }
    }

    fn solve_sim_j2000(sim: &Arc<Mutex<SimRawState>>, ts: &Timescale) -> (f64, f64) {
        let s = sim.lock().unwrap();
        let m = Vector3::new(
            s.ax2.cos() * s.ax1.cos(),
            s.ax2.cos() * s.ax1.sin(),
            s.ax2.sin(),
        );
        let (ra_jnow, dec_jnow) = earth_vec_to_jnow(s.rot * m, Utc::now(), ts);
        precess_coordinates(ra_jnow, dec_jnow, current_julian_epoch(ts), 2000.0, ts)
    }

    fn start_sim_class_b(
        ax1: f64,
        ax2: f64,
        rot: Matrix3<f64>,
        site: Option<(f64, f64)>,
        slew_polls_per_goto: usize,
    ) -> (Arc<Timescale>, Arc<Mutex<SimRawState>>, TestMount) {
        let ts = Arc::new(Timescale::default());
        let sim = Arc::new(Mutex::new(SimRawState {
            ax1,
            ax2,
            rot,
            slew_polls_per_goto,
            slew_polls_remaining: 0,
            goto_count: 0,
            rate_updates: 0,
        }));
        let sim_factory = sim.clone();
        let factory: DriverFactory = Box::new(move |cfg, _conn, ts_ref| {
            Ok(Box::new(KinematicMount::new(
                SimRawMount(sim_factory.clone()),
                ts_ref.clone(),
                cfg.gem_meridian_flip,
            )))
        });
        let (handle, join) = MountHandle::start_internal(ts.clone(), None, factory, fast_timings());
        let tm = TestMount {
            handle,
            join: Some(join),
        };
        if let Some((lat, lon)) = site {
            tm.handle.set_site_location(Some(lat), Some(lon));
        }
        tm.handle
            .configure(Some(MountConfig::skywatcher("127.0.0.1", 11880)));
        wait_until(
            || tm.handle.telemetry().state == MountUiState::NeedsCalibration,
            Duration::from_secs(2),
        );
        (ts, sim, tm)
    }

    #[test]
    fn test_class_b_single_solve_unlocks_goto_auto_probes_and_chains_slew() {
        let rot = *Rotation3::from_euler_angles(0.5, -0.3, 1.1).matrix();
        let (ts, sim, tm) = start_sim_class_b(0.4, 0.7, rot, None, 2);
        assert!(!tm.handle.telemetry().goto_ready);

        // Before ANY plate solve, Class B mount MUST reject GoTo.
        tm.handle.goto(120.0, 25.0);
        wait_until(
            || !tm.handle.is_goto_active() && tm.handle.telemetry().notice.is_some(),
            Duration::from_secs(2),
        );
        assert_eq!(sim.lock().unwrap().goto_count, 0);
        assert!(
            tm.handle
                .telemetry()
                .notice
                .unwrap()
                .contains("Initial plate solve required")
        );

        // A single stationary plate solve unlocks GoTo WITHOUT moving motors!
        let (ra0, dec0) = solve_sim_j2000(&sim, &ts);
        tm.handle.on_solved(ra0, dec0, tm.handle.slew_epoch());
        wait_until(
            || {
                tm.handle.telemetry().goto_ready
                    && tm.handle.telemetry().state == MountUiState::Connected
            },
            Duration::from_secs(2),
        );
        assert_eq!(sim.lock().unwrap().goto_count, 0);

        // First GoTo automatically runs the 2-hop probe and chains straight into slewing to target!
        let (tgt_a_ra, tgt_a_dec) = ((ra0 + 8.0).rem_euclid(360.0), dec0);
        let (tgt_b_ra, tgt_b_dec) = (
            (ra0 + 14.0).rem_euclid(360.0),
            (dec0 + 4.0).clamp(-70.0, 70.0),
        );
        tm.handle.goto(tgt_a_ra, tgt_a_dec);
        wait_until(
            || {
                tm.handle.telemetry().state == MountUiState::Calibrating
                    && !tm.handle.is_physically_slewing()
            },
            Duration::from_secs(2),
        );
        assert!(tm.handle.is_goto_active());

        // Deliver initial probe solve -> triggers Axis-2-first jog (goto_count == 1).
        tm.handle.on_solved(ra0, dec0, tm.handle.slew_epoch());
        wait_until(
            || sim.lock().unwrap().goto_count == 1 && !tm.handle.is_physically_slewing(),
            Duration::from_secs(2),
        );

        // Mid-probe retargeting updates pending target without aborting calibration or wedging epoch!
        tm.handle.goto(tgt_b_ra, tgt_b_dec);
        wait_until(
            || !tm.handle.is_physically_slewing(),
            Duration::from_secs(2),
        );

        let (ra1, dec1) = solve_sim_j2000(&sim, &ts);
        tm.handle.on_solved(ra1, dec1, tm.handle.slew_epoch());
        wait_until(
            || sim.lock().unwrap().goto_count == 2 && !tm.handle.is_physically_slewing(),
            Duration::from_secs(2),
        );

        // Final probe solve fits the sky model and immediately chains coarse GoTo -> tgt_b (goto_count == 3)!
        let (ra2, dec2) = solve_sim_j2000(&sim, &ts);
        tm.handle.on_solved(ra2, dec2, tm.handle.slew_epoch());
        wait_until(
            || {
                sim.lock().unwrap().goto_count == 3
                    && tm.handle.telemetry().state == MountUiState::Centering
            },
            Duration::from_secs(2),
        );

        // Post-slew solve confirms we landed on tgt_b (the retargeted destination) -> Tracking!
        let (landed_ra, landed_dec) = solve_sim_j2000(&sim, &ts);
        assert!(
            angular_separation_deg(tgt_b_ra, tgt_b_dec, landed_ra, landed_dec) < 0.25,
            "Chained GoTo did not land on retargeted destination"
        );
        tm.handle
            .on_solved(landed_ra, landed_dec, tm.handle.slew_epoch());
        wait_until(
            || !tm.handle.is_goto_active() && tm.handle.telemetry().state == MountUiState::Tracking,
            Duration::from_secs(2),
        );
        wait_until(
            || sim.lock().unwrap().rate_updates >= 1,
            Duration::from_secs(2),
        );

        tm.stop();
    }

    fn drive_cal_probe_steps(ts: &Timescale, sim: &Arc<Mutex<SimRawState>>, handle: &MountHandle) {
        for expected_gotos in [1usize, 2] {
            let (ra, dec) = solve_sim_j2000(sim, ts);
            handle.on_solved(ra, dec, handle.slew_epoch());
            wait_until(
                || {
                    sim.lock().unwrap().goto_count == expected_gotos
                        && !handle.is_physically_slewing()
                },
                Duration::from_secs(2),
            );
        }
    }

    #[test]
    fn test_class_b_eq_polaris_home_auto_probe_and_manual_calibrate() {
        // 1) Start dead-center on Polaris (Dec = +89.2°); +9° Axis-2 jog crosses straight over the pole!
        let (ts, sim, tm) = start_sim_class_b(
            0.2,
            89.2_f64.to_radians(),
            Matrix3::identity(),
            Some((38.0, -122.0)),
            1,
        );

        // Single Polaris solve unlocks GoTo.
        let (ra0, dec0) = solve_sim_j2000(&sim, &ts);
        tm.handle.on_solved(ra0, dec0, tm.handle.slew_epoch());
        wait_until(|| tm.handle.telemetry().goto_ready, Duration::from_secs(2));

        // Pick an Eastern target (HA = -3h, Dec = +40°) and a Western target (HA = +3h, Dec = +40°)
        // to verify BOTH unflipped East GoTo AND counterweight-down meridian flip to West!
        let lst_deg = (ts.from_datetime(Utc::now()).gast() * 15.0 - 122.0).rem_euclid(360.0);
        let east_ra = (lst_deg + 45.0).rem_euclid(360.0);
        let west_ra = (lst_deg - 45.0).rem_euclid(360.0);
        tm.handle.goto(east_ra, 40.0);
        wait_until(
            || {
                tm.handle.telemetry().state == MountUiState::Calibrating
                    && !tm.handle.is_physically_slewing()
            },
            Duration::from_secs(2),
        );

        drive_cal_probe_steps(&ts, &sim, &tm.handle);

        // Final probe solve fits EQ model across the pole and chains straight into East GoTo!
        let (ra2, dec2) = solve_sim_j2000(&sim, &ts);
        tm.handle.on_solved(ra2, dec2, tm.handle.slew_epoch());
        wait_until(
            || {
                sim.lock().unwrap().goto_count == 3
                    && tm.handle.telemetry().state == MountUiState::Centering
            },
            Duration::from_secs(2),
        );
        let (land_ra, land_dec) = solve_sim_j2000(&sim, &ts);
        assert!(
            angular_separation_deg(east_ra, 40.0, land_ra, land_dec) < 0.25,
            "Polaris EQ auto-probe GoTo missed East target"
        );
        let east_ax2_deg = sim.lock().unwrap().ax2.to_degrees();
        tm.handle
            .on_solved(land_ra, land_dec, tm.handle.slew_epoch());
        wait_until(
            || !tm.handle.is_goto_active() && tm.handle.telemetry().state == MountUiState::Tracking,
            Duration::from_secs(2),
        );

        // Now GoTo West (HA = +3h) MUST meridian-flip pier side across the North Pole!
        tm.handle.goto(west_ra, 40.0);
        wait_until(
            || {
                sim.lock().unwrap().goto_count == 4
                    && tm.handle.telemetry().state == MountUiState::Centering
            },
            Duration::from_secs(2),
        );
        let (west_land_ra, west_land_dec) = solve_sim_j2000(&sim, &ts);
        assert!(
            angular_separation_deg(west_ra, 40.0, west_land_ra, west_land_dec) < 0.25,
            "Meridian-flipped West GoTo missed target"
        );
        let west_ax2_deg = sim.lock().unwrap().ax2.to_degrees();
        // East and West pier sides MUST be symmetric across +90° (40° vs 140°)!
        assert!(
            ((east_ax2_deg - 40.0).abs() < 1.0 && (west_ax2_deg - 140.0).abs() < 1.0)
                || ((east_ax2_deg - 140.0).abs() < 1.0 && (west_ax2_deg - 40.0).abs() < 1.0),
            "GEM East->West GoTo did not meridian flip across North Pole (east={:.1}°, west={:.1}°)",
            east_ax2_deg,
            west_ax2_deg
        );
        tm.stop();

        // 2) Direct `handle.calibrate()` finishes into Tracking without a GoTo target.
        let (ts2, sim2, tm2) = start_sim_class_b(0.5, 0.7, Matrix3::identity(), None, 1);
        tm2.handle.calibrate();
        wait_until(
            || tm2.handle.telemetry().state == MountUiState::Calibrating,
            Duration::from_secs(2),
        );
        drive_cal_probe_steps(&ts2, &sim2, &tm2.handle);
        let (ra2, dec2) = solve_sim_j2000(&sim2, &ts2);
        tm2.handle.on_solved(ra2, dec2, tm2.handle.slew_epoch());
        wait_until(
            || {
                tm2.handle.telemetry().goto_ready
                    && tm2.handle.telemetry().state == MountUiState::Tracking
            },
            Duration::from_secs(2),
        );
        tm2.stop();
    }

    #[test]
    fn test_set_tracking_class_a_and_class_b_probe_return() {
        // 1) Class A: stop_tracking -> Connected (and background solves do NOT re-enable tracking!),
        //    start_tracking -> Tracking, and `abort()` followed immediately by `set_tracking(false)`
        //    still executes the abort.
        let mock = MockMountState::idle_tracking(true, 100.0, 20.0);
        let tm_a = start_mock_controller(mock.clone(), fast_timings());
        assert_eq!(tm_a.handle.telemetry().state, MountUiState::Tracking);

        tm_a.handle.set_tracking(false);
        wait_until(
            || {
                mock.lock().unwrap().stop_tracking_calls == 1
                    && tm_a.handle.telemetry().state == MountUiState::Connected
            },
            Duration::from_secs(2),
        );
        // Background solve while tracking is paused must NOT silently turn tracking back on!
        tm_a.handle.on_solved(105.0, 25.0, tm_a.handle.slew_epoch());
        sleep(Duration::from_millis(25));
        assert_eq!(mock.lock().unwrap().start_tracking_calls, 0);
        assert_eq!(tm_a.handle.telemetry().state, MountUiState::Connected);

        tm_a.handle.set_tracking(true);
        wait_until(
            || {
                mock.lock().unwrap().start_tracking_calls == 1
                    && tm_a.handle.telemetry().state == MountUiState::Tracking
            },
            Duration::from_secs(2),
        );

        // `abort()` followed immediately by `set_tracking(false)` must NOT drop the abort.
        mock.lock().unwrap().slew_polls_remaining = 50;
        tm_a.handle.goto(130.0, 30.0);
        wait_until(
            || tm_a.handle.telemetry().state == MountUiState::Slewing,
            Duration::from_secs(2),
        );
        tm_a.handle.abort();
        tm_a.handle.set_tracking(false);
        wait_until(
            || mock.lock().unwrap().abort_calls == 1 && !tm_a.handle.is_physically_slewing(),
            Duration::from_secs(2),
        );
        tm_a.stop();

        // 2) Class B (uncalibrated, manually aimed): 1 plate solve + set_tracking(true)
        //    runs the 2-hop probe AND returns straight back to the aimed coordinate + tracks!
        let rot = *Rotation3::from_euler_angles(0.4, -0.25, 0.9).matrix();
        let (ts, sim, tm_b) = start_sim_class_b(0.6, 0.75, rot, None, 1);

        // Before any solve, set_tracking(true) refuses cleanly without moving motors.
        tm_b.handle.set_tracking(true);
        wait_until(
            || tm_b.handle.telemetry().notice.is_some(),
            Duration::from_secs(2),
        );
        assert_eq!(sim.lock().unwrap().goto_count, 0);

        // Single solve at aimed position unlocks tracking.
        let (aim_ra, aim_dec) = solve_sim_j2000(&sim, &ts);
        tm_b.handle
            .on_solved(aim_ra, aim_dec, tm_b.handle.slew_epoch());
        wait_until(
            || tm_b.handle.telemetry().goto_ready,
            Duration::from_secs(2),
        );

        // Enabling tracking probes and chains GoTo right back to `(aim_ra, aim_dec)`!
        tm_b.handle.set_tracking(true);
        wait_until(
            || {
                tm_b.handle.telemetry().state == MountUiState::Calibrating
                    && !tm_b.handle.is_physically_slewing()
            },
            Duration::from_secs(2),
        );
        drive_cal_probe_steps(&ts, &sim, &tm_b.handle);
        let (ra2, dec2) = solve_sim_j2000(&sim, &ts);
        tm_b.handle.on_solved(ra2, dec2, tm_b.handle.slew_epoch());
        wait_until(
            || {
                sim.lock().unwrap().goto_count == 3
                    && tm_b.handle.telemetry().state == MountUiState::Centering
            },
            Duration::from_secs(2),
        );
        let (ret_ra, ret_dec) = solve_sim_j2000(&sim, &ts);
        assert!(
            angular_separation_deg(aim_ra, aim_dec, ret_ra, ret_dec) < 0.25,
            "Class B set_tracking(true) did not return to original aimed position"
        );
        tm_b.handle
            .on_solved(ret_ra, ret_dec, tm_b.handle.slew_epoch());
        wait_until(
            || {
                !tm_b.handle.is_goto_active()
                    && tm_b.handle.telemetry().state == MountUiState::Tracking
            },
            Duration::from_secs(2),
        );

        // Stopping & restarting tracking keeps the calibrated sky model intact!
        tm_b.handle.set_tracking(false);
        wait_until(
            || tm_b.handle.telemetry().state == MountUiState::Connected,
            Duration::from_secs(2),
        );
        let rates_before = sim.lock().unwrap().rate_updates;
        tm_b.handle.set_tracking(true);
        wait_until(
            || {
                tm_b.handle.telemetry().state == MountUiState::Tracking
                    && sim.lock().unwrap().rate_updates > rates_before
            },
            Duration::from_secs(2),
        );
        assert_eq!(sim.lock().unwrap().goto_count, 3);

        tm_b.stop();
    }
}
