// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

//! Wireless motorized telescope mount control and closed-loop astrometric GoTo.
//!
//! Runs a single blocking `std::thread` controller (`MountController`) over a
//! small synchronous `MountDriver` trait. Keeping the mount loop off Tokio
//! avoids cancelling half-duplex serial-over-WiFi TCP reads mid-flight and lets
//! `:Q#` aborts fire immediately on the already-open socket.

mod ioptron;

use std::io::Result as IoResult;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::{Builder, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use chrono::Utc;
use log::{debug, info, warn};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use starfield::time::Timescale;

use crate::engine::calculate_alt_az;
use crate::telescope_server::{AtomicOptionF64, precess_coordinates};
use ioptron::IoptronMount;

const CENTERING_TOLERANCE_DEG: f64 = 0.05;
const HORIZON_SAFETY_MIN_ALT_DEG: f64 = 5.0;
const POLE_SINGULARITY_SKIP_DEG: f64 = 88.0;
const FRAME_FENCE_MARGIN: Duration = Duration::from_millis(100);

/// Optional platform TCP socket factory (e.g. Android `ConnectivityManager` `TRANSPORT_WIFI` socket).
pub type TcpConnector = Arc<dyn Fn(&str, u16) -> IoResult<TcpStream> + Send + Sync>;

/// Runtime configuration for motorized mount control (iOptron V2.5 / V3.10 TCP Wi-Fi).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MountConfig {
    pub host: String,
    pub port: u16,
    /// When `true`, performs one post-slew plate solve + `:CM#` sync + fine re-GoTo pass.
    pub closed_loop_goto: bool,
}

impl MountConfig {
    /// Convenience constructor for an iOptron Wi-Fi mount with closed-loop GoTo enabled.
    pub fn ioptron(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            closed_loop_goto: true,
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
}

trait MountDriver: Send {
    fn model_name(&self) -> &str;
    fn is_alt_az(&self) -> bool;
    fn poll(&mut self) -> IoResult<MountRawStatus>;
    fn goto_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<bool>;
    /// Syncs the mount to `(ra_deg, dec_deg)` JNow and verifies readback.
    /// Returns `Ok(Ok(()))` on verified convergence, or `Ok(Err(offset_deg))` if rejected/divergent.
    fn sync_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<Result<(), f64>>;
    fn start_tracking(&mut self) -> IoResult<bool>;
    fn abort(&mut self) -> IoResult<()>;
    fn unpark(&mut self) -> IoResult<bool>;
    fn sync_site_time(&mut self, lat_deg: f64, lon_deg: f64, utc_ms: i64) -> IoResult<()>;
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

fn j2000_to_jnow(ra_deg: f64, dec_deg: f64, ts: &Timescale) -> (f64, f64) {
    let t = ts.from_datetime(Utc::now());
    let epoch = 2000.0 + (t.tdb() - 2451545.0) / 365.25;
    precess_coordinates(ra_deg, dec_deg, 2000.0, epoch, ts)
}

enum MountCmd {
    Configure(Option<MountConfig>),
    Reconnect,
    GoTo {
        ra_j2000: f64,
        dec_j2000: f64,
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

enum ControllerState {
    Disconnected,
    Idle,
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
            fine_start_grace: Duration::from_millis(150),
            reconnect_backoff: Duration::from_secs(2),
            min_solve_timeout: Duration::from_secs(8),
            max_solve_timeout: Duration::from_secs(20),
        }
    }
}

type DriverFactory =
    Box<dyn Fn(&MountConfig, Option<&TcpConnector>) -> IoResult<Box<dyn MountDriver>> + Send>;

fn default_driver_factory(
    cfg: &MountConfig,
    connector: Option<&TcpConnector>,
) -> IoResult<Box<dyn MountDriver>> {
    Ok(Box::new(IoptronMount::connect(
        &cfg.host, cfg.port, connector,
    )?))
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
        self.abort_flag.store(false, Ordering::SeqCst);
        let seq = self.cmd_seq.fetch_add(1, Ordering::SeqCst) + 1;
        self.mark_slew_started();
        self.goto_active.store(true, Ordering::Release);
        let _ = self.cmd_tx.send(MountCmd::GoTo {
            ra_j2000,
            dec_j2000,
            seq,
        });
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

    /// Re-arms the background `:CM#` sync latch after `target_pixel` eyepiece alignment changes.
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

    fn set_connected_model(&self, model: Option<String>) {
        let mut guard = self.telemetry.lock().unwrap();
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        guard.0.model = model;
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
            next_poll_at: Instant::now(),
            next_reconnect_at: None,
        }
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
            ControllerState::WaitingPostStopSolve { deadline, .. } => self
                .next_poll_at
                .min(deadline)
                .saturating_duration_since(now),
            ControllerState::Slewing { .. } | ControllerState::Idle => {
                self.next_poll_at.saturating_duration_since(now)
            }
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
                if old.host == new.host && old.port == new.port && self.driver.is_some() =>
            {
                self.config = new_cfg;
                return;
            }
            _ => {}
        }

        self.disconnect(false);
        self.config = new_cfg;
        if self.config.is_some() {
            self.handle.set_connected_model(None);
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
        if let ControllerState::WaitingPostStopSolve { deadline, .. } = self.state {
            if now >= deadline {
                warn!("Mount centering timed out waiting for plate solve; tracking open-loop");
                self.finish_goto_idle(
                    false,
                    false,
                    MountUiState::TrackingNoSolve,
                    Some("Centering timed out (no plate solve) — tracking open-loop".to_string()),
                );
                return;
            }
        }

        if now < self.next_poll_at {
            return;
        }
        let interval = match self.state {
            ControllerState::Slewing { .. } | ControllerState::WaitingPostStopSolve { .. } => {
                self.timings.poll_interval_slewing
            }
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

        if let Err(e) = self.sync_site_if_needed(raw.gps_locked) {
            warn!("Mount site/time sync error: {}", e);
            self.disconnect(true);
            return;
        }

        // If a new GoTo/Abort was queued while `driver.poll()` blocked on TCP I/O,
        // defer state transitions so the queued command is handled cleanly next loop.
        if self.handle.cmd_seq.load(Ordering::Acquire) != cmd_seq_before_poll {
            return;
        }

        match self.state {
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
                    } else {
                        self.finish_goto_idle(false, false, MountUiState::Tracking, None);
                    }
                }
            }
            ControllerState::WaitingPostStopSolve { .. } => {
                if raw.is_slewing {
                    info!("Manual/external slew detected during centering; cancelling centering");
                    self.hc_slewing = true;
                    self.handle.goto_active.store(false, Ordering::Release);
                    self.handle.mark_slew_started();
                    self.state = ControllerState::Idle;
                    self.update_wants_solve();
                    self.handle
                        .set_state_and_notice(MountUiState::Slewing, None);
                }
            }
            ControllerState::Idle => {
                let ui_state = raw.ui_state();
                if raw.is_slewing && !self.hc_slewing {
                    self.hc_slewing = true;
                    self.handle.mark_slew_started();
                } else if !raw.is_slewing && self.hc_slewing {
                    self.hc_slewing = false;
                    self.handle.clear_slewing(true);
                }

                let prev = self.handle.telemetry();
                let notice = if raw.is_slewing { None } else { prev.notice };
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
        match (self.factory)(&cfg, connector.as_ref()) {
            Ok(mut driver) => {
                let model = driver.model_name().to_string();
                let init_res = (|| -> IoResult<(MountRawStatus, bool)> {
                    let mut status = driver.poll()?;
                    if status.is_parked {
                        info!("Mount {} is parked; sending unpark (:MP0#)", model);
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
                        let ui_state = status.ui_state();
                        self.driver = Some(driver);
                        self.last_status = Some(status);
                        self.hc_slewing = status.is_slewing;
                        self.site_synced = site_synced;
                        self.needs_bg_sync = true;
                        self.sync_rejected = false;
                        self.state = ControllerState::Idle;
                        self.next_reconnect_at = None;
                        self.next_poll_at = Instant::now() + self.timings.poll_interval_idle;
                        // Discard any stale abort requested while disconnected/connecting.
                        self.handle.abort_flag.store(false, Ordering::SeqCst);
                        self.update_wants_solve();
                        self.handle.set_connected_model(Some(model));
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

    fn handle_goto(&mut self, ra_j2000: f64, dec_j2000: f64, cmd_seq: u64) {
        // Skip GoTos superseded by a newer GoTo, Abort, disable, or shutdown.
        if !self.handle.is_enabled()
            || self.handle.abort_flag.load(Ordering::Acquire)
            || cmd_seq != self.handle.cmd_seq.load(Ordering::Acquire)
        {
            return;
        }
        self.sync_rejected = false;

        if self.driver.is_none() {
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
                self.finish_goto_idle(true, false, idle_state, Some(notice));
                return;
            }
        }

        let mut aborted_prior_motion = false;
        if matches!(
            self.state,
            ControllerState::Slewing { .. } | ControllerState::WaitingPostStopSolve { .. }
        ) || self.hc_slewing
        {
            let Some(driver) = self.driver.as_mut() else {
                return;
            };
            if let Err(e) = driver.abort() {
                warn!("Abort before new GoTo failed: {}", e);
                self.disconnect(true);
                return;
            }
            aborted_prior_motion =
                matches!(self.state, ControllerState::Slewing { .. }) || self.hc_slewing;
            self.hc_slewing = false;
        }

        if self.last_status.is_some_and(|s| s.is_parked) {
            let Some(driver) = self.driver.as_mut() else {
                return;
            };
            match driver.unpark() {
                Ok(true) => {
                    if let Some(st) = self.last_status.as_mut() {
                        st.is_parked = false;
                    }
                }
                Ok(false) => {
                    self.finish_goto_idle(
                        true,
                        aborted_prior_motion,
                        MountUiState::Parked,
                        Some("Mount refused to unpark".to_string()),
                    );
                    return;
                }
                Err(e) => {
                    warn!("Unpark before GoTo failed: {}", e);
                    self.disconnect(true);
                    return;
                }
            }
        }

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
                self.finish_goto_idle(
                    true,
                    aborted_prior_motion,
                    idle_state,
                    Some("Mount rejected GoTo command".to_string()),
                );
            }
            Err(e) => {
                warn!("Mount GoTo I/O error: {}", e);
                self.disconnect(true);
            }
        }
    }

    fn handle_abort(&mut self) {
        let was_active = matches!(
            self.state,
            ControllerState::Slewing { .. } | ControllerState::WaitingPostStopSolve { .. }
        ) || self.hc_slewing
            || self.last_status.is_some_and(|s| s.is_slewing);

        if !was_active {
            self.handle.goto_active.store(false, Ordering::Release);
            return;
        }

        let motors_moving = matches!(self.state, ControllerState::Slewing { .. })
            || self.hc_slewing
            || self.last_status.is_some_and(|s| s.is_slewing);

        if let Some(driver) = self.driver.as_mut() {
            if let Err(e) = driver.abort() {
                warn!("Mount abort I/O error: {}", e);
                self.disconnect(true);
                return;
            }
            self.finish_goto_idle(
                true,
                motors_moving,
                MountUiState::Connected,
                Some("Slew aborted".to_string()),
            );
        } else {
            self.handle.goto_active.store(false, Ordering::Release);
            self.handle.clear_slewing(false);
        }
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
            ControllerState::Idle => {
                if !self.needs_bg_sync || self.sync_rejected {
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

    fn ensure_tracking(&mut self) -> IoResult<bool> {
        if self.last_status.is_some_and(|s| s.is_tracking) {
            return Ok(true);
        }
        let Some(driver) = self.driver.as_mut() else {
            return Ok(false);
        };
        let ok = driver.start_tracking()?;
        if ok {
            if let Some(st) = self.last_status.as_mut() {
                st.is_tracking = true;
            }
        }
        Ok(ok)
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
            self.finish_goto_idle(false, false, MountUiState::Tracking, None);
            return;
        }

        info!(
            "Post-slew error {:.3}° > {:.3}°; syncing mount and issuing fine re-GoTo",
            err_deg, CENTERING_TOLERANCE_DEG
        );
        let (solved_ra_jnow, solved_dec_jnow) = self.to_jnow(solved_ra_j2000, solved_dec_j2000);

        match self.ensure_tracking() {
            Ok(true) => {}
            Ok(false) => {
                warn!("Mount rejected :ST1# before post-slew sync; skipping re-GoTo");
                self.finish_goto_idle(
                    false,
                    false,
                    MountUiState::Connected,
                    Some("Mount refused to enable tracking".to_string()),
                );
                return;
            }
            Err(e) => {
                warn!("Mount :ST1# before sync failed: {}", e);
                self.disconnect(true);
                return;
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
                        self.finish_goto_idle(
                            true,
                            false,
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
                    "Mount rejected :CM# sync (readback offset {:.2}°); skipping fine re-GoTo",
                    offset_deg
                );
                self.sync_rejected = true;
                self.finish_goto_idle(
                    false,
                    false,
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
        let Some(is_alt_az) = self.driver.as_ref().map(|d| d.is_alt_az()) else {
            return;
        };

        let (solved_ra_jnow, solved_dec_jnow) = self.to_jnow(solved_ra_j2000, solved_dec_j2000);

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
            return;
        }

        match self.ensure_tracking() {
            Ok(true) => {}
            Ok(false) => {
                self.sync_rejected = true;
                self.update_wants_solve();
                return;
            }
            Err(e) => {
                warn!("Mount :ST1# before background sync failed: {}", e);
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
            ControllerState::WaitingPostStopSolve { .. } => true,
            ControllerState::Idle => self.needs_bg_sync && !self.sync_rejected,
            ControllerState::Disconnected | ControllerState::Slewing { .. } => false,
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
        let was_slewing = matches!(self.state, ControllerState::Slewing { .. })
            || self.hc_slewing
            || self.last_status.is_some_and(|s| s.is_slewing)
            || self.handle.is_physically_slewing();

        if !schedule_reconnect && was_slewing {
            if let Some(driver) = self.driver.as_mut() {
                let _ = driver.abort();
            }
        }

        self.driver = None;
        self.last_status = None;
        self.hc_slewing = false;
        self.site_synced = false;
        self.state = ControllerState::Disconnected;
        self.update_wants_solve();
        self.handle.goto_active.store(false, Ordering::Release);
        self.handle.clear_slewing(was_slewing);
        if schedule_reconnect && self.config.is_some() {
            self.schedule_reconnect("Mount disconnected — reconnecting...".to_string());
        } else {
            self.next_reconnect_at = None;
        }
    }

    fn schedule_reconnect(&mut self, notice: String) {
        self.next_reconnect_at = Some(Instant::now() + self.timings.reconnect_backoff);
        self.handle.set_connected_model(None);
        self.handle
            .set_state_and_notice(MountUiState::Reconnecting, Some(notice));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

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

        fn start_tracking(&mut self) -> IoResult<bool> {
            let mut st = self.shared.lock().unwrap();
            st.start_tracking_calls += 1;
            st.status.is_tracking = true;
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
        fn stop(mut self) {
            self.handle.shutdown();
            if let Some(j) = self.join.take() {
                let _ = j.join();
            }
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
        let factory: DriverFactory = Box::new(move |_cfg, _conn| {
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
        let factory: DriverFactory = Box::new(move |_cfg, _conn| {
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
        sleep(Duration::from_millis(30));
        assert_eq!(mock.lock().unwrap().goto_calls.len(), gotos_before);
        assert!(!tm.handle.is_goto_active());

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
        assert_eq!(handle.frame_fence(SystemTime::now(), 1_000_000), (false, 0));
        handle.goto(10.0, 20.0);
        assert!(!handle.is_goto_active());
        handle.shutdown();
    }
}
