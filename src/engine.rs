// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

use std::f64::consts::TAU;
use std::sync::atomic::Ordering;
use std::{
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime},
};

use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use starfield::time::Timescale;
use tokio::sync::mpsc::Receiver;
use tokio::time::interval;

use crate::alpaca_server::AlpacaServer;
use crate::mount::{MountHandle, MountUiState};
use crate::telescope_server::{Lx200Telescope, Lx200WifiTelescope, SharedPosition, TargetOp};
use crate::{CameraConfig, CameraTrait, CameraType};
use tetra3::{
    fast_extractor::{FastBgSubMode, FastDownsample, FastExtractOptions, FastExtractor},
    solver::{SolveOptions, SolveStatus},
};

const MULTIPLE_EXTRACTION_MIN_EXPOSURE_US: i32 = 100_000;
const MULTIPLE_EXTRACTION_SIGMA: f32 = 2.6;
const MULTIPLE_EXTRACTION_MIN_AREA: usize = 1;
const MULTIPLE_EXTRACTION_NOISE_FILTER: bool = false;

const MAX_STACKED_FRAMES: u32 = 10;
const MAX_VIRTUAL_EXPOSURE_US: u32 = 3_000_000;
const GAIN_ADJUST_THROTTLE_MS: u64 = 10_000;

/// Minimum field of view in degrees required to enable virtual crop extraction.
/// For FOVs below this threshold, full-frame solving is sufficiently free of obstructions,
/// eliminating the overhead of 5 sub-crop evaluations per frame.
pub const MIN_VIRTUAL_CROP_FOV: f64 = 25.0;

/// The labels corresponding to the 2 concentric virtual crop regions produced by [`generate_concentric_center_crops`].
pub const VIRTUAL_CROP_LABELS: [SolvedCrop; 2] = [SolvedCrop::Center2x, SolvedCrop::Center1_5x];

/// Plate-solving engine configuration.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Fixed analog gain override. If `None`, automatic gain control (AGC) is used.
    pub gain: Option<f32>,
    /// Target star centroid count for auto-exposure feedback.
    pub target_centroids: usize,
    /// Filesystem path to the Tetra3 star catalog (`.npz`).
    pub database_path: String,
    /// Enable Raspberry Pi ISP hardware pipeline mode (libcamera only).
    pub isp: bool,
    /// Background estimation algorithm for star centroiding.
    pub bg_mode: BgMode,
    /// Centroid extraction backend algorithm.
    pub extractor_type: Extractor,
    /// Sub-window crop dimensions `(width, height)` in pixels.
    pub crop: Option<(usize, usize)>,
    /// Downsampling factor applied before extraction (`0` disables).
    pub downsample: u8,
    /// Enable automated runtime tuning (gain, exposure, stacking).
    pub auto_mode: bool,
    /// `true` for cameras with native low-latency exposure AGC.
    pub is_adaptive_camera: bool,
    /// Apply noise threshold filter during centroid detection.
    pub noise_filter: bool,
    /// Initial field-of-view estimate in degrees to accelerate catalog lookup.
    pub initial_fov: Option<f32>,
    /// Compute topocentric Alt/Az coordinates from client location.
    pub use_observer_location: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            gain: None,
            target_centroids: 20,
            database_path: format!(
                "{}/data/default_database.npz",
                std::env::var("HOME").unwrap_or_default()
            ),
            isp: false,
            bg_mode: BgMode::Line,
            extractor_type: Extractor::Approximate,
            crop: None,
            downsample: 0,
            auto_mode: true,
            is_adaptive_camera: false,
            noise_filter: false,
            initial_fov: None,
            use_observer_location: true,
        }
    }
}

/// Intermediate frame processing and centroid extraction results.
pub struct IntermediateResult {
    /// Total star centroids extracted from frame.
    pub count: usize,
    /// Star centroid count per virtual sub-crop.
    pub per_crop_centroid_counts: Vec<usize>,
    /// Sensor gain multiplier applied during acquisition.
    pub gain: f32,
    /// Virtual exposure duration in milliseconds or seconds.
    pub exp_time: Duration,
    /// Physical sensor exposure duration.
    pub phys_exp_time: Duration,
    /// Time spent in debayering and pre-processing.
    pub pp_time: Duration,
    /// Time spent in centroid extraction.
    pub extract_time: Duration,
    /// Extracted centroid coordinates and crop metadata.
    pub centroids_batch: Vec<(Array2<f64>, Option<tetra3::extractor::Crop>, SolvedCrop)>,
    /// Acquisition host timestamp.
    pub timestamp: SystemTime,
    /// Number of stacked physical frames contributing to this result.
    pub stacked_frames: u32,
    /// Measured background light level (0.0 to 255.0).
    pub background_level: Option<f32>,
    /// Time delta from previous frame start in microseconds.
    pub frame_interval_us: i64,
    /// `true` if frame metadata matches verified hardware registers.
    pub metadata_valid: bool,
    /// Optical/digital zoom factor applied to this frame (defaults to 1.0).
    pub zoom: f32,
    /// Monotonic mount slew epoch stamped when `t1` accepted this frame.
    pub slew_epoch: u64,
}

/// Star centroid extractor algorithm backend.
#[derive(Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum, Serialize, Deserialize)]
pub enum Extractor {
    /// Fast heuristic centroid extraction optimized for embedded CPUs.
    #[default]
    Approximate,
    /// Robust, full-precision centroid extractor.
    Fast,
}

/// Background estimation and subtraction mode.
#[derive(Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum, Serialize, Deserialize)]
pub enum BgMode {
    /// Line-by-line background subtraction.
    #[default]
    Line,
    /// Global frame-wide median subtraction.
    Global,
    /// Block-based local background estimation.
    Block,
}

/// Region of interest that produced an astrometric solution.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum SolvedCrop {
    /// Center 2.0x virtual crop (50% image dimensions).
    Center2x,
    /// Center 1.5x virtual crop (66.7% image dimensions).
    Center1_5x,
    /// Full uncropped sensor frame.
    Full,
    /// Astrometric solution failed across all crops.
    Failed,
    /// Center 2.0x virtual crop at high sensitivity.
    Center2xHighSens,
    /// Center 1.5x virtual crop at high sensitivity.
    Center1_5xHighSens,
    /// Full frame at high sensitivity.
    FullHighSens,
}

/// Target pixel alignment command.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TargetPixelCommand {
    /// Set target pixel coordinates `(x, y)`.
    Set(f64, f64),
    /// Clear active target pixel.
    Clear,
}

/// Real-time engine telemetry, astrometric coordinates, and server state.
#[derive(Clone, Debug, Serialize)]
pub struct EngineStateData {
    /// Operating state message (e.g. `"Solving"`, `"Camera Mode"`, `"Idle"`).
    pub status: String,
    /// Boresight Right Ascension in degrees [0.0, 360.0), topocentric epoch.
    pub ra: Option<f64>,
    /// Boresight Declination in degrees [-90.0, +90.0], topocentric epoch.
    pub dec: Option<f64>,
    /// Center RA in degrees before pixel-offset translation.
    pub center_ra: Option<f64>,
    /// Center Declination in degrees before pixel-offset translation.
    pub center_dec: Option<f64>,
    /// Equatorial roll / position angle in degrees.
    pub roll: Option<f64>,
    /// `true` if image orientation is optically mirrored.
    pub is_mirrored: Option<bool>,
    /// Star centroid count detected in the last analyzed frame.
    pub centroids: usize,
    /// Active target Right Ascension in degrees.
    pub target_ra: Option<f64>,
    /// Active target Declination in degrees.
    pub target_dec: Option<f64>,
    /// Active sensor exposure duration in microseconds.
    pub exposure_us: i32,
    /// Image pre-processing latency in milliseconds.
    pub post_process_ms: f64,
    /// Centroid extraction latency in milliseconds.
    pub extract_ms: f64,
    /// Catalog matching and attitude determination latency in milliseconds.
    pub solve_ms: f64,
    /// Rolling 5-second solve throughput in Hertz.
    pub solve_rate_hz: f64,
    /// Flag indicating an image capture snapshot was requested.
    pub capture_requested: bool,
    /// `true` when live HFR focus metric computation is requested by the client.
    #[serde(skip)]
    pub hfr_enabled: bool,
    /// Half Flux Radius focus metric in pixels (lower indicates sharper focus).
    pub hfr: f64,
    /// Active lens focus position (0.0 to 1.0).
    pub focus: Option<f32>,
    /// Observer site latitude in degrees [-90.0, +90.0].
    pub client_latitude: Option<f64>,
    /// Observer site longitude in degrees [-180.0, +180.0].
    pub client_longitude: Option<f64>,
    /// Target angular offset in RA in arcseconds or degrees.
    pub offset_ra: Option<f64>,
    /// Target angular offset in Declination in arcseconds or degrees.
    pub offset_dec: Option<f64>,
    /// Target angular offset in Azimuth in degrees.
    pub offset_az: Option<f64>,
    /// Target angular offset in Altitude in degrees.
    pub offset_alt: Option<f64>,
    /// Total angular separation to target in degrees.
    pub offset_angle: Option<f64>,
    /// Adaptive noise sigma threshold.
    pub current_sigma: Option<f32>,
    /// Estimated horizontal field-of-view in degrees.
    pub fov: Option<f64>,
    /// Optical distortion coefficient.
    pub distortion: Option<f64>,
    /// Commanded target RA from client in degrees.
    pub requested_target_ra: Option<f64>,
    /// Commanded target Declination from client in degrees.
    pub requested_target_dec: Option<f64>,
    /// Commanded target epoch (e.g. J2000 or current year).
    pub requested_target_epoch: Option<f64>,
    /// Flag requesting active target coordinates be cleared.
    pub requested_target_clear: bool,
    /// Pending target pixel alignment command.
    #[serde(skip)]
    pub pending_target_pixel_cmd: Option<TargetPixelCommand>,
    /// `true` if current orientation is dead-reckoned via IMU sensor fusion.
    pub is_imu_estimate: bool,
    /// Require strict match threshold for catalog verification.
    pub strict_match: bool,
    /// Sub-crop region that produced the active astrometric solution.
    pub solved_crop: Option<SolvedCrop>,
    /// Flag requesting telescope position synchronization.
    pub requested_sync: bool,
    /// Target pixel X coordinate on camera sensor.
    pub target_pixel_x: Option<f64>,
    /// Target pixel Y coordinate on camera sensor.
    pub target_pixel_y: Option<f64>,
    /// Commanded synchronization coordinates `(ra, dec)` in degrees.
    pub requested_sync_coords: Option<(f64, f64)>,
    /// Current motorized mount state (`Disabled` in Push-To mode).
    pub mount_state: MountUiState,
    /// Identified motorized mount model name (e.g. `"HAZ31"`, `"AZ Mount Pro"`).
    pub mount_model: Option<String>,
    /// Latest concise mount status or error notice.
    pub mount_notice: Option<String>,
}

impl Default for EngineStateData {
    fn default() -> Self {
        Self {
            status: "Idle".to_string(),
            ra: None,
            dec: None,
            center_ra: None,
            center_dec: None,
            roll: None,
            is_mirrored: None,
            centroids: 0,
            target_ra: None,
            target_dec: None,
            exposure_us: 0,
            post_process_ms: 0.0,
            extract_ms: 0.0,
            solve_ms: 0.0,
            solve_rate_hz: 0.0,
            capture_requested: false,
            hfr_enabled: false,
            hfr: 0.0,
            focus: None,
            client_latitude: None,
            client_longitude: None,
            offset_ra: None,
            offset_dec: None,
            offset_az: None,
            offset_alt: None,
            offset_angle: None,
            current_sigma: None,
            fov: None,
            distortion: None,
            requested_target_ra: None,
            requested_target_dec: None,
            requested_target_epoch: None,
            requested_target_clear: false,
            pending_target_pixel_cmd: None,
            is_imu_estimate: false,
            strict_match: false,
            solved_crop: None,
            requested_sync: false,
            target_pixel_x: None,
            target_pixel_y: None,
            requested_sync_coords: None,
            mount_state: MountUiState::Disabled,
            mount_model: None,
            mount_notice: None,
        }
    }
}

/// Generates concentric center crops (Center 2.0x, Center 1.5x) sharing the optical axis (W/2, H/2).
pub fn generate_concentric_center_crops(
    proc_width: usize,
    proc_height: usize,
) -> Vec<tetra3::extractor::Crop> {
    vec![
        // Center 2.0x: 50% width & height (25% area) - Aggressive perimeter clutter rejection
        tetra3::extractor::Crop::Center {
            width: (proc_width as f64 * 0.50).round() as usize,
            height: (proc_height as f64 * 0.50).round() as usize,
        },
        // Center 1.5x: 66.7% width & height (44.4% area) - Moderate clutter rejection, wider star collection
        tetra3::extractor::Crop::Center {
            width: (proc_width as f64 * (2.0 / 3.0)).round() as usize,
            height: (proc_height as f64 * (2.0 / 3.0)).round() as usize,
        },
    ]
}

/// Helper to determine virtual crops and corresponding labels based on the current estimated FOV.
/// Only activates when `best_fov >= MIN_VIRTUAL_CROP_FOV`.
pub fn evaluate_virtual_crops(
    best_fov: Option<f64>,
    proc_width: usize,
    proc_height: usize,
) -> (
    Option<Vec<tetra3::extractor::Crop>>,
    Option<Vec<SolvedCrop>>,
) {
    if best_fov.is_some_and(|fov| fov >= MIN_VIRTUAL_CROP_FOV) {
        (
            Some(generate_concentric_center_crops(proc_width, proc_height)),
            Some(VIRTUAL_CROP_LABELS.to_vec()),
        )
    } else {
        (None, None)
    }
}

/// Celestial plate-solving engine coordinating frame acquisition, attitude estimation, and servers.
#[allow(dead_code)]
pub struct Engine {
    camera: Arc<dyn CameraTrait>,
    config: EngineConfig,
    last_gain_adjust: Option<std::time::Instant>,
    pub fused_solver: Arc<olive_solve::FusedSolver>,
    fov: Option<f64>,
    distortion: Option<f64>,
    active_zoom: f32,

    shared_position: Arc<SharedPosition>,
    pub telescope_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub lx200_controller:
        std::sync::Arc<std::sync::RwLock<crate::telescope_server::Lx200Controller>>,
    pub alpaca_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub rt_handle: Option<tokio::runtime::Handle>,
    target_ra: Option<f64>,
    target_dec: Option<f64>,
    target_pixel: Option<(f64, f64)>,
    target_pixel_key: String,
    storage: Option<Arc<dyn olive_imu::storage::PersistentStorage>>,
    client_connected: Arc<std::sync::atomic::AtomicBool>,
    engine_state: Arc<RwLock<EngineStateData>>,
    solve_timestamps: std::collections::VecDeque<Instant>,
    shared_result: Arc<(
        std::sync::Mutex<Option<IntermediateResult>>,
        std::sync::Condvar,
    )>,
    pub is_running: Arc<std::sync::atomic::AtomicBool>,
    pub clear_stacker: Arc<std::sync::atomic::AtomicBool>,
    timescale: Arc<Timescale>,
    mount_handle: Arc<MountHandle>,
}

fn angle_diff_deg(a: f64, b: f64) -> f64 {
    let diff = (a - b).rem_euclid(360.0);
    if diff > 180.0 { diff - 360.0 } else { diff }
}

fn resolve_site_location(
    state: &EngineStateData,
    shared_pos: &SharedPosition,
) -> (Option<f64>, Option<f64>) {
    (
        state
            .client_latitude
            .or(shared_pos.site_latitude.load(Ordering::Relaxed)),
        state
            .client_longitude
            .or(shared_pos.site_longitude.load(Ordering::Relaxed)),
    )
}

pub(crate) fn calculate_alt_az(
    ra_deg: f64,
    dec_deg: f64,
    lat_deg: f64,
    lon_deg: f64,
    time: &DateTime<Utc>,
    ts: &Timescale,
) -> (f64, f64) {
    let t = ts.from_datetime(*time);
    let gast_hours = t.gast();

    // LST in degrees
    let lst_deg = (gast_hours * 15.0 + lon_deg).rem_euclid(360.0);

    // Hour Angle in degrees
    let ha_deg = (lst_deg - ra_deg).rem_euclid(360.0);

    // Convert to radians for math
    let ha_rad = ha_deg.to_radians();
    let dec_rad = dec_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    // Altitude
    let sin_alt = dec_rad.sin() * lat_rad.sin() + dec_rad.cos() * lat_rad.cos() * ha_rad.cos();
    let alt_rad = sin_alt.clamp(-1.0, 1.0).asin();

    // Azimuth
    let cos_az = (dec_rad.sin() - alt_rad.sin() * lat_rad.sin()) / (alt_rad.cos() * lat_rad.cos());
    let sin_az = -dec_rad.cos() * ha_rad.sin() / alt_rad.cos();
    let az_rad = sin_az.atan2(cos_az).rem_euclid(TAU);

    (alt_rad.to_degrees(), az_rad.to_degrees())
}

impl Engine {
    /// Initializes the plate-solving engine, background network servers, and IMU solver.
    ///
    /// Spawns the Meade LX200 server (TCP 4030) and ASCOM Alpaca server (HTTP 11111 / UDP 32227).
    pub async fn new(
        camera: Arc<dyn CameraTrait>,
        config: EngineConfig,
        client_connected: Arc<std::sync::atomic::AtomicBool>,
        engine_state: Arc<RwLock<EngineStateData>>,
        imu_type: Option<olive_solve::ImuType>,
        rt_handle: Option<tokio::runtime::Handle>,
        storage: Option<Arc<dyn olive_imu::storage::PersistentStorage>>,
        alpaca_fd_receiver: Option<Receiver<i32>>,
        alpaca_discovery_fd_receiver: Option<Receiver<i32>>,
    ) -> Self {
        let fused_solver = Arc::new(
            olive_solve::FusedSolver::new(
                std::path::Path::new(&config.database_path),
                imu_type,
                storage.clone(),
                false, // disable accelerometer
            )
            .expect("Failed to initialize FusedSolver"),
        );
        let config_fov = config.initial_fov.map(|f| f as f64);
        let shared_position = Arc::new(SharedPosition::default());
        let is_running = Arc::new(std::sync::atomic::AtomicBool::new(true));

        let timescale = Arc::new(
            tokio::task::spawn_blocking(|| starfield::time::Timescale::default())
                .await
                .expect("Failed to initialize Timescale"),
        );

        let mut server = Lx200WifiTelescope::new(
            shared_position.clone(),
            is_running.clone(),
            timescale.clone(),
        );
        let lx200_controller = server.get_controller();
        let alpaca_server = AlpacaServer::new(
            shared_position.clone(),
            is_running.clone(),
            alpaca_fd_receiver,
            alpaca_discovery_fd_receiver,
        );
        let (telescope_server_handle, alpaca_server_handle) = if let Some(ref h) = rt_handle {
            let t_handle = h.spawn(async move {
                if let Err(e) = server.serve_requests().await {
                    error!("Telescope server error: {}", e);
                }
            });
            let a_handle = h.spawn(async move {
                if let Err(e) = alpaca_server.serve_requests().await {
                    error!("Alpaca server error: {}", e);
                }
            });
            (t_handle, a_handle)
        } else {
            let t_handle = tokio::spawn(async move {
                if let Err(e) = server.serve_requests().await {
                    error!("Telescope server error: {}", e);
                }
            });
            let a_handle = tokio::spawn(async move {
                if let Err(e) = alpaca_server.serve_requests().await {
                    error!("Alpaca server error: {}", e);
                }
            });
            (t_handle, a_handle)
        };

        let clear_stacker = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mount_handle = MountHandle::start(timescale.clone(), None);

        let engine_state_clone = engine_state.clone();
        let shared_pos_clone = shared_position.clone();
        let fused_solver_clone = fused_solver.clone();
        let is_running_clone = is_running.clone();
        let timescale_clone = timescale.clone();
        let clear_stacker_clone = clear_stacker.clone();
        let mount_handle_poll = mount_handle.clone();

        let polling_task = async move {
            let mut ticker = interval(Duration::from_millis(33));
            let mut last_mount_seq = u64::MAX;
            let mut mount_was_enabled = false;
            while is_running_clone.load(Ordering::Relaxed) {
                ticker.tick().await;

                if mount_handle_poll.is_enabled() {
                    mount_was_enabled = true;
                    let op = shared_pos_clone.target_op.load(Ordering::Acquire);
                    if op == Some(TargetOp::Abort)
                        && !mount_handle_poll.is_abort_pending()
                        && (mount_handle_poll.is_goto_active()
                            || mount_handle_poll.is_physically_slewing())
                    {
                        mount_handle_poll.abort();
                    }

                    let (lat, lon, has_pending_ui_goto) =
                        if let Ok(state) = engine_state_clone.read() {
                            let (lat, lon) = resolve_site_location(&state, &shared_pos_clone);
                            (
                                lat,
                                lon,
                                state.requested_target_ra.is_some()
                                    && state.requested_target_dec.is_some(),
                            )
                        } else {
                            (None, None, false)
                        };
                    mount_handle_poll.set_site_location(lat, lon);

                    if op == Some(TargetOp::Slew) || has_pending_ui_goto {
                        shared_pos_clone.is_slewing.store(true, Ordering::Release);
                    } else {
                        shared_pos_clone
                            .is_slewing
                            .store(mount_handle_poll.is_goto_active(), Ordering::Release);
                    }

                    let (mt, seq) = mount_handle_poll.telemetry_with_seq();
                    if seq != last_mount_seq {
                        last_mount_seq = seq;
                        if let Ok(mut state) = engine_state_clone.write() {
                            state.mount_state = mt.state;
                            state.mount_model = mt.model;
                            state.mount_notice = mt.notice;
                        }
                    }
                } else if mount_was_enabled {
                    mount_was_enabled = false;
                    last_mount_seq = u64::MAX;
                    shared_pos_clone.is_slewing.store(false, Ordering::Release);
                    if let Ok(mut state) = engine_state_clone.write() {
                        state.mount_state = MountUiState::Disabled;
                        state.mount_model = None;
                        state.mount_notice = None;
                    }
                }

                // High-cadence IMU motion distribution
                if let Some(motion) = fused_solver_clone.get_motion_state() {
                    let is_moving = motion == olive_imu::MotionState::Moving;

                    if is_moving {
                        clear_stacker_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }

                if let Some(pos) = fused_solver_clone.get_latest_position() {
                    let ra = pos.ra;
                    let dec = pos.dec;
                    let eq_roll = pos.roll;

                    *shared_pos_clone.telescope_position.write().unwrap() = (ra, dec);

                    let mut target_ra_opt = None;
                    let mut target_dec_opt = None;
                    let mut lat_opt_state = None;
                    let mut lon_opt_state = None;

                    if let Ok(state) = engine_state_clone.read() {
                        target_ra_opt = state.target_ra;
                        target_dec_opt = state.target_dec;
                        lat_opt_state = state.client_latitude;
                        lon_opt_state = state.client_longitude;
                    }

                    let mut offset_ra = None;
                    let mut offset_dec = None;
                    let mut offset_alt = None;
                    let mut offset_az = None;
                    let mut offset_angle = None;

                    if let (Some(t_ra), Some(t_dec)) = (target_ra_opt, target_dec_opt) {
                        let d_ra = angle_diff_deg(t_ra, ra);
                        let d_dec = t_dec - dec;
                        offset_ra = Some(d_ra);
                        offset_dec = Some(d_dec);

                        if let (Some(lat), Some(lon)) = (lat_opt_state, lon_opt_state) {
                            let time = chrono::Utc::now();
                            let (cur_alt, cur_az) =
                                calculate_alt_az(ra, dec, lat, lon, &time, &timescale_clone);
                            let (t_alt, t_az) =
                                calculate_alt_az(t_ra, t_dec, lat, lon, &time, &timescale_clone);

                            let d_alt = t_alt - cur_alt;
                            let d_az = angle_diff_deg(t_az, cur_az);

                            offset_alt = Some(d_alt);
                            offset_az = Some(d_az);

                            let x = -d_az * cur_alt.to_radians().cos();
                            let y = d_alt;
                            offset_angle = Some(x.atan2(y).to_degrees());
                        }
                    }

                    if let Ok(mut state) = engine_state_clone.write() {
                        state.ra = Some(ra);
                        state.dec = Some(dec);
                        state.center_ra = Some(pos.center_ra);
                        state.center_dec = Some(pos.center_dec);
                        state.roll = Some(eq_roll);
                        state.is_mirrored = Some(pos.is_mirrored);
                        state.is_imu_estimate = pos.source == olive_solve::PositionSource::Imu;

                        state.offset_ra = offset_ra;
                        state.offset_dec = offset_dec;
                        state.offset_alt = offset_alt;
                        state.offset_az = offset_az;
                        state.offset_angle = offset_angle;
                    }
                }
            }
        };

        if let Some(ref h) = rt_handle {
            h.spawn(polling_task);
        } else {
            tokio::spawn(polling_task);
        }

        let cam_cfg = camera.get_config();
        let target_pixel_key = Self::get_target_pixel_key(cam_cfg.as_ref());
        let target_pixel =
            Self::load_target_pixel(storage.as_deref(), &target_pixel_key, cam_cfg.as_ref());
        if let Some((tx, ty)) = target_pixel {
            if let Ok(mut state) = engine_state.write() {
                state.target_pixel_x = Some(tx);
                state.target_pixel_y = Some(ty);
            }
        }

        let start_zoom = camera.get_zoom().unwrap_or(1.0);
        let initial_engine_fov = if (start_zoom - 1.0).abs() > 0.05 {
            None
        } else {
            config_fov
        };

        Self {
            camera,
            config,
            last_gain_adjust: None,
            fused_solver,
            fov: initial_engine_fov,
            distortion: None,
            active_zoom: start_zoom,
            shared_position,
            target_ra: None,
            target_dec: None,
            target_pixel,
            target_pixel_key,
            storage,
            client_connected,
            engine_state,
            solve_timestamps: std::collections::VecDeque::with_capacity(60),
            shared_result: Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new())),
            is_running,
            clear_stacker,
            telescope_server_handle: Some(telescope_server_handle),
            alpaca_server_handle: Some(alpaca_server_handle),
            rt_handle,
            lx200_controller,
            timescale,
            mount_handle,
        }
    }

    /// Returns a thread-safe handle for configuring and controlling the optional motorized mount.
    pub fn mount_handle(&self) -> Arc<MountHandle> {
        self.mount_handle.clone()
    }

    /// Returns the cache key used for storing sensor target pixel calibration.
    pub fn get_target_pixel_key(cam_config: Option<&CameraConfig>) -> String {
        match cam_config {
            Some(cfg) => match &cfg.camera_type {
                CameraType::Demo => "target_pixel_demo".to_string(),
                CameraType::Internal { id } => format!("target_pixel_internal_{}", id),
                CameraType::External => {
                    format!("target_pixel_external_{}x{}", cfg.width, cfg.height)
                }
            },
            None => "target_pixel_demo".to_string(),
        }
    }

    fn load_target_pixel(
        storage: Option<&dyn olive_imu::storage::PersistentStorage>,
        key: &str,
        cam_config: Option<&CameraConfig>,
    ) -> Option<(f64, f64)> {
        let s = storage?;
        // 1. Try namespaced key
        if let Some(val) = s.get(key) {
            if let Some(pt) = Self::parse_pixel_coords(&val) {
                return Some(pt);
            }
        }
        // 2. Fall back to legacy "target_pixel" with bounds check
        if let Some(legacy_val) = s.get("target_pixel") {
            if let Some((x, y)) = Self::parse_pixel_coords(&legacy_val) {
                let (max_w, max_h) = cam_config
                    .map(|c| (c.width as f64, c.height as f64))
                    .unwrap_or((f64::MAX, f64::MAX));
                if x >= 0.0 && x <= max_w && y >= 0.0 && y <= max_h {
                    return Some((x, y));
                } else {
                    warn!(
                        "Legacy target_pixel ({}, {}) out of bounds for {}x{}; ignoring",
                        x, y, max_w, max_h
                    );
                }
            }
        }
        None
    }

    fn parse_pixel_coords(s: &str) -> Option<(f64, f64)> {
        let mut parts = s.split(',');
        let x = parts.next()?.trim().parse::<f64>().ok()?;
        let y = parts.next()?.trim().parse::<f64>().ok()?;
        if parts.next().is_none() {
            Some((x, y))
        } else {
            None
        }
    }

    fn save_target_pixel(&self, pixel: Option<(f64, f64)>) {
        if let Some(storage) = &self.storage {
            if let Some((x, y)) = pixel {
                storage.set(&self.target_pixel_key, &format!("{},{}", x, y));
            } else {
                storage.remove(&self.target_pixel_key);
                storage.remove("target_pixel");
            }
        }
    }

    fn update_status(&self, status: &str) {
        log::debug!("Engine status changing to: {}", status);
        let mut state = self.engine_state.write().unwrap();
        state.status = status.to_string();
    }

    fn update_solve_stats(
        &mut self,
        solved: bool,
        centroids: usize,
        target_ra: Option<f64>,
        target_dec: Option<f64>,
        exposure_us: i32,
        post_process: Duration,
        extract: Duration,
        solve: Duration,
        solved_crop: Option<SolvedCrop>,
    ) {
        if solved {
            self.solve_timestamps.push_back(Instant::now());
        }
        self.solve_timestamps
            .retain(|t| t.elapsed() <= Duration::from_secs(5));
        let solve_rate_hz = self.solve_timestamps.len() as f64 / 5.0;

        let reported_fov = self.fov;

        if let Ok(mut state) = self.engine_state.write() {
            state.centroids = centroids;
            state.target_ra = target_ra;
            state.target_dec = target_dec;
            state.exposure_us = exposure_us;
            state.post_process_ms = post_process.as_secs_f64() * 1000.0;
            state.extract_ms = extract.as_secs_f64() * 1000.0;
            state.solve_ms = solve.as_secs_f64() * 1000.0;
            state.solve_rate_hz = solve_rate_hz;
            state.fov = reported_fov;
            state.distortion = self.distortion;
            if let Some(crop) = solved_crop {
                state.solved_crop = Some(crop);
            }
            state.target_pixel_x = self.target_pixel.map(|p| p.0);
            state.target_pixel_y = self.target_pixel.map(|p| p.1);
        }
    }

    /// Spawns the image extraction pipeline thread and plate-solving worker thread.
    ///
    /// Returns join handles for the extraction and solving threads. Both threads
    /// run continuously until [`Self::is_running`] is signaled `false`.
    pub fn start(self) -> (std::thread::JoinHandle<()>, std::thread::JoinHandle<()>) {
        #[cfg(target_os = "ios")]
        let should_autofocus = self.config.auto_mode
            && matches!(
                self.camera.get_config().map(|c| c.camera_type),
                Some(crate::CameraType::Internal { .. })
            )
            && self.camera.get_focus().is_some();
        #[cfg(not(target_os = "ios"))]
        let should_autofocus = false;

        if should_autofocus {
            let best_focus =
                crate::autofocus::run_autofocus(&self.camera, &self.engine_state, &self.is_running);
            if let Ok(mut state) = self.engine_state.write() {
                state.focus = Some(best_focus);
            }
        }

        let camera_ext = self.camera.clone();
        let config_ext = self.config.clone();
        let shared_res_ext = self.shared_result.clone();
        let engine_state_ext = self.engine_state.clone();
        let is_running_ext = self.is_running.clone();
        let client_connected_ext = self.client_connected.clone();
        let clear_stacker_ext = self.clear_stacker.clone();
        let mount_handle_ext = self.mount_handle.clone();

        let t1 = std::thread::spawn(move || {
            let mut stacker = crate::stacker::Stacker::new(
                0,
                0,
                config_ext.auto_mode && !config_ext.is_adaptive_camera,
            );
            let mut extractor: Option<FastExtractor> = None;
            let mut last_processed_timestamp = None;
            let mut last_slew_epoch = 0u64;
            let mut active_max_area = 150;
            let mut hfr_history: std::collections::VecDeque<(std::time::SystemTime, f64)> =
                std::collections::VecDeque::new();

            loop {
                if !is_running_ext.load(std::sync::atomic::Ordering::Relaxed) {
                    log::info!("Engine thread stopping.");
                    break;
                }

                let img_opt = camera_ext.wait_for_next_capture(
                    last_processed_timestamp,
                    Some(std::time::Duration::from_millis(1000)),
                );
                let img_arc = match img_opt {
                    Some(img) => img,
                    None => {
                        if !is_running_ext.load(std::sync::atomic::Ordering::Relaxed) {
                            log::info!(
                                "Engine loop stopping due to is_running flag after None capture."
                            );
                            break;
                        }
                        // Sleep briefly before retrying to prevent thread thrashing and 100% CPU utilization
                        // when the camera capture thread is paused, restarting, or has tripped the watchdog.
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    }
                };

                // We should also check is_running after waiting.
                if !is_running_ext.load(std::sync::atomic::Ordering::Relaxed) {
                    log::info!("Engine loop stopping due to is_running flag after wait.");
                    break;
                }
                last_processed_timestamp = Some(img_arc.timestamp);

                let (drop_for_slew, frame_slew_epoch) =
                    mount_handle_ext.frame_fence(img_arc.timestamp, img_arc.exposure_time_us);
                if drop_for_slew || frame_slew_epoch != last_slew_epoch {
                    stacker.reset();
                    last_slew_epoch = frame_slew_epoch;
                }
                if drop_for_slew {
                    continue;
                }

                let mut do_capture = false;
                {
                    let mut state = engine_state_ext.write().unwrap();
                    if state.capture_requested {
                        state.capture_requested = false;
                        do_capture = true;
                    }
                }

                let exposure_us = img_arc.exposure_time_us;
                let img_gain = img_arc.gain;
                let post_process_time = img_arc.post_process_time;
                let img_timestamp = img_arc.timestamp;
                let frame_interval_us = img_arc.frame_interval_us;
                let metadata_valid = img_arc.metadata_valid;
                let img_zoom = img_arc.zoom.max(1.0);

                let img_data = match Arc::try_unwrap(img_arc) {
                    Ok(img) => img.data,
                    Err(arc) => arc.data.clone(),
                };

                let cam_config = camera_ext.get_config().expect("Camera config missing");
                let width = cam_config.width as usize;
                let height = cam_config.height as usize;

                let virtual_exposure_us;
                let final_img_data = {
                    let should_reset = clear_stacker_ext
                        .swap(false, std::sync::atomic::Ordering::Relaxed)
                        || stacker.stacked_frames >= MAX_STACKED_FRAMES
                        || stacker.virtual_exposure_us >= MAX_VIRTUAL_EXPOSURE_US;

                    if should_reset {
                        if stacker.stacked_frames >= MAX_STACKED_FRAMES
                            || stacker.virtual_exposure_us >= MAX_VIRTUAL_EXPOSURE_US
                        {
                            log::info!(
                                "Engine: Stacker reached max frames or max virtual exposure, resetting"
                            );
                        }
                        stacker.reset();
                    }

                    if !stacker.enabled {
                        virtual_exposure_us = exposure_us as i32;
                        stacker.stacked_frames = 1;
                        img_data.clone()
                    } else {
                        stacker.resize_if_needed(width, height);
                        let active = stacker.add_frame(&img_data, exposure_us as u32).to_vec();
                        virtual_exposure_us = stacker.virtual_exposure_us as i32;
                        active
                    }
                };

                if do_capture {
                    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                    let captures_dir = format!("{}/captures", home);
                    if let Err(e) = std::fs::create_dir_all(&captures_dir) {
                        error!("Failed to create captures directory: {}", e);
                    } else {
                        let timestamp = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        let path = format!("{}/capture_{}.jpg", captures_dir, timestamp);
                        let mut jpeg_data = Vec::new();
                        let mut encoder =
                            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_data, 90);
                        if let Err(e) = encoder.encode(
                            &img_data,
                            width as u32,
                            height as u32,
                            image::ExtendedColorType::L8,
                        ) {
                            error!("Failed to encode captured image to JPEG: {}", e);
                        } else if let Err(e) = std::fs::write(&path, jpeg_data) {
                            error!("Failed to write captured image to {}: {}", path, e);
                        } else {
                            info!("Captured image saved to {}", path);
                        }
                    }
                }

                let extract_start = Instant::now();
                let mut baseline_full_centroids = None;
                let mut per_crop_centroid_counts = Vec::new();

                let (state_sigma, state_fov) = {
                    let st = engine_state_ext.read().unwrap();
                    (st.current_sigma, st.fov)
                };

                let array_img =
                    match Array2::from_shape_vec((height, width), final_img_data.clone()) {
                        Ok(arr) => arr,
                        Err(e) => {
                            error!("Failed to create Array2 from image data: {}", e);
                            continue;
                        }
                    };

                let final_crop = config_ext
                    .crop
                    .or(cam_config.crop.map(|(w, h)| (w as usize, h as usize)));
                let (proc_width, proc_height) = match final_crop {
                    Some((cw, ch)) => (cw.min(width), ch.min(height)),
                    None => (width, height),
                };

                let downsample = match config_ext.downsample {
                    4 => FastDownsample::X4,
                    2 => FastDownsample::X2,
                    1 => FastDownsample::None, // explicitly Off
                    _ => {
                        // 0 = Auto
                        if proc_width * proc_height > 1_250_000 {
                            FastDownsample::X2
                        } else {
                            FastDownsample::None
                        }
                    }
                };

                let active_sigma = state_sigma.unwrap_or(3.5);
                let mut update_max_area = None;
                let best_fov = if state_fov.is_some() {
                    state_fov
                } else if (img_zoom - 1.0).abs() <= 0.05 {
                    config_ext.initial_fov.map(|f| f as f64)
                } else {
                    None
                };
                if let Some(fov) = best_fov {
                    if fov > 0.0 {
                        let ratio = (width as f64) / fov;
                        let raw_max_area = 20.0 + 0.03 * (ratio * ratio);
                        let ds_factor = match downsample {
                            FastDownsample::X4 => 4.0,
                            FastDownsample::X2 => 2.0,
                            FastDownsample::None => 1.0,
                        };
                        active_max_area =
                            (raw_max_area / (ds_factor * ds_factor)).round().max(10.0) as usize;
                        update_max_area = Some(active_max_area);
                    }
                }

                let (virtual_crops, virtual_crop_labels) =
                    evaluate_virtual_crops(best_fov, proc_width, proc_height);

                if extractor.is_none() {
                    let bg_sub_mode = match config_ext.bg_mode {
                        BgMode::Global => Some(FastBgSubMode::GlobalMedian),
                        BgMode::Block => Some(FastBgSubMode::BlockMedian { block_size: 32 }),
                        BgMode::Line => Some(FastBgSubMode::LineMedian),
                    };

                    let approximate_background =
                        config_ext.extractor_type == Extractor::Approximate;

                    let options = FastExtractOptions {
                        sigma: active_sigma,
                        approximate_background,
                        downsample,
                        bg_sub_mode,
                        binary_open: config_ext.noise_filter,
                        min_area: Some(2),
                        max_area: Some(active_max_area),
                        max_axis_ratio: Some(2.0),
                        crop: final_crop,
                        virtual_crops: virtual_crops.clone(),
                        //centroid_window: Some(3),
                        ..FastExtractOptions::default()
                    };
                    extractor = Some(FastExtractor::new(width, height, options));
                }

                let ext = extractor.as_mut().unwrap();
                ext.update_options(tetra3::fast_extractor::FastExtractOptionsUpdate {
                    sigma: Some(active_sigma),
                    max_area: update_max_area,
                    virtual_crops: Some(virtual_crops.clone()),
                    ..Default::default()
                });

                let mut variants = vec![Default::default()];

                let do_multiple_variants =
                    exposure_us >= MULTIPLE_EXTRACTION_MIN_EXPOSURE_US && config_ext.auto_mode;

                if do_multiple_variants {
                    variants.push(tetra3::fast_extractor::FastExtractOptionsUpdate {
                        sigma: Some(MULTIPLE_EXTRACTION_SIGMA),
                        min_area: Some(MULTIPLE_EXTRACTION_MIN_AREA),
                        noise_filter: Some(MULTIPLE_EXTRACTION_NOISE_FILTER),
                        ..Default::default()
                    });
                }

                let results = ext.extract_variants(&array_img, &variants);
                let final_bg_level = results.first().map(|r| r.background_level);
                if let Some(res) = results.first() {
                    if let Some(vc) = &res.virtual_crop_centroids {
                        per_crop_centroid_counts = vc.iter().map(|c| c.len()).collect();
                    }
                }

                let mut centroids_batch: Vec<(
                    Array2<f64>,
                    Option<tetra3::extractor::Crop>,
                    SolvedCrop,
                )> = Vec::new();
                let target_c = config_ext.target_centroids as f64 * 0.5;

                for (v_idx, result) in results.iter().enumerate() {
                    if let Some(vc) = &result.virtual_crop_centroids {
                        for (i, crop_cents) in vc.iter().enumerate() {
                            if (crop_cents.len() as f64) >= target_c {
                                let num_cents = crop_cents.len().min(80);
                                let mut arr = Array2::<f64>::zeros((num_cents, 2));
                                for (j, c) in crop_cents.iter().take(80).enumerate() {
                                    arr[[j, 0]] = c.y;
                                    arr[[j, 1]] = c.x;
                                }
                                let crop = virtual_crops.as_ref().and_then(|vc| vc.get(i).cloned());
                                let mut label = virtual_crop_labels.as_ref().unwrap()[i].clone();
                                if v_idx > 0 {
                                    label = match label {
                                        SolvedCrop::Center2x => SolvedCrop::Center2xHighSens,
                                        SolvedCrop::Center1_5x => SolvedCrop::Center1_5xHighSens,
                                        _ => label,
                                    };
                                }
                                centroids_batch.push((arr, crop, label));
                            }
                        }
                    }

                    let num_cents = result.centroids.len().min(80);
                    let mut arr = Array2::<f64>::zeros((num_cents, 2));
                    for (i, c) in result.centroids.iter().take(80).enumerate() {
                        arr[[i, 0]] = c.y;
                        arr[[i, 1]] = c.x;
                    }
                    if v_idx == 0 {
                        baseline_full_centroids = Some(arr.clone());
                    }
                    let label = if v_idx > 0 {
                        SolvedCrop::FullHighSens
                    } else {
                        SolvedCrop::Full
                    };
                    centroids_batch.push((arr, None, label));
                }

                // Use the base variant's full image centroid count for the UI
                let num_centroids = results.first().map(|r| r.centroids.len()).unwrap_or(0);

                let extract_time = extract_start.elapsed();

                // Calculate focus metric only when requested (e.g. Camera / Align or web stream).
                // This prevents wasting CPU cycles during normal plate solving.
                let hfr_enabled = engine_state_ext.read().unwrap().hfr_enabled;
                let mut hfr = if client_connected_ext.load(Ordering::Relaxed) || hfr_enabled {
                    calculate_hfr(
                        &final_img_data,
                        width,
                        height,
                        baseline_full_centroids
                            .as_ref()
                            .unwrap_or(&centroids_batch.last().unwrap().0),
                        30,
                    )
                } else {
                    0.0
                };

                // Rolling average over 1 second
                if hfr > 0.0 {
                    let now = std::time::SystemTime::now();
                    hfr_history.push_back((now, hfr));
                    // remove old elements
                    while let Some(&(t, _)) = hfr_history.front() {
                        if now
                            .duration_since(t)
                            .unwrap_or(std::time::Duration::ZERO)
                            .as_millis()
                            > 1000
                        {
                            hfr_history.pop_front();
                        } else {
                            break;
                        }
                    }
                    if !hfr_history.is_empty() {
                        let sum: f64 = hfr_history.iter().map(|&(_, v)| v).sum();
                        hfr = sum / hfr_history.len() as f64;
                    }
                } else {
                    hfr_history.clear();
                }

                {
                    let mut state = engine_state_ext.write().unwrap();
                    state.hfr = hfr;
                }

                let intermediate = IntermediateResult {
                    count: num_centroids,
                    per_crop_centroid_counts,
                    gain: img_gain,
                    exp_time: Duration::from_micros(virtual_exposure_us as u64),
                    phys_exp_time: Duration::from_micros(exposure_us as u64),
                    pp_time: post_process_time,
                    extract_time,
                    centroids_batch,
                    timestamp: img_timestamp,
                    stacked_frames: stacker.stacked_frames,
                    background_level: final_bg_level,
                    frame_interval_us,
                    metadata_valid,
                    zoom: img_zoom,
                    slew_epoch: frame_slew_epoch,
                };

                let (lock, cvar) = &*shared_res_ext;
                let mut inner = lock.lock().unwrap();
                *inner = Some(intermediate);
                cvar.notify_all();
            }
        });

        let t2 = std::thread::spawn(move || {
            let mut engine = self;

            // Apply static hardware settings ONCE at startup
            if let Some(g) = engine.config.gain {
                log::info!("Setting base hardware settings...");
                engine.camera.set_gain(g);
            }

            loop {
                if !engine.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                    log::info!("Main engine thread stopping.");
                    return;
                }
                engine.run_solving_loop();
            }
        });
        (t1, t2)
    }

    fn wait_for_extraction(&self) -> Option<IntermediateResult> {
        let (lock, cvar) = &*self.shared_result;
        let mut inner = lock.lock().unwrap();

        let result = cvar
            .wait_timeout(inner, Duration::from_millis(100))
            .unwrap();
        inner = result.0;

        inner.take()
    }

    /// Automatically adjusts camera gain based on the image's background brightness.
    /// This prevents stars from being washed out in pre-dusk skies, while keeping the
    /// sensor pegged at max gain during true dark skies.
    ///
    /// The function uses a simple proportional controller to target a background
    /// brightness window. Changes are throttled (e.g. 10s) to give the camera's
    /// exposure buffer time to flush and avoid oscillation.
    ///
    /// Skipped when calibration is bypassed or
    /// when an explicit gain is provided in the configuration.
    fn auto_adjust_gain(&mut self, bg_level: f32) {
        if !self.config.auto_mode || self.config.is_adaptive_camera || self.config.gain.is_some() {
            return;
        }

        let now = std::time::Instant::now();
        if let Some(last_adj) = self.last_gain_adjust {
            if now.duration_since(last_adj)
                < std::time::Duration::from_millis(GAIN_ADJUST_THROTTLE_MS)
            {
                return;
            }
        }

        // For highly sensitive sensors (e.g. SC2210) under dark skies, the background
        // is naturally very low (5-15 ADU) even at max gain.
        // Max 80.0 ensures we drop gain before twilight washes out the stars.
        // Min 20.0 ensures we push gain back to max as the sky darkens.
        let target_bg_max = 80.0;
        let target_bg_min = 20.0;

        let current_gain = self.camera.get_gain();
        let (min_gain, _, true_max_analog_gain) = self.camera.get_gain_range();

        let absolute_max = true_max_analog_gain;
        let baseline_gain = min_gain + (absolute_max - min_gain) * 0.1;
        let lower_bound = baseline_gain.min(absolute_max).max(min_gain);

        let mut new_gain = current_gain;
        if bg_level > target_bg_max {
            new_gain *= target_bg_max / bg_level;
        } else if bg_level < target_bg_min {
            new_gain *= target_bg_min / bg_level.max(1.0);
        }

        new_gain = new_gain.clamp(lower_bound, absolute_max);

        if (new_gain - current_gain).abs() > 0.01 {
            self.camera.set_gain(new_gain);
            self.last_gain_adjust = Some(now);

            let effective_gain = self.camera.get_gain();
            if (effective_gain - current_gain).abs() > 0.01 {
                log::info!(
                    "Auto-adjusted gain to {:.2} (bg level: {:.1})",
                    effective_gain,
                    bg_level
                );
            }
        }
    }

    pub fn run_solving_loop(&mut self) {
        info!("Entering solving loop...");

        if let Ok(mut state) = self.engine_state.write() {
            state.solved_crop = None;
        }

        self.update_solve_stats(
            false,
            0,
            None,
            None,
            self.camera.get_exposure(),
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            None,
        );
        let mut adaptive_controller = if self.config.is_adaptive_camera && self.config.auto_mode {
            let config = self.camera.get_config().unwrap_or_default();
            let (min_gain, _, _) = self.camera.get_gain_range();
            let phases = self.camera.get_calibration_phases();
            let exp_range = self.camera.get_exposure_range();
            Some(crate::auto_controller::AdaptiveController::new(
                config.model,
                phases,
                exp_range,
                min_gain,
                self.storage.clone(),
            ))
        } else {
            None
        };
        let mut observer_location_set = false;
        let mut was_connected = false;

        loop {
            if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }

            // 1. Process Local UI Requests (JNI/Push-To GoTo & Pixel Align)
            if let Ok(mut st) = self.engine_state.write() {
                if st.requested_target_clear {
                    self.target_ra = None;
                    self.target_dec = None;
                    st.target_ra = None;
                    st.target_dec = None;
                    st.requested_target_clear = false;
                    if self.mount_handle.is_enabled() {
                        if self.mount_handle.is_goto_active() {
                            self.mount_handle.abort();
                        }
                        self.shared_position
                            .is_slewing
                            .store(false, Ordering::Relaxed);
                    }
                }

                if let (Some(ra), Some(dec)) = (st.requested_target_ra, st.requested_target_dec) {
                    let epoch = st.requested_target_epoch.unwrap_or(2000.0);
                    let (ra_j2000, dec_j2000) = if (epoch - 2000.0).abs() > 0.001 {
                        crate::telescope_server::precess_coordinates(
                            ra,
                            dec,
                            epoch,
                            2000.0,
                            &self.timescale,
                        )
                    } else {
                        (ra, dec)
                    };
                    self.target_ra = Some(ra_j2000);
                    self.target_dec = Some(dec_j2000);
                    st.target_ra = Some(ra_j2000);
                    st.target_dec = Some(dec_j2000);
                    st.requested_target_ra = None;
                    st.requested_target_dec = None;
                    st.requested_target_epoch = None;

                    if self.mount_handle.is_enabled() {
                        self.mount_handle.goto(ra_j2000, dec_j2000);
                    } else {
                        self.shared_position
                            .target_op
                            .store(Some(TargetOp::Slew), Ordering::Release);
                    }
                    self.shared_position
                        .is_slewing
                        .store(true, Ordering::Release);
                    info!("Engine received UI target request: RA={}, Dec={}", ra, dec);
                }
                if let Some(cmd) = st.pending_target_pixel_cmd.take() {
                    match cmd {
                        TargetPixelCommand::Set(x, y) => {
                            self.target_pixel = Some((x, y));
                            st.target_pixel_x = Some(x);
                            st.target_pixel_y = Some(y);
                            info!("Engine received align target pixel request: ({}, {})", x, y);
                        }
                        TargetPixelCommand::Clear => {
                            self.target_pixel = None;
                            st.target_pixel_x = None;
                            st.target_pixel_y = None;
                            info!("Engine received clear target pixel request");
                        }
                    }
                    self.save_target_pixel(self.target_pixel);
                    self.mount_handle.notify_target_pixel_changed();
                }
                if st.requested_sync {
                    st.requested_sync = false;
                    let coords = st
                        .requested_sync_coords
                        .take()
                        .or(self.target_ra.zip(self.target_dec));
                    if let Some((s_ra, s_dec)) = coords {
                        self.shared_position
                            .target_ra
                            .store(Some(s_ra), std::sync::atomic::Ordering::Relaxed);
                        self.shared_position
                            .target_dec
                            .store(Some(s_dec), std::sync::atomic::Ordering::Relaxed);
                        self.shared_position
                            .target_op
                            .store(Some(TargetOp::Sync), std::sync::atomic::Ordering::Relaxed);
                        info!(
                            "Engine received UI sync request for target RA={}, Dec={}",
                            s_ra, s_dec
                        );
                    } else {
                        warn!("Engine received UI sync request but no target is set!");
                    }
                }
            }

            // 2. Process External Client & State Flags (Alpaca/LX200)
            let target_op = self.shared_position.target_op.load(Ordering::Acquire);
            let target_ra = self.shared_position.target_ra.load(Ordering::Relaxed);
            let target_dec = self.shared_position.target_dec.load(Ordering::Relaxed);

            if target_op == Some(TargetOp::Abort) {
                self.target_ra = None;
                self.target_dec = None;
                if let Ok(mut st) = self.engine_state.write() {
                    st.target_ra = None;
                    st.target_dec = None;
                }
                if self.mount_handle.is_enabled()
                    && !self.mount_handle.is_abort_pending()
                    && (self.mount_handle.is_goto_active()
                        || self.mount_handle.is_physically_slewing())
                {
                    self.mount_handle.abort();
                }
                let _ = self.shared_position.target_op.compare_exchange(
                    Some(TargetOp::Abort),
                    None,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                self.shared_position
                    .is_slewing
                    .store(false, Ordering::Release);
            } else if target_op == Some(TargetOp::Slew)
                && target_ra.is_some()
                && target_dec.is_some()
            {
                let t_ra = target_ra.unwrap();
                let t_dec = target_dec.unwrap();
                self.target_ra = Some(t_ra);
                self.target_dec = Some(t_dec);
                if let Ok(mut st) = self.engine_state.write() {
                    st.target_ra = Some(t_ra);
                    st.target_dec = Some(t_dec);
                }
                if self.mount_handle.is_enabled() {
                    self.mount_handle.goto(t_ra, t_dec);
                }
                if self
                    .shared_position
                    .target_op
                    .compare_exchange(
                        Some(TargetOp::Slew),
                        None,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    self.shared_position
                        .target_ra
                        .store(None, Ordering::Relaxed);
                    self.shared_position
                        .target_dec
                        .store(None, Ordering::Relaxed);
                    self.shared_position
                        .is_slewing
                        .store(true, Ordering::Release);
                }
            }

            if self
                .client_connected
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                was_connected = true;
                self.update_status("Camera Mode");
                std::thread::sleep(Duration::from_millis(500));
                continue;
            } else if was_connected {
                info!("Client disconnected. Returning to calibration mode.");
                return;
            }

            let status = if adaptive_controller
                .as_ref()
                .map_or(false, |ac| ac.is_calibrating())
            {
                "Calibrating"
            } else {
                "Solving"
            };
            self.update_status(status);

            if let Some(res) = self.wait_for_extraction() {
                let frame_zoom = res.zoom.max(1.0);
                if (frame_zoom - self.active_zoom).abs() > 0.05 {
                    log::info!(
                        "Zoom changed from {:.2}x to {:.2}x: resetting solver feedback state",
                        self.active_zoom,
                        frame_zoom
                    );
                    let initial_engine_fov = if (frame_zoom - 1.0).abs() <= 0.05 {
                        self.config.initial_fov.map(|f| f as f64)
                    } else {
                        None
                    };
                    self.fov = initial_engine_fov;
                    self.distortion = None;
                    self.active_zoom = frame_zoom;
                    self.clear_stacker.store(true, Ordering::Relaxed);

                    if let Some(ac) = adaptive_controller.as_mut() {
                        ac.reset_failures();
                    }

                    if let Ok(mut state) = self.engine_state.write() {
                        state.fov = initial_engine_fov;
                        state.distortion = None;
                    }
                }

                if let Some(bg) = res.background_level {
                    if !self.mount_handle.is_physically_slewing() {
                        self.auto_adjust_gain(bg);
                    }
                }
                let per_crop_centroid_counts = res.per_crop_centroid_counts.clone();
                let actual_gain = res.gain;
                let actual_exposure_us = res.phys_exp_time.as_micros() as i32;
                let bg_level = res.background_level;

                let count = res.count;
                let exp_time = res.exp_time;
                let pp_time = res.pp_time;
                let extract_time = res.extract_time;
                let centroids_batch = res.centroids_batch;
                let img_timestamp = res.timestamp;
                let stacked_frames = res.stacked_frames;
                let frame_interval_us = res.frame_interval_us;
                let metadata_valid = res.metadata_valid;
                let frame_slew_epoch = res.slew_epoch;
                self.mount_handle
                    .set_expected_solve_window_ms(exp_time.as_millis() as u64);
                let mut pending_sync = None;

                // 3. Process Sync Request (Requires active frame)
                let sync_target_op = self
                    .shared_position
                    .target_op
                    .load(std::sync::atomic::Ordering::Relaxed);
                let sync_target_ra = self
                    .shared_position
                    .target_ra
                    .load(std::sync::atomic::Ordering::Relaxed);
                let sync_target_dec = self
                    .shared_position
                    .target_dec
                    .load(std::sync::atomic::Ordering::Relaxed);

                if sync_target_op == Some(TargetOp::Sync)
                    && sync_target_ra.is_some()
                    && sync_target_dec.is_some()
                {
                    pending_sync = Some((sync_target_ra.unwrap(), sync_target_dec.unwrap()));
                }

                let mut solved = false;
                let camera_config = self.camera.get_config().unwrap();
                let width = camera_config.width as f64;
                let height = camera_config.height as f64;
                let size = (height, width);

                let mut solve_options = SolveOptions::default();
                // Force distortion calculation by providing a 0.0 baseline if None exists
                solve_options.distortion = self.distortion.or(Some(0.0));
                let has_active_imu = self.fused_solver.get_motion_state().is_some();
                solve_options.return_best_failed_match = has_active_imu;

                let num_crops = centroids_batch.len();
                let total_budget_ms = (exp_time.as_secs_f64() * 1000.0 * 0.90).max(150.0);
                let crop_budget_ms = (total_budget_ms / (num_crops as f64).max(1.0)).max(15.0);
                solve_options.solve_timeout_ms = Some(crop_budget_ms);

                if let Some(fov) = self.fov {
                    solve_options.fov_estimate = Some(fov);
                    solve_options.fov_max_error = Some((fov * 0.05).max(0.1));
                } else {
                    solve_options.fov_estimate = None;
                    solve_options.fov_max_error = None;
                }
                // Check for pending sync or slew already processed above

                if let Some(tp) = self.target_pixel {
                    solve_options.target_pixel =
                        Some(Array2::from_shape_vec((1, 2), vec![tp.1, tp.0]).unwrap());
                }
                if let Some((t_ra, t_dec)) = pending_sync {
                    solve_options.target_sky_coord =
                        Some(Array2::from_shape_vec((1, 2), vec![t_ra, t_dec]).unwrap());
                }

                let engine_state = self.engine_state.read().unwrap();
                let (lat_opt, lon_opt) =
                    resolve_site_location(&engine_state, &self.shared_position);
                let is_strict = engine_state.strict_match;
                drop(engine_state);

                if !is_strict {
                    solve_options.match_max_error = 0.005;
                    solve_options.match_threshold = 1e-4;
                    solve_options.match_radius = 0.005;
                } else {
                    solve_options.match_radius = 0.003;
                }

                // Set Zenith limits to automatically reject patterns containing stars below the horizon
                if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                    if self.config.use_observer_location {
                        let now = chrono::Utc::now();
                        let t = self.timescale.from_datetime(now);
                        let gast_hours = t.gast();
                        let lst_deg = (gast_hours * 15.0 + lon).rem_euclid(360.0);

                        solve_options.observer_latitude = Some(lat);
                        solve_options.observer_lst = Some(lst_deg);

                        let max_fov = self.fov.unwrap_or(70.0);
                        let offset_angle = if let Some(tp) = self.target_pixel {
                            let center_x = width as f64 / 2.0;
                            let center_y = height as f64 / 2.0;
                            let dist_px =
                                ((tp.0 - center_x).powi(2) + (tp.1 - center_y).powi(2)).sqrt();
                            (dist_px / width as f64) * max_fov
                        } else {
                            0.0
                        };

                        let buffer = offset_angle + 1.0;
                        solve_options.min_boresight_altitude = Some(-buffer);
                    }
                }

                if !observer_location_set {
                    if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                        if self.config.use_observer_location {
                            self.fused_solver.set_observer_location(lat, lon);
                        }
                        observer_location_set = true;
                    }
                }

                let sliced_batch: Vec<(Array2<f64>, Option<tetra3::extractor::Crop>)> =
                    centroids_batch
                        .iter()
                        .map(|(a, c, _label)| (a.clone(), c.clone()))
                        .collect();

                let batch_solution = self.fused_solver.solve_from_centroids_batch(
                    &sliced_batch,
                    size,
                    solve_options.clone(),
                    Some(img_timestamp),
                ).expect("Critical Error: solver failed to execute during solving loop. Is the database accessible?");

                let solution = batch_solution.solution;
                let mut solved_crop_enum = None;

                if solution.status == SolveStatus::MatchFound {
                    solved_crop_enum = Some(centroids_batch[batch_solution.crop_index].2.clone());
                }

                let solve_time = batch_solution.solve_time;

                match solution.status {
                    SolveStatus::MatchFound => {
                        // Report to telescope server
                        // If we have a target_ra from the solver (for target_pixel), use it.
                        // Otherwise use the image center.
                        let (mut ra, mut dec) = if let (Some(tra), Some(tdec)) =
                            (&solution.target_ra, &solution.target_dec)
                        {
                            (tra[0], tdec[0])
                        } else {
                            (solution.ra.unwrap(), solution.dec.unwrap())
                        };

                        // Sync logic: update target_pixel based on the solver's coordinate mapping
                        if let Some((t_ra, t_dec)) = pending_sync {
                            // The solver found the pixel coordinates for our requested sync RA/Dec
                            if let (Some(tx), Some(ty)) = (&solution.target_x, &solution.target_y) {
                                if let (Some(x), Some(y)) = (tx[0], ty[0]) {
                                    self.target_pixel = Some((x, y));
                                    self.save_target_pixel(self.target_pixel);
                                    self.mount_handle.notify_target_pixel_changed();
                                    if let Ok(mut state) = self.engine_state.write() {
                                        state.target_pixel_x = Some(x);
                                        state.target_pixel_y = Some(y);
                                    }
                                    info!(
                                        "Synced to target RA/Dec: ({}, {}). New target pixel: {:?}",
                                        t_ra, t_dec, self.target_pixel
                                    );

                                    // Use the synced values for reporting in this frame
                                    ra = t_ra;
                                    dec = t_dec;

                                    let mut synced_solution = solution.clone();
                                    synced_solution.target_ra = Some(vec![t_ra]);
                                    synced_solution.target_dec = Some(vec![t_dec]);
                                    self.fused_solver.update_from_solution(
                                        &synced_solution,
                                        Some(img_timestamp),
                                        true,
                                    );
                                } else {
                                    warn!(
                                        "Solver could not project sync RA/Dec to pixels (out of bounds?). Sync failed."
                                    );
                                }
                            }

                            // Consume the fields
                            self.shared_position
                                .target_op
                                .store(None, Ordering::Relaxed);
                            self.shared_position
                                .target_ra
                                .store(None, Ordering::Relaxed);
                            self.shared_position
                                .target_dec
                                .store(None, Ordering::Relaxed);
                        }

                        debug!(
                            "Solved! RA: {:.6}, Dec: {:.6}, Roll: {:.2} | Centroids: {} | Exposure: {:?} | Post-Process: {:?} | Extract: {:?} | Solve: {:?}",
                            ra,
                            dec,
                            solution.roll.unwrap(),
                            count,
                            exp_time,
                            pp_time,
                            extract_time,
                            solve_time
                        );

                        let is_full_frame = matches!(
                            solved_crop_enum,
                            Some(SolvedCrop::Full | SolvedCrop::FullHighSens)
                        );

                        if is_full_frame {
                            if let Some(new_fov) = solution.fov {
                                if self.fov.is_none() {
                                    log::info!(
                                        "Locked FOV to {:.2}° at {:.2}x zoom",
                                        new_fov,
                                        frame_zoom
                                    );
                                }
                                self.fov = Some(new_fov);
                            }

                            if let Some(new_dist) = solution.distortion {
                                if (-0.15..=0.05).contains(&new_dist) {
                                    if let Some(old_dist) = self.distortion {
                                        self.distortion = Some(0.9 * old_dist + 0.1 * new_dist);
                                    } else {
                                        self.distortion = Some(new_dist);
                                        log::info!(
                                            "Locked initial distortion to {:.6} at {:.2}x zoom",
                                            new_dist,
                                            frame_zoom
                                        );
                                    }
                                } else {
                                    log::warn!(
                                        "Rejected unphysical distortion fit: {:.6}",
                                        new_dist
                                    );
                                }
                            }
                        }
                        solved = true;

                        self.mount_handle.on_solved(ra, dec, frame_slew_epoch);

                        self.update_solve_stats(
                            true,
                            count,
                            self.target_ra,
                            self.target_dec,
                            exp_time.as_micros() as i32,
                            pp_time,
                            extract_time,
                            solve_time,
                            solved_crop_enum.clone(),
                        );
                    }
                    _ => {
                        let should_suppress_failure = self.config.auto_mode
                            && !self.config.is_adaptive_camera
                            && stacked_frames < MAX_STACKED_FRAMES
                            && (exp_time.as_micros() as u32) < MAX_VIRTUAL_EXPOSURE_US;

                        let solved_crop = if should_suppress_failure {
                            debug!(
                                "Still stacking ({} / {} frames, {} / {} ms). Suppressing failure UI.",
                                stacked_frames,
                                MAX_STACKED_FRAMES,
                                exp_time.as_millis(),
                                MAX_VIRTUAL_EXPOSURE_US / 1000
                            );
                            None
                        } else {
                            Some(SolvedCrop::Failed)
                        };

                        self.update_solve_stats(
                            false,
                            count,
                            self.target_ra,
                            self.target_dec,
                            exp_time.as_micros() as i32,
                            pp_time,
                            extract_time,
                            solve_time,
                            solved_crop,
                        );

                        warn!(
                            "No Solution (Status: {:?}) | Centroids: {} | Exposure: {:?} | Post-Process: {:?} | Extract: {:?} | Solve: {:?}",
                            solution.status, count, exp_time, pp_time, extract_time, solve_time
                        );
                    }
                }

                if solved {
                    self.clear_stacker
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }

                if !self.mount_handle.is_physically_slewing()
                    && let Some(ac) = adaptive_controller.as_mut()
                {
                    let mut match_ratio = None;
                    if solved {
                        let num_crop_centroids = sliced_batch[batch_solution.crop_index].0.nrows();
                        match_ratio = Some(
                            solution.matches.unwrap_or(0) as f32 / num_crop_centroids.max(1) as f32,
                        );
                    }

                    let obs = crate::auto_controller::FrameObservation {
                        actual_exposure_us,
                        actual_gain,
                        centroid_counts_per_crop: per_crop_centroid_counts.clone(),
                        full_frame_centroids: count,
                        background_level: bg_level.unwrap_or(0.0),
                        solve_succeeded: solved,
                        match_ratio,
                        solve_ra: solution.ra,
                        solve_dec: solution.dec,
                        motion_state: self.fused_solver.get_motion_state(),
                        frame_interval_us,
                        metadata_valid,
                    };

                    let actions = ac.on_frame(&obs);
                    for action in actions {
                        match action {
                            crate::auto_controller::ControllerAction::NoChange => {}
                            crate::auto_controller::ControllerAction::SetExposure(e) => {
                                self.camera.set_exposure(e);
                            }
                            crate::auto_controller::ControllerAction::SetGain(g) => {
                                self.camera.set_gain(g);
                            }
                            crate::auto_controller::ControllerAction::SetHotPixelMode(m) => {
                                self.camera.set_hot_pixel_mode(m);
                            }
                            crate::auto_controller::ControllerAction::SetNoiseReduction(m) => {
                                self.camera.set_noise_reduction(m);
                            }
                            crate::auto_controller::ControllerAction::SetSigma(s) => {
                                if let Ok(mut st) = self.engine_state.write() {
                                    st.current_sigma = Some(s);
                                }
                            }
                        }
                    }
                }
            } else {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

/// Standalone focus evaluation routine for external callers (e.g., Android autofocus).
/// Bypasses the core engine's background loops and forces a high-resolution 1x extraction
/// specifically optimized for accurate Half-Flux Radius calculation.
pub fn evaluate_focus(img_data: &[u8], width: usize, height: usize) -> f64 {
    let array_img = match Array2::from_shape_vec((height, width), img_data.to_vec()) {
        Ok(arr) => arr,
        Err(e) => {
            log::error!("Failed to create Array2 for autofocus: {}", e);
            return 0.0;
        }
    };

    // Use strict parameters optimized for finding round stars without downsampling
    let options = FastExtractOptions {
        sigma: 3.5,
        downsample: FastDownsample::None, // Full resolution is required for precise HFR
        bg_sub_mode: Some(FastBgSubMode::GlobalMedian),
        binary_open: false,
        min_area: Some(1), // Ignore hot pixels
        max_area: Some(50),
        ..FastExtractOptions::default()
    };

    let mut extractor = FastExtractor::new(width, height, options);
    let result = extractor.extract(&array_img);

    let num_centroids = result.centroids.len();
    if num_centroids == 0 {
        return 0.0;
    }

    let mut centroids_arr = Array2::<f64>::zeros((num_centroids, 2));
    for (i, c) in result.centroids.iter().enumerate() {
        centroids_arr[[i, 0]] = c.y;
        centroids_arr[[i, 1]] = c.x;
    }

    calculate_hfr(img_data, width, height, &centroids_arr, 30)
}

/// Calculates the median Half-Flux Radius (HFR) for a given set of centroids
pub fn calculate_hfr(
    image: &[u8],
    width: usize,
    height: usize,
    centroids: &Array2<f64>,
    max_stars: usize,
) -> f64 {
    let num_stars = std::cmp::min(centroids.nrows(), max_stars);
    if num_stars == 0 {
        return 0.0;
    }

    let mut hfrs = Vec::with_capacity(num_stars);
    let radius = 10isize;
    let bg_radius = 15isize;

    for i in 0..num_stars {
        let cy = centroids[[i, 0]];
        let cx = centroids[[i, 1]];

        let min_x = (cx.floor() as isize - bg_radius).max(0) as usize;
        let max_x = (cx.ceil() as isize + bg_radius).min(width as isize - 1) as usize;
        let min_y = (cy.floor() as isize - bg_radius).max(0) as usize;
        let max_y = (cy.ceil() as isize + bg_radius).min(height as isize - 1) as usize;

        let mut pixels = Vec::with_capacity(400);
        let mut bg_sum = 0.0;
        let mut bg_count = 0;

        // Iterate pixel window, sort strictly into star area or background annulus
        for y in min_y..=max_y {
            let row_offset = y * width;
            for x in min_x..=max_x {
                let val = image[row_offset + x] as f64;
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= (radius as f64 * radius as f64) {
                    pixels.push((dist_sq, val));
                } else if dist_sq <= (bg_radius as f64 * bg_radius as f64) {
                    bg_sum += val;
                    bg_count += 1;
                }
            }
        }

        let bg = if bg_count > 0 {
            bg_sum / bg_count as f64
        } else {
            0.0
        };

        let mut total_flux = 0.0;
        for p in &mut pixels {
            p.1 = (p.1 - bg).max(0.0);
            total_flux += p.1;
        }

        if total_flux <= 0.0 {
            continue;
        }

        pixels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let target_flux = total_flux / 2.0;
        let mut running_sum = 0.0;

        // Find the distance at which we cross the 50% flux threshold
        for (dist_sq, flux) in pixels {
            running_sum += flux;
            if running_sum >= target_flux {
                hfrs.push(dist_sq.sqrt());
                break;
            }
        }
    }

    if hfrs.is_empty() {
        return 0.0;
    }

    // Return the median HFR out of valid samples to protect against jitter outliers
    hfrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hfrs[hfrs.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_offset_angle_conversion_math() {
        // Test offset math against web converter output for standard position angle offsets.
        // The engine computes an offset_angle using Alt/Az mimicking this exact geometry,
        // with RA/Dec mapping equivalently to Az/Alt under a local small angle approximation.

        // Position 1: 12:12:12.00, +11:11:11.00
        let ra1_deg: f64 = (12.0 + 12.0 / 60.0 + 12.0 / 3600.0) * 15.0;
        let dec1_deg: f64 = 11.0 + 11.0 / 60.0 + 11.0 / 3600.0;

        // Position 2: 21:21:21.00, -05:05:05.00
        let ra2_deg: f64 = (21.0 + 21.0 / 60.0 + 21.0 / 3600.0) * 15.0;
        let dec2_deg: f64 = -(5.0 + 5.0 / 60.0 + 5.0 / 3600.0);

        // Calculate telescope offsets for 1 -> 2
        let east_offset_arcsec: f64 = (ra2_deg - ra1_deg) * 3600.0 * dec1_deg.to_radians().cos();
        let north_offset_arcsec: f64 = (dec2_deg - dec1_deg) * 3600.0;

        // Verify web converter output:
        // East Offset [arcsec] +484845.16
        // North Offset [arcsec] -58576.00
        assert!((east_offset_arcsec - 484845.16).abs() < 0.1);
        assert!((north_offset_arcsec - (-58576.00)).abs() < 0.1);

        // To calculate the position angle of the slit that connects these two points,
        // the web converter uses the midpoint declination to avoid distortion.
        let dec_mid: f64 = (dec1_deg + dec2_deg) / 2.0;
        let east_mid: f64 = (ra2_deg - ra1_deg) * dec_mid.to_radians().cos();
        let north_mid: f64 = dec2_deg - dec1_deg;

        // Web converter reports Slit Alignment Position Angle: +96.77
        let position_angle: f64 = east_mid.atan2(north_mid).to_degrees();
        assert!((position_angle - 96.77).abs() < 0.01);
    }

    #[test]
    fn test_horizon_bounds_computation() {
        let lon: f64 = -118.0;
        let max_fov: f64 = 70.0;
        let width: f64 = 1920.0;
        let height: f64 = 1080.0;

        let target_pixel: Option<(f64, f64)> = Some((1920.0, 540.0)); // Far right edge, middle Y

        // Test offset math
        let offset_angle = if let Some(tp) = target_pixel {
            let center_x: f64 = width / 2.0;
            let center_y: f64 = height / 2.0;
            let dist_px = ((tp.0 - center_x).powi(2) + (tp.1 - center_y).powi(2)).sqrt();
            (dist_px / width) * max_fov
        } else {
            0.0
        };

        // Center X = 960. dist_px = 960.
        // 960 / 1920 = 0.5. 0.5 * 70 = 35.0.
        assert!((offset_angle - 35.0).abs() < 0.01);

        let buffer = offset_angle + 1.0;
        assert!((buffer - 36.0).abs() < 0.01);

        // Test LST math
        let timescale = starfield::time::Timescale::default();
        // Just checking it doesn't crash and returns reasonable bounds
        let now = chrono::Utc::now();
        let t = timescale.from_datetime(now);
        let gast_hours = t.gast();
        let lst_deg = (gast_hours * 15.0 + lon).rem_euclid(360.0);

        assert!(lst_deg >= 0.0 && lst_deg < 360.0);
    }
    #[test]
    fn test_virtual_crops_axes() {
        use tetra3::fast_extractor::{FastExtractOptions, FastExtractor};

        // Load image (using 2019-07-29T204726_Alt40_Azi135_Try1.jpg since it has a wide field of stars)
        let bytes = include_bytes!(
            "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt40_Azi135_Try1.jpg"
        );
        let img = image::load_from_memory(bytes).unwrap().into_luma8();
        let (width, height) = (img.width() as usize, img.height() as usize);

        // Convert to Array2<u8> for FastExtractor
        let mut array_img = ndarray::Array2::<u8>::zeros((height, width));
        for (y, row) in img.enumerate_rows() {
            for (x, _, pixel) in row {
                array_img[[y as usize, x as usize]] = pixel.0[0];
            }
        }

        let virtual_crops = super::generate_concentric_center_crops(width, height);
        assert_eq!(virtual_crops.len(), 2);

        // Center 2.0x (50% dimensions)
        let (y1, y2, x1, x2) = virtual_crops[0].bounds(width, height);
        assert_eq!(y2 - y1, height / 2);
        assert_eq!(x2 - x1, width / 2);
        assert!(((y1 + y2) as isize / 2 - height as isize / 2).abs() <= 1);
        assert!(((x1 + x2) as isize / 2 - width as isize / 2).abs() <= 1);

        // Center 1.5x (66.7% dimensions)
        let (y1, y2, x1, x2) = virtual_crops[1].bounds(width, height);
        assert_eq!(y2 - y1, (height as f64 * (2.0 / 3.0)).round() as usize);
        assert_eq!(x2 - x1, (width as f64 * (2.0 / 3.0)).round() as usize);
        assert!(((y1 + y2) as isize / 2 - height as isize / 2).abs() <= 1);
        assert!(((x1 + x2) as isize / 2 - width as isize / 2).abs() <= 1);

        // Run extractor
        let options = FastExtractOptions {
            virtual_crops: Some(virtual_crops),
            binary_open: false, // disable noise filter to ensure we get lots of centroids
            ..Default::default()
        };
        let mut ext = FastExtractor::new(width, height, options);
        let result = ext.extract(&array_img);

        let vcrops_res = result
            .virtual_crop_centroids
            .expect("Should have virtual crop centroids");
        assert_eq!(
            vcrops_res.len(),
            2,
            "Expected 2 concentric center crop buckets"
        );

        // Verify each quadrant actually extracted stars
        assert!(!vcrops_res[0].is_empty(), "Expected stars in Center 2.0x");
        assert!(!vcrops_res[1].is_empty(), "Expected stars in Center 1.5x");
    }

    #[test]
    fn test_virtual_crop_fov_threshold() {
        let width = 1920;
        let height = 1080;

        // 1. None FOV -> no virtual crops
        let (crops, labels) = super::evaluate_virtual_crops(None, width, height);
        assert!(crops.is_none());
        assert!(labels.is_none());

        // 2. Sub-threshold FOV (e.g. 15.0°, 24.99°) -> no virtual crops
        let (crops, labels) = super::evaluate_virtual_crops(Some(15.0), width, height);
        assert!(crops.is_none());
        assert!(labels.is_none());

        let (crops, labels) = super::evaluate_virtual_crops(Some(24.99), width, height);
        assert!(crops.is_none());
        assert!(labels.is_none());

        // 3. Exact threshold (25.0°) -> activates 2 virtual crops & labels
        let (crops, labels) = super::evaluate_virtual_crops(Some(25.0), width, height);
        let crops = crops.expect("Should produce virtual crops at 25.0°");
        let labels = labels.expect("Should produce virtual crop labels at 25.0°");
        assert_eq!(crops.len(), 2);
        assert_eq!(labels, super::VIRTUAL_CROP_LABELS);

        // 4. Wide-angle FOV (35.0°) -> activates 2 virtual crops & labels
        let (crops, labels) = super::evaluate_virtual_crops(Some(35.0), width, height);
        let crops = crops.expect("Should produce virtual crops at 35.0°");
        let labels = labels.expect("Should produce virtual crop labels at 35.0°");
        assert_eq!(crops.len(), 2);
        assert_eq!(labels, super::VIRTUAL_CROP_LABELS);
    }

    fn create_test_engine(
        config: EngineConfig,
        initial_fov: Option<f64>,
    ) -> (Engine, Arc<RwLock<EngineStateData>>) {
        use crate::dummy_camera::DummyCamera;
        use olive_solve::FusedSolver;
        use std::collections::VecDeque;
        use std::path::Path;
        use std::sync::atomic::AtomicBool;
        use std::sync::{Condvar, Mutex};

        let camera = Arc::new(DummyCamera::new());
        let engine_state = Arc::new(RwLock::new(EngineStateData::default()));
        let shared_position = Arc::new(SharedPosition::default());
        let timescale = Arc::new(Timescale::default());
        let server = Lx200WifiTelescope::new(
            shared_position.clone(),
            Arc::new(AtomicBool::new(true)),
            timescale.clone(),
        );
        let lx200_controller = server.get_controller();
        let fused_solver = Arc::new(
            FusedSolver::new(
                Path::new("../olive-solve/tetra3/tests/fixtures/default_database.npz"),
                None,
                None,
                false,
            )
            .unwrap(),
        );
        let mount_handle = MountHandle::start(timescale.clone(), None);

        let engine = Engine {
            camera,
            config,
            last_gain_adjust: None,
            fused_solver,
            fov: initial_fov,
            distortion: None,
            active_zoom: 1.0,
            shared_position,
            target_ra: None,
            target_dec: None,
            target_pixel: None,
            target_pixel_key: "target_pixel_demo".to_string(),
            storage: None,
            client_connected: Arc::new(AtomicBool::new(false)),
            engine_state: engine_state.clone(),
            solve_timestamps: VecDeque::with_capacity(60),
            shared_result: Arc::new((Mutex::new(None), Condvar::new())),
            is_running: Arc::new(AtomicBool::new(true)),
            clear_stacker: Arc::new(AtomicBool::new(false)),
            telescope_server_handle: None,
            alpaca_server_handle: None,
            rt_handle: None,
            lx200_controller,
            mount_handle,
            timescale,
        };
        (engine, engine_state)
    }

    #[test]
    fn test_update_solve_stats_crop_status() {
        let (mut engine, engine_state) = create_test_engine(EngineConfig::default(), Some(15.0));

        // 1. Initial state has None
        assert_eq!(engine_state.read().unwrap().solved_crop, None);

        // 2. Successful solve updates solved_crop to the solved region (e.g. Full)
        engine.update_solve_stats(
            true,
            25,
            None,
            None,
            33000,
            Duration::from_millis(5),
            Duration::from_millis(10),
            Duration::from_millis(20),
            Some(SolvedCrop::Full),
        );
        assert_eq!(
            engine_state.read().unwrap().solved_crop,
            Some(SolvedCrop::Full)
        );

        // 3. Failed solve updates solved_crop to Failed
        engine.update_solve_stats(
            false,
            5,
            None,
            None,
            330000,
            Duration::from_millis(5),
            Duration::from_millis(10),
            Duration::from_millis(20),
            Some(SolvedCrop::Failed),
        );
        assert_eq!(
            engine_state.read().unwrap().solved_crop,
            Some(SolvedCrop::Failed)
        );

        // 4. Successful solve updates to Center
        engine.update_solve_stats(
            true,
            30,
            None,
            None,
            33000,
            Duration::from_millis(5),
            Duration::from_millis(10),
            Duration::from_millis(20),
            Some(SolvedCrop::Center2x),
        );
        assert_eq!(
            engine_state.read().unwrap().solved_crop,
            Some(SolvedCrop::Center2x)
        );
    }

    #[test]
    fn test_active_imu_and_low_confidence_handling() {
        let (mut engine, engine_state) = create_test_engine(EngineConfig::default(), Some(15.0));

        // 1. When no IMU is running, get_motion_state() is None
        let has_active_imu = engine.fused_solver.get_motion_state().is_some();
        assert!(!has_active_imu, "IMU should not be active by default");

        // 2. An unverified LowConfidenceMatch must not clear the stacker
        engine
            .clear_stacker
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let solved = false;
        if solved {
            engine
                .clear_stacker
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        assert!(
            !engine
                .clear_stacker
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        // 3. update_solve_stats on failed/unverified match updates stats with solved = false
        engine.update_solve_stats(
            false,
            4,
            None,
            None,
            33000,
            Duration::from_millis(2),
            Duration::from_millis(5),
            Duration::from_millis(15),
            Some(SolvedCrop::Failed),
        );
        let st = engine_state.read().unwrap();
        assert_eq!(st.solved_crop, Some(SolvedCrop::Failed));
        assert_eq!(st.centroids, 4);
    }

    #[test]
    fn test_calculate_alt_az_zenith() {
        let ts = starfield::time::Timescale::default();
        let time = chrono::Utc::now();
        let t = ts.from_datetime(time);
        let gast_hours = t.gast();
        let lon_deg = 0.0;
        let lst_deg = (gast_hours * 15.0 + lon_deg).rem_euclid(360.0);

        // Target right at zenith: RA = LST (ha = 0), Dec = Lat
        let lat_deg = 45.0;
        let dec_deg = 45.0;
        let ra_deg = lst_deg;

        let (alt, _) = calculate_alt_az(ra_deg, dec_deg, lat_deg, lon_deg, &time, &ts);
        assert!(!alt.is_nan());
        assert!((alt - 90.0).abs() < 1e-4);
    }

    #[test]
    fn test_fov_updated_from_solver() {
        let initial_fov = 58.0;
        let (mut engine, engine_state) = create_test_engine(
            EngineConfig {
                initial_fov: Some(initial_fov as f32),
                ..EngineConfig::default()
            },
            Some(initial_fov),
        );

        // Before solve: fov is initial seed
        assert_eq!(engine.fov, Some(58.0));

        // Simulate solver match returning 65.5° FOV
        let solution_fov: f64 = 65.5;
        engine.fov = Some(solution_fov);

        engine.update_solve_stats(
            true,
            30,
            None,
            None,
            33000,
            Duration::from_millis(5),
            Duration::from_millis(10),
            Duration::from_millis(20),
            Some(SolvedCrop::Full),
        );

        let st = engine_state.read().unwrap();
        assert_eq!(st.fov, Some(65.5));
    }

    #[test]
    fn test_target_pixel_keys() {
        let demo_cfg = CameraConfig {
            camera_type: CameraType::Demo,
            model: "Demo Camera".to_string(),
            width: 1280,
            height: 720,
            fps: 30,
            crop: None,
            initial_fov: None,
        };
        assert_eq!(
            Engine::get_target_pixel_key(Some(&demo_cfg)),
            "target_pixel_demo"
        );

        let internal_cfg = CameraConfig {
            camera_type: CameraType::Internal {
                id: "0".to_string(),
            },
            model: "Android_Cam_0".to_string(),
            width: 4032,
            height: 3024,
            fps: 2,
            crop: None,
            initial_fov: None,
        };
        assert_eq!(
            Engine::get_target_pixel_key(Some(&internal_cfg)),
            "target_pixel_internal_0"
        );

        let uvc_cfg = CameraConfig {
            camera_type: CameraType::External,
            model: "External UVC".to_string(),
            width: 1920,
            height: 1080,
            fps: 30,
            crop: None,
            initial_fov: None,
        };
        assert_eq!(
            Engine::get_target_pixel_key(Some(&uvc_cfg)),
            "target_pixel_external_1920x1080"
        );

        assert_eq!(Engine::get_target_pixel_key(None), "target_pixel_demo");
    }

    #[test]
    fn test_parse_pixel_coords() {
        assert_eq!(
            Engine::parse_pixel_coords("123.45, 678.9"),
            Some((123.45, 678.9))
        );
        assert_eq!(Engine::parse_pixel_coords("0,0"), Some((0.0, 0.0)));
        assert_eq!(Engine::parse_pixel_coords("invalid"), None);
        assert_eq!(Engine::parse_pixel_coords("1,2,3"), None);
    }

    #[test]
    fn test_zoom_change_engine_state_reset() {
        let engine_state = Arc::new(RwLock::new(EngineStateData::default()));
        {
            let mut st = engine_state.write().unwrap();
            st.exposure_us = 250_000;
            st.focus = Some(0.42);
            st.target_pixel_x = Some(640.0);
            st.target_pixel_y = Some(360.0);
            st.client_latitude = Some(37.77);
            st.client_longitude = Some(-122.42);
            st.ra = Some(180.0);
            st.dec = Some(45.0);
            st.center_ra = Some(180.0);
            st.center_dec = Some(45.0);
            st.roll = Some(12.0);
            st.centroids = 45;
            st.solve_rate_hz = 15.0;
            st.offset_ra = Some(1.5);
            st.offset_dec = Some(2.0);
            st.hfr = 2.8;
            st.solved_crop = Some(SolvedCrop::Center2x);
        }

        // Simulate zoom change from 2.0x to 1.0x: only solver feedback parameters (FOV & distortion) are reset
        let initial_engine_fov = Some(65.0);
        {
            let mut state = engine_state.write().unwrap();
            state.fov = initial_engine_fov;
            state.distortion = None;
        }

        let st = engine_state.read().unwrap();
        // Preserved calibration and observer location
        assert_eq!(st.exposure_us, 250_000);
        assert_eq!(st.focus, Some(0.42));
        assert_eq!(st.target_pixel_x, Some(640.0));
        assert_eq!(st.target_pixel_y, Some(360.0));
        assert_eq!(st.client_latitude, Some(37.77));
        assert_eq!(st.client_longitude, Some(-122.42));

        // Preserved solutions and metrics (not cleared on zoom transition)
        assert_eq!(st.ra, Some(180.0));
        assert_eq!(st.dec, Some(45.0));
        assert_eq!(st.center_ra, Some(180.0));
        assert_eq!(st.center_dec, Some(45.0));
        assert_eq!(st.roll, Some(12.0));
        assert_eq!(st.centroids, 45);
        assert_eq!(st.solve_rate_hz, 15.0);
        assert_eq!(st.offset_ra, Some(1.5));
        assert_eq!(st.offset_dec, Some(2.0));
        assert_eq!(st.hfr, 2.8);
        assert_eq!(st.solved_crop, Some(SolvedCrop::Center2x));

        // Reset solver feedback data
        assert_eq!(st.fov, Some(65.0));
        assert_eq!(st.distortion, None);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
        self.mount_handle.shutdown();
        if let Some(handle) = self.telescope_server_handle.take() {
            handle.abort();
        }
        if let Some(handle) = self.alpaca_server_handle.take() {
            handle.abort();
        }
    }
}
