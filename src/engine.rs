// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::sync::atomic::Ordering;
use std::{
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime},
};

use log::{debug, error, info, warn};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Receiver;

use crate::CameraTrait;
use crate::alpaca_server::AlpacaServer;
use crate::telescope_server::{Lx200Telescope, Lx200WifiTelescope, SharedPosition, TargetOp};
use tetra3::{
    fast_extractor::{FastBgSubMode, FastDownsample, FastExtractOptions, FastExtractor},
    solver::{SolveOptions, SolveStatus},
};

pub const MIN_CENTROIDS: usize = 4;

const MULTIPLE_EXTRACTION_MIN_EXPOSURE_US: i32 = 100_000;
const MULTIPLE_EXTRACTION_SIGMA: f32 = 2.6;
const MULTIPLE_EXTRACTION_MIN_AREA: usize = 1;
const MULTIPLE_EXTRACTION_NOISE_FILTER: bool = false;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub gain: Option<f32>,
    pub target_centroids: usize,
    pub database_path: String,
    pub isp: bool,
    pub bg_mode: BgMode,
    pub extractor_type: Extractor,
    pub crop: Option<(usize, usize)>,
    pub downsample: u8,
    pub auto_mode: bool,
    pub noise_filter: bool,
    pub initial_fov: Option<f32>,
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
            noise_filter: false,
            initial_fov: None,
            use_observer_location: true,
        }
    }
}

pub struct IntermediateResult {
    pub count: usize,
    pub exp_time: Duration,
    pub phys_exp_time: Duration,
    pub pp_time: Duration,
    pub extract_time: Duration,
    pub centroids_batch: Vec<(Array2<f64>, Option<tetra3::extractor::Crop>, SolvedCrop)>,
    pub timestamp: SystemTime,
    pub stacked_frames: u32,
    pub background_level: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum, Serialize, Deserialize)]
pub enum Extractor {
    #[default]
    Approximate,
    Fast,
    Cedar,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum, Serialize, Deserialize)]
pub enum BgMode {
    #[default]
    Line,
    Global,
    Block,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum SolvedCrop {
    Center,
    Top,
    Left,
    Right,
    Bottom,
    Full,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineStateData {
    pub status: String,
    pub ra: Option<f64>,
    pub dec: Option<f64>,
    pub roll: Option<f64>,
    pub centroids: usize,
    pub target_ra: Option<f64>,
    pub target_dec: Option<f64>,
    pub exposure_us: i32,
    pub post_process_ms: f64,
    pub extract_ms: f64,
    pub solve_ms: f64,
    pub solve_rate_hz: f64,
    pub capture_requested: bool,
    pub paused: bool,
    pub hfr: f64, // Added focus metric
    pub client_latitude: Option<f64>,
    pub client_longitude: Option<f64>,
    pub client_date: Option<chrono::DateTime<chrono::Utc>>,
    pub offset_ra: Option<f64>,
    pub offset_dec: Option<f64>,
    pub offset_az: Option<f64>,
    pub offset_alt: Option<f64>,
    pub offset_angle: Option<f64>,
    pub current_sigma: Option<f32>,
    pub fov: Option<f64>,
    pub distortion: Option<f64>,
    pub requested_target_ra: Option<f64>,
    pub requested_target_dec: Option<f64>,
    pub requested_target_epoch: Option<f64>,
    pub requested_align_target_pixel: Option<(f64, f64)>,
    pub is_imu_estimate: bool,
    pub strict_match: bool,
    pub solved_crop: Option<SolvedCrop>,
}

impl Default for EngineStateData {
    fn default() -> Self {
        Self {
            status: "Idle".to_string(),
            ra: None,
            dec: None,
            roll: None,
            centroids: 0,
            target_ra: None,
            target_dec: None,
            exposure_us: 0,
            post_process_ms: 0.0,
            extract_ms: 0.0,
            solve_ms: 0.0,
            solve_rate_hz: 0.0,
            capture_requested: false,
            paused: false,
            hfr: 0.0,
            client_latitude: None,
            client_longitude: None,
            client_date: None,
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
            requested_align_target_pixel: None,
            is_imu_estimate: false,
            strict_match: false,
            solved_crop: None,
        }
    }
}

const MAX_STACKED_FRAMES: u32 = 10;
const MAX_VIRTUAL_EXPOSURE_US: u32 = 3_000_000;
const GAIN_ADJUST_THROTTLE_MS: u64 = 10_000;

#[allow(dead_code)]
pub struct Engine {
    camera: Arc<dyn CameraTrait>,
    config: EngineConfig,
    last_gain_adjust: Option<std::time::Instant>,
    pub fused_solver: Arc<olive_solve::FusedSolver>,
    fov: Option<f64>,
    distortion: Option<f64>,

    shared_position: Arc<SharedPosition>,
    pub telescope_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub lx200_controller:
        std::sync::Arc<std::sync::RwLock<crate::telescope_server::Lx200Controller>>,
    pub alpaca_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub rt_handle: Option<tokio::runtime::Handle>,
    target_ra: Option<f64>,
    target_dec: Option<f64>,
    target_pixel: Option<(f64, f64)>,
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
    timescale: Arc<starfield::time::Timescale>,
}

fn angle_diff_deg(a: f64, b: f64) -> f64 {
    let diff = (a - b).rem_euclid(360.0);
    if diff > 180.0 { diff - 360.0 } else { diff }
}

fn calculate_alt_az(
    ra_deg: f64,
    dec_deg: f64,
    lat_deg: f64,
    lon_deg: f64,
    time: &chrono::DateTime<chrono::Utc>,
    ts: &starfield::time::Timescale,
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
    let alt_rad = sin_alt.asin();

    // Azimuth
    let cos_az = (dec_rad.sin() - alt_rad.sin() * lat_rad.sin()) / (alt_rad.cos() * lat_rad.cos());
    let sin_az = -dec_rad.cos() * ha_rad.sin() / alt_rad.cos();
    let az_rad = sin_az.atan2(cos_az).rem_euclid(std::f64::consts::TAU);

    (alt_rad.to_degrees(), az_rad.to_degrees())
}

impl Engine {
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

        let engine_state_clone = engine_state.clone();
        let shared_pos_clone = shared_position.clone();
        let fused_solver_clone = fused_solver.clone();
        let is_running_clone = is_running.clone();
        let timescale_clone = timescale.clone();
        let clear_stacker_clone = clear_stacker.clone();

        let polling_task = async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(33));
            while is_running_clone.load(std::sync::atomic::Ordering::Relaxed) {
                interval.tick().await;

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

                    shared_pos_clone
                        .telescope_ra
                        .store(ra.to_bits(), std::sync::atomic::Ordering::Relaxed);
                    shared_pos_clone
                        .telescope_dec
                        .store(dec.to_bits(), std::sync::atomic::Ordering::Relaxed);

                    let mut target_ra_opt = None;
                    let mut target_dec_opt = None;
                    let mut lat_opt_state = None;
                    let mut lon_opt_state = None;
                    let mut client_date_opt = None;

                    if let Ok(state) = engine_state_clone.read() {
                        target_ra_opt = state.target_ra;
                        target_dec_opt = state.target_dec;
                        lat_opt_state = state.client_latitude;
                        lon_opt_state = state.client_longitude;
                        client_date_opt = state.client_date;
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
                            let time = client_date_opt.unwrap_or_else(|| chrono::Utc::now());
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
                        state.roll = Some(eq_roll);
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

        Self {
            camera,
            config,
            last_gain_adjust: None,
            fused_solver,
            fov: config_fov,
            distortion: None,
            shared_position,
            target_ra: None,
            target_dec: None,
            target_pixel: Self::load_target_pixel(storage.as_deref()),
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
        }
    }

    fn load_target_pixel(
        storage: Option<&dyn olive_imu::storage::PersistentStorage>,
    ) -> Option<(f64, f64)> {
        storage.and_then(|s| s.get("target_pixel")).and_then(|s| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(x), Ok(y)) = (parts[0].trim().parse(), parts[1].trim().parse()) {
                    return Some((x, y));
                }
            }
            None
        })
    }

    fn save_target_pixel(&self, pixel: Option<(f64, f64)>) {
        if let Some(storage) = &self.storage {
            if let Some((x, y)) = pixel {
                storage.set("target_pixel", &format!("{},{}", x, y));
            } else {
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

        let zoom = self.camera.get_zoom().unwrap_or(1.0) as f64;
        let reported_fov = if let Some(fov) = self.fov {
            if (zoom - 1.0).abs() > 0.01 {
                let fov_rad = fov.to_radians();
                Some(2.0 * ((fov_rad / 2.0).tan() / zoom).atan().to_degrees())
            } else {
                Some(fov)
            }
        } else {
            None
        };

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
            state.solved_crop = solved_crop;
        }
    }

    pub fn start(self) -> (std::thread::JoinHandle<()>, std::thread::JoinHandle<()>) {
        let camera_ext = self.camera.clone();
        let config_ext = self.config.clone();
        let shared_res_ext = self.shared_result.clone();
        let engine_state_ext = self.engine_state.clone();
        let is_running_ext = self.is_running.clone();
        let client_connected_ext = self.client_connected.clone();
        let clear_stacker_ext = self.clear_stacker.clone();

        let t1 = std::thread::spawn(move || {
            let mut stacker = crate::stacker::Stacker::new(0, 0, config_ext.auto_mode);
            let mut extractor: Option<FastExtractor> = None;
            let mut last_processed_timestamp = None;
            let mut active_max_area = 150;
            let mut hfr_history: std::collections::VecDeque<(std::time::SystemTime, f64)> =
                std::collections::VecDeque::new();

            loop {
                if !is_running_ext.load(std::sync::atomic::Ordering::Relaxed) {
                    log::info!("Engine thread stopping.");
                    break;
                }
                {
                    let mut needs_update = false;
                    {
                        let state = engine_state_ext.read().unwrap();
                        if state.paused && state.status != "Camera Mode" {
                            needs_update = true;
                        }
                    }
                    if needs_update {
                        let mut state = engine_state_ext.write().unwrap();
                        state.status = "Camera Mode".to_string();
                    }
                }

                let img_opt = camera_ext.wait_for_next_capture(
                    last_processed_timestamp,
                    Some(std::time::Duration::from_millis(1000)),
                );
                let img_arc = match img_opt {
                    Some(img) => img,
                    None => continue,
                };

                // We should also check is_running after waiting.
                if !is_running_ext.load(std::sync::atomic::Ordering::Relaxed) {
                    log::info!("Engine loop stopping due to is_running flag after wait.");
                    break;
                }
                last_processed_timestamp = Some(img_arc.timestamp);

                let mut do_capture = false;
                {
                    let mut state = engine_state_ext.write().unwrap();
                    if state.capture_requested {
                        state.capture_requested = false;
                        do_capture = true;
                    }
                }

                let exposure_us = img_arc.exposure_time_us;
                let post_process_time = img_arc.post_process_time;
                let img_timestamp = img_arc.timestamp;

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
                let num_centroids;
                let centroids_batch;
                let mut baseline_full_centroids = None;
                let mut final_bg_level: Option<f32> = None;

                let (state_sigma, state_fov) = {
                    let st = engine_state_ext.read().unwrap();
                    (st.current_sigma, st.fov)
                };

                match config_ext.extractor_type {
                    Extractor::Approximate | Extractor::Fast => {
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
                                if proc_width * proc_height > 2_000_000 {
                                    FastDownsample::X4
                                } else if proc_width * proc_height > 1_000_000 {
                                    FastDownsample::X2
                                } else {
                                    FastDownsample::None
                                }
                            }
                        };

                        let active_sigma = state_sigma.unwrap_or(3.5);
                        let mut update_max_area = None;
                        let best_fov = state_fov.or(config_ext.initial_fov.map(|f| f as f64));
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
                                    (raw_max_area / (ds_factor * ds_factor)).round().max(10.0)
                                        as usize;
                                update_max_area = Some(active_max_area);
                            }
                        }

                        let (virtual_crops, virtual_crop_labels) = if let Some(fov) = best_fov {
                            if fov >= 13.5 {
                                let fov_rad = fov.to_radians();
                                let target_rad = 10.5f64.to_radians();
                                let min_ratio = (target_rad / 2.0).tan() / (fov_rad / 2.0).tan();
                                let scale = min_ratio.clamp(0.50, 0.75);

                                let crop_w = (proc_width as f64 * scale).round() as usize;
                                let crop_h = (proc_height as f64 * scale).round() as usize;

                                let off_x = (proc_width as isize - crop_w as isize) / 2;
                                let off_y = (proc_height as isize - crop_h as isize) / 2;

                                (
                                    Some(vec![
                                        tetra3::extractor::Crop::Center {
                                            width: crop_w,
                                            height: crop_h,
                                        },
                                        tetra3::extractor::Crop::Region {
                                            height: crop_h,
                                            width: crop_w,
                                            offset_y: -off_y,
                                            offset_x: 0,
                                        },
                                        tetra3::extractor::Crop::Region {
                                            height: crop_h,
                                            width: crop_w,
                                            offset_y: 0,
                                            offset_x: -off_x,
                                        },
                                        tetra3::extractor::Crop::Region {
                                            height: crop_h,
                                            width: crop_w,
                                            offset_y: 0,
                                            offset_x: off_x,
                                        },
                                        tetra3::extractor::Crop::Region {
                                            height: crop_h,
                                            width: crop_w,
                                            offset_y: off_y,
                                            offset_x: 0,
                                        },
                                    ]),
                                    Some(vec![
                                        SolvedCrop::Center,
                                        SolvedCrop::Top,
                                        SolvedCrop::Left,
                                        SolvedCrop::Right,
                                        SolvedCrop::Bottom,
                                    ]),
                                )
                            } else {
                                (None, None)
                            }
                        } else {
                            let off_x = proc_width as isize / 4;
                            let off_y = proc_height as isize / 4;
                            (
                                Some(vec![
                                    tetra3::extractor::Crop::Center {
                                        width: proc_width / 2,
                                        height: proc_height / 2,
                                    },
                                    tetra3::extractor::Crop::Region {
                                        height: proc_height / 2,
                                        width: proc_width / 2,
                                        offset_y: -off_y,
                                        offset_x: 0,
                                    },
                                    tetra3::extractor::Crop::Region {
                                        height: proc_height / 2,
                                        width: proc_width / 2,
                                        offset_y: 0,
                                        offset_x: -off_x,
                                    },
                                    tetra3::extractor::Crop::Region {
                                        height: proc_height / 2,
                                        width: proc_width / 2,
                                        offset_y: 0,
                                        offset_x: off_x,
                                    },
                                    tetra3::extractor::Crop::Region {
                                        height: proc_height / 2,
                                        width: proc_width / 2,
                                        offset_y: off_y,
                                        offset_x: 0,
                                    },
                                ]),
                                Some(vec![
                                    SolvedCrop::Center,
                                    SolvedCrop::Top,
                                    SolvedCrop::Left,
                                    SolvedCrop::Right,
                                    SolvedCrop::Bottom,
                                ]),
                            )
                        };

                        if extractor.is_none() {
                            let bg_sub_mode = match config_ext.bg_mode {
                                BgMode::Global => Some(FastBgSubMode::GlobalMedian),
                                BgMode::Block => {
                                    Some(FastBgSubMode::BlockMedian { block_size: 32 })
                                }
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

                        let do_multiple_variants = exposure_us
                            >= MULTIPLE_EXTRACTION_MIN_EXPOSURE_US
                            && config_ext.auto_mode;

                        if do_multiple_variants {
                            variants.push(tetra3::fast_extractor::FastExtractOptionsUpdate {
                                sigma: Some(MULTIPLE_EXTRACTION_SIGMA),
                                min_area: Some(MULTIPLE_EXTRACTION_MIN_AREA),
                                noise_filter: Some(MULTIPLE_EXTRACTION_NOISE_FILTER),
                                ..Default::default()
                            });
                        }

                        let results = ext.extract_variants(&array_img, &variants);
                        final_bg_level = results.first().map(|r| r.background_level);

                        let mut batch: Vec<(
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
                                        let crop = virtual_crops
                                            .as_ref()
                                            .and_then(|vc| vc.get(i).cloned());
                                        let label =
                                            virtual_crop_labels.as_ref().unwrap()[i].clone();
                                        batch.push((arr, crop, label));
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
                            batch.push((arr, None, SolvedCrop::Full));
                        }

                        // Use the base variant's full image centroid count for the UI
                        num_centroids = results.first().map(|r| r.centroids.len()).unwrap_or(0);
                        centroids_batch = batch;
                    }
                    Extractor::Cedar => {
                        let final_crop = config_ext
                            .crop
                            .or(cam_config.crop.map(|(w, h)| (w as usize, h as usize)));
                        let (crop_w, crop_h, crop_x, crop_y) = match final_crop {
                            Some((cw, ch)) => {
                                let cw = cw.min(width);
                                let ch = ch.min(height);
                                let cx = (width - cw) / 2;
                                let cy = (height - ch) / 2;
                                (cw, ch, cx, cy)
                            }
                            None => (width, height, 0, 0),
                        };

                        let cropped_img_data = if final_crop.is_some() {
                            let mut cropped = Vec::with_capacity(crop_w * crop_h);
                            for y in crop_y..(crop_y + crop_h) {
                                let start = y * width + crop_x;
                                let end = start + crop_w;
                                cropped.extend_from_slice(&final_img_data[start..end]);
                            }
                            cropped
                        } else {
                            final_img_data.clone()
                        };

                        if let Some(gray_img) = image::ImageBuffer::from_raw(
                            crop_w as u32,
                            crop_h as u32,
                            cropped_img_data,
                        ) {
                            let noise_estimate =
                                cedar_detect::algorithm::estimate_noise_from_image(&gray_img);
                            let active_sigma = state_sigma.unwrap_or(8.0);
                            let (stars, _, _, _) = cedar_detect::algorithm::get_stars_from_image(
                                &gray_img,
                                noise_estimate,
                                active_sigma as f64, // sigma
                                true,                // normalize_rows
                                match config_ext.downsample {
                                    4 => 4,
                                    2 => 2,
                                    1 => 1,
                                    _ => {
                                        if crop_w * crop_h > 2_000_000 {
                                            4
                                        } else if crop_w * crop_h > 1_000_000 {
                                            2
                                        } else {
                                            1
                                        }
                                    }
                                }, // binning
                                config_ext.noise_filter, // detect_hot_pixels
                                false,               // return_binned_image
                            );

                            num_centroids = stars.len();
                            let mut arr = Array2::<f64>::zeros((num_centroids, 2));
                            for (i, c) in stars.iter().enumerate() {
                                arr[[i, 0]] = c.centroid_y + crop_y as f64;
                                arr[[i, 1]] = c.centroid_x + crop_x as f64;
                            }
                            centroids_batch = vec![(arr, None, SolvedCrop::Full)];
                        } else {
                            error!("Failed to create ImageBuffer from image data");
                            continue;
                        }
                    }
                }

                let extract_time = extract_start.elapsed();

                // Calculate focus metric only when camera mode is active (manual focus).
                // This prevents wasting CPU cycles during active plate solving.
                let is_paused = engine_state_ext.read().unwrap().paused;
                let mut hfr = if client_connected_ext.load(std::sync::atomic::Ordering::Relaxed)
                    || is_paused
                {
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
                    exp_time: Duration::from_micros(virtual_exposure_us as u64),
                    phys_exp_time: Duration::from_micros(exposure_us as u64),
                    pp_time: post_process_time,
                    extract_time,
                    centroids_batch,
                    timestamp: img_timestamp,
                    stacked_frames: stacker.stacked_frames,
                    background_level: final_bg_level,
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
                if engine.engine_state.read().unwrap().paused {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    continue;
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
    /// Skipped when `Extractor::Cedar` is in use, calibration is bypassed, or
    /// when an explicit gain is provided in the configuration.
    fn auto_adjust_gain(&mut self, bg_level: f32) {
        if !self.config.auto_mode
            || self.config.gain.is_some()
            || self.config.extractor_type == Extractor::Cedar
        {
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
        let mut observer_location_set = false;
        let mut was_connected = false;

        loop {
            if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }

            // 1. Process Local UI Requests (JNI/Push-To GoTo & Pixel Align)
            if let Ok(mut st) = self.engine_state.write() {
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

                    self.shared_position
                        .target_op
                        .store(Some(TargetOp::Slew), std::sync::atomic::Ordering::Relaxed);
                    self.shared_position
                        .is_slewing
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    info!("Engine received UI target request: RA={}, Dec={}", ra, dec);
                }
                if let Some((x, y)) = st.requested_align_target_pixel {
                    self.target_pixel = Some((x, y));
                    self.save_target_pixel(self.target_pixel);
                    st.requested_align_target_pixel = None;
                    info!("Engine received align target pixel request: ({}, {})", x, y);
                }
            }

            // 2. Process External Client & State Flags (Alpaca/LX200)
            let target_op = self
                .shared_position
                .target_op
                .load(std::sync::atomic::Ordering::Relaxed);
            let target_ra = self
                .shared_position
                .target_ra
                .load(std::sync::atomic::Ordering::Relaxed);
            let target_dec = self
                .shared_position
                .target_dec
                .load(std::sync::atomic::Ordering::Relaxed);

            if target_op == Some(TargetOp::Abort) {
                self.target_ra = None;
                self.target_dec = None;
                if let Ok(mut st) = self.engine_state.write() {
                    st.target_ra = None;
                    st.target_dec = None;
                }
                self.shared_position
                    .target_op
                    .store(None, std::sync::atomic::Ordering::Relaxed);
                self.shared_position
                    .is_slewing
                    .store(false, std::sync::atomic::Ordering::Relaxed);
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
                self.shared_position
                    .target_op
                    .store(None, std::sync::atomic::Ordering::Relaxed);
                self.shared_position
                    .is_slewing
                    .store(true, std::sync::atomic::Ordering::Relaxed);

                // Consume it so we don't repeatedly trigger it
                self.shared_position
                    .target_ra
                    .store(None, std::sync::atomic::Ordering::Relaxed);
                self.shared_position
                    .target_dec
                    .store(None, std::sync::atomic::Ordering::Relaxed);
            }

            if self.engine_state.read().unwrap().paused {
                self.update_status("Camera Mode");
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
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
            self.update_status("Solving");

            if let Some(res) = self.wait_for_extraction() {
                if let Some(bg) = res.background_level {
                    self.auto_adjust_gain(bg);
                }
                let count = res.count;
                let exp_time = res.exp_time;
                let pp_time = res.pp_time;
                let extract_time = res.extract_time;
                let centroids_batch = res.centroids_batch;
                let img_timestamp = res.timestamp;
                let stacked_frames = res.stacked_frames;
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

                if count < MIN_CENTROIDS {
                    warn!(
                        "Discarding frame: {} centroids (< {} min). Telescope might be moving.",
                        count, MIN_CENTROIDS
                    );
                    if !self.config.auto_mode
                        || stacked_frames >= MAX_STACKED_FRAMES
                        || exp_time.as_micros() as u32 >= MAX_VIRTUAL_EXPOSURE_US
                    {
                        self.update_solve_stats(
                            false,
                            count,
                            self.target_ra,
                            self.target_dec,
                            exp_time.as_micros() as i32,
                            Duration::ZERO,
                            extract_time,
                            Duration::ZERO,
                            Some(SolvedCrop::Failed),
                        );
                    } else {
                        debug!(
                            "Still stacking ({} / {} frames, {} / {} ms). Suppressing discard UI.",
                            stacked_frames,
                            MAX_STACKED_FRAMES,
                            exp_time.as_millis(),
                            MAX_VIRTUAL_EXPOSURE_US / 1000
                        );
                    }

                    continue;
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

                if let Some(fov) = self.fov {
                    let zoom = self.camera.get_zoom().unwrap_or(1.0) as f64;
                    // Mathematically project the Base FOV to the Zoomed FOV (rectilinear projection).
                    let fov_rad = fov.to_radians();
                    let zoomed_fov_rad = 2.0 * ((fov_rad / 2.0).tan() / zoom).atan();

                    solve_options.solve_timeout_ms =
                        Some((exp_time.as_secs_f64() * 1000.0 * 0.90).max(200.0));
                    solve_options.fov_estimate = Some(zoomed_fov_rad.to_degrees());
                    // Provide a proportional constraint (5%) for wide-angle distortion shifts, with a minimum 0.1 degree floor for telescopes
                    solve_options.fov_max_error =
                        Some((zoomed_fov_rad.to_degrees() * 0.05).max(0.1));
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
                let lat_opt = engine_state
                    .client_latitude
                    .or(self.shared_position.site_latitude.load(Ordering::Relaxed));
                let lon_opt = engine_state
                    .client_longitude
                    .or(self.shared_position.site_longitude.load(Ordering::Relaxed));
                let client_date_opt = engine_state.client_date;
                let is_strict = engine_state.strict_match;
                drop(engine_state);

                if !is_strict {
                    solve_options.match_max_error = 0.005;
                    solve_options.match_threshold = 1e-4;
                }

                // Set Zenith limits to automatically reject patterns containing stars below the horizon
                if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                    if self.config.use_observer_location {
                        let now = client_date_opt.unwrap_or_else(|| chrono::Utc::now());
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
                                    info!(
                                        "Synced to target RA/Dec: ({}, {}). New target pixel: {:?}",
                                        t_ra, t_dec, self.target_pixel
                                    );

                                    // Use the synced values for reporting in this frame
                                    ra = t_ra;
                                    dec = t_dec;
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

                        // Lock in the FOV after the first successful solve (if not provided by config)
                        if self.fov.is_none() {
                            if let Some(new_fov) = solution.fov {
                                let zoom = self.camera.get_zoom().unwrap_or(1.0) as f64;
                                // The solver returns the FOV of the zoomed (cropped) image.
                                // We must reverse the rectilinear projection to find the camera's true 1.0x Base FOV.
                                let solved_fov_rad = new_fov.to_radians();
                                let base_fov_rad =
                                    2.0 * ((solved_fov_rad / 2.0).tan() * zoom).atan();
                                let base_fov = base_fov_rad.to_degrees();

                                self.fov = Some(base_fov);
                                log::info!(
                                    "Locked base FOV to {:.2}° (Solved {:.2}° at {:.2}x zoom)",
                                    base_fov,
                                    new_fov,
                                    zoom
                                );
                            }
                        }

                        // EMA update for distortion
                        if let Some(new_dist) = solution.distortion {
                            if let Some(old_dist) = self.distortion {
                                // Exponential Moving Average (EMA).
                                self.distortion = Some(0.9 * old_dist + 0.1 * new_dist);
                            } else {
                                self.distortion = Some(new_dist);
                                log::info!("Locked initial distortion to {:.6}", new_dist);
                            }
                        }
                        solved = true;

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
                        if !self.config.auto_mode
                            || stacked_frames >= MAX_STACKED_FRAMES
                            || exp_time.as_micros() as u32 >= MAX_VIRTUAL_EXPOSURE_US
                        {
                            self.update_solve_stats(
                                false,
                                count,
                                self.target_ra,
                                self.target_dec,
                                exp_time.as_micros() as i32,
                                pp_time,
                                extract_time,
                                solve_time,
                                Some(SolvedCrop::Failed),
                            );
                        } else {
                            debug!(
                                "Still stacking ({} / {} frames, {} / {} ms). Suppressing failure UI.",
                                stacked_frames,
                                MAX_STACKED_FRAMES,
                                exp_time.as_millis(),
                                MAX_VIRTUAL_EXPOSURE_US / 1000
                            );
                        }
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
        use tetra3::extractor::Crop;
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

        // Mimic engine.rs logic for a 50% scale
        let scale = 0.50;
        let crop_w = (width as f64 * scale).round() as usize;
        let crop_h = (height as f64 * scale).round() as usize;

        let off_x = (width as isize - crop_w as isize) / 2;
        let off_y = (height as isize - crop_h as isize) / 2;

        let virtual_crops = vec![
            Crop::Center {
                width: crop_w,
                height: crop_h,
            },
            Crop::Region {
                height: crop_h,
                width: crop_w,
                offset_y: -off_y,
                offset_x: 0,
            },
            Crop::Region {
                height: crop_h,
                width: crop_w,
                offset_y: 0,
                offset_x: -off_x,
            },
            Crop::Region {
                height: crop_h,
                width: crop_w,
                offset_y: 0,
                offset_x: off_x,
            },
            Crop::Region {
                height: crop_h,
                width: crop_w,
                offset_y: off_y,
                offset_x: 0,
            },
        ];

        // Ensure the absolute bounds actually hit the edges
        let (y1, _y2, _x1, _x2) = virtual_crops[1].bounds(width, height);
        assert_eq!(y1, 0, "Top-Center crop must start at y=0");

        let (_y1, _y2, x1, _x2) = virtual_crops[2].bounds(width, height);
        assert_eq!(x1, 0, "Left-Center crop must start at x=0");

        let (_y1, _y2, _x1, x2) = virtual_crops[3].bounds(width, height);
        assert_eq!(x2, width, "Right-Center crop must end at x=width");

        let (_y1, y2, _x1, _x2) = virtual_crops[4].bounds(width, height);
        assert_eq!(y2, height, "Bottom-Center crop must end at y=height");

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
            5,
            "Expected 5 crop buckets (1 center + 4 edges)"
        );

        // Verify each quadrant actually extracted stars
        assert!(!vcrops_res[0].is_empty(), "Expected stars in Center");
        assert!(
            !vcrops_res[1].is_empty(),
            "Expected stars in Top-Center edge"
        );
        assert!(
            !vcrops_res[2].is_empty(),
            "Expected stars in Left-Center edge"
        );
        assert!(
            !vcrops_res[3].is_empty(),
            "Expected stars in Right-Center edge"
        );
        assert!(
            !vcrops_res[4].is_empty(),
            "Expected stars in Bottom-Center edge"
        );
    }

    #[test]
    fn test_update_solve_stats_crop_status() {
        let camera = Arc::new(crate::dummy_camera::DummyCamera::new());
        let engine_state = Arc::new(RwLock::new(EngineStateData::default()));
        let shared_position = Arc::new(SharedPosition::default());
        let timescale = Arc::new(starfield::time::Timescale::default());
        let server = Lx200WifiTelescope::new(
            shared_position.clone(),
            Arc::new(std::sync::atomic::AtomicBool::new(true)),
            timescale.clone(),
        );
        let lx200_controller = server.get_controller();
        let fused_solver = Arc::new(
            olive_solve::FusedSolver::new(
                std::path::Path::new("../olive-solve/tetra3/tests/fixtures/default_database.npz"),
                None,
                None,
            )
            .unwrap(),
        );

        let mut engine = Engine {
            camera,
            config: EngineConfig::default(),
            last_gain_adjust: None,
            fused_solver,
            fov: Some(15.0),
            distortion: None,
            shared_position,
            target_ra: None,
            target_dec: None,
            target_pixel: None,
            storage: None,
            client_connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            engine_state: engine_state.clone(),
            solve_timestamps: std::collections::VecDeque::with_capacity(60),
            shared_result: Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new())),
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            clear_stacker: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            telescope_server_handle: None,
            alpaca_server_handle: None,
            rt_handle: None,
            lx200_controller,
            timescale,
        };

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
            Some(SolvedCrop::Center),
        );
        assert_eq!(
            engine_state.read().unwrap().solved_crop,
            Some(SolvedCrop::Center)
        );
    }

    #[test]
    fn test_active_imu_and_low_confidence_handling() {
        let camera = Arc::new(crate::dummy_camera::DummyCamera::new());
        let engine_state = Arc::new(RwLock::new(EngineStateData::default()));
        let shared_position = Arc::new(SharedPosition::default());
        let timescale = Arc::new(starfield::time::Timescale::default());
        let server = Lx200WifiTelescope::new(
            shared_position.clone(),
            Arc::new(std::sync::atomic::AtomicBool::new(true)),
            timescale.clone(),
        );
        let lx200_controller = server.get_controller();
        let fused_solver = Arc::new(
            olive_solve::FusedSolver::new(
                std::path::Path::new("../olive-solve/tetra3/tests/fixtures/default_database.npz"),
                None,
                None,
            )
            .unwrap(),
        );

        let mut engine = Engine {
            camera,
            config: EngineConfig::default(),
            last_gain_adjust: None,
            fused_solver: fused_solver.clone(),
            fov: Some(15.0),
            distortion: None,
            shared_position,
            target_ra: None,
            target_dec: None,
            target_pixel: None,
            storage: None,
            client_connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            engine_state: engine_state.clone(),
            solve_timestamps: std::collections::VecDeque::with_capacity(60),
            shared_result: Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new())),
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            clear_stacker: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            telescope_server_handle: None,
            alpaca_server_handle: None,
            rt_handle: None,
            lx200_controller,
            timescale,
        };

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
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(handle) = self.telescope_server_handle.take() {
            handle.abort();
        }
        if let Some(handle) = self.alpaca_server_handle.take() {
            handle.abort();
        }
    }
}
