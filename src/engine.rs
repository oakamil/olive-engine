// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::sync::atomic::Ordering;
use std::{
    cmp::min,
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime},
};

use log::{debug, error, info, warn};
use ndarray::{Array2, s};
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
pub const CALIBRATION_INTERVAL: Duration = Duration::from_secs(10);
pub const MAX_RETRIES: usize = 15; // Increased to allow downward floor probing

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
    pub bypass_calibration: bool,
    pub noise_filter: bool,
    pub initial_fov: Option<f32>,
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
            extractor_type: Extractor::Sequential,
            crop: None,
            downsample: 0,
            bypass_calibration: false,
            noise_filter: false,
            initial_fov: None,
        }
    }
}

pub struct IntermediateResult {
    pub count: usize,
    pub exp_time: Duration,
    pub pp_time: Duration,
    pub extract_time: Duration,
    pub centroids_arr: Array2<f64>,
    pub timestamp: SystemTime,
    pub mean_brightness: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum, Serialize, Deserialize)]
pub enum Extractor {
    #[default]
    Sequential,
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
    pub requested_target_ra: Option<f64>,
    pub requested_target_dec: Option<f64>,
    pub requested_align_target_pixel: Option<(f64, f64)>,
    pub is_imu_estimate: bool,
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
            requested_target_ra: None,
            requested_target_dec: None,
            requested_align_target_pixel: None,
            is_imu_estimate: false,
        }
    }
}

#[allow(dead_code)]
pub struct Engine {
    camera: Arc<dyn CameraTrait>,
    config: EngineConfig,
    pub fused_solver: Arc<olive_solve::FusedSolver>,
    fov: Option<f64>,

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
            .await
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

        let engine_state_clone = engine_state.clone();
        let shared_pos_clone = shared_position.clone();
        let fused_solver_clone = fused_solver.clone();
        let is_running_clone = is_running.clone();
        let timescale_clone = timescale.clone();

        let polling_task = async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(33));
            while is_running_clone.load(std::sync::atomic::Ordering::Relaxed) {
                interval.tick().await;
                if let Some(pos) = fused_solver_clone.get_latest_position().await {
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
            fused_solver,
            fov: config_fov,
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
            telescope_server_handle: Some(telescope_server_handle),
            alpaca_server_handle: Some(alpaca_server_handle),
            rt_handle,
            lx200_controller,
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
    ) {
        if solved {
            self.solve_timestamps.push_back(Instant::now());
        }
        self.solve_timestamps
            .retain(|t| t.elapsed() <= Duration::from_secs(5));
        let solve_rate_hz = self.solve_timestamps.len() as f64 / 5.0;

        let mut state = self.engine_state.write().unwrap();
        state.centroids = centroids;
        state.target_ra = target_ra;
        state.target_dec = target_dec;
        state.exposure_us = exposure_us;
        state.post_process_ms = post_process.as_secs_f64() * 1000.0;
        state.extract_ms = extract.as_secs_f64() * 1000.0;
        state.solve_ms = solve.as_secs_f64() * 1000.0;
        state.solve_rate_hz = solve_rate_hz;
        state.fov = self.fov;
    }

    pub fn start(self) -> (std::thread::JoinHandle<()>, std::thread::JoinHandle<()>) {
        let camera_ext = self.camera.clone();
        let config_ext = self.config.clone();
        let shared_res_ext = self.shared_result.clone();
        let engine_state_ext = self.engine_state.clone();
        let is_running_ext = self.is_running.clone();
        let client_connected_ext = self.client_connected.clone();

        let t1 = std::thread::spawn(move || {
            let mut extractor: Option<FastExtractor> = None;
            let mut last_processed_timestamp = None;
            let mut active_max_area = 150;
            let mut fov_locked_in = false;
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

                {
                    let mut state = engine_state_ext.write().unwrap();
                    state.exposure_us = exposure_us;
                }

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
                let mut centroids_arr;

                let (state_sigma, state_fov) = {
                    let st = engine_state_ext.read().unwrap();
                    (st.current_sigma, st.fov)
                };

                match config_ext.extractor_type {
                    Extractor::Sequential | Extractor::Fast => {
                        let array_img =
                            match Array2::from_shape_vec((height, width), img_data.clone()) {
                                Ok(arr) => arr,
                                Err(e) => {
                                    error!("Failed to create Array2 from image data: {}", e);
                                    continue;
                                }
                            };

                        let active_sigma = state_sigma.unwrap_or(3.5);
                        let mut update_max_area = None;
                        if !fov_locked_in {
                            let best_fov = state_fov.or(config_ext.initial_fov.map(|f| f as f64));
                            if let Some(fov) = best_fov {
                                if fov > 0.0 {
                                    let ratio = (width as f64) / fov;
                                    active_max_area =
                                        (20.0 + 0.03 * (ratio * ratio)).round() as usize;
                                    fov_locked_in = true;
                                    update_max_area = Some(active_max_area);
                                }
                            }
                        }

                        if extractor.is_none() {
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

                            let bg_sub_mode = match config_ext.bg_mode {
                                BgMode::Global => Some(FastBgSubMode::GlobalMedian),
                                BgMode::Block => {
                                    Some(FastBgSubMode::BlockMedian { block_size: 32 })
                                }
                                BgMode::Line => Some(FastBgSubMode::LineMedian),
                            };

                            let options = FastExtractOptions {
                                sigma: active_sigma,
                                downsample,
                                bg_sub_mode,
                                binary_open: config_ext.noise_filter,
                                min_area: Some(2),
                                max_area: Some(active_max_area),
                                crop: final_crop,
                                //centroid_window: Some(3),
                                ..FastExtractOptions::default()
                            };
                            extractor = Some(FastExtractor::new(width, height, options));
                        }

                        let ext = extractor.as_mut().unwrap();
                        ext.update_options(tetra3::fast_extractor::FastExtractOptionsUpdate {
                            sigma: Some(active_sigma),
                            max_area: update_max_area,
                            ..Default::default()
                        });

                        let result = if config_ext.extractor_type == Extractor::Sequential {
                            ext.extract_sequential(&array_img)
                        } else {
                            ext.extract(&array_img)
                        };

                        num_centroids = result.len();
                        centroids_arr = Array2::<f64>::zeros((num_centroids, 2));
                        for (i, c) in result.iter().enumerate() {
                            centroids_arr[[i, 0]] = c.y;
                            centroids_arr[[i, 1]] = c.x;
                        }
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
                                cropped.extend_from_slice(&img_data[start..end]);
                            }
                            cropped
                        } else {
                            img_data.clone()
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
                            centroids_arr = Array2::<f64>::zeros((num_centroids, 2));
                            for (i, c) in stars.iter().enumerate() {
                                centroids_arr[[i, 0]] = c.centroid_y + crop_y as f64;
                                centroids_arr[[i, 1]] = c.centroid_x + crop_x as f64;
                            }
                        } else {
                            error!("Failed to create ImageBuffer from image data");
                            continue;
                        }
                    }
                }

                let target = config_ext.target_centroids as usize;
                let needs_brightness =
                    num_centroids < 4 || num_centroids < (target as f64 * 0.75) as usize;
                let mean_brightness = if needs_brightness {
                    let step = 256;
                    let count = img_data.len() / step;
                    if count > 0 {
                        let sum: u64 = img_data.iter().step_by(step).map(|&p| p as u64).sum();
                        sum as f64 / count as f64
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let extract_time = extract_start.elapsed();

                // Calculate focus metric only when camera mode is active (manual focus).
                // This prevents wasting CPU cycles during active plate solving.
                let is_paused = engine_state_ext.read().unwrap().paused;
                let mut hfr = if client_connected_ext.load(std::sync::atomic::Ordering::Relaxed)
                    || is_paused
                {
                    calculate_hfr(&img_data, width, height, &centroids_arr, 30)
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
                    state.centroids = num_centroids;
                }

                let intermediate = IntermediateResult {
                    count: num_centroids,
                    exp_time: Duration::from_micros(exposure_us as u64),
                    pp_time: post_process_time,
                    extract_time,
                    centroids_arr,
                    timestamp: img_timestamp,
                    mean_brightness,
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

    fn wait_for_extraction(
        &self,
    ) -> Option<(
        usize,
        Duration,
        Duration,
        Duration,
        Array2<f64>,
        SystemTime,
        f64,
    )> {
        let (lock, cvar) = &*self.shared_result;
        let mut inner = lock.lock().unwrap();

        let result = cvar
            .wait_timeout(inner, Duration::from_millis(100))
            .unwrap();
        inner = result.0;

        if let Some(res) = inner.take() {
            Some((
                res.count,
                res.exp_time,
                res.pp_time,
                res.extract_time,
                res.centroids_arr,
                res.timestamp,
                res.mean_brightness,
            ))
        } else {
            None
        }
    }

    pub fn run_calibration_loop(&mut self) {
        info!("Starting auto-calibration loop...");
        self.update_status("Calibration");

        info!("Waiting for stream initialization...");
        while self.wait_for_extraction().is_none() {
            if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
        }

        // Apply static hardware settings ONCE at startup
        let initial_gain = self.config.gain;
        if let Some(g) = initial_gain {
            info!("Setting base hardware settings...");
            self.camera.set_gain(g);
        }

        if self.config.bypass_calibration {
            info!("Bypassing calibration loop based on configuration.");
            return;
        }

        let target_min = self.config.target_centroids;
        let target_max = (self.config.target_centroids as f64 * 1.25).round() as usize;

        let mut was_connected = false;

        loop {
            if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                return;
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
                info!("Client disconnected. Restarting calibration hardware reset.");
                was_connected = false;
                self.camera.restore_state();
                let g = initial_gain.unwrap_or(self.camera.get_gain_range().2);
                self.camera.set_gain(g);
            }

            // Immediate UI update for exposure and status
            self.update_status("Calibration");
            self.update_solve_stats(
                false,
                0,
                None,
                None,
                self.camera.get_exposure(), // Pull actual starting exposure
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
            );

            let mut calibrated = false;
            let mut has_saved_state = false;

            for retry in 0..MAX_RETRIES {
                if self
                    .client_connected
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    break;
                }

                self.update_status(&format!(
                    "Calibration (Attempt {}/{})",
                    retry + 1,
                    MAX_RETRIES
                ));

                info!(
                    "Calibration attempt {}/{} (Sensitivity State -> Exp: {} us, Gain: {:.2})",
                    retry + 1,
                    MAX_RETRIES,
                    self.camera.get_exposure(),
                    self.camera.get_gain()
                );

                let mut total_centroids = 0;
                let mut last_centroids_arr = None;
                let mut valid_samples = 0;
                let required_samples = if self.camera.get_exposure() < 500_000 {
                    3
                } else {
                    1
                };

                // Fetch frames until we verify the hardware has applied our new setting
                while valid_samples < required_samples {
                    if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    if self
                        .client_connected
                        .load(std::sync::atomic::Ordering::Relaxed)
                    {
                        break;
                    }

                    if let Some((
                        count,
                        exp_time,
                        pp_time,
                        extract_time,
                        arr,
                        _timestamp,
                        _mean_brightness,
                    )) = self.wait_for_extraction()
                    {
                        self.update_solve_stats(
                            false,
                            count,
                            self.target_ra,
                            self.target_dec,
                            self.camera.get_exposure(),
                            pp_time,
                            extract_time,
                            Duration::ZERO,
                        );

                        let actual_exp = exp_time.as_micros() as i32;

                        // Hardware quantization tolerance (5% or 100us, whichever is larger)
                        let tolerance =
                            (self.camera.get_exposure() as f32 * 0.05).max(100.0) as i32;

                        if (actual_exp - self.camera.get_exposure()).abs() > tolerance {
                            // Frame is from the old exposure pipeline. Skip it without spamming the logs.
                            continue;
                        }

                        total_centroids += count;
                        last_centroids_arr = Some(arr);
                        valid_samples += 1;
                    } else {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }

                if self
                    .client_connected
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    break;
                }

                // Use floating point math to prevent truncation (e.g. 11.8 becoming 11)
                let avg_centroids =
                    (total_centroids as f64 / valid_samples as f64).round() as usize;
                info!(
                    "Averaged {} centroids over {} images. (Target: {}-{})",
                    avg_centroids, valid_samples, target_min, target_max
                );

                let mut needs_standard_adjustment = true;

                // Sanity check: If we have at least the minimum centroids, make sure it actually solves
                if avg_centroids >= target_min {
                    info!(
                        "Centroid count >= min ({}). Running sanity check...",
                        avg_centroids
                    );

                    let camera_config = self.camera.get_config().unwrap();
                    let size = (camera_config.height as f64, camera_config.width as f64);

                    if let Some(arr) = &last_centroids_arr {
                        let solve_start = Instant::now();
                        let solution = futures::executor::block_on(self.fused_solver.solve_from_centroids(&arr.slice(s![..min(40, arr.nrows()), ..]).to_owned(), size, SolveOptions::default(), None))
                            .expect("Critical Error: solver failed to execute during calibration. Is the database path valid?");
                        let solve_time = solve_start.elapsed();

                        if matches!(solution.status, SolveStatus::MatchFound) {
                            self.fov = solution.fov;
                            self.update_solve_stats(
                                true,
                                0,
                                None,
                                None,
                                self.camera.get_exposure(),
                                Duration::ZERO,
                                Duration::ZERO,
                                solve_time,
                            );
                            info!("Sanity check passed! FOV: {:?}", self.fov);

                            // Now check if we are below the ceiling
                            if avg_centroids <= target_max {
                                // Save current optimal state
                                self.camera.save_state();
                                has_saved_state = true;
                                info!(
                                    "Target range reached and verified. Testing a 10% lower sensitivity to find the optimal floor..."
                                );

                                let applied = self.camera.adjust_sensitivity(0.9);
                                if !applied {
                                    info!(
                                        "Sensitivity cannot be lowered further. Calibration complete."
                                    );
                                    calibrated = true;
                                    break;
                                }
                                needs_standard_adjustment = false;
                            } else {
                                if retry == MAX_RETRIES - 1 {
                                    info!(
                                        "Centroid count {} is high but sanity check passed. Max retries reached, accepting current state as calibration.",
                                        avg_centroids
                                    );
                                    calibrated = true;
                                    break;
                                } else {
                                    info!(
                                        "Centroid count {} is too high (max {}). Scaling down sensitivity...",
                                        avg_centroids, target_max
                                    );
                                    // Let it fall through to standard downward adjustment logic
                                }
                            }
                        } else {
                            warn!(
                                "Sanity check failed (Status: {:?}). Camera may not be pointing at the sky.",
                                solution.status
                            );

                            if has_saved_state {
                                info!(
                                    "Sanity check failed after step-down. Reverting to optimal floor state. Calibration complete."
                                );
                                self.camera.restore_state();
                                calibrated = true;
                                break;
                            } else {
                                break; // Normal failure, escape the retry loop
                            }
                        }
                    }
                } else {
                    // avg_centroids < target_min
                    if has_saved_state {
                        info!(
                            "Centroid count {} fell below target minimum ({}). Reverting to optimal floor state. Calibration complete.",
                            avg_centroids, target_min
                        );
                        self.camera.restore_state();
                        calibrated = true;
                        break;
                    }
                }

                if needs_standard_adjustment {
                    // Finer Adjustment: We only move 33% of the way to the "ideal" mathematical factor.
                    // This dampens the change so we creep up on the target range and settle lower.
                    let raw_factor =
                        self.config.target_centroids as f64 / (avg_centroids.max(1)) as f64;
                    let factor = 1.0 + (raw_factor - 1.0) * 0.33;

                    let applied = self.camera.adjust_sensitivity(factor);

                    if !applied {
                        info!("Sensitivity limit reached. Checking if it solves...");
                        let camera_config = self.camera.get_config().unwrap();
                        let size = (camera_config.height as f64, camera_config.width as f64);

                        if let Some(arr) = &last_centroids_arr {
                            let solution = futures::executor::block_on(
                                self.fused_solver.solve_from_centroids(
                                    &arr.slice(s![..min(40, arr.nrows()), ..]).to_owned(),
                                    size,
                                    SolveOptions::default(),
                                    None,
                                ),
                            )
                            .expect("Critical Error: solver failed to execute during calibration.");

                            if matches!(solution.status, SolveStatus::MatchFound) {
                                self.fov = solution.fov;
                                info!(
                                    "Solves at limit! FOV: {:?}. Calibration complete.",
                                    self.fov
                                );
                                calibrated = true;
                                break;
                            }
                        }

                        warn!(
                            "Hardware/config limits reached and still outside target range. Aborting attempt."
                        );
                        break; // We hit our limits, break the retry loop and sleep
                    }
                }
            }

            // If we exhausted all retries but had successfully verified a state, use it!
            if !calibrated {
                if has_saved_state {
                    info!(
                        "Max retries reached during floor search. Reverting to optimal floor state. Calibration complete."
                    );
                    self.camera.restore_state();
                    calibrated = true;
                }
            }

            if calibrated {
                break;
            }

            if self
                .client_connected
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                continue;
            }

            info!(
                "Calibration failed or in progress, waiting {:?}...",
                CALIBRATION_INTERVAL
            );

            // Responsive wait that checks for connections
            let wait_start = Instant::now();
            while wait_start.elapsed() < CALIBRATION_INTERVAL {
                if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                if self
                    .client_connected
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    break;
                }
                std::thread::sleep(Duration::from_millis(250));
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
        );
        let mut consecutive_failures = 0;
        let mut observer_location_set = false;

        // Rolling average buffer to smooth out frame-to-frame scintillation dips
        let mut recent_centroids: Vec<usize> = Vec::with_capacity(5);
        let mut was_connected = false;

        loop {
            if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
                return;
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

            if let Some((
                count,
                exp_time,
                pp_time,
                extract_time,
                centroids_arr,
                img_timestamp,
                mean_brightness,
            )) = self.wait_for_extraction()
            {
                let mut pending_sync = None;
                {
                    if let Ok(mut st) = self.engine_state.write() {
                        if let (Some(ra), Some(dec)) =
                            (st.requested_target_ra, st.requested_target_dec)
                        {
                            self.target_ra = Some(ra);
                            self.target_dec = Some(dec);
                            st.requested_target_ra = None;
                            st.requested_target_dec = None;

                            // Let the shared position know we are slewing so external clients get updated state if any
                            self.shared_position
                                .target_op
                                .store(Some(TargetOp::Slew), std::sync::atomic::Ordering::Relaxed);
                            self.shared_position
                                .is_slewing
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                            info!("Engine received target request: RA={}, Dec={}", ra, dec);
                        }

                        if let Some((x, y)) = st.requested_align_target_pixel {
                            self.target_pixel = Some((x, y));
                            self.save_target_pixel(self.target_pixel);
                            st.requested_align_target_pixel = None;
                            info!("Engine received align target pixel request: ({}, {})", x, y);
                        }
                    }

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
                        self.shared_position
                            .target_op
                            .store(None, std::sync::atomic::Ordering::Relaxed);
                        self.shared_position
                            .is_slewing
                            .store(false, std::sync::atomic::Ordering::Relaxed);
                    } else if target_ra.is_some() && target_dec.is_some() {
                        let t_ra = target_ra.unwrap();
                        let t_dec = target_dec.unwrap();
                        if target_op == Some(TargetOp::Sync) {
                            pending_sync = Some((t_ra, t_dec));
                        } else if target_op == Some(TargetOp::Slew) {
                            self.target_ra = Some(t_ra);
                            self.target_dec = Some(t_dec);
                            self.shared_position
                                .target_op
                                .store(None, std::sync::atomic::Ordering::Relaxed);
                            self.shared_position
                                .is_slewing
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }

                if count < MIN_CENTROIDS {
                    warn!(
                        "Discarding frame: {} centroids (< {} min). Telescope might be moving.",
                        count, MIN_CENTROIDS
                    );
                    self.update_solve_stats(
                        false,
                        count,
                        self.target_ra,
                        self.target_dec,
                        exp_time.as_micros() as i32,
                        Duration::ZERO,
                        extract_time,
                        Duration::ZERO,
                    );

                    // Recovery check if we are lost
                    if consecutive_failures > 3 {
                        if mean_brightness < 20.0 {
                            if self.camera.adjust_sensitivity(1.5) {
                                recent_centroids.clear();
                            }
                        } else if mean_brightness > 220.0 {
                            if self.camera.adjust_sensitivity(0.5) {
                                recent_centroids.clear();
                            }
                        }
                    }

                    continue;
                }

                let mut solved = false;
                let solve_start = Instant::now();
                let camera_config = self.camera.get_config().unwrap();
                let width = camera_config.width as f64;
                let height = camera_config.height as f64;
                let size = (height, width);

                let mut solve_options = SolveOptions::default();
                if let Some(fov) = self.fov {
                    solve_options.solve_timeout_ms = Some(exp_time.as_secs_f64() * 1000.0 * 0.90);
                    solve_options.fov_estimate = Some(fov);
                    solve_options.fov_max_error = Some(0.5); // Provide a reasonable constraint based on FOV
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
                drop(engine_state);

                if !observer_location_set {
                    if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                        futures::executor::block_on(
                            self.fused_solver.set_observer_location(lat, lon),
                        );
                        observer_location_set = true;
                    }
                }

                let solution = futures::executor::block_on(self.fused_solver.solve_from_centroids(&centroids_arr.slice(s![..min(40, centroids_arr.nrows()), ..]).to_owned(), size, solve_options, Some(img_timestamp)))
                    .expect("Critical Error: solver failed to execute during solving loop. Is the database accessible?");

                let solve_time = solve_start.elapsed();

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

                        // Update FOV if available
                        if let Some(new_fov) = solution.fov {
                            self.fov = Some(new_fov);
                        }
                        solved = true;

                        self.update_solve_stats(
                            true,
                            count,
                            self.target_ra,
                            self.target_dec,
                            self.camera.get_exposure(),
                            pp_time,
                            extract_time,
                            solve_time,
                        );
                    }
                    _ => {
                        self.update_solve_stats(
                            false,
                            count,
                            self.target_ra,
                            self.target_dec,
                            self.camera.get_exposure(),
                            pp_time,
                            extract_time,
                            solve_time,
                        );
                        warn!(
                            "No Solution (Status: {:?}) | Centroids: {} | Exposure: {:?} | Post-Process: {:?} | Extract: {:?} | Solve: {:?}",
                            solution.status, count, exp_time, pp_time, extract_time, solve_time
                        );
                    }
                }

                let actual_exp = exp_time.as_micros() as i32;
                // Tighter 1% tolerance prevents overlapping with our 5% adjustment steps (stops ratcheting)
                let tolerance = (self.camera.get_exposure() as f32 * 0.01).max(500.0) as i32;

                // For UVC, actual_exp will match get_exposure() immediately since it is static, passing naturally.
                let matches_current = (actual_exp - self.camera.get_exposure()).abs() <= tolerance;

                // Update our continuous failure tracker
                if solved {
                    consecutive_failures = 0;
                } else if matches_current {
                    consecutive_failures += 1;
                }

                // Continuous exposure/gain adjustment
                if !self.config.bypass_calibration && matches_current {
                    let target = self.config.target_centroids as usize;

                    if recent_centroids.len() >= 5 {
                        recent_centroids.remove(0);
                    }
                    recent_centroids.push(count);

                    if recent_centroids.len() >= 5 {
                        let avg_count = (recent_centroids.iter().sum::<usize>() as f64
                            / recent_centroids.len() as f64)
                            .round() as usize;

                        let lower_bound = (target as f64 * 0.75) as usize;
                        let upper_bound = (target as f64 * 1.5) as usize;

                        let mut factor: Option<f64> = None;

                        if avg_count < lower_bound {
                            let f = target as f64 / avg_count.max(1) as f64;
                            factor = Some(f.min(1.5)); // max 50% increase
                        } else if avg_count > upper_bound {
                            let f = target as f64 / avg_count as f64;
                            factor = Some(f.max(0.5)); // max 50% decrease
                        }

                        if let Some(mut f) = factor {
                            // Bright sky check
                            if f > 1.0 && mean_brightness > 200.0 {
                                f = 1.0;
                            }
                            if f != 1.0 {
                                let applied = self.camera.adjust_sensitivity(f);
                                if applied {
                                    recent_centroids.clear();
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

    let num_centroids = result.len();
    if num_centroids == 0 {
        return 0.0;
    }

    let mut centroids_arr = Array2::<f64>::zeros((num_centroids, 2));
    for (i, c) in result.iter().enumerate() {
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
