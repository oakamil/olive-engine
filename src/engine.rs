// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use chrono::{Datelike, Timelike};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use log::{error, info, warn};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::CameraTrait;
use crate::telescope_server::{Lx200Telescope, Lx200WifiTelescope, SharedPosition, TargetOp};
use tetra3::{
    fast_extractor::{FastBgSubMode, FastDownsample, FastExtractOptions, FastExtractor},
    solver::{SolveOptions, SolveStatus},
    tetra3::Tetra3,
};

pub const MIN_CENTROIDS: usize = 4;
pub const CALIBRATION_INTERVAL: Duration = Duration::from_secs(10);
pub const MAX_RETRIES: usize = 15; // Increased to allow downward floor probing
pub const CALIBRATION_EXPOSURE_US: i32 = 50000;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub min_exposure_us: i32,
    pub max_exposure_us: i32,
    pub gain: Option<f32>,
    pub target_centroids: usize,
    pub database_path: String,
    pub isp: bool,
    pub bg_mode: BgMode,
    pub extractor_type: Extractor,
    pub uvc: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            min_exposure_us: 16670,
            max_exposure_us: 1000000,
            gain: None,
            target_centroids: 20,
            database_path: format!(
                "{}/data/default_database.npz",
                std::env::var("HOME").unwrap_or_default()
            ),
            isp: false,
            bg_mode: BgMode::Line,
            extractor_type: Extractor::Sequential,
            uvc: false,
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
        }
    }
}

pub struct Engine {
    camera: Arc<dyn CameraTrait>,
    config: EngineConfig,
    solver: Tetra3,
    fov: Option<f64>,
    imu: Option<Arc<crate::imu::Imu>>,

    shared_position: Arc<std::sync::RwLock<SharedPosition>>,
    target_ra: Option<f64>,
    target_dec: Option<f64>,
    target_pixel: Option<(f64, f64)>,
    client_connected: Arc<std::sync::atomic::AtomicBool>,
    engine_state: Arc<RwLock<EngineStateData>>,
    solve_timestamps: std::collections::VecDeque<Instant>,
    shared_result: Arc<(
        std::sync::Mutex<Option<IntermediateResult>>,
        std::sync::Condvar,
    )>,
}

impl Engine {
    pub fn new(
        camera: Arc<dyn CameraTrait>,
        config: EngineConfig,
        client_connected: Arc<std::sync::atomic::AtomicBool>,
        engine_state: Arc<RwLock<EngineStateData>>,
        imu: Option<Arc<crate::imu::Imu>>,
    ) -> Self {
        let solver = Tetra3::new(&config.database_path);
        let shared_position = Arc::new(RwLock::new(SharedPosition::default()));

        let mut server = Lx200WifiTelescope::new(shared_position.clone());
        tokio::spawn(async move {
            if let Err(e) = server.serve_requests().await {
                error!("Telescope server error: {}", e);
            }
        });

        Self {
            camera,
            config,
            solver,
            fov: None,
            imu,
            shared_position,
            target_ra: None,
            target_dec: None,
            target_pixel: None,
            client_connected,
            engine_state,
            solve_timestamps: std::collections::VecDeque::new(),
            shared_result: Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new())),
        }
    }

    fn clamp_exposure(&self, exposure: i32) -> i32 {
        exposure.clamp(self.config.min_exposure_us, self.config.max_exposure_us)
    }

    fn update_status(&self, status: &str) {
        log::debug!("Engine status changing to: {}", status);
        let mut state = self.engine_state.write().unwrap();
        state.status = status.to_string();
    }

    fn update_solve_stats(
        &mut self,
        solved: bool,
        ra: Option<f64>,
        dec: Option<f64>,
        roll: Option<f64>,
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
        state.ra = ra;
        state.dec = dec;
        state.roll = roll;
        state.centroids = centroids;
        state.target_ra = target_ra;
        state.target_dec = target_dec;
        state.exposure_us = exposure_us;
        state.post_process_ms = post_process.as_secs_f64() * 1000.0;
        state.extract_ms = extract.as_secs_f64() * 1000.0;
        state.solve_ms = solve.as_secs_f64() * 1000.0;
        state.solve_rate_hz = solve_rate_hz;
    }

    pub fn start(self) {
        let camera_ext = self.camera.clone();
        let config_ext = self.config.clone();
        let shared_res_ext = self.shared_result.clone();
        let engine_state_ext = self.engine_state.clone();

        std::thread::spawn(move || {
            let mut extractor: Option<FastExtractor> = None;
            let mut last_processed_timestamp = None;

            loop {
                let img_arc = camera_ext.wait_for_next_capture(last_processed_timestamp);
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

                match config_ext.extractor_type {
                    Extractor::Sequential | Extractor::Fast => {
                        let array_img = match Array2::from_shape_vec((height, width), img_data) {
                            Ok(arr) => arr,
                            Err(e) => {
                                error!("Failed to create Array2 from image data: {}", e);
                                continue;
                            }
                        };

                        if extractor.is_none() {
                            let downsample = if width * height > 1_000_000 {
                                FastDownsample::X2
                            } else {
                                FastDownsample::None
                            };

                            let bg_sub_mode = match config_ext.bg_mode {
                                BgMode::Global => Some(FastBgSubMode::GlobalMedian),
                                BgMode::Block => {
                                    Some(FastBgSubMode::BlockMedian { block_size: 32 })
                                }
                                BgMode::Line => Some(FastBgSubMode::LineMedian),
                            };

                            let options = FastExtractOptions {
                                sigma: 3.5,
                                downsample,
                                bg_sub_mode,
                                binary_open: false,
                                min_area: Some(1),
                                max_area: Some(20),
                                //centroid_window: Some(3),
                                ..FastExtractOptions::default()
                            };
                            extractor = Some(FastExtractor::new(width, height, options));
                        }

                        let ext = extractor.as_mut().unwrap();

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
                        if let Some(gray_img) =
                            image::ImageBuffer::from_raw(width as u32, height as u32, img_data)
                        {
                            let noise_estimate =
                                cedar_detect::algorithm::estimate_noise_from_image(&gray_img);
                            let (stars, _, _, _) = cedar_detect::algorithm::get_stars_from_image(
                                &gray_img,
                                noise_estimate,
                                8.0,                                            // sigma
                                true,                                           // normalize_rows
                                if width * height > 1_000_000 { 2 } else { 1 }, // binning
                                false,                                          // detect_hot_pixels
                                false, // return_binned_image
                            );

                            num_centroids = stars.len();
                            centroids_arr = Array2::<f64>::zeros((num_centroids, 2));
                            for (i, c) in stars.iter().enumerate() {
                                centroids_arr[[i, 0]] = c.centroid_y;
                                centroids_arr[[i, 1]] = c.centroid_x;
                            }
                        } else {
                            error!("Failed to create ImageBuffer from image data");
                            continue;
                        }
                    }
                }

                let extract_time = extract_start.elapsed();

                let intermediate = IntermediateResult {
                    count: num_centroids,
                    exp_time: Duration::from_micros(exposure_us as u64),
                    pp_time: post_process_time,
                    extract_time,
                    centroids_arr,
                    timestamp: img_timestamp,
                };

                let (lock, cvar) = &*shared_res_ext;
                let mut inner = lock.lock().unwrap();
                *inner = Some(intermediate);
                cvar.notify_all();
            }
        });

        std::thread::spawn(move || {
            let mut engine = self;
            loop {
                engine.run_calibration_loop();
                engine.run_solving_loop();
            }
        });
    }

    fn wait_for_extraction(
        &self,
    ) -> Option<(usize, Duration, Duration, Duration, Array2<f64>, SystemTime)> {
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
            ))
        } else {
            None
        }
    }

    pub fn run_calibration_loop(&mut self) {
        info!("Starting auto-calibration loop...");
        self.update_status("Calibration");

        // Apply static hardware settings ONCE at startup
        let gain = self.config.gain.unwrap_or(self.camera.get_max_gain());
        if !self.config.uvc {
            info!("Setting gain to: {}", gain);
            self.camera.set_gain(gain);
        }
        self.camera.enable_hcg();

        let target_min = self.config.target_centroids;
        let target_max = (self.config.target_centroids as f64 * 1.25).round() as usize;

        let mut imu_calibration_start: Option<Instant> = None;
        let mut first_solve: Option<(f64, f64)> = None;

        let mut was_connected = false;

        loop {
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
                self.camera.set_gain(gain);
                self.camera.enable_hcg();
            }

            let mut current_exposure = CALIBRATION_EXPOSURE_US;
            let mut current_gain = gain;
            
            if self.config.uvc {
                current_gain = current_gain.clamp(self.camera.get_min_gain(), self.camera.get_max_gain());
                self.camera.set_gain(current_gain);
                current_exposure = self.camera.get_exposure();
            } else {
                current_exposure = self.clamp_exposure(current_exposure);
                self.camera.set_exposure(current_exposure);
            }

            // Immediate UI update for exposure and status
            self.update_status("Calibration");
            self.update_solve_stats(
                false,
                None,
                None,
                None,
                0,
                None,
                None,
                current_exposure,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
            );

            let mut calibrated = false;
            let mut last_good_exposure: Option<i32> = None;
            let mut last_good_gain: Option<f32> = None;

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

                if self.config.uvc {
                    info!(
                        "Calibration attempt {}/{} (Gain: {:.2})",
                        retry + 1,
                        MAX_RETRIES,
                        current_gain
                    );
                } else {
                    info!(
                        "Calibration attempt {}/{} (Exposure: {} us)",
                        retry + 1,
                        MAX_RETRIES,
                        current_exposure
                    );
                }

                let mut total_centroids = 0;
                let mut last_centroids_arr = None;
                let mut valid_samples = 0;
                let mut found_new_exposure = false;

                // Fetch frames until we verify the hardware has applied our new exposure, then collect 5
                while valid_samples < 5 {
                    if self
                        .client_connected
                        .load(std::sync::atomic::Ordering::Relaxed)
                    {
                        break;
                    }

                    if let Some((count, exp_time, pp_time, extract_time, arr, _timestamp)) =
                        self.wait_for_extraction()
                    {
                        self.update_solve_stats(
                            false,
                            None,
                            None,
                            None,
                            count,
                            self.target_ra,
                            self.target_dec,
                            current_exposure,
                            pp_time,
                            extract_time,
                            Duration::ZERO,
                        );
                        let actual_exp = exp_time.as_micros() as i32;

                        // Hardware quantization tolerance (5% or 100us, whichever is larger)
                        let tolerance = (current_exposure as f32 * 0.05).max(100.0) as i32;

                        if !found_new_exposure {
                            if self.config.uvc {
                                found_new_exposure = true; // Assume gain applies quickly for UVC
                            } else if (actual_exp - current_exposure).abs() > tolerance {
                                // Frame is from the old exposure pipeline. Skip it without spamming the logs.
                                continue;
                            } else {
                                found_new_exposure = true;
                            }
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
                        let solution = self.solver.solve_from_centroids(arr, size, SolveOptions::default())
                            .expect("Critical Error: solver failed to execute during calibration. Is the database path valid?");
                        let solve_time = solve_start.elapsed();

                        if matches!(solution.status, SolveStatus::MatchFound) {
                            if let Some(imu) = &self.imu {
                                if imu_calibration_start.is_none() {
                                    imu.start_calibration();
                                    imu_calibration_start = Some(Instant::now());
                                    if let (Some(ra), Some(dec)) = (solution.ra, solution.dec) {
                                        first_solve = Some((ra, dec));
                                    }
                                }
                            }
                            self.fov = solution.fov;
                            self.update_solve_stats(
                                true,
                                solution.ra,
                                solution.dec,
                                solution.roll,
                                0,
                                None,
                                None,
                                current_exposure,
                                Duration::ZERO,
                                Duration::ZERO,
                                solve_time,
                            );
                            info!("Sanity check passed! FOV: {:?}", self.fov);

                            // Now check if we are below the ceiling
                            if avg_centroids <= target_max {
                                if self.config.uvc {
                                    last_good_gain = Some(current_gain);
                                    info!("Target range reached and verified at gain {:.2}. Testing a 10% lower gain to find the optimal floor...", current_gain);
                                    let next_gain = (current_gain * 0.9).max(self.camera.get_min_gain());
                                    if (next_gain - current_gain).abs() < 0.1 {
                                        info!("Gain cannot be lowered further. Calibration complete.");
                                        calibrated = true;
                                        break;
                                    }
                                    current_gain = next_gain;
                                    self.camera.set_gain(current_gain);
                                    needs_standard_adjustment = false;
                                } else {
                                    last_good_exposure = Some(current_exposure);
                                    info!(
                                        "Target range reached and verified at {} us. Testing a 10% lower exposure to find the optimal floor...",
                                        current_exposure
                                    );

                                    let next_exposure =
                                        self.clamp_exposure((current_exposure as f64 * 0.9) as i32);

                                    if next_exposure == current_exposure {
                                        info!(
                                            "Exposure cannot be lowered further. Calibration complete."
                                        );
                                        calibrated = true;
                                        break;
                                    }

                                    current_exposure = next_exposure;
                                    self.camera.set_exposure(current_exposure);
                                    needs_standard_adjustment = false;
                                }
                            } else {
                                info!(
                                    "Centroid count {} is too high (max {}). Scaling down exposure...",
                                    avg_centroids, target_max
                                );
                                // Let it fall through to standard downward adjustment logic
                            }
                        } else {
                            warn!(
                                "Sanity check failed (Status: {:?}). Camera may not be pointing at the sky.",
                                solution.status
                            );

                            if self.config.uvc {
                                if let Some(good_gain) = last_good_gain {
                                    info!("Sanity check failed after step-down. Reverting to optimal floor gain: {:.2}. Calibration complete.", good_gain);
                                    current_gain = good_gain;
                                    self.camera.set_gain(current_gain);
                                    calibrated = true;
                                    break;
                                } else {
                                    break;
                                }
                            } else {
                                if let Some(good_exp) = last_good_exposure {
                                    info!(
                                        "Sanity check failed after step-down. Reverting to optimal floor exposure: {} us. Calibration complete.",
                                        good_exp
                                    );
                                    current_exposure = good_exp;
                                    self.camera.set_exposure(current_exposure);
                                    calibrated = true;
                                    break;
                                } else {
                                    break; // Normal failure, escape the retry loop
                                }
                            }
                        }
                    }
                } else {
                    // avg_centroids < target_min
                    if self.config.uvc {
                        if let Some(good_gain) = last_good_gain {
                            info!("Centroid count {} fell below target minimum ({}). Reverting to optimal floor gain: {:.2}. Calibration complete.", avg_centroids, target_min, good_gain);
                            current_gain = good_gain;
                            self.camera.set_gain(current_gain);
                            calibrated = true;
                            break;
                        }
                    } else {
                        if let Some(good_exp) = last_good_exposure {
                            info!(
                                "Centroid count {} fell below target minimum ({}). Reverting to optimal floor exposure: {} us. Calibration complete.",
                                avg_centroids, target_min, good_exp
                            );
                            current_exposure = good_exp;
                            self.camera.set_exposure(current_exposure);
                            calibrated = true;
                            break;
                        }
                    }
                }

                if needs_standard_adjustment {
                    // Finer Adjustment: We only move 33% of the way to the "ideal" mathematical factor.
                    // This dampens the change so we creep up on the target range and settle lower.
                    let raw_factor =
                        self.config.target_centroids as f64 / (avg_centroids.max(1)) as f64;
                    let factor = 1.0 + (raw_factor - 1.0) * 0.33;

                    if self.config.uvc {
                        let next_gain = (current_gain * factor as f32).clamp(self.camera.get_min_gain(), self.camera.get_max_gain());
                        if (next_gain - current_gain).abs() < 0.1 {
                            if next_gain <= self.camera.get_min_gain() + 0.1 {
                                info!("Minimum gain limit reached ({:.2}). Checking if it solves...", next_gain);
                                let camera_config = self.camera.get_config().unwrap();
                                let size = (camera_config.height as f64, camera_config.width as f64);
                                if let Some(arr) = &last_centroids_arr {
                                    let solution = self.solver.solve_from_centroids(arr, size, SolveOptions::default())
                                        .expect("Critical Error: solver failed to execute during calibration.");
                                    if matches!(solution.status, SolveStatus::MatchFound) {
                                        self.fov = solution.fov;
                                        info!("Solves at minimum gain! FOV: {:?}. Calibration complete.", self.fov);
                                        calibrated = true;
                                        break;
                                    }
                                }
                            }
                            warn!("Gain limit reached ({:.2}) and still outside target range. Aborting attempt.", current_gain);
                            break;
                        }
                        current_gain = next_gain;
                        self.camera.set_gain(current_gain);
                    } else {
                        let next_exposure =
                            self.clamp_exposure((current_exposure as f64 * factor) as i32);

                        if next_exposure == current_exposure {
                            if next_exposure == self.config.min_exposure_us {
                                info!(
                                    "Minimum exposure limit reached ({} us). Checking if it solves...",
                                    next_exposure
                                );
                                let camera_config = self.camera.get_config().unwrap();
                                let size = (camera_config.height as f64, camera_config.width as f64);

                                if let Some(arr) = &last_centroids_arr {
                                    let solution = self.solver.solve_from_centroids(arr, size, SolveOptions::default())
                                        .expect("Critical Error: solver failed to execute during calibration.");

                                    if matches!(solution.status, SolveStatus::MatchFound) {
                                        self.fov = solution.fov;
                                        info!(
                                            "Solves at minimum exposure! FOV: {:?}. Calibration complete.",
                                            self.fov
                                        );
                                        calibrated = true;
                                        break;
                                    }
                                }
                            }

                            warn!(
                                "Exposure limit reached ({} us) and still outside target range. Aborting attempt.",
                                current_exposure
                            );
                            break; // We hit our limits, break the retry loop and sleep
                        }

                        current_exposure = next_exposure;
                        self.camera.set_exposure(current_exposure);
                    }
                }
            }

            // If we exhausted all retries but had successfully verified an exposure, use it!
            if !calibrated {
                if self.config.uvc {
                    if let Some(good_gain) = last_good_gain {
                        info!("Max retries reached during floor search. Reverting to optimal floor gain: {:.2}. Calibration complete.", good_gain);
                        current_gain = good_gain;
                        self.camera.set_gain(current_gain);
                        calibrated = true;
                    }
                } else {
                    if let Some(good_exp) = last_good_exposure {
                        info!(
                            "Max retries reached during floor search. Reverting to optimal floor exposure: {} us. Calibration complete.",
                            good_exp
                        );
                        current_exposure = good_exp;
                        self.camera.set_exposure(current_exposure);
                        calibrated = true;
                    }
                }
            }

            if calibrated {
                if let Some(imu) = &self.imu {
                    if let Some(start_time) = imu_calibration_start {
                        let elapsed = start_time.elapsed();
                        if elapsed < Duration::from_secs(5) {
                            info!("Waiting for IMU calibration to reach 5 seconds...");
                            std::thread::sleep(Duration::from_secs(5) - elapsed);
                        }

                        let mut verified = false;
                        info!("Taking final frame for IMU calibration verification...");

                        if let Some((_, _, _, _, arr, _)) = self.wait_for_extraction() {
                            let camera_config = self.camera.get_config().unwrap();
                            let size = (camera_config.height as f64, camera_config.width as f64);
                            if let Ok(solution) = self.solver.solve_from_centroids(
                                &arr,
                                size,
                                SolveOptions::default(),
                            ) {
                                if matches!(solution.status, SolveStatus::MatchFound) {
                                    if let Some((first_ra, first_dec)) = first_solve {
                                        let time_diff = start_time.elapsed().as_secs_f64();
                                        let max_expected_drift_deg =
                                            (15.0 / 3600.0) * time_diff * 1.5;

                                        if let (Some(ra), Some(dec)) = (solution.ra, solution.dec) {
                                            let mut ra_diff = (ra - first_ra).abs();
                                            if ra_diff > 180.0 {
                                                ra_diff = 360.0 - ra_diff;
                                            }
                                            let dec_diff = (dec - first_dec).abs();

                                            let total_drift_deg =
                                                (ra_diff.powi(2) + dec_diff.powi(2)).sqrt();

                                            if total_drift_deg <= max_expected_drift_deg + 0.1 {
                                                verified = true;
                                            } else {
                                                warn!(
                                                    "IMU calibration rejected due to large movement: {:.3} deg drift (expected max {:.3} deg)",
                                                    total_drift_deg, max_expected_drift_deg
                                                );
                                            }
                                        } else {
                                            warn!(
                                                "IMU calibration verification failed: missing ra/dec in final frame"
                                            );
                                        }
                                    } else {
                                        verified = true;
                                    }
                                } else {
                                    warn!(
                                        "IMU calibration verification failed: could not solve final frame"
                                    );
                                }
                            } else {
                                warn!("IMU calibration verification failed: solver error");
                            }
                        } else {
                            warn!(
                                "IMU calibration verification failed: could not extract centroids"
                            );
                        }

                        if verified {
                            imu.stop_calibration();
                        } else {
                            imu.abort_calibration();
                            calibrated = false;
                            imu_calibration_start = None;
                            first_solve = None;
                        }
                    }
                }
            } else {
                if let Some(imu) = &self.imu {
                    if imu_calibration_start.is_some() {
                        imu.abort_calibration();
                        imu_calibration_start = None;
                        first_solve = None;
                    }
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
        let mut current_exposure = self.camera.get_exposure();
        let mut current_gain = self.camera.get_gain();
        self.update_solve_stats(
            false,
            None,
            None,
            None,
            0,
            None,
            None,
            current_exposure,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
        );
        let target_max = (self.config.target_centroids as f64 * 1.5).round() as usize;

        let mut consecutive_failures = 0;
        const MAX_FAILURES: usize = 3;

        // Rolling average buffer to smooth out frame-to-frame scintillation dips
        let mut recent_centroids: Vec<usize> = Vec::with_capacity(5);
        let mut was_connected = false;

        loop {
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

            if let Some((count, exp_time, pp_time, extract_time, centroids_arr, img_timestamp)) =
                self.wait_for_extraction()
            {
                if count < MIN_CENTROIDS {
                    // Frame discarded completely. Not added to buffer, does not count as a failure.
                    // Prevents telescope smearing movement from spiking exposure.
                    warn!(
                        "Discarding frame: {} centroids (< {} min). Telescope might be moving.",
                        count, MIN_CENTROIDS
                    );
                    continue;
                }

                let mut solved = false;
                let solve_start = Instant::now();
                let camera_config = self.camera.get_config().unwrap();
                let width = camera_config.width as f64;
                let height = camera_config.height as f64;
                let size = (height, width);

                let mut solve_options = SolveOptions::default();
                solve_options.solve_timeout_ms = Some(15.0);
                if let Some(fov) = self.fov {
                    solve_options.fov_estimate = Some(fov);
                    solve_options.fov_max_error = Some(0.5); // Provide a reasonable constraint based on FOV
                }

                // Check for pending sync to include in the solve
                let mut pending_sync = None;
                {
                    let locked = self.shared_position.read().unwrap();
                    if locked.target_op == Some(TargetOp::Sync)
                        && locked.target_ra.is_some()
                        && locked.target_dec.is_some()
                    {
                        pending_sync =
                            Some((locked.target_ra.unwrap(), locked.target_dec.unwrap()));
                    }
                }

                if let Some(tp) = self.target_pixel {
                    solve_options.target_pixel =
                        Some(Array2::from_shape_vec((1, 2), vec![tp.1, tp.0]).unwrap());
                }
                if let Some((t_ra, t_dec)) = pending_sync {
                    solve_options.target_sky_coord =
                        Some(Array2::from_shape_vec((1, 2), vec![t_ra, t_dec]).unwrap());
                }

                let solution = self.solver.solve_from_centroids(&centroids_arr, size, solve_options)
                    .expect("Critical Error: solver failed to execute during solving loop. Is the database accessible?");

                let solve_time = solve_start.elapsed();

                match solution.status {
                    SolveStatus::MatchFound => {
                        let mut locked = self.shared_position.write().unwrap();

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
                                    self.target_ra = Some(t_ra);
                                    self.target_dec = Some(t_dec);
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
                            locked.target_op = None;
                            locked.target_ra = None;
                            locked.target_dec = None;
                        }

                        info!(
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

                        locked.telescope_ra = ra;
                        locked.telescope_dec = dec;
                        let lat_opt = locked.site_latitude;
                        let lon_opt = locked.site_longitude;
                        drop(locked);

                        if let Some(imu) = &self.imu {
                            if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                                let dt: chrono::DateTime<chrono::Utc> = img_timestamp.into();
                                let (alt, az, alt_az_roll) =
                                    ra_dec_to_alt_az(ra, dec, solution.roll.unwrap(), lat, lon, dt);
                                let mount_coords = crate::imu::MountCoordinates {
                                    pitch: alt,
                                    yaw: az,
                                    roll: alt_az_roll,
                                };
                                futures::executor::block_on(
                                    imu.update_anchor(&mount_coords, &img_timestamp),
                                );
                            }
                        }

                        self.update_solve_stats(
                            true,
                            Some(ra),
                            Some(dec),
                            solution.roll,
                            count,
                            self.target_ra,
                            self.target_dec,
                            current_exposure,
                            pp_time,
                            extract_time,
                            solve_time,
                        );
                    }
                    _ => {
                        let mut est_ra = None;
                        let mut est_dec = None;
                        let mut est_roll = None;

                        let locked = self.shared_position.read().unwrap();
                        let lat_opt = locked.site_latitude;
                        let lon_opt = locked.site_longitude;
                        drop(locked);

                        if let Some(imu) = &self.imu {
                            if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                                if let Ok(est) = futures::executor::block_on(
                                    imu.get_estimated_pointing(&SystemTime::now()),
                                ) {
                                    let dt = chrono::Utc::now();
                                    let (ra, dec, eq_roll) = alt_az_to_ra_dec(
                                        est.pitch, est.yaw, est.roll, lat, lon, dt,
                                    );

                                    est_ra = Some(ra);
                                    est_dec = Some(dec);
                                    est_roll = Some(eq_roll);

                                    let mut locked = self.shared_position.write().unwrap();
                                    locked.telescope_ra = ra;
                                    locked.telescope_dec = dec;
                                }
                            }
                        }

                        self.update_solve_stats(
                            false,
                            est_ra,
                            est_dec,
                            est_roll,
                            count,
                            self.target_ra,
                            self.target_dec,
                            current_exposure,
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

                // Update our continuous failure tracker
                if solved {
                    consecutive_failures = 0;
                } else {
                    consecutive_failures += 1;
                }

                // Continuous exposure/gain adjustment
                let actual_exp = exp_time.as_micros() as i32;
                // Tighter 1% tolerance prevents overlapping with our 5% adjustment steps (stops ratcheting)
                let tolerance = (current_exposure as f32 * 0.01).max(500.0) as i32;

                // For UVC, we assume gain changes take effect quickly, so we don't strictly gate on exposure match.
                // For regular cameras, we only evaluate if actual_exp matches current_exposure.
                // ONLY evaluate adjustment if this image matches our currently requested exposure (or is UVC).
                let matches_current = self.config.uvc || (actual_exp - current_exposure).abs() <= tolerance;

                if matches_current {
                    if recent_centroids.len() >= 5 {
                        recent_centroids.remove(0);
                    }
                    recent_centroids.push(count);

                    // Strict requirement: Buffer must be full to trigger a downward adjustment.
                    let buffer_full = recent_centroids.len() >= 5;
                    // Use floating point math to prevent integer truncation
                    let avg_count = (recent_centroids.iter().sum::<usize>() as f64
                        / recent_centroids.len() as f64)
                        .round() as usize;

                    // Only trigger a downward adjustment if the buffer is full and we exceed the ceiling (125% of target)
                    let needs_downward = buffer_full && avg_count > target_max;
                    // Only trigger an upward adjustment if we repeatedly fail to solve (regardless of buffer fullness)
                    let needs_upward = consecutive_failures >= MAX_FAILURES;

                    if needs_downward || needs_upward {
                        let safe_count = avg_count.max(1);
                        let raw_factor = self.config.target_centroids as f64 / safe_count as f64;

                        let factor = if needs_downward {
                            // Needs less exposure. Drop quickly (max 50% per frame)
                            1.0 + (raw_factor - 1.0).max(-0.5) * 0.8
                        } else if needs_upward && avg_count < target_max {
                            // Failed to solve and we are below 150% limit. Increase exposure by a small amount (5%)
                            1.05
                        } else {
                            // Star count is already very high. Don't blindly increase exposure further.
                            1.0
                        };

                        // Only command the camera if the clamped exposure/gain actually changes
                        if self.config.uvc {
                            let next_gain = (current_gain * factor as f32).clamp(self.camera.get_min_gain(), self.camera.get_max_gain());
                            if (next_gain - current_gain).abs() > 0.1 {
                                current_gain = next_gain;
                                self.camera.set_gain(current_gain);
                                // Flush the history so we evaluate the new exposure fresh, stopping the ratchet
                                recent_centroids.clear();
                                consecutive_failures = 0;
                            } else if needs_upward {
                                // Prevent spamming failure adjustments if we hit the hardware max limit
                                consecutive_failures = 0;
                            }
                        } else {
                            let next_exposure = self.clamp_exposure((current_exposure as f64 * factor) as i32);
                            if next_exposure != current_exposure {
                                current_exposure = next_exposure;
                                self.camera.set_exposure(current_exposure);
                                // Flush the history so we evaluate the new exposure fresh, stopping the ratchet
                                recent_centroids.clear();
                                consecutive_failures = 0;
                            } else if needs_upward {
                                // Prevent spamming failure adjustments if we hit the hardware max limit
                                consecutive_failures = 0;
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

// Meeus algorithms for RA/Dec <-> Alt/Az conversions and Parallactic Angle
fn ra_dec_to_alt_az(
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    lat_deg: f64,
    lon_deg: f64,
    utc_time: chrono::DateTime<chrono::Utc>,
) -> (f64, f64, f64) {
    let y = utc_time.year() as f64;
    let m = utc_time.month() as f64;
    let d = utc_time.day() as f64;
    let h = utc_time.hour() as f64
        + utc_time.minute() as f64 / 60.0
        + utc_time.second() as f64 / 3600.0;

    let (mut y_jd, mut m_jd) = (y, m);
    if m <= 2.0 {
        y_jd -= 1.0;
        m_jd += 12.0;
    }

    let a = (y_jd / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd =
        (365.25 * (y_jd + 4716.0)).floor() + (30.6001 * (m_jd + 1.0)).floor() + d + h / 24.0 + b
            - 1524.5;

    let d_jd = jd - 2451545.0; // days since J2000.0
    let mut gmst = 280.46061837 + 360.98564736629 * d_jd;
    gmst %= 360.0;
    if gmst < 0.0 {
        gmst += 360.0;
    }

    let mut lst = gmst + lon_deg;
    lst %= 360.0;
    if lst < 0.0 {
        lst += 360.0;
    }

    let mut ha = lst - ra_deg;
    ha %= 360.0;
    if ha < 0.0 {
        ha += 360.0;
    }

    let ha_rad = ha.to_radians();
    let dec_rad = dec_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    let sin_alt = dec_rad.sin() * lat_rad.sin() + dec_rad.cos() * lat_rad.cos() * ha_rad.cos();
    let alt_rad = sin_alt.asin();
    let alt_deg = alt_rad.to_degrees();

    let cos_az = (dec_rad.sin() - sin_alt * lat_rad.sin()) / (alt_rad.cos() * lat_rad.cos());
    let mut az_rad = cos_az.clamp(-1.0, 1.0).acos();
    if ha_rad.sin() > 0.0 {
        az_rad = std::f64::consts::TAU - az_rad;
    }
    let az_deg = az_rad.to_degrees();

    let q_rad = (ha_rad.sin()).atan2(lat_rad.tan() * dec_rad.cos() - dec_rad.sin() * ha_rad.cos());
    let q_deg = q_rad.to_degrees();

    let alt_az_roll = roll_deg + q_deg;

    (alt_deg, az_deg, alt_az_roll)
}

fn alt_az_to_ra_dec(
    alt_deg: f64,
    az_deg: f64,
    alt_az_roll: f64,
    lat_deg: f64,
    lon_deg: f64,
    utc_time: chrono::DateTime<chrono::Utc>,
) -> (f64, f64, f64) {
    let y = utc_time.year() as f64;
    let m = utc_time.month() as f64;
    let d = utc_time.day() as f64;
    let h = utc_time.hour() as f64
        + utc_time.minute() as f64 / 60.0
        + utc_time.second() as f64 / 3600.0;

    let (mut y_jd, mut m_jd) = (y, m);
    if m <= 2.0 {
        y_jd -= 1.0;
        m_jd += 12.0;
    }

    let a = (y_jd / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd =
        (365.25 * (y_jd + 4716.0)).floor() + (30.6001 * (m_jd + 1.0)).floor() + d + h / 24.0 + b
            - 1524.5;

    let d_jd = jd - 2451545.0;
    let mut gmst = 280.46061837 + 360.98564736629 * d_jd;
    gmst %= 360.0;
    if gmst < 0.0 {
        gmst += 360.0;
    }

    let mut lst = gmst + lon_deg;
    lst %= 360.0;
    if lst < 0.0 {
        lst += 360.0;
    }

    let alt_rad = alt_deg.to_radians();
    let az_rad = az_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    let sin_dec = alt_rad.sin() * lat_rad.sin() + alt_rad.cos() * lat_rad.cos() * az_rad.cos();
    let dec_rad = sin_dec.asin();
    let dec_deg = dec_rad.to_degrees();

    let cos_ha = (alt_rad.sin() - sin_dec * lat_rad.sin()) / (dec_rad.cos() * lat_rad.cos());
    let mut ha_rad = cos_ha.clamp(-1.0, 1.0).acos();
    if az_rad.sin() > 0.0 {
        ha_rad = std::f64::consts::TAU - ha_rad;
    }
    let ha_deg = ha_rad.to_degrees();

    let mut ra_deg = lst - ha_deg;
    ra_deg %= 360.0;
    if ra_deg < 0.0 {
        ra_deg += 360.0;
    }

    let q_rad = (ha_rad.sin()).atan2(lat_rad.tan() * dec_rad.cos() - dec_rad.sin() * ha_rad.cos());
    let q_deg = q_rad.to_degrees();

    let eq_roll = alt_az_roll - q_deg;

    (ra_deg, dec_deg, eq_roll)
}
