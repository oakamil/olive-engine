// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{error, info, warn};
use ndarray::Array2;

use crate::camera::Camera;
use tetra3::{
    extractor::{BgSubMode, ExtractOptions, Extractor},
    solver::{SolveOptions, SolveStatus},
    tetra3::Tetra3,
};

pub const TARGET_CENTROIDS: usize = 16;
pub const MIN_CENTROIDS: usize = 4;
pub const CALIBRATION_INTERVAL: Duration = Duration::from_secs(10);
pub const MAX_RETRIES: usize = 15; // Increased to allow downward floor probing
pub const CALIBRATION_EXPOSURE_US: i32 = 50000;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub min_exposure_us: i32,
    pub max_exposure_us: i32,
    pub gain: f32,
    pub database_path: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            min_exposure_us: 16670,
            max_exposure_us: 1000000,
            gain: 31.0,
            database_path: format!(
                "{}/data/default_database.npz",
                std::env::var("HOME").unwrap_or_default()
            ),
        }
    }
}

pub struct Engine {
    camera: Arc<Camera>,
    config: EngineConfig,
    solver: Tetra3,
    extractor: Extractor,
    fov: Option<f64>,
    last_processed_timestamp: Option<std::time::SystemTime>,
}

impl Engine {
    pub fn new(camera: Arc<Camera>, config: EngineConfig) -> Self {
        let solver = Tetra3::new(&config.database_path);
        let extractor = Extractor::new();
        Self {
            camera,
            config,
            solver,
            extractor,
            fov: None,
            last_processed_timestamp: None,
        }
    }

    fn clamp_exposure(&self, exposure: i32) -> i32 {
        exposure.clamp(self.config.min_exposure_us, self.config.max_exposure_us)
    }

    fn extract_image_centroids(
        &mut self,
    ) -> Option<(usize, Duration, Duration, Duration, Array2<f64>)> {
        let img = self
            .camera
            .wait_for_next_capture(self.last_processed_timestamp);
        self.last_processed_timestamp = Some(img.timestamp);

        let config = self.camera.get_config().expect("Camera config missing");

        let width = config.width as usize;
        let height = config.height as usize;

        // Build ndarray from data
        let array_img = match Array2::from_shape_vec((height, width), img.data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                error!("Failed to create Array2 from image data: {}", e);
                return None;
            }
        };

        let mut options = ExtractOptions::default();
        options.bg_sub_mode = Some(BgSubMode::GlobalMean);
        //options.downsample = Some(2);

        let extract_start = Instant::now();
        let result = self.extractor.extract_u8_fast(&array_img, options);
        let extract_time = extract_start.elapsed();

        let num_centroids = result.centroids.len();

        let mut centroids_arr = Array2::<f64>::zeros((num_centroids, 2));
        for (i, c) in result.centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = c.y;
            centroids_arr[[i, 1]] = c.x;
        }

        Some((
            num_centroids,
            Duration::from_micros(img.exposure_time_us as u64),
            img.post_process_time,
            extract_time,
            centroids_arr,
        ))
    }

    pub fn run_calibration_loop(&mut self) {
        info!("Starting auto-calibration loop...");

        // Apply static hardware settings ONCE at startup
        self.camera.set_gain(self.config.gain);
        self.camera.enable_hcg();

        let target_min = TARGET_CENTROIDS;
        let target_max = (TARGET_CENTROIDS as f64 * 1.25).round() as usize;

        loop {
            let mut current_exposure = CALIBRATION_EXPOSURE_US;
            current_exposure = self.clamp_exposure(current_exposure);
            self.camera.set_exposure(current_exposure);

            let mut calibrated = false;
            let mut last_good_exposure: Option<i32> = None;

            for retry in 0..MAX_RETRIES {
                info!(
                    "Calibration attempt {}/{} (Exposure: {} us)",
                    retry + 1,
                    MAX_RETRIES,
                    current_exposure
                );

                let mut total_centroids = 0;
                let mut last_centroids_arr = None;
                let mut valid_samples = 0;
                let mut found_new_exposure = false;

                // Fetch frames until we verify the hardware has applied our new exposure, then collect 5
                while valid_samples < 5 {
                    if let Some((count, exp_time, _, _, arr)) = self.extract_image_centroids() {
                        let actual_exp = exp_time.as_micros() as i32;

                        // Hardware quantization tolerance (5% or 100us, whichever is larger)
                        let tolerance = (current_exposure as f32 * 0.05).max(100.0) as i32;

                        if !found_new_exposure {
                            if (actual_exp - current_exposure).abs() > tolerance {
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
                        // We unwrap here because an Err means a catastrophic failure (e.g., missing database), not a failure to match stars.
                        let solution = self.solver.solve_from_centroids(arr, size, SolveOptions::default())
                            .expect("Critical Error: solver failed to execute during calibration. Is the database path valid?");

                        if matches!(solution.status, SolveStatus::MatchFound) {
                            self.fov = solution.fov;
                            info!("Sanity check passed! FOV: {:?}", self.fov);

                            // Now check if we are below the ceiling
                            if avg_centroids <= target_max {
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
                } else {
                    // avg_centroids < target_min
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

                if needs_standard_adjustment {
                    // Finer Adjustment: We only move 33% of the way to the "ideal" mathematical factor.
                    // This dampens the change so we creep up on the target range and settle lower.
                    let raw_factor = TARGET_CENTROIDS as f64 / (avg_centroids.max(1)) as f64;
                    let factor = 1.0 + (raw_factor - 1.0) * 0.33;

                    let next_exposure =
                        self.clamp_exposure((current_exposure as f64 * factor) as i32);

                    if next_exposure == current_exposure {
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

            // If we exhausted all retries but had successfully verified an exposure, use it!
            if !calibrated && last_good_exposure.is_some() {
                let good_exp = last_good_exposure.unwrap();
                info!(
                    "Max retries reached during floor search. Reverting to optimal floor exposure: {} us. Calibration complete.",
                    good_exp
                );
                current_exposure = good_exp;
                self.camera.set_exposure(current_exposure);
                calibrated = true;
            }

            if calibrated {
                break;
            }

            info!(
                "Calibration failed or in progress, waiting {:?}...",
                CALIBRATION_INTERVAL
            );
            std::thread::sleep(CALIBRATION_INTERVAL);
        }
    }

    pub fn run_solving_loop(&mut self) {
        info!("Entering solving loop...");
        let mut current_exposure = self.camera.get_exposure();
        let target_max = (TARGET_CENTROIDS as f64 * 1.5).round() as usize;

        let mut consecutive_failures = 0;
        const MAX_FAILURES: usize = 3;

        // Rolling average buffer to smooth out frame-to-frame scintillation dips
        let mut recent_centroids: Vec<usize> = Vec::with_capacity(5);

        loop {
            if let Some((count, exp_time, pp_time, extract_time, centroids_arr)) =
                self.extract_image_centroids()
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
                let size = (camera_config.height as f64, camera_config.width as f64);

                let mut solve_options = SolveOptions::default();
                if let Some(fov) = self.fov {
                    solve_options.fov_estimate = Some(fov);
                    solve_options.fov_max_error = Some(0.5); // Provide a reasonable constraint based on FOV
                }

                let solution = self.solver.solve_from_centroids(&centroids_arr, size, solve_options)
                    .expect("Critical Error: solver failed to execute during solving loop. Is the database accessible?");

                let solve_time = solve_start.elapsed();

                match solution.status {
                    SolveStatus::MatchFound => {
                        info!(
                            "Solved! RA: {:?}, Dec: {:?}, Roll: {:?} | Centroids: {} | Exposure: {:?} | Post-Process: {:?} | Extract: {:?} | Solve: {:?}",
                            solution.ra,
                            solution.dec,
                            solution.roll,
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
                    }
                    _ => {
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

                // Continuous exposure adjustment
                let actual_exp = exp_time.as_micros() as i32;

                // Tighter 1% tolerance prevents overlapping with our 5% adjustment steps (stops ratcheting)
                let tolerance = (current_exposure as f32 * 0.01).max(500.0) as i32;

                // ONLY evaluate adjustment if this image matches our currently requested exposure.
                if (actual_exp - current_exposure).abs() <= tolerance {
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
                        let raw_factor = TARGET_CENTROIDS as f64 / safe_count as f64;

                        let factor = if needs_downward {
                            // Needs less exposure. Drop quickly (max 50% per frame)
                            1.0 + (raw_factor - 1.0).max(-0.5) * 0.8
                        } else if needs_upward && raw_factor > 1.0 {
                            // Failed to solve and average is below target. Increase VERY slowly (max 5% per frame)
                            1.0 + (raw_factor - 1.0).min(0.5) * 0.1
                        } else {
                            // Failed to solve, but star count is already high. Don't blindly increase exposure.
                            1.0
                        };

                        let next_exposure =
                            self.clamp_exposure((current_exposure as f64 * factor) as i32);

                        // Only command the camera if the clamped exposure actually changes
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
            } else {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}
