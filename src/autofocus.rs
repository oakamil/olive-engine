// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

use crate::engine::{EngineStateData, calculate_hfr};
use crate::{CameraTrait, Image};
use log::info;
use ndarray::Array2;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tetra3::fast_extractor::{FastBgSubMode, FastDownsample, FastExtractOptions, FastExtractor};

pub const DEFAULT_INFINITY_FOCUS: f32 = 0.92;

/// Runs an autofocus sweep across internal camera focus positions for iOS.
///
/// Only invoked when auto mode is enabled on internal iOS cameras.
/// Sweeps from 0.92 down to 0.76 at 250ms exposure. Evaluates HFR strictly on the
/// central 50% x 50% box (Center 2.0x crop) to eliminate perimeter obstructions (trees/eaves).
/// Exits early as soon as HFR increases (hfr > min_hfr).
/// Defaults to 0.92 if no stars are detected.
pub fn run_autofocus(
    camera: &Arc<dyn CameraTrait>,
    engine_state: &Arc<RwLock<EngineStateData>>,
    is_running: &Arc<AtomicBool>,
) -> f32 {
    let focus_points = [0.92f32, 0.88f32, 0.84f32, 0.80f32, 0.76f32];
    let mut best_focus = DEFAULT_INFINITY_FOCUS;
    let mut min_hfr = f64::MAX;

    info!("Starting iOS internal camera autofocus sweep...");

    if let Ok(mut state) = engine_state.write() {
        state.status = "Calibrating Focus".to_string();
    }

    // Set 250ms exposure for the sweep
    camera.set_exposure(250_000);

    let mut last_timestamp = None;

    // Camera pipeline warm up: wait up to 2 seconds for initial frame streaming
    if let Some(first_img) = camera.wait_for_next_capture(None, Some(Duration::from_millis(2000))) {
        last_timestamp = Some(first_img.timestamp);
    }

    for &focus in &focus_points {
        if !is_running.load(Ordering::Relaxed) {
            info!("Autofocus cancelled by is_running flag");
            camera.set_focus(best_focus);
            return best_focus;
        }

        camera.set_focus(focus);

        // Discard 1 transition frame exposed while the voice coil motor was in transit
        if let Some(t_img) =
            camera.wait_for_next_capture(last_timestamp, Some(Duration::from_millis(1000)))
        {
            last_timestamp = Some(t_img.timestamp);
        }

        if !is_running.load(Ordering::Relaxed) {
            info!("Autofocus cancelled by is_running flag");
            camera.set_focus(best_focus);
            return best_focus;
        }

        // Capture 1 fully settled frame
        let img =
            match camera.wait_for_next_capture(last_timestamp, Some(Duration::from_millis(1000))) {
                Some(i) => {
                    last_timestamp = Some(i.timestamp);
                    i
                }
                None => {
                    info!("Autofocus: No frame received for focus {:.2}", focus);
                    continue;
                }
            };

        // Extract HFR from Center 2.0x crop (central 50% x 50% box)
        let (hfr, star_count) = evaluate_center_crop_hfr(&img);
        info!(
            "Autofocus probe: focus={:.2}, HFR={:.2}, stars={}",
            focus, hfr, star_count
        );

        if star_count >= 2 && hfr > 0.0 {
            if let Ok(mut state) = engine_state.write() {
                state.hfr = hfr;
                state.status = "Calibrating Focus".to_string();
            }

            if hfr < min_hfr {
                min_hfr = hfr;
                best_focus = focus;
            } else if hfr > min_hfr {
                info!(
                    "Autofocus early exit: HFR rose from {:.2} to {:.2}. Best focus is {:.2}",
                    min_hfr, hfr, best_focus
                );
                break;
            }
        }
    }

    info!(
        "Autofocus sweep complete. Locking focus at {:.2} (min HFR: {:.2})",
        best_focus, min_hfr
    );
    camera.set_focus(best_focus);
    best_focus
}

/// Evaluates median Half-Flux Radius (HFR) strictly within the central 50% x 50% box
/// of the image (Center 2.0x virtual crop).
pub fn evaluate_center_crop_hfr(img: &Image) -> (f64, usize) {
    let w = img.width as usize;
    let h = img.height as usize;
    if w < 10 || h < 10 || img.data.len() < w * h {
        return (0.0, 0);
    }

    let crop_w = w / 2;
    let crop_h = h / 2;
    let start_x = w / 4;
    let start_y = h / 4;

    let mut crop_data = Vec::with_capacity(crop_w * crop_h);
    for y in 0..crop_h {
        let offset = (start_y + y) * w + start_x;
        crop_data.extend_from_slice(&img.data[offset..offset + crop_w]);
    }

    let arr = match Array2::from_shape_vec((crop_h, crop_w), crop_data.clone()) {
        Ok(a) => a,
        Err(_) => return (0.0, 0),
    };

    let options = FastExtractOptions {
        sigma: 3.5,
        downsample: FastDownsample::None,
        bg_sub_mode: Some(FastBgSubMode::GlobalMedian),
        binary_open: false,
        min_area: Some(2), // Ignore single hot pixels
        max_area: Some(60),
        ..FastExtractOptions::default()
    };

    let mut extractor = FastExtractor::new(crop_w, crop_h, options);
    let result = extractor.extract(&arr);
    let count = result.centroids.len();
    if count < 2 {
        return (0.0, count);
    }

    let mut centroids_arr = Array2::<f64>::zeros((count, 2));
    for (i, c) in result.centroids.iter().enumerate() {
        centroids_arr[[i, 0]] = c.y;
        centroids_arr[[i, 1]] = c.x;
    }

    let hfr = calculate_hfr(&crop_data, crop_w, crop_h, &centroids_arr, 20);
    (hfr, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CameraConfig;
    use std::sync::Mutex;
    use std::time::SystemTime;

    fn make_star_image(
        width: usize,
        height: usize,
        stars: &[(usize, usize)],
        radius: f64,
        timestamp: SystemTime,
    ) -> Image {
        let mut data = vec![10u8; width * height];
        for &(cx, cy) in stars {
            for dy in -8..=8 {
                for dx in -8..=8 {
                    let x = cx as isize + dx;
                    let y = cy as isize + dy;
                    if x >= 0 && x < width as isize && y >= 0 && y < height as isize {
                        let dist_sq = (dx * dx + dy * dy) as f64;
                        let profile = (-0.5 * dist_sq / (radius * radius)).exp();
                        let intensity = (profile * 230.0) as u8;
                        let idx = (y as usize) * width + (x as usize);
                        data[idx] = data[idx].saturating_add(intensity);
                    }
                }
            }
        }
        Image {
            data,
            width: width as u32,
            height: height as u32,
            timestamp,
            exposure_time_us: 250_000,
            gain: 100.0,
            post_process_time: Duration::ZERO,
            frame_interval_us: 33_333,
            metadata_valid: true,
            zoom: 1.0,
        }
    }

    struct MockAutofocusCamera {
        current_focus: Mutex<f32>,
        focus_probes: Mutex<Vec<f32>>,
        star_radius_fn: Box<dyn Fn(f32) -> Option<f64> + Send + Sync>,
    }

    impl MockAutofocusCamera {
        fn new<F>(radius_fn: F) -> Self
        where
            F: Fn(f32) -> Option<f64> + Send + Sync + 'static,
        {
            Self {
                current_focus: Mutex::new(DEFAULT_INFINITY_FOCUS),
                focus_probes: Mutex::new(Vec::new()),
                star_radius_fn: Box::new(radius_fn),
            }
        }
    }

    impl CameraTrait for MockAutofocusCamera {
        fn get_config(&self) -> Option<CameraConfig> {
            None
        }
        fn start_capture(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn get_gain(&self) -> f32 {
            1.0
        }
        fn get_gain_range(&self) -> (f32, f32, f32) {
            (1.0, 1.0, 1.0)
        }
        fn set_gain(&self, _gain: f32) {}
        fn get_exposure(&self) -> i32 {
            250_000
        }
        fn set_exposure(&self, _exposure_us: i32) {}
        fn get_focus(&self) -> Option<f32> {
            Some(*self.current_focus.lock().unwrap())
        }
        fn set_focus(&self, distance: f32) {
            *self.current_focus.lock().unwrap() = distance;
            self.focus_probes.lock().unwrap().push(distance);
        }
        fn wait_for_next_capture(
            &self,
            _last_timestamp: Option<SystemTime>,
            _timeout: Option<Duration>,
        ) -> Option<Arc<Image>> {
            let cur_f = *self.current_focus.lock().unwrap();
            let radius_opt = (self.star_radius_fn)(cur_f);
            let img = match radius_opt {
                Some(r) => make_star_image(
                    200,
                    200,
                    &[(80, 80), (100, 100), (120, 80)],
                    r,
                    SystemTime::now(),
                ),
                None => make_star_image(200, 200, &[], 1.0, SystemTime::now()),
            };
            Some(Arc::new(img))
        }
        fn get_last_capture(&self) -> Option<Arc<Image>> {
            None
        }
        fn adjust_sensitivity(&self, _factor: f64) -> bool {
            true
        }
        fn save_state(&self) {}
        fn restore_state(&self) {}
    }

    #[test]
    fn test_center_2x_crop_dimensions() {
        let width = 100usize;
        let height = 80usize;
        let mut data = vec![0u8; width * height];
        // Mark center pixel
        data[40 * width + 50] = 255;

        let img = Image {
            data,
            width: width as u32,
            height: height as u32,
            timestamp: SystemTime::now(),
            exposure_time_us: 250_000,
            gain: 100.0,
            post_process_time: Duration::ZERO,
            frame_interval_us: 33_333,
            metadata_valid: true,
            zoom: 1.0,
        };

        let (hfr, count) = evaluate_center_crop_hfr(&img);
        // Only 1 hot pixel in center -> rejected by min_area or count < 2
        assert_eq!(hfr, 0.0);
        assert!(count <= 1);
    }

    #[test]
    fn test_center_crop_ignores_perimeter_stars() {
        let width = 200usize;
        let height = 200usize;
        // Center crop is [50..150, 50..150]. Place stars only in perimeter ([0..50] and [150..200]).
        let perimeter_stars = vec![(20, 20), (30, 180), (180, 20)];
        let perimeter_img =
            make_star_image(width, height, &perimeter_stars, 1.5, SystemTime::now());

        let (hfr, count) = evaluate_center_crop_hfr(&perimeter_img);
        assert_eq!(
            count, 0,
            "Perimeter stars outside center 2x crop must be ignored"
        );
        assert_eq!(hfr, 0.0);

        // Place stars strictly inside the center 2x crop ([50..150, 50..150])
        let center_stars = vec![(80, 80), (100, 100), (120, 80)];
        let center_img = make_star_image(width, height, &center_stars, 1.5, SystemTime::now());

        let (hfr, count) = evaluate_center_crop_hfr(&center_img);
        assert_eq!(count, 3, "Center 2x crop stars must be detected");
        assert!(
            hfr > 0.0,
            "Center 2x crop stars must have valid positive HFR"
        );
    }

    #[test]
    fn test_run_autofocus_early_exit() {
        // Probe sequence: 0.92 -> 0.88 -> 0.84 -> 0.80 -> 0.76
        // Set radii: 0.92 => 2.0 (HFR ~2.55), 0.88 => 1.2 (sharpest / minimum HFR ~1.58), 0.84 => 2.0 (HFR ~2.55 -> break!)
        let mock_camera = Arc::new(MockAutofocusCamera::new(|f| {
            if (f - 0.92).abs() < 0.01 {
                Some(2.0)
            } else if (f - 0.88).abs() < 0.01 {
                Some(1.2)
            } else if (f - 0.84).abs() < 0.01 {
                Some(2.0)
            } else {
                Some(2.5)
            }
        }));
        let camera: Arc<dyn CameraTrait> = mock_camera.clone();

        let engine_state = Arc::new(RwLock::new(EngineStateData::default()));
        let is_running = Arc::new(AtomicBool::new(true));

        let best_focus = run_autofocus(&camera, &engine_state, &is_running);
        assert_eq!(
            best_focus, 0.88,
            "Best focus should be 0.88 where HFR was minimum"
        );

        // Verify status and HFR in engine_state
        let state = engine_state.read().unwrap();
        assert_eq!(state.status, "Calibrating Focus");
        assert!(state.hfr > 0.0);

        // Verify early exit: 0.80 and 0.76 were never probed
        let probes = mock_camera.focus_probes.lock().unwrap().clone();
        assert!(probes.contains(&0.92));
        assert!(probes.contains(&0.88));
        assert!(probes.contains(&0.84));
        assert!(
            !probes.contains(&0.80),
            "0.80 must not be probed due to early exit"
        );
        assert!(
            !probes.contains(&0.76),
            "0.76 must not be probed due to early exit"
        );

        // Ensure final camera focus is set to 0.88
        assert_eq!(camera.get_focus(), Some(0.88));
    }

    #[test]
    fn test_run_autofocus_fallback_insufficient_stars() {
        // No stars detected across all frames
        let camera: Arc<dyn CameraTrait> = Arc::new(MockAutofocusCamera::new(|_| None));
        let engine_state = Arc::new(RwLock::new(EngineStateData::default()));
        let is_running = Arc::new(AtomicBool::new(true));

        let best_focus = run_autofocus(&camera, &engine_state, &is_running);
        assert_eq!(
            best_focus, DEFAULT_INFINITY_FOCUS,
            "Should fallback to 0.92 when insufficient stars detected"
        );
        assert_eq!(camera.get_focus(), Some(DEFAULT_INFINITY_FOCUS));
    }

    #[test]
    fn test_run_autofocus_cancelled_when_not_running() {
        let camera: Arc<dyn CameraTrait> = Arc::new(MockAutofocusCamera::new(|_| Some(1.5)));
        let engine_state = Arc::new(RwLock::new(EngineStateData::default()));
        let is_running = Arc::new(AtomicBool::new(false));

        let best_focus = run_autofocus(&camera, &engine_state, &is_running);
        assert_eq!(best_focus, DEFAULT_INFINITY_FOCUS);
    }
}
