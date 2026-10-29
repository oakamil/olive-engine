// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

use crate::{CameraConfig, CameraTrait, CameraType, Image};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

const FRAME_DELAY_MS: u64 = 33;
const FRAMES_PER_IMAGE: usize = 120;

/// Synthetic camera driver playing back canned astronomical sky images.
pub struct DummyCamera {
    images: Vec<Vec<u8>>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    current_idx: Arc<Mutex<usize>>,
    frame_count: Arc<Mutex<usize>>,
    is_running: Arc<Mutex<bool>>,
    last_frame_time: Arc<Mutex<Instant>>,
}

impl DummyCamera {
    /// Creates a new synthetic dummy camera instance loaded with test fixtures.
    pub fn new() -> Self {
        let bytes_list: [&[u8]; 8] = [
            include_bytes!(
                "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt40_Azi-135_Try1.jpg"
            ),
            include_bytes!(
                "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt40_Azi-45_Try1.jpg"
            ),
            include_bytes!(
                "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt40_Azi135_Try1.jpg"
            ),
            include_bytes!(
                "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt40_Azi45_Try1.jpg"
            ),
            include_bytes!(
                "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt60_Azi-135_Try1.jpg"
            ),
            include_bytes!(
                "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt60_Azi-45_Try1.jpg"
            ),
            include_bytes!(
                "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt60_Azi135_Try1.jpg"
            ),
            include_bytes!(
                "../../olive-solve/tetra3/tests/fixtures/sample_images/2019-07-29T204726_Alt60_Azi45_Try1.jpg"
            ),
        ];

        let mut images = Vec::new();
        let mut widths = Vec::new();
        let mut heights = Vec::new();

        for &bytes in bytes_list.iter() {
            let img = image::load_from_memory(bytes)
                .expect("Failed to decode dummy image")
                .into_luma8();
            widths.push(img.width());
            heights.push(img.height());
            images.push(img.into_raw());
        }

        Self {
            images,
            widths,
            heights,
            current_idx: Arc::new(Mutex::new(0)),
            frame_count: Arc::new(Mutex::new(0)),
            is_running: Arc::new(Mutex::new(false)),
            last_frame_time: Arc::new(Mutex::new(Instant::now())),
        }
    }
}

impl CameraTrait for DummyCamera {
    fn get_config(&self) -> Option<CameraConfig> {
        Some(CameraConfig {
            camera_type: CameraType::Demo,
            model: "Demo Camera".to_string(),
            width: self.widths[0],
            height: self.heights[0],
            fps: (1000 / FRAME_DELAY_MS) as u32,
            crop: None,
            initial_fov: None,
        })
    }

    fn start_capture(&mut self) -> Result<(), String> {
        *self.is_running.lock().unwrap() = true;
        Ok(())
    }

    fn stop_capture(&self) {
        *self.is_running.lock().unwrap() = false;
    }

    fn get_gain(&self) -> f32 {
        1.0
    }
    fn get_gain_range(&self) -> (f32, f32, f32) {
        (1.0, 1.0, 1.0)
    }
    fn set_gain(&self, _gain: f32) {}
    fn get_exposure(&self) -> i32 {
        100_000
    }
    fn set_exposure(&self, _exposure_us: i32) {}

    fn wait_for_next_capture(
        &self,
        _last_timestamp: Option<SystemTime>,
        _timeout: Option<std::time::Duration>,
    ) -> Option<Arc<Image>> {
        if !*self.is_running.lock().unwrap() {
            return None;
        }

        let mut last_time = self.last_frame_time.lock().unwrap();
        let now = Instant::now();
        let delay = Duration::from_millis(FRAME_DELAY_MS);
        if now.duration_since(*last_time) < delay {
            thread::sleep(delay - now.duration_since(*last_time));
        }
        *last_time = Instant::now();

        let mut fc = self.frame_count.lock().unwrap();
        *fc += 1;
        if *fc >= FRAMES_PER_IMAGE {
            *fc = 0;
            let mut idx = self.current_idx.lock().unwrap();
            *idx = (*idx + 1) % self.images.len();
        }

        let idx = *self.current_idx.lock().unwrap();

        Some(Arc::new(Image {
            data: self.images[idx].clone(),
            width: self.widths[idx],
            height: self.heights[idx],
            timestamp: SystemTime::now(),
            exposure_time_us: 100_000,
            gain: 1.0,
            post_process_time: Duration::from_millis(0),
            frame_interval_us: 0,
            metadata_valid: true,
            zoom: 1.0,
        }))
    }

    fn get_last_capture(&self) -> Option<Arc<Image>> {
        let idx = *self.current_idx.lock().unwrap();
        Some(Arc::new(Image {
            data: self.images[idx].clone(),
            width: self.widths[idx],
            height: self.heights[idx],
            timestamp: SystemTime::now(),
            exposure_time_us: 100_000,
            gain: 1.0,
            post_process_time: Duration::from_millis(0),
            frame_interval_us: 0,
            metadata_valid: true,
            zoom: 1.0,
        }))
    }

    fn adjust_sensitivity(&self, _factor: f64) -> bool {
        false
    }
    fn save_state(&self) {}
    fn restore_state(&self) {}
}
