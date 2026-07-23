// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

#[cfg(not(target_os = "android"))]
pub mod camera;
pub mod control_server;
pub mod engine;
#[cfg(not(target_os = "android"))]
pub mod python;

pub mod alpaca_server;
pub mod dummy_camera;

pub mod telescope_server;
pub mod uvc_camera;

#[derive(Clone, Debug)]
pub struct Image {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp: SystemTime,
    pub exposure_time_us: i32,
    pub post_process_time: Duration,
}

#[derive(Clone, Debug)]
pub struct CameraConfig {
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub crop: Option<(u32, u32)>,
    pub initial_fov: Option<f32>,
}

pub trait CameraTrait: Send + Sync {
    fn get_config(&self) -> Option<CameraConfig>;
    fn start_capture(&mut self) -> Result<(), String>;
    fn stop_capture(&self) {}
    fn get_gain(&self) -> f32;
    /// Returns a tuple of (min_gain, max_gain, true_max_analog_gain)
    /// - min_gain: Minimum possible gain.
    /// - max_gain: Maximum possible gain (including digital, useful for UI sliders).
    /// - true_max_analog_gain: Maximum analog gain before switching to digital. Used as the default max sensitivity.
    fn get_gain_range(&self) -> (f32, f32, f32);
    fn set_gain(&self, gain: f32);
    fn get_exposure(&self) -> i32;
    fn set_exposure(&self, exposure_us: i32);
    fn set_stack_count(&self, _count: usize) {}
    fn enable_hcg(&self);
    fn disable_hcg(&self);
    fn get_focus(&self) -> Option<f32> {
        None
    }
    fn set_focus(&self, _distance: f32) {}
    fn set_noise_reduction(&self, _mode: u8) {}
    fn get_brightness_range(&self) -> Option<(i32, i32, i32)> {
        None
    }
    fn get_brightness(&self) -> Option<i32> {
        None
    }
    fn set_brightness(&self, _brightness: i32) {}
    fn get_zoom(&self) -> Option<f32> {
        None
    }
    fn set_zoom(&self, _zoom: f32) {}
    fn wait_for_next_capture(
        &self,
        last_timestamp: Option<SystemTime>,
        timeout: Option<std::time::Duration>,
    ) -> Option<Arc<Image>>;
    fn get_last_capture(&self) -> Option<Arc<Image>>;

    /// Adjusts the camera's sensitivity (light gathering) by the given factor.
    /// Returns true if the adjustment was applied and changed the state,
    /// false if hardware/configured limits were hit.
    fn adjust_sensitivity(&self, factor: f64) -> bool;
    /// Saves the current hardware sensitivity state internally.
    fn save_state(&self);
    /// Restores the hardware to the last saved sensitivity state.
    fn restore_state(&self);
}
