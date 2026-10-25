// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

pub mod auto_controller;
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod camera;
pub mod control_server;
pub mod engine;
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod python;
pub mod stacker;

pub mod alpaca_server;
pub mod dummy_camera;

pub mod telescope_server;
#[cfg(not(target_os = "ios"))]
pub mod uvc_camera;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwarePhase {
    pub gain: f32,
    pub hot_pixel_mode: u8,
    pub noise_reduction_mode: u8,
}

pub fn build_isp_phases(
    analog: f32,
    max_gain: f32,
    hp_supported: bool,
    nr_modes: &[u8],
) -> Vec<HardwarePhase> {
    let mut phases = Vec::new();
    let nr_off = *nr_modes
        .iter()
        .find(|&&m| m == 0 || m == 3 || m == 1)
        .unwrap_or(&0);

    let mut add_tier = |hp: u8, nr: u8| {
        phases.push(HardwarePhase {
            gain: analog,
            hot_pixel_mode: hp,
            noise_reduction_mode: nr,
        });
        let digital = (analog * 1.5).min(max_gain);
        if digital > analog {
            phases.push(HardwarePhase {
                gain: digital,
                hot_pixel_mode: hp,
                noise_reduction_mode: nr,
            });
        }
    };

    add_tier(0, nr_off);
    if hp_supported {
        add_tier(1, nr_off);
        if nr_modes.contains(&1) {
            add_tier(1, 1);
        }
    }

    // Deduplicate consecutive identical phases (e.g. if max_gain == analog)
    phases.dedup();
    phases
}

#[derive(Clone, Debug)]
pub struct Image {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp: SystemTime,
    pub exposure_time_us: i32,
    pub gain: f32,
    pub post_process_time: Duration,
    pub frame_interval_us: i64,
    pub metadata_valid: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CameraConfig {
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub crop: Option<(u32, u32)>,
    pub initial_fov: Option<f32>,
}

pub trait CameraTrait: Send + Sync {
    fn get_calibration_phases(&self) -> Vec<HardwarePhase> {
        vec![HardwarePhase {
            gain: self.get_gain_range().2,
            hot_pixel_mode: 0,
            noise_reduction_mode: 0,
        }]
    }
    fn get_exposure_range(&self) -> Option<(i32, i32)> {
        None
    }
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
    fn get_focus(&self) -> Option<f32> {
        None
    }
    fn set_focus(&self, _distance: f32) {}
    fn set_noise_reduction(&self, _mode: u8) {}
    fn get_noise_reduction(&self) -> u8 {
        0
    }
    fn set_hot_pixel_mode(&self, _mode: u8) {}
    fn get_hot_pixel_mode(&self) -> u8 {
        0
    }
    fn supports_hot_pixel_mode(&self) -> bool {
        false
    }
    fn supports_noise_reduction(&self) -> bool {
        false
    }
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
