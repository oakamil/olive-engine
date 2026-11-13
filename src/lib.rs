// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

//! High-performance astronomical plate-solving engine and telescope daemon.
//!
//! `olive-engine` provides real-time sky orientation tracking at up to 60 Hz by integrating:
//! - Hardware camera capture (Raspberry Pi ISP via `libcamera`, or USB UVC).
//! - Sub-pixel star centroid extraction (`tetra3` fast/approximate).
//! - Lost-in-space plate solving (`tetra3`, `olive-solve`).
//! - IMU sensor fusion (BNO085, BMI160).
//! - Telescope server protocols: Meade LX200 (TCP 4030) and ASCOM Alpaca (HTTP 11111 / UDP 32227).
//! - Web management UI and diagnostic MJPEG stream (HTTP 8080).

use std::sync::Arc;
use std::time::{Duration, SystemTime};

pub mod auto_controller;
pub mod autofocus;
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
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod uvc_camera;

use serde::{Deserialize, Serialize};

/// Hardware sensor configuration phase used during exposure calibration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwarePhase {
    /// Analog sensor gain multiplier.
    pub gain: f32,
    /// Hot pixel filter mode (`0` = off, `1` = on).
    pub hot_pixel_mode: u8,
    /// Hardware noise reduction algorithm mode.
    pub noise_reduction_mode: u8,
}

/// Builds discrete exposure tuning phases for the Raspberry Pi ISP pipeline.
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

/// Captured video frame and associated acquisition metadata.
#[derive(Clone, Debug)]
pub struct Image {
    /// Raw 8-bit grayscale pixel buffer (row-major).
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Acquisition timestamp measured by the host clock.
    pub timestamp: SystemTime,
    /// Exposure duration in microseconds.
    pub exposure_time_us: i32,
    /// Total sensor gain (analog multiplier, typically >= 1.0).
    pub gain: f32,
    /// Duration spent in debayering, ISP processing, or format conversion.
    pub post_process_time: Duration,
    /// Time delta from previous frame start in microseconds.
    pub frame_interval_us: i64,
    /// `true` if exposure and gain reflect verified hardware registers.
    pub metadata_valid: bool,
    /// Optical/digital zoom factor applied to this frame (defaults to 1.0).
    pub zoom: f32,
}

/// Classification of camera hardware backend.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CameraType {
    /// Synthetic demo camera returning canned star images.
    #[default]
    Demo,
    /// Internal device camera (e.g. Raspberry Pi CSI via libcamera).
    Internal {
        /// Camera device identifier.
        id: String,
    },
    /// External USB Video Class (UVC) camera.
    External,
}

/// Static descriptor and geometry of a camera.
#[derive(Clone, Debug, Default)]
pub struct CameraConfig {
    /// Hardware backend classification.
    pub camera_type: CameraType,
    /// Sensor or driver identifier string.
    pub model: String,
    /// Sensor capture width in pixels.
    pub width: u32,
    /// Sensor capture height in pixels.
    pub height: u32,
    /// Target frame rate in frames per second.
    pub fps: u32,
    /// Optional hardware or sensor crop dimensions `(width, height)`.
    pub crop: Option<(u32, u32)>,
    /// Initial field-of-view estimate in degrees.
    pub initial_fov: Option<f32>,
}

/// Unified interface for camera hardware drivers and synthetic frame sources.
pub trait CameraTrait: Send + Sync {
    /// Returns the discrete hardware tuning phases (gain, hot pixel, noise reduction).
    fn get_calibration_phases(&self) -> Vec<HardwarePhase> {
        vec![HardwarePhase {
            gain: self.get_gain_range().2,
            hot_pixel_mode: 0,
            noise_reduction_mode: 0,
        }]
    }

    /// Returns hardware exposure limits in microseconds `(min_us, max_us)`.
    fn get_exposure_range(&self) -> Option<(i32, i32)> {
        None
    }

    /// Returns the static camera descriptor and stream geometry.
    fn get_config(&self) -> Option<CameraConfig>;

    /// Initializes hardware pipelines and starts video streaming.
    ///
    /// # Errors
    /// Returns an error string if the device node cannot be opened or configured.
    fn start_capture(&mut self) -> Result<(), String>;

    /// Halts video streaming and releases hardware resources.
    fn stop_capture(&self) {}

    /// Returns current analog gain multiplier.
    fn get_gain(&self) -> f32;

    /// Returns a tuple of `(min_gain, max_gain, true_max_analog_gain)`.
    ///
    /// - `min_gain`: Minimum achievable gain.
    /// - `max_gain`: Maximum achievable gain including digital amplification.
    /// - `true_max_analog_gain`: Maximum analog gain prior to digital multiplication.
    fn get_gain_range(&self) -> (f32, f32, f32);

    /// Sets the target analog gain multiplier.
    fn set_gain(&self, gain: f32);

    /// Returns current exposure time in microseconds.
    fn get_exposure(&self) -> i32;

    /// Sets target exposure time in microseconds.
    fn set_exposure(&self, exposure_us: i32);

    /// Configures rolling frame accumulator stack depth (`1` disables stacking).
    fn set_stack_count(&self, _count: usize) {}

    /// Returns current focus position or distance if supported.
    fn get_focus(&self) -> Option<f32> {
        None
    }

    /// Sets lens focus position or distance.
    fn set_focus(&self, _distance: f32) {}

    /// Sets noise reduction algorithm mode.
    fn set_noise_reduction(&self, _mode: u8) {}

    /// Returns active noise reduction algorithm mode.
    fn get_noise_reduction(&self) -> u8 {
        0
    }

    /// Sets hot pixel suppression mode.
    fn set_hot_pixel_mode(&self, _mode: u8) {}

    /// Returns active hot pixel suppression mode.
    fn get_hot_pixel_mode(&self) -> u8 {
        0
    }

    /// Returns `true` if camera driver supports hot pixel suppression.
    fn supports_hot_pixel_mode(&self) -> bool {
        false
    }

    /// Returns `true` if camera driver supports noise reduction modes.
    fn supports_noise_reduction(&self) -> bool {
        false
    }

    /// Returns hardware brightness limits `(min, max, default)`.
    fn get_brightness_range(&self) -> Option<(i32, i32, i32)> {
        None
    }

    /// Returns active brightness level.
    fn get_brightness(&self) -> Option<i32> {
        None
    }

    /// Sets brightness level.
    fn set_brightness(&self, _brightness: i32) {}

    /// Returns current optical or digital zoom factor.
    fn get_zoom(&self) -> Option<f32> {
        None
    }

    /// Sets zoom factor.
    fn set_zoom(&self, _zoom: f32) {}

    /// Blocks until a new frame arrives after `last_timestamp`, or until `timeout` expires.
    fn wait_for_next_capture(
        &self,
        last_timestamp: Option<SystemTime>,
        timeout: Option<std::time::Duration>,
    ) -> Option<Arc<Image>>;

    /// Returns the most recently captured frame without blocking.
    fn get_last_capture(&self) -> Option<Arc<Image>>;

    /// Scales exposure and analog gain by `factor`. Returns `false` if limits are hit.
    fn adjust_sensitivity(&self, factor: f64) -> bool;

    /// Stores the current exposure and gain registers for subsequent rollback.
    fn save_state(&self);

    /// Restores exposure and gain registers saved by [`Self::save_state`].
    fn restore_state(&self);
}
