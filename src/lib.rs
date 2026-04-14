// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

pub mod camera;
pub mod control_server;
pub mod engine;
pub mod python;
pub mod telescope_server;
pub mod uvc_camera;

#[derive(Clone, Debug)]
pub struct Image {
    pub data: Vec<u8>,
    pub timestamp: SystemTime,
    pub exposure_time_us: i32,
    pub post_process_time: Duration,
}

#[derive(Clone, Debug)]
pub struct CameraConfig {
    pub model: String,
    pub width: u32,
    pub height: u32,
}

pub trait CameraTrait: Send + Sync {
    fn get_config(&self) -> Option<CameraConfig>;
    fn start_capture(&mut self) -> Result<(), String>;
    fn get_gain(&self) -> f32;
    fn get_min_gain(&self) -> f32;
    fn get_max_gain(&self) -> f32;
    fn set_gain(&self, gain: f32);
    fn get_exposure(&self) -> i32;
    fn set_exposure(&self, exposure_us: i32);
    fn enable_hcg(&self);
    fn disable_hcg(&self);
    fn wait_for_next_capture(&self, last_timestamp: Option<SystemTime>) -> Arc<Image>;
    fn get_last_capture(&self) -> Option<Arc<Image>>;
}
