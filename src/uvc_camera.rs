// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use crate::{CameraConfig, CameraTrait, Image};
use log::{info, warn};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicI32, AtomicU32, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};
use turbojpeg;
use uvc::{AutoExposureMode, FrameFormat};

// --- Data Structures ---

#[derive(Clone, Debug, Default)]
pub struct UvcRanges {
    pub saturation: Option<uvc::Range<i32>>,
    pub contrast: Option<uvc::Range<i32>>,
    pub white_balance_temperature: Option<uvc::Range<i32>>,
    pub sharpness: Option<uvc::Range<i32>>,
    pub brightness: Option<uvc::Range<i32>>,
    pub gamma: Option<uvc::Range<i32>>,
    pub gain: Option<uvc::Range<i32>>,
}

#[derive(Clone, Debug, Default)]
pub struct UvcControls {
    pub ae_mode: Option<u8>,
    pub exposure_abs: Option<u32>,
    pub saturation: Option<i32>,
    pub contrast: Option<i32>,
    pub white_balance_temperature: Option<i32>,
    pub sharpness: Option<i32>,
    pub brightness: Option<i32>,
    pub gamma: Option<i32>,
    pub gain: Option<i32>,
}

/// Thread-safe atomic store for the latest capture.
struct CaptureStore {
    image: Mutex<Option<Arc<Image>>>,
    cvar: Condvar,
}

/// Commands sent from the main thread to the capture thread.
enum CaptureCommand {
    SetGain(f32),
    SetUvcControls(UvcControls),
    StopCaptureThread,
}

pub struct Camera {
    command_tx: Option<mpsc::Sender<CaptureCommand>>,
    store: Arc<CaptureStore>,
    capture_thread: Option<JoinHandle<()>>,

    // Cached atomic values for lock-free getter access
    current_gain: Arc<AtomicU32>, // f32 stored as u32 bits for atomic operations
    current_exposure: Arc<AtomicI32>,
    saved_gain: Arc<AtomicU32>,

    config: Arc<Mutex<Option<CameraConfig>>>,
    pub ranges: Arc<Mutex<Option<UvcRanges>>>,
    target_fps: u32,
    mjpeg_enabled: bool,
    target_res: Option<(u32, u32)>,
    fd: Option<i32>,
}

impl CameraTrait for Camera {
    fn get_config(&self) -> Option<CameraConfig> {
        self.config.lock().unwrap().as_ref().cloned()
    }

    /// Kicks off the capture operation on a separate thread.
    fn start_capture(&mut self) -> Result<(), String> {
        if self.capture_thread.is_some() {
            return Ok(()); // Already running
        }

        let (tx, rx) = mpsc::channel();
        self.command_tx = Some(tx);

        let store_clone = Arc::clone(&self.store);
        let config_clone = Arc::clone(&self.config);
        let ranges_clone = Arc::clone(&self.ranges);
        let current_gain = Arc::clone(&self.current_gain);
        let current_exposure = Arc::clone(&self.current_exposure);
        let target_fps = self.target_fps;
        let mjpeg_enabled = self.mjpeg_enabled;
        let target_res = self.target_res;
        let fd = self.fd;

        self.capture_thread = Some(thread::spawn(move || {
            Self::capture_loop(
                rx,
                store_clone,
                config_clone,
                ranges_clone,
                current_gain,
                current_exposure,
                target_fps,
                mjpeg_enabled,
                target_res,
                fd,
            );
        }));

        Ok(())
    }

    // --- Configuration Getters & Setters ---

    fn get_gain(&self) -> f32 {
        f32::from_bits(self.current_gain.load(Ordering::Relaxed))
    }

    fn get_min_gain(&self) -> f32 {
        self.ranges
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|r| r.gain.as_ref())
            .map(|g| g.min as f32)
            .unwrap_or(0.0)
    }

    fn get_max_gain(&self) -> f32 {
        self.ranges
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|r| r.gain.as_ref())
            .map(|g| g.max as f32)
            .unwrap_or(100.0)
    }

    fn set_gain(&self, gain: f32) {
        let min = self.get_min_gain();
        let max = self.get_max_gain();
        let clamped_gain = gain.clamp(min, max).round();
        self.current_gain
            .store(clamped_gain.to_bits(), Ordering::Relaxed);
        self.send_command(CaptureCommand::SetGain(clamped_gain));
    }

    fn get_exposure(&self) -> i32 {
        self.current_exposure.load(Ordering::Relaxed)
    }

    fn set_exposure(&self, _exposure_us: i32) {}

    fn enable_hcg(&self) {}

    fn disable_hcg(&self) {}

    fn get_last_capture(&self) -> Option<Arc<Image>> {
        let lock = self.store.image.lock().unwrap();
        lock.clone()
    }

    fn wait_for_next_capture(
        &self,
        last_timestamp: Option<SystemTime>,
        timeout: Option<Duration>,
    ) -> Option<Arc<Image>> {
        let mut lock = self.store.image.lock().unwrap();
        loop {
            if let Some(img) = &*lock {
                if last_timestamp.is_none() || Some(img.timestamp) != last_timestamp {
                    return Some(img.clone());
                }
            }
            if let Some(duration) = timeout {
                let (new_lock, result) = self.store.cvar.wait_timeout(lock, duration).unwrap();
                lock = new_lock;
                if result.timed_out() {
                    // Check one last time before returning None
                    if let Some(img) = &*lock {
                        if last_timestamp.is_none() || Some(img.timestamp) != last_timestamp {
                            return Some(img.clone());
                        }
                    }
                    return None;
                }
            } else {
                lock = self.store.cvar.wait(lock).unwrap();
            }
        }
    }

    fn adjust_sensitivity(&self, _factor: f64) -> bool {
        // Exposure can't be easily adjusted. In the future we can select a slower frame rate
        // if supported, if we need to increase sensitivity.
        return false;
    }

    fn save_state(&self) {
        self.saved_gain
            .store(self.current_gain.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    fn restore_state(&self) {
        let gain = f32::from_bits(self.saved_gain.load(Ordering::Relaxed));
        self.set_gain(gain);
    }
}

impl Camera {
    /// Instantiates the camera module and sets initial cached state.
    pub fn new(target_fps: u32, mjpeg_enabled: bool, target_res: Option<(u32, u32)>) -> Self {
        Self::with_fd(target_fps, mjpeg_enabled, target_res, None)
    }

    pub fn with_fd(
        target_fps: u32,
        mjpeg_enabled: bool,
        target_res: Option<(u32, u32)>,
        fd: Option<i32>,
    ) -> Self {
        let store = Arc::new(CaptureStore {
            image: Mutex::new(None),
            cvar: Condvar::new(),
        });

        let current_exposure = Arc::new(AtomicI32::new(50000));
        let config = Arc::new(Mutex::new(None));
        let ranges = Arc::new(Mutex::new(None));

        Self {
            command_tx: None,
            store,
            capture_thread: None,
            current_gain: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            current_exposure,
            saved_gain: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            config,
            ranges,
            target_fps,
            mjpeg_enabled,
            target_res,
            fd,
        }
    }

    pub fn set_uvc_controls(&self, controls: UvcControls) {
        self.send_command(CaptureCommand::SetUvcControls(controls));
    }

    fn send_command(&self, cmd: CaptureCommand) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(cmd);
        }
    }

    fn capture_loop(
        rx: mpsc::Receiver<CaptureCommand>,
        store: Arc<CaptureStore>,
        config_store: Arc<Mutex<Option<CameraConfig>>>,
        ranges_store: Arc<Mutex<Option<UvcRanges>>>,
        current_gain_atomic: Arc<AtomicU32>,
        current_exposure_atomic: Arc<AtomicI32>,
        target_fps: u32,
        mjpeg_enabled: bool,
        target_res: Option<(u32, u32)>,
        fd: Option<i32>,
    ) {
        let ctx = uvc::Context::new().expect("Failed to create UVC context");

        let mut _dev_keep_alive = None;
        let devh = if let Some(fd) = fd {
            ctx.wrap_fd(fd).expect("Failed to open UVC device with FD")
        } else {
            let dev = ctx
                .find_device(None, None, None)
                .expect("Failed to find UVC device");
            _dev_keep_alive = Some(dev);
            _dev_keep_alive
                .as_ref()
                .unwrap()
                .open()
                .expect("Failed to open UVC device")
        };

        // Find highest resolution format (format depends on the new mjpeg option) that exactly
        // matches the frame rate given or quit with an error if none is found
        let mut best_format = None;
        let mut max_res = 0;
        let mut camera_max_res = 0;

        for format_desc in devh.supported_formats() {
            for frame_desc in format_desc.supported_formats() {
                let width = frame_desc.width() as u32;
                let height = frame_desc.height() as u32;
                let res = width * height;
                if res > camera_max_res {
                    camera_max_res = res;
                }

                let format_type = match frame_desc.subtype() {
                    uvc::DescriptionSubtype::FormatMJPEG | uvc::DescriptionSubtype::FrameMJPEG => {
                        FrameFormat::MJPEG
                    }
                    uvc::DescriptionSubtype::FormatUncompressed
                    | uvc::DescriptionSubtype::FrameUncompressed => FrameFormat::Uncompressed,
                    _ => FrameFormat::Any,
                };

                if format_type == FrameFormat::Any {
                    continue;
                }

                let is_mjpeg = format_type == FrameFormat::MJPEG;
                if is_mjpeg != mjpeg_enabled {
                    // Only match exactly the required mjpeg state
                    continue;
                }

                for &interval in frame_desc.intervals() {
                    let fps = 10_000_000 / interval;
                    if fps == target_fps {
                        let width = frame_desc.width() as u32;
                        let height = frame_desc.height() as u32;
                        let res = width * height;

                        let is_match = match target_res {
                            Some((tw, th)) => width == tw && height == th,
                            None => true, // If no exact res is requested, any matching fps/mjpeg is a candidate
                        };

                        if is_match && res > max_res {
                            max_res = res;
                            best_format = Some(uvc::StreamFormat {
                                width,
                                height,
                                fps,
                                format: format_type,
                            });
                        }
                    }
                }
            }
        }

        let stream_format = best_format.unwrap_or_else(|| {
            panic!(
                "Error: No format found matching exactly the specified fps of {} and mjpeg enabled: {}",
                target_fps, mjpeg_enabled
            );
        });

        info!("Selected UVC format: {:?}", stream_format);

        let mut uvc_ranges = UvcRanges::default();

        macro_rules! get_range {
            ($getter:ident, $range_getter:ident, $field:ident) => {
                if let Ok(range) = devh.$range_getter() {
                    uvc_ranges.$field = Some(uvc::Range {
                        min: range.min as i32,
                        max: range.max as i32,
                        step: range.step as i32,
                        default: range.default as i32,
                    });
                }
            };
        }

        get_range!(saturation, saturation_range, saturation);
        get_range!(contrast, contrast_range, contrast);
        get_range!(
            white_balance_temperature,
            white_balance_temperature_range,
            white_balance_temperature
        );
        get_range!(sharpness, sharpness_range, sharpness);
        get_range!(brightness, brightness_range, brightness);
        get_range!(gamma, gamma_range, gamma);
        get_range!(gain, gain_range, gain);

        let mut retry_count = 0;
        let mut streamh = loop {
            match devh.get_stream_handle_with_format(stream_format) {
                Ok(h) => break h,
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= 3 {
                        panic!("Failed to get stream handle after 3 attempts: {:?}", e);
                    }
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
        };

        // Per empirical results with an SC2210 camera, set all values to their defaults
        if let Some(range) = uvc_ranges.saturation {
            let _ = devh.set_saturation(range.default.try_into().unwrap());
        }
        if let Some(range) = uvc_ranges.contrast {
            let _ = devh.set_contrast(range.default.try_into().unwrap());
        }
        if let Some(range) = uvc_ranges.white_balance_temperature {
            let _ = devh.set_white_balance_temperature(range.default.try_into().unwrap());
        }
        if let Some(range) = uvc_ranges.sharpness {
            let _ = devh.set_sharpness(range.default.try_into().unwrap());
        }
        if let Some(range) = uvc_ranges.brightness {
            let _ = devh.set_brightness(range.default.try_into().unwrap());
        }
        if let Some(range) = uvc_ranges.gamma {
            let _ = devh.set_gamma(range.default.try_into().unwrap());
        }

        {
            let mut lock = ranges_store.lock().unwrap();
            *lock = Some(uvc_ranges);
        }

        // Turn off auto-exposure and auto-white-balance
        match devh.set_ae_mode(AutoExposureMode::Manual) {
            Err(_) => {
                // Fall back to ShutterPriority if supported
                info!("Failed to set auto-exposure mode to manual");
                let supported_ae_modes = devh.supported_ae_modes().unwrap_or(Vec::new());
                if supported_ae_modes.contains(&AutoExposureMode::ShutterPriority) {
                    match devh.set_ae_mode(AutoExposureMode::ShutterPriority) {
                        Err(_) => warn!("Failed to set shutter priority mode"),
                        _ => {}
                    };
                } else {
                    warn!("Shutter priority not supported!");
                }
            }
            _ => {}
        };
        let _ = devh.set_white_balance_temperature_auto(false);

        let initial_gain = f32::from_bits(current_gain_atomic.load(Ordering::Relaxed));
        current_exposure_atomic.store((1_000_000 / stream_format.fps) as i32, Ordering::Relaxed);

        let _ = devh.set_gain(initial_gain as u16);
        let _ = devh.set_exposure_abs(10000 / stream_format.fps);

        {
            let mut lock = config_store.lock().unwrap();
            *lock = Some(CameraConfig {
                model: "UVC Camera".to_string(),
                width: stream_format.width,
                height: stream_format.height,
                fps: stream_format.fps,
            });
        }

        let discard_frames = Arc::new(AtomicU32::new(stream_format.fps * 2));
        let discard_frames_clone = Arc::clone(&discard_frames);
        let store_clone = Arc::clone(&store);

        let _stream = streamh
            .start_stream(
                move |frame, _| {
                    if discard_frames_clone
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                            if x > 0 { Some(x - 1) } else { None }
                        })
                        .is_ok()
                    {
                        return;
                    }

                    let receive_time = std::time::Instant::now();
                    let timestamp = SystemTime::now();

                    let mut width = frame.width() as usize;
                    let mut height = frame.height() as usize;
                    let data = frame.to_bytes();

                    let mut image_data;
                    if frame.format() == FrameFormat::MJPEG {
                        let mut decompressor = match turbojpeg::Decompressor::new() {
                            Ok(d) => d,
                            Err(_) => return,
                        };
                        let header = match decompressor.read_header(data) {
                            Ok(h) => h,
                            Err(_) => return,
                        };
                        width = header.width;
                        height = header.height;
                        image_data = vec![0u8; width * height];
                        let image = turbojpeg::Image {
                            pixels: image_data.as_mut_slice(),
                            width,
                            pitch: width,
                            height,
                            format: turbojpeg::PixelFormat::GRAY,
                        };
                        if decompressor.decompress(data, image).is_err() {
                            return;
                        }
                    } else {
                        // Highly optimized loop for 8-bit mono extraction (Y from YUYV)
                        // YUYV sequence: Y0, U0, Y1, V0... (2 bytes per pixel, even bytes are Y)
                        image_data = Vec::with_capacity(width * height);
                        unsafe {
                            image_data.set_len(width * height);
                            let mut j = 0;
                            // Manual loop unrolling or just a tight loop
                            for i in (0..data.len()).step_by(2) {
                                if j < width * height {
                                    *image_data.get_unchecked_mut(j) = *data.get_unchecked(i);
                                    j += 1;
                                }
                            }
                        }
                    }

                    let post_process_time = receive_time.elapsed();
                    let mut lock = store_clone.image.lock().unwrap();
                    *lock = Some(Arc::new(Image {
                        data: image_data,
                        width: width as u32,
                        height: height as u32,
                        timestamp,
                        exposure_time_us: current_exposure_atomic.load(Ordering::Relaxed),
                        post_process_time,
                    }));
                    store_clone.cvar.notify_all();
                },
                (),
            )
            .expect("Failed to start stream");

        loop {
            // Handle commands
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(CaptureCommand::StopCaptureThread) => break,
                Ok(CaptureCommand::SetGain(gain)) => {
                    let _ = devh.set_gain(gain as u16);
                    let _ = discard_frames.fetch_add(10, Ordering::Relaxed);
                    *store.image.lock().unwrap() = None;
                }
                Ok(CaptureCommand::SetUvcControls(controls)) => {
                    let mut current_ae_mode = AutoExposureMode::Manual;
                    let supported_ae_modes = devh.supported_ae_modes().unwrap_or(Vec::new());
                    if supported_ae_modes.contains(&AutoExposureMode::ShutterPriority)
                        && !supported_ae_modes.contains(&AutoExposureMode::Manual)
                    {
                        current_ae_mode = AutoExposureMode::ShutterPriority;
                    }
                    if let Some(val) = controls.ae_mode {
                        current_ae_mode = match val {
                            1 => AutoExposureMode::Manual,
                            2 => AutoExposureMode::Auto,
                            4 => AutoExposureMode::ShutterPriority,
                            8 => AutoExposureMode::AperturePriority,
                            _ => AutoExposureMode::Manual,
                        };
                        // Enforce fallback if Manual isn't supported
                        if current_ae_mode == AutoExposureMode::Manual
                            && supported_ae_modes.contains(&AutoExposureMode::ShutterPriority)
                            && !supported_ae_modes.contains(&AutoExposureMode::Manual)
                        {
                            current_ae_mode = AutoExposureMode::ShutterPriority;
                        }
                    }
                    let _ = devh.set_ae_mode(current_ae_mode);
                    let _ = devh.set_white_balance_temperature_auto(false);

                    if let Some(val) = controls.saturation {
                        let _ = devh.set_saturation(val.try_into().unwrap_or(0));
                    }
                    if let Some(val) = controls.contrast {
                        let _ = devh.set_contrast(val.try_into().unwrap_or(0));
                    }
                    if let Some(val) = controls.white_balance_temperature {
                        let _ = devh.set_white_balance_temperature(val.try_into().unwrap_or(0));
                    }
                    if let Some(val) = controls.sharpness {
                        let _ = devh.set_sharpness(val.try_into().unwrap_or(0));
                    }
                    if let Some(val) = controls.brightness {
                        let _ = devh.set_brightness(val.try_into().unwrap_or(0));
                    }
                    if let Some(val) = controls.gamma {
                        let _ = devh.set_gamma(val.try_into().unwrap_or(0));
                    }
                    if let Some(val) = controls.gain {
                        let _ = devh.set_gain(val.try_into().unwrap_or(0));
                    }

                    if let Some(val) = controls.exposure_abs {
                        let _ = devh.set_exposure_abs(val);
                    } else {
                        let _ = devh.set_exposure_abs(10000 / stream_format.fps);
                    }
                    discard_frames.store(stream_format.fps, Ordering::Relaxed);
                    *store.image.lock().unwrap() = None;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.send_command(CaptureCommand::StopCaptureThread);
        if let Some(handle) = self.capture_thread.take() {
            let _ = handle.join();
        }
    }
}
