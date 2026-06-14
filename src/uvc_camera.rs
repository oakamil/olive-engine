// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use crate::{CameraConfig, CameraTrait, Image};
use log::{info, warn};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};
use turbojpeg;
use uvc::{AutoExposureMode, FrameFormat};

macro_rules! try_set_control {
    ($name:expr, $action:expr) => {
        if let Err(e) = $action {
            log::warn!("Failed to set {}: {:?}. Retrying in 100ms...", $name, e);
            std::thread::sleep(std::time::Duration::from_millis(100));
            if let Err(e2) = $action {
                log::warn!("Failed to set {} on retry: {:?}", $name, e2);
            } else {
                log::info!("Successfully set {} on retry.", $name);
            }
        }
    };
}

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
    SetExposure(i32),
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
    current_brightness: Arc<AtomicI32>,
    saved_gain: Arc<AtomicU32>,
    stack_count_config: Arc<AtomicUsize>,
    allow_stacking: bool,

    config: Arc<Mutex<Option<CameraConfig>>>,
    pub ranges: Arc<Mutex<Option<UvcRanges>>>,
    target_fps: u32,
    mjpeg_enabled: bool,
    target_res: Option<(u32, u32)>,
    fd: Option<i32>,
    descriptors: Option<Vec<u8>>,
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
        let current_brightness = Arc::clone(&self.current_brightness);
        let stack_count_config = Arc::clone(&self.stack_count_config);
        let target_fps = self.target_fps;
        let mjpeg_enabled = self.mjpeg_enabled;
        let target_res = self.target_res;
        let fd = self.fd;
        let descriptors = self.descriptors.clone();

        self.capture_thread = Some(thread::spawn(move || {
            Self::capture_loop(
                rx,
                store_clone,
                config_clone,
                ranges_clone,
                current_gain,
                current_exposure,
                current_brightness,
                target_fps,
                mjpeg_enabled,
                target_res,
                fd,
                descriptors,
                stack_count_config,
            );
        }));

        Ok(())
    }
    fn stop_capture(&self) {
        self.send_command(CaptureCommand::StopCaptureThread);
        self.store.cvar.notify_all();
    }

    // --- Configuration Getters & Setters ---

    fn get_gain(&self) -> f32 {
        f32::from_bits(self.current_gain.load(Ordering::Relaxed))
    }

    fn get_gain_range(&self) -> (f32, f32, f32) {
        let lock = self.ranges.lock().unwrap();
        if let Some(r) = lock.as_ref() {
            if let Some(g) = r.gain.as_ref() {
                return (g.min as f32, g.max as f32, g.max as f32);
            }
        }
        (0.0, 100.0, 100.0)
    }

    fn set_gain(&self, gain: f32) {
        let (min, max, _) = self.get_gain_range();
        let clamped_gain = gain.clamp(min, max).round();
        self.current_gain
            .store(clamped_gain.to_bits(), Ordering::Relaxed);
        self.send_command(CaptureCommand::SetGain(clamped_gain));
    }

    fn get_exposure(&self) -> i32 {
        self.current_exposure.load(Ordering::Relaxed)
            * (self.stack_count_config.load(Ordering::Relaxed).max(1) as i32)
    }

    fn set_exposure(&self, exposure_us: i32) {
        self.current_exposure.store(exposure_us, Ordering::Relaxed);
        self.send_command(CaptureCommand::SetExposure(exposure_us));
    }

    fn get_brightness(&self) -> Option<i32> {
        Some(self.current_brightness.load(Ordering::Relaxed))
    }

    fn set_stack_count(&self, count: usize) {
        self.stack_count_config
            .store(count.clamp(1, 10), Ordering::Relaxed);
    }

    fn enable_hcg(&self) {}

    fn disable_hcg(&self) {}

    fn get_last_capture(&self) -> Option<Arc<Image>> {
        let lock = self.store.image.lock().unwrap();
        lock.clone()
    }

    fn get_brightness_range(&self) -> Option<(i32, i32, i32)> {
        let ranges = self.ranges.lock().unwrap();
        if let Some(r) = ranges.as_ref() {
            if let Some(b) = r.brightness {
                return Some((
                    b.min,
                    b.max,
                    self.current_brightness.load(Ordering::Relaxed),
                ));
            }
        }
        None
    }

    fn set_brightness(&self, brightness: i32) {
        let mut controls = UvcControls::default();
        controls.brightness = Some(brightness);
        self.send_command(CaptureCommand::SetUvcControls(controls));
        self.current_brightness.store(brightness, Ordering::Relaxed);
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

    fn adjust_sensitivity(&self, factor: f64) -> bool {
        if !self.allow_stacking {
            return false;
        }
        let current = self.stack_count_config.load(Ordering::Relaxed);
        let new_count = if factor > 1.0 {
            (current + 1).min(5)
        } else if factor < 1.0 {
            (current.saturating_sub(1)).max(1)
        } else {
            current
        };
        if new_count != current {
            log::info!(
                "Adjusting sensitivity via stacking. Stack count changed from {} to {}",
                current,
                new_count
            );
            self.stack_count_config.store(new_count, Ordering::Relaxed);
            return true;
        }
        false
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
    pub fn new(
        target_fps: u32,
        mjpeg_enabled: bool,
        target_res: Option<(u32, u32)>,
        allow_stacking: bool,
        initial_gain: Option<f32>,
    ) -> Self {
        Self::with_fd(
            target_fps,
            mjpeg_enabled,
            target_res,
            None,
            None,
            allow_stacking,
            initial_gain,
        )
    }

    pub fn with_fd(
        target_fps: u32,
        mjpeg_enabled: bool,
        target_res: Option<(u32, u32)>,
        fd: Option<i32>,
        descriptors: Option<Vec<u8>>,
        allow_stacking: bool,
        initial_gain: Option<f32>,
    ) -> Self {
        log::info!(
            "Initializing UvcCamera with_fd: target_fps={}, mjpeg={}, fd={:?}, allow_stacking={}, initial_gain={:?}",
            target_fps,
            mjpeg_enabled,
            fd,
            allow_stacking,
            initial_gain
        );

        let store = Arc::new(CaptureStore {
            image: Mutex::new(None),
            cvar: Condvar::new(),
        });

        let current_exposure = Arc::new(AtomicI32::new(33_000)); // Default 30fps = 33ms
        let config = Arc::new(Mutex::new(None));
        let ranges = Arc::new(Mutex::new(None));

        Self {
            command_tx: None,
            store,
            capture_thread: None,
            current_gain: Arc::new(AtomicU32::new(initial_gain.unwrap_or(f32::NAN).to_bits())),
            current_exposure,
            current_brightness: Arc::new(AtomicI32::new(0)),
            saved_gain: Arc::new(AtomicU32::new(initial_gain.unwrap_or(f32::NAN).to_bits())),
            stack_count_config: Arc::new(AtomicUsize::new(1)),
            allow_stacking,
            config,
            ranges,
            target_fps,
            mjpeg_enabled,
            target_res,
            fd,
            descriptors,
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
        current_brightness_atomic: Arc<AtomicI32>,
        target_fps: u32,
        mjpeg_enabled: bool,
        target_res: Option<(u32, u32)>,
        fd: Option<i32>,
        descriptors: Option<Vec<u8>>,
        stack_count_config: Arc<AtomicUsize>,
    ) {
        let ctx = uvc::Context::new().expect("Failed to create UVC context");

        let mut _dev_keep_alive = None;
        let devh = if let Some(fd) = fd {
            if let Some(ref d) = descriptors {
                ctx.wrap_fd_with_descriptors(fd, d).unwrap_or_else(|e| {
                    panic!(
                        "Failed to open UVC device with descriptors. Error: {:?}, raw err code",
                        e
                    )
                })
            } else {
                ctx.wrap_fd(fd).expect("Failed to open UVC device with FD")
            }
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

        if best_format.is_none() {
            warn!("Strict MJPEG/FPS match failed. Trying FPS match only...");
            for format_desc in devh.supported_formats() {
                for frame_desc in format_desc.supported_formats() {
                    for &interval in frame_desc.intervals() {
                        let fps = 10_000_000 / interval;
                        if fps == target_fps {
                            best_format = Some(uvc::StreamFormat {
                                width: frame_desc.width() as u32,
                                height: frame_desc.height() as u32,
                                fps,
                                format: match frame_desc.subtype() {
                                    uvc::DescriptionSubtype::FormatMJPEG
                                    | uvc::DescriptionSubtype::FrameMJPEG => FrameFormat::MJPEG,
                                    _ => FrameFormat::Uncompressed,
                                },
                            });
                            break;
                        }
                    }
                    if best_format.is_some() {
                        break;
                    }
                }
                if best_format.is_some() {
                    break;
                }
            }
        }

        if best_format.is_none() {
            warn!("FPS match failed. Picking largest resolution...");
            for format_desc in devh.supported_formats() {
                for frame_desc in format_desc.supported_formats() {
                    let width = frame_desc.width() as u32;
                    let height = frame_desc.height() as u32;
                    if width * height > max_res {
                        max_res = width * height;
                        let fps = 10_000_000 / frame_desc.intervals()[0];
                        best_format = Some(uvc::StreamFormat {
                            width,
                            height,
                            fps,
                            format: match frame_desc.subtype() {
                                uvc::DescriptionSubtype::FormatMJPEG
                                | uvc::DescriptionSubtype::FrameMJPEG => FrameFormat::MJPEG,
                                _ => FrameFormat::Uncompressed,
                            },
                        });
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
                    info!(
                        "UVC Range for {}: min={}, max={}, default={}",
                        stringify!($field),
                        range.min,
                        range.max,
                        range.default
                    );
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

        {
            let mut lock = ranges_store.lock().unwrap();
            *lock = Some(uvc_ranges.clone());
        }

        {
            let mut lock = config_store.lock().unwrap();
            *lock = Some(CameraConfig {
                model: "UVC Camera".to_string(),
                width: stream_format.width,
                height: stream_format.height,
                fps: stream_format.fps,
                crop: None,
                initial_fov: None,
            });
        }

        let store_clone = Arc::clone(&store);
        let stack_count_config_clone = Arc::clone(&stack_count_config);
        let current_exposure_atomic_clone = Arc::clone(&current_exposure_atomic);

        struct StreamContext {
            stacked_count: usize,
            accumulator: Vec<u8>,
        }

        let stream_context = StreamContext {
            stacked_count: 0,
            accumulator: Vec::new(),
        };

        let _stream_handle = streamh
            .start_stream(
                move |frame, ctx: &mut StreamContext| {
                    log::info!(
                        "UVC callback fired! format={:?}, len={}",
                        frame.format(),
                        frame.to_bytes().len()
                    );
                    log::info!(
                        "Callback fired! Format={:?}, len={}",
                        frame.format(),
                        frame.to_bytes().len()
                    );

                    let receive_time = std::time::Instant::now();
                    let timestamp = SystemTime::now();

                    let mut width = frame.width() as usize;
                    let mut height = frame.height() as usize;
                    let data = frame.to_bytes();
                    log::info!(
                        "UVC frame arrived: {}x{}, format={:?}",
                        width,
                        height,
                        frame.format()
                    );

                    let mut image_data = Vec::new();
                    let mut direct_accumulate = false;
                    let target_stack = stack_count_config_clone.load(Ordering::Relaxed);

                    if frame.format() == FrameFormat::MJPEG {
                        let mut decompressor = match turbojpeg::Decompressor::new() {
                            Ok(d) => d,
                            Err(e) => {
                                log::error!("Failed to create turbojpeg decompressor: {:?}", e);
                                return;
                            }
                        };
                        let header = match decompressor.read_header(data) {
                            Ok(h) => h,
                            Err(e) => {
                                log::error!("MJPEG header read failed: {:?}", e);
                                return;
                            }
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
                        if let Err(e) = decompressor.decompress(data, image) {
                            log::error!("MJPEG decompression failed: {:?}", e);
                            return;
                        }
                    } else {
                        // Highly optimized loop for 8-bit mono extraction (Y from YUYV)
                        // YUYV sequence: Y0, U0, Y1, V0... (2 bytes per pixel, even bytes are Y)
                        if target_stack > 1
                            && ctx.stacked_count > 0
                            && ctx.accumulator.len() == width * height
                        {
                            direct_accumulate = true;
                            unsafe {
                                let mut j = 0;
                                for i in (0..data.len()).step_by(2) {
                                    if j < width * height {
                                        let a = ctx.accumulator.get_unchecked_mut(j);
                                        *a = a.saturating_add(*data.get_unchecked(i));
                                        j += 1;
                                    }
                                }
                            }
                        } else {
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
                    }

                    let mut is_ready = false;

                    if target_stack <= 1 {
                        ctx.accumulator = image_data;
                        is_ready = true;
                        ctx.stacked_count = 0;
                    } else {
                        if !direct_accumulate {
                            if ctx.stacked_count == 0 || ctx.accumulator.len() != width * height {
                                ctx.accumulator = image_data;
                            } else {
                                for (a, b) in ctx.accumulator.iter_mut().zip(image_data.iter()) {
                                    *a = a.saturating_add(*b);
                                }
                            }
                        }
                        ctx.stacked_count += 1;
                        if ctx.stacked_count >= target_stack {
                            is_ready = true;
                            ctx.stacked_count = 0;
                        }
                    }

                    if is_ready {
                        let final_data = ctx.accumulator.clone();
                        let post_process_time = receive_time.elapsed();
                        let mut lock = store_clone.image.lock().unwrap();
                        *lock = Some(Arc::new(Image {
                            data: final_data,
                            width: width as u32,
                            height: height as u32,
                            timestamp,
                            exposure_time_us: current_exposure_atomic_clone.load(Ordering::Relaxed)
                                * (target_stack.max(1) as i32),
                            post_process_time,
                        }));
                        store_clone.cvar.notify_all();
                    }
                },
                stream_context,
            )
            .expect("Failed to start stream");
        log::info!("UVC stream started successfully");

        std::thread::sleep(Duration::from_millis(500));

        // Turn off auto-exposure
        let supported_ae_modes = devh.supported_ae_modes().unwrap_or(Vec::new());
        if supported_ae_modes.contains(&AutoExposureMode::Manual) {
            try_set_control!("ae_mode manual", devh.set_ae_mode(AutoExposureMode::Manual));
        } else if supported_ae_modes.contains(&AutoExposureMode::ShutterPriority) {
            try_set_control!(
                "ae_mode shutter priority",
                devh.set_ae_mode(AutoExposureMode::ShutterPriority)
            );
        } else {
            log::warn!("Neither Manual nor Shutter priority auto exposure modes are supported!");
        }

        // Disable AWB
        try_set_control!("awb off", devh.set_white_balance_temperature_auto(false));

        // Per empirical results with an SC2210 camera, set all values to their defaults
        if let Some(range) = uvc_ranges.saturation {
            try_set_control!(
                "saturation",
                devh.set_saturation(range.default.try_into().unwrap())
            );
        }
        if let Some(range) = uvc_ranges.contrast {
            try_set_control!(
                "contrast",
                devh.set_contrast(range.default.try_into().unwrap())
            );
        }
        if let Some(range) = uvc_ranges.white_balance_temperature {
            try_set_control!(
                "white_balance_temp",
                devh.set_white_balance_temperature(range.default.try_into().unwrap())
            );
        }
        if let Some(range) = uvc_ranges.sharpness {
            try_set_control!(
                "sharpness",
                devh.set_sharpness(range.default.try_into().unwrap())
            );
        }
        if let Some(range) = uvc_ranges.brightness {
            let def: i16 = range.default.try_into().unwrap();
            try_set_control!("brightness", devh.set_brightness(def));
            current_brightness_atomic.store(range.default, Ordering::Relaxed);
        }
        if let Some(range) = uvc_ranges.gamma {
            try_set_control!("gamma", devh.set_gamma(range.default.try_into().unwrap()));
        }

        // Delaying to ensure that the camera fully transitions to manual / shutter priority mode
        std::thread::sleep(Duration::from_millis(100));

        current_exposure_atomic.store((1_000_000 / stream_format.fps) as i32, Ordering::Relaxed);
        try_set_control!("exposure", devh.set_exposure_abs(10000 / stream_format.fps));

        let mut initial_gain = f32::from_bits(current_gain_atomic.load(Ordering::Relaxed));
        if initial_gain.is_nan() {
            if let Some(r) = uvc_ranges.gain.as_ref() {
                initial_gain = r.max as f32;
                log::info!(
                    "No initial gain provided, defaulting to max gain: {}",
                    initial_gain
                );
            } else {
                initial_gain = 50.0;
                log::info!("No initial gain and no range, defaulting to 50.0");
            }
            current_gain_atomic.store(initial_gain.to_bits(), Ordering::Relaxed);
        }
        try_set_control!("gain", devh.set_gain(initial_gain as u16));

        // Keep stream_handle alive for the duration of the loop

        // Keep stream_handle alive for the duration of the loop

        loop {
            // Handle commands
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(CaptureCommand::StopCaptureThread) => break,
                Ok(CaptureCommand::SetGain(gain)) => {
                    try_set_control!("gain", devh.set_gain(gain as u16));

                    *store.image.lock().unwrap() = None;
                }
                Ok(CaptureCommand::SetExposure(exposure)) => {
                    try_set_control!("exposure", devh.set_exposure_abs((exposure / 100) as u32));

                    *store.image.lock().unwrap() = None;
                }
                Ok(CaptureCommand::SetUvcControls(controls)) => {
                    if let Some(val) = controls.ae_mode {
                        let mut current_ae_mode = match val {
                            1 => AutoExposureMode::Manual,
                            2 => AutoExposureMode::Auto,
                            4 => AutoExposureMode::ShutterPriority,
                            8 => AutoExposureMode::AperturePriority,
                            _ => AutoExposureMode::Manual,
                        };

                        let supported_ae_modes = devh.supported_ae_modes().unwrap_or(Vec::new());
                        // Enforce fallback if Manual isn't supported
                        if current_ae_mode == AutoExposureMode::Manual
                            && supported_ae_modes.contains(&AutoExposureMode::ShutterPriority)
                            && !supported_ae_modes.contains(&AutoExposureMode::Manual)
                        {
                            current_ae_mode = AutoExposureMode::ShutterPriority;
                        }

                        try_set_control!("ae_mode", devh.set_ae_mode(current_ae_mode));
                        try_set_control!("awb off", devh.set_white_balance_temperature_auto(false));
                    }
                    if let Some(val) = controls.saturation {
                        try_set_control!(
                            "saturation",
                            devh.set_saturation(val.try_into().unwrap_or(0))
                        );
                    }
                    if let Some(val) = controls.contrast {
                        try_set_control!(
                            "contrast",
                            devh.set_contrast(val.try_into().unwrap_or(0))
                        );
                    }
                    if let Some(val) = controls.white_balance_temperature {
                        try_set_control!(
                            "white_balance_temp",
                            devh.set_white_balance_temperature(val.try_into().unwrap_or(0))
                        );
                    }
                    if let Some(val) = controls.sharpness {
                        try_set_control!(
                            "sharpness",
                            devh.set_sharpness(val.try_into().unwrap_or(0))
                        );
                    }
                    if let Some(val) = controls.brightness {
                        try_set_control!(
                            "brightness",
                            devh.set_brightness(val.try_into().unwrap_or(0))
                        );
                    }
                    if let Some(val) = controls.gamma {
                        try_set_control!("gamma", devh.set_gamma(val.try_into().unwrap_or(0)));
                    }
                    if let Some(val) = controls.gain {
                        try_set_control!("gain", devh.set_gain(val.try_into().unwrap_or(0)));
                    }

                    if let Some(val) = controls.exposure_abs {
                        try_set_control!("exposure_abs", devh.set_exposure_abs(val));
                    }

                    *store.image.lock().unwrap() = None;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        log::info!("UVC capture loop exiting cleanly...");
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.send_command(CaptureCommand::StopCaptureThread);
        self.store.cvar.notify_all();
        if let Some(handle) = self.capture_thread.take() {
            let _ = handle.join();
        }
    }
}
