// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use crate::{CameraConfig, CameraTrait, Image};
use log::info;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicI32, AtomicU32, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};
use uvc::{AutoExposureMode, FrameFormat};

// --- Data Structures ---

/// Thread-safe atomic store for the latest capture.
struct CaptureStore {
    image: Mutex<Option<Arc<Image>>>,
    cvar: Condvar,
}

/// Commands sent from the main thread to the capture thread.
enum CaptureCommand {
    SetGain(f32),
    SetExposure(i32),
    StopCaptureThread,
}

pub struct Camera {
    command_tx: Option<mpsc::Sender<CaptureCommand>>,
    store: Arc<CaptureStore>,
    capture_thread: Option<JoinHandle<()>>,

    // Cached atomic values for lock-free getter access
    current_gain: Arc<AtomicU32>, // f32 stored as u32 bits for atomic operations
    current_exposure: Arc<AtomicI32>,
    min_gain: Arc<AtomicU32>,
    max_gain: Arc<AtomicU32>,

    config: Arc<Mutex<Option<CameraConfig>>>,
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
        let current_gain = Arc::clone(&self.current_gain);
        let current_exposure = Arc::clone(&self.current_exposure);
        let min_gain_store = Arc::clone(&self.min_gain);
        let max_gain_store = Arc::clone(&self.max_gain);

        self.capture_thread = Some(thread::spawn(move || {
            Self::capture_loop(
                rx,
                store_clone,
                config_clone,
                current_gain,
                current_exposure,
                min_gain_store,
                max_gain_store,
            );
        }));

        Ok(())
    }

    // --- Configuration Getters & Setters ---

    fn get_gain(&self) -> f32 {
        f32::from_bits(self.current_gain.load(Ordering::Relaxed))
    }

    fn get_min_gain(&self) -> f32 {
        f32::from_bits(self.min_gain.load(Ordering::Relaxed))
    }

    fn get_max_gain(&self) -> f32 {
        f32::from_bits(self.max_gain.load(Ordering::Relaxed))
    }

    fn set_gain(&self, gain: f32) {
        let min = self.get_min_gain();
        let max = self.get_max_gain();
        let clamped_gain = gain.clamp(min, max);
        self.current_gain
            .store(clamped_gain.to_bits(), Ordering::Relaxed);
        self.send_command(CaptureCommand::SetGain(clamped_gain));
    }

    fn get_exposure(&self) -> i32 {
        self.current_exposure.load(Ordering::Relaxed)
    }

    fn set_exposure(&self, exposure_us: i32) {
        self.current_exposure.store(exposure_us, Ordering::Relaxed);
        self.send_command(CaptureCommand::SetExposure(exposure_us));
    }

    fn enable_hcg(&self) {}

    fn disable_hcg(&self) {}

    fn get_last_capture(&self) -> Option<Arc<Image>> {
        let lock = self.store.image.lock().unwrap();
        lock.clone()
    }

    fn wait_for_next_capture(&self, last_timestamp: Option<SystemTime>) -> Arc<Image> {
        let mut lock = self.store.image.lock().unwrap();
        loop {
            if let Some(img) = &*lock {
                if last_timestamp.is_none() || Some(img.timestamp) != last_timestamp {
                    return img.clone();
                }
            }
            lock = self.store.cvar.wait(lock).unwrap();
        }
    }
}

impl Camera {
    /// Instantiates the camera module and sets initial cached state.
    pub fn new(_isp_mode: bool) -> Self {
        let store = Arc::new(CaptureStore {
            image: Mutex::new(None),
            cvar: Condvar::new(),
        });

        let current_exposure = Arc::new(AtomicI32::new(50000));
        let min_gain = Arc::new(AtomicU32::new(0.0f32.to_bits()));
        let max_gain = Arc::new(AtomicU32::new(100.0f32.to_bits()));
        let config = Arc::new(Mutex::new(None));

        Self {
            command_tx: None,
            store,
            capture_thread: None,
            current_gain: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            current_exposure,
            min_gain,
            max_gain,
            config,
        }
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
        current_gain_atomic: Arc<AtomicU32>,
        current_exposure_atomic: Arc<AtomicI32>,
        min_gain_store: Arc<AtomicU32>,
        max_gain_store: Arc<AtomicU32>,
    ) {
        let ctx = uvc::Context::new().expect("Failed to create UVC context");
        let dev = ctx
            .find_device(None, None, None)
            .expect("Failed to find UVC device");
        let devh = dev.open().expect("Failed to open UVC device");

        // Find highest resolution uncompressed (YUV) format that supports at least 30fps
        let stream_format = devh
            .get_preferred_format(|f1, f2| {
                let f1_is_uncomp = f1.format == FrameFormat::Uncompressed;
                let f2_is_uncomp = f2.format == FrameFormat::Uncompressed;

                if f1_is_uncomp != f2_is_uncomp {
                    return if f1_is_uncomp { f1 } else { f2 };
                }

                let f1_fps_ok = f1.fps >= 30;
                let f2_fps_ok = f2.fps >= 30;

                if f1_fps_ok != f2_fps_ok {
                    return if f1_fps_ok { f1 } else { f2 };
                }

                let f1_res = f1.width * f1.height;
                let f2_res = f2.width * f2.height;

                if f1_res != f2_res {
                    return if f1_res > f2_res { f1 } else { f2 };
                }

                if f1.fps > f2.fps { f1 } else { f2 }
            })
            .expect("Failed to find suitable YUV format");

        info!("Selected UVC format: {:?}", stream_format);

        // Set default controls
        //if let Ok(range) = devh.saturation_range() {
        //    let _ = devh.set_saturation(range.default);
        //}
        //if let Ok(range) = devh.contrast_range() {
        //    let _ = devh.set_contrast(range.default);
        //}
        //if let Ok(range) = devh.white_balance_temperature_range() {
        //    let _ = devh.set_white_balance_temperature(range.default);
        //}
        if let Ok(range) = devh.sharpness_range() {
            let _ = devh.set_sharpness(range.min);
        }
        //if let Ok(range) = devh.brightness_range() {
        //    let _ = devh.set_brightness(range.default);
        //}
        if let Ok(range) = devh.gamma_range() {
            let _ = devh.set_gamma(range.min);
        }

        // Turn off auto-exposure and auto-white-balance
        let _ = devh.set_ae_mode(AutoExposureMode::Manual);
        let _ = devh.set_white_balance_temperature_auto(false);

        // Update gain limits
        if let Ok(range) = devh.gain_range() {
            min_gain_store.store((range.min as f32).to_bits(), Ordering::Relaxed);
            max_gain_store.store((range.max as f32).to_bits(), Ordering::Relaxed);
        }

        let initial_gain = f32::from_bits(current_gain_atomic.load(Ordering::Relaxed));
        let initial_exposure = current_exposure_atomic.load(Ordering::Relaxed);

        let _ = devh.set_gain(initial_gain as u16);
        let _ = devh.set_exposure_abs(initial_exposure as u32 / 100); // libuvc exposure is usually in 0.1ms units

        {
            let mut lock = config_store.lock().unwrap();
            *lock = Some(CameraConfig {
                model: "UVC Camera".to_string(),
                width: stream_format.width,
                height: stream_format.height,
            });
        }

        let mut streamh = devh
            .get_stream_handle_with_format(stream_format)
            .expect("Failed to get stream handle");

        let _stream = streamh
            .start_stream(
                move |frame, _| {
                    let receive_time = std::time::Instant::now();
                    let timestamp = SystemTime::now();

                    let width = frame.width() as usize;
                    let height = frame.height() as usize;
                    let data = frame.to_bytes();

                    // Highly optimized loop for 8-bit mono extraction (Y from YUYV)
                    // YUYV sequence: Y0, U0, Y1, V0... (2 bytes per pixel, even bytes are Y)
                    let mut image_data = Vec::with_capacity(width * height);
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

                    let post_process_time = receive_time.elapsed();
                    let mut lock = store.image.lock().unwrap();
                    *lock = Some(Arc::new(Image {
                        data: image_data,
                        timestamp,
                        exposure_time_us: current_exposure_atomic.load(Ordering::Relaxed),
                        post_process_time,
                    }));
                    store.cvar.notify_all();
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
                }
                Ok(CaptureCommand::SetExposure(exp)) => {
                    let _ = devh.set_exposure_abs(exp as u32 / 100);
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
