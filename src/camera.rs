// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use crate::{CameraConfig, CameraTrait, Image};
use libcamera::{
    camera::ActiveCamera,
    camera_manager::CameraManager,
    control::ControlList,
    controls::{AeEnable, AnalogueGain, ExposureTime, FrameDurationLimits},
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    properties,
    request::ReuseFlag,
    stream::StreamRole,
};
use log::info;
use rayon::prelude::*;
use rppal::i2c::I2c;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::asm;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicI32, AtomicU32, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

// --- Defined Constants ---
const I2C_ADDR: u16 = 0x1A;
const INITIAL_EXPOSURE_US: i32 = 50_000;
const I2C_BUSES_TO_SCAN: &[u8] = &[0, 1, 4, 6, 10, 11, 12, 13, 14, 15];

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
    SetHcg(bool),
    StopCaptureThread,
}

/// Represents metadata for the selected camera configuration.
#[derive(Clone)]
pub struct LibCameraConfiguration {
    pub model: String,
    pub format: PixelFormat,
    pub bit_depth: u8,
    pub is_color: bool,
    pub is_pisp: bool,
    pub is_packed: bool,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
}

fn parse_bit_depth(s: &str) -> u8 {
    let mut depth = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() {
            depth.push(c);
        } else if !depth.is_empty() {
            // Stop at the first non-digit after we've started collecting digits
            break;
        }
    }
    depth.parse().unwrap_or(8)
}

fn is_color(s: &str) -> bool {
    !s.contains("GREY")
        && (s.contains('R')
            || s.contains('G')
            || s.contains('B')
            || s.contains('Y')
            || s.contains('V'))
}

fn is_hcg_supported(model: &str) -> bool {
    let model = model.to_lowercase();
    model.contains("imx290")
        || model.contains("imx327")
        || model.contains("imx462")
        || model.contains("imx662")
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

    isp_mode: bool,
    config: Arc<Mutex<Option<LibCameraConfiguration>>>,
}

impl CameraTrait for Camera {
    fn get_config(&self) -> Option<CameraConfig> {
        self.config.lock().unwrap().as_ref().map(|c| CameraConfig {
            model: c.model.clone(),
            width: c.width,
            height: c.height,
            fps: 0,
        })
    }

    /// Kicks off the capture operation on a separate thread.
    /// This thread handles libcamera initialization, format selection, and the capture loop.
    fn start_capture(&mut self) -> Result<(), String> {
        if self.capture_thread.is_some() {
            return Ok(()); // Already running
        }

        let (tx, rx) = mpsc::channel();
        self.command_tx = Some(tx);

        let store_clone = Arc::clone(&self.store);
        let isp_mode = self.isp_mode;
        let config_clone = Arc::clone(&self.config);
        let max_gain = Arc::clone(&self.max_gain);
        self.capture_thread = Some(thread::spawn(move || {
            Self::capture_loop(rx, store_clone, isp_mode, config_clone, max_gain);
        }));

        Ok(())
    }

    // --- Configuration Getters & Setters ---

    /// Returns the currently set analog gain (lock-free).
    fn get_gain(&self) -> f32 {
        f32::from_bits(self.current_gain.load(Ordering::Relaxed))
    }

    /// Returns the minimum analog gain allowed by the sensor (lock-free).
    fn get_min_gain(&self) -> f32 {
        f32::from_bits(self.min_gain.load(Ordering::Relaxed))
    }

    /// Returns the maximum analog gain allowed by the sensor (lock-free).
    fn get_max_gain(&self) -> f32 {
        f32::from_bits(self.max_gain.load(Ordering::Relaxed))
    }

    /// Sets the analog gain and notifies the capture thread to apply it.
    /// The value is clamped to the sensor's supported range.
    fn set_gain(&self, gain: f32) {
        let min = self.get_min_gain();
        let max = self.get_max_gain();
        let clamped_gain = gain.clamp(min, max);
        self.current_gain
            .store(clamped_gain.to_bits(), Ordering::Relaxed);
        self.send_command(CaptureCommand::SetGain(clamped_gain));
    }

    /// Returns the currently set exposure in microseconds (lock-free).
    fn get_exposure(&self) -> i32 {
        self.current_exposure.load(Ordering::Relaxed)
    }

    /// Sets the exposure in microseconds and notifies the capture thread to apply it.
    fn set_exposure(&self, exposure_us: i32) {
        self.current_exposure.store(exposure_us, Ordering::Relaxed);
        self.send_command(CaptureCommand::SetExposure(exposure_us));
    }

    /// Enables High Conversion Gain (HCG) via I2C on address 0x1A.
    fn enable_hcg(&self) {
        self.send_command(CaptureCommand::SetHcg(true));
    }

    /// Disables High Conversion Gain (HCG) via I2C on address 0x1A.
    fn disable_hcg(&self) {
        self.send_command(CaptureCommand::SetHcg(false));
    }

    // --- Data Retrieval ---

    /// Returns the last capture from the store. Returns None if empty.
    /// Optimization: Returns an Arc to avoid copying the heap buffer.
    fn get_last_capture(&self) -> Option<Arc<Image>> {
        let lock = self.store.image.lock().unwrap();
        lock.clone()
    }

    /// Blocks until an image newer than `last_timestamp` is available, then returns it.
    /// Optimization: Returns an Arc to avoid copying the heap buffer.
    fn wait_for_next_capture(&self, last_timestamp: Option<SystemTime>) -> Arc<Image> {
        let mut lock = self.store.image.lock().unwrap();
        loop {
            if let Some(img) = &*lock {
                if last_timestamp.is_none() || Some(img.timestamp) != last_timestamp {
                    return img.clone();
                }
            }
            // Wait to be notified by the capture thread that a new image is ready
            lock = self.store.cvar.wait(lock).unwrap();
        }
    }
}

impl Camera {
    /// Instantiates the camera module and sets initial cached state.
    /// Performs hardware discovery immediately to populate gain limits.
    pub fn new(isp_mode: bool) -> Self {
        let store = Arc::new(CaptureStore {
            image: Mutex::new(None),
            cvar: Condvar::new(),
        });

        let current_exposure = Arc::new(AtomicI32::new(INITIAL_EXPOSURE_US));
        let min_gain = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let max_gain = Arc::new(AtomicU32::new(31.0f32.to_bits()));
        let config = Arc::new(Mutex::new(None));

        // Perform hardware discovery immediately
        {
            unsafe {
                std::env::set_var("LIBCAMERA_LOG_LEVELS", "*:ERROR");
            }
            libcamera::logging::log_set_target(libcamera::logging::LoggingTarget::None).ok();

            if let Ok(mut mgr) = CameraManager::new() {
                if mgr.start().is_ok() {
                    let cameras = mgr.cameras();
                    if let Some(camera) = cameras.get(0) {
                        if let Ok(mut active_camera) = camera.acquire() {
                            if let Ok((cam_config, _)) = Self::configure_camera(
                                &mut active_camera,
                                StreamRole::Raw,
                                isp_mode,
                            ) {
                                *config.lock().unwrap() = Some(cam_config);

                                use libcamera::control::ControlEntry;
                                if let Ok(info) = active_camera.controls().find(AnalogueGain::ID) {
                                    let min_val = info.min().clone();
                                    let max_val = info.max().clone();
                                    if let libcamera::control_value::ControlValue::Float(min_vec) =
                                        min_val
                                    {
                                        if let libcamera::control_value::ControlValue::Float(
                                            max_vec,
                                        ) = max_val
                                        {
                                            if let (Some(&min), Some(&max)) =
                                                (min_vec.first(), max_vec.first())
                                            {
                                                min_gain.store(min.to_bits(), Ordering::Relaxed);
                                                max_gain.store(max.to_bits(), Ordering::Relaxed);
                                                info!(
                                                    "Camera analogue gain range discovered: {} - {}",
                                                    min, max
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let current_gain = Arc::new(AtomicU32::new(max_gain.load(Ordering::Relaxed)));

        Self {
            command_tx: None,
            store,
            capture_thread: None,
            current_gain,
            current_exposure,
            min_gain,
            max_gain,
            isp_mode,
            config,
        }
    }

    fn store_as_8bit(
        store: &Arc<CaptureStore>,
        config: &LibCameraConfiguration,
        buf_data: &[u8],
        exposure_time_us: i32,
        timestamp: SystemTime,
        receive_time: std::time::Instant,
    ) {
        let width = config.width as usize;
        let height = config.height as usize;
        let stride = config.stride as usize;

        // Optimization: Use with_capacity and set_len to avoid zero-initializing 2MB+ of memory.
        // This is safe because we guaranteed to overwrite every byte in the loops below.
        let mut image_data: Vec<u8> = Vec::with_capacity(width * height);
        unsafe { image_data.set_len(width * height) };

        let is_10_bit = config.bit_depth == 10;
        let is_12_bit = config.bit_depth == 12;
        let is_16_bit = config.bit_depth == 16;

        let rows_per_core = height / 2;

        if is_16_bit {
            buf_data
                .par_chunks_exact(stride)
                .zip(image_data.par_chunks_exact_mut(width))
                .with_min_len(rows_per_core)
                .for_each(|(buf_row, pix_row)| {
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        let buf_ptr = buf_row.as_ptr();
                        let pix_ptr = pix_row.as_mut_ptr();

                        let mut i = 0;
                        let mut j = 0;

                        // Process 32 bytes (16 pixels) at a time
                        while i + 32 <= width * 2 {
                            // vld2q_u8 deinterleaves: v.0 gets even bytes, v.1 gets odd bytes
                            // Since ARM is little-endian and data is MSB-justified in 16-bit words,
                            // the high byte (MSB) is the second byte of each pair (odd index).
                            let v = vld2q_u8(buf_ptr.add(i));
                            vst1q_u8(pix_ptr.add(j), v.1);
                            i += 32;
                            j += 16;
                        }

                        // Fallback for remaining pixels in the row
                        for k in 0..(width - j) {
                            *pix_ptr.add(j + k) = *buf_ptr.add((j + k) * 2 + 1);
                        }
                    }

                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        let buf_row_end = width * 2;
                        for (src_chunk, dst) in buf_row[..buf_row_end]
                            .chunks_exact(2)
                            .zip(pix_row.iter_mut())
                        {
                            *dst = src_chunk[1];
                        }
                    }
                });
        } else if !is_10_bit && !is_12_bit {
            if stride == width {
                // A single contiguous memory copy is extremely fast and doesn't need rayon overhead
                image_data[..width * height].copy_from_slice(&buf_data[..width * height]);
            } else {
                buf_data
                    .par_chunks_exact(stride)
                    .zip(image_data.par_chunks_exact_mut(width))
                    .with_min_len(rows_per_core)
                    .for_each(|(buf_row, pix_row)| {
                        pix_row.copy_from_slice(&buf_row[..width]);
                    });
            }
        } else if config.is_packed {
            // 10-bit path
            if is_10_bit {
                let valid_buf_len = width * 5 / 4;

                buf_data
                    .par_chunks_exact(stride)
                    .zip(image_data.par_chunks_exact_mut(width))
                    .with_min_len(rows_per_core)
                    .for_each(|(buf_row_full, pix_row_full)| {
                        let buf_slice = &buf_row_full[..valid_buf_len];
                        let pix_slice = &mut pix_row_full[..width];

                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            let mask0 = vld1q_u8(
                                [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18].as_ptr(),
                            );
                            let mask1 = vld1q_u8(
                                [4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22]
                                    .as_ptr(),
                            );
                            let mask2 = vld1q_u8(
                                [8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26]
                                    .as_ptr(),
                            );
                            let mask3 = vld1q_u8(
                                [
                                    12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30,
                                ]
                                .as_ptr(),
                            );

                            let mut i = 0;
                            let mut j = 0;

                            // 80 bytes in -> 64 pixels out
                            while i + 80 <= buf_slice.len() && j + 64 <= pix_slice.len() {
                                let in_ptr = buf_slice.as_ptr().add(i);
                                let out_ptr = pix_slice.as_mut_ptr().add(j);

                                asm!(
                                    "prfm pldl1keep, [{0}, #256]",
                                    in(reg) in_ptr,
                                    options(readonly, preserves_flags, nostack)
                                );

                                let v0 = vld1q_u8(in_ptr);
                                let v1 = vld1q_u8(in_ptr.add(16));
                                let v2 = vld1q_u8(in_ptr.add(32));

                                let out0 = vqtbl2q_u8(uint8x16x2_t(v0, v1), mask0);
                                let v3 = vld1q_u8(in_ptr.add(48));

                                let out1 = vqtbl2q_u8(uint8x16x2_t(v1, v2), mask1);
                                let v4 = vld1q_u8(in_ptr.add(64));

                                let out2 = vqtbl2q_u8(uint8x16x2_t(v2, v3), mask2);
                                vst1q_u8(out_ptr, out0);

                                let out3 = vqtbl2q_u8(uint8x16x2_t(v3, v4), mask3);
                                vst1q_u8(out_ptr.add(16), out1);
                                vst1q_u8(out_ptr.add(32), out2);
                                vst1q_u8(out_ptr.add(48), out3);

                                i += 80;
                                j += 64;
                            }

                            // Edge remainder
                            for (buf_chunk, pix_chunk) in buf_slice[i..]
                                .chunks_exact(5)
                                .zip(pix_slice[j..].chunks_exact_mut(4))
                            {
                                pix_chunk[0] = buf_chunk[0];
                                pix_chunk[1] = buf_chunk[1];
                                pix_chunk[2] = buf_chunk[2];
                                pix_chunk[3] = buf_chunk[3];
                            }
                        }

                        // Scalar Fallback for x86 / non-aarch64 machines
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            for (buf_chunk, pix_chunk) in
                                buf_slice.chunks_exact(5).zip(pix_slice.chunks_exact_mut(4))
                            {
                                pix_chunk[0] = buf_chunk[0];
                                pix_chunk[1] = buf_chunk[1];
                                pix_chunk[2] = buf_chunk[2];
                                pix_chunk[3] = buf_chunk[3];
                            }
                        }
                    });
            }
            // 12-bit path
            else if is_12_bit {
                let valid_buf_len = width * 3 / 2;

                buf_data
                    .par_chunks_exact(stride)
                    .zip(image_data.par_chunks_exact_mut(width))
                    .with_min_len(rows_per_core)
                    .for_each(|(buf_row_full, pix_row_full)| {
                        let buf_slice = &buf_row_full[..valid_buf_len];
                        let pix_slice = &mut pix_row_full[..width];

                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            // For 12-bit, we drop every 3rd byte.
                            // Lowest Common Multiple for 16-byte registers is 48 bytes (3 registers).
                            // This outputs 32 bytes (2 registers).
                            let mask0 = vld1q_u8(
                                [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22].as_ptr(),
                            );
                            let mask1 = vld1q_u8(
                                [8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30]
                                    .as_ptr(),
                            );

                            let mut i = 0;
                            let mut j = 0;

                            // 48 bytes in -> 32 pixels out
                            while i + 48 <= buf_slice.len() && j + 32 <= pix_slice.len() {
                                let in_ptr = buf_slice.as_ptr().add(i);
                                let out_ptr = pix_slice.as_mut_ptr().add(j);

                                asm!(
                                    "prfm pldl1keep, [{0}, #256]",
                                    in(reg) in_ptr,
                                    options(readonly, preserves_flags, nostack)
                                );

                                let v0 = vld1q_u8(in_ptr);
                                let v1 = vld1q_u8(in_ptr.add(16));
                                let v2 = vld1q_u8(in_ptr.add(32));

                                let out0 = vqtbl2q_u8(uint8x16x2_t(v0, v1), mask0);
                                let out1 = vqtbl2q_u8(uint8x16x2_t(v1, v2), mask1);

                                vst1q_u8(out_ptr, out0);
                                vst1q_u8(out_ptr.add(16), out1);

                                i += 48;
                                j += 32;
                            }

                            // Edge remainder
                            for (buf_chunk, pix_pair) in buf_slice[i..]
                                .chunks_exact(3)
                                .zip(pix_slice[j..].chunks_exact_mut(2))
                            {
                                pix_pair[0] = buf_chunk[0];
                                pix_pair[1] = buf_chunk[1];
                            }
                        }

                        // Scalar Fallback for x86 / non-aarch64 machines
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            for (buf_chunk, pix_pair) in
                                buf_slice.chunks_exact(3).zip(pix_slice.chunks_exact_mut(2))
                            {
                                pix_pair[0] = buf_chunk[0];
                                pix_pair[1] = buf_chunk[1];
                            }
                        }
                    });
            }
        }

        let post_process_time = receive_time.elapsed();
        let mut lock = store.image.lock().unwrap();
        *lock = Some(Arc::new(Image {
            data: image_data,
            timestamp,
            exposure_time_us,
            post_process_time,
        }));
        store.cvar.notify_all();
    }

    // --- Private Helpers ---

    /// Applies the standard control settings to a ControlList.
    fn apply_controls(controls: &mut ControlList, gain: f32, exposure_us: i32) {
        let _ = controls.set(AeEnable(false));
        let _ = controls.set(AnalogueGain(gain));
        let _ = controls.set(ExposureTime(exposure_us));
        let _ = controls.set(FrameDurationLimits([16667, 1000000]));
    }

    /// Queues a command to the capture thread.
    fn send_command(&self, cmd: CaptureCommand) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(cmd); // Ignore send errors if thread has died
        }
    }

    const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
        (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
    }

    /// Scans the defined I2C buses to find the camera at address 0x1A using rppal.
    fn detect_i2c_bus() -> Option<u8> {
        for &bus in I2C_BUSES_TO_SCAN {
            if let Ok(mut i2c) = I2c::with_bus(bus) {
                // Without I2C_SLAVE_FORCE, set_slave_address will fail with OS Error 16 (EBUSY)
                // if the V4L2 kernel driver has already claimed the camera sensor.
                match i2c.set_slave_address(I2C_ADDR) {
                    Ok(_) => {
                        // The address is free. Try a simple read to ping the device.
                        if i2c.smbus_receive_byte().is_ok() {
                            return Some(bus);
                        }
                    }
                    Err(rppal::i2c::Error::Io(e)) if e.raw_os_error() == Some(16) => {
                        // EBUSY (16) explicitly means a driver is using this address on this bus.
                        // Since we are looking for the active camera, this is exactly what we want!
                        return Some(bus);
                    }
                    Err(_) => {} // Other I/O errors, move on to the next bus
                }
            }
        }
        None
    }

    /// Writes raw bytes to the detected I2C bus to enable/disable HCG using `i2cset`.
    /// Bypasses kernel driver locks by enforcing the proper I2C_RDWR ioctl block via i2c-tools.
    fn write_i2c_hcg(enable: bool) -> std::io::Result<()> {
        let bus = Self::detect_i2c_bus().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "Camera I2C bus not found")
        })?;

        // The target value for register 0x3009
        let val = if enable { "0x11" } else { "0x01" };

        log::debug!("Executing: i2cset -y -f {} 0x1a 0x30 0x09 {} i", bus, val);

        // i2cset -y (yes) -f (force) <bus> <chip-addr> <reg-high> <reg-low> <val> i (block data)
        let output = std::process::Command::new("i2cset")
            .arg("-y")
            .arg("-f")
            .arg(bus.to_string())
            .arg("0x1a") // I2C Address
            .arg("0x30") // Register High Byte
            .arg("0x09") // Register Low Byte
            .arg(val) // Data Byte
            .arg("i") // I2C Block Write Mode
            .output()?;

        if !output.status.success() {
            let err_msg = String::from_utf8_lossy(&output.stderr);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("i2cset failed: {}", err_msg.trim()),
            ));
        }

        Ok(())
    }

    fn configure_camera(
        active_camera: &mut ActiveCamera<'_>,
        role: StreamRole,
        isp_mode: bool,
    ) -> Result<(LibCameraConfiguration, libcamera::stream::Stream), String> {
        let model = active_camera
            .properties()
            .get::<properties::Model>()
            .map(|m| m.to_string())
            .unwrap_or_else(|_| "Unknown".to_string());
        info!("Camera model: {}", model);

        if isp_mode {
            info!("Using ISP mode");

            let mut cfgs = active_camera
                .generate_configuration(&[StreamRole::VideoRecording])
                .ok_or_else(|| "Failed to generate configuration".to_string())?;

            // --- 1. Identify Target Formats ---
            let stream_cfg = cfgs
                .get_mut(0)
                .ok_or_else(|| "Failed to get stream configuration".to_string())?;
            let formats = stream_cfg.formats().pixel_formats();
            let mut format_r8 = None;
            let mut format_yuv = None;

            for i in 0..formats.len() {
                let format = formats.get(i).unwrap();
                let fmt_str = format!("{:?}", format);
                if fmt_str == "R8" || fmt_str == "GREY" || fmt_str == "Y8" {
                    format_r8 = Some(format);
                }
                if fmt_str == "YUV420" {
                    format_yuv = Some(format);
                }
            }

            // --- 2. Sniff Native Sensor Mode ---
            // We generate a temporary Raw configuration to read the exact dimensions
            // and bit-depth the sensor driver inherently supports.
            let raw_cfg = active_camera
                .generate_configuration(&[StreamRole::Raw])
                .ok_or_else(|| "Failed to generate raw configuration".to_string())?;
            let raw_stream = raw_cfg.get(0).unwrap();
            let raw_size = raw_stream.get_size();
            let raw_fmt_str = format!("{:?}", raw_stream.get_pixel_format());

            // Extract the bit depth from the raw format string (e.g., "SRGGB12" -> 12)
            let bit_depth = if raw_fmt_str.contains("12") {
                12
            } else if raw_fmt_str.contains("10") {
                10
            } else if raw_fmt_str.contains("14") {
                14
            } else {
                8
            };

            info!(
                "Sniffed native sensor mode: {}x{} at {}-bit",
                raw_size.width, raw_size.height, bit_depth
            );

            // --- 3. Apply Sensor Configuration (Lock the FOV) ---
            let mut sensor_config = libcamera::camera::SensorConfiguration::new();
            sensor_config.set_output_size(raw_size.width, raw_size.height);
            sensor_config.set_bit_depth(bit_depth);

            // The ISP Scaler MUST know the analog crop bounds to calculate the scale factor.
            sensor_config.set_analog_crop(libcamera::geometry::Rectangle {
                x: 0,
                y: 0,
                width: raw_size.width,
                height: raw_size.height,
            });
            // Explicitly disable any hardware binning/skipping at the physical sensor level
            sensor_config.set_binning(1, 1);
            sensor_config.set_skipping(1, 1, 1, 1);

            cfgs.set_sensor_configuration(sensor_config);

            // --- 4. Calculate Strict Aspect Ratio Downsampling ---
            // Because we locked the sensor to its native dimensions, our requested stream
            // MUST mathematically match its aspect ratio to keep the hardware scaler happy.
            let downsample_factor = if (raw_size.width * raw_size.height) < 3_000_000 {
                2
            } else {
                4
            };
            let target_width = raw_size.width / downsample_factor;
            let target_height = raw_size.height / downsample_factor;

            // --- 5. Configure Stream & Fallback Logic ---
            // We prioritize R8/GREY (ideal for Mono tuning files). If the pipeline rejects it
            // and falls back to RAW (common with Color tuning files), we seamlessly retry with YUV420.
            let mut success = false;
            let format_candidates = [("Mono 8-bit", format_r8), ("YUV420 Planar", format_yuv)];

            for (desc, format_opt) in format_candidates.iter() {
                if let Some(fmt) = format_opt {
                    // Scope block to safely drop the mutable reference before validating
                    {
                        let mut stream0 = cfgs.get_mut(0).unwrap();
                        stream0.set_pixel_format(*fmt);
                        stream0.set_size(libcamera::geometry::Size {
                            width: target_width,
                            height: target_height,
                        });
                    }

                    cfgs.validate();
                    let resulted_fmt = format!("{:?}", cfgs.get(0).unwrap().get_pixel_format());

                    // If the pipeline didn't panic and force a high-bit/RAW format, it worked!
                    if !resulted_fmt.contains("16")
                        && !resulted_fmt.contains("10")
                        && !resulted_fmt.contains("12")
                        && !resulted_fmt.starts_with('S')
                    {
                        info!(
                            "ISP pipeline securely locked to {} ({:?}) at {}x{}",
                            desc,
                            cfgs.get(0).unwrap().get_pixel_format(),
                            target_width,
                            target_height
                        );
                        success = true;
                        break;
                    }
                }
            }

            if !success {
                return Err("ISP hardware scaler rejected all 8-bit/YUV formats".to_string());
            }

            active_camera
                .configure(&mut cfgs)
                .map_err(|e| format!("Failed to configure camera: {}", e))?;

            let final_cfg = cfgs.get(0).unwrap();
            let selected_format = final_cfg.get_pixel_format();
            let size = final_cfg.get_size();
            let stride = final_cfg.get_stride();
            let stream = final_cfg
                .stream()
                .expect("Failed to get stream from config");
            drop(final_cfg);

            let format_str = format!("{:?}", selected_format);
            let bit_depth = 8;
            let is_color_val = is_color(&format_str);
            let is_pisp_val = false;
            let is_packed_val = false;

            info!("Final ISP configuration:");
            info!("  Format: {:?}", selected_format);
            info!("  Bit Depth: {}", bit_depth);
            info!("  Size: {}x{}", size.width, size.height);
            info!("  Is Color: {}", is_color_val);
            info!("  Is PiSP: {}", is_pisp_val);
            info!("  Is Packed: {}", is_packed_val);

            return Ok((
                LibCameraConfiguration {
                    model,
                    format: selected_format,
                    bit_depth,
                    is_color: is_color_val,
                    is_pisp: is_pisp_val,
                    is_packed: is_packed_val,
                    width: size.width,
                    height: size.height,
                    stride,
                },
                stream,
            ));
        }

        let mut config = active_camera
            .generate_configuration(&[role])
            .ok_or_else(|| "Failed to generate configuration".to_string())?;

        let mut stream_config = config
            .get_mut(0)
            .ok_or_else(|| "Failed to get stream configuration".to_string())?;
        let original_format = stream_config.get_pixel_format();
        let mut best_format = original_format;

        let original_format_str = format!("{:?}", original_format);
        let mut best_bit_depth = parse_bit_depth(&original_format_str);

        // Iterate through the supported formats
        let formats = stream_config.formats().pixel_formats();
        for i in 0..formats.len() {
            let format = formats.get(i).expect("Failed to get format");
            let format_str = format!("{:?}", format);
            info!("Supported format: {}", format_str);

            let bit_depth = parse_bit_depth(&format_str);
            if bit_depth < best_bit_depth {
                best_bit_depth = bit_depth;
                best_format = format;
            }
        }

        // Validate the format
        stream_config.set_pixel_format(best_format);
        drop(stream_config);

        let status = config.validate();

        let mut stream_config = config
            .get_mut(0)
            .ok_or_else(|| "Failed to get stream configuration".to_string())?;

        let selected_format = if status.is_invalid() {
            stream_config.set_pixel_format(original_format);
            original_format
        } else {
            stream_config.get_pixel_format()
        };

        if status.is_adjusted() && format!("{:?}", selected_format).contains("PISP_COMP") {
            let target_fmt_str = format!("{:?}", best_format);
            let is_12_bit = target_fmt_str.contains("12");
            let bit_char = if is_12_bit { b'2' } else { b'0' };

            let code = if target_fmt_str.contains("RGGB") {
                Some(Self::fourcc(b'R', b'G', b'1', bit_char))
            } else if target_fmt_str.contains("BGGR") {
                Some(Self::fourcc(b'B', b'G', b'1', bit_char))
            } else if target_fmt_str.contains("GBRG") {
                Some(Self::fourcc(b'G', b'B', b'1', bit_char))
            } else if target_fmt_str.contains("GRBG") {
                Some(Self::fourcc(b'g', b'r', b'1', bit_char))
            } else if target_fmt_str.contains("GREY")
                || target_fmt_str.contains("Y10")
                || target_fmt_str.contains("Y12")
            {
                Some(Self::fourcc(b'R', b'1', bit_char, b' '))
            } else {
                None
            };

            if let Some(c) = code {
                let fmt_unpacked = PixelFormat::new(c, 0);
                stream_config.set_pixel_format(fmt_unpacked);
                drop(stream_config);

                let _ = config.validate();
                stream_config = config.get_mut(0).unwrap();
            }
        }

        drop(stream_config);

        // Configure the update configs.
        active_camera
            .configure(&mut config)
            .map_err(|e| format!("Failed to configure camera: {}", e))?;

        let final_cfg = config.get(0).unwrap();
        let selected_format = final_cfg.get_pixel_format();
        let size = final_cfg.get_size();
        let stride = final_cfg.get_stride();
        let stream = final_cfg
            .stream()
            .expect("Failed to get stream from config");
        drop(final_cfg);

        let final_format_str = format!("{:?}", selected_format);
        let final_bit_depth = if final_format_str.contains("16")
            || (final_format_str.starts_with('R')
                && final_format_str.len() >= 4
                && final_format_str.chars().nth(2) == Some('1'))
        {
            16
        } else {
            parse_bit_depth(&final_format_str)
        };

        let is_color_val = is_color(&final_format_str);
        let is_pisp_val = final_format_str.contains("PISP_COMP");
        let is_packed_val =
            final_format_str.ends_with("_CSI2P") || final_format_str.ends_with("_P");

        info!("Final Raw configuration:");
        info!("  Format: {:?}", selected_format);
        info!("  Bit Depth: {}", final_bit_depth);
        info!("  Size: {}x{}", size.width, size.height);
        info!("  Is Color: {}", is_color_val);
        info!("  Is PiSP: {}", is_pisp_val);
        info!("  Is Packed: {}", is_packed_val);

        Ok((
            LibCameraConfiguration {
                model,
                format: selected_format,
                bit_depth: final_bit_depth,
                is_color: is_color_val,
                is_pisp: is_pisp_val,
                is_packed: is_packed_val,
                width: size.width,
                height: size.height,
                stride,
            },
            stream,
        ))
    }

    /// The isolated thread loop handling the entire libcamera hardware lifecycle.
    fn capture_loop(
        rx: mpsc::Receiver<CaptureCommand>,
        store: Arc<CaptureStore>,
        isp_mode: bool,
        config_store: Arc<Mutex<Option<LibCameraConfiguration>>>,
        max_gain: Arc<AtomicU32>,
    ) {
        // Aggressively suppress libcamera logs via environment variable
        unsafe {
            std::env::set_var("LIBCAMERA_LOG_LEVELS", "*:ERROR");
        }

        // Also set via internal API as backup
        libcamera::logging::log_set_target(libcamera::logging::LoggingTarget::None).ok();

        let mut mgr = CameraManager::new().expect("Failed to create CameraManager");
        mgr.start().expect("Failed to start CameraManager");

        let cameras = mgr.cameras();
        let camera = cameras.get(0).expect("No cameras available");

        // In the newer API, we acquire the ActiveCamera context first
        let mut active_camera = camera.acquire().expect("Failed to acquire camera");

        // --- 1. Camera Setup & Format Selection ---
        let (cam_config, stream) =
            Self::configure_camera(&mut active_camera, StreamRole::Raw, isp_mode)
                .expect("Failed to configure camera");
        {
            let mut lock = config_store.lock().unwrap();
            *lock = Some(cam_config.clone());
        }

        // --- 2. Buffer Allocation & Request Generation ---
        let mut allocator = FrameBufferAllocator::new(&active_camera);

        // `.allocate()` was renamed to `.alloc()`
        let buffers = allocator
            .alloc(&stream)
            .expect("Failed to allocate buffers");

        // Setup request completion channel
        let (req_tx, req_rx) = mpsc::channel();
        active_camera.on_request_completed(move |req| {
            let _ = req_tx.send(req);
        });

        // We build the requests and hold them in a Vec temporarily
        let mut initial_requests = Vec::new();
        for buffer in buffers {
            let mapped_buffer = MemoryMappedFrameBuffer::new(buffer).expect("Failed to map buffer");
            let mut request = active_camera
                .create_request(None)
                .expect("Failed to create request");
            request
                .add_buffer(&stream, mapped_buffer)
                .expect("Failed to add buffer to request");
            initial_requests.push(request);
        }

        // Local state for actively driven exposure and gain
        let sensor_max = f32::from_bits(max_gain.load(Ordering::Relaxed));
        let mut target_gain = sensor_max;
        let mut target_exposure = INITIAL_EXPOSURE_US;

        // --- 3. Initial Controls Application ---
        let mut control_list = ControlList::new();
        Self::apply_controls(&mut control_list, target_gain, target_exposure);

        active_camera
            .start(Some(&control_list))
            .expect("Failed to start camera");

        // Queue all initial requests by consuming our Vec.
        // We pass ownership of the Request directly to the camera.
        for req in initial_requests {
            active_camera
                .queue_request(req)
                .expect("Failed to queue request");
        }

        // --- 4. Main Event/Capture Loop ---
        'capture: loop {
            // Check for software configuration changes
            while let Ok(cmd) = rx.try_recv() {
                match cmd {
                    CaptureCommand::StopCaptureThread => break 'capture,
                    CaptureCommand::SetHcg(enable) => {
                        if !is_hcg_supported(&cam_config.model) {
                            log::info!(
                                "HCG isn't supported on the current camera: {}",
                                cam_config.model
                            );
                            continue;
                        }
                        log::info!("Initiating HCG toggle sequence (Target: {})", enable);

                        // 1. Drain any pending/completed requests into a holding vector
                        let mut held_requests = Vec::new();
                        while let Ok(req) = req_rx.try_recv() {
                            held_requests.push(req);
                        }

                        // 2. Stop the pipeline to halt hardware processing.
                        // Any frames actively exposed or moving through the pipeline will be cancelled
                        // and immediately pushed into `req_rx`
                        log::debug!(
                            "Stopping camera pipeline... (held {} completed requests)",
                            held_requests.len()
                        );
                        if let Err(e) = active_camera.stop() {
                            log::error!("Failed to stop camera for HCG toggle: {}", e);
                        }

                        // 3. Restart the pipeline
                        log::debug!("Restarting camera pipeline...");
                        if let Err(e) = active_camera.start(None) {
                            log::error!("Failed to restart camera for HCG toggle: {}", e);
                        }

                        // 4. Attempt to write I2C while the pipeline clock is running
                        log::debug!("Writing HCG via I2C (started state)...");
                        if let Err(e) = Self::write_i2c_hcg(enable) {
                            log::warn!("HCG write failed: {}", e);
                        } else {
                            log::info!("HCG write succeeded.");
                        }

                        // 5. Rest before we flood the queue with buffers again
                        std::thread::sleep(Duration::from_millis(100));

                        // 6. Grab any cancelled requests that were dumped during the `stop()` call
                        while let Ok(req) = req_rx.try_recv() {
                            held_requests.push(req);
                        }

                        // 7. Re-queue everything with our latest controls
                        log::debug!(
                            "Re-queuing {} requests after HCG toggle...",
                            held_requests.len()
                        );
                        for mut req in held_requests {
                            req.reuse(ReuseFlag::REUSE_BUFFERS);
                            {
                                let req_ctrls = req.controls_mut();
                                Self::apply_controls(req_ctrls, target_gain, target_exposure);
                            }
                            let _ = active_camera.queue_request(req);
                        }
                        log::info!("HCG toggle sequence complete.");
                    }
                    CaptureCommand::SetGain(gain) => {
                        target_gain = gain;
                    }
                    CaptureCommand::SetExposure(exp) => {
                        target_exposure = exp;
                    }
                }
            }

            // Wait for at least one completed frame from libcamera
            if let Ok(completed_request) = req_rx.recv_timeout(Duration::from_millis(50)) {
                let receive_time = std::time::Instant::now();
                let mut latest_req = completed_request;

                // Drain the channel to ensure we drop older backlog frames.
                // Re-queue older frames with the absolute latest controls applied.
                while let Ok(newer_req) = req_rx.try_recv() {
                    latest_req.reuse(ReuseFlag::REUSE_BUFFERS);
                    Self::apply_controls(latest_req.controls_mut(), target_gain, target_exposure);
                    let _ = active_camera.queue_request(latest_req);
                    latest_req = newer_req;
                }

                // GUARD: Skip processing entirely if the hardware aborted this request.
                // Since we now handle cancelled frames explicitly during the HCG toggle,
                // this mostly catches unexpected hardware stutters.
                if latest_req.status() == libcamera::request::RequestStatus::Cancelled {
                    latest_req.reuse(ReuseFlag::REUSE_BUFFERS);
                    Self::apply_controls(latest_req.controls_mut(), target_gain, target_exposure);
                    let _ = active_camera.queue_request(latest_req);
                    continue;
                }

                // --- Process ONLY the newest request ---
                let meta = latest_req.metadata();

                // Extract exposure time (in microseconds), fallback to a sensible default if unavailable
                let exposure_time_us = meta
                    .get::<libcamera::controls::ExposureTime>()
                    .map(|e| e.0)
                    .unwrap_or(16667);

                // Extract hardware sensor timestamp (in nanoseconds since boot)
                let sensor_timestamp_ns = meta
                    .get::<libcamera::controls::SensorTimestamp>()
                    .map(|t| t.0)
                    .unwrap_or(0);

                let capture_sys_time = if sensor_timestamp_ns > 0 {
                    // Calculate pipeline delay using CLOCK_MONOTONIC
                    let mut ts = libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 0,
                    };
                    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
                    let current_mono_ns = ts.tv_sec as i64 * 1_000_000_000 + ts.tv_nsec as i64;

                    let pipeline_delay_ns = (current_mono_ns - sensor_timestamp_ns).max(0) as u64;

                    // SensorTimestamp is typically Start of Frame. Add half exposure to get to the middle.
                    let halfway_offset_ns = (exposure_time_us as u64) * 1000 / 2;

                    // SystemTime now minus the pipeline delivery delay, plus the offset to the middle of exposure
                    SystemTime::now()
                        .checked_sub(Duration::from_nanos(pipeline_delay_ns))
                        .unwrap_or_else(SystemTime::now)
                        + Duration::from_nanos(halfway_offset_ns)
                } else {
                    // Fallback if hardware timestamp is missing
                    let halfway_duration = Duration::from_micros((exposure_time_us as u64) / 2);
                    SystemTime::now()
                        .checked_sub(halfway_duration)
                        .unwrap_or_else(SystemTime::now)
                };

                if let Some(mapped_buffer) =
                    latest_req.buffer::<MemoryMappedFrameBuffer<FrameBuffer>>(&stream)
                {
                    Self::store_as_8bit(
                        &store,
                        &cam_config,
                        &mapped_buffer.data()[0],
                        exposure_time_us,
                        capture_sys_time,
                        receive_time,
                    );
                }

                // Re-queue the newest request, forcing the active target controls onto it.
                latest_req.reuse(ReuseFlag::REUSE_BUFFERS);
                {
                    let req_ctrls = latest_req.controls_mut();
                    Self::apply_controls(req_ctrls, target_gain, target_exposure);
                }

                let _ = active_camera.queue_request(latest_req);
            }
        }

        // Clean up hardware resources on exit
        active_camera.stop().unwrap();
        // `active_camera` automatically drops/releases out of scope here.
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
