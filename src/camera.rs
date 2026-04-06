// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use i2c_linux::{I2c, ReadWrite};
use libcamera::{
    camera::ActiveCamera,
    camera_manager::CameraManager,
    control::ControlList,
    controls::{AeEnable, AnalogueGain, ExposureTime, FrameDurationLimits},
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    request::ReuseFlag,
    stream::StreamRole,
};
use log::info;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::asm;
use std::io::Write;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicI32, AtomicU32, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

// --- Defined Constants ---
const I2C_ADDR: u16 = 0x1A;
const INITIAL_GAIN: f32 = 31.0;
const INITIAL_EXPOSURE_US: i32 = 50_000;
const I2C_BUSES_TO_SCAN: &[u8] = &[0, 1, 4, 6, 10, 11, 12, 13, 14, 15];

// --- Data Structures ---

/// Represents a captured image frame with its timestamp.
pub struct Image {
    pub data: Vec<u8>,
    pub timestamp: SystemTime,
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
    SetHcg(bool),
    StopCaptureThread,
}

/// Represents metadata for the selected camera configuration.
#[derive(Clone)]
pub struct CameraConfiguration {
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

pub struct Camera {
    command_tx: Option<mpsc::Sender<CaptureCommand>>,
    store: Arc<CaptureStore>,
    capture_thread: Option<JoinHandle<()>>,

    // Cached atomic values for lock-free getter access
    current_gain: Arc<AtomicU32>, // f32 stored as u32 bits for atomic operations
    current_exposure: Arc<AtomicI32>,

    isp_mode: bool,
    config: Arc<Mutex<Option<CameraConfiguration>>>,
}

impl Camera {
    /// Instantiates the camera module and sets initial cached state.
    /// Actual hardware setup occurs when `start_capture` is called to keep
    /// the libcamera context localized to the capture thread.
    pub fn new() -> Self {
        let store = Arc::new(CaptureStore {
            image: Mutex::new(None),
            cvar: Condvar::new(),
        });

        let current_gain = Arc::new(AtomicU32::new(INITIAL_GAIN.to_bits()));
        let current_exposure = Arc::new(AtomicI32::new(INITIAL_EXPOSURE_US));

        Self {
            command_tx: None,
            store,
            capture_thread: None,
            current_gain,
            current_exposure,
            isp_mode: false,
            config: Arc::new(Mutex::new(None)),
        }
    }

    pub fn get_config(&self) -> Option<CameraConfiguration> {
        self.config.lock().unwrap().as_ref().cloned()
    }

    /// Set ISP mode for the camera.
    pub fn set_isp_mode(&mut self, enabled: bool) {
        self.isp_mode = enabled;
    }

    /// Kicks off the capture operation on a separate thread.
    /// This thread handles libcamera initialization, format selection, and the capture loop.
    pub fn start_capture(&mut self) -> Result<(), String> {
        if self.capture_thread.is_some() {
            return Ok(()); // Already running
        }

        let (tx, rx) = mpsc::channel();
        self.command_tx = Some(tx);

        let store_clone = Arc::clone(&self.store);
        let isp_mode = self.isp_mode;
        let config_clone = Arc::clone(&self.config);

        self.capture_thread = Some(thread::spawn(move || {
            Self::capture_loop(rx, store_clone, isp_mode, config_clone);
        }));

        Ok(())
    }

    // --- Configuration Getters & Setters ---

    /// Returns the currently set analog gain (lock-free).
    pub fn get_gain(&self) -> f32 {
        f32::from_bits(self.current_gain.load(Ordering::Relaxed))
    }

    /// Sets the analog gain and notifies the capture thread to apply it.
    pub fn set_gain(&self, gain: f32) {
        self.current_gain.store(gain.to_bits(), Ordering::Relaxed);
        self.send_command(CaptureCommand::SetGain(gain));
    }

    /// Returns the currently set exposure in microseconds (lock-free).
    pub fn get_exposure(&self) -> i32 {
        self.current_exposure.load(Ordering::Relaxed)
    }

    /// Sets the exposure in microseconds and notifies the capture thread to apply it.
    pub fn set_exposure(&self, exposure_us: i32) {
        self.current_exposure.store(exposure_us, Ordering::Relaxed);
        self.send_command(CaptureCommand::SetExposure(exposure_us));
    }

    /// Enables High Conversion Gain (HCG) via I2C on address 0x1A.
    pub fn enable_hcg(&self) {
        self.send_command(CaptureCommand::SetHcg(true));
    }

    /// Disables High Conversion Gain (HCG) via I2C on address 0x1A.
    pub fn disable_hcg(&self) {
        self.send_command(CaptureCommand::SetHcg(false));
    }

    // --- Data Retrieval ---

    /// Returns the last capture from the store. Returns None if empty.
    /// Optimization: Returns an Arc to avoid copying the heap buffer.
    pub fn get_last_capture(&self) -> Option<Arc<Image>> {
        let lock = self.store.image.lock().unwrap();
        lock.clone()
    }

    /// Blocks until an image is available, then returns it.
    /// Optimization: Returns an Arc to avoid copying the heap buffer.
    pub fn get_last_capture_sync(&self) -> Arc<Image> {
        let mut lock = self.store.image.lock().unwrap();
        loop {
            if let Some(img) = &*lock {
                return img.clone();
            }
            // Wait to be notified by the capture thread that a new image is ready
            lock = self.store.cvar.wait(lock).unwrap();
        }
    }

    fn store_as_8bit(store: &Arc<CaptureStore>, config: &CameraConfiguration, buf_data: &[u8]) {
        let width = config.width as usize;
        let height = config.height as usize;
        let stride = config.stride as usize;
        let mut image_data = vec![0u8; width * height];

        let is_10_bit = config.bit_depth == 10;
        let is_12_bit = config.bit_depth == 12;
        let is_16_bit = config.bit_depth == 16;

        if is_16_bit {
            for row in 0..height {
                let buf_row_start = row * stride;
                let buf_row_end = buf_row_start + width * 2;
                let pix_row_start = row * width;
                let pix_row_end = pix_row_start + width;

                for (src_chunk, dst) in buf_data[buf_row_start..buf_row_end]
                    .chunks_exact(2)
                    .zip(image_data[pix_row_start..pix_row_end].iter_mut())
                {
                    // Because the Pi 5 MSB-justifies, and ARM is little-endian,
                    // src_chunk[1] contains exactly the top 8 bits we need!
                    *dst = src_chunk[1];
                }
            }
        } else if !is_10_bit && !is_12_bit {
            if stride == width {
                image_data[..width * height].copy_from_slice(&buf_data[..width * height]);
            } else {
                for row in 0..height {
                    let src_start = row * stride;
                    let dst_start = row * width;
                    image_data[dst_start..dst_start + width]
                        .copy_from_slice(&buf_data[src_start..src_start + width]);
                }
            }
        } else if config.is_packed {
            if is_10_bit {
                let valid_buf_len = width * 5 / 4;
                for row in 0..height {
                    let buf_row_full = &buf_data[row * stride..(row + 1) * stride];
                    let pix_row_full = &mut image_data[row * width..(row + 1) * width];
                    let buf_slice = &buf_row_full[..valid_buf_len];
                    let pix_slice = &mut pix_row_full[..width];

                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        let mask0 = vld1q_u8(
                            [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18].as_ptr(),
                        );
                        let mask1 = vld1q_u8(
                            [4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22].as_ptr(),
                        );
                        let mask2 = vld1q_u8(
                            [8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26].as_ptr(),
                        );
                        let mask3 = vld1q_u8(
                            [
                                12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30,
                            ]
                            .as_ptr(),
                        );

                        let mut i = 0;
                        let mut j = 0;

                        while i + 80 <= buf_slice.len() && j + 64 <= pix_slice.len() {
                            let in_ptr = buf_slice.as_ptr().add(i);
                            let out_ptr = pix_slice.as_mut_ptr().add(j);

                            asm!("prfm pldl1keep, [{0}, #256]", in(reg) in_ptr, options(readonly, preserves_flags, nostack));

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
                }
            } else if is_12_bit {
                let valid_buf_len = width * 3 / 2;
                for row in 0..height {
                    let buf_row_full = &buf_data[row * stride..(row + 1) * stride];
                    let pix_row_full = &mut image_data[row * width..(row + 1) * width];
                    let buf_slice = &buf_row_full[..valid_buf_len];
                    let pix_slice = &mut pix_row_full[..width];

                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        let mask0 = vld1q_u8(
                            [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22].as_ptr(),
                        );
                        let mask1 = vld1q_u8(
                            [8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30].as_ptr(),
                        );

                        let mut i = 0;
                        let mut j = 0;

                        while i + 48 <= buf_slice.len() && j + 32 <= pix_slice.len() {
                            let in_ptr = buf_slice.as_ptr().add(i);
                            let out_ptr = pix_slice.as_mut_ptr().add(j);

                            asm!("prfm pldl1keep, [{0}, #256]", in(reg) in_ptr, options(readonly, preserves_flags, nostack));

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

                        for (buf_chunk, pix_pair) in buf_slice[i..]
                            .chunks_exact(3)
                            .zip(pix_slice[j..].chunks_exact_mut(2))
                        {
                            pix_pair[0] = buf_chunk[0];
                            pix_pair[1] = buf_chunk[1];
                        }
                    }

                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for (buf_chunk, pix_pair) in
                            buf_slice.chunks_exact(3).zip(pix_slice.chunks_exact_mut(2))
                        {
                            pix_pair[0] = buf_chunk[0];
                            pix_pair[1] = buf_chunk[1];
                        }
                    }
                }
            }
        }

        let mut lock = store.image.lock().unwrap();
        *lock = Some(Arc::new(Image {
            data: image_data,
            timestamp: SystemTime::now(),
        }));
        store.cvar.notify_all();
    }

    // --- Private Helpers ---

    /// Queues a command to the capture thread.
    fn send_command(&self, cmd: CaptureCommand) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(cmd); // Ignore send errors if thread has died
        }
    }

    const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
        (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
    }

    /// Scans the defined I2C buses to find the camera at address 0x1A.
    fn detect_i2c_bus() -> Option<u8> {
        for &bus in I2C_BUSES_TO_SCAN {
            let path = format!("/dev/i2c-{}", bus);
            if let Ok(mut i2c) = I2c::from_path(&path) {
                // Force mode bypasses driver reservation conflicts (true = I2C_SLAVE_FORCE)
                if i2c.smbus_set_slave_address(I2C_ADDR, true).is_ok() {
                    // Try a quick write to ping the device; if it ACKs, it's our bus.
                    if i2c.smbus_write_quick(ReadWrite::Write).is_ok() {
                        return Some(bus);
                    }
                }
            }
        }
        None
    }

    /// Writes raw bytes to the detected I2C bus to enable/disable HCG.
    fn write_i2c_hcg(enable: bool) -> std::io::Result<()> {
        let bus = Self::detect_i2c_bus().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "Camera I2C bus not found")
        })?;

        let path = format!("/dev/i2c-{}", bus);
        let mut i2c = I2c::from_path(&path)?;

        i2c.smbus_set_slave_address(I2C_ADDR, true)?;

        // Specific payloads required for 0x1A HCG toggle
        let payload: [u8; 3] = if enable {
            [0x30, 0x09, 0x11]
        } else {
            [0x30, 0x09, 0x01]
        };

        // Extract the underlying File to write the raw bytes directly to the register
        let mut file = i2c.into_inner();
        file.write_all(&payload)?;

        Ok(())
    }

    fn configure_camera(
        active_camera: &mut ActiveCamera<'_>,
        role: StreamRole,
        isp_mode: bool,
    ) -> Result<(CameraConfiguration, libcamera::stream::Stream), String> {
        if isp_mode {
            info!("Using ISP mode");
            let mut cfgs = active_camera
                .generate_configuration(&[StreamRole::VideoRecording])
                .ok_or_else(|| "Failed to generate configuration".to_string())?;

            let stream_cfg = cfgs
                .get(0)
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

            // Sniff native sensor mode
            let raw_cfg = active_camera
                .generate_configuration(&[StreamRole::Raw])
                .ok_or_else(|| "Failed to generate raw configuration".to_string())?;
            let raw_stream = raw_cfg.get(0).unwrap();
            let raw_size = raw_stream.get_size();
            let raw_fmt_str = format!("{:?}", raw_stream.get_pixel_format());

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

            let mut sensor_config = libcamera::camera::SensorConfiguration::new();
            sensor_config.set_output_size(raw_size.width, raw_size.height);
            sensor_config.set_bit_depth(bit_depth);
            sensor_config.set_analog_crop(libcamera::geometry::Rectangle {
                x: 0,
                y: 0,
                width: raw_size.width,
                height: raw_size.height,
            });
            sensor_config.set_binning(1, 1);
            sensor_config.set_skipping(1, 1, 1, 1);

            cfgs.set_sensor_configuration(sensor_config);

            let downsample_factor = if (raw_size.width * raw_size.height) < 3_000_000 {
                2
            } else {
                4
            };
            let target_width = raw_size.width / downsample_factor;
            let target_height = raw_size.height / downsample_factor;

            let mut success = false;
            let format_candidates = [("Mono 8-bit", format_r8), ("YUV420 Planar", format_yuv)];

            for (desc, format_opt) in format_candidates.iter() {
                if let Some(fmt) = format_opt {
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
                CameraConfiguration {
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
            CameraConfiguration {
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
        config_store: Arc<Mutex<Option<CameraConfiguration>>>,
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

        // --- 3. Initial Controls Application ---
        let mut control_list = ControlList::new();
        // Controls are now strongly typed tuple structs
        let _ = control_list.set(AeEnable(false));
        let _ = control_list.set(AnalogueGain(INITIAL_GAIN));
        let _ = control_list.set(ExposureTime(INITIAL_EXPOSURE_US));
        let _ = control_list.set(FrameDurationLimits([16667, 1000000]));

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
        loop {
            // Check for software configuration changes
            if let Ok(cmd) = rx.try_recv() {
                match cmd {
                    CaptureCommand::StopCaptureThread => break,
                    CaptureCommand::SetHcg(enable) => {
                        let _ = Self::write_i2c_hcg(enable);
                    }
                    CaptureCommand::SetGain(gain) => {
                        active_camera.stop().unwrap();
                        let mut new_ctrls = ControlList::new();
                        let _ = new_ctrls.set(AnalogueGain(gain));
                        active_camera.start(Some(&new_ctrls)).unwrap();

                        // When stopped, pending requests are returned to the channel as cancelled.
                        // Drain them, refresh them, and pass them back to the active camera.
                        while let Ok(mut req) = req_rx.try_recv() {
                            req.reuse(ReuseFlag::REUSE_BUFFERS);
                            active_camera.queue_request(req).unwrap();
                        }
                    }
                    CaptureCommand::SetExposure(exp) => {
                        active_camera.stop().unwrap();
                        let mut new_ctrls = ControlList::new();
                        let _ = new_ctrls.set(ExposureTime(exp));
                        active_camera.start(Some(&new_ctrls)).unwrap();

                        while let Ok(mut req) = req_rx.try_recv() {
                            req.reuse(ReuseFlag::REUSE_BUFFERS);
                            active_camera.queue_request(req).unwrap();
                        }
                    }
                }
            }

            // Wait for at least one completed frame from libcamera
            if let Ok(completed_request) = req_rx.recv_timeout(Duration::from_millis(50)) {
                let mut latest_req = completed_request;

                // Drain the channel to ensure we drop older backlog frames.
                // We must re-queue them immediately without processing to prevent libcamera from starving.
                while let Ok(newer_req) = req_rx.try_recv() {
                    latest_req.reuse(ReuseFlag::REUSE_BUFFERS);
                    let _ = active_camera.queue_request(latest_req);
                    latest_req = newer_req;
                }

                // --- Process ONLY the newest request ---
                if let Some(mapped_buffer) =
                    latest_req.buffer::<MemoryMappedFrameBuffer<FrameBuffer>>(&stream)
                {
                    // Extract the bytes from the first plane and store as 8-bit.
                    Self::store_as_8bit(&store, &cam_config, &mapped_buffer.data()[0]);
                }

                // Re-queue the newest request (passing ownership) to keep the hardware pipeline flowing
                latest_req.reuse(ReuseFlag::REUSE_BUFFERS);
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
