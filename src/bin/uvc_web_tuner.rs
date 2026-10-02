use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime};
use tetra3::{
    fast_extractor::{FastBgSubMode, FastDownsample, FastExtractOptions, FastExtractor},
    solver::{SolveOptions, SolveStatus},
    tetra3::Tetra3,
};
use tokio::sync::{Semaphore, mpsc as tmpsc};
use tokio_stream::wrappers::ReceiverStream;

// --- Data Structures ---

#[derive(Clone, Serialize, Default)]
struct AppState {
    formats: Vec<FormatInfo>,
    current_format_id: usize,
    ranges: UvcRanges,
    current_values: UvcValues,
    is_benchmarking: bool,
    benchmark_stats: BenchmarkStats,
    camera_error: Option<String>,
}

#[derive(Clone, Serialize)]
struct FormatInfo {
    id: usize,
    format_type: String,
    width: u32,
    height: u32,
    fps: u32,
}

#[derive(Clone, Serialize)]
struct UvcValues {
    ae_mode: u8,
    gain: i32,
    contrast: i32,
    saturation: i32,
    brightness: i32,
    white_balance_temperature: i32,
    sharpness: i32,
    gamma: i32,
    backlight_compensation: i32,
}

impl Default for UvcValues {
    fn default() -> Self {
        Self {
            ae_mode: 1,
            gain: 0,
            contrast: 0,
            saturation: 0,
            brightness: 0,
            white_balance_temperature: 0,
            sharpness: 0,
            gamma: 0,
            backlight_compensation: 0,
        }
    }
}

#[derive(Clone, Serialize, Default)]
struct UvcRanges {
    gain: Option<RangeDef>,
    contrast: Option<RangeDef>,
    saturation: Option<RangeDef>,
    brightness: Option<RangeDef>,
    white_balance_temperature: Option<RangeDef>,
    sharpness: Option<RangeDef>,
    gamma: Option<RangeDef>,
    backlight_compensation: Option<RangeDef>,
}

#[derive(Clone, Serialize)]
struct RangeDef {
    min: i32,
    max: i32,
    default: i32,
    step: i32,
}

impl From<uvc::Range<i32>> for RangeDef {
    fn from(r: uvc::Range<i32>) -> Self {
        Self {
            min: r.min as i32,
            max: r.max as i32,
            default: r.default as i32,
            step: r.step as i32,
        }
    }
}

#[derive(Clone, Serialize, Default)]
struct BenchmarkStats {
    total: usize,
    solves: usize,
    solve_rate: f64,
    avg_centroids: f64,
}

struct SharedImageStore {
    image: Mutex<Option<Arc<olive_engine::Image>>>,
    cvar: Condvar,
}

struct ServerState {
    cam_tx: std::sync::mpsc::Sender<CamCommand>,
    store: Arc<SharedImageStore>,
    app_state: Arc<Mutex<AppState>>,
    semaphore: Arc<Semaphore>,
}

enum CamCommand {
    SetFormat(usize),
    SetControl(String, i32),
}

#[derive(Deserialize)]
struct ControlUpdate {
    control: String,
    value: i32,
}

// --- Thread Functions ---

fn get_range(devh: &uvc::DeviceHandle, ctrl: &str) -> Option<uvc::Range<i32>> {
    match ctrl {
        "gain" => devh.gain_range().ok().map(|r| uvc::Range {
            min: r.min as i32,
            max: r.max as i32,
            step: r.step as i32,
            default: r.default as i32,
        }),
        "contrast" => devh.contrast_range().ok().map(|r| uvc::Range {
            min: r.min as i32,
            max: r.max as i32,
            step: r.step as i32,
            default: r.default as i32,
        }),
        "saturation" => devh.saturation_range().ok().map(|r| uvc::Range {
            min: r.min as i32,
            max: r.max as i32,
            step: r.step as i32,
            default: r.default as i32,
        }),
        "brightness" => devh.brightness_range().ok().map(|r| uvc::Range {
            min: r.min as i32,
            max: r.max as i32,
            step: r.step as i32,
            default: r.default as i32,
        }),
        "white_balance_temperature" => {
            devh.white_balance_temperature_range()
                .ok()
                .map(|r| uvc::Range {
                    min: r.min as i32,
                    max: r.max as i32,
                    step: r.step as i32,
                    default: r.default as i32,
                })
        }
        "sharpness" => devh.sharpness_range().ok().map(|r| uvc::Range {
            min: r.min as i32,
            max: r.max as i32,
            step: r.step as i32,
            default: r.default as i32,
        }),
        "gamma" => devh.gamma_range().ok().map(|r| uvc::Range {
            min: r.min as i32,
            max: r.max as i32,
            step: r.step as i32,
            default: r.default as i32,
        }),
        "backlight_compensation" => devh
            .backlight_compensation_range()
            .ok()
            .map(|r| uvc::Range {
                min: r.min as i32,
                max: r.max as i32,
                step: r.step as i32,
                default: r.default as i32,
            }),
        _ => None,
    }
}

fn run_camera(
    cmd_rx: std::sync::mpsc::Receiver<CamCommand>,
    state_mutex: Arc<Mutex<AppState>>,
    store: Arc<SharedImageStore>,
) {
    let ctx = uvc::Context::new().expect("Failed to create UVC context");
    let dev = ctx
        .find_device(None, None, None)
        .expect("Failed to find UVC device");
    let devh = dev.open().expect("Failed to open UVC device");

    let mut formats = Vec::new();
    for i in devh.supported_formats() {
        for j in i.supported_formats() {
            for k in j.intervals() {
                let format = uvc::StreamFormat {
                    width: u32::from(j.width()),
                    height: u32::from(j.height()),
                    fps: 10_000_000 / *k,
                    format: match j.subtype() {
                        uvc::DescriptionSubtype::FormatMJPEG
                        | uvc::DescriptionSubtype::FrameMJPEG => uvc::FrameFormat::MJPEG,
                        uvc::DescriptionSubtype::FormatUncompressed
                        | uvc::DescriptionSubtype::FrameUncompressed => {
                            uvc::FrameFormat::Uncompressed
                        }
                        _ => uvc::FrameFormat::Any,
                    },
                };
                if format.format != uvc::FrameFormat::Any {
                    formats.push(format);
                }
            }
        }
    }

    let mut ranges = UvcRanges::default();
    ranges.gain = get_range(&devh, "gain").map(Into::into);
    ranges.contrast = get_range(&devh, "contrast").map(Into::into);
    ranges.saturation = get_range(&devh, "saturation").map(Into::into);
    ranges.brightness = get_range(&devh, "brightness").map(Into::into);
    ranges.white_balance_temperature =
        get_range(&devh, "white_balance_temperature").map(Into::into);
    ranges.sharpness = get_range(&devh, "sharpness").map(Into::into);
    ranges.gamma = get_range(&devh, "gamma").map(Into::into);
    ranges.backlight_compensation = get_range(&devh, "backlight_compensation").map(Into::into);

    let mut current_values = UvcValues {
        ae_mode: 1,
        gain: ranges.gain.as_ref().map(|r| r.default).unwrap_or(0),
        contrast: ranges.contrast.as_ref().map(|r| r.default).unwrap_or(0),
        saturation: ranges.saturation.as_ref().map(|r| r.default).unwrap_or(0),
        brightness: ranges.brightness.as_ref().map(|r| r.default).unwrap_or(0),
        white_balance_temperature: ranges
            .white_balance_temperature
            .as_ref()
            .map(|r| r.default)
            .unwrap_or(0),
        sharpness: ranges.sharpness.as_ref().map(|r| r.default).unwrap_or(0),
        gamma: ranges.gamma.as_ref().map(|r| r.default).unwrap_or(0),
        backlight_compensation: ranges
            .backlight_compensation
            .as_ref()
            .map(|r| r.default)
            .unwrap_or(0),
    };

    let supported_ae_modes = devh.supported_ae_modes().unwrap_or(Vec::new());
    if supported_ae_modes.contains(&uvc::AutoExposureMode::ShutterPriority)
        && !supported_ae_modes.contains(&uvc::AutoExposureMode::Manual)
    {
        let _ = devh.set_ae_mode(uvc::AutoExposureMode::ShutterPriority);
    } else {
        let _ = devh.set_ae_mode(uvc::AutoExposureMode::Manual);
    }

    let format_infos: Vec<_> = formats
        .iter()
        .enumerate()
        .map(|(id, f)| FormatInfo {
            id,
            format_type: if f.format == uvc::FrameFormat::MJPEG {
                "MJPEG".to_string()
            } else {
                "Uncompressed".to_string()
            },
            width: f.width,
            height: f.height,
            fps: f.fps,
        })
        .collect();

    let mut current_format_id = 0;
    let target_fps = 10;
    if let Some(preferred_format) = devh.get_preferred_format(move |f1, f2| {
        let f1_is_uncomp = f1.format == uvc::FrameFormat::Uncompressed;
        let f2_is_uncomp = f2.format == uvc::FrameFormat::Uncompressed;
        if f1_is_uncomp != f2_is_uncomp {
            return if f1_is_uncomp { f1 } else { f2 };
        }
        let f1_fps_ok = f1.fps >= target_fps;
        let f2_fps_ok = f2.fps >= target_fps;
        if f1_fps_ok != f2_fps_ok {
            return if f1_fps_ok { f1 } else { f2 };
        }
        let f1_res = f1.width * f1.height;
        let f2_res = f2.width * f2.height;
        if f1_res != f2_res {
            return if f1_res > f2_res { f1 } else { f2 };
        }
        if f1.fps > f2.fps { f1 } else { f2 }
    }) {
        current_format_id = formats
            .iter()
            .position(|f| {
                f.width == preferred_format.width
                    && f.height == preferred_format.height
                    && f.fps == preferred_format.fps
                    && f.format == preferred_format.format
            })
            .unwrap_or(0);
    }

    {
        let mut st = state_mutex.lock().unwrap();
        st.formats = format_infos.clone();
        st.current_format_id = current_format_id;
        st.ranges = ranges.clone();
        st.current_values = current_values.clone();
    }

    loop {
        let stream_format = formats[current_format_id].clone();
        println!(
            "Attempting to open stream with format ID {}: {:?} {}x{} @ {} fps",
            current_format_id,
            stream_format.format,
            stream_format.width,
            stream_format.height,
            stream_format.fps
        );
        let mut stream_attempts = 0;
        let mut format_changed = false;

        loop {
            stream_attempts += 1;
            let store_clone = store.clone();

            let mut h = match devh.get_stream_handle_with_format(stream_format) {
                Ok(h) => h,
                Err(e) => {
                    if stream_attempts >= 3 {
                        let err_msg =
                            format!("Failed to get stream handle after 3 attempts: {:?}", e);
                        eprintln!("{}", err_msg);
                        state_mutex.lock().unwrap().camera_error = Some(err_msg);
                        break;
                    }
                    eprintln!(
                        "Attempt {} failed to get stream handle: {:?}. Retrying in 1s...",
                        stream_attempts, e
                    );
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };

            let start_stream_res = h.start_stream(
                move |frame, _| {
                    let width = frame.width() as usize;
                    let height = frame.height() as usize;
                    let format = frame.format();
                    let data = frame.to_bytes();

                    let mut actual_width = width as u32;
                    let mut actual_height = height as u32;

                    let image_data = if format == uvc::FrameFormat::MJPEG {
                        let mut decompressor = match turbojpeg::Decompressor::new() {
                            Ok(d) => d,
                            Err(_) => return,
                        };
                        let header = match decompressor.read_header(data) {
                            Ok(h) => h,
                            Err(_) => return,
                        };
                        actual_width = header.width as u32;
                        actual_height = header.height as u32;

                        let mut pixels = vec![0u8; header.width * header.height];
                        let image = turbojpeg::Image {
                            pixels: pixels.as_mut_slice(),
                            width: header.width,
                            pitch: header.width,
                            height: header.height,
                            format: turbojpeg::PixelFormat::GRAY,
                        };
                        match decompressor.decompress(data, image) {
                            Ok(_) => pixels,
                            Err(_) => return,
                        }
                    } else {
                        let mut img = Vec::with_capacity(width * height);
                        unsafe {
                            img.set_len(width * height);
                            let mut j = 0;
                            for i in (0..data.len()).step_by(2) {
                                if j < width * height {
                                    *img.get_unchecked_mut(j) = *data.get_unchecked(i);
                                    j += 1;
                                }
                            }
                        }
                        img
                    };

                    let mut lock = store_clone.image.lock().unwrap();
                    *lock = Some(Arc::new(olive_engine::Image {
                        data: image_data,
                        width: actual_width,
                        height: actual_height,
                        timestamp: SystemTime::now(),
                        exposure_time_us: 0,
                        gain: 1.0,
                        post_process_time: Duration::from_millis(0),
                        frame_interval_us: 0,
                        metadata_valid: true,
                    }));
                    store_clone.cvar.notify_all();
                },
                (),
            );

            let _stream = match start_stream_res {
                Ok(s) => s,
                Err(e) => {
                    if stream_attempts >= 3 {
                        let err_msg = format!("Failed to start stream after 3 attempts: {:?}", e);
                        eprintln!("{}", err_msg);
                        state_mutex.lock().unwrap().camera_error = Some(err_msg);
                        break;
                    }
                    eprintln!(
                        "Attempt {} failed to start stream: {:?}. Retrying in 1s...",
                        stream_attempts, e
                    );
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };

            {
                let mut st = state_mutex.lock().unwrap();
                st.camera_error = None;
            }
            println!("Successfully opened stream format ID {}", current_format_id);

            // Apply controls only AFTER successfully getting the stream handle
            let ae_mode = match current_values.ae_mode {
                1 => uvc::AutoExposureMode::Manual,
                2 => uvc::AutoExposureMode::Auto,
                4 => uvc::AutoExposureMode::ShutterPriority,
                8 => uvc::AutoExposureMode::AperturePriority,
                _ => uvc::AutoExposureMode::Manual,
            };
            let _ = devh.set_ae_mode(ae_mode);
            let _ = devh.set_white_balance_temperature_auto(false);
            let _ = devh.set_gain(current_values.gain as u16);
            let _ = devh.set_contrast(current_values.contrast.try_into().unwrap_or(0));
            let _ = devh.set_saturation(current_values.saturation.try_into().unwrap_or(0));
            let _ = devh.set_brightness(current_values.brightness.try_into().unwrap_or(0));
            let _ = devh.set_white_balance_temperature(
                current_values
                    .white_balance_temperature
                    .try_into()
                    .unwrap_or(0),
            );
            let _ = devh.set_sharpness(current_values.sharpness.try_into().unwrap_or(0));
            let _ = devh.set_gamma(current_values.gamma.try_into().unwrap_or(0));
            let _ = devh.set_backlight_compensation(
                current_values
                    .backlight_compensation
                    .try_into()
                    .unwrap_or(0),
            );
            let _ = devh.set_exposure_abs(10000 / stream_format.fps);

            loop {
                match cmd_rx.recv() {
                    Ok(CamCommand::SetFormat(id)) => {
                        if id < formats.len() {
                            current_format_id = id;
                            {
                                let mut st = state_mutex.lock().unwrap();
                                st.current_format_id = id;
                                st.camera_error = None;
                            }
                            format_changed = true;
                            break;
                        }
                    }
                    Ok(CamCommand::SetControl(ctrl, val)) => {
                        let mut changed = true;
                        match ctrl.as_str() {
                            "gain" => current_values.gain = val,
                            "contrast" => current_values.contrast = val,
                            "saturation" => current_values.saturation = val,
                            "brightness" => current_values.brightness = val,
                            "white_balance_temperature" => {
                                current_values.white_balance_temperature = val
                            }
                            "sharpness" => current_values.sharpness = val,
                            "gamma" => current_values.gamma = val,
                            "backlight_compensation" => current_values.backlight_compensation = val,
                            _ => changed = false,
                        }
                        if changed {
                            let _ = devh.set_ae_mode(ae_mode);
                            let _ = devh.set_white_balance_temperature_auto(false);
                            let _ = devh.set_gain(current_values.gain as u16);
                            let _ =
                                devh.set_contrast(current_values.contrast.try_into().unwrap_or(0));
                            let _ = devh
                                .set_saturation(current_values.saturation.try_into().unwrap_or(0));
                            let _ = devh
                                .set_brightness(current_values.brightness.try_into().unwrap_or(0));
                            let _ = devh.set_white_balance_temperature(
                                current_values
                                    .white_balance_temperature
                                    .try_into()
                                    .unwrap_or(0),
                            );
                            let _ = devh
                                .set_sharpness(current_values.sharpness.try_into().unwrap_or(0));
                            let _ = devh.set_gamma(current_values.gamma.try_into().unwrap_or(0));
                            let _ = devh.set_backlight_compensation(
                                current_values
                                    .backlight_compensation
                                    .try_into()
                                    .unwrap_or(0),
                            );
                            let _ = devh.set_exposure_abs(10000 / stream_format.fps);

                            state_mutex.lock().unwrap().current_values = current_values.clone();
                        }
                    }
                    Err(_) => return,
                }
            }

            if format_changed {
                break;
            }
        }

        if !format_changed {
            loop {
                match cmd_rx.recv() {
                    Ok(CamCommand::SetFormat(id)) => {
                        if id < formats.len() {
                            current_format_id = id;
                            {
                                let mut st = state_mutex.lock().unwrap();
                                st.current_format_id = id;
                                st.camera_error = None;
                            }
                            break;
                        }
                    }
                    Ok(_) => {} // Ignore controls when stream is inactive
                    Err(_) => return,
                }
            }
        }
    }
}

fn run_benchmark(state_mutex: Arc<Mutex<AppState>>, store: Arc<SharedImageStore>) {
    let db_path = format!(
        "{}/data/default_database.npz",
        std::env::var("HOME").unwrap_or_default()
    );
    let mut solver = Tetra3::new(&db_path);
    let mut extractor: Option<FastExtractor> = None;
    let mut ext_width = 0;

    let mut last_ts = None;
    let mut recent_centroids = std::collections::VecDeque::new();

    loop {
        let is_active = state_mutex.lock().unwrap().is_benchmarking;
        if !is_active {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        let img = {
            let mut lock = store.image.lock().unwrap();
            loop {
                let result = store
                    .cvar
                    .wait_timeout(lock, Duration::from_millis(200))
                    .unwrap();
                lock = result.0;

                if !state_mutex.lock().unwrap().is_benchmarking {
                    break None;
                }

                if let Some(i) = &*lock {
                    if Some(i.timestamp) != last_ts {
                        break Some(i.clone());
                    }
                }
            }
        };

        if let Some(img_arc) = img {
            last_ts = Some(img_arc.timestamp);
            let img_data = &img_arc.data;

            let (width, height) = {
                let st = state_mutex.lock().unwrap();
                let f = &st.formats[st.current_format_id];
                (f.width as usize, f.height as usize)
            };

            let array_img = match Array2::from_shape_vec((height, width), img_data.clone()) {
                Ok(arr) => arr,
                Err(_) => continue,
            };

            if extractor.is_none() || ext_width != width {
                ext_width = width;
                let downsample = if width * height > 1_000_000 {
                    FastDownsample::X2
                } else {
                    FastDownsample::None
                };

                let options = FastExtractOptions {
                    sigma: 3.5,
                    downsample,
                    bg_sub_mode: Some(FastBgSubMode::LineMedian),
                    binary_open: false,
                    min_area: Some(1),
                    max_area: Some(20),
                    ..FastExtractOptions::default()
                };
                extractor = Some(FastExtractor::new(width, height, options));
            }

            let ext = extractor.as_mut().unwrap();
            let result = ext.extract(&array_img);

            let num_centroids = result.centroids.len();
            recent_centroids.push_back(num_centroids);
            if recent_centroids.len() > 20 {
                recent_centroids.pop_front();
            }

            let mut centroids_arr = Array2::<f64>::zeros((num_centroids, 2));
            for (i, c) in result.centroids.iter().enumerate() {
                centroids_arr[[i, 0]] = c.y;
                centroids_arr[[i, 1]] = c.x;
            }

            let size = (height as f64, width as f64);
            let mut solved = false;
            if num_centroids >= 4 {
                if let Ok(solution) =
                    solver.solve_from_centroids(&centroids_arr, size, SolveOptions::default())
                {
                    if matches!(solution.status, SolveStatus::MatchFound) {
                        solved = true;
                    }
                }
            }

            {
                let mut st = state_mutex.lock().unwrap();
                st.benchmark_stats.total += 1;
                if solved {
                    st.benchmark_stats.solves += 1;
                }
                st.benchmark_stats.solve_rate =
                    (st.benchmark_stats.solves as f64 / st.benchmark_stats.total as f64) * 100.0;
                st.benchmark_stats.avg_centroids =
                    recent_centroids.iter().sum::<usize>() as f64 / recent_centroids.len() as f64;
            }
        }
    }
}

// --- Web Server Functions ---

const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
<title>UVC Web Tuner</title>
<style>
  body { font-family: monospace; background: #000; color: #f00; margin: 0; display: flex; height: 100vh; flex-direction: column; overflow: hidden; }
  a, button, select { color: #f00; background: #110000; border: 1px solid #f00; cursor: pointer; padding: 10px; font-family: inherit; font-size: 1em; }
  a:hover, button:hover { background: #330000; }
  input[type=range] { accent-color: #f00; width: 100%; }
  .view { display: none; flex-grow: 1; padding: 20px; box-sizing: border-box; overflow-y: auto; }
  .active { display: flex !important; }
  #main-view { display: flex; flex-direction: row; height: 100%; width: 100%; overflow: hidden; }
  #controls { width: 350px; padding: 20px; border-right: 1px solid #f00; display: flex; flex-direction: column; box-sizing: border-box; overflow-y: auto; flex-shrink: 0; }
  #viewer { flex-grow: 1; overflow: hidden; position: relative; display: flex; justify-content: center; align-items: center; background: #000; cursor: grab; }
  #viewer:active { cursor: grabbing; }
  #cam-image { transform-origin: center; transition: transform 0.1s ease-out; pointer-events: none; max-width: 100%; max-height: 100%; image-rendering: pixelated; display: none; }
  .control-group { margin-top: 15px; }
  .control-row { display: flex; align-items: center; gap: 10px; }
  label { display: block; font-size: 0.9em; color: #f00; margin-bottom: 5px; }
  .value { float: right; color: #f00; }
  h1, h3 { margin-top: 0; }
  button:disabled, input:disabled, select:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
  <div id="main-view" class="view active">
    <div id="controls">
      <h3>UVC Tuner</h3>
      
      <div class="control-group">
        <button id="camera-btn" onclick="toggleCamera()">Show Live Camera</button>
      </div>

      <div class="control-group" style="border: 1px solid #f00; padding: 10px; margin-top: 20px;">
        <button id="benchmark-btn" onclick="toggleBenchmark()">Start Benchmark</button>
        <div id="benchmark-stats" style="display:none; margin-top: 10px;">
          <p>Solves: <span id="bench-solves">0</span> / <span id="bench-total">0</span></p>
          <p>Solve Rate: <span id="bench-rate">0.00</span>%</p>
          <p>Avg Centroids: <span id="bench-centroids">0.0</span></p>
        </div>
      </div>

      <div class="control-group">
        <label>Format</label>
        <select id="format" onchange="setControl('format', parseInt(this.value))"></select>
      </div>

      <div id="sliders"></div>

    </div>
    <div id="viewer">
      <div id="stream-error-msg" style="display: none; color: red; font-size: 2em; position: absolute; text-align: center; z-index: 10;">Failed to open stream</div>
      <img id="cam-image" alt="Camera Stream" draggable="false">
    </div>
  </div>

  <script>
    let stateData = null;
    let cameraActive = false;
    const controlsList = ['gain', 'contrast', 'saturation', 'brightness', 'white_balance_temperature', 'sharpness', 'gamma', 'backlight_compensation'];

    function createSliders() {
      const slidersDiv = document.getElementById('sliders');
      controlsList.forEach(ctrl => {
        const div = document.createElement('div');
        div.className = 'control-group';
        div.id = `group-${ctrl}`;
        
        div.innerHTML = `
          <label>${ctrl} <span id="val-${ctrl}" class="value">0</span></label>
          <div class="control-row">
            <input type="range" id="slider-${ctrl}" oninput="updateValDisplay('${ctrl}')" onchange="setControl('${ctrl}', parseInt(this.value))">
            <button onclick="resetDefault('${ctrl}')">Default</button>
          </div>
        `;
        slidersDiv.appendChild(div);
      });
    }
    createSliders();

    function updateValDisplay(ctrl) {
      document.getElementById(`val-${ctrl}`).innerText = document.getElementById(`slider-${ctrl}`).value;
    }

    async function setControl(ctrl, val) {
      await fetch('/api/control', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ control: ctrl, value: val })
      });
    }

    function resetDefault(ctrl) {
      if (stateData && stateData.ranges[ctrl]) {
        setControl(ctrl, stateData.ranges[ctrl].default);
      }
    }

    async function toggleBenchmark() {
      await fetch('/api/benchmark', { method: 'POST' });
    }

    function toggleCamera() {
      cameraActive = !cameraActive;
      const btn = document.getElementById('camera-btn');
      const img = document.getElementById('cam-image');
      const errDiv = document.getElementById('stream-error-msg');
      if (cameraActive) {
        btn.innerText = "Hide Live Camera";
        if (stateData && stateData.camera_error) {
            img.style.display = 'none';
            errDiv.style.display = 'block';
            errDiv.innerText = stateData.camera_error;
        } else {
            img.style.display = 'block';
            errDiv.style.display = 'none';
            img.src = '/stream';
        }
      } else {
        btn.innerText = "Show Live Camera";
        img.style.display = 'none';
        errDiv.style.display = 'none';
        img.src = '';
      }
    }

    async function pollState() {
      try {
        const r = await fetch('/api/state');
        if (r.ok) {
          stateData = await r.json();
          updateUI();
        }
      } catch (e) { }
    }
    setInterval(pollState, 250);

    function updateUI() {
      if (!stateData) return;

      const benchmarking = stateData.is_benchmarking;
      const bBtn = document.getElementById('benchmark-btn');
      bBtn.innerText = benchmarking ? "Stop Benchmark" : "Start Benchmark";
      
      const bStats = document.getElementById('benchmark-stats');
      bStats.style.display = benchmarking ? 'block' : 'none';
      
      if (benchmarking) {
        document.getElementById('bench-solves').innerText = stateData.benchmark_stats.solves;
        document.getElementById('bench-total').innerText = stateData.benchmark_stats.total;
        document.getElementById('bench-rate').innerText = stateData.benchmark_stats.solve_rate.toFixed(2);
        document.getElementById('bench-centroids').innerText = stateData.benchmark_stats.avg_centroids.toFixed(1);
      }

      const errDiv = document.getElementById('stream-error-msg');
      const img = document.getElementById('cam-image');
      if (cameraActive && stateData.camera_error) {
        img.style.display = 'none';
        errDiv.style.display = 'block';
        errDiv.innerText = stateData.camera_error;
      } else if (cameraActive && !stateData.camera_error) {
        errDiv.style.display = 'none';
        img.style.display = 'block';
        if (!img.src.includes('/stream')) {
            img.src = '/stream';
        }
      } else {
          errDiv.style.display = 'none';
      }

      const fSelect = document.getElementById('format');
      if (fSelect.options.length === 0 || fSelect.dataset.formatsHash !== stateData.formats.length.toString()) {
        fSelect.innerHTML = '';
        stateData.formats.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f.id;
          opt.text = `${f.format_type} ${f.width}x${f.height} @ ${f.fps}fps`;
          fSelect.appendChild(opt);
        });
        fSelect.dataset.formatsHash = stateData.formats.length.toString();
      }
      if (fSelect.value !== stateData.current_format_id.toString()) {
        fSelect.value = stateData.current_format_id;
      }
      fSelect.disabled = benchmarking;

      controlsList.forEach(ctrl => {
        const range = stateData.ranges[ctrl];
        const group = document.getElementById(`group-${ctrl}`);
        if (!range) {
          group.style.display = 'none';
          return;
        } else {
          group.style.display = 'block';
        }

        const slider = document.getElementById(`slider-${ctrl}`);
        const currentVal = stateData.current_values[ctrl];
        
        slider.disabled = benchmarking;
        group.querySelector('button').disabled = benchmarking;

        if (document.activeElement !== slider) {
          slider.min = range.min;
          slider.max = range.max;
          slider.step = range.step;
          slider.value = currentVal;
          document.getElementById(`val-${ctrl}`).innerText = currentVal;
        }
      });
    }

    let scale = 1, panX = 0, panY = 0;
    let isViewDragging = false, startX, startY;
    const img = document.getElementById('cam-image');
    const viewer = document.getElementById('viewer');
    
    function updateTransform() {
      img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }

    viewer.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomSensitivity = 0.001;
      const delta = -e.deltaY * zoomSensitivity;
      scale = Math.max(0.1, Math.min(scale * Math.exp(delta), 200));
      updateTransform();
    });

    viewer.addEventListener('mousedown', (e) => {
      isViewDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    });

    window.addEventListener('mouseup', () => isViewDragging = false);
    window.addEventListener('mousemove', (e) => {
      if (!isViewDragging) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      updateTransform();
    });

    let initialPinchDistance = null;
    let initialScale = 1;

    viewer.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        isViewDragging = true;
        startX = e.touches[0].clientX - panX;
        startY = e.touches[0].clientY - panY;
      } else if (e.touches.length === 2) {
        isViewDragging = false;
        initialPinchDistance = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        initialScale = scale;
      }
    }, {passive: false});

    viewer.addEventListener('touchend', () => {
      isViewDragging = false;
      initialPinchDistance = null;
    });

    viewer.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && isViewDragging) {
        e.preventDefault();
        panX = e.touches[0].clientX - startX;
        panY = e.touches[0].clientY - startY;
        updateTransform();
      } else if (e.touches.length === 2 && initialPinchDistance !== null) {
        e.preventDefault();
        const currentDistance = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        let newScale = initialScale * (currentDistance / initialPinchDistance);
        scale = Math.max(0.1, Math.min(newScale, 200));
        updateTransform();
      }
    }, {passive: false});
  </script>
</body>
</html>"#;

async fn get_state(State(state): State<Arc<ServerState>>) -> Json<AppState> {
    let app_state = state.app_state.lock().unwrap().clone();
    Json(app_state)
}

async fn post_control(
    State(state): State<Arc<ServerState>>,
    Json(update): Json<ControlUpdate>,
) -> StatusCode {
    let is_benchmarking = state.app_state.lock().unwrap().is_benchmarking;
    if is_benchmarking {
        return StatusCode::FORBIDDEN;
    }

    if update.control == "format" {
        let _ = state
            .cam_tx
            .send(CamCommand::SetFormat(update.value as usize));
    } else {
        let _ = state
            .cam_tx
            .send(CamCommand::SetControl(update.control, update.value));
    }
    StatusCode::OK
}

async fn post_benchmark(State(state): State<Arc<ServerState>>) -> StatusCode {
    let mut st = state.app_state.lock().unwrap();
    st.is_benchmarking = !st.is_benchmarking;
    if st.is_benchmarking {
        st.benchmark_stats = BenchmarkStats::default();
    }
    StatusCode::OK
}

async fn stream_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Too many active streams. Limited to 1 connection.",
            )
                .into_response();
        }
    };

    let (tx, rx) = tmpsc::channel(1);
    let store_clone = state.store.clone();

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut last_ts = None;
        loop {
            let img_arc = {
                let mut lock = store_clone.image.lock().unwrap();
                loop {
                    if let Some(img) = &*lock {
                        if Some(img.timestamp) != last_ts {
                            break img.clone();
                        }
                    }
                    lock = store_clone.cvar.wait(lock).unwrap();
                }
            };
            last_ts = Some(img_arc.timestamp);

            let (width, height) = (img_arc.width, img_arc.height);

            let mut jpeg_data = Vec::new();
            let mut encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_data, 75);
            if encoder
                .encode(&img_arc.data, width, height, image::ExtendedColorType::L8)
                .is_err()
            {
                break;
            }

            let mut chunk = Vec::new();
            chunk.extend_from_slice(b"--frame\r\n");
            chunk.extend_from_slice(b"Content-Type: image/jpeg\r\n\r\n");
            chunk.extend_from_slice(&jpeg_data);
            chunk.extend_from_slice(b"\r\n");

            if tx.blocking_send(Ok::<_, std::io::Error>(chunk)).is_err() {
                break;
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Response::builder()
        .header(
            header::CONTENT_TYPE,
            "multipart/x-mixed-replace; boundary=frame",
        )
        .body(body)
        .unwrap()
}

#[tokio::main]
async fn main() {
    let store = Arc::new(SharedImageStore {
        image: Mutex::new(None),
        cvar: Condvar::new(),
    });

    let (cam_tx, cam_rx) = std::sync::mpsc::channel();
    let app_state = Arc::new(Mutex::new(AppState::default()));

    let state_clone = app_state.clone();
    let store_clone = store.clone();
    std::thread::spawn(move || run_camera(cam_rx, state_clone, store_clone));

    let state_clone2 = app_state.clone();
    let store_clone2 = store.clone();
    std::thread::spawn(move || run_benchmark(state_clone2, store_clone2));

    let server_state = Arc::new(ServerState {
        cam_tx,
        store,
        app_state,
        semaphore: Arc::new(Semaphore::new(1)),
    });

    let app = Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route("/stream", get(stream_handler))
        .route("/api/state", get(get_state))
        .route("/api/control", post(post_control))
        .route("/api/benchmark", post(post_benchmark))
        .with_state(server_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Web Tuner listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}
