// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use clap::Parser;
use log::info;
use std::sync::Arc;

use olive_engine::CameraTrait;
use olive_engine::camera::Camera as LibCamera;
use olive_engine::control_server;
use olive_engine::engine::{Engine, EngineConfig};
use olive_engine::uvc_camera::Camera as UvcCamera;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Minimum exposure time in microseconds
    #[arg(long, default_value_t = 16670)]
    min_exposure: i32,

    /// Maximum exposure time in microseconds
    #[arg(long, default_value_t = 1000000)]
    max_exposure: i32,

    /// Analog gain to apply to the camera
    #[arg(long)]
    gain: Option<f32>,

    /// Target number of centroids for auto-exposure
    #[arg(long, default_value_t = 20)]
    target_centroids: usize,

    /// Path to the Tetra3 database NPZ file
    #[arg(long, default_value = "$HOME/data/default_database.npz")]
    database_path: String,

    /// ISP mode configuration
    #[arg(long)]
    isp: bool,

    /// Use UVC camera instead of libcamera
    #[arg(long)]
    uvc: bool,

    /// Target FPS for UVC camera
    #[arg(long, default_value_t = 30)]
    uvc_fps: u32,

    /// Background subtraction mode for extraction
    #[arg(long, value_enum, default_value_t = olive_engine::engine::BgMode::Line)]
    bg_mode: olive_engine::engine::BgMode,

    /// Extractor type to use
    #[arg(long, value_enum, default_value_t = olive_engine::engine::Extractor::Sequential)]
    extractor_type: olive_engine::engine::Extractor,

    /// Rayon threads configuration
    #[arg(long, default_value_t = 2)]
    rayon_threads: usize,

    /// IMU to use (none, bno085, bmi160)
    #[arg(long, default_value = "none")]
    imu: String,
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args = Args::parse();

    // Initialize IMU if requested
    let imu: Option<Arc<olive_engine::imu::Imu>> = match args.imu.as_str() {
        "bno085" => {
            let device = olive_engine::bno08x::Bno08xDevice::new(5).expect("Failed to init BNO085");
            Some(Arc::new(
                olive_engine::imu::Imu::start(device).expect("Failed to start IMU thread"),
            ))
        }
        "bmi160" => {
            let device = olive_engine::bmi160::Bmi160Device::new().expect("Failed to init BMI160");
            Some(Arc::new(
                olive_engine::imu::Imu::start(device).expect("Failed to start IMU thread"),
            ))
        }
        "none" => None,
        _ => panic!("Invalid IMU selected. Options are: none, bno085, bmi160"),
    };

    // Initialize Rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.rayon_threads)
        .build_global()
        .expect("Failed to initialize Rayon thread pool");

    // Resolve $HOME in database path if present
    let database_path = if args.database_path.starts_with("$HOME") {
        let home = std::env::var("HOME").unwrap_or_default();
        args.database_path.replace("$HOME", &home)
    } else {
        args.database_path.clone()
    };

    let config = EngineConfig {
        min_exposure_us: args.min_exposure,
        max_exposure_us: args.max_exposure,
        gain: args.gain,
        target_centroids: args.target_centroids,
        database_path,
        isp: if args.uvc { false } else { args.isp },
        bg_mode: args.bg_mode,
        extractor_type: args.extractor_type,
        uvc: args.uvc,
    };

    info!("Starting Olive Engine with configuration: {:?}", config);

    // Initialize and start the camera
    let camera: Arc<dyn CameraTrait> = if args.uvc {
        let mut cam = UvcCamera::new(args.uvc_fps);
        cam.start_capture().expect("Failed to start UVC camera");
        Arc::new(cam)
    } else {
        let mut cam = LibCamera::new(args.isp);
        cam.start_capture().expect("Failed to start libcamera");
        Arc::new(cam)
    };

    let client_connected = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let engine_state = Arc::new(std::sync::RwLock::new(
        olive_engine::engine::EngineStateData::default(),
    ));

    // Start the control server in the background
    let server_camera = camera.clone();
    let server_client_connected = client_connected.clone();
    let server_engine_state = engine_state.clone();
    tokio::spawn(async move {
        control_server::start_server(server_camera, server_client_connected, server_engine_state)
            .await;
    });

    // Initialize the engine
    let engine = Engine::new(camera, config, client_connected, engine_state, imu);
    engine.start();

    // Trap Ctrl-C to handle graceful shutdown
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl-C");
    info!("Received Ctrl-C, shutting down...");
    std::process::exit(0);
}
