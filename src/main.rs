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

    /// Enable MJPEG support for UVC camera
    #[arg(long)]
    uvc_mjpeg: bool,

    /// Background subtraction mode for extraction
    #[arg(long, value_enum, default_value_t = olive_engine::engine::BgMode::Line)]
    bg_mode: olive_engine::engine::BgMode,

    /// Extractor type to use
    #[arg(long, value_enum, default_value_t = olive_engine::engine::Extractor::Approximate)]
    extractor_type: olive_engine::engine::Extractor,

    /// IMU to use (none, bno085, bmi160)
    #[arg(long, default_value = "none")]
    imu: String,

    /// Disable automatic image downsampling
    #[arg(long)]
    no_downsample: bool,

    /// Enable automatic tuning (gain, stacking, extraction)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    auto_mode: bool,
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args = Args::parse();

    let storage_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let storage: Option<Arc<dyn olive_imu::storage::PersistentStorage>> =
        Some(Arc::new(olive_imu::storage::FileStorage::new(storage_dir)));

    // Initialize IMU if requested
    let imu: Option<olive_solve::ImuType> = match args.imu.as_str() {
        "bno085" => Some(olive_solve::ImuType::Bno085),
        "bmi160" => Some(olive_solve::ImuType::Bmi160),
        "none" => None,
        _ => panic!("Invalid IMU selected. Options are: none, bno085, bmi160"),
    };

    // Resolve $HOME in database path if present
    let database_path = if args.database_path.starts_with("$HOME") {
        let home = std::env::var("HOME").unwrap_or_default();
        args.database_path.replace("$HOME", &home)
    } else {
        args.database_path.clone()
    };

    let config = EngineConfig {
        gain: args.gain,
        target_centroids: args.target_centroids,
        database_path,
        isp: if args.uvc { false } else { args.isp },
        bg_mode: args.bg_mode,
        extractor_type: args.extractor_type,
        downsample: if args.no_downsample { 0 } else { 1 },
        auto_mode: args.auto_mode,
        is_adaptive_camera: !args.uvc,
        ..Default::default()
    };

    info!("Starting Olive Engine with configuration: {:?}", config);

    // Initialize and start the camera
    let camera: Arc<dyn CameraTrait> = if args.uvc {
        let mut cam = UvcCamera::new(args.uvc_fps, args.uvc_mjpeg, None, false, args.gain);
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
    let engine = Engine::new(
        camera,
        config,
        client_connected,
        engine_state,
        imu,
        None,
        storage,
        None,
        None,
    )
    .await;
    let _ = engine.fused_solver.start_imu();
    let _ = engine.start();

    // Trap Ctrl-C to handle graceful shutdown
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl-C");
    info!("Received Ctrl-C, shutting down...");
    std::process::exit(0);
}
