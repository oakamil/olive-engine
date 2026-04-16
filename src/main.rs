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
    #[arg(long, default_value_t = 31.0)]
    gain: f32,

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

    /// Use GlobalMedian instead of BlockMedian for extraction
    #[arg(long)]
    fast_bg: bool,

    /// Extractor type to use
    #[arg(long, value_enum, default_value_t = olive_engine::engine::ExtractorType::FastExtractSequential)]
    extractor_type: olive_engine::engine::ExtractorType,

    /// Rayon threads configuration
    #[arg(long, default_value_t = 2)]
    rayon_threads: usize,
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args = Args::parse();

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
        fast_bg: args.fast_bg,
        extractor_type: args.extractor_type,
        uvc: args.uvc,
    };

    info!("Starting Olive Engine with configuration: {:?}", config);

    // Initialize and start the camera
    let camera: Arc<dyn CameraTrait> = if args.uvc {
        let mut cam = UvcCamera::new(false);
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
    let engine = Engine::new(camera, config, client_connected, engine_state);
    engine.start();

    // Trap Ctrl-C to handle graceful shutdown
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl-C");
    info!("Received Ctrl-C, shutting down...");
    std::process::exit(0);
}
