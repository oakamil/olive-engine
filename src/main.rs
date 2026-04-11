// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use clap::Parser;
use log::info;
use std::sync::Arc;

use olive_engine::camera::Camera;
use olive_engine::control_server;
use olive_engine::engine::{Engine, EngineConfig};

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

    /// Path to the Tetra3 database NPZ file
    #[arg(long, default_value = "$HOME/data/default_database.npz")]
    database_path: String,

    /// ISP mode configuration
    #[arg(long)]
    isp: bool,

    /// Use GlobalMedian instead of BlockMedian for extraction
    #[arg(long)]
    fast: bool,
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args = Args::parse();

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
        database_path,
        isp: args.isp,
        fast: args.fast,
    };

    info!("Starting Olive Engine with configuration: {:?}", config);

    // Initialize and start the camera
    let mut camera = Camera::new(args.isp);
    camera
        .start_capture()
        .expect("Failed to start camera capture");
    let camera = Arc::new(camera);

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
    let mut engine = Engine::new(camera, config, client_connected, engine_state);

    // Trap Ctrl-C to handle graceful shutdown
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl-C");
        info!("Received Ctrl-C, shutting down...");
        std::process::exit(0);
    });

    // Run the engine
    loop {
        engine.run_calibration_loop().await;
        engine.run_solving_loop().await;
    }
}
