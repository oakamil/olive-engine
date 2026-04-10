// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use clap::Parser;
use log::info;
use std::sync::Arc;

use olive_engine::camera::Camera;
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
    use_isp_mode: bool,
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
    };

    info!("Starting Olive Engine with configuration: {:?}", config);

    // Initialize and start the camera
    let mut camera = Camera::new();
    camera.set_isp_mode(args.use_isp_mode);
    camera
        .start_capture()
        .expect("Failed to start camera capture");
    let camera = Arc::new(camera);

    // Initialize the engine
    let mut engine = Engine::new(camera, config);

    // Trap Ctrl-C to handle graceful shutdown
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl-C");
        info!("Received Ctrl-C, shutting down...");
        std::process::exit(0);
    });

    // Run the engine
    engine.run_calibration_loop().await;
    engine.run_solving_loop().await;
}
