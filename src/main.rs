// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

use clap::Parser;
use log::{error, info};
use std::net::{IpAddr, SocketAddr};
use std::process::exit;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use olive_engine::CameraTrait;
use olive_engine::camera::Camera as LibCamera;
use olive_engine::control_server;
use olive_engine::engine::{Engine, EngineConfig};
use olive_engine::mount::{MountConfig, MountProtocol};
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

    /// Optional Wi-Fi mount address (`<host>` or `<host>:<port>`)
    #[arg(long)]
    mount_host: Option<String>,

    /// Wireless mount protocol (`ioptron`, `skywatcher`, `celestron`)
    #[arg(long, value_enum, default_value_t = MountProtocol::Ioptron)]
    mount_protocol: MountProtocol,
}

fn parse_mount_host(raw: &str, default_port: u16) -> Result<(String, u16), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Mount host cannot be empty".to_string());
    }
    let (host, port) = if let Some(rest) = trimmed.strip_prefix('[') {
        let Some((h, after)) = rest.split_once(']') else {
            return Err(format!("Unclosed '[' in --mount-host {:?}", raw));
        };
        let port = if after.is_empty() {
            default_port
        } else if let Some(p_str) = after.strip_prefix(':') {
            p_str
                .parse::<u16>()
                .map_err(|_| format!("Invalid port {:?} in --mount-host {:?}", p_str, raw))?
        } else {
            return Err(format!("Invalid bracketed --mount-host {:?}", raw));
        };
        (h.to_string(), port)
    } else if let Ok(sock) = trimmed.parse::<SocketAddr>() {
        (sock.ip().to_string(), sock.port())
    } else if let Ok(ip) = trimmed.parse::<IpAddr>() {
        (ip.to_string(), default_port)
    } else if let Some((h, p_str)) = trimmed.rsplit_once(':') {
        let port = p_str
            .parse::<u16>()
            .map_err(|_| format!("Invalid port {:?} in --mount-host {:?}", p_str, raw))?;
        (h.to_string(), port)
    } else {
        (trimmed.to_string(), default_port)
    };

    if host.is_empty() {
        return Err(format!("Missing host in --mount-host {:?}", raw));
    }
    if port == 0 {
        return Err(format!("Port cannot be 0 in --mount-host {:?}", raw));
    }
    Ok((host, port))
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
    let mount_handle = engine.mount_handle();
    if let Some(ref mount_addr) = args.mount_host {
        let protocol = args.mount_protocol;
        match parse_mount_host(mount_addr, protocol.default_port()) {
            Ok((host, port)) => {
                info!("Enabling {:?} mount control at {}:{}", protocol, host, port);
                mount_handle.configure(Some(MountConfig::new(protocol, host, port)));
            }
            Err(err) => {
                error!("Invalid --mount-host: {}", err);
                exit(2);
            }
        }
    }
    let _ = engine.fused_solver.start_imu();
    let _ = engine.start();

    // Trap Ctrl-C to handle graceful shutdown
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl-C");
    info!("Received Ctrl-C, shutting down...");
    mount_handle.shutdown();
    // Allow the non-blocking MountController thread to send abort/halt commands before exit.
    sleep(Duration::from_millis(500)).await;
    exit(0);
}
