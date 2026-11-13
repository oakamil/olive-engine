// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, RwLock};
use tokio::runtime::Runtime;

use crate::CameraTrait;
use crate::camera::Camera as LibCamera;
use crate::control_server;
use crate::engine::{BgMode, Engine, EngineConfig, EngineStateData, Extractor};
use crate::uvc_camera::Camera as UvcCamera;

#[pyfunction]
#[pyo3(
    signature = (
        isp=false,
        bg_mode="line".to_string(),
        extractor_type="approximate".to_string(),
        uvc=false,
        uvc_fps=None,
        uvc_mjpeg=false,
        gain=None,
        target_centroids=20,
        database_path=None,
        callback=None
    )
)]
/// Starts the Olive Engine plate-solving daemon and blocks until interrupted.
///
/// Spawns camera acquisition, the web diagnostic server (port 8080), LX200
/// telescope server (port 4030), and ASCOM Alpaca services (port 11111).
///
/// Parameters
/// ----------
/// isp : bool, default=False
///     Enable Raspberry Pi ISP hardware pipeline (libcamera backend only).
/// bg_mode : {"line", "global", "block"}, default="line"
///     Background subtraction method for star extraction.
/// extractor_type : {"approximate", "fast"}, default="approximate"
///     Centroid extraction backend algorithm.
/// uvc : bool, default=False
///     Use USB UVC camera backend instead of libcamera.
/// uvc_fps : int, optional
///     Target framerate for UVC capture (default: 30).
/// uvc_mjpeg : bool, default=False
///     Request MJPEG stream format from UVC camera.
/// gain : float, optional
///     Manual analog gain override. Disables adaptive gain when set.
/// target_centroids : int, default=20
///     Target centroid count for auto-exposure regulation.
/// database_path : str, optional
///     Path to Tetra3 star catalog (`.npz`). Defaults to `~/data/default_database.npz`.
/// callback : callable, optional
///     Unary function invoked periodically with telemetry dict:
///     `status`, `ra`, `dec`, `roll`, `exposure_us`, `post_process_ms`,
///     `extract_ms`, `solve_ms`, `solve_rate_hz`, `centroids`,
///     `target_ra`, `target_dec`.
fn start_engine(
    py: Python,
    isp: bool,
    bg_mode: String,
    extractor_type: String,

    uvc: bool,
    uvc_fps: Option<u32>,
    uvc_mjpeg: bool,
    gain: Option<f32>,
    target_centroids: usize,
    database_path: Option<String>,
    callback: Option<PyObject>,
) -> PyResult<()> {
    let db_path = database_path.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}/data/default_database.npz", home)
    });

    let ext_type = match extractor_type.to_lowercase().as_str() {
        "fast" => Extractor::Fast,
        _ => Extractor::Approximate,
    };

    let bg_sub_mode = match bg_mode.to_lowercase().as_str() {
        "global" => BgMode::Global,
        "block" => BgMode::Block,
        _ => BgMode::Line,
    };

    let config = EngineConfig {
        gain,
        target_centroids,
        database_path: db_path,
        isp,
        bg_mode: bg_sub_mode,
        extractor_type: ext_type,
        is_adaptive_camera: !uvc,
        ..Default::default()
    };

    log::info!(
        "Starting Olive Engine from Python with configuration: {:?}",
        config
    );

    let camera: Arc<dyn CameraTrait> = if uvc {
        let mut cam = UvcCamera::new(uvc_fps.unwrap_or(30), uvc_mjpeg, None, false, None);
        cam.start_capture().expect("Failed to start UVC camera");
        Arc::new(cam)
    } else {
        let mut cam = LibCamera::new(isp);
        cam.start_capture().expect("Failed to start libcamera");
        Arc::new(cam)
    };

    let client_connected = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let engine_state = Arc::new(RwLock::new(EngineStateData::default()));

    let rt = Runtime::new().unwrap();

    let server_camera = camera.clone();
    let server_client_connected = client_connected.clone();
    let server_engine_state = engine_state.clone();
    rt.spawn(async move {
        control_server::start_server(server_camera, server_client_connected, server_engine_state)
            .await;
    });

    let engine = rt.block_on(Engine::new(
        camera,
        config,
        client_connected,
        engine_state.clone(),
        None,
        None,
        None,
        None,
        None,
    ));
    let _ = engine.start();

    loop {
        py.check_signals()?;
        if let Some(cb) = &callback {
            let state = engine_state.read().unwrap().clone();
            let dict = PyDict::new_bound(py);
            dict.set_item("status", state.status)?;
            dict.set_item("ra", state.ra)?;
            dict.set_item("dec", state.dec)?;
            dict.set_item("roll", state.roll)?;
            dict.set_item("exposure_us", state.exposure_us)?;
            dict.set_item("post_process_ms", state.post_process_ms)?;
            dict.set_item("extract_ms", state.extract_ms)?;
            dict.set_item("solve_ms", state.solve_ms)?;
            dict.set_item("solve_rate_hz", state.solve_rate_hz)?;
            dict.set_item("centroids", state.centroids)?;
            dict.set_item("target_ra", state.target_ra)?;
            dict.set_item("target_dec", state.target_dec)?;
            cb.call1(py, (dict,))?;
        }
        py.allow_threads(|| std::thread::sleep(std::time::Duration::from_millis(100)));
    }
}

#[pymodule]
fn olive_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_engine, m)?)?;
    Ok(())
}
