// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, RwLock};
use tokio::runtime::Runtime;

use crate::CameraTrait;
use crate::camera::Camera as LibCamera;
use crate::control_server;
use crate::engine::{Engine, EngineConfig, EngineStateData};
use crate::uvc_camera::Camera as UvcCamera;

#[pyfunction]
#[pyo3(signature = (isp=false, fast_bg=false, extractor_type="fast_extract_sequential".to_string(), rayon_threads=2, uvc=false, target_centroids=20, database_path=None, callback=None))]
fn start_engine(
    py: Python,
    isp: bool,
    fast_bg: bool,
    extractor_type: String,
    rayon_threads: usize,
    uvc: bool,
    target_centroids: usize,
    database_path: Option<String>,
    callback: Option<PyObject>,
) -> PyResult<()> {
    // Initialize Rayon thread pool (ignore if already initialized)
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .build_global();

    let db_path = database_path.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}/data/default_database.npz", home)
    });

    let ext_type = match extractor_type.to_lowercase().as_str() {
        "sequential" | "sequential" => crate::engine::ExtractorType::Sequential,
        "fast_extract" | "fastextract" => crate::engine::ExtractorType::FastExtract,
        "cedar_detect" | "cedardetect" | "cedar-detect" => {
            crate::engine::ExtractorType::CedarDetect
        }
        _ => crate::engine::ExtractorType::Sequential,
    };

    let config = EngineConfig {
        min_exposure_us: 16670,
        max_exposure_us: 1000000,
        gain: 31.0,
        target_centroids,
        database_path: db_path,
        isp,
        fast_bg,
        extractor_type: ext_type,
        uvc,
    };

    log::info!(
        "Starting Olive Engine from Python with configuration: {:?}",
        config
    );

    let camera: Arc<dyn CameraTrait> = if uvc {
        let mut cam = UvcCamera::new(false);
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

    let engine = Engine::new(camera, config, client_connected, engine_state.clone());
    engine.start();

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
