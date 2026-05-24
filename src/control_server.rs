use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::CameraTrait;
use std::sync::RwLock;

pub struct ServerState {
    pub camera: Arc<dyn CameraTrait>,
    pub semaphore: Arc<Semaphore>,
    pub hcg_enabled: AtomicBool,
    pub client_connected: Arc<AtomicBool>,
    pub engine_state: Arc<RwLock<crate::engine::EngineStateData>>,
}

#[derive(Serialize, Deserialize)]
pub struct ControlsData {
    pub gain: f32,
    pub exposure_ms: f32,
    pub hcg: bool,
    #[serde(default)]
    pub min_gain: f32,
    #[serde(default)]
    pub max_gain: f32,
}

const INDEX_HTML: &str = r#"
<!DOCTYPE html>
<html>
<head>
<title>Olive Control</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
  body { font-family: monospace; background: #000; color: #f00; margin: 0; display: flex; height: 100vh; flex-direction: column; overflow: hidden; }
  a, button { color: #f00; background: #110000; border: 1px solid #f00; cursor: pointer; padding: 10px; text-decoration: none; font-family: inherit; font-size: 1em; }
  a:hover, button:hover { background: #330000; }
  input[type=range] { accent-color: #f00; }
  input[type=checkbox] { accent-color: #f00; }
  .view { display: none; flex-grow: 1; padding: 20px; box-sizing: border-box; overflow-y: auto; }
  .active { display: block !important; }
  /* camera viewer layout */
  #camera-mode.active { display: flex; flex-direction: column; padding: 0; overflow: hidden; }
  #controls { width: 100%; padding: 20px; border-bottom: 1px solid #f00; display: flex; flex-direction: column; box-sizing: border-box; overflow-y: auto; flex-shrink: 0; max-height: 50vh; }
  #viewer { flex-grow: 1; overflow: hidden; position: relative; display: flex; justify-content: center; align-items: center; background: #000; cursor: grab; touch-action: none; }
  #viewer:active { cursor: grabbing; }
  #cam-image { transform-origin: center; transition: transform 0.1s ease-out; pointer-events: none; max-width: 100%; max-height: 100%; image-rendering: pixelated; }
  label { display: block; margin-top: 15px; font-size: 0.9em; color: #f00; }
  input[type=range] { width: 100%; }
  .value { float: right; color: #f00; }
  h1, h3 { margin-top: 0; }
  
  @media (min-width: 768px) {
    #camera-mode.active { flex-direction: row; }
    #controls { width: 300px; border-right: 1px solid #f00; border-bottom: none; max-height: none; }
  }
</style>
</head>
<body>
  <div id="home" class="view active">
    <h1>Olive Engine Status</h1>
    <p>Status: <span id="status">-</span></p>
    <p>RA: <span id="ra">-</span></p>
    <p>Dec: <span id="dec">-</span></p>
    <p>Roll: <span id="roll">-</span></p>
    <p>Exposure: <span id="exposure_time">-</span> ms</p>
    <p>Post-Process: <span id="post_process">-</span> ms</p>
    <p>Extraction: <span id="extract">-</span> ms</p>
    <p>Solve: <span id="solve">-</span> ms</p>
    <p>Solve Rate (5s): <span id="solve_rate">-</span> Hz</p>
    <button onclick="goCamera()" style="margin-top: 20px;">Camera Mode</button>
    <button onclick="captureImage()" style="margin-top: 10px;">Capture Image</button>
  </div>

  <div id="camera-mode" class="view">
    <div id="controls">
      <button onclick="goHome()" style="margin-bottom: 20px;">Back</button>
      <h3>Camera Controls</h3>
      
      <div style="margin-bottom: 15px; padding: 10px; border: 1px dashed #f00;">
         <strong>Focus Metric (HFR):</strong> <span id="hfr-val" class="value">0.00</span>
      </div>
      
      <label>Gain <span id="gain-val" class="value">0</span></label>
      <input type="range" id="gain" step="0.1">
      
      <label>Exposure (ms) <span id="exp-val" class="value">0</span></label>
      <input type="range" id="exposure" min="16.667" max="100" step="0.1">
      
      <label style="display:flex; align-items:center; margin-top: 25px;">
        <input type="checkbox" id="hcg" style="width:auto; margin-right:10px;">
        Enable HCG
      </label>
    </div>
    <div id="viewer">
      <img id="cam-image" alt="Camera Stream" draggable="false">
    </div>
  </div>

  <script>
    let pollInterval;

    // View Routing
    function goCamera() {
      document.getElementById('home').classList.remove('active');
      document.getElementById('camera-mode').classList.add('active');
      document.getElementById('cam-image').src = '/stream';
      loadControls();
    }

    function goHome() {
      document.getElementById('camera-mode').classList.remove('active');
      document.getElementById('home').classList.add('active');
      document.getElementById('cam-image').src = ''; // stop stream
    }

    async function captureImage() {
      try {
        await fetch('/api/capture', { method: 'POST' });
        alert('Capture requested');
      } catch (e) {
        alert('Failed to request capture');
      }
    }

    // Engine Polling
    async function pollEngine() {
      try {
        const r = await fetch('/api/engine_state');
        if (r.ok) {
          const data = await r.json();
          // Status Screen Sync
          document.getElementById('status').innerText = data.status;
          document.getElementById('ra').innerText = data.ra !== null ? data.ra.toFixed(6) : 'N/A';
          document.getElementById('dec').innerText = data.dec !== null ? data.dec.toFixed(6) : 'N/A';
          document.getElementById('roll').innerText = data.roll !== null ? data.roll.toFixed(2) : 'N/A';
          document.getElementById('exposure_time').innerText = (data.exposure_us / 1000).toFixed(2);
          document.getElementById('post_process').innerText = data.post_process_ms.toFixed(2);
          document.getElementById('extract').innerText = data.extract_ms.toFixed(2);
          document.getElementById('solve').innerText = data.solve_ms.toFixed(2);
          document.getElementById('solve_rate').innerText = data.solve_rate_hz.toFixed(2);
          
          // Auxiliary Control Sync (Only populates during active Camera Mode)
          if(document.getElementById('hfr-val')) {
              document.getElementById('hfr-val').innerText = data.hfr.toFixed(2);
          }
        }
      } catch (e) {
        document.getElementById('status').innerText = 'Disconnected';
      }
    }
    
    pollInterval = setInterval(pollEngine, 250);

    // Camera Viewer logic
    let scale = 1, panX = 0, panY = 0;
    let isDragging = false, startX, startY;
    const img = document.getElementById('cam-image');
    const viewer = document.getElementById('viewer');
    
    function updateTransform() {
      img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }

    viewer.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomSensitivity = 0.001;
      const delta = -e.deltaY * zoomSensitivity;
      let newScale = scale * Math.exp(delta);
      newScale = Math.max(0.1, Math.min(newScale, 200));
      scale = newScale;
      updateTransform();
    });

    viewer.addEventListener('mousedown', (e) => {
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    });

    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      updateTransform();
    });

    let initialPinchDistance = null;
    let initialScale = 1;

    viewer.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        startX = e.touches[0].clientX - panX;
        startY = e.touches[0].clientY - panY;
      } else if (e.touches.length === 2) {
        isDragging = false;
        initialPinchDistance = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        initialScale = scale;
      }
    }, {passive: false});

    viewer.addEventListener('touchend', () => {
      isDragging = false;
      initialPinchDistance = null;
    });

    viewer.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && isDragging) {
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

    // Camera Controls logic
    const gain = document.getElementById('gain');
    const gainVal = document.getElementById('gain-val');
    const exp = document.getElementById('exposure');
    const expVal = document.getElementById('exp-val');
    const hcg = document.getElementById('hcg');

    async function loadControls() {
      const r = await fetch('/api/controls');
      const data = await r.json();
      
      gain.min = data.min_gain;
      gain.max = data.max_gain;
      gain.value = data.gain;
      gainVal.innerText = data.gain.toFixed(2);
      
      exp.value = data.exposure_ms;
      expVal.innerText = data.exposure_ms.toFixed(3);
      
      hcg.checked = data.hcg;
    }

    async function setControls() {
      const data = {
        gain: parseFloat(gain.value),
        exposure_ms: parseFloat(exp.value),
        hcg: hcg.checked
      };
      await fetch('/api/controls', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      });
    }

    gain.addEventListener('input', () => { gainVal.innerText = parseFloat(gain.value).toFixed(2); });
    exp.addEventListener('input', () => { expVal.innerText = parseFloat(exp.value).toFixed(3); });

    gain.addEventListener('change', setControls);
    exp.addEventListener('change', setControls);
    hcg.addEventListener('change', setControls);
  </script>
</body>
</html>
"#;

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn get_controls(State(state): State<Arc<ServerState>>) -> Json<ControlsData> {
    let camera = &state.camera;
    let gain_range = camera.get_gain_range();
    Json(ControlsData {
        gain: camera.get_gain(),
        exposure_ms: camera.get_exposure() as f32 / 1000.0,
        hcg: state.hcg_enabled.load(Ordering::Relaxed),
        min_gain: gain_range.0,
        max_gain: gain_range.1,
    })
}

async fn post_controls(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ControlsData>,
) -> StatusCode {
    let camera = &state.camera;
    camera.set_gain(payload.gain);
    camera.set_exposure((payload.exposure_ms * 1000.0) as i32);

    let current_hcg = state.hcg_enabled.load(Ordering::Relaxed);
    if payload.hcg != current_hcg {
        if payload.hcg {
            camera.enable_hcg();
        } else {
            camera.disable_hcg();
        }
        state.hcg_enabled.store(payload.hcg, Ordering::Relaxed);
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

    let (tx, rx) = mpsc::channel(1);
    let state_clone = state.clone();

    tokio::task::spawn_blocking(move || {
        let _permit = permit; // Hold permit for duration of the stream

        // Setup: mark connected and set to min bounds
        state_clone.client_connected.store(true, Ordering::SeqCst);
        let camera = &state_clone.camera;
        camera.set_gain(camera.get_gain_range().0);
        camera.set_exposure(16670); // Default to a low exposure for responsive UI
        camera.disable_hcg();
        state_clone.hcg_enabled.store(false, Ordering::Relaxed);

        let mut last_ts = None;
        loop {
            // Block until next capture
            let image_opt =
                camera.wait_for_next_capture(last_ts, Some(std::time::Duration::from_millis(500)));
            let image_arc = match image_opt {
                Some(img) => img,
                None => continue,
            };
            last_ts = Some(image_arc.timestamp);

            let config = if let Some(c) = camera.get_config() {
                c
            } else {
                continue;
            };
            let width = config.width;
            let height = config.height;
            let data = &image_arc.data;

            // Encode 8-bit luma to JPEG
            let mut jpeg_data = Vec::new();
            let mut encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_data, 75);
            if let Err(e) = encoder.encode(data, width, height, image::ExtendedColorType::L8) {
                log::error!("JPEG encoding failed: {}", e);
                break;
            }

            let mut chunk = Vec::new();
            chunk.extend_from_slice(b"--frame\r\n");
            chunk.extend_from_slice(b"Content-Type: image/jpeg\r\n\r\n");
            chunk.extend_from_slice(&jpeg_data);
            chunk.extend_from_slice(b"\r\n");

            // Send chunk to client; breaks if the client disconnected
            if tx.blocking_send(Ok::<_, std::io::Error>(chunk)).is_err() {
                log::info!("Stream client disconnected.");
                break;
            }
        }

        // Teardown
        state_clone.client_connected.store(false, Ordering::SeqCst);
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

async fn get_engine_state(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let engine_state = state.engine_state.read().unwrap().clone();
    (
        [
            (
                header::CACHE_CONTROL,
                "no-store, no-cache, must-revalidate, proxy-revalidate",
            ),
            (header::PRAGMA, "no-cache"),
            (header::EXPIRES, "0"),
        ],
        Json(engine_state),
    )
}

async fn post_capture(State(state): State<Arc<ServerState>>) -> StatusCode {
    let mut engine_state = state.engine_state.write().unwrap();
    engine_state.capture_requested = true;
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct LocationTimeData {
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub timestamp_ms: Option<u64>,
}

async fn post_location_time(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<LocationTimeData>,
) -> StatusCode {
    let mut engine_state = state.engine_state.write().unwrap();
    if let Some(lat) = payload.latitude {
        engine_state.client_latitude = Some(lat);
    }
    if let Some(lon) = payload.longitude {
        engine_state.client_longitude = Some(lon);
    }
    if let Some(ms) = payload.timestamp_ms {
        if let Some(dt) =
            chrono::DateTime::from_timestamp((ms / 1000) as i64, ((ms % 1000) * 1_000_000) as u32)
        {
            engine_state.client_date = Some(dt);
        }
    }
    StatusCode::OK
}

pub async fn start_server(
    camera: Arc<dyn CameraTrait>,
    client_connected: Arc<AtomicBool>,
    engine_state: Arc<RwLock<crate::engine::EngineStateData>>,
) {
    let state = Arc::new(ServerState {
        camera,
        semaphore: Arc::new(Semaphore::new(1)), // Limit to 1 connection to minimize memory footprint
        hcg_enabled: AtomicBool::new(false),
        client_connected,
        engine_state,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/stream", get(stream_handler))
        .route("/api/controls", get(get_controls).post(post_controls))
        .route("/api/capture", post(post_capture))
        .route("/api/engine_state", get(get_engine_state))
        .route("/api/location_time", post(post_location_time))
        .with_state(state);

    let listener = match tokio::net::TcpListener::bind("0.0.0.0:8080").await {
        Ok(l) => l,
        Err(e) => {
            log::error!("Failed to bind Control Server port 8080: {}", e);
            return;
        }
    };
    log::info!("Control server listening on http://0.0.0.0:8080");

    if let Err(e) = axum::serve(listener, app).await {
        log::error!("Control server failed: {}", e);
    }
}
