// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use axum::{
    Router,
    extract::State,
    response::{Html, Response},
    routing::get,
};
use clap::Parser;
use futures_util::stream;
use olive_engine::Camera;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Use ISP mode instead of Raw mode
    #[arg(short, long, default_value_t = false)]
    isp: bool,
}

struct AppState {
    camera: Arc<Camera>,
    fps: Mutex<f64>,
}

#[tokio::main]
async fn main() {
    // Initialize logger to see info! output from camera.rs
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    let mut camera = Camera::new();
    camera.set_isp_mode(args.isp);
    camera
        .start_capture()
        .expect("Failed to start camera capture");

    // Set exposure to 16.667ms (16667 us)
    camera.set_exposure(16667);

    let camera = Arc::new(camera);
    let state = Arc::new(AppState {
        camera: camera.clone(),
        fps: Mutex::new(0.0),
    });

    // Background thread to track actual throughput FPS
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut frame_count = 0;
        let mut last_check = Instant::now();
        let mut last_timestamp = None;
        loop {
            // Non-blocking check for new image
            if let Some(img) = state_clone.camera.get_last_capture() {
                if Some(img.timestamp) != last_timestamp {
                    frame_count += 1;
                    last_timestamp = Some(img.timestamp);
                }
            }

            let now = Instant::now();
            if now.duration_since(last_check) >= Duration::from_secs(1) {
                let mut fps_lock = state_clone.fps.lock().unwrap();
                *fps_lock = frame_count as f64 / now.duration_since(last_check).as_secs_f64();
                frame_count = 0;
                last_check = now;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/mjpeg", get(mjpeg_handler))
        .route("/fps", get(get_fps))
        .with_state(state);

    let listener = TcpListener::bind("0.0.0.0:8081").await.unwrap();
    println!("Web server running on http://localhost:8081");
    axum::serve(listener, app).await.unwrap();
}

async fn get_fps(State(state): State<Arc<AppState>>) -> String {
    format!("{:.2}", *state.fps.lock().unwrap())
}

async fn index() -> Html<String> {
    Html(format!(
        r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Camera MJPEG Preview</title>
            <script>
                setInterval(async () => {{
                    const response = await fetch('/fps');
                    const fps = await response.text();
                    document.getElementById('fps').innerText = fps;
                }}, 1000);
            </script>
        </head>
        <body>
            <h1>Camera MJPEG Preview</h1>
            <p>Current Throughput FPS: <span id="fps">0.00</span></p>
            <img src="/mjpeg" style="max-width: 100%;">
        </body>
        </html>
    "#
    ))
}

async fn mjpeg_handler(State(state): State<Arc<AppState>>) -> Response {
    let stream = stream::unfold(state, |state| async move {
        let img = state.camera.get_last_capture_sync();
        let config = state
            .camera
            .get_config()
            .expect("Camera configuration not available");

        let mut buffer = std::io::Cursor::new(Vec::new());
        let gray_image = image::GrayImage::from_raw(config.width, config.height, img.data.clone())
            .expect("Failed to create image from raw data");

        image::DynamicImage::ImageLuma8(gray_image)
            .write_to(&mut buffer, image::ImageFormat::Jpeg)
            .expect("Failed to encode image as JPEG");

        let data = buffer.into_inner();
        let chunk = format!(
            "--boundary\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            data.len()
        );
        let mut full_chunk = chunk.into_bytes();
        full_chunk.extend_from_slice(&data);
        full_chunk.extend_from_slice(b"\r\n");

        Some((Ok::<_, std::convert::Infallible>(full_chunk), state))
    });

    Response::builder()
        .header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=boundary",
        )
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}
