use std::net::SocketAddr;
use std::os::unix::io::FromRawFd;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc::UnboundedReceiver;

use axum::{
    Json, Router,
    extract::{Form, Path, Query, State},
    response::IntoResponse,
    routing::get,
};
use log::{error, info};
use serde::{Deserialize, Serialize};
use tokio::net::UdpSocket;

use crate::telescope_server::{SharedPosition, TargetOp};

const ALPACA_PORT: u16 = 11111;
const DISCOVERY_PORT: u16 = 32227;

#[derive(Clone)]
pub struct AlpacaServer {
    shared_position: Arc<SharedPosition>,
    is_running: Arc<AtomicBool>,
    fd_receiver: Arc<tokio::sync::Mutex<Option<UnboundedReceiver<i32>>>>,
    discovery_fd_receiver: Arc<tokio::sync::Mutex<Option<UnboundedReceiver<i32>>>>,
}

impl AlpacaServer {
    pub fn new(
        shared_position: Arc<SharedPosition>,
        is_running: Arc<AtomicBool>,
        fd_receiver: Option<UnboundedReceiver<i32>>,
        discovery_fd_receiver: Option<UnboundedReceiver<i32>>,
    ) -> Self {
        Self {
            shared_position,
            is_running,
            fd_receiver: Arc::new(tokio::sync::Mutex::new(fd_receiver)),
            discovery_fd_receiver: Arc::new(tokio::sync::Mutex::new(discovery_fd_receiver)),
        }
    }

    pub async fn serve_requests(&self) -> Result<(), Box<dyn std::error::Error + 'static>> {
        // Start Discovery Server
        let discovery_running = self.is_running.clone();
        tokio::spawn(async move {
            if let Err(e) = run_discovery_server(discovery_running, None).await {
                error!("Alpaca discovery server error: {}", e);
            }
        });

        // Start REST API
        // Start REST API
        let app = Router::new()
            .route("/management/*path", get(management_handler))
            .route(
                "/api/v1/telescope/0/:method",
                get(handle_get).put(handle_put),
            )
            .fallback(|uri: hyper::Uri| async move {
                error!("Alpaca REST API 404 Not Found: {}", uri);
                (
                    hyper::StatusCode::NOT_FOUND,
                    format!("No route for {}", uri),
                )
            })
            .layer(axum::middleware::from_fn(lowercase_path))
            .with_state(self.clone());

        let fd_receiver_opt = self.fd_receiver.lock().await.take();

        if let Some(mut rx) = fd_receiver_opt {
            error!("!!! Running Alpaca REST API via Kotlin FDs !!!");
            use hyper::server::conn::http1;
            use hyper_util::rt::TokioIo;
            use std::net::TcpStream as StdTcpStream;
            use std::os::unix::io::FromRawFd;
            use tokio::net::TcpStream;
            use tower::ServiceExt;

            loop {
                match rx.recv().await {
                    Some(fd) => {
                        error!("!!! Alpaca received FD {} from Kotlin !!!", fd);
                        let std_stream = unsafe { StdTcpStream::from_raw_fd(fd) };
                        if let Err(e) = std_stream.set_nonblocking(true) {
                            error!("Failed to set FD nonblocking: {}", e);
                            continue;
                        }
                        if let Ok(stream) = TcpStream::from_std(std_stream) {
                            let io = TokioIo::new(stream);
                            let app_clone = app.clone();
                            tokio::spawn(async move {
                                let hyper_service = hyper::service::service_fn(
                                    move |req: hyper::Request<hyper::body::Incoming>| {
                                        error!(
                                            "!!! Alpaca HTTP Request: {} {} !!!",
                                            req.method(),
                                            req.uri()
                                        );
                                        let req = req.map(axum::body::Body::new);
                                        app_clone.clone().oneshot(req)
                                    },
                                );
                                if let Err(err) = http1::Builder::new()
                                    .serve_connection(io, hyper_service)
                                    .await
                                {
                                    error!("Alpaca serve_connection error: {}", err);
                                }
                            });
                        }
                    }
                    None => break,
                }
            }
        } else {
            let addr = SocketAddr::from(([0, 0, 0, 0], ALPACA_PORT));
            info!("Running Alpaca REST API on: {}", addr);

            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
        }

        Ok(())
    }
}

async fn run_discovery_server(
    is_running: Arc<AtomicBool>,
    fd: Option<i32>,
) -> Result<(), Box<dyn std::error::Error + 'static>> {
    let socket = if let Some(f) = fd {
        let std_sock = unsafe { std::net::UdpSocket::from_raw_fd(f) };
        std_sock.set_nonblocking(true)?;
        UdpSocket::from_std(std_sock)?
    } else {
        let sock = socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::DGRAM,
            Some(socket2::Protocol::UDP),
        )?;
        sock.set_reuse_address(true)?;
        sock.set_broadcast(true)?;
        sock.bind(&socket2::SockAddr::from(SocketAddr::from((
            [0, 0, 0, 0],
            DISCOVERY_PORT,
        ))))?;
        sock.set_nonblocking(true)?;
        let std_sock: std::net::UdpSocket = sock.into();
        UdpSocket::from_std(std_sock)?
    };
    info!(
        "Running Alpaca Discovery Server on UDP port {}",
        DISCOVERY_PORT
    );

    let mut buf = [0u8; 1024];
    let mut tick = 0;

    loop {
        if !is_running.load(Ordering::Relaxed) {
            break;
        }

        tick += 1;
        if tick % 10 == 0 {
            error!(
                "Alpaca UDP Discovery thread is ALIVE and looping. tick={}",
                tick
            );
        }

        match tokio::time::timeout(
            std::time::Duration::from_secs(1),
            socket.recv_from(&mut buf),
        )
        .await
        {
            Ok(Ok((len, addr))) => {
                let msg = String::from_utf8_lossy(&buf[..len]);
                error!("Alpaca UDP Discovery Received: {} from {}", msg, addr);
                if msg.to_lowercase().contains("alpacadiscovery1") {
                    let response = format!("{{\"AlpacaPort\": {}}}", ALPACA_PORT);
                    if let Err(e) = socket.send_to(response.as_bytes(), addr).await {
                        error!("Failed to send Alpaca discovery response: {}", e);
                    }
                }
            }
            Ok(Err(e)) => {
                error!("Alpaca discovery recv error: {}", e);
            }
            Err(_) => {
                // timeout, continue checking is_running
            }
        }
    }

    Ok(())
}

#[derive(Serialize)]
struct AlpacaResponse<T> {
    #[serde(rename = "ClientTransactionID")]
    client_transaction_id: u32,
    #[serde(rename = "ServerTransactionID")]
    server_transaction_id: u32,
    #[serde(rename = "ErrorNumber")]
    error_number: i32,
    #[serde(rename = "ErrorMessage")]
    error_message: String,
    #[serde(rename = "Value", skip_serializing_if = "Option::is_none")]
    value: Option<T>,
}

impl<T> AlpacaResponse<T> {
    fn success(client_transaction_id: u32, value: Option<T>) -> Self {
        Self {
            client_transaction_id,
            server_transaction_id: 0,
            error_number: 0,
            error_message: String::new(),
            value,
        }
    }

    fn error(client_transaction_id: u32, msg: &str) -> Self {
        Self {
            client_transaction_id,
            server_transaction_id: 0,
            error_number: 1024,
            error_message: msg.to_string(),
            value: None,
        }
    }
}

#[derive(Deserialize)]
struct AlpacaParams {
    #[serde(rename = "ClientTransactionID", default)]
    client_transaction_id: u32,
    #[serde(rename = "ClientID", default)]
    _client_id: u32,
    #[serde(rename = "RightAscension", default)]
    right_ascension: Option<f64>,
    #[serde(rename = "Declination", default)]
    declination: Option<f64>,
    #[serde(rename = "Connected", default)]
    _connected: Option<bool>,
}

#[derive(Serialize)]
struct ServerDescription {
    #[serde(rename = "ServerName")]
    server_name: String,
    #[serde(rename = "Manufacturer")]
    manufacturer: String,
    #[serde(rename = "ManufacturerVersion")]
    manufacturer_version: String,
    #[serde(rename = "Location")]
    location: String,
}

#[derive(Serialize)]
struct Device {
    #[serde(rename = "DeviceName")]
    device_name: String,
    #[serde(rename = "DeviceType")]
    device_type: String,
    #[serde(rename = "DeviceNumber")]
    device_number: u32,
    #[serde(rename = "UniqueID")]
    unique_id: String,
}

async fn lowercase_path(
    mut req: hyper::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let uri = req.uri().clone();
    let mut parts = uri.into_parts();
    if let Some(ref path_and_query) = parts.path_and_query {
        let new_path = path_and_query.path().to_lowercase();
        let new_paq = if let Some(query) = path_and_query.query() {
            format!("{}?{}", new_path, query)
        } else {
            new_path
        };
        if let Ok(new_paq) = new_paq.parse() {
            parts.path_and_query = Some(new_paq);
        }
    }
    if let Ok(new_uri) = hyper::Uri::from_parts(parts) {
        *req.uri_mut() = new_uri;
    }
    next.run(req).await
}

async fn management_handler(
    Path(path): Path<String>,
    params: Option<Query<AlpacaParams>>,
) -> impl IntoResponse {
    let lower_path = path.to_lowercase();
    let client_id = params.map(|Query(p)| p.client_transaction_id).unwrap_or(0);
    info!("Alpaca Management GET: {}", lower_path);
    if lower_path == "apiversions" || lower_path == "/apiversions" {
        return Json(AlpacaResponse::success(client_id, Some(vec![1]))).into_response();
    } else if lower_path == "v1/description" || lower_path == "/v1/description" {
        return Json(AlpacaResponse::success(
            client_id,
            Some(ServerDescription {
                server_name: "Olive Engine".to_string(),
                manufacturer: "Omair Kamil".to_string(),
                manufacturer_version: "1.0".to_string(),
                location: "Unknown".to_string(),
            }),
        ))
        .into_response();
    } else if lower_path == "v1/configureddevices" || lower_path == "/v1/configureddevices" {
        return Json(AlpacaResponse::success(
            client_id,
            Some(vec![Device {
                device_name: "Telescope".to_string(),
                device_type: "Telescope".to_string(),
                device_number: 0,
                unique_id: "7B228B0A-5A0D-48E0-8025-E93C355EBE28".to_string(),
            }]),
        ))
        .into_response();
    }

    (hyper::StatusCode::NOT_FOUND, "Not Found").into_response()
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum AlignmentMode {
    AltAz = 0,
    Polar = 1,
    GermanPolar = 2,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum EquatorialSystem {
    Other = 0,
    J2000 = 1,
    JNow = 2,
}

async fn configured_devices() -> Json<AlpacaResponse<Vec<Device>>> {
    Json(AlpacaResponse::success(
        0,
        Some(vec![Device {
            device_name: "Telescope".to_string(),
            device_type: "Telescope".to_string(),
            device_number: 0,
            unique_id: "olive-engine-telescope-0".to_string(),
        }]),
    ))
}

async fn handle_get(
    State(server): State<AlpacaServer>,
    Path(method): Path<String>,
    params: Option<Query<AlpacaParams>>,
) -> impl IntoResponse {
    let client_id = params.map(|Query(p)| p.client_transaction_id).unwrap_or(0);
    info!("Alpaca GET: {}", method);
    match method.to_lowercase().as_str() {
        "connected" => Json(AlpacaResponse::success(client_id, Some(true))).into_response(),
        "name" => Json(AlpacaResponse::success(
            client_id,
            Some("Olive Engine Alpaca"),
        ))
        .into_response(),
        "description" => Json(AlpacaResponse::success(
            client_id,
            Some("Olive Engine Telescope Server"),
        ))
        .into_response(),
        "driverinfo" => {
            Json(AlpacaResponse::success(client_id, Some("Olive Engine"))).into_response()
        }
        "driverversion" => Json(AlpacaResponse::success(client_id, Some("1.0"))).into_response(),
        "interfaceversion" => Json(AlpacaResponse::success(client_id, Some(3))).into_response(),
        "supportedactions" => Json(AlpacaResponse::success(
            client_id,
            Some(Vec::<String>::new()),
        ))
        .into_response(),
        "equatorialsystem" => Json(AlpacaResponse::success(
            client_id,
            Some(EquatorialSystem::J2000 as i32),
        ))
        .into_response(),
        "alignmentmode" => Json(AlpacaResponse::success(
            client_id,
            Some(AlignmentMode::Polar as i32),
        ))
        .into_response(),
        "canabortslew" => Json(AlpacaResponse::success(client_id, Some(true))).into_response(),
        "cansetrightascension" => {
            Json(AlpacaResponse::success(client_id, Some(true))).into_response()
        }
        "cansetdeclination" => Json(AlpacaResponse::success(client_id, Some(true))).into_response(),
        "canslewasync" => Json(AlpacaResponse::success(client_id, Some(true))).into_response(),
        "cansync" => Json(AlpacaResponse::success(client_id, Some(true))).into_response(),
        "canmoveaxis" => Json(AlpacaResponse::success(client_id, Some(false))).into_response(),
        "cansettracking" => Json(AlpacaResponse::success(client_id, Some(false))).into_response(),
        "rightascension" => {
            let bits = server.shared_position.telescope_ra.load(Ordering::Relaxed);
            let ra_deg = f64::from_bits(bits);
            let dec_bits = server.shared_position.telescope_dec.load(Ordering::Relaxed);
            server
                .shared_position
                .snapshot_dec
                .store(Some(f64::from_bits(dec_bits)), Ordering::Relaxed);
            Json(AlpacaResponse::success(client_id, Some(ra_deg / 15.0))).into_response()
        }
        "declination" => {
            let dec_deg = if let Some(snap_dec) =
                server.shared_position.snapshot_dec.load(Ordering::Relaxed)
            {
                server
                    .shared_position
                    .snapshot_dec
                    .store(None, Ordering::Relaxed);
                snap_dec
            } else {
                f64::from_bits(server.shared_position.telescope_dec.load(Ordering::Relaxed))
            };
            Json(AlpacaResponse::success(client_id, Some(dec_deg))).into_response()
        }
        "targetrightascension" => {
            if let Some(ra_deg) = server.shared_position.target_ra.load(Ordering::Relaxed) {
                Json(AlpacaResponse::success(client_id, Some(ra_deg / 15.0))).into_response()
            } else {
                Json(AlpacaResponse::<f64>::error(client_id, "Target RA not set")).into_response()
            }
        }
        "targetdeclination" => {
            if let Some(dec_deg) = server.shared_position.target_dec.load(Ordering::Relaxed) {
                Json(AlpacaResponse::success(client_id, Some(dec_deg))).into_response()
            } else {
                Json(AlpacaResponse::<f64>::error(
                    client_id,
                    "Target Dec not set",
                ))
                .into_response()
            }
        }
        "slewing" => {
            let is_slewing =
                server.shared_position.target_op.load(Ordering::Relaxed) == Some(TargetOp::Slew);
            Json(AlpacaResponse::success(client_id, Some(is_slewing))).into_response()
        }
        "tracking" => Json(AlpacaResponse::success(client_id, Some(false))).into_response(),
        "sitelatitude" => {
            if let Some(lat) = server.shared_position.site_latitude.load(Ordering::Relaxed) {
                Json(AlpacaResponse::success(client_id, Some(lat))).into_response()
            } else {
                Json(AlpacaResponse::success(client_id, Some(0.0))).into_response()
            }
        }
        "sitelongitude" => {
            if let Some(lon) = server
                .shared_position
                .site_longitude
                .load(Ordering::Relaxed)
            {
                Json(AlpacaResponse::success(client_id, Some(lon))).into_response()
            } else {
                Json(AlpacaResponse::success(client_id, Some(0.0))).into_response()
            }
        }
        "siteelevation" => Json(AlpacaResponse::success(client_id, Some(0.0))).into_response(),
        "altitude" | "azimuth" | "aperturearea" | "aperturediameter" | "focallength"
        | "siderealtime" | "rightascensionrate" | "declinationrate" => {
            Json(AlpacaResponse::success(client_id, Some(0.0))).into_response()
        }
        "athome" | "ispulseguiding" => {
            Json(AlpacaResponse::success(client_id, Some(false))).into_response()
        }
        "sideofpier" | "trackingrate" => {
            Json(AlpacaResponse::success(client_id, Some(0))).into_response()
        }
        "trackingrates" => Json(AlpacaResponse::success(client_id, Some(vec![1]))).into_response(),
        "utcdate" => Json(AlpacaResponse::success(
            client_id,
            Some("2026-06-08T00:00:00.000Z"),
        ))
        .into_response(),
        m if m.starts_with("can") => {
            Json(AlpacaResponse::success(client_id, Some(false))).into_response()
        }
        _ => Json(AlpacaResponse::<()>::error(client_id, "Not implemented")).into_response(),
    }
}

async fn handle_put(
    State(server): State<AlpacaServer>,
    Path(method): Path<String>,
    params: Option<Form<AlpacaParams>>,
) -> impl IntoResponse {
    let p = params.map(|Form(p)| p).unwrap_or_else(|| AlpacaParams {
        client_transaction_id: 0,
        _client_id: 0,
        right_ascension: None,
        declination: None,
        _connected: None,
    });
    let client_id = p.client_transaction_id;
    info!("Alpaca PUT: {}", method);
    match method.to_lowercase().as_str() {
        "connected" => Json(AlpacaResponse::<()>::success(client_id, None)).into_response(),
        "targetrightascension" => {
            if let Some(ra_hours) = p.right_ascension {
                server
                    .shared_position
                    .target_ra
                    .store(Some(ra_hours * 15.0), Ordering::Relaxed);
                Json(AlpacaResponse::<()>::success(client_id, None)).into_response()
            } else {
                Json(AlpacaResponse::<()>::error(
                    client_id,
                    "Missing RightAscension",
                ))
                .into_response()
            }
        }
        "targetdeclination" => {
            if let Some(dec_deg) = p.declination {
                server
                    .shared_position
                    .target_dec
                    .store(Some(dec_deg), Ordering::Relaxed);
                Json(AlpacaResponse::<()>::success(client_id, None)).into_response()
            } else {
                Json(AlpacaResponse::<()>::error(
                    client_id,
                    "Missing Declination",
                ))
                .into_response()
            }
        }
        "slewtocoordinatesasync" => {
            if let (Some(ra_hours), Some(dec_deg)) = (p.right_ascension, p.declination) {
                server
                    .shared_position
                    .target_ra
                    .store(Some(ra_hours * 15.0), Ordering::Relaxed);
                server
                    .shared_position
                    .target_dec
                    .store(Some(dec_deg), Ordering::Relaxed);
                server
                    .shared_position
                    .target_op
                    .store(Some(TargetOp::Slew), Ordering::Relaxed);
                Json(AlpacaResponse::<()>::success(client_id, None)).into_response()
            } else {
                Json(AlpacaResponse::<()>::error(
                    client_id,
                    "Missing RightAscension or Declination",
                ))
                .into_response()
            }
        }
        "synctocoordinates" => {
            if let (Some(ra_hours), Some(dec_deg)) = (p.right_ascension, p.declination) {
                server
                    .shared_position
                    .target_ra
                    .store(Some(ra_hours * 15.0), Ordering::Relaxed);
                server
                    .shared_position
                    .target_dec
                    .store(Some(dec_deg), Ordering::Relaxed);
                server
                    .shared_position
                    .target_op
                    .store(Some(TargetOp::Sync), Ordering::Relaxed);
                Json(AlpacaResponse::<()>::success(client_id, None)).into_response()
            } else {
                Json(AlpacaResponse::<()>::error(
                    client_id,
                    "Missing RightAscension or Declination",
                ))
                .into_response()
            }
        }
        "abortslew" => {
            server
                .shared_position
                .target_op
                .store(Some(TargetOp::Abort), Ordering::Relaxed);
            Json(AlpacaResponse::<()>::success(client_id, None)).into_response()
        }
        "tracking" => {
            // Ignore but acknowledge
            Json(AlpacaResponse::<()>::success(client_id, None)).into_response()
        }
        "moveaxis" => {
            // Silently ignore
            Json(AlpacaResponse::<()>::success(client_id, None)).into_response()
        }
        _ => Json(AlpacaResponse::<()>::error(client_id, "Not implemented")).into_response(),
    }
}
