// Required Notice: Copyright (c) 2025 Omair Kamil
// See LICENSE file in root directory for license terms.
//
// References for LX200 command set include:
//    https://www.astro.louisville.edu/software/xmtel/archive/xmtel-indi-6.0/xmtel-6.0l/support/lx200/CommandSet.html
//    https://interactiveastronomy.com/lx-200gps_telescope_protocol_2010-10.pdf
//    https://skymtn.com/mapug-astronomy/ragreiner/LX200Commands.html
//

use std::sync::atomic::{AtomicI64, AtomicU8, AtomicU64, Ordering};
use std::{error::Error, f64::consts::TAU, sync::Arc, sync::OnceLock};

use async_trait::async_trait;
use chrono::{DateTime, Datelike, FixedOffset, Local, Utc};
use log::{debug, info, warn};
use memchr::memchr;
use nalgebra::Vector3;
use starfield::{precessionlib::compute_precession, time::Timescale};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
    net::TcpListener,
};

static TIMESCALE: OnceLock<Timescale> = OnceLock::new();

fn get_timescale() -> &'static Timescale {
    TIMESCALE.get_or_init(|| Timescale::default())
}

fn precess_coordinates(ra: f64, dec: f64, epoch_from_j: f64, epoch_to_j: f64) -> (f64, f64) {
    let ts = get_timescale();
    let epoch_from = ts.j(epoch_from_j);
    let epoch_to = ts.j(epoch_to_j);

    let p_from = compute_precession(epoch_from.tdb());
    let p_to = compute_precession(epoch_to.tdb());

    let rot_matrix = p_to * p_from.transpose();

    let ra_rad = ra.to_radians();
    let dec_rad = dec.to_radians();

    let v_initial = Vector3::new(
        dec_rad.cos() * ra_rad.cos(),
        dec_rad.cos() * ra_rad.sin(),
        dec_rad.sin(),
    );

    let v_precessed = rot_matrix * v_initial;

    let precessed_ra = v_precessed.y.atan2(v_precessed.x).rem_euclid(TAU);
    let precessed_dec = v_precessed.z.asin();

    (precessed_ra.to_degrees(), precessed_dec.to_degrees())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetOp {
    Sync,
    Slew,
    Abort,
}

#[derive(Debug)]
pub struct AtomicOptionF64(AtomicU64);

impl AtomicOptionF64 {
    pub fn new(val: Option<f64>) -> Self {
        let bits = match val {
            Some(v) => v.to_bits(),
            None => f64::NAN.to_bits(),
        };
        AtomicOptionF64(AtomicU64::new(bits))
    }

    pub fn load(&self, order: Ordering) -> Option<f64> {
        let bits = self.0.load(order);
        let val = f64::from_bits(bits);
        if val.is_nan() { None } else { Some(val) }
    }

    pub fn store(&self, val: Option<f64>, order: Ordering) {
        let bits = match val {
            Some(v) => v.to_bits(),
            None => f64::NAN.to_bits(),
        };
        self.0.store(bits, order);
    }
}

#[derive(Debug)]
pub struct AtomicOptionI64(AtomicI64);

impl AtomicOptionI64 {
    pub fn new(val: Option<i64>) -> Self {
        let bits = match val {
            Some(v) => v,
            None => i64::MIN,
        };
        AtomicOptionI64(AtomicI64::new(bits))
    }

    pub fn load(&self, order: Ordering) -> Option<i64> {
        let val = self.0.load(order);
        if val == i64::MIN { None } else { Some(val) }
    }

    pub fn store(&self, val: Option<i64>, order: Ordering) {
        let bits = match val {
            Some(v) => v,
            None => i64::MIN,
        };
        self.0.store(bits, order);
    }
}

#[derive(Debug)]
pub struct AtomicTargetOp(AtomicU8);

impl AtomicTargetOp {
    pub fn new(val: Option<TargetOp>) -> Self {
        AtomicTargetOp(AtomicU8::new(Self::to_u8(val)))
    }

    fn to_u8(val: Option<TargetOp>) -> u8 {
        match val {
            None => 0,
            Some(TargetOp::Sync) => 1,
            Some(TargetOp::Slew) => 2,
            Some(TargetOp::Abort) => 3,
        }
    }

    pub fn load(&self, order: Ordering) -> Option<TargetOp> {
        match self.0.load(order) {
            1 => Some(TargetOp::Sync),
            2 => Some(TargetOp::Slew),
            3 => Some(TargetOp::Abort),
            _ => None,
        }
    }

    pub fn store(&self, val: Option<TargetOp>, order: Ordering) {
        self.0.store(Self::to_u8(val), order);
    }
}

#[derive(Debug)]
pub struct SharedPosition {
    pub target_ra: AtomicOptionF64,
    pub target_dec: AtomicOptionF64,
    pub site_date_ms: AtomicOptionI64,
    pub site_latitude: AtomicOptionF64,
    pub site_longitude: AtomicOptionF64,
    pub target_op: AtomicTargetOp,

    // For reporting back to the client
    pub telescope_ra: AtomicU64,
    pub telescope_dec: AtomicU64,
}

impl Default for SharedPosition {
    fn default() -> Self {
        SharedPosition {
            target_ra: AtomicOptionF64::new(None),
            target_dec: AtomicOptionF64::new(None),
            site_date_ms: AtomicOptionI64::new(None),
            site_latitude: AtomicOptionF64::new(None),
            site_longitude: AtomicOptionF64::new(None),
            target_op: AtomicTargetOp::new(None),
            telescope_ra: AtomicU64::new(0f64.to_bits()),
            telescope_dec: AtomicU64::new(0f64.to_bits()),
        }
    }
}

#[async_trait]
pub trait Lx200Telescope {
    async fn serve_requests(&mut self) -> Result<(), Box<dyn Error + 'static>>;
}

pub struct Lx200WifiTelescope {
    controller: Arc<std::sync::RwLock<Lx200Controller>>,
    is_running: Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait]
impl Lx200Telescope for Lx200WifiTelescope {
    async fn serve_requests(&mut self) -> Result<(), Box<dyn Error + 'static>> {
        let listener = TcpListener::bind("0.0.0.0:4030").await?;
        info!(
            "Running LX200 server on: {}",
            listener.local_addr().unwrap()
        );

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    debug!("WiFi LX200 connection opened");
                    let _ = stream.set_nodelay(true);
                    let (reader, writer) = stream.into_split();
                    let controller = self.controller.clone();
                    let is_running = self.is_running.clone();
                    tokio::spawn(async move {
                        Lx200Controller::handle_connection(controller, reader, writer, is_running)
                            .await;
                        debug!("WiFi LX200 connection closed");
                    });
                }
                Err(e) => {
                    warn!("Failed to accept WiFi LX200 connection: {:?}", e);
                }
            }
        }
    }
}

impl Lx200WifiTelescope {
    pub fn new(
        shared_position: Arc<SharedPosition>,
        is_running: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Lx200WifiTelescope {
            controller: Arc::new(std::sync::RwLock::new(Lx200Controller::new(
                shared_position,
            ))),
            is_running,
        }
    }
}

#[derive(Default, Debug)]
struct Lx200Controller {
    shared_position: Arc<SharedPosition>,

    // Pending target coordinates (in jnow_epoch) for sync/slew
    target_ra: Option<f64>,
    target_dec: Option<f64>,

    // Pending time
    // Timezone format is sHH:MM as the offset from GMT
    timezone: Option<String>,
    // Time format is HH:MM:SS
    time: Option<String>,

    // Local time, possibly provided by the client
    datetime: DateTime<FixedOffset>,

    // Current site set by the client. These values are stored here in the same
    // format provided by the client, for possible retrieval by the client.
    // Latitude format: sDD:MM where s is +/-
    latitude: String,
    // Longitude format: DDD:MM where DDD is defined as degrees west of the
    //                   Prime Meridian, from 0-359
    longitude: String,

    // Current epoch to the closest tenth of a year
    jnow_epoch: f64,

    // Stellarium seems to use J2000 epoch. If Stellarium is connected do not
    // precess.
    is_stellarium: bool,
}

impl Lx200Controller {
    pub fn new(shared_position: Arc<SharedPosition>) -> Self {
        let dt = Local::now();
        let jnow = ((dt.year() as f64 + dt.ordinal0() as f64 / 365.0) * 10.0).round() / 10.0;
        debug!("Using now epoch: {}", jnow);
        Lx200Controller {
            shared_position,
            datetime: dt.with_timezone(dt.offset()),
            latitude: "+00:00".to_string(),
            longitude: "000:00".to_string(),
            jnow_epoch: jnow,
            target_ra: None,
            target_dec: None,
            timezone: None,
            time: None,
            is_stellarium: false,
        }
    }

    // Returns the command portion of the input string
    fn extract_command(s: &str) -> Option<&str> {
        if !s.starts_with(':') {
            return None;
        }

        for (index, c) in s.char_indices().skip(1) {
            if !c.is_alphabetic() {
                if index == 1 {
                    return None;
                }
                return Some(&s[1..index]);
            }
        }

        return Some(&s[1..]);
    }

    fn get_failure() -> String {
        "0".to_string()
    }

    fn get_success() -> String {
        "1".to_string()
    }

    fn convert_to_j2000(&self, ra: f64, dec: f64) -> (f64, f64) {
        if self.is_stellarium {
            return (ra, dec);
        }
        precess_coordinates(ra, dec, self.jnow_epoch, 2000.0)
    }

    fn convert_to_jnow(&self, ra: f64, dec: f64) -> (f64, f64) {
        if self.is_stellarium {
            return (ra, dec);
        }
        precess_coordinates(ra, dec, 2000.0, self.jnow_epoch)
    }

    fn get_ra(&self) -> String {
        let (tel_ra, tel_dec) = (
            f64::from_bits(self.shared_position.telescope_ra.load(Ordering::Relaxed)),
            f64::from_bits(self.shared_position.telescope_dec.load(Ordering::Relaxed)),
        );
        let (ra, _) = self.convert_to_jnow(tel_ra, tel_dec);
        let (h, m, s) = Self::to_hms((ra / 15.0).rem_euclid(24.0));
        format!("{h:02}:{m:02}:{s:02}#")
    }

    fn get_dec(&self) -> String {
        let (tel_ra, tel_dec) = (
            f64::from_bits(self.shared_position.telescope_ra.load(Ordering::Relaxed)),
            f64::from_bits(self.shared_position.telescope_dec.load(Ordering::Relaxed)),
        );
        let dec = {
            let (_, dec) = self.convert_to_jnow(tel_ra, tel_dec);
            dec
        };

        let (h, m, s) = Self::to_hms(dec);
        let sign = if dec < 0.0 && (h != 0 || m != 0 || s != 0) {
            "-"
        } else {
            "+"
        };

        format!("{sign}{h:02}*{m:02}'{s:02}#")
    }

    fn set_target_ra(&mut self, cmd: &str) -> String {
        if cmd.len() < 11 {
            warn!("Unexpected ra length, cmd: {}", cmd);
            return Self::get_failure();
        }
        let hours = Self::parse_coordinates(&cmd[3..5], &cmd[6..8], &cmd[9..11]);
        match hours {
            Some(n) => {
                self.target_ra = Some(n * 15.0);
                Self::get_success()
            }
            None => Self::get_failure(),
        }
    }

    fn set_target_dec(&mut self, cmd: &str) -> String {
        if cmd.len() < 11 {
            warn!("Unexpected dec length, cmd: {}", cmd);
            return Self::get_failure();
        }
        let degrees = Self::parse_coordinates(&cmd[3..6], &cmd[7..9], &cmd[10..12]);
        match degrees {
            Some(n) => {
                self.target_dec = Some(n);
                Self::get_success()
            }
            None => Self::get_failure(),
        }
    }

    fn slew(&mut self) -> String {
        if let (Some(ra), Some(dec)) = (self.target_ra.take(), self.target_dec.take()) {
            let (j2000_ra, j2000_dec) = self.convert_to_j2000(ra, dec);
            debug!("Slewing to {}, {}", j2000_ra, j2000_dec);
            self.shared_position
                .target_ra
                .store(Some(j2000_ra), Ordering::Relaxed);
            self.shared_position
                .target_dec
                .store(Some(j2000_dec), Ordering::Relaxed);
            self.shared_position
                .target_op
                .store(Some(TargetOp::Slew), Ordering::Relaxed);
            "0".to_string()
        } else {
            "1No object#".to_string()
        }
    }

    fn sync(&mut self) -> String {
        if let (Some(ra), Some(dec)) = (self.target_ra.take(), self.target_dec.take()) {
            let (j2000_ra, j2000_dec) = self.convert_to_j2000(ra, dec);
            debug!("Syncing to {}, {}", j2000_ra, j2000_dec);
            self.shared_position
                .target_ra
                .store(Some(j2000_ra), Ordering::Relaxed);
            self.shared_position
                .target_dec
                .store(Some(j2000_dec), Ordering::Relaxed);
            self.shared_position
                .target_op
                .store(Some(TargetOp::Sync), Ordering::Relaxed);
        }
        " M31 EX GAL MAG 3.5 SZ178.0'#".to_string()
    }

    fn abort(&mut self) {
        self.shared_position
            .target_op
            .store(Some(TargetOp::Abort), Ordering::Relaxed);
        debug!("Stopping slew");
    }

    fn set_latitude(&mut self, cmd: &str) -> String {
        if cmd.len() < 9 {
            warn!("Unexpected lat length, cmd: {}", cmd);
            return Self::get_failure();
        }
        let location = Self::parse_location(&cmd[3..6], &cmd[7..9]);
        match location {
            Some(n) => {
                info!("Set observer latitude");
                self.latitude = cmd[3..9].to_string();
                self.shared_position
                    .site_latitude
                    .store(Some(n), Ordering::Relaxed);
                Self::get_success()
            }
            None => Self::get_failure(),
        }
    }

    fn set_longitude(&mut self, cmd: &str) -> String {
        if cmd.len() < 9 {
            warn!("Unexpected lon length, cmd: {}", cmd);
            return Self::get_failure();
        }
        let location = Self::parse_location(&cmd[3..6], &cmd[7..9]);
        match location {
            Some(n) => {
                self.longitude = cmd[3..9].to_string();
                let longitude = if n > 180.0 { 360.0 - n } else { -n };
                info!("Set observer longitude");
                self.shared_position
                    .site_longitude
                    .store(Some(longitude), Ordering::Relaxed);
                Self::get_success()
            }
            None => Self::get_failure(),
        }
    }

    fn set_hours_to_utc(&mut self, cmd: &str) -> String {
        if cmd.len() < 8 {
            warn!("Unexpected tz length, cmd: {}", cmd);
            return Self::get_failure();
        }
        let sign = if &cmd[3..4] == "-" { "+" } else { "-" };
        let mut timezone = format!("{}{}", sign, &cmd[4..6]);
        match &cmd[6..8] {
            ".0" => timezone.push_str("00"),
            ".2" => timezone.push_str("15"),
            ".5" => timezone.push_str("30"),
            ".8" => timezone.push_str("45"),
            _ => {
                warn!("Error parsing timezone: {}", &cmd[3..8]);
                return Self::get_failure();
            }
        }
        self.timezone = Some(timezone);
        Self::get_success()
    }

    fn set_time(&mut self, cmd: &str) -> String {
        if cmd.len() < 11 {
            warn!("Unexpected time length, cmd: {}", cmd);
            return Self::get_failure();
        }
        self.time = Some(cmd[3..11].to_string());
        Self::get_success()
    }

    fn set_date(&mut self, cmd: &str) -> String {
        if cmd.len() < 11 {
            warn!("Unexpected date length, cmd: {}", cmd);
            return Self::get_failure();
        }
        if let (Some(timezone), Some(time)) = (self.timezone.take(), self.time.take()) {
            const FMT: &str = "%z%H:%M:%S%D";
            let dtstr = timezone.clone() + &time + &cmd[3..11];
            let datetime: Result<DateTime<FixedOffset>, _> = DateTime::parse_from_str(&dtstr, FMT);
            match datetime {
                Ok(dt) => {
                    info!("Set date/time to {}", dt);
                    self.datetime = dt;
                    self.jnow_epoch =
                        ((dt.year() as f64 + dt.ordinal0() as f64 / 365.0) * 10.0).round() / 10.0;
                    self.shared_position.site_date_ms.store(
                        Some(dt.with_timezone(&Utc).timestamp_millis()),
                        Ordering::Relaxed,
                    );
                    return "1Updating Planetary Data# #".to_string();
                }
                Err(e) => {
                    warn!("Error parsing date/time: {:?}", e);
                }
            }
        }
        Self::get_failure()
    }

    fn get_hours_to_utc(&self) -> String {
        let timezone = self.datetime.format("%z").to_string();
        let sign = if &timezone[0..1] == "-" { "+" } else { "-" };
        let hours = &timezone[1..3];
        let partial = match &timezone[3..] {
            "15" => ".2",
            "30" => ".5",
            "45" => ".8",
            _ => ".0",
        };
        format!("{sign}{hours}{partial}#")
    }

    fn get_time(&self) -> String {
        self.datetime.format("%H:%M:%S#").to_string()
    }

    fn get_date(&self) -> String {
        self.datetime.format("%D#").to_string()
    }

    fn get_latitude(&self) -> String {
        format!("{0}#", self.latitude)
    }

    fn get_longitude(&self) -> String {
        format!("{0}#", self.longitude)
    }

    fn get_distance_bars(&self) -> String {
        if self.shared_position.target_op.load(Ordering::Relaxed) == Some(TargetOp::Slew) {
            "\x7f#".to_string()
        } else {
            "#".to_string()
        }
    }

    fn to_hms(n: f64) -> (i64, i64, i64) {
        let n_abs = n.abs();
        let mut hours = n_abs.trunc() as i64;
        let h_rem = n_abs.fract() * 60.0;
        let mut minutes = h_rem.trunc() as i64;
        let m_rem = h_rem.fract() * 60.0;
        let mut seconds = m_rem.round() as i64;
        if seconds == 60 {
            seconds = 0;
            minutes += 1;
            if minutes == 60 {
                minutes = 0;
                hours += 1;
            }
        }
        if hours == 24 {
            hours = 0;
        }
        if n < 0.0 {
            hours = -hours;
        }
        (hours, minutes, seconds)
    }

    fn parse_coordinates(d: &str, m: &str, s: &str) -> Option<f64> {
        let degrees: Result<i32, _> = d.parse();
        let is_negative = match degrees {
            Err(e) => {
                warn!("Error parsing degrees: {:?}", e);
                return None;
            }
            Ok(deg) => deg < 0,
        };
        let minutes: Result<i32, _> = m.parse();
        match minutes {
            Err(e) => {
                warn!("Error parsing minutes: {:?}", e);
                return None;
            }
            Ok(_) => {}
        }
        let seconds: Result<i32, _> = s.parse();
        match seconds {
            Err(e) => {
                warn!("Error parsing seconds: {:?}", e);
                return None;
            }
            Ok(_) => {}
        }
        let deg = degrees.unwrap().abs() as f64
            + minutes.unwrap() as f64 / 60.0
            + seconds.unwrap() as f64 / 3600.00;
        Some(if is_negative { -deg } else { deg })
    }

    fn parse_location(deg: &str, min: &str) -> Option<f64> {
        Self::parse_coordinates(deg, min, "0")
    }

    async fn handle_connection<R, W>(
        controller: Arc<std::sync::RwLock<Self>>,
        reader: R,
        mut writer: W,
        is_running: Arc<std::sync::atomic::AtomicBool>,
    ) where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut buf_reader = BufReader::new(reader);
        let mut buffer = Vec::new();

        const ACK_CMD: [u8; 1] = [0x06];
        const SKY_SAFARI_INIT: [u8; 3] = [b'$', b'$', b'$'];
        const HASH: u8 = b'#';

        loop {
            if !is_running.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            buffer.clear();

            loop {
                let chunk = match tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    buf_reader.fill_buf(),
                )
                .await
                {
                    Ok(Ok(c)) => c,
                    Ok(Err(e)) => {
                        match e.kind() {
                            std::io::ErrorKind::ConnectionReset
                            | std::io::ErrorKind::BrokenPipe
                            | std::io::ErrorKind::ConnectionAborted => {
                                debug!("Client disconnected: {:?}", e);
                            }
                            _ => {
                                warn!("Error reading from stream: {:?}", e);
                            }
                        }
                        return;
                    }
                    Err(_) => {
                        debug!("Connection timed out waiting for data");
                        return;
                    }
                };

                if chunk.is_empty() {
                    debug!("Client closed connection");
                    return;
                }

                if buffer.is_empty() {
                    if chunk[0] == ACK_CMD[0] {
                        buf_reader.consume(1);
                        debug!("Received ack command");
                        controller.write().unwrap().is_stellarium = true;
                        if let Err(e) = Self::process_output(&mut writer, "A".to_string()).await {
                            warn!("Failed to send ack response: {:?}", e);
                            return;
                        }
                        continue;
                    }
                    if chunk.len() >= 3 && &chunk[0..3] == SKY_SAFARI_INIT {
                        buf_reader.consume(3);
                        debug!("Received 35092$");
                        continue;
                    }
                }

                let (found, consumed) = match memchr(HASH, chunk) {
                    Some(i) => {
                        buffer.extend_from_slice(&chunk[..=i]);
                        (true, i + 1)
                    }
                    None => {
                        buffer.extend_from_slice(chunk);
                        (false, chunk.len())
                    }
                };

                buf_reader.consume(consumed);
                if found {
                    break;
                }

                if buffer == ACK_CMD {
                    debug!("Received ack command");
                    buffer.clear();
                    controller.write().unwrap().is_stellarium = true;
                    if let Err(e) = Self::process_output(&mut writer, "A".to_string()).await {
                        warn!("Failed to send ack response: {:?}", e);
                        return;
                    }
                } else if buffer == SKY_SAFARI_INIT {
                    debug!("Received 35092$");
                    buffer.clear();
                }
            }

            if buffer.len() == 1 {
                continue;
            }
            let result = controller.write().unwrap().process_input(&buffer);
            match result {
                Some(res) => {
                    debug!("Writing to client: {}", res);

                    if let Err(e) = Self::process_output(&mut writer, res).await {
                        warn!("Failed to send data to client: {:?}", e);
                        return;
                    }
                }
                None => {}
            }
        }
    }

    async fn process_output<W>(writer: &mut W, output: String) -> std::io::Result<()>
    where
        W: AsyncWrite + Unpin,
    {
        writer.write_all(output.as_bytes()).await?;
        writer.flush().await?;
        Ok(())
    }

    fn process_input(&mut self, buffer: &[u8]) -> Option<String> {
        let input = String::from_utf8_lossy(&buffer);
        let in_data = input.as_ref();
        if let Some(command) = Self::extract_command(in_data) {
            match command {
                "CM" => {
                    debug!("Received sync command");
                    Some(self.sync())
                }
                "D" => {
                    debug!("Received distance bars command");
                    Some(self.get_distance_bars())
                }
                "GC" => {
                    debug!("Received get date command");
                    Some(self.get_date())
                }
                "GD" => {
                    debug!("Received get declination command");
                    Some(self.get_dec())
                }
                "GG" => {
                    debug!("Received get hours to UTC command");
                    Some(self.get_hours_to_utc())
                }
                "GL" => {
                    debug!("Received get time command");
                    Some(self.get_time())
                }
                "GR" => {
                    debug!("Received get ra command");
                    Some(self.get_ra())
                }
                "GVD" => {
                    debug!("Received get firmware date command");
                    Some("Nov 14 2025#".to_string())
                }
                "GVN" => {
                    debug!("Received get firmware version command");
                    Some("01.0#".to_string())
                }
                "GVP" => {
                    debug!("Received get product command");
                    Some("Lx200Server#".to_string())
                }
                "GVT" => {
                    debug!("Received get firmware time command");
                    Some("23:00:00#".to_string())
                }
                "GW" => {
                    debug!("Received get status command");
                    Some("AT1".to_string())
                }
                "Gg" => {
                    debug!("Received get longitude command");
                    Some(self.get_longitude())
                }
                "Gt" => {
                    debug!("Received get latitude command");
                    Some(self.get_latitude())
                }
                "MS" => {
                    debug!("Received slew to object command");
                    Some(self.slew())
                }
                "Me" | "Mn" | "Ms" | "Mw" => {
                    debug!("Received movement command");
                    None
                }
                "Q" => {
                    debug!("Received abort command");
                    self.abort();
                    None
                }
                "Qe" | "Qn" | "Qs" | "Qw" => {
                    debug!("Received stop movement command");
                    None
                }
                "RS" => {
                    debug!("Received set rate to slew command");
                    None
                }
                "SC" => {
                    debug!("Received set date command: {}", in_data);
                    Some(self.set_date(in_data))
                }
                "SG" => {
                    debug!("Received set hours to UTC command: {}", in_data);
                    Some(self.set_hours_to_utc(in_data))
                }
                "SL" => {
                    debug!("Received set time command: {}", in_data);
                    Some(self.set_time(in_data))
                }
                "Sd" => {
                    debug!("Received set declination command: {}", in_data);
                    Some(self.set_target_dec(in_data))
                }
                "Sg" => {
                    debug!("Received set longitude command: {}", in_data);
                    Some(self.set_longitude(in_data))
                }
                "Sr" => {
                    debug!("Received set ra command: {}", in_data);
                    Some(self.set_target_ra(in_data))
                }
                "St" => {
                    debug!("Received set latitude command: {}", in_data);
                    Some(self.set_latitude(in_data))
                }
                "Sw" => {
                    debug!("Received set max slew rate command: {}", in_data);
                    Some(Self::get_success())
                }
                "U" => {
                    debug!("Received precision toggle command");
                    None
                }
                _ => {
                    warn!("Unknown command: {}", in_data);
                    None
                }
            }
        } else {
            warn!("Malformed input: {}", in_data);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate approx;

    use approx::assert_abs_diff_eq;
    use chrono::TimeZone;

    use super::*;

    async fn setup_controller() -> (Lx200Controller, Arc<RwLock<SharedPosition>>) {
        let shared_position = Arc::new(RwLock::new(SharedPosition::default()));
        let controller = Lx200Controller::new(shared_position.clone());
        (controller, shared_position)
    }

    // --- Utility Function Tests ---

    fn assert_approx_eq(x: f64, y: f64) {
        assert_abs_diff_eq!(x, y, epsilon = 0.0001);
    }

    #[test]
    fn test_extract_command() {
        assert_eq!(
            Lx200Controller::extract_command(":GR#").as_deref(),
            Some("GR")
        );
        assert_eq!(
            Lx200Controller::extract_command(":RS#:GR#").as_deref(),
            Some("RS")
        );
        assert_eq!(
            Lx200Controller::extract_command(":Sd+00*00:00#").as_deref(),
            Some("Sd")
        );
        assert_eq!(
            Lx200Controller::extract_command(":Sr00:00:00#").as_deref(),
            Some("Sr")
        );
        // Invalid cases - commands start with : and contain letters
        assert_eq!(Lx200Controller::extract_command(":123#"), None);
        assert_eq!(Lx200Controller::extract_command(":#"), None);
        assert_eq!(Lx200Controller::extract_command("GR#"), None);
    }

    #[test]
    fn test_to_hms() {
        assert_eq!(Lx200Controller::to_hms(0.0), (0, 0, 0));
        assert_eq!(Lx200Controller::to_hms(1.5), (1, 30, 0));
        assert_eq!(Lx200Controller::to_hms(-10.5083), (-10, 30, 30));
        assert_eq!(Lx200Controller::to_hms(23.99972), (23, 59, 59));
        // Floating point math is hard
        assert_eq!(Lx200Controller::to_hms(10.1), (10, 6, 0));
        // Make sure we dont end up with 60 minutes or seconds
        assert_eq!(Lx200Controller::to_hms(23.999859), (23, 59, 59));
        // Checking 2 possible values due to floating point imprecision
        let v = Lx200Controller::to_hms(23.999861);
        assert!(v == (23, 59, 59) || v == (0, 0, 0), "Incorrect: {:?}", v);
        assert_eq!(Lx200Controller::to_hms(23.999862), (0, 0, 0));
        assert_eq!(Lx200Controller::to_hms(-23.999999), (0, 0, 0));
    }

    #[test]
    fn test_parse_coordinates() {
        assert_approx_eq(
            Lx200Controller::parse_coordinates("01", "30", "00").unwrap(),
            1.5,
        );
        assert_approx_eq(
            Lx200Controller::parse_coordinates("10", "30", "30").unwrap(),
            10.5083,
        );
        assert_approx_eq(
            Lx200Controller::parse_coordinates("-15", "30", "45").unwrap(),
            -15.5125,
        );
        // Invalid
        assert_eq!(Lx200Controller::parse_coordinates("xx", "30", "00"), None);
        assert_eq!(Lx200Controller::parse_coordinates("01", "xx", "00"), None);
        assert_eq!(Lx200Controller::parse_coordinates("01", "30", "xx"), None);
    }

    #[test]
    fn test_parse_location() {
        assert_eq!(Lx200Controller::parse_location("01", "30"), Some(1.5));
        assert_approx_eq(
            Lx200Controller::parse_location("+37", "46").unwrap(),
            37.7667,
        );
        assert_approx_eq(Lx200Controller::parse_location("-90", "30").unwrap(), -90.5);
        assert_eq!(Lx200Controller::parse_location("a0", "00"), None);
    }

    // --- Get RA/Dec Command Tests ---

    #[tokio::test]
    async fn test_get_ra_and_dec() {
        let (mut controller, position_arc) = setup_controller().await;
        // Set epoch to J2000 for no change
        controller.jnow_epoch = 2000.0;

        let ra_j2000 = 1.5 * 15.0;
        let dec_j2000 = 30.5;

        // Set position in a block to release the lock at the end
        {
            let mut locked_position = position_arc.write().unwrap();
            locked_position
                .telescope_ra
                .store(ra_j2000.to_bits(), Ordering::Relaxed);
            locked_position
                .telescope_dec
                .store(dec_j2000.to_bits(), Ordering::Relaxed);
        }

        let result_ra = controller.process_input(b":GR#");
        assert_eq!(result_ra.as_deref(), Some("01:30:00#"));

        let result_dec = controller.process_input(b":GD#");
        assert_eq!(result_dec.as_deref(), Some("+30*30'00#"));
    }

    // --- Sync/Slew Flow Tests ---

    #[tokio::test]
    async fn test_set_ra_dec_slew_abort() {
        let (mut controller, position_arc) = setup_controller().await;
        // Set epoch to J2000 for no change
        controller.jnow_epoch = 2000.0;

        let set_ra = controller.process_input(b":Sr10:30:00#");
        assert_eq!(set_ra.as_deref(), Some("1"));
        assert_approx_eq(controller.target_ra.unwrap(), 157.5);

        let set_dec = controller.process_input(b":Sd-15*30:00#");
        assert_eq!(set_dec.as_deref(), Some("1"));
        assert_approx_eq(controller.target_dec.unwrap(), -15.5);

        let slew_result = controller.process_input(b":MS#");
        // 0 is success for slew
        assert_eq!(slew_result.as_deref(), Some("0"));

        // Check shared position state
        {
            let locked_position = position_arc.read().unwrap();
            assert_approx_eq(locked_position.target_ra.unwrap(), 157.5);
            assert_approx_eq(locked_position.target_dec.unwrap(), -15.5);
            assert_eq!(locked_position.target_op, Some(TargetOp::Slew));
        }

        let abort = controller.process_input(b":Q#");
        assert_eq!(abort, None);

        {
            let locked_position = position_arc.read().unwrap();
            assert!(locked_position.target_op.is_none());
        }

        // Slew should fail if no target
        let slew_fail_result = controller.process_input(b":MS#");
        assert_eq!(slew_fail_result.as_deref(), Some("1No object#"));
    }

    #[tokio::test]
    async fn test_set_ra_dec_sync() {
        let (mut controller, position_arc) = setup_controller().await;
        // Set epoch to J2000 for no change
        controller.jnow_epoch = 2000.0;

        // Even with no target we should still return canned result
        let mut sync = controller.process_input(b":CM#");
        assert_eq!(sync.as_deref(), Some(" M31 EX GAL MAG 3.5 SZ178.0'#"));

        controller.process_input(b":Sr01:00:00#");
        controller.process_input(b":Sd+20*00:00#");

        sync = controller.process_input(b":CM#");
        assert_eq!(sync.as_deref(), Some(" M31 EX GAL MAG 3.5 SZ178.0'#"));

        let locked_position = position_arc.read().unwrap();
        assert_approx_eq(locked_position.target_ra.unwrap(), 15.0);
        assert_approx_eq(locked_position.target_dec.unwrap(), 20.0);
        assert_eq!(locked_position.target_op, Some(TargetOp::Sync));
    }

    // --- Location and Date Tests ---
    #[tokio::test]
    async fn test_set_get_location() {
        let (mut controller, position_arc) = setup_controller().await;

        let set_lat_n = controller.process_input(b":St+37:46#");
        assert_eq!(set_lat_n.as_deref(), Some("1"));
        assert_eq!(controller.latitude, "+37:46");
        let lat_n = position_arc.read().unwrap().site_latitude.unwrap();
        assert_approx_eq(lat_n, 37.7667);

        let set_lat_s = controller.process_input(b":St-37:46#");
        assert_eq!(set_lat_s.as_deref(), Some("1"));
        assert_eq!(controller.latitude, "-37:46");
        let lat_s = position_arc.read().unwrap().site_latitude.unwrap();
        assert_approx_eq(lat_s, -37.7667);

        // Less than 180 is West for longitude
        let set_lon_w = controller.process_input(b":Sg122:25#");
        assert_eq!(set_lon_w.as_deref(), Some("1"));
        assert_eq!(controller.longitude, "122:25");
        let lon_w = position_arc.read().unwrap().site_longitude.unwrap();
        assert_approx_eq(lon_w, -122.4167);

        // Longitude values are > 180 for East
        let set_lon_e = controller.process_input(b":Sg300:00#");
        assert_eq!(set_lon_e.as_deref(), Some("1"));
        assert_eq!(controller.longitude, "300:00");
        let lon_e = position_arc.read().unwrap().site_longitude.unwrap();
        // 360 - 300 = 60.0
        assert_approx_eq(lon_e, 60.0);

        let get_lat_result = controller.process_input(b":Gt#");
        assert_eq!(get_lat_result.as_deref(), Some("-37:46#"));
        let get_lon_result = controller.process_input(b":Gg#");
        assert_eq!(get_lon_result.as_deref(), Some("300:00#"));
    }

    #[tokio::test]
    async fn test_set_get_date_time_timezone() {
        let (mut controller, position_arc) = setup_controller().await;

        // Since LX200 protocol involves passing hours to UTC, flip the sign
        let set_time_to_utc = controller.process_input(b":SG+08.0#");
        assert_eq!(set_time_to_utc.as_deref(), Some("1"));
        assert_eq!(controller.timezone, Some("-0800".to_string()));

        let set_time = controller.process_input(b":SL10:30:00#");
        assert_eq!(set_time.as_deref(), Some("1"));
        assert_eq!(controller.time, Some("10:30:00".to_string()));

        let set_date = controller.process_input(b":SC11/15/25#");
        assert_eq!(set_date.as_deref(), Some("1Updating Planetary Data# #"));

        // Pending time and timezone should be consumed
        assert!(controller.timezone.is_none());
        assert!(controller.time.is_none());

        // Check the controllers time
        let expected_dt = FixedOffset::west_opt(8 * 3600)
            .unwrap()
            .with_ymd_and_hms(2025, 11, 15, 10, 30, 0)
            .unwrap();
        assert_eq!(controller.datetime, expected_dt);

        // Check the SharedPositions time
        {
            let locked_position = position_arc.read().unwrap();
            assert_eq!(
                locked_position.site_date,
                Some(expected_dt.with_timezone(&Utc))
            );
        }

        assert_eq!(
            controller.process_input(b":GL#").as_deref(),
            Some("10:30:00#")
        );
        assert_eq!(
            controller.process_input(b":GC#").as_deref(),
            Some("11/15/25#")
        );
        // Sign flip here as well to match the LX200 protocol
        assert_eq!(controller.process_input(b":GG#").as_deref(), Some("+08.0#"));
    }

    // --- Tests for Remaining Get Commands ---

    #[tokio::test]
    async fn test_canned_get_commands() {
        let (mut controller, _) = setup_controller().await;
        // Firmware date should be something like "Nov 14 2025#"
        assert_eq!(controller.process_input(b":GVD#").unwrap().len(), 12);
        // Firmware version should be something like "01.0#"
        assert_eq!(controller.process_input(b":GVN#").unwrap().len(), 5);
        // Model shouldnt change
        assert_eq!(
            controller.process_input(b":GVP#").as_deref(),
            Some("Lx200Server#")
        );
        // Firmware time should be something like "23:00:00#"
        assert_eq!(controller.process_input(b":GVT#").unwrap().len(), 9);
        // Status command indicates A for Alt-Az, T for tracking state, and 1
        // for 1-star aligned
        assert_eq!(controller.process_input(b":GW#").as_deref(), Some("AT1"));
    }

    // --- Special Case Tests ---

    #[tokio::test]
    async fn test_get_distance_bars() {
        let (mut controller, position_arc) = setup_controller().await;

        // Set position in a block to release the lock at the end
        {
            let mut locked_position = position_arc.write().unwrap();
            locked_position.target_op = None;
        }
        assert_eq!(controller.process_input(b":D#").as_deref(), Some("#"));

        // Set position in a block to release the lock at the end
        {
            let mut locked_position = position_arc.write().unwrap();
            locked_position.target_op = Some(TargetOp::Slew);
        }
        assert_eq!(controller.process_input(b":D#").as_deref(), Some("\x7f#"));
    }

    #[tokio::test]
    async fn test_handle_connection_ack() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (controller, _) = setup_controller().await;
        let controller_arc = Arc::new(RwLock::new(controller));
        let (client, server) = tokio::io::duplex(1024);
        let (server_read, server_write) = tokio::io::split(server);
        let (mut client_read, mut client_write) = tokio::io::split(client);

        let server_handle = tokio::spawn(async move {
            Lx200Controller::handle_connection(controller_arc.clone(), server_read, server_write)
                .await;
            controller_arc
        });

        client_write.write_all(b"\x06").await.unwrap();

        let mut buf = [0; 64];
        let len = client_read.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..len], b"A");

        // Shut down the client write half so the server sees EOF and exits
        client_write.shutdown().await.unwrap();

        let controller_res = server_handle.await.unwrap();
        assert!(controller_res.read().unwrap().is_stellarium);
    }

    #[tokio::test]
    async fn test_process_input_leading_hash() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut controller, _) = setup_controller().await;
        // Set epoch to J2000 for no change
        controller.jnow_epoch = 2000.0;
        let controller_arc = Arc::new(RwLock::new(controller));

        let (client, server) = tokio::io::duplex(1024);
        let (server_read, server_write) = tokio::io::split(server);
        let (mut client_read, mut client_write) = tokio::io::split(client);

        let server_handle = tokio::spawn(async move {
            Lx200Controller::handle_connection(controller_arc, server_read, server_write).await;
        });

        // Stellarium sends a leading #
        client_write.write_all(b"#:GR#").await.unwrap();

        let mut buf = [0; 64];
        let len = client_read.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..len], b"00:00:00#");

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_process_input_fragmented_command() {
        use futures::poll;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut controller, _) = setup_controller().await;
        // Set epoch to J2000 for no change
        controller.jnow_epoch = 2000.0;
        let controller_arc = Arc::new(RwLock::new(controller));

        let (client, server) = tokio::io::duplex(1024);
        let (server_read, server_write) = tokio::io::split(server);
        let (mut client_read, mut client_write) = tokio::io::split(client);

        let server_handle = tokio::spawn(async move {
            Lx200Controller::handle_connection(controller_arc, server_read, server_write).await;
        });

        // Send the first fragment of the command
        client_write.write_all(b"#:G").await.unwrap();

        // Yield execution to give the server task a chance to process the first
        // fragment
        tokio::task::yield_now().await;

        let mut buf = [0; 64];
        // Verify that no data is returned yet
        {
            let read_fut = client_read.read(&mut buf);
            tokio::pin!(read_fut);
            assert!(poll!(&mut read_fut).is_pending());
        } // Drop read_fut to allow reading again and checking the buffer

        assert_eq!(buf, [0; 64]);

        // Send the rest of the command
        client_write.write_all(b"R#").await.unwrap();
        let len = client_read.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..len], b"00:00:00#");

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_process_input_multiple_commands() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut controller, position_arc) = setup_controller().await;
        // Set epoch to J2000 for no change
        controller.jnow_epoch = 2000.0;
        let controller_arc = Arc::new(RwLock::new(controller));

        // Set position in a block to release the lock at the end
        {
            let mut locked_position = position_arc.write().unwrap();
            locked_position
                .telescope_ra
                .store(0.0f64.to_bits(), Ordering::Relaxed);
            locked_position
                .telescope_dec
                .store(0.0f64.to_bits(), Ordering::Relaxed);
        }

        let (client, server) = tokio::io::duplex(1024);
        let (server_read, server_write) = tokio::io::split(server);
        let (mut client_read, mut client_write) = tokio::io::split(client);

        let server_handle = tokio::spawn(async move {
            Lx200Controller::handle_connection(controller_arc, server_read, server_write).await;
        });

        // SkySafari in Bluetooth mode sets slew rate and gets RA together
        client_write.write_all(b":RS#:GR#").await.unwrap();
        let mut buf = [0; 64];
        let mut len = client_read.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..len], b"00:00:00#");

        // Make sure 2 commands that need responses get a concatenated response
        client_write.write_all(b":GR#:GD#").await.unwrap();
        buf = [0; 64];
        len = client_read.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..len], b"00:00:00#+00*00'00#");

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_stellarium_mode_no_precess() {
        let (mut controller, position_arc) = setup_controller().await;
        controller.jnow_epoch = 2026.0;
        // Toggle is_stellarium directly as the ack command is tested separately
        controller.is_stellarium = true;

        // Use 10h RA (150 degrees) and +45d Dec
        let internal_ra = 150.0;
        let internal_dec = 45.0;
        // Set position in a block to release the lock at the end
        {
            let mut locked_position = position_arc.write().unwrap();
            locked_position
                .telescope_ra
                .store(internal_ra.to_bits(), Ordering::Relaxed);
            locked_position
                .telescope_dec
                .store(internal_dec.to_bits(), Ordering::Relaxed);
        }

        // Ensure no precessing for telescope position
        let ra_result = controller.process_input(b":GR#");
        assert_eq!(ra_result.as_deref(), Some("10:00:00#"));
        let dec_result = controller.process_input(b":GD#");
        assert_eq!(dec_result.as_deref(), Some("+45*00'00#"));

        // Target: 20h RA (300 degrees), -30d Dec
        let set_ra_res = controller.process_input(b":Sr20:00:00#");
        assert_eq!(set_ra_res.as_deref(), Some("1"));
        let set_dec_res = controller.process_input(b":Sd-30*00:00#");
        assert_eq!(set_dec_res.as_deref(), Some("1"));
        let slew_res = controller.process_input(b":MS#");
        assert_eq!(slew_res.as_deref(), Some("0"));

        // Check the SharedPositions time
        // Ensure no precessing for target
        {
            let locked_position = position_arc.read().unwrap();
            assert_eq!(locked_position.target_op, Some(TargetOp::Slew));
            assert_approx_eq(locked_position.target_ra.unwrap(), 300.0);
            assert_approx_eq(locked_position.target_dec.unwrap(), -30.0);
        }
    }
}
