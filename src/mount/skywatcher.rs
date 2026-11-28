// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

//! Sky-Watcher SynScan Wi-Fi Motor Controller driver (`UDP 11880`, default `192.168.4.1`).
//!
//! Speaks the native Sky-Watcher ASCII motor-controller protocol (`:a`, `:b`, `:e`,
//! `:f`, `:F`, `:g`, `:G`, `:H`, `:I`, `:j`, `:J`, `:K`, `:M`) with 24-bit
//! little-endian hex framing, stale-packet draining, and UDP timeout retries.

use std::f64::consts::TAU;
use std::io::{Error, ErrorKind, Result as IoResult};
use std::net::{ToSocketAddrs, UdpSocket};
use std::str::from_utf8;
use std::thread::sleep;
use std::time::Duration;

use log::{info, trace};

use super::invalid_data;
use super::kinematic::{AxisUnwrapper, MIN_RATE_RAD_PER_SEC, RawAxisMount};

const UDP_TIMEOUT: Duration = Duration::from_millis(250);
const UDP_RETRIES: usize = 3;
const STOP_POLL_MAX: usize = 20;
const STOP_POLL_DELAY: Duration = Duration::from_millis(50);
const ENCODER_ORIGIN: i64 = 0x80_0000;
const SIDEREAL_RAD_PER_SEC: f64 = 7.292_115_855_3e-5;
const LOW_SPEED_RATE_LIMIT: f64 = 128.0 * SIDEREAL_RAD_PER_SEC;
const HIGH_SPEED_RAMP_MAX: u32 = 3_200;
const LOW_SPEED_RAMP_MAX: u32 = 200;

struct AxisParams {
    counts_per_rev: u32,
    timer_freq: u32,
    high_speed_ratio: u32,
    low_speed_goto_margin: u32,
    unwrapper: AxisUnwrapper,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AxisStatus {
    running: bool,
    rate_mode: bool,
    forward: bool,
    high_speed: bool,
    initialized: bool,
}

/// Blocking UDP client for Sky-Watcher SynScan Wi-Fi motor controllers.
pub(super) struct SkyWatcherMount {
    socket: UdpSocket,
    model_name: String,
    is_alt_az: bool,
    axes: [AxisParams; 2],
}

impl SkyWatcherMount {
    /// Connects to a Sky-Watcher SynScan Wi-Fi motor controller (`UDP 11880`) and queries axis parameters.
    pub(super) fn connect(host: &str, port: u16) -> IoResult<Self> {
        let mut last_err = None;
        for remote in (host, port).to_socket_addrs()?.take(2) {
            let bind_addr = if remote.is_ipv6() {
                "[::]:0"
            } else {
                "0.0.0.0:0"
            };
            let socket = match UdpSocket::bind(bind_addr) {
                Ok(s) => s,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };
            if let Err(e) = socket.connect(remote) {
                last_err = Some(e);
                continue;
            }
            socket.set_read_timeout(Some(UDP_TIMEOUT))?;
            socket.set_write_timeout(Some(UDP_TIMEOUT))?;

            match Self::handshake(socket) {
                Ok(m) => return Ok(m),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err
            .unwrap_or_else(|| Error::new(ErrorKind::AddrNotAvailable, "No UDP address resolved")))
    }

    fn handshake(socket: UdpSocket) -> IoResult<Self> {
        let ver_hex = talk_axis(&socket, 1, 'e', "")?;
        let (model_name, is_alt_az) = identify_mount(&ver_hex)?;

        let ax1 = init_axis(&socket, 1)?;
        let ax2 = init_axis(&socket, 2)?;

        info!(
            "Sky-Watcher handshake OK: {} (alt_az={}, cpr=[{}, {}], freq=[{}, {}])",
            model_name,
            is_alt_az,
            ax1.counts_per_rev,
            ax2.counts_per_rev,
            ax1.timer_freq,
            ax2.timer_freq
        );

        Ok(Self {
            socket,
            model_name,
            is_alt_az,
            axes: [ax1, ax2],
        })
    }

    fn read_axis_rad(&mut self, idx: usize) -> IoResult<f64> {
        let axis_num = (idx + 1) as u8;
        let reply = talk_axis(&self.socket, axis_num, 'j', "")?;
        let raw = decode_le24(&reply)?;
        let steps = self.axes[idx].unwrapper.update(raw);
        Ok(steps as f64 * TAU / self.axes[idx].counts_per_rev as f64)
    }

    fn stop_axis_and_wait(&self, axis_num: u8) -> IoResult<()> {
        talk_axis(&self.socket, axis_num, 'K', "")?;
        for _ in 0..STOP_POLL_MAX {
            let st = query_axis_status(&self.socket, axis_num)?;
            if !st.running {
                return Ok(());
            }
            sleep(STOP_POLL_DELAY);
        }
        Err(Error::new(
            ErrorKind::TimedOut,
            format!("Sky-Watcher axis {} did not stop after :K", axis_num),
        ))
    }

    fn goto_axis_rad(&mut self, idx: usize, target_rad: f64) -> IoResult<()> {
        if !target_rad.is_finite() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Target axis angle must be finite",
            ));
        }
        let axis_num = (idx + 1) as u8;
        // Stop any active tracking/slew before sampling position so coasting doesn't bias :H
        // and so an offset==0 move still halts a tracking axis.
        let st = query_axis_status(&self.socket, axis_num)?;
        if st.running {
            self.stop_axis_and_wait(axis_num)?;
        }

        let _ = self.read_axis_rad(idx)?;
        let cpr = self.axes[idx].counts_per_rev;
        let target_steps = (target_rad * cpr as f64 / TAU).round() as i64;
        let offset = target_steps - self.axes[idx].unwrapper.current();
        if offset == 0 {
            return Ok(());
        }

        let abs_u64 = offset.unsigned_abs();
        if abs_u64 > 0xFF_FFFF {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Sky-Watcher goto offset {} exceeds 24-bit limit", abs_u64),
            ));
        }
        let abs_off = abs_u64 as u32;
        let dir = if offset >= 0 { '0' } else { '1' };
        let high_speed = abs_off > self.axes[idx].low_speed_goto_margin;
        let mode = if high_speed { '0' } else { '2' };
        let ramp_cap = if high_speed {
            HIGH_SPEED_RAMP_MAX
        } else {
            LOW_SPEED_RAMP_MAX
        };
        let ramp = abs_off.min(ramp_cap);

        talk_axis(&self.socket, axis_num, 'G', &format!("{}{}", mode, dir))?;
        talk_axis(&self.socket, axis_num, 'H', &encode_le24(abs_off))?;
        talk_axis(&self.socket, axis_num, 'M', &encode_le24(ramp))?;
        talk_axis(&self.socket, axis_num, 'J', "")?;
        Ok(())
    }

    fn set_axis_rate_rad_per_sec(&self, idx: usize, rate_rad_s: f64) -> IoResult<()> {
        let axis_num = (idx + 1) as u8;
        if !rate_rad_s.is_finite() || rate_rad_s.abs() < MIN_RATE_RAD_PER_SEC {
            let st = query_axis_status(&self.socket, axis_num)?;
            if st.running {
                talk_axis(&self.socket, axis_num, 'K', "")?;
            }
            return Ok(());
        }

        let forward = rate_rad_s > 0.0;
        let abs_rate = rate_rad_s.abs();
        let high_speed = abs_rate > LOW_SPEED_RATE_LIMIT;
        let ax = &self.axes[idx];
        let adj_rate = if high_speed {
            abs_rate / ax.high_speed_ratio as f64
        } else {
            abs_rate
        };

        let steps_per_sec = adj_rate * (ax.counts_per_rev as f64 / TAU);
        let t1_ticks =
            ((ax.timer_freq as f64 / steps_per_sec).round() as u64).clamp(6, 0xFF_FFFF) as u32;

        let st = query_axis_status(&self.socket, axis_num)?;
        let mode_matches =
            st.running && st.rate_mode && st.forward == forward && st.high_speed == high_speed;

        if !mode_matches {
            if st.running {
                self.stop_axis_and_wait(axis_num)?;
            }
            let mode = if high_speed { '3' } else { '1' };
            let dir = if forward { '0' } else { '1' };
            talk_axis(&self.socket, axis_num, 'G', &format!("{}{}", mode, dir))?;
        }

        talk_axis(&self.socket, axis_num, 'I', &encode_le24(t1_ticks))?;
        if !mode_matches {
            talk_axis(&self.socket, axis_num, 'J', "")?;
        }
        Ok(())
    }
}

impl RawAxisMount for SkyWatcherMount {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_alt_az(&self) -> bool {
        self.is_alt_az
    }

    fn read_axes_rad(&mut self) -> IoResult<(f64, f64)> {
        Ok((self.read_axis_rad(0)?, self.read_axis_rad(1)?))
    }

    fn is_slewing(&mut self) -> IoResult<bool> {
        let s1 = query_axis_status(&self.socket, 1)?;
        let s2 = query_axis_status(&self.socket, 2)?;
        Ok((s1.running && !s1.rate_mode) || (s2.running && !s2.rate_mode))
    }

    fn goto_axes_rad(&mut self, ax1_rad: f64, ax2_rad: f64) -> IoResult<()> {
        self.goto_axis_rad(0, ax1_rad)?;
        self.goto_axis_rad(1, ax2_rad)
    }

    fn set_rates_rad_per_sec(&mut self, rate1: f64, rate2: f64) -> IoResult<()> {
        self.set_axis_rate_rad_per_sec(0, rate1)?;
        self.set_axis_rate_rad_per_sec(1, rate2)
    }

    fn abort(&mut self) -> IoResult<()> {
        let r1 = talk_axis(&self.socket, 1, 'K', "").map(|_| ());
        let r2 = talk_axis(&self.socket, 2, 'K', "").map(|_| ());
        r1.and(r2)
    }
}

/// Drains any late UDP replies sitting in the receive buffer before issuing a new command.
fn drain_stale_udp(socket: &UdpSocket) {
    if socket.set_nonblocking(true).is_ok() {
        let mut discard = [0u8; 64];
        while socket.recv(&mut discard).is_ok() {}
        let _ = socket.set_nonblocking(false);
    }
}

fn talk_axis(socket: &UdpSocket, axis: u8, cmd: char, params: &str) -> IoResult<String> {
    let tx = format!(":{}{}{}\r", cmd, axis, params);
    let mut buf = [0u8; 64];

    for attempt in 0..UDP_RETRIES {
        drain_stale_udp(socket);
        trace!("Sky-Watcher TX: {:?}", tx);
        socket.send(tx.as_bytes())?;
        match socket.recv(&mut buf) {
            Ok(n) => {
                let raw = from_utf8(&buf[..n])
                    .map_err(|_| invalid_data("Non-UTF8 Sky-Watcher UDP reply"))?
                    .trim_end_matches(['\r', '\n']);
                trace!("Sky-Watcher RX: {:?}", raw);
                if let Some(payload) = raw.strip_prefix('=') {
                    return Ok(payload.to_string());
                }
                return Err(invalid_data(format!(
                    "Sky-Watcher error reply {:?} to {:?}",
                    raw, tx
                )));
            }
            Err(e)
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
                    && attempt + 1 < UDP_RETRIES =>
            {
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    Err(Error::new(ErrorKind::TimedOut, "Sky-Watcher UDP timeout"))
}

fn init_axis(socket: &UdpSocket, axis: u8) -> IoResult<AxisParams> {
    let cpr = decode_le24(&talk_axis(socket, axis, 'a', "")?)?;
    if cpr == 0 {
        return Err(invalid_data("Sky-Watcher counts_per_rev is zero"));
    }
    let timer_freq = decode_le24(&talk_axis(socket, axis, 'b', "")?)?;
    if timer_freq == 0 {
        return Err(invalid_data("Sky-Watcher timer_freq is zero"));
    }
    let hs_raw = talk_axis(socket, axis, 'g', "")?;
    let high_speed_ratio = decode_u8_hex(&hs_raw)?.max(1) as u32;

    let st = query_axis_status(socket, axis)?;
    if !st.initialized {
        talk_axis(socket, axis, 'F', "")?;
    }

    let initial_raw = decode_le24(&talk_axis(socket, axis, 'j', "")?)?;
    Ok(AxisParams {
        counts_per_rev: cpr,
        timer_freq,
        high_speed_ratio,
        low_speed_goto_margin: (cpr / 36).max(3_200),
        unwrapper: AxisUnwrapper::new(initial_raw, ENCODER_ORIGIN),
    })
}

fn query_axis_status(socket: &UdpSocket, axis: u8) -> IoResult<AxisStatus> {
    let reply = talk_axis(socket, axis, 'f', "")?;
    parse_axis_status(&reply)
}

fn parse_axis_status(reply: &str) -> IoResult<AxisStatus> {
    let bytes = reply.as_bytes();
    if bytes.len() < 3 {
        return Err(invalid_data(format!(
            "Short Sky-Watcher :f reply {:?}",
            reply
        )));
    }
    let n0 = hex_nibble(bytes[0])?;
    let n1 = hex_nibble(bytes[1])?;
    let n2 = hex_nibble(bytes[2])?;
    Ok(AxisStatus {
        running: (n1 & 0x01) != 0,
        rate_mode: (n0 & 0x01) != 0,
        forward: (n0 & 0x02) == 0,
        high_speed: (n0 & 0x04) != 0,
        initialized: (n2 & 0x01) != 0,
    })
}

fn identify_mount(ver_hex: &str) -> IoResult<(String, bool)> {
    if !ver_hex.is_ascii() {
        return Err(invalid_data("Non-ASCII Sky-Watcher :e1 reply"));
    }
    let bytes = ver_hex.as_bytes();
    if bytes.len() < 4 {
        return Err(invalid_data(format!(
            "Short Sky-Watcher :e1 reply {:?}",
            ver_hex
        )));
    }
    if bytes.len() < 6 {
        return Ok((format!("Sky-Watcher ({})", ver_hex), false));
    }
    let model_code = decode_u8_hex(&ver_hex[4..6])?;
    Ok(match model_code {
        0x00 => ("Sky-Watcher EQ6".to_string(), false),
        0x01 => ("Sky-Watcher HEQ5".to_string(), false),
        0x02 => ("Sky-Watcher EQ5".to_string(), false),
        0x03 => ("Sky-Watcher EQ3".to_string(), false),
        0x04 => ("Sky-Watcher EQ8".to_string(), false),
        0x05 => ("Sky-Watcher AZ-EQ6".to_string(), true),
        0x06 => ("Sky-Watcher AZ-EQ5".to_string(), true),
        0x80 => ("Sky-Watcher GT".to_string(), true),
        0x81 => ("Sky-Watcher MF".to_string(), true),
        0x82 => ("Sky-Watcher 114GT".to_string(), true),
        0x90 => ("Sky-Watcher Dobsonian".to_string(), true),
        0xA0 => ("Sky-Watcher AllView / Virtuoso".to_string(), true),
        0xA2 => ("Sky-Watcher AZ-GTe".to_string(), true),
        0xA5 => ("Sky-Watcher AZ-GTi".to_string(), true),
        0xA6 => ("Sky-Watcher AZ-GTiX".to_string(), true),
        _ => (format!("Sky-Watcher (0x{:02X})", model_code), false),
    })
}

fn encode_le24(value: u32) -> String {
    let v = value & 0xFF_FFFF;
    format!(
        "{:02X}{:02X}{:02X}",
        v & 0xFF,
        (v >> 8) & 0xFF,
        (v >> 16) & 0xFF
    )
}

fn decode_le24(hex: &str) -> IoResult<u32> {
    let b = hex.as_bytes();
    if b.len() != 6 {
        return Err(invalid_data(format!("Expected 6 hex chars, got {:?}", hex)));
    }
    let b0 = (hex_nibble(b[0])? << 4) | hex_nibble(b[1])?;
    let b1 = (hex_nibble(b[2])? << 4) | hex_nibble(b[3])?;
    let b2 = (hex_nibble(b[4])? << 4) | hex_nibble(b[5])?;
    Ok((b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16))
}

fn decode_u8_hex(hex: &str) -> IoResult<u8> {
    let b = hex.as_bytes();
    if b.len() < 2 {
        return Err(invalid_data(format!("Expected 2 hex chars, got {:?}", hex)));
    }
    Ok((hex_nibble(b[0])? << 4) | hex_nibble(b[1])?)
}

fn hex_nibble(c: u8) -> IoResult<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        _ => Err(invalid_data(format!("Invalid hex char 0x{:02X}", c))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread::spawn;

    #[test]
    fn test_le24_encoding_status_and_model_codes() {
        assert_eq!(encode_le24(1_000_000), "40420F");
        assert_eq!(decode_le24("40420F").unwrap(), 1_000_000);
        assert_eq!(decode_le24("000080").unwrap(), 0x80_0000);

        let st = parse_axis_status("311").unwrap();
        assert!(st.running && st.rate_mode && !st.forward && !st.high_speed && st.initialized);
        let st2 = parse_axis_status("000").unwrap();
        assert!(!st2.running && !st2.rate_mode && st2.forward && !st2.initialized);

        assert_eq!(
            identify_mount("0305A5").unwrap(),
            ("Sky-Watcher AZ-GTi".to_string(), true)
        );
        assert_eq!(
            identify_mount("020705").unwrap(),
            ("Sky-Watcher AZ-EQ6".to_string(), true)
        );
        assert_eq!(
            identify_mount("020706").unwrap(),
            ("Sky-Watcher AZ-EQ5".to_string(), true)
        );
        assert_eq!(
            identify_mount("020790").unwrap(),
            ("Sky-Watcher Dobsonian".to_string(), true)
        );
        assert_eq!(
            identify_mount("0106").unwrap(),
            ("Sky-Watcher (0106)".to_string(), false)
        );
    }

    #[test]
    fn test_udp_roundtrip_az_gti_handshake_goto_and_tracking() {
        let srv = UdpSocket::bind("127.0.0.1:0").unwrap();
        let port = srv.local_addr().unwrap().port();
        srv.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let log_c = log.clone();

        spawn(move || {
            let mut pos = [0x80_0000_u32, 0x80_0000_u32];
            let mut initialized = [false, false];
            let mut running = [false, false];
            let mut rate_mode = [false, false];
            let mut forward = [true, true];
            let mut pending_offset = [0_i64, 0_i64];
            let mut buf = [0u8; 64];

            while let Ok((n, peer)) = srv.recv_from(&mut buf) {
                let cmd = from_utf8(&buf[..n]).unwrap().trim_end_matches('\r');
                log_c.lock().unwrap().push(cmd.to_string());
                let b = cmd.as_bytes();
                let axis = (b[2] - b'1') as usize;
                let reply = match b[1] {
                    b'e' => "=0305A5\r".to_string(),
                    b'a' => format!("={}\r", encode_le24(2_073_600)),
                    b'b' => format!("={}\r", encode_le24(16_000_000)),
                    b'g' => "=10\r".to_string(),
                    b'F' => {
                        initialized[axis] = true;
                        "=\r".to_string()
                    }
                    b'f' => {
                        let n0 = (u8::from(rate_mode[axis])) | (u8::from(!forward[axis]) << 1);
                        let n1 = u8::from(running[axis]);
                        let n2 = u8::from(initialized[axis]);
                        format!("={:X}{:X}{:X}\r", n0, n1, n2)
                    }
                    b'j' => format!("={}\r", encode_le24(pos[axis])),
                    b'G' => {
                        let mode = b[3];
                        rate_mode[axis] = mode == b'1' || mode == b'3';
                        forward[axis] = b[4] == b'0';
                        "=\r".to_string()
                    }
                    b'H' => {
                        let mag = decode_le24(&cmd[3..9]).unwrap() as i64;
                        pending_offset[axis] = if forward[axis] { mag } else { -mag };
                        "=\r".to_string()
                    }
                    b'J' => {
                        if !rate_mode[axis] {
                            pos[axis] = ((pos[axis] as i64 + pending_offset[axis])
                                .rem_euclid(1 << 24))
                                as u32;
                            running[axis] = false;
                        } else {
                            running[axis] = true;
                        }
                        "=\r".to_string()
                    }
                    b'K' => {
                        running[axis] = false;
                        "=\r".to_string()
                    }
                    _ => "=\r".to_string(),
                };
                srv.send_to(reply.as_bytes(), peer).unwrap();
            }
        });

        let mut m = SkyWatcherMount::connect("127.0.0.1", port).unwrap();
        assert_eq!(m.model_name(), "Sky-Watcher AZ-GTi");
        assert!(m.is_alt_az());

        let (a1, a2) = m.read_axes_rad().unwrap();
        assert!(a1.abs() < 1e-6 && a2.abs() < 1e-6);

        assert!(m.goto_axes_rad(f64::NAN, 0.0).is_err());

        let target1 = 10.0_f64.to_radians();
        let target2 = 20.0_f64.to_radians();
        m.goto_axes_rad(target1, target2).unwrap();
        assert!(!m.is_slewing().unwrap());
        let (n1, n2) = m.read_axes_rad().unwrap();
        assert!((n1 - target1).abs() < 1e-4);
        assert!((n2 - target2).abs() < 1e-4);

        m.set_rates_rad_per_sec(SIDEREAL_RAD_PER_SEC, -0.5 * SIDEREAL_RAD_PER_SEC)
            .unwrap();
        assert!(!m.is_slewing().unwrap());
        m.set_rates_rad_per_sec(1.1 * SIDEREAL_RAD_PER_SEC, -0.6 * SIDEREAL_RAD_PER_SEC)
            .unwrap();
        // Zero-offset goto on axis 2 while tracking must still stop axis 2!
        m.goto_axes_rad(target1 + 0.1, target2).unwrap();
        m.abort().unwrap();

        let cmds = log.lock().unwrap();
        assert!(cmds.iter().any(|c| c == ":F1"));
        assert!(cmds.iter().any(|c| c == ":F2"));
        assert!(cmds.iter().any(|c| c.starts_with(":H1")));
        assert!(cmds.iter().any(|c| c.starts_with(":I1")));
        assert!(cmds.iter().any(|c| c == ":K1"));
        assert!(cmds.iter().any(|c| c == ":K2"));
    }
}
