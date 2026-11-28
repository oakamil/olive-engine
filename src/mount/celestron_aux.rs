// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

//! Celestron NexStar AUX Wi-Fi Motor Controller driver (`TCP 2000`, default `1.2.3.4`).
//!
//! Speaks the native Celestron `0x3B`-framed binary AUX bus protocol directly to
//! the Azimuth/RA (`0x10`) and Altitude/Dec (`0x11`) motor controllers, filtering
//! unrelated shared-bus broadcasts and using `1/1024 arcsec/sec` 24-bit rates.

use std::f64::consts::{PI, TAU};
use std::io::{Error, ErrorKind, Read, Result as IoResult, Write};
use std::net::TcpStream;
use std::thread::sleep;
use std::time::{Duration, Instant};

use log::{info, trace};

use super::kinematic::{AxisUnwrapper, MIN_RATE_RAD_PER_SEC, RawAxisMount};
use super::{
    MANUAL_SLEW_MAX_SPEED, MANUAL_SLEW_MIN_SPEED, TcpConnector, connect_tcp_stream, invalid_data,
};

const READ_REPLY_DEADLINE: Duration = Duration::from_millis(700);
const MANUAL_DECEL_POLL_MAX: usize = 12;
const MANUAL_DECEL_POLL_DELAY: Duration = Duration::from_millis(50);
const MANUAL_HIGH_SPEED_SETTLE_MIN: u8 = 6;
const MAX_SKIPPED_PACKETS: usize = 32;
const MAX_PREAMBLE_NOISE_BYTES: usize = 128;

const PREAMBLE: u8 = 0x3B;
const DEV_AZM: u8 = 0x10;
const DEV_ALT: u8 = 0x11;
const DEV_APP: u8 = 0x20;

const CMD_GET_POSITION: u8 = 0x01;
const CMD_GOTO_FAST: u8 = 0x02;
const CMD_SET_POS_GUIDERATE: u8 = 0x06;
const CMD_SET_NEG_GUIDERATE: u8 = 0x07;
const CMD_SLEW_DONE: u8 = 0x13;
const CMD_MOVE_POS: u8 = 0x24;
const CMD_MOVE_NEG: u8 = 0x25;
const CMD_GET_VER: u8 = 0xFE;

const STEPS_PER_REV: f64 = 16_777_216.0; // 2^24
const ARCSEC_PER_RAD: f64 = 206_264.806_247_096_36;
/// Celestron AUX 24-bit rate scale: `STEPS_PER_ARCSEC * (10125/128) = 1024.0` LSB per arcsec/sec.
const RATE_LSB_PER_ARCSEC_S: f64 = 1024.0;

/// Blocking TCP client for Celestron NexStar AUX Wi-Fi mounts (Evolution, SkyPortal WiFi).
pub(super) struct CelestronAuxMount {
    stream: TcpStream,
    model_name: String,
    unwrappers: [AxisUnwrapper; 2],
    last_rates: [Option<i32>; 2],
    manual_state: [(i8, u8); 2],
}

impl CelestronAuxMount {
    /// Connects to a Celestron AUX Wi-Fi module (`TCP 2000`) and handshakes with both motor boards.
    pub(super) fn connect(
        host: &str,
        port: u16,
        connector: Option<&TcpConnector>,
    ) -> IoResult<Self> {
        let stream = connect_tcp_stream(host, port, connector)?;
        Self::from_stream(stream)
    }

    fn from_stream(mut stream: TcpStream) -> IoResult<Self> {
        let ver = exchange_aux(&mut stream, DEV_AZM, CMD_GET_VER, &[])?;
        let model_name = if ver.len() >= 2 {
            format!("Celestron AUX (MC v{}.{})", ver[0], ver[1])
        } else {
            "Celestron AUX".to_string()
        };

        let raw1 = query_raw_position(&mut stream, DEV_AZM)?;
        let raw2 = query_raw_position(&mut stream, DEV_ALT)?;

        info!("Celestron AUX handshake OK: {}", model_name);

        Ok(Self {
            stream,
            model_name,
            unwrappers: [AxisUnwrapper::new(raw1, 0), AxisUnwrapper::new(raw2, 0)],
            last_rates: [None, None],
            manual_state: [(0, 0); 2],
        })
    }

    fn set_axis_rate_steps(&mut self, idx: usize, rate_lsb: i32, force: bool) -> IoResult<()> {
        if !force && self.last_rates[idx] == Some(rate_lsb) {
            return Ok(());
        }
        let dev = axis_dev(idx);
        let cmd = if rate_lsb < 0 {
            CMD_SET_NEG_GUIDERATE
        } else {
            CMD_SET_POS_GUIDERATE
        };
        let mag = rate_lsb.unsigned_abs().min(0xFF_FFFF);
        let data = encode_be24(mag);
        exchange_aux(&mut self.stream, dev, cmd, &data)?;
        self.last_rates[idx] = Some(rate_lsb);
        Ok(())
    }

    fn stop_manual_axis(&mut self, idx: usize) -> IoResult<()> {
        let cmd = if self.manual_state[idx].0 < 0 {
            CMD_MOVE_NEG
        } else {
            CMD_MOVE_POS
        };
        exchange_aux(&mut self.stream, axis_dev(idx), cmd, &[0x00])?;
        self.manual_state[idx] = (0, 0);
        Ok(())
    }

    fn read_raw_positions(&mut self) -> IoResult<(u32, u32)> {
        Ok((
            query_raw_position(&mut self.stream, DEV_AZM)?,
            query_raw_position(&mut self.stream, DEV_ALT)?,
        ))
    }

    fn goto_axis_rad(&mut self, idx: usize, target_rad: f64) -> IoResult<()> {
        if !target_rad.is_finite() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Target axis angle must be finite",
            ));
        }
        if self.manual_state[idx] != (0, 0) {
            self.stop_manual_axis(idx)?;
        }
        self.set_axis_rate_steps(idx, 0, true)?;
        let raw_target =
            ((target_rad * STEPS_PER_REV / TAU).round() as i64).rem_euclid(1 << 24) as u32;
        let data = encode_be24(raw_target);
        exchange_aux(&mut self.stream, axis_dev(idx), CMD_GOTO_FAST, &data)?;
        Ok(())
    }

    fn manual_slew_axis(&mut self, idx: usize, dir: i8, speed: u8) -> IoResult<bool> {
        let target = if dir == 0 || speed == 0 {
            (0, 0)
        } else {
            (
                dir.clamp(-1, 1),
                speed.clamp(MANUAL_SLEW_MIN_SPEED, MANUAL_SLEW_MAX_SPEED),
            )
        };
        let prev = self.manual_state[idx];
        if prev == target {
            // If transitioning from tracking into a single-axis manual slew, halt the orthogonal
            // axis's active guiderate so it doesn't keep drifting!
            if target.0 == 0 && self.last_rates[idx].is_some_and(|r| r != 0) {
                self.set_axis_rate_steps(idx, 0, true)?;
            }
            return Ok(false);
        }

        self.last_rates[idx] = None;
        if target.0 == 0 {
            self.stop_manual_axis(idx)?;
            return Ok(prev.1 >= MANUAL_HIGH_SPEED_SETTLE_MIN);
        }

        if prev.0 != 0 && prev.0 != target.0 {
            self.stop_manual_axis(idx)?;
        }
        let cmd = if target.0 > 0 {
            CMD_MOVE_POS
        } else {
            CMD_MOVE_NEG
        };
        exchange_aux(&mut self.stream, axis_dev(idx), cmd, &[target.1])?;
        self.manual_state[idx] = target;
        Ok(false)
    }

    fn wait_for_manual_settle(&mut self) -> IoResult<()> {
        let mut prev = self.read_raw_positions()?;
        for _ in 0..MANUAL_DECEL_POLL_MAX {
            sleep(MANUAL_DECEL_POLL_DELAY);
            let cur = self.read_raw_positions()?;
            if cur == prev {
                return Ok(());
            }
            prev = cur;
        }
        Ok(())
    }

    fn halt_axis(&mut self, idx: usize) -> IoResult<()> {
        // Sending MC_MOVE_*(0) aborts both manual slews and active MC_GOTO_FAST slews on AUX.
        let r_move = self.stop_manual_axis(idx);
        let r_rate = self.set_axis_rate_steps(idx, 0, true);
        r_move.and(r_rate)
    }
}

impl RawAxisMount for CelestronAuxMount {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_alt_az(&self) -> bool {
        true
    }

    fn read_axes_rad(&mut self) -> IoResult<(f64, f64)> {
        let (r1, r2) = self.read_raw_positions()?;
        let s1 = self.unwrappers[0].update(r1);
        let s2 = self.unwrappers[1].update(r2);
        Ok((
            s1 as f64 * TAU / STEPS_PER_REV,
            s2 as f64 * TAU / STEPS_PER_REV,
        ))
    }

    fn max_goto_step_rad(&self) -> Option<f64> {
        // `MC_GOTO_FAST` takes a 24-bit modulo-2π target and always slews along the shorter arc;
        // splitting slews > 135° guarantees > 180° GEM meridian flips always go over the pole.
        Some(0.75 * PI)
    }

    fn is_slewing(&mut self) -> IoResult<bool> {
        let d1 = exchange_aux(&mut self.stream, DEV_AZM, CMD_SLEW_DONE, &[])?;
        let d2 = exchange_aux(&mut self.stream, DEV_ALT, CMD_SLEW_DONE, &[])?;
        let Some(&b1) = d1.first() else {
            return Err(invalid_data("Empty AZM MC_SLEW_DONE reply"));
        };
        let Some(&b2) = d2.first() else {
            return Err(invalid_data("Empty ALT MC_SLEW_DONE reply"));
        };
        let slewing = !(b1 == 0xFF && b2 == 0xFF);
        if slewing {
            self.last_rates = [None, None];
        }
        Ok(slewing)
    }

    fn goto_axes_rad(&mut self, ax1_rad: f64, ax2_rad: f64) -> IoResult<()> {
        self.goto_axis_rad(0, ax1_rad)?;
        self.goto_axis_rad(1, ax2_rad)
    }

    fn set_rates_rad_per_sec(&mut self, rate1: f64, rate2: f64) -> IoResult<()> {
        let r1 = rad_per_sec_to_aux_rate(rate1);
        let r2 = rad_per_sec_to_aux_rate(rate2);
        self.set_axis_rate_steps(0, r1, false)?;
        self.set_axis_rate_steps(1, r2, false)
    }

    fn manual_slew(&mut self, axis1_dir: i8, axis2_dir: i8, speed: u8) -> IoResult<()> {
        let hi0 = self.manual_slew_axis(0, axis1_dir, speed)?;
        let hi1 = self.manual_slew_axis(1, axis2_dir, speed)?;
        if (hi0 || hi1) && self.manual_state == [(0, 0); 2] {
            self.wait_for_manual_settle()?;
        }
        Ok(())
    }

    fn abort(&mut self) -> IoResult<()> {
        let r1 = self.halt_axis(0);
        let r2 = self.halt_axis(1);
        r1.and(r2)
    }
}

fn axis_dev(idx: usize) -> u8 {
    if idx == 0 { DEV_AZM } else { DEV_ALT }
}

fn rad_per_sec_to_aux_rate(rate_rad_s: f64) -> i32 {
    if !rate_rad_s.is_finite() || rate_rad_s.abs() < MIN_RATE_RAD_PER_SEC {
        return 0;
    }
    let arcsec_s = rate_rad_s * ARCSEC_PER_RAD;
    (arcsec_s * RATE_LSB_PER_ARCSEC_S)
        .round()
        .clamp(-16_777_215.0, 16_777_215.0) as i32
}

fn encode_be24(val: u32) -> [u8; 3] {
    let v = val & 0xFF_FFFF;
    [
        ((v >> 16) & 0xFF) as u8,
        ((v >> 8) & 0xFF) as u8,
        (v & 0xFF) as u8,
    ]
}

fn decode_be24(bytes: &[u8]) -> IoResult<u32> {
    if bytes.len() < 3 {
        return Err(invalid_data(format!(
            "Expected 3-byte AUX position, got {}",
            bytes.len()
        )));
    }
    Ok(((bytes[0] as u32) << 16) | ((bytes[1] as u32) << 8) | (bytes[2] as u32))
}

fn aux_checksum(fields: &[u8]) -> u8 {
    let sum = fields.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    0u8.wrapping_sub(sum)
}

fn build_aux_packet(src: u8, dst: u8, cmd: u8, data: &[u8]) -> Vec<u8> {
    let len = (3 + data.len()) as u8;
    let mut pkt = Vec::with_capacity(5 + data.len() + 1);
    pkt.push(PREAMBLE);
    pkt.push(len);
    pkt.push(src);
    pkt.push(dst);
    pkt.push(cmd);
    pkt.extend_from_slice(data);
    let cksum = aux_checksum(&pkt[1..]);
    pkt.push(cksum);
    pkt
}

/// Reads one `0x3B`-framed AUX packet before `deadline`. Returns `InvalidData` on checksum/length
/// corruption so `exchange_aux` can resync on the next preamble before timing out.
fn read_aux_packet<R: Read>(reader: &mut R, deadline: Instant) -> IoResult<(u8, u8, u8, Vec<u8>)> {
    let mut b = [0u8; 1];
    let mut noise = 0usize;
    loop {
        if Instant::now() >= deadline {
            return Err(Error::new(ErrorKind::TimedOut, "AUX preamble timeout"));
        }
        reader.read_exact(&mut b)?;
        if b[0] == PREAMBLE {
            break;
        }
        noise += 1;
        if noise >= MAX_PREAMBLE_NOISE_BYTES {
            return Err(invalid_data("Excess AUX bus noise"));
        }
    }
    let len = loop {
        if Instant::now() >= deadline {
            return Err(Error::new(ErrorKind::TimedOut, "AUX length timeout"));
        }
        reader.read_exact(&mut b)?;
        if b[0] == PREAMBLE {
            // Consecutive 0x3B bytes: treat the second byte as the start of a fresh packet.
            noise += 1;
            if noise >= MAX_PREAMBLE_NOISE_BYTES {
                return Err(invalid_data("Excess AUX bus noise"));
            }
            continue;
        }
        break b[0] as usize;
    };
    if !(3..=32).contains(&len) {
        return Err(invalid_data(format!("Invalid AUX packet length {}", len)));
    }

    let mut body = vec![0u8; len + 1];
    reader.read_exact(&mut body)?;

    let mut sum = len as u8;
    for &byte in &body[..len] {
        sum = sum.wrapping_add(byte);
    }
    let expected = 0u8.wrapping_sub(sum);
    if body[len] != expected {
        return Err(invalid_data(format!(
            "AUX checksum mismatch: expected 0x{:02X}, got 0x{:02X}",
            expected, body[len]
        )));
    }

    let src = body[0];
    let dst = body[1];
    let cmd = body[2];
    let data = body[3..len].to_vec();
    Ok((src, dst, cmd, data))
}

fn exchange_aux(stream: &mut TcpStream, dst: u8, cmd: u8, data: &[u8]) -> IoResult<Vec<u8>> {
    let pkt = build_aux_packet(DEV_APP, dst, cmd, data);
    trace!("AUX TX: {:02X?}", pkt);
    stream.write_all(&pkt)?;
    stream.flush()?;

    let deadline = Instant::now() + READ_REPLY_DEADLINE;
    for _ in 0..MAX_SKIPPED_PACKETS {
        if Instant::now() >= deadline {
            break;
        }
        match read_aux_packet(stream, deadline) {
            Ok((src, rx_dst, rx_cmd, rx_data)) => {
                trace!(
                    "AUX RX: src=0x{:02X} dst=0x{:02X} cmd=0x{:02X} data={:02X?}",
                    src, rx_dst, rx_cmd, rx_data
                );
                if rx_dst == DEV_APP && src == dst && rx_cmd == cmd {
                    return Ok(rx_data);
                }
            }
            // Shared-bus collision or stray 0x3B byte: resync on next preamble until deadline.
            Err(e) if e.kind() == ErrorKind::InvalidData => continue,
            Err(e) => return Err(e),
        }
    }
    Err(Error::new(
        ErrorKind::TimedOut,
        format!(
            "No matching AUX reply from 0x{:02X} for cmd 0x{:02X}",
            dst, cmd
        ),
    ))
}

fn query_raw_position(stream: &mut TcpStream, dev: u8) -> IoResult<u32> {
    let data = exchange_aux(stream, dev, CMD_GET_POSITION, &[])?;
    decode_be24(&data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread::spawn;

    #[test]
    fn test_aux_packet_checksum_and_rate_scaling() {
        let pkt = build_aux_packet(DEV_APP, DEV_AZM, CMD_GET_POSITION, &[]);
        assert_eq!(pkt, vec![0x3B, 0x03, 0x20, 0x10, 0x01, 0xCC]);

        let sidereal_rad_s = 15.041_068_6 / ARCSEC_PER_RAD;
        let rate = rad_per_sec_to_aux_rate(sidereal_rad_s);
        assert!((rate - 15_402).abs() <= 1);
        assert_eq!(rad_per_sec_to_aux_rate(-sidereal_rad_s), -rate);
        assert_eq!(rad_per_sec_to_aux_rate(0.0), 0);
    }

    #[test]
    fn test_tcp_roundtrip_with_bus_noise_corrupt_packet_and_24bit_wrap() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Mutex::new(Vec::<(u8, u8, Vec<u8>)>::new()));
        let log_c = log.clone();

        spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut pos = [0xFF_FF00_u32, 0x10_0000_u32];
            let mut slewing = [false, false];
            let deadline = Instant::now() + Duration::from_secs(5);

            while let Ok((src, dst, cmd, data)) = read_aux_packet(&mut sock, deadline) {
                log_c.lock().unwrap().push((dst, cmd, data.clone()));
                // Inject stray bytes, bad-checksum packet, unrelated HC broadcast, and a stray 0x3B
                // right before the valid reply to verify resync across consecutive preambles.
                sock.write_all(&[0xAA, 0x55, 0x3B, 0x03, 0x04, 0x10, 0x01, 0x00])
                    .unwrap();
                let noise = build_aux_packet(0x04, 0x10, 0x01, &[]);
                sock.write_all(&noise).unwrap();
                sock.write_all(&[PREAMBLE]).unwrap();

                let idx = usize::from(dst == DEV_ALT);
                let reply_data = match cmd {
                    CMD_GET_VER => vec![7, 17],
                    CMD_GET_POSITION => encode_be24(pos[idx]).to_vec(),
                    CMD_GOTO_FAST => {
                        pos[idx] = decode_be24(&data).unwrap();
                        slewing[idx] = false;
                        vec![]
                    }
                    CMD_SLEW_DONE => vec![if slewing[idx] { 0x00 } else { 0xFF }],
                    CMD_SET_POS_GUIDERATE | CMD_SET_NEG_GUIDERATE | CMD_MOVE_POS | CMD_MOVE_NEG => {
                        slewing[idx] = false;
                        vec![]
                    }
                    _ => vec![],
                };
                let reply = build_aux_packet(dst, src, cmd, &reply_data);
                sock.write_all(&reply).unwrap();
            }
        });

        let mut m = CelestronAuxMount::connect("127.0.0.1", port, None).unwrap();
        assert_eq!(m.model_name(), "Celestron AUX (MC v7.17)");
        assert!(m.goto_axes_rad(f64::NAN, 0.0).is_err());

        let (a1, _a2) = m.read_axes_rad().unwrap();
        let target1 = a1 + 1.0_f64.to_radians();
        let target2 = 30.0_f64.to_radians();
        m.goto_axes_rad(target1, target2).unwrap();
        assert!(!m.is_slewing().unwrap());

        let (n1, n2) = m.read_axes_rad().unwrap();
        assert!((n1 - target1).abs() < 1e-4);
        assert!((n2 - target2).abs() < 1e-4);

        m.set_rates_rad_per_sec(15.0 / ARCSEC_PER_RAD, -5.0 / ARCSEC_PER_RAD)
            .unwrap();
        // Single-axis manual slew while tracking must zero the orthogonal axis's guiderate!
        m.manual_slew(1, 0, 5).unwrap();
        assert!(log.lock().unwrap().iter().any(|(d, c, data)| {
            *d == DEV_ALT && *c == CMD_SET_POS_GUIDERATE && data == &[0, 0, 0]
        }));
        // Dual-axis manual slew (pos/neg + reversal + high-speed stop) must invalidate last_rates!
        m.manual_slew(1, -1, 7).unwrap();
        m.manual_slew(-1, -1, 7).unwrap();
        m.manual_slew(0, 0, 0).unwrap();
        let guide_before = log
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, c, _)| *c == CMD_SET_POS_GUIDERATE)
            .count();
        m.set_rates_rad_per_sec(15.0 / ARCSEC_PER_RAD, -5.0 / ARCSEC_PER_RAD)
            .unwrap();
        let guide_after = log
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, c, _)| *c == CMD_SET_POS_GUIDERATE)
            .count();
        assert_eq!(guide_after, guide_before + 1);
        m.abort().unwrap();

        let seen = log.lock().unwrap();
        assert!(seen.iter().any(|(_, c, _)| *c == CMD_GET_VER));
        assert!(seen.iter().any(|(_, c, _)| *c == CMD_GOTO_FAST));
        assert!(seen.iter().any(|(_, c, _)| *c == CMD_SET_POS_GUIDERATE));
        assert!(seen.iter().any(|(_, c, _)| *c == CMD_SET_NEG_GUIDERATE));
        assert!(
            seen.iter()
                .any(|(d, c, data)| *d == DEV_AZM && *c == CMD_MOVE_POS && data == &[7])
        );
        assert!(
            seen.iter()
                .any(|(d, c, data)| *d == DEV_ALT && *c == CMD_MOVE_NEG && data == &[7])
        );
        assert!(
            seen.iter()
                .any(|(d, c, data)| *d == DEV_AZM && *c == CMD_MOVE_POS && data == &[0])
        );
    }
}
