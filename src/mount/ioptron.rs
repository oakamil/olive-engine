// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

//! iOptron V2.5 & V3.10 TCP Wi-Fi mount driver (`TCP 8899`).
//!
//! Supports HAZ31, AZ Mount Pro, HAZ46/71/130, HAE/HEM, CEM, GEM, and iEQ Pro
//! mounts with automatic V2.5 vs V3.10 dialect detection via `:GLS#` length.

use std::io::{Error, ErrorKind, Read, Result as IoResult, Write};
use std::net::TcpStream;
use std::str::from_utf8;

use chrono::{DateTime, Datelike, Timelike};
use log::{info, trace};

use super::{
    MANUAL_SLEW_MAX_SPEED, MANUAL_SLEW_MIN_SPEED, MountDriver, MountRawStatus,
    StartTrackingOutcome, TcpConnector, angular_separation_deg, connect_tcp_stream, invalid_data,
};

const J2000_NOON_UNIX_MS: i64 = 946_728_000_000;
const SYNC_READBACK_MAX_ERROR_DEG: f64 = 0.25;
const MAX_HASH_REPLY_LEN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IoptronDialect {
    /// V2.5 (`:GLS#` = 19 bytes + `#`, `:GEC#` = 17 bytes + `#`, `:Sr` ms, `:MS#`).
    V25,
    /// V3.10 (`:GLS#` = 23 bytes + `#`, `:GEP#` = 20 bytes + `#`, `:SRA` 0.01", `:MS1#`).
    V310,
}

/// Blocking TCP client for iOptron V2.5 / V3.10 mounts.
pub(super) struct IoptronMount {
    stream: TcpStream,
    dialect: IoptronDialect,
    model_name: String,
    is_alt_az: bool,
    manual_axes: (i8, i8),
    last_manual_speed: Option<u8>,
}

impl IoptronMount {
    /// Connects to an iOptron Wi-Fi mount over TCP and performs the `:MountInfo#` + `:GLS#` handshake.
    pub(super) fn connect(
        host: &str,
        port: u16,
        connector: Option<&TcpConnector>,
    ) -> IoResult<Self> {
        let stream = connect_tcp_stream(host, port, connector)?;
        Self::from_stream(stream)
    }

    fn from_stream(mut stream: TcpStream) -> IoResult<Self> {
        // `:MountInfo#` returns exactly 4 ASCII digits with no `#` terminator.
        send_raw(&mut stream, b":MountInfo#")?;
        let mut info_buf = [0u8; 4];
        stream.read_exact(&mut info_buf)?;
        if !info_buf.iter().all(u8::is_ascii_digit) {
            return Err(invalid_data("Non-numeric :MountInfo# reply"));
        }
        let code = from_utf8(&info_buf).map_err(|_| invalid_data("Non-UTF8 :MountInfo# reply"))?;
        let (model_name, is_alt_az) = identify_model(code);

        // `:GLS#` returns 19 bytes + `#` (V2.5) or 23 bytes + `#` (V3.10).
        send_raw(&mut stream, b":GLS#")?;
        let gls = read_until_hash(&mut stream)?;
        let (dialect, _gps, _sys) = parse_gls(&gls)?;

        info!(
            "iOptron handshake OK: {} (code={}, alt_az={}, dialect={:?})",
            model_name, code, is_alt_az, dialect
        );

        Ok(Self {
            stream,
            dialect,
            model_name,
            is_alt_az,
            manual_axes: (0, 0),
            last_manual_speed: None,
        })
    }

    fn send_action(&mut self, cmd: &str) -> IoResult<bool> {
        trace!("iOptron TX: {}", cmd);
        send_raw(&mut self.stream, cmd.as_bytes())?;
        let mut reply = [0u8; 1];
        self.stream.read_exact(&mut reply)?;
        trace!("iOptron RX: {:?}", reply[0] as char);
        match reply[0] {
            b'1' => Ok(true),
            b'0' => Ok(false),
            other => Err(invalid_data(format!(
                "Unexpected 1-byte iOptron reply 0x{:02x} to {}",
                other, cmd
            ))),
        }
    }

    fn require_action(&mut self, cmd: &str) -> IoResult<()> {
        if self.send_action(cmd)? {
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::Other,
                format!("iOptron mount rejected command {}", cmd),
            ))
        }
    }

    fn read_equatorial_jnow(&mut self) -> IoResult<(f64, f64)> {
        match self.dialect {
            IoptronDialect::V25 => {
                send_raw(&mut self.stream, b":GEC#")?;
                let payload = read_until_hash(&mut self.stream)?;
                parse_gec_v25(&payload)
            }
            IoptronDialect::V310 => {
                send_raw(&mut self.stream, b":GEP#")?;
                let payload = read_until_hash(&mut self.stream)?;
                parse_gep_v310(&payload)
            }
        }
    }

    fn set_target_ra_dec_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<bool> {
        let ra_cmd = format_set_ra(self.dialect, ra_deg)?;
        if !self.send_action(&ra_cmd)? {
            return Ok(false);
        }
        let dec_cmd = format_set_dec(dec_deg)?;
        self.send_action(&dec_cmd)
    }

    fn stop_manual_axes(&mut self) -> IoResult<()> {
        if self.manual_axes != (0, 0) {
            let _ = self.send_action(":q#")?;
            self.manual_axes = (0, 0);
        }
        Ok(())
    }

    fn set_manual_axis(
        &mut self,
        axis_idx: usize,
        target: i8,
        stop_cmd: &str,
        pos_cmd: &[u8],
        neg_cmd: &[u8],
    ) -> IoResult<()> {
        let cur = if axis_idx == 0 {
            self.manual_axes.0
        } else {
            self.manual_axes.1
        };
        if cur == target {
            return Ok(());
        }
        if cur != 0 {
            let _ = self.send_action(stop_cmd)?;
            if axis_idx == 0 {
                self.manual_axes.0 = 0;
            } else {
                self.manual_axes.1 = 0;
            }
        }
        if target > 0 {
            trace!("iOptron TX: {}", from_utf8(pos_cmd).unwrap_or("?"));
            send_raw(&mut self.stream, pos_cmd)?;
        } else if target < 0 {
            trace!("iOptron TX: {}", from_utf8(neg_cmd).unwrap_or("?"));
            send_raw(&mut self.stream, neg_cmd)?;
        }
        if axis_idx == 0 {
            self.manual_axes.0 = target;
        } else {
            self.manual_axes.1 = target;
        }
        Ok(())
    }
}

impl MountDriver for IoptronMount {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_alt_az(&self) -> bool {
        self.is_alt_az
    }

    fn poll(&mut self) -> IoResult<MountRawStatus> {
        send_raw(&mut self.stream, b":GLS#")?;
        let gls = read_until_hash(&mut self.stream)?;
        let (dialect, gps_locked, sys_status) = parse_gls(&gls)?;
        if dialect != self.dialect {
            return Err(invalid_data(format!(
                "iOptron :GLS# dialect mismatch: expected {:?}, got {:?}",
                self.dialect, dialect
            )));
        }
        let (ra_jnow_deg, dec_jnow_deg) = self.read_equatorial_jnow()?;

        Ok(MountRawStatus {
            is_slewing: matches!(sys_status, b'2' | b'4'),
            is_tracking: matches!(sys_status, b'1' | b'3' | b'5'),
            is_parked: sys_status == b'6',
            at_zero: sys_status == b'7',
            gps_locked,
            ra_jnow_deg,
            dec_jnow_deg,
            safety_notice: None,
        })
    }

    fn goto_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<bool> {
        self.stop_manual_axes()?;
        if !self.set_target_ra_dec_jnow(ra_deg, dec_deg)? {
            return Ok(false);
        }
        let cmd = match self.dialect {
            IoptronDialect::V25 => ":MS#",
            IoptronDialect::V310 => ":MS1#",
        };
        self.send_action(cmd)
    }

    fn sync_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<Result<(), f64>> {
        let target_ok = self.set_target_ra_dec_jnow(ra_deg, dec_deg)?;
        let cm_ok = if target_ok {
            self.send_action(":CM#")?
        } else {
            false
        };

        let (rb_ra, rb_dec) = self.read_equatorial_jnow()?;
        let offset = angular_separation_deg(ra_deg, dec_deg, rb_ra, rb_dec);

        if cm_ok && offset < SYNC_READBACK_MAX_ERROR_DEG {
            Ok(Ok(()))
        } else {
            Ok(Err(offset))
        }
    }

    fn start_tracking(&mut self) -> IoResult<StartTrackingOutcome> {
        Ok(if self.send_action(":ST1#")? {
            StartTrackingOutcome::Started
        } else {
            StartTrackingOutcome::Refused
        })
    }

    fn stop_tracking(&mut self) -> IoResult<bool> {
        self.send_action(":ST0#")
    }

    fn abort(&mut self) -> IoResult<()> {
        let q_res = self.send_action(":Q#");
        let m_res = self.stop_manual_axes();
        q_res.map(|_| ()).and(m_res)
    }

    fn manual_slew(&mut self, axis1_dir: i8, axis2_dir: i8, speed: u8) -> IoResult<()> {
        let a1 = axis1_dir.clamp(-1, 1);
        let a2 = axis2_dir.clamp(-1, 1);
        if (a1 == 0 && a2 == 0) || speed == 0 {
            return self.stop_manual_axes();
        }

        let spd = speed.clamp(MANUAL_SLEW_MIN_SPEED, MANUAL_SLEW_MAX_SPEED);
        let speed_changed = self.last_manual_speed != Some(spd);
        if speed_changed {
            // Many V2.5/V3.10 firmwares ignore `:SRn#` for already-running arrow motion.
            self.stop_manual_axes()?;
            self.require_action(&format!(":SR{}#", spd))?;
            self.last_manual_speed = Some(spd);
        }

        self.set_manual_axis(0, a1, ":qR#", b":me#", b":mw#")?;
        self.set_manual_axis(1, a2, ":qD#", b":mn#", b":ms#")
    }

    fn unpark(&mut self) -> IoResult<bool> {
        self.send_action(":MP0#")
    }

    fn sync_site_time(&mut self, lat_deg: f64, lon_deg: f64, utc_ms: i64) -> IoResult<()> {
        let (lon_cmd, lat_cmd) = format_site_commands(self.dialect, lat_deg, lon_deg)?;
        self.require_action(&lon_cmd)?;
        self.require_action(&lat_cmd)?;
        self.require_action(":SG+000#")?;
        self.require_action(":SDS0#")?;

        match self.dialect {
            IoptronDialect::V25 => {
                let (date_cmd, time_cmd) = format_utc_v25(utc_ms)?;
                self.require_action(&date_cmd)?;
                self.require_action(&time_cmd)?;
            }
            IoptronDialect::V310 => {
                let sut_cmd = format_sut_v310(utc_ms)?;
                self.require_action(&sut_cmd)?;
            }
        }
        Ok(())
    }
}

fn invalid_input(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, msg.into())
}

fn send_raw(stream: &mut TcpStream, bytes: &[u8]) -> IoResult<()> {
    stream.write_all(bytes)?;
    stream.flush()
}

fn read_until_hash<R: Read>(reader: &mut R) -> IoResult<Vec<u8>> {
    let mut buf = Vec::with_capacity(32);
    let mut byte = [0u8; 1];
    loop {
        reader.read_exact(&mut byte)?;
        if byte[0] == b'#' {
            return Ok(buf);
        }
        if buf.len() >= MAX_HASH_REPLY_LEN {
            return Err(invalid_data("iOptron '#' reply exceeded maximum length"));
        }
        buf.push(byte[0]);
    }
}

fn identify_model(code: &str) -> (String, bool) {
    match code {
        "0035" => ("HAZ31".to_string(), true),
        "8035" => ("HAZ31SM".to_string(), true),
        "0052" => ("HAZ46".to_string(), true),
        "8052" => ("HAZ46SM".to_string(), true),
        "0073" => ("HAZ71".to_string(), true),
        "0123" => ("HAZ130".to_string(), true),
        "5035" => ("AZ Mount Pro".to_string(), true),
        "9035" => ("AZ Mount Pro SM".to_string(), true),
        "8033" => ("HAE29 AA".to_string(), true),
        "8034" => ("HAE29EC AA".to_string(), true),
        "8050" => ("HAE43 AA".to_string(), true),
        "8051" => ("HAE43EC AA".to_string(), true),
        "8068" => ("HAE69 AA".to_string(), true),
        "8069" => ("HAE69EC AA".to_string(), true),
        "0046" | "5045" => ("iEQ45 Pro AA".to_string(), true),
        "0013" | "0022" | "0032" | "0038" | "0055" | "0056" | "0064" | "0065" | "5010" => {
            (format!("iOptron AA ({})", code), true)
        }
        "0011" => ("SmartEQ Pro+".to_string(), false),
        "0015" => ("HEM15".to_string(), false),
        "0025" => ("CEM25".to_string(), false),
        "0026" => ("CEM26".to_string(), false),
        "0027" => ("HEM27".to_string(), false),
        "0028" => ("GEM28".to_string(), false),
        "0029" => ("GEM28EC".to_string(), false),
        "0030" => ("iEQ30 Pro".to_string(), false),
        "0033" => ("HAE29".to_string(), false),
        "0034" => ("HAE29EC".to_string(), false),
        "0040" => ("CEM40".to_string(), false),
        "0041" => ("CEM40EC".to_string(), false),
        "0043" => ("GEM45".to_string(), false),
        "0044" => ("GEM45EC".to_string(), false),
        "0045" => ("iEQ45 Pro".to_string(), false),
        "0050" => ("HAE43".to_string(), false),
        "0051" => ("HAE43EC".to_string(), false),
        "0060" => ("CEM60".to_string(), false),
        "0061" => ("CEM60EC".to_string(), false),
        "0068" => ("HAE69".to_string(), false),
        "0069" => ("HAE69EC".to_string(), false),
        "0070" => ("CEM70".to_string(), false),
        "0071" => ("CEM70G".to_string(), false),
        "0120" => ("CEM120".to_string(), false),
        _ => (format!("iOptron ({})", code), false),
    }
}

fn parse_gls(payload: &[u8]) -> IoResult<(IoptronDialect, bool, u8)> {
    match payload.len() {
        19 => {
            let (gps, sys) = (payload[13], payload[14]);
            validate_sys_status(sys)?;
            Ok((IoptronDialect::V25, gps == b'2', sys))
        }
        23 => {
            let (gps, sys) = (payload[17], payload[18]);
            validate_sys_status(sys)?;
            Ok((IoptronDialect::V310, gps == b'2', sys))
        }
        len => Err(invalid_data(format!(
            "Unexpected :GLS# payload length {} (expected 19 or 23)",
            len
        ))),
    }
}

fn validate_sys_status(sys: u8) -> IoResult<()> {
    if (b'0'..=b'7').contains(&sys) {
        Ok(())
    } else {
        Err(invalid_data(format!(
            "Invalid iOptron SysStatus byte 0x{:02x}",
            sys
        )))
    }
}

fn parse_dec_centi_arcsec(bytes: &[u8]) -> IoResult<f64> {
    let dec_centi = parse_signed_ascii_i64(bytes)?;
    Ok((dec_centi as f64 / 360_000.0).clamp(-90.0, 90.0))
}

fn parse_gec_v25(payload: &[u8]) -> IoResult<(f64, f64)> {
    if payload.len() != 17 {
        return Err(invalid_data(format!(
            "Expected 17-byte :GEC# payload, got {}",
            payload.len()
        )));
    }
    let dec_deg = parse_dec_centi_arcsec(&payload[0..9])?;
    let ra_ms = parse_unsigned_ascii_u64(&payload[9..17])?;
    let ra_deg = (ra_ms as f64 / 240_000.0).rem_euclid(360.0);
    Ok((ra_deg, dec_deg))
}

fn parse_gep_v310(payload: &[u8]) -> IoResult<(f64, f64)> {
    if payload.len() != 20 {
        return Err(invalid_data(format!(
            "Expected 20-byte :GEP# payload, got {}",
            payload.len()
        )));
    }
    let dec_deg = parse_dec_centi_arcsec(&payload[0..9])?;
    let ra_centi = parse_unsigned_ascii_u64(&payload[9..18])?;
    let ra_deg = (ra_centi as f64 / 360_000.0).rem_euclid(360.0);
    Ok((ra_deg, dec_deg))
}

fn parse_signed_ascii_i64(bytes: &[u8]) -> IoResult<i64> {
    let s = from_utf8(bytes).map_err(|_| invalid_data("Non-UTF8 signed field"))?;
    if !(s.starts_with('+') || s.starts_with('-')) {
        return Err(invalid_data(format!("Missing sign in field {:?}", s)));
    }
    s.parse::<i64>()
        .map_err(|_| invalid_data(format!("Invalid signed field {:?}", s)))
}

fn parse_unsigned_ascii_u64(bytes: &[u8]) -> IoResult<u64> {
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return Err(invalid_data("Non-digit unsigned field"));
    }
    let s = from_utf8(bytes).map_err(|_| invalid_data("Non-UTF8 unsigned field"))?;
    s.parse::<u64>()
        .map_err(|_| invalid_data(format!("Invalid unsigned field {:?}", s)))
}

fn format_signed_scaled(deg: f64, limit: f64, scale: f64, max_mag: u64) -> IoResult<(char, u64)> {
    if !deg.is_finite() {
        return Err(invalid_input("Coordinate must be finite"));
    }
    let clamped = deg.clamp(-limit, limit);
    let sign = if clamped < 0.0 { '-' } else { '+' };
    let mag = ((clamped.abs() * scale).round() as u64).min(max_mag);
    Ok((sign, mag))
}

fn format_set_ra(dialect: IoptronDialect, ra_deg: f64) -> IoResult<String> {
    if !ra_deg.is_finite() {
        return Err(invalid_input("RA must be finite"));
    }
    let ra_norm = ra_deg.rem_euclid(360.0);
    Ok(match dialect {
        IoptronDialect::V25 => {
            let ms = ((ra_norm * 240_000.0).round() as u64) % 86_400_000;
            format!(":Sr{:08}#", ms)
        }
        IoptronDialect::V310 => {
            let centi = ((ra_norm * 360_000.0).round() as u64) % 129_600_000;
            format!(":SRA{:09}#", centi)
        }
    })
}

fn format_set_dec(dec_deg: f64) -> IoResult<String> {
    let (sign, centi) = format_signed_scaled(dec_deg, 90.0, 360_000.0, 32_400_000)?;
    Ok(format!(":Sd{}{:08}#", sign, centi))
}

fn format_site_commands(
    dialect: IoptronDialect,
    lat_deg: f64,
    lon_deg: f64,
) -> IoResult<(String, String)> {
    match dialect {
        IoptronDialect::V25 => {
            let (lon_sign, lon_as) = format_signed_scaled(lon_deg, 180.0, 3_600.0, 648_000)?;
            let (lat_sign, lat_as) = format_signed_scaled(lat_deg, 90.0, 3_600.0, 324_000)?;
            Ok((
                format!(":Sg{}{:06}#", lon_sign, lon_as),
                format!(":St{}{:06}#", lat_sign, lat_as),
            ))
        }
        IoptronDialect::V310 => {
            let (lon_sign, lon_c) = format_signed_scaled(lon_deg, 180.0, 360_000.0, 64_800_000)?;
            let (lat_sign, lat_c) = format_signed_scaled(lat_deg, 90.0, 360_000.0, 32_400_000)?;
            Ok((
                format!(":SLO{}{:08}#", lon_sign, lon_c),
                format!(":SLA{}{:08}#", lat_sign, lat_c),
            ))
        }
    }
}

fn format_sut_v310(utc_ms: i64) -> IoResult<String> {
    if utc_ms < J2000_NOON_UNIX_MS {
        return Err(invalid_input(format!(
            "UTC timestamp {} precedes J2000.0 noon",
            utc_ms
        )));
    }
    let j2000_ms = (utc_ms - J2000_NOON_UNIX_MS) as u64;
    Ok(format!(":SUT{:013}#", j2000_ms))
}

fn format_utc_v25(utc_ms: i64) -> IoResult<(String, String)> {
    let dt = DateTime::from_timestamp_millis(utc_ms)
        .ok_or_else(|| invalid_input("Invalid UTC timestamp"))?;
    let yy = dt.year().rem_euclid(100);
    Ok((
        format!(":SC{:02}{:02}{:02}#", yy, dt.month(), dt.day()),
        format!(":SL{:02}{:02}{:02}#", dt.hour(), dt.minute(), dt.second()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread::spawn;

    fn format_mock_equatorial(dialect: IoptronDialect, ra_deg: f64, dec_deg: f64) -> String {
        let (sign, dec_c) = format_signed_scaled(dec_deg, 90.0, 360_000.0, 32_400_000).unwrap();
        match dialect {
            IoptronDialect::V25 => {
                let ra_ms = ((ra_deg.rem_euclid(360.0) * 240_000.0).round() as u64) % 86_400_000;
                format!("{}{:08}{:08}#", sign, dec_c, ra_ms)
            }
            IoptronDialect::V310 => {
                let ra_c = ((ra_deg.rem_euclid(360.0) * 360_000.0).round() as u64) % 129_600_000;
                format!("{}{:08}{:09}10#", sign, dec_c, ra_c)
            }
        }
    }

    #[test]
    fn test_identify_model_alt_az_and_eq() {
        assert_eq!(identify_model("0035"), ("HAZ31".to_string(), true));
        assert_eq!(identify_model("8035"), ("HAZ31SM".to_string(), true));
        assert_eq!(identify_model("5035"), ("AZ Mount Pro".to_string(), true));
        assert_eq!(identify_model("8033"), ("HAE29 AA".to_string(), true));
        assert_eq!(identify_model("0033"), ("HAE29".to_string(), false));
        assert_eq!(identify_model("0011"), ("SmartEQ Pro+".to_string(), false));
        assert_eq!(identify_model("0027"), ("HEM27".to_string(), false));
        assert_eq!(identify_model("0040"), ("CEM40".to_string(), false));
        assert_eq!(
            identify_model("9999"),
            ("iOptron (9999)".to_string(), false)
        );
    }

    #[test]
    fn test_parse_gls_v25_and_v310() {
        let v25 = b"-440712459820210111";
        assert_eq!(v25.len(), 19);
        assert_eq!(parse_gls(v25).unwrap(), (IoptronDialect::V25, true, b'1'));

        let v310 = b"-4407120045982000171011";
        assert_eq!(v310.len(), 23);
        assert_eq!(
            parse_gls(v310).unwrap(),
            (IoptronDialect::V310, false, b'7')
        );

        assert!(parse_gls(b"short").is_err());
    }

    #[test]
    fn test_ra_wraparound_dec_clamping_and_nan_rejection() {
        assert_eq!(
            format_set_ra(IoptronDialect::V310, 359.9999999).unwrap(),
            ":SRA000000000#"
        );
        assert_eq!(
            format_set_ra(IoptronDialect::V25, 359.9999999).unwrap(),
            ":Sr00000000#"
        );
        assert_eq!(
            format_set_ra(IoptronDialect::V310, 180.0).unwrap(),
            ":SRA064800000#"
        );
        assert_eq!(
            format_set_ra(IoptronDialect::V25, 180.0).unwrap(),
            ":Sr43200000#"
        );
        assert!(format_set_ra(IoptronDialect::V310, f64::NAN).is_err());

        assert_eq!(format_set_dec(45.0).unwrap(), ":Sd+16200000#");
        assert_eq!(format_set_dec(-95.0).unwrap(), ":Sd-32400000#");
        assert_eq!(format_set_dec(95.0).unwrap(), ":Sd+32400000#");
        assert!(format_set_dec(f64::INFINITY).is_err());
    }

    #[test]
    fn test_parse_gec_v25_and_gep_v310() {
        let (ra, dec) = parse_gec_v25(b"+1620000043200000").unwrap();
        assert!((ra - 180.0).abs() < 1e-6);
        assert!((dec - 45.0).abs() < 1e-6);

        let (ra, dec) = parse_gep_v310(b"-1098000003240000010").unwrap();
        assert!((ra - 90.0).abs() < 1e-6);
        assert!((dec - (-30.5)).abs() < 1e-6);
    }

    #[test]
    fn test_site_and_time_formatting() {
        assert_eq!(
            format_site_commands(IoptronDialect::V25, 37.75, -122.5).unwrap(),
            (":Sg-441000#".to_string(), ":St+135900#".to_string())
        );
        assert_eq!(
            format_site_commands(IoptronDialect::V310, 37.75, -122.5).unwrap(),
            (":SLO-44100000#".to_string(), ":SLA+13590000#".to_string())
        );
        assert!(format_site_commands(IoptronDialect::V310, f64::NAN, 0.0).is_err());

        assert!(format_sut_v310(J2000_NOON_UNIX_MS - 1).is_err());
        assert_eq!(
            format_sut_v310(J2000_NOON_UNIX_MS).unwrap(),
            ":SUT0000000000000#"
        );
        assert_eq!(
            format_sut_v310(J2000_NOON_UNIX_MS + 123_456).unwrap(),
            ":SUT0000000123456#"
        );

        let (sc, sl) = format_utc_v25(J2000_NOON_UNIX_MS).unwrap();
        assert_eq!(sc, ":SC000101#");
        assert_eq!(sl, ":SL120000#");
    }

    fn spawn_mock_ioptron_server(
        dialect: IoptronDialect,
        model_code: &'static [u8; 4],
        gps_byte: u8,
        sys_byte: u8,
    ) -> (u16, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut ra_deg = 10.0_f64;
            let mut dec_deg = 20.0_f64;
            let mut target_ra = 10.0_f64;
            let mut target_dec = 20.0_f64;

            loop {
                let mut cmd = Vec::new();
                let mut b = [0u8; 1];
                loop {
                    if sock.read_exact(&mut b).is_err() {
                        return;
                    }
                    cmd.push(b[0]);
                    if b[0] == b'#' {
                        break;
                    }
                }
                let cmd_str = String::from_utf8_lossy(&cmd).to_string();
                log_clone.lock().unwrap().push(cmd_str.clone());

                match cmd_str.as_str() {
                    ":MountInfo#" => sock.write_all(model_code).unwrap(),
                    ":GLS#" => {
                        let reply = match dialect {
                            IoptronDialect::V25 => {
                                format!(
                                    "-440712459820{}{}0111#",
                                    gps_byte as char, sys_byte as char
                                )
                            }
                            IoptronDialect::V310 => {
                                format!(
                                    "-4407120045982000{}{}0111#",
                                    gps_byte as char, sys_byte as char
                                )
                            }
                        };
                        sock.write_all(reply.as_bytes()).unwrap();
                    }
                    ":GEC#" | ":GEP#" => {
                        let reply = format_mock_equatorial(dialect, ra_deg, dec_deg);
                        sock.write_all(reply.as_bytes()).unwrap();
                    }
                    ":CM#" => {
                        ra_deg = target_ra;
                        dec_deg = target_dec;
                        sock.write_all(b"1").unwrap();
                    }
                    s if s.starts_with(":SRA") => {
                        target_ra = s[4..s.len() - 1].parse::<u64>().unwrap() as f64 / 360_000.0;
                        sock.write_all(b"1").unwrap();
                    }
                    s if s.starts_with(":Sr") => {
                        target_ra = s[3..s.len() - 1].parse::<u64>().unwrap() as f64 / 240_000.0;
                        sock.write_all(b"1").unwrap();
                    }
                    s if s.starts_with(":Sd") => {
                        target_dec = s[3..s.len() - 1].parse::<i64>().unwrap() as f64 / 360_000.0;
                        sock.write_all(b"1").unwrap();
                    }
                    ":mn#" | ":ms#" | ":me#" | ":mw#" => {
                        // Directional arrow commands produce no reply on iOptron V2.5/V3.10.
                    }
                    _ => sock.write_all(b"1").unwrap(),
                }
            }
        });

        (port, log)
    }

    #[test]
    fn test_tcp_roundtrip_v310_haz31() {
        let (port, log) = spawn_mock_ioptron_server(IoptronDialect::V310, b"0035", b'1', b'1');
        let mut m = IoptronMount::connect("127.0.0.1", port, None).unwrap();
        assert_eq!(m.dialect, IoptronDialect::V310);
        assert_eq!(m.model_name(), "HAZ31");
        assert!(m.is_alt_az());

        let st = m.poll().unwrap();
        assert!(st.is_tracking);
        assert!(!st.gps_locked);
        assert!((st.ra_jnow_deg - 10.0).abs() < 1e-4);

        m.sync_site_time(37.75, -122.5, J2000_NOON_UNIX_MS + 1000)
            .unwrap();
        assert!(m.sync_jnow(120.5, 33.25).unwrap().is_ok());
        assert!(m.goto_jnow(125.0, 35.0).unwrap());
        assert!(m.stop_tracking().unwrap());
        m.abort().unwrap();

        let cmds = log.lock().unwrap();
        assert!(cmds.iter().any(|c| c == ":MountInfo#"));
        assert!(cmds.iter().any(|c| c == ":SG+000#"));
        assert!(cmds.iter().any(|c| c == ":SDS0#"));
        assert!(cmds.iter().any(|c| c.starts_with(":SUT")));
        assert!(cmds.iter().any(|c| c == ":MS1#"));
        assert!(cmds.iter().any(|c| c == ":ST0#"));
        assert!(cmds.iter().any(|c| c == ":Q#"));
    }

    #[test]
    fn test_tcp_roundtrip_v25_az_mount_pro() {
        let (port, log) = spawn_mock_ioptron_server(IoptronDialect::V25, b"5035", b'2', b'6');
        let mut m = IoptronMount::connect("127.0.0.1", port, None).unwrap();
        assert_eq!(m.dialect, IoptronDialect::V25);
        assert_eq!(m.model_name(), "AZ Mount Pro");
        assert!(m.is_alt_az());

        let st = m.poll().unwrap();
        assert!(st.is_parked);
        assert!(st.gps_locked);
        assert!(m.unpark().unwrap());
        assert!(m.sync_jnow(210.25, -15.5).unwrap().is_ok());
        assert!(m.goto_jnow(215.0, -10.0).unwrap());

        let cmds = log.lock().unwrap();
        assert!(cmds.iter().any(|c| c == ":GEC#"));
        assert!(cmds.iter().any(|c| c == ":MP0#"));
        assert!(cmds.iter().any(|c| c == ":MS#"));
    }

    #[test]
    fn test_rejects_non_numeric_mount_info() {
        let (port, _) = spawn_mock_ioptron_server(IoptronDialect::V310, b"1#\r\n", b'2', b'1');
        assert!(IoptronMount::connect("127.0.0.1", port, None).is_err());
    }

    #[test]
    fn test_manual_slew_protocol_reversals_speed_change_and_abort() {
        let (port, log) = spawn_mock_ioptron_server(IoptronDialect::V310, b"0035", b'1', b'7');
        let mut m = IoptronMount::connect("127.0.0.1", port, None).unwrap();

        // Start East at speed 5 -> :SR5# then :me# (no reply).
        m.manual_slew(1, 0, 5).unwrap();
        // Repeating the same state is a no-op on the wire.
        m.manual_slew(1, 0, 5).unwrap();
        // Reverse Axis 1 (East -> West) -> :qR# before :mw#.
        m.manual_slew(-1, 0, 5).unwrap();
        // Change speed mid-slew (5 -> 9) and add North -> :q#, :SR9#, :mw#, :mn#.
        m.manual_slew(-1, 1, 9).unwrap();
        // Drop Axis 2 only -> :qD#.
        m.manual_slew(-1, 0, 9).unwrap();
        // Abort while manual motion is active must send BOTH :Q# and :q#!
        m.abort().unwrap();
        // Follow-up poll confirms TCP stream stayed completely synchronized.
        let _ = m.poll().unwrap();

        let cmds = log.lock().unwrap();
        let manual_seq: Vec<&str> = cmds
            .iter()
            .skip_while(|c| !c.starts_with(":SR"))
            .take_while(|c| *c != ":GLS#")
            .map(String::as_str)
            .collect();
        assert_eq!(
            manual_seq,
            vec![
                ":SR5#", ":me#", ":qR#", ":mw#", ":q#", ":SR9#", ":mw#", ":mn#", ":qD#", ":Q#",
                ":q#",
            ]
        );
    }
}
