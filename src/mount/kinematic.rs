// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

//! Generic Class B kinematic wrapper (`KinematicMount<M: RawAxisMount>`).
//!
//! Bridges raw two-axis motor-controller mounts (Sky-Watcher SynScan UDP,
//! Celestron AUX TCP) onto [`MountDriver`] by owning the `SO(3)`
//! [`GimbalSkyModel`] and refreshing dual-axis tracking velocities inside `poll()`.

use std::f64::consts::PI;
use std::io::Result as IoResult;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use log::{debug, info, warn};
use starfield::time::Timescale;

use super::sky_model::{GimbalSkyModel, TrackingHaltReason, wrap_pi};
use super::{MountDriver, MountRawStatus, StartTrackingOutcome, angular_separation_deg};

/// Minimum non-zero tracking velocity below which axes are commanded to halt (rad/s).
pub(super) const MIN_RATE_RAD_PER_SEC: f64 = 1e-7;
/// Maximum closed-loop encoder phase-lock trim added to tracking velocity (10 arcsec/sec).
const MAX_TRACK_TRIM_RAD_PER_SEC: f64 = (10.0 / 3600.0) * PI / 180.0;
const TRACK_TRIM_GAIN: f64 = 0.2;
const SYNC_VERIFY_MAX_ERR_DEG: f64 = 0.25;
const SLEW_START_GRACE: Duration = Duration::from_millis(350);

/// Unwraps 24-bit unsigned motor encoder counts (`0..2^24`) into continuous signed steps.
pub(super) struct AxisUnwrapper {
    last_raw: u32,
    unwrapped: i64,
}

impl AxisUnwrapper {
    pub(super) fn new(initial_raw: u32, origin: i64) -> Self {
        let r = initial_raw & 0xFF_FFFF;
        Self {
            last_raw: r,
            unwrapped: r as i64 - origin,
        }
    }

    pub(super) fn update(&mut self, raw: u32) -> i64 {
        let r = raw & 0xFF_FFFF;
        let delta = ((r.wrapping_sub(self.last_raw) << 8) as i32) >> 8;
        self.last_raw = r;
        self.unwrapped += delta as i64;
        self.unwrapped
    }

    pub(super) fn current(&self) -> i64 {
        self.unwrapped
    }
}

/// Minimal synchronous two-axis raw motor controller interface.
pub(super) trait RawAxisMount: Send {
    fn model_name(&self) -> &str;
    fn is_alt_az(&self) -> bool;
    /// Returns unwrapped encoder shaft angles `(axis1_rad, axis2_rad)`.
    fn read_axes_rad(&mut self) -> IoResult<(f64, f64)>;
    /// Returns `true` if either axis is currently executing a GoTo slew.
    fn is_slewing(&mut self) -> IoResult<bool>;
    /// Commands both axes to slew to unwrapped target angles `(axis1_rad, axis2_rad)`.
    fn goto_axes_rad(&mut self, ax1_rad: f64, ax2_rad: f64) -> IoResult<()>;
    /// Sets continuous tracking velocities `(rad/sec, rad/sec)` on both axes.
    fn set_rates_rad_per_sec(&mut self, rate1: f64, rate2: f64) -> IoResult<()>;
    /// Halts all motion on both axes.
    fn abort(&mut self) -> IoResult<()>;
}

/// Class B mount driver owning the `SO(3)` sky model and dual-axis tracking loop.
pub(super) struct KinematicMount<M: RawAxisMount> {
    raw: M,
    timescale: Arc<Timescale>,
    model: GimbalSkyModel,
    tracking: bool,
    our_slew_started_at: Option<Instant>,
    our_slew_seen: bool,
    track_target_jnow: Option<(f64, f64)>,
}

impl<M: RawAxisMount> KinematicMount<M> {
    pub(super) fn new(raw: M, timescale: Arc<Timescale>, gem_meridian_flip: bool) -> Self {
        let mut model = GimbalSkyModel::default();
        model.set_gem_meridian_flip(gem_meridian_flip);
        Self {
            raw,
            timescale,
            model,
            tracking: false,
            our_slew_started_at: None,
            our_slew_seen: false,
            track_target_jnow: None,
        }
    }

    /// Refreshes dual-axis tracking rates, returning `Ok(Some(notice))` if a safety limit halted tracking.
    fn refresh_tracking(
        &mut self,
        cur_ax1: f64,
        cur_ax2: f64,
        now: DateTime<Utc>,
    ) -> IoResult<Option<&'static str>> {
        let Some((ra_jnow, dec_jnow)) = self.track_target_jnow else {
            return Ok(None);
        };
        let ((t1, t2), (v1, v2)) = match self.model.tracking_target_and_rates(
            ra_jnow,
            dec_jnow,
            cur_ax1,
            cur_ax2,
            now,
            &self.timescale,
        ) {
            Ok(res) => res,
            Err(reason) => {
                let notice = match reason {
                    TrackingHaltReason::GimbalCone => {
                        warn!(
                            "Tracked target entered 85° gimbal cone; halting tracking for safety"
                        );
                        "Tracking stopped — 85° zenith/pole limit reached"
                    }
                    TrackingHaltReason::MeridianLimit => {
                        warn!(
                            "EQ mount reached 10° past meridian limit; halting tracking for safety"
                        );
                        "Tracking stopped — meridian limit reached (GoTo to flip)"
                    }
                };
                self.tracking = false;
                self.track_target_jnow = None;
                self.raw.abort()?;
                return Ok(Some(notice));
            }
        };

        let trim1 = (wrap_pi(t1 - cur_ax1) * TRACK_TRIM_GAIN)
            .clamp(-MAX_TRACK_TRIM_RAD_PER_SEC, MAX_TRACK_TRIM_RAD_PER_SEC);
        let trim2 = (wrap_pi(t2 - cur_ax2) * TRACK_TRIM_GAIN)
            .clamp(-MAX_TRACK_TRIM_RAD_PER_SEC, MAX_TRACK_TRIM_RAD_PER_SEC);

        self.raw.set_rates_rad_per_sec(v1 + trim1, v2 + trim2)?;
        Ok(None)
    }
}

impl<M: RawAxisMount> MountDriver for KinematicMount<M> {
    fn model_name(&self) -> &str {
        self.raw.model_name()
    }

    fn is_alt_az(&self) -> bool {
        self.model
            .detected_alt_az()
            .unwrap_or_else(|| self.raw.is_alt_az())
    }

    fn is_calibrated(&self) -> bool {
        self.model.is_calibrated()
    }

    fn is_goto_ready(&self) -> bool {
        self.model.is_calibrated() || self.model.sample_count() > 0
    }

    fn requires_host_tracking(&self) -> bool {
        true
    }

    fn set_gem_meridian_flip(&mut self, enabled: bool) {
        self.model.set_gem_meridian_flip(enabled);
    }

    fn reset_calibration(&mut self) {
        self.model.reset();
        self.tracking = false;
        self.our_slew_started_at = None;
        self.our_slew_seen = false;
        self.track_target_jnow = None;
    }

    fn poll(&mut self) -> IoResult<MountRawStatus> {
        let (ax1, ax2) = self.raw.read_axes_rad()?;
        let is_slewing = self.raw.is_slewing()?;
        let now = Utc::now();

        let mut in_start_grace = false;
        let mut safety_notice = None;
        if is_slewing {
            if self.our_slew_started_at.is_some() {
                self.our_slew_seen = true;
            } else {
                // An external/hand-controller slew invalidates our previous tracking target.
                self.tracking = false;
                self.track_target_jnow = None;
            }
        } else {
            in_start_grace = self
                .our_slew_started_at
                .is_some_and(|t| !self.our_slew_seen && t.elapsed() < SLEW_START_GRACE);
            if !in_start_grace {
                self.our_slew_started_at = None;
                self.our_slew_seen = false;
                if self.tracking && self.model.is_calibrated() {
                    safety_notice = self.refresh_tracking(ax1, ax2, now)?;
                }
            }
        }

        let (ra_jnow_deg, dec_jnow_deg) = self
            .model
            .axes_to_jnow(ax1, ax2, now, &self.timescale)
            .unwrap_or((0.0, 0.0));

        Ok(MountRawStatus {
            is_slewing,
            is_tracking: self.tracking && !is_slewing && !in_start_grace,
            is_parked: false,
            at_zero: false,
            gps_locked: self.model.has_site(),
            ra_jnow_deg,
            dec_jnow_deg,
            safety_notice,
        })
    }

    fn goto_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<bool> {
        if !self.model.is_calibrated() {
            return Ok(false);
        }
        let (cur_ax1, cur_ax2) = self.raw.read_axes_rad()?;
        let Some((t1, t2)) = self.model.jnow_to_axes(
            ra_deg,
            dec_deg,
            cur_ax1,
            cur_ax2,
            Utc::now(),
            &self.timescale,
        ) else {
            warn!(
                "KinematicMount rejected GoTo ({:.3}, {:.3}): inside 85° gimbal cone",
                ra_deg, dec_deg
            );
            return Ok(false);
        };

        self.raw.goto_axes_rad(t1, t2)?;
        self.track_target_jnow = Some((ra_deg, dec_deg));
        self.our_slew_started_at = Some(Instant::now());
        self.our_slew_seen = false;
        self.tracking = true;
        Ok(true)
    }

    fn sync_jnow(&mut self, ra_deg: f64, dec_deg: f64) -> IoResult<Result<(), f64>> {
        let (ax1, ax2) = self.raw.read_axes_rad()?;
        let now = Utc::now();
        let was_cal = self.model.is_calibrated();
        let was_samples = self.model.sample_count();

        if was_cal {
            let accepted = self
                .model
                .sync_local(ax1, ax2, ra_deg, dec_deg, now, &self.timescale);
            let Some((rb_ra, rb_dec)) = self.model.axes_to_jnow(ax1, ax2, now, &self.timescale)
            else {
                return Ok(Err(180.0));
            };
            let err = angular_separation_deg(ra_deg, dec_deg, rb_ra, rb_dec);
            if !accepted || err >= SYNC_VERIFY_MAX_ERR_DEG {
                return Ok(Err(err));
            }
            // Update track target only while actively tracking so paused syncs don't leave a stale target.
            if self.tracking {
                self.track_target_jnow = Some((ra_deg, dec_deg));
            }
            return Ok(Ok(()));
        }

        let is_cal = self
            .model
            .add_observation(ax1, ax2, ra_deg, dec_deg, now, &self.timescale);
        if !is_cal {
            if was_samples == 0 && self.model.sample_count() > 0 {
                info!("Class B initial plate solve recorded — GoTo ready");
            } else {
                debug!(
                    "Class B sky model collecting samples ({} stored)",
                    self.model.sample_count()
                );
            }
            return Ok(Ok(()));
        }

        info!("Class B SO(3) sky model calibrated — GoTo & tracking ready");
        Ok(Ok(()))
    }

    fn start_tracking(&mut self) -> IoResult<StartTrackingOutcome> {
        if !self.model.is_calibrated() {
            return Ok(StartTrackingOutcome::Refused);
        }
        let (ax1, ax2) = self.raw.read_axes_rad()?;
        let now = Utc::now();
        self.our_slew_started_at = None;
        self.our_slew_seen = false;
        if !self.tracking || self.track_target_jnow.is_none() {
            self.track_target_jnow = self.model.axes_to_jnow(ax1, ax2, now, &self.timescale);
        }
        self.tracking = self.track_target_jnow.is_some();
        if !self.tracking {
            return Ok(StartTrackingOutcome::Refused);
        }
        if let Some(notice) = self.refresh_tracking(ax1, ax2, now)? {
            return Ok(StartTrackingOutcome::SafetyHalt(notice));
        }
        Ok(StartTrackingOutcome::Started)
    }

    fn stop_tracking(&mut self) -> IoResult<bool> {
        self.abort()?;
        Ok(true)
    }

    fn jog_axes_rad(&mut self, d_ax1: f64, d_ax2: f64) -> IoResult<bool> {
        let (cur_ax1, cur_ax2) = self.raw.read_axes_rad()?;
        self.raw.goto_axes_rad(cur_ax1 + d_ax1, cur_ax2 + d_ax2)?;
        self.tracking = false;
        self.our_slew_started_at = Some(Instant::now());
        self.our_slew_seen = false;
        self.track_target_jnow = None;
        Ok(true)
    }

    fn abort(&mut self) -> IoResult<()> {
        self.tracking = false;
        self.our_slew_started_at = None;
        self.our_slew_seen = false;
        self.track_target_jnow = None;
        self.raw.abort()
    }

    fn unpark(&mut self) -> IoResult<bool> {
        Ok(true)
    }

    fn sync_site_time(&mut self, lat_deg: f64, lon_deg: f64, _utc_ms: i64) -> IoResult<()> {
        self.model.set_site(lat_deg, lon_deg);
        Ok(())
    }
}
