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
use super::{
    MountDriver, MountRawStatus, SafetyEvent, StartTrackingOutcome, angular_separation_deg,
};

/// Minimum non-zero tracking velocity below which axes are commanded to halt (rad/s).
pub(super) const MIN_RATE_RAD_PER_SEC: f64 = 1e-7;
/// Maximum closed-loop encoder phase-lock trim added to tracking velocity (10 arcsec/sec).
const MAX_TRACK_TRIM_RAD_PER_SEC: f64 = (10.0 / 3600.0) * PI / 180.0;
const TRACK_TRIM_GAIN: f64 = 0.2;
const SYNC_VERIFY_MAX_ERR_DEG: f64 = 0.25;
const SLEW_START_GRACE: Duration = Duration::from_millis(350);
const MAX_GOTO_CONTINUATION_LEGS: u8 = 4;
const MIN_GOTO_LEG_PROGRESS_RAD: f64 = PI / 180.0;

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
    /// Optional maximum single-command GoTo arc (radians, `< π`) for mounts whose hardware GoTo
    /// takes modulo-2π absolute targets and would otherwise wrap `> π` meridian flips backward.
    fn max_goto_step_rad(&self) -> Option<f64> {
        None
    }
    /// Sets continuous tracking velocities `(rad/sec, rad/sec)` on both axes.
    fn set_rates_rad_per_sec(&mut self, rate1: f64, rate2: f64) -> IoResult<()>;
    /// Commands continuous manual slewing (`axis*_dir` in `-1..=1`, `speed` in `1..=9`),
    /// or stops and waits for deceleration settle when `(0, 0, _)`.
    fn manual_slew(&mut self, axis1_dir: i8, axis2_dir: i8, speed: u8) -> IoResult<()>;
    /// Halts all motion on both axes.
    fn abort(&mut self) -> IoResult<()>;
}

#[derive(Clone, Copy, Debug)]
struct PendingGoto {
    final_t1: f64,
    final_t2: f64,
    remaining_legs: u8,
    prev_max_err: f64,
}

/// Class B mount driver owning the `SO(3)` sky model and dual-axis tracking loop.
pub(super) struct KinematicMount<M: RawAxisMount> {
    raw: M,
    timescale: Arc<Timescale>,
    model: GimbalSkyModel,
    tracking: bool,
    manual_moving: bool,
    pending_goto: Option<PendingGoto>,
    in_continuation_leg: bool,
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
            manual_moving: false,
            pending_goto: None,
            in_continuation_leg: false,
            our_slew_started_at: None,
            our_slew_seen: false,
            track_target_jnow: None,
        }
    }

    fn clear_motion_state(&mut self) {
        self.tracking = false;
        self.manual_moving = false;
        self.pending_goto = None;
        self.in_continuation_leg = false;
        self.our_slew_started_at = None;
        self.our_slew_seen = false;
        self.track_target_jnow = None;
    }

    fn command_goto_leg(
        &mut self,
        cur_ax1: f64,
        cur_ax2: f64,
        final_t1: f64,
        final_t2: f64,
        remaining_legs: u8,
        is_continuation: bool,
    ) -> IoResult<()> {
        let err1 = final_t1 - cur_ax1;
        let err2 = final_t2 - cur_ax2;
        let max_err = err1.abs().max(err2.abs());
        let (leg_t1, leg_t2, next_pending) = match self.raw.max_goto_step_rad() {
            Some(max_step) if max_err > max_step => {
                let d1 = err1.clamp(-max_step, max_step);
                let d2 = err2.clamp(-max_step, max_step);
                (
                    cur_ax1 + d1,
                    cur_ax2 + d2,
                    Some(PendingGoto {
                        final_t1,
                        final_t2,
                        remaining_legs,
                        prev_max_err: max_err,
                    }),
                )
            }
            _ => (final_t1, final_t2, None),
        };
        self.raw.goto_axes_rad(leg_t1, leg_t2)?;
        self.pending_goto = next_pending;
        self.in_continuation_leg = is_continuation;
        self.our_slew_started_at = Some(Instant::now());
        self.our_slew_seen = false;
        Ok(())
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
        self.clear_motion_state();
    }

    fn poll(&mut self) -> IoResult<MountRawStatus> {
        let (ax1, ax2) = self.raw.read_axes_rad()?;
        let now = Utc::now();

        if self.manual_moving {
            let (ra_jnow_deg, dec_jnow_deg) = self
                .model
                .axes_to_jnow(ax1, ax2, now, &self.timescale)
                .unwrap_or((0.0, 0.0));
            return Ok(MountRawStatus {
                is_slewing: true,
                is_tracking: false,
                is_parked: false,
                at_zero: false,
                gps_locked: self.model.has_site(),
                ra_jnow_deg,
                dec_jnow_deg,
                safety_notice: None,
            });
        }

        let mut is_slewing = self.raw.is_slewing()?;

        let mut in_start_grace = false;
        let mut safety_notice = None;
        if is_slewing {
            if self.our_slew_started_at.is_some() {
                self.our_slew_seen = true;
                self.in_continuation_leg = false;
            } else {
                // An external/hand-controller slew invalidates our previous tracking/slew target.
                self.tracking = false;
                self.pending_goto = None;
                self.in_continuation_leg = false;
                self.track_target_jnow = None;
            }
        } else {
            in_start_grace = self
                .our_slew_started_at
                .is_some_and(|t| !self.our_slew_seen && t.elapsed() < SLEW_START_GRACE);
            if in_start_grace && self.in_continuation_leg {
                // Keep reporting slewing during a continuation leg's start grace so the controller
                // (which already observed `slew_seen == true` on leg 1) never exits Slewing early.
                is_slewing = true;
            } else if !in_start_grace {
                self.in_continuation_leg = false;
                if let Some(pending) = self.pending_goto {
                    let cur_err = (pending.final_t1 - ax1)
                        .abs()
                        .max((pending.final_t2 - ax2).abs());
                    let stalled = pending.remaining_legs == 0
                        || (pending.prev_max_err - cur_err) < MIN_GOTO_LEG_PROGRESS_RAD;
                    if stalled {
                        warn!(
                            "Multi-leg GoTo stalled ({:.1}° remaining); aborting remaining legs",
                            cur_err.to_degrees()
                        );
                        self.clear_motion_state();
                        self.raw.abort()?;
                        safety_notice = Some(SafetyEvent::Slew(
                            "GoTo stalled during meridian flip — check mount obstruction",
                        ));
                    } else {
                        self.command_goto_leg(
                            ax1,
                            ax2,
                            pending.final_t1,
                            pending.final_t2,
                            pending.remaining_legs - 1,
                            true,
                        )?;
                        is_slewing = true;
                    }
                } else {
                    self.our_slew_started_at = None;
                    self.our_slew_seen = false;
                    if self.tracking && self.model.is_calibrated() {
                        safety_notice = self
                            .refresh_tracking(ax1, ax2, now)?
                            .map(SafetyEvent::Tracking);
                    }
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

        self.command_goto_leg(cur_ax1, cur_ax2, t1, t2, MAX_GOTO_CONTINUATION_LEGS, false)?;
        self.manual_moving = false;
        self.track_target_jnow = Some((ra_deg, dec_deg));
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
        if self.manual_moving {
            self.raw.manual_slew(0, 0, 0)?;
            self.manual_moving = false;
        }
        let (ax1, ax2) = self.raw.read_axes_rad()?;
        let now = Utc::now();
        self.pending_goto = None;
        self.in_continuation_leg = false;
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
        self.clear_motion_state();
        self.our_slew_started_at = Some(Instant::now());
        Ok(true)
    }

    fn manual_slew(&mut self, axis1_dir: i8, axis2_dir: i8, speed: u8) -> IoResult<()> {
        let active = (axis1_dir != 0 || axis2_dir != 0) && speed > 0;
        self.clear_motion_state();
        self.raw.manual_slew(axis1_dir, axis2_dir, speed)?;
        self.manual_moving = active;
        Ok(())
    }

    fn abort(&mut self) -> IoResult<()> {
        self.clear_motion_state();
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    use super::super::sky_model::earth_vec_to_jnow;

    struct LegSimMount {
        ax1: f64,
        ax2: f64,
        start_delay_polls: usize,
        slewing_polls: usize,
        stalled: bool,
        legs: Vec<(f64, f64)>,
    }

    impl RawAxisMount for LegSimMount {
        fn model_name(&self) -> &str {
            "LegSim"
        }
        fn is_alt_az(&self) -> bool {
            false
        }
        fn read_axes_rad(&mut self) -> IoResult<(f64, f64)> {
            Ok((self.ax1, self.ax2))
        }
        fn is_slewing(&mut self) -> IoResult<bool> {
            if self.start_delay_polls > 0 {
                self.start_delay_polls -= 1;
                Ok(false)
            } else if self.slewing_polls > 0 {
                self.slewing_polls -= 1;
                Ok(true)
            } else {
                Ok(false)
            }
        }
        fn goto_axes_rad(&mut self, ax1_rad: f64, ax2_rad: f64) -> IoResult<()> {
            // Mirror Celestron MC_GOTO_FAST: verify every leg stays strictly < π so hardware
            // can never wrap the wrong way around the circle!
            assert!(
                (ax1_rad - self.ax1).abs() < PI && (ax2_rad - self.ax2).abs() < PI,
                "Single GoTo leg exceeded π: d1={:.1}°, d2={:.1}°",
                (ax1_rad - self.ax1).to_degrees(),
                (ax2_rad - self.ax2).to_degrees()
            );
            self.legs.push((ax1_rad, ax2_rad));
            if !self.stalled {
                self.ax1 = ax1_rad;
                self.ax2 = ax2_rad;
            }
            self.slewing_polls = 1;
            Ok(())
        }
        fn max_goto_step_rad(&self) -> Option<f64> {
            Some(0.75 * PI)
        }
        fn set_rates_rad_per_sec(&mut self, _r1: f64, _r2: f64) -> IoResult<()> {
            Ok(())
        }
        fn manual_slew(&mut self, _a1: i8, _a2: i8, _speed: u8) -> IoResult<()> {
            Ok(())
        }
        fn abort(&mut self) -> IoResult<()> {
            self.start_delay_polls = 0;
            self.slewing_polls = 0;
            Ok(())
        }
    }

    #[test]
    fn test_multileg_goto_splits_over_pole_gem_meridian_flip() {
        let ts = Arc::new(Timescale::default());
        let lat = 37.75_f64;
        let lon = -122.5_f64;
        let now = Utc::now();
        let (lst_deg, _) = earth_vec_to_jnow(
            Vector3::new(lon.to_radians().cos(), lon.to_radians().sin(), 0.0),
            now,
            &ts,
        );

        let east_ax1 = (lon + 40.0).to_radians();
        let east_ax2 = -20.0_f64.to_radians();
        let raw = LegSimMount {
            ax1: east_ax1,
            ax2: east_ax2,
            start_delay_polls: 0,
            slewing_polls: 0,
            stalled: false,
            legs: Vec::new(),
        };
        let mut km = KinematicMount::new(raw, ts.clone(), true);
        km.sync_site_time(lat, lon, 0).unwrap();

        // Calibrate as ideal Northern Hemisphere GEM on East pier side (Dec = -20°).
        let step = 5.0_f64.to_radians();
        for (a1, a2) in [
            (east_ax1, east_ax2),
            (east_ax1, east_ax2 + step),
            (east_ax1 + step, east_ax2 + step),
        ] {
            km.raw.ax1 = a1;
            km.raw.ax2 = a2;
            let u = Vector3::new(a2.cos() * a1.cos(), a2.cos() * a1.sin(), a2.sin());
            let (ra, dec) = earth_vec_to_jnow(u, Utc::now(), &ts);
            assert!(km.sync_jnow(ra, dec).unwrap().is_ok());
        }
        assert!(km.is_calibrated());
        km.raw.ax1 = east_ax1;
        km.raw.ax2 = east_ax2;

        // 1) GoTo West sky Dec = -20° requires a +220° over-the-North-Pole Dec flip (-20° -> +90° -> +200°).
        let west_ra = (lst_deg - 40.0).rem_euclid(360.0);
        assert!(km.goto_jnow(west_ra, -20.0).unwrap());
        assert_eq!(km.raw.legs.len(), 1);
        assert!(km.pending_goto.is_some());

        // Poll 1: leg 1 slewing.
        assert!(km.poll().unwrap().is_slewing);
        // Poll 2: leg 1 done -> commands leg 2 and reports is_slewing=true.
        assert!(km.poll().unwrap().is_slewing);
        assert_eq!(km.raw.legs.len(), 2);
        assert!(km.pending_goto.is_none());
        // Simulate MC start delay on leg 2: raw.is_slewing() returns false for 1 poll while in_start_grace!
        km.raw.start_delay_polls = 1;
        // Poll 3: leg 2 still in start_delay (raw.is_slewing() == false), MUST still report is_slewing=true!
        assert!(km.poll().unwrap().is_slewing);
        // Poll 4: leg 2 actively slewing; Poll 5: leg 2 done -> lands at +200° and starts tracking!
        assert!(km.poll().unwrap().is_slewing);
        let st = km.poll().unwrap();
        assert!(!st.is_slewing);
        assert!(st.is_tracking);
        assert!(
            (km.raw.ax2.to_degrees() - 200.0).abs() < 0.5,
            "Multi-leg GEM flip did not land at +200°: {:.1}°",
            km.raw.ax2.to_degrees()
        );

        // 2) Motor stall on leg 1 must abort cleanly with a safety notice instead of looping forever.
        km.raw.ax1 = east_ax1;
        km.raw.ax2 = east_ax2;
        km.raw.stalled = true;
        km.raw.legs.clear();
        assert!(km.goto_jnow(west_ra, -20.0).unwrap());
        assert!(km.poll().unwrap().is_slewing);
        let stalled_st = km.poll().unwrap();
        assert!(!stalled_st.is_slewing);
        assert!(!stalled_st.is_tracking);
        assert!(stalled_st.safety_notice.is_some());
        assert!(km.pending_goto.is_none());
        assert_eq!(km.raw.legs.len(), 1);
    }
}
