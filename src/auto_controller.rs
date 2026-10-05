//! Adaptive exposure for cameras we fully control (Android Camera2, RPi).
//! UVC cameras never create this; they use the auto-stacker.
//!
//!   Startup ─► Calibrating ─► Tracking (terminal)
//!                ▲      │
//!                └──────┘  retry: sky washed out, or nothing to fall back on
//!
//! Calibrating: for each hardware phase (least → most star-destructive), sweep exposure
//!   upward and keep the shortest exposure that solves reliably. With a saved profile,
//!   try only that phase first (ProfileCheck).
//! Tracking: keep solving with small exposure changes. Dim for bright skies and restore
//!   afterwards. Never recalibrates.
//!
//! Before any of that, each frame passes the settled check. Camera pipelines lag several
//! frames, so frames taken with old settings are ignored. If the camera never delivers a
//! setting, that's a hardware limit: we learn it and clamp all future commands to it.

use crate::HardwarePhase;
use log::{debug, info};
use olive_imu::MotionState;
use olive_imu::storage::PersistentStorage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;

#[derive(Clone, Debug, PartialEq)]
pub enum ControllerAction {
    NoChange,
    SetExposure(i32),
    SetGain(f32),
    SetHotPixelMode(u8),
    SetNoiseReduction(u8),
    SetSigma(f32),
}

#[derive(Clone, Debug, Default)]
pub struct FrameObservation {
    pub actual_exposure_us: i32,
    pub actual_gain: f32,
    pub centroid_counts_per_crop: Vec<usize>,
    pub full_frame_centroids: usize,
    pub background_level: f32,
    pub solve_succeeded: bool,
    pub match_ratio: Option<f32>,
    pub solve_ra: Option<f64>,
    pub solve_dec: Option<f64>,
    pub motion_state: Option<MotionState>,
    pub frame_interval_us: i64,
    pub metadata_valid: bool,
}

fn is_moving(obs: &FrameObservation) -> bool {
    matches!(obs.motion_state, Some(MotionState::Moving))
}

/// Frames to wait for a setting to apply before calling it a hardware limit.
/// Deeper than any HAL pipeline we've seen (4–6 frames).
const UNSETTLED_LIMIT: u32 = 8;
/// First exposure of each sweep phase.
const SWEEP_START_US: i32 = 250_000;
/// Sweep step. Large enough that the settled check can tell old frames from new ones
/// in both directions (1/1.5 < 0.8, and 1.5 > 1.2 + 50 ms for exposures >= 167 ms).
const SWEEP_STEP: f32 = 1.5;
/// Background levels (8-bit mean).
const BG_SATURATED: f32 = 200.0; // washed out: more exposure can't help
const BG_BRIGHT: f32 = 80.0; // dim above this
const BG_DARK: f32 = 50.0; // raise sensitivity only below this; the gap is hysteresis
const DIM_STEP: f32 = 0.9;
/// Gain restore step. Below 80/50 = 1.6, so one step can't jump from dark to bright.
const RESTORE_STEP: f32 = 1.25;
#[derive(Clone, Debug, PartialEq)]
enum Kind {
    ProfileCheck,
    FullSweep,
}

#[derive(Clone, Debug)]
struct Calibration {
    kind: Kind,
    phases: Vec<HardwarePhase>,
    idx: usize,
    solves: u32,
    failures: u32,
    best: Option<(HardwarePhase, i32)>,
    saw_saturation: bool,
}

#[derive(Clone, Debug)]
struct Tracking {
    ema_solve_rate: Option<f32>,
    frames_since_change: u32,
    failures: u32,
}

#[derive(Clone, Debug)]
pub enum State {
    Startup,
    Calibrating,
    Tracking,
}

#[derive(Clone, Debug)]
enum InternalState {
    Startup,
    Calibrating(Calibration),
    Tracking(Tracking),
}

#[derive(Clone, Debug)]
struct Limits {
    exposure_min: i32,
    exposure_max: i32,
    gain_min: f32,
    gain_max: f32,
}

/// Same JSON keys as today's profiles (flattened), so existing files still load.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NightProfile {
    pub camera_id: String,
    pub exposure_us: i32,
    #[serde(flatten)]
    pub phase: HardwarePhase,
    pub saved_at: SystemTime,
}

pub struct AdaptiveController {
    internal_state: InternalState,
    phases: Vec<HardwarePhase>,
    limits: Limits,
    exposure_us: i32,
    gain: f32,
    base_gain: f32,
    unsettled: u32,
    profile: Option<NightProfile>,
    camera_id: String,
    storage: Option<Arc<dyn PersistentStorage>>,
}

impl AdaptiveController {
    pub fn new(
        camera_id: String,
        phases: Vec<HardwarePhase>,
        exposure_range: Option<(i32, i32)>,
        min_gain: f32,
        storage: Option<Arc<dyn PersistentStorage>>,
    ) -> Self {
        let (exposure_min, exposure_max) = exposure_range.unwrap_or((16_667, 5_000_000));

        let profile = storage.as_ref().and_then(|s| {
            s.get(&format!("night_profile_{}", camera_id))
                .and_then(|json| serde_json::from_str::<NightProfile>(&json).ok())
        });

        Self {
            internal_state: InternalState::Startup,
            phases,
            limits: Limits {
                exposure_min,
                exposure_max,
                gain_min: min_gain,
                gain_max: 10_000.0, // High default, will be clamped by hardware limit learning
            },
            exposure_us: SWEEP_START_US,
            gain: min_gain,
            base_gain: min_gain,
            unsettled: 0,
            profile,
            camera_id,
            storage,
        }
    }

    pub fn is_calibrating(&self) -> bool {
        matches!(
            self.internal_state,
            InternalState::Startup | InternalState::Calibrating(_)
        )
    }

    pub fn on_frame(&mut self, obs: &FrameObservation) -> Vec<ControllerAction> {
        if matches!(self.internal_state, InternalState::Startup) {
            return self.start();
        }

        // Moving, or settings unknown: this frame tells us nothing.
        if is_moving(obs) || !obs.metadata_valid {
            return vec![ControllerAction::NoChange];
        }

        if !self.is_settled(obs) {
            self.unsettled += 1;
            if self.unsettled % 5 == 1 {
                log::info!(
                    "Waiting for camera to settle at {}us (measured {}us)",
                    self.exposure_us,
                    self.measured_exposure_us(obs)
                );
            }
            if self.unsettled < UNSETTLED_LIMIT {
                return vec![ControllerAction::NoChange];
            }
            self.learn_limits(obs);
        }
        self.unsettled = 0;

        // Cannot borrow `self` directly in match because the step functions borrow `self` mutably.
        // We temporarily replace the state to appease the borrow checker.
        let state = std::mem::replace(&mut self.internal_state, InternalState::Startup);
        let (next_state, actions) = match state {
            InternalState::Calibrating(cal) => self.calibration_step(cal, obs),
            InternalState::Tracking(trk) => self.tracking_step(trk, obs),
            InternalState::Startup => unreachable!(),
        };
        self.internal_state = next_state;
        actions
    }

    /// Was this frame captured with the settings we last commanded?
    /// Exposure is measured from the sensor frame interval, the ground truth (HAL metadata
    /// can report a new value before it applies). The interval is the *previous* frame's
    /// duration, so a frame passes only after one full frame at the new settings.
    /// 0.8× rejects the old frame after a step up; 1.2× + 50 ms rejects it after a step down
    /// (the 50 ms covers readout and minimum frame time).
    fn is_settled(&self, obs: &FrameObservation) -> bool {
        let measured = self.measured_exposure_us(obs);
        let target = self.exposure_us as i64;

        // Step up check (new frame must be longer)
        if measured < (target * 8 / 10) {
            return false;
        }
        // Step down check (new frame must be shorter)
        if measured > (target * 12 / 10) + 50_000 {
            return false;
        }

        // Gain check
        let diff = obs.actual_gain - self.gain;
        if diff.abs() > (self.gain * 0.15).max(2.0) {
            return false;
        }

        true
    }

    /// The interval if known, else metadata (cameras without timestamps).
    fn measured_exposure_us(&self, obs: &FrameObservation) -> i64 {
        if obs.frame_interval_us > 0 {
            obs.frame_interval_us
        } else {
            obs.actual_exposure_us as i64
        }
    }

    /// The camera has ignored a setting for UNSETTLED_LIMIT frames: that's a hardware limit,
    /// not lag. Adopt what it actually delivers as the new limit. Uses the same measurement
    /// as is_settled, so the next frame passes and this can't repeat.
    fn learn_limits(&mut self, obs: &FrameObservation) {
        let measured = self.measured_exposure_us(obs) as i32;
        let target = self.exposure_us;

        let mut limit_learned = false;

        if measured < (target * 8 / 10) {
            self.limits.exposure_max = measured;
            limit_learned = true;
            info!("Learned exposure_max = {} us", measured);
        } else if measured > (target * 12 / 10) + 50_000 {
            self.limits.exposure_min = measured;
            limit_learned = true;
            info!("Learned exposure_min = {} us", measured);
        }

        if obs.actual_gain < self.gain * 0.85 {
            self.limits.gain_max = obs.actual_gain;
            limit_learned = true;
            info!("Learned gain_max = {}", obs.actual_gain);
        } else if obs.actual_gain > self.gain * 1.15 {
            self.limits.gain_min = obs.actual_gain;
            limit_learned = true;
            info!("Learned gain_min = {}", obs.actual_gain);
        }

        if limit_learned {
            // Re-clamp current commands
            self.exposure_us = self
                .exposure_us
                .clamp(self.limits.exposure_min, self.limits.exposure_max);
            self.gain = self.gain.clamp(self.limits.gain_min, self.limits.gain_max);
            self.base_gain = self
                .base_gain
                .clamp(self.limits.gain_min, self.limits.gain_max);

            // Re-clamp phase list bounds
            for p in &mut self.phases {
                p.gain = p.gain.clamp(self.limits.gain_min, self.limits.gain_max);
            }
        }
    }

    fn set_exposure(&mut self, us: i32) -> ControllerAction {
        self.exposure_us = us.clamp(self.limits.exposure_min, self.limits.exposure_max);
        ControllerAction::SetExposure(self.exposure_us)
    }

    fn set_phase(&mut self, p: &HardwarePhase) -> Vec<ControllerAction> {
        let mut actions = vec![
            ControllerAction::SetHotPixelMode(p.hot_pixel_mode),
            ControllerAction::SetNoiseReduction(p.noise_reduction_mode),
        ];
        self.gain = p.gain.clamp(self.limits.gain_min, self.limits.gain_max);
        self.base_gain = self.gain;
        actions.push(ControllerAction::SetGain(self.gain));
        actions
    }

    /// A phase as the hardware will actually run it (gain clamped to limits).
    fn effective(&self, p: &HardwarePhase) -> HardwarePhase {
        HardwarePhase {
            gain: p.gain.clamp(self.limits.gain_min, self.limits.gain_max),
            hot_pixel_mode: p.hot_pixel_mode,
            noise_reduction_mode: p.noise_reduction_mode,
        }
    }

    fn start(&mut self) -> Vec<ControllerAction> {
        let mut actions = vec![ControllerAction::SetSigma(3.5)];

        if let Some(profile) = self.profile.clone() {
            info!("Starting ProfileCheck with saved profile.");
            let cal = Calibration {
                kind: Kind::ProfileCheck,
                phases: vec![profile.phase.clone()],
                idx: 0,
                solves: 0,
                failures: 0,
                best: None,
                saw_saturation: false,
            };

            let phase_actions = self.set_phase(&profile.phase);
            actions.extend(phase_actions);
            actions.push(self.set_exposure(profile.exposure_us));

            self.internal_state = InternalState::Calibrating(cal);
            actions
        } else {
            info!("Starting FullSweep.");
            let mut cal = Calibration {
                kind: Kind::FullSweep,
                phases: self.phases.clone(),
                idx: 0,
                solves: 0,
                failures: 0,
                best: None,
                saw_saturation: false,
            };

            if cal.phases.is_empty() {
                // Fallback if camera provided no phases
                cal.phases.push(HardwarePhase {
                    gain: self.limits.gain_min,
                    hot_pixel_mode: 0,
                    noise_reduction_mode: 0,
                });
            }

            let phase = cal.phases[0].clone();
            let phase_actions = self.set_phase(&phase);
            actions.extend(phase_actions);
            actions.push(self.set_exposure(SWEEP_START_US));

            self.internal_state = InternalState::Calibrating(cal);
            actions
        }
    }

    /// Fast exposures need more confirmations (a lucky solve is cheap);
    /// slow ones need fewer (each frame costs seconds).
    fn required_solves(us: i32) -> u32 {
        match us {
            ..=500_000 => 5,
            ..=1_500_000 => 3,
            _ => 2,
        }
    }

    fn calibration_step(
        &mut self,
        mut cal: Calibration,
        obs: &FrameObservation,
    ) -> (InternalState, Vec<ControllerAction>) {
        if obs.background_level > BG_SATURATED {
            cal.saw_saturation = true;
        }

        if obs.solve_succeeded {
            cal.solves += 1;
            cal.failures = 0;
            log::info!(
                "Calibration testing {}us: Solved ({} total)",
                self.exposure_us,
                cal.solves
            );
        } else {
            cal.failures += 1;
            cal.solves = 0;
            log::info!(
                "Calibration testing {}us: Failed ({} total)",
                self.exposure_us,
                cal.failures
            );
        }

        if cal.solves >= Self::required_solves(self.exposure_us) {
            let phase = cal.phases[cal.idx].clone();
            let eff_phase = self.effective(&phase);
            if cal.best.is_none() || self.exposure_us < cal.best.as_ref().unwrap().1 {
                debug!("Calibration: found best at {}us", self.exposure_us);
                cal.best = Some((eff_phase, self.exposure_us));
            }
            return self.next_phase(cal, obs);
        }

        if cal.failures >= 2 {
            if cal.saw_saturation {
                debug!(
                    "Calibration: saturated at {}us, skipping rest of phase",
                    self.exposure_us
                );
                return self.next_phase(cal, obs);
            }

            let next_us = (self.exposure_us as f32 * SWEEP_STEP) as i32;

            if self.exposure_us >= self.limits.exposure_max
                || cal.best.as_ref().map_or(false, |b| next_us >= b.1)
            {
                return self.next_phase(cal, obs);
            }

            cal.solves = 0;
            cal.failures = 0;
            cal.saw_saturation = false;
            let act = self.set_exposure(next_us);
            return (InternalState::Calibrating(cal), vec![act]);
        }

        (
            InternalState::Calibrating(cal),
            vec![ControllerAction::NoChange],
        )
    }

    fn next_phase(
        &mut self,
        mut cal: Calibration,
        obs: &FrameObservation,
    ) -> (InternalState, Vec<ControllerAction>) {
        cal.idx += 1;
        let start_us = SWEEP_START_US.clamp(self.limits.exposure_min, self.limits.exposure_max);

        while cal.idx < cal.phases.len() {
            let phase = &cal.phases[cal.idx];
            let eff = self.effective(phase);

            // Skip if this phase can't possibly beat best
            if let Some((_, best_us)) = &cal.best {
                if start_us >= *best_us {
                    cal.idx += 1;
                    continue;
                }
            }

            // Skip if identical to an already tested phase (due to limits)
            if cal.idx > 0 {
                let prev_eff = self.effective(&cal.phases[cal.idx - 1]);
                if eff == prev_eff {
                    cal.idx += 1;
                    continue;
                }
            }

            // Try this phase
            cal.solves = 0;
            cal.failures = 0;
            cal.saw_saturation = false;

            let mut actions = self.set_phase(&eff);
            actions.push(self.set_exposure(start_us));
            return (InternalState::Calibrating(cal), actions);
        }

        self.finish(cal, obs)
    }

    fn finish(
        &mut self,
        cal: Calibration,
        obs: &FrameObservation,
    ) -> (InternalState, Vec<ControllerAction>) {
        if let Some((best_phase, best_us)) = cal.best {
            info!("Calibration successful: {}us", best_us);

            // Save profile
            let profile = NightProfile {
                camera_id: self.camera_id.clone(),
                exposure_us: best_us,
                phase: best_phase.clone(),
                saved_at: SystemTime::now(),
            };

            if let Some(storage) = &self.storage {
                if let Ok(json) = serde_json::to_string(&profile) {
                    storage.set(&format!("night_profile_{}", self.camera_id), &json);
                }
            }

            self.profile = Some(profile);

            // Enter Tracking
            let mut actions = self.set_phase(&best_phase);
            actions.push(self.set_exposure(best_us));

            let trk = Tracking {
                ema_solve_rate: None,
                frames_since_change: 0,
                failures: 0,
            };
            return (InternalState::Tracking(trk), actions);
        }

        // No best found
        if cal.kind == Kind::ProfileCheck {
            info!("ProfileCheck failed. Falling back to FullSweep.");
            self.internal_state = InternalState::Startup;
            let prof = self.profile.take();
            let actions = self.start();
            self.profile = prof;

            let state = std::mem::replace(&mut self.internal_state, InternalState::Startup);
            return (state, actions);
        }

        // FullSweep failed
        if obs.background_level < BG_DARK {
            if let Some(prof) = self.profile.clone() {
                info!(
                    "FullSweep failed, but have profile in dark. Entering Tracking to wait it out."
                );
                let mut actions = self.set_phase(&prof.phase);
                actions.push(self.set_exposure(prof.exposure_us));

                let trk = Tracking {
                    ema_solve_rate: None,
                    frames_since_change: 0,
                    failures: 0,
                };
                return (InternalState::Tracking(trk), actions);
            }
        }

        info!("FullSweep failed, no fallback. Retrying.");
        self.internal_state = InternalState::Startup;
        let actions = self.start();
        let state = std::mem::replace(&mut self.internal_state, InternalState::Startup);
        (state, actions)
    }

    /// Three bands by background level. The 50–80 gap is hysteresis so restoring gain
    /// can't immediately trigger dimming again.
    ///   bright (> 80): dim gain first (keeps exposure short), then exposure.
    ///   hold (50–80):  EMA tuning only. The gentle ×1.1 step can't overshoot the band
    ///                  (65 × 1.1 = 71), but the aggressive ×1.5 failure ramp would
    ///                  (65 × 1.5 = 97), causing the dim/ramp sawtooth oscillation.
    ///   dark (< 50):   full recovery: restore gain first, then the failure ramp and EMA.
    fn tracking_step(
        &mut self,
        mut trk: Tracking,
        obs: &FrameObservation,
    ) -> (InternalState, Vec<ControllerAction>) {
        trk.frames_since_change += 1;

        if obs.solve_succeeded {
            trk.failures = 0;
        } else {
            trk.failures += 1;
        }

        let outcome = if obs.solve_succeeded { 1.0 } else { 0.0 };
        let ema_rate = trk.ema_solve_rate.unwrap_or(outcome);
        trk.ema_solve_rate = Some(0.3 * outcome + 0.7 * ema_rate);

        let bg = obs.background_level;
        let rate = trk.ema_solve_rate.unwrap();

        let mut next_exp = self.exposure_us;
        let mut next_gain = self.gain;

        if bg > BG_BRIGHT {
            if self.gain > self.limits.gain_min {
                next_gain *= DIM_STEP;
            } else {
                next_exp = (self.exposure_us as f32 * DIM_STEP) as i32;
            }
        } else if bg < BG_DARK && self.gain < self.base_gain {
            next_gain = (self.gain * RESTORE_STEP).min(self.base_gain);
        } else if trk.failures >= 2 {
            next_exp = (self.exposure_us as f32 * 1.5) as i32;
        } else if trk.frames_since_change >= 5 {
            if bg >= BG_DARK && bg <= BG_BRIGHT {
                if rate > 0.90 {
                    next_exp = (self.exposure_us as f32 * DIM_STEP) as i32;
                }
            } else if bg < BG_DARK {
                if rate < 0.60 {
                    next_exp = (self.exposure_us as f32 * 1.25) as i32;
                } else if rate > 0.90 {
                    next_exp = (self.exposure_us as f32 * DIM_STEP) as i32;
                }
            }
        }

        let mut actions = Vec::new();

        let clamped_gain = next_gain.clamp(self.limits.gain_min, self.limits.gain_max);
        if clamped_gain != self.gain {
            self.gain = clamped_gain;
            actions.push(ControllerAction::SetGain(self.gain));
        }

        let clamped_exp = next_exp.clamp(self.limits.exposure_min, self.limits.exposure_max);
        if clamped_exp != self.exposure_us {
            self.exposure_us = clamped_exp;
            actions.push(ControllerAction::SetExposure(self.exposure_us));
        }

        if !actions.is_empty() {
            trk.ema_solve_rate = None;
            trk.frames_since_change = 0;
            trk.failures = 0;
        } else {
            actions.push(ControllerAction::NoChange);
        }

        (InternalState::Tracking(trk), actions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct SimCamera {
        exposure_cap: i32,
        gain_cap: f32,
        gain_floor: f32,
        min_frame_us: i64,

        command_q: Vec<(i32, f32)>, // queue of (exposure, gain)
        current_exp: i32,
        current_gain: f32,

        target_exp: i32,
        target_gain: f32,
    }

    impl SimCamera {
        fn new(lag: usize, e_cap: i32, g_cap: f32, g_floor: f32, min_frame: i64) -> Self {
            Self {
                exposure_cap: e_cap,
                gain_cap: g_cap,
                gain_floor: g_floor,
                min_frame_us: min_frame,
                command_q: vec![(SWEEP_START_US, 100.0); lag + 1],
                current_exp: SWEEP_START_US,
                current_gain: 100.0,
                target_exp: SWEEP_START_US,
                target_gain: 100.0,
            }
        }

        fn apply_actions(&mut self, actions: &[ControllerAction]) {
            for a in actions {
                match a {
                    ControllerAction::SetExposure(e) => self.target_exp = *e,
                    ControllerAction::SetGain(g) => self.target_gain = *g,
                    _ => {}
                }
            }
        }

        fn step(&mut self) -> (i32, f32, i64) {
            // Apply physical limits
            let clamped_e = self.target_exp.min(self.exposure_cap);
            let clamped_g = self.target_gain.clamp(self.gain_floor, self.gain_cap);

            self.command_q.push((clamped_e, clamped_g));
            let old = self.command_q.remove(0);

            let previous_duration = (self.current_exp as i64).max(self.min_frame_us);

            self.current_exp = old.0;
            self.current_gain = old.1;

            (self.current_exp, self.current_gain, previous_duration)
        }
    }

    fn default_phases() -> Vec<HardwarePhase> {
        vec![
            HardwarePhase {
                gain: 3200.0,
                hot_pixel_mode: 0,
                noise_reduction_mode: 0,
            },
            HardwarePhase {
                gain: 4800.0,
                hot_pixel_mode: 0,
                noise_reduction_mode: 0,
            },
            HardwarePhase {
                gain: 3200.0,
                hot_pixel_mode: 1,
                noise_reduction_mode: 0,
            },
        ]
    }

    fn make_obs(e: i32, g: f32, interval: i64, bg: f32, stars: usize) -> FrameObservation {
        FrameObservation {
            actual_exposure_us: e,
            actual_gain: g,
            centroid_counts_per_crop: vec![],
            full_frame_centroids: stars,
            background_level: bg,
            solve_succeeded: stars >= 10,
            match_ratio: Some(0.8),
            solve_ra: Some(0.0),
            solve_dec: Some(0.0),
            motion_state: None,
            frame_interval_us: interval,
            metadata_valid: true,
        }
    }

    #[test]
    fn test_t1_t2_lag_and_step_down() {
        let mut ctrl = AdaptiveController::new("test".into(), default_phases(), None, 100.0, None);
        let mut cam = SimCamera::new(3, 5_000_000, 10_000.0, 100.0, 16_667);

        let acts = ctrl.on_frame(&make_obs(
            SWEEP_START_US,
            100.0,
            SWEEP_START_US as i64,
            50.0,
            0,
        ));
        cam.apply_actions(&acts);
        assert!(matches!(ctrl.internal_state, InternalState::Calibrating(_)));

        // Let it start
        for _ in 0..4 {
            let (e, g, i) = cam.step();
            let acts = ctrl.on_frame(&make_obs(e, g, i, 50.0, 0));
            cam.apply_actions(&acts);
        }

        // It failed twice at 250ms, so it stepped to 375ms.
        assert_eq!(cam.target_exp, 375_000);

        // Next 3 frames are stale (still 250ms), should be NoChange and not counted as failures
        for _ in 0..3 {
            let (e, g, i) = cam.step();
            assert_eq!(e, 250_000);
            let acts = ctrl.on_frame(&make_obs(e, g, i, 50.0, 0));
            assert_eq!(acts, vec![ControllerAction::NoChange]);
        }

        // Now 375ms arrives
        let (e, g, i) = cam.step();
        assert_eq!(e, 375_000);
        // Step down test (T2): Force a reset back to 250ms by changing phase
        // We simulate this by manually setting the controller's target
        ctrl.exposure_us = 250_000;
        let obs = make_obs(e, g, i, 50.0, 0); // frame is 375ms, target is 250ms
        assert!(!ctrl.is_settled(&obs)); // Should reject 375ms as stale
    }

    #[test]
    fn test_t3_exposure_cap() {
        let mut ctrl = AdaptiveController::new("test".into(), default_phases(), None, 100.0, None);
        let mut cam = SimCamera::new(0, 1_000_000, 10_000.0, 100.0, 16_667); // cap at 1s

        ctrl.exposure_us = 1_500_000; // Ask for 1.5s

        for _ in 0..10 {
            let (e, g, i) = cam.step(); // Camera delivers 1.0s
            ctrl.on_frame(&make_obs(e, g, i, 50.0, 0));
        }

        // Should have learned limit
        assert_eq!(ctrl.limits.exposure_max, 1_000_000);
    }
}
