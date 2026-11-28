// Copyright (c) 2026 Omair Kamil
// See LICENSE.md in the repository root for license terms.

//! Universal two-axis `SO(3)` kinematic sky model for Class B mounts.
//!
//! Operates in the Earth-fixed Greenwich equatorial frame (`GAST` from UTC),
//! so a stationary mount's two orthogonal mechanical axes map to a single
//! constant rotation matrix `R` + elevation zero offset `theta2_0` regardless
//! of whether the mount is Alt-Az, tilted Alt-Az, or Equatorial, and without
//! requiring GPS site coordinates.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use chrono::{DateTime, TimeDelta, Utc};
use nalgebra::{Matrix3, Vector3};
use starfield::time::Timescale;

const MAX_SAMPLES: usize = 6;
const MIN_ENCODER_SEPARATION_RAD: f64 = 0.035; // ~2.0 deg
const MIN_ENCODER_AREA_RAD2: f64 = 1e-3;
const WIDE_BASELINE_SPAN_RAD: f64 = 15.0 * PI / 180.0;
const MAX_FIT_RMS_RAD: f64 = 0.035; // ~2.0 deg sanity ceiling
const MAX_AXIS2_ELEVATION_RAD: f64 = 85.0 * PI / 180.0;
const MERIDIAN_GOTO_HYSTERESIS_RAD: f64 = 2.0 * PI / 180.0;
const MAX_MERIDIAN_PAST_RAD: f64 = 10.0 * PI / 180.0;
const COARSE_THETA2_STEPS: usize = 72;
const GOLDEN_ITERS: usize = 28;
const GOLDEN_RATIO_INV: f64 = 0.618_033_988_749_894_9;

#[derive(Clone, Copy, Debug)]
struct Sample {
    ax1: f64,
    ax2: f64,
    u_earth: Vector3<f64>,
}

#[derive(Clone, Copy, Debug)]
struct FitSolution {
    rot: Matrix3<f64>,
    theta2_0: f64,
    s1: f64,
    s2: f64,
    rms_rad: f64,
    mean_abs_elev: f64,
    s0_unflipped: bool,
}

fn is_better_fit(cand: &FitSolution, prev: Option<&FitSolution>) -> bool {
    match prev {
        None => true,
        Some(p) => {
            if (cand.rms_rad - p.rms_rad).abs() <= 1e-6 {
                if cand.s0_unflipped != p.s0_unflipped {
                    cand.s0_unflipped
                } else {
                    cand.mean_abs_elev < p.mean_abs_elev
                }
            } else {
                cand.rms_rad < p.rms_rad
            }
        }
    }
}

fn rz_pi() -> Matrix3<f64> {
    Matrix3::new(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
}

fn rx_pi() -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
}

fn try_record_fit(
    best: &mut Option<FitSolution>,
    samples: &[Sample],
    rot: Matrix3<f64>,
    theta2_0: f64,
    s1: f64,
    s2: f64,
    rms_rad: f64,
) {
    // Evaluate both this parameterization and its exact 180°-flipped twin (identical rms_rad)
    // so both hemispheres always compete on equal footing regardless of candidate source.
    let pairs = [
        (rot, theta2_0, s2),
        (rot * rz_pi(), wrap_pi(theta2_0 + PI), -s2),
    ];
    for (r, th2_0, sign2) in pairs {
        let mean_abs_elev = samples
            .iter()
            .map(|s| wrap_pi(sign2 * (s.ax2 - th2_0)).abs())
            .sum::<f64>()
            / samples.len() as f64;
        let s0_unflipped = wrap_pi(sign2 * (samples[0].ax2 - th2_0)).abs() <= FRAC_PI_2;
        if s0_unflipped || mean_abs_elev <= FRAC_PI_2 {
            let cand = FitSolution {
                rot: r,
                theta2_0: th2_0,
                s1,
                s2: sign2,
                rms_rad,
                mean_abs_elev,
                s0_unflipped,
            };
            if is_better_fit(&cand, best.as_ref()) {
                *best = Some(cand);
            }
        }
    }
}

fn align_hemisphere(mut sol: FitSolution, prev: &FitSolution) -> FitSolution {
    // Normalize across all 4 exact SO(3) twins: first match `s1` via Rx(π), then `s2`/`theta2_0` via Rz(π).
    if sol.s1 != prev.s1 {
        sol.rot *= rx_pi();
        sol.s1 = -sol.s1;
        sol.s2 = -sol.s2;
    }
    if sol.s2 != prev.s2 && wrap_pi(sol.theta2_0 - prev.theta2_0).abs() > FRAC_PI_2 {
        sol.rot *= rz_pi();
        sol.theta2_0 = wrap_pi(sol.theta2_0 + PI);
        sol.s2 = -sol.s2;
    }
    sol
}

fn jnow_to_earth_vec(
    ra_jnow_deg: f64,
    dec_jnow_deg: f64,
    time: DateTime<Utc>,
    ts: &Timescale,
) -> Vector3<f64> {
    let gast_deg = ts.from_datetime(time).gast() * 15.0;
    let lon_rad = (ra_jnow_deg - gast_deg).to_radians();
    let dec_rad = dec_jnow_deg.to_radians();
    Vector3::new(
        dec_rad.cos() * lon_rad.cos(),
        dec_rad.cos() * lon_rad.sin(),
        dec_rad.sin(),
    )
}

pub(super) fn earth_vec_to_jnow(
    u: Vector3<f64>,
    time: DateTime<Utc>,
    ts: &Timescale,
) -> (f64, f64) {
    let gast_deg = ts.from_datetime(time).gast() * 15.0;
    let lon_deg = u.y.atan2(u.x).to_degrees();
    let ra_deg = (lon_deg + gast_deg).rem_euclid(360.0);
    let dec_deg = u.z.clamp(-1.0, 1.0).asin().to_degrees();
    (ra_deg, dec_deg)
}

/// Wraps an angle delta into `(-PI, PI]`.
pub(super) fn wrap_pi(angle: f64) -> f64 {
    let mut a = (angle + PI).rem_euclid(TAU) - PI;
    if a <= -PI {
        a += TAU;
    }
    a
}

fn mount_vec(ax1: f64, ax2: f64, theta2_0: f64, s1: f64, s2: f64) -> Vector3<f64> {
    let phi1 = s1 * ax1;
    let phi2 = s2 * (ax2 - theta2_0);
    Vector3::new(phi2.cos() * phi1.cos(), phi2.cos() * phi1.sin(), phi2.sin())
}

/// Exact Kabsch SVD solver finding optimal `R in SO(3)` minimizing `sum ||R * m_i - u_i||^2`.
fn kabsch_so3(samples: &[Sample], theta2_0: f64, s1: f64, s2: f64) -> (Matrix3<f64>, f64) {
    let mut h = Matrix3::<f64>::zeros();
    let mut mount_vecs = Vec::with_capacity(samples.len());
    for s in samples {
        let m = mount_vec(s.ax1, s.ax2, theta2_0, s1, s2);
        h += m * s.u_earth.transpose();
        mount_vecs.push(m);
    }

    let svd = h.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return (Matrix3::identity(), f64::INFINITY);
    };
    let v = v_t.transpose();
    let mut d = Matrix3::identity();
    if (v * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let rot = v * d * u.transpose();

    let mut sq_err = 0.0;
    for (s, m) in samples.iter().zip(mount_vecs.iter()) {
        sq_err += (rot * m - s.u_earth).norm_squared();
    }
    (rot, (sq_err / samples.len() as f64).sqrt())
}

fn has_non_collinear_geometry(samples: &[Sample]) -> bool {
    if samples.len() < 3 {
        return false;
    }
    for i in 0..samples.len() {
        let si = samples[i];
        for j in (i + 1)..samples.len() {
            let d1_j = wrap_pi(samples[j].ax1 - si.ax1);
            let d2_j = wrap_pi(samples[j].ax2 - si.ax2);
            let e1 = samples[j].u_earth - si.u_earth;
            for k in (j + 1)..samples.len() {
                let d1_k = wrap_pi(samples[k].ax1 - si.ax1);
                let d2_k = wrap_pi(samples[k].ax2 - si.ax2);
                let enc_area = (d1_j * d2_k - d2_j * d1_k).abs();
                let sky_area = e1.cross(&(samples[k].u_earth - si.u_earth)).norm();
                if enc_area > MIN_ENCODER_AREA_RAD2 && sky_area > 2e-4 {
                    return true;
                }
            }
        }
    }
    false
}

fn solve_samples(samples: &[Sample]) -> Option<FitSolution> {
    let step = TAU / COARSE_THETA2_STEPS as f64;
    let mut best: Option<FitSolution> = None;

    for &s1 in &[1.0_f64, -1.0] {
        for &s2 in &[1.0_f64, -1.0] {
            let mut best_k = 0usize;
            let mut best_err = f64::INFINITY;
            for k in 0..COARSE_THETA2_STEPS {
                let th2_0 = -PI + k as f64 * step;
                let (_, err) = kabsch_so3(samples, th2_0, s1, s2);
                if err < best_err {
                    best_err = err;
                    best_k = k;
                }
            }

            let mut a = -PI + (best_k as f64 - 1.0) * step;
            let mut b = -PI + (best_k as f64 + 1.0) * step;
            let mut c = b - GOLDEN_RATIO_INV * (b - a);
            let mut d = a + GOLDEN_RATIO_INV * (b - a);
            let mut fc = kabsch_so3(samples, c, s1, s2).1;
            let mut fd = kabsch_so3(samples, d, s1, s2).1;

            for _ in 0..GOLDEN_ITERS {
                if fc < fd {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - GOLDEN_RATIO_INV * (b - a);
                    fc = kabsch_so3(samples, c, s1, s2).1;
                } else {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + GOLDEN_RATIO_INV * (b - a);
                    fd = kabsch_so3(samples, d, s1, s2).1;
                }
            }

            let th2_0 = wrap_pi(0.5 * (a + b));
            let (rot, rms_rad) = kabsch_so3(samples, th2_0, s1, s2);
            try_record_fit(&mut best, samples, rot, th2_0, s1, s2, rms_rad);
        }
    }

    best.filter(|sol| sol.rms_rad <= MAX_FIT_RMS_RAD)
}

fn zenith_earth_vec(lat_deg: f64, lon_deg: f64) -> Vector3<f64> {
    let (sl, cl) = lat_deg.to_radians().sin_cos();
    let (so, co) = lon_deg.to_radians().sin_cos();
    Vector3::new(cl * co, cl * so, sl)
}

fn basis_from_z(z: Vector3<f64>) -> Matrix3<f64> {
    let ref_v = if z.z.abs() < 0.9 {
        Vector3::z()
    } else {
        Vector3::x()
    };
    let x = ref_v.cross(&z).normalize();
    let y = z.cross(&x);
    Matrix3::from_columns(&[x, y, z])
}

fn max_sky_span_rad(samples: &[Sample]) -> f64 {
    let mut max_span = 0.0_f64;
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let dot = samples[i].u_earth.dot(&samples[j].u_earth).clamp(-1.0, 1.0);
            max_span = max_span.max(dot.acos());
        }
    }
    max_span
}

/// Closed-form fit constraining Axis 1 to a known physical unit vector `axis1_earth`
/// (local Zenith for Alt-Az, or Earth rotation axis for Equatorial). Immune to gear
/// backlash overfitting and `±phi2` flips during small-step calibration probes.
fn solve_constrained_prior(samples: &[Sample], axis1_earth: Vector3<f64>) -> Option<FitSolution> {
    let basis = basis_from_z(axis1_earth);
    let local: Vec<(f64, f64, f64)> = samples
        .iter()
        .map(|s| {
            let v = basis.transpose() * s.u_earth;
            let elev = v.z.clamp(-1.0, 1.0).asin();
            let az = v.y.atan2(v.x);
            let horiz = v.x.hypot(v.y);
            (elev, az, horiz)
        })
        .collect();

    let mut best: Option<FitSolution> = None;
    let mut candidates = Vec::with_capacity(2 * samples.len() + 1);
    for &s1 in &[1.0_f64, -1.0] {
        for &s2 in &[1.0_f64, -1.0] {
            candidates.clear();
            let (mut th_sin, mut th_cos) = (0.0, 0.0);
            let (mut al_sin, mut al_cos) = (0.0, 0.0);
            for (s, &(elev, az, horiz)) in samples.iter().zip(local.iter()) {
                let (ts, tc) = (s.ax2 - s2 * elev).sin_cos();
                th_sin += ts;
                th_cos += tc;
                let (as_, ac) = (az - s1 * s.ax1).sin_cos();
                al_sin += horiz * as_;
                al_cos += horiz * ac;
                // Emit both unflipped and over-pole folded anchors so probes stepping across
                // a pole recover the unflipped pre-probe posture rather than the 180° twin.
                if horiz > 0.02 {
                    candidates.push((wrap_pi(s.ax2 - s2 * elev), wrap_pi(az - s1 * s.ax1)));
                    let elev_folded = if elev >= 0.0 { PI - elev } else { -PI - elev };
                    candidates.push((
                        wrap_pi(s.ax2 - s2 * elev_folded),
                        wrap_pi(az + PI - s1 * s.ax1),
                    ));
                }
            }
            if th_sin.hypot(th_cos) >= 1e-6 && al_sin.hypot(al_cos) >= 1e-6 {
                candidates.push((wrap_pi(th_sin.atan2(th_cos)), al_sin.atan2(al_cos)));
            }

            for &(theta2_0, alpha) in &candidates {
                let (sa, ca) = alpha.sin_cos();
                let rz = Matrix3::new(ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0);
                let rot = basis * rz;

                let mut sq_err = 0.0;
                for s in samples {
                    let m = mount_vec(s.ax1, s.ax2, theta2_0, s1, s2);
                    sq_err += (rot * m - s.u_earth).norm_squared();
                }
                let rms_rad = (sq_err / samples.len() as f64).sqrt();
                try_record_fit(&mut best, samples, rot, theta2_0, s1, s2, rms_rad);
            }
        }
    }

    best.filter(|sol| sol.rms_rad <= MAX_FIT_RMS_RAD)
}

fn solve_constrained_priors(samples: &[Sample], lat_deg: f64, lon_deg: f64) -> Option<FitSolution> {
    let aa = solve_constrained_prior(samples, zenith_earth_vec(lat_deg, lon_deg));
    let eq = solve_constrained_prior(samples, Vector3::z());
    match (aa, eq) {
        (Some(a), Some(e)) => Some(if a.rms_rad <= e.rms_rad { a } else { e }),
        (Some(a), None) => Some(a),
        (None, Some(e)) => Some(e),
        (None, None) => None,
    }
}

/// Selects which index to evict when the sample ring is full so that adding `new_sample`
/// preserves both 2D non-collinear geometry and maximum pairwise sky span.
fn choose_eviction_index(samples: &[Sample], new_sample: Sample) -> usize {
    let mut best_idx = 0usize;
    let mut best_non_collinear = false;
    let mut best_span = -1.0_f64;
    let mut scratch = Vec::with_capacity(samples.len());
    for skip in 0..samples.len() {
        scratch.clear();
        for (i, &s) in samples.iter().enumerate() {
            if i != skip {
                scratch.push(s);
            }
        }
        scratch.push(new_sample);
        let non_collinear = has_non_collinear_geometry(&scratch);
        let span = max_sky_span_rad(&scratch);
        let better = match (non_collinear, best_non_collinear) {
            (true, false) => true,
            (false, true) => false,
            _ => span > best_span + 1e-9,
        };
        if better {
            best_non_collinear = non_collinear;
            best_span = span;
            best_idx = skip;
        }
    }
    best_idx
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TrackingHaltReason {
    GimbalCone,
    MeridianLimit,
}

#[derive(Clone, Copy, Debug)]
struct GemMeridianState {
    branch_a_is_east: bool,
    pole_pos_phi2: bool,
    ax1_cw_down: f64,
    ax2_elevated_pole: f64,
}

fn ha_earth_rad(u_earth: Vector3<f64>, lon_deg: f64) -> f64 {
    let lon_u = u_earth.y.atan2(u_earth.x);
    wrap_pi(lon_deg.to_radians() - lon_u)
}

fn compute_gem_state(
    sol: &FitSolution,
    samples: &[Sample],
    lat_deg: f64,
    lon_deg: f64,
    prev: Option<&GemMeridianState>,
) -> Option<GemMeridianState> {
    let (branch_a_is_east, ref_ax1, ref_ax2) = if let Some(p) = prev {
        (p.branch_a_is_east, p.ax1_cw_down, p.ax2_elevated_pole)
    } else {
        // Pick the sample farthest from both the celestial pole and the meridian.
        let p_ref = samples.iter().copied().max_by(|a, b| {
            let score = |s: &Sample| {
                let horiz = s.u_earth.x.hypot(s.u_earth.y);
                horiz * ha_earth_rad(s.u_earth, lon_deg).sin().abs()
            };
            score(a).total_cmp(&score(b))
        })?;
        let horiz = p_ref.u_earth.x.hypot(p_ref.u_earth.y);
        let ha = ha_earth_rad(p_ref.u_earth, lon_deg);
        if horiz * ha.sin().abs() < 1e-3 {
            return None;
        }
        let on_branch_a = wrap_pi(sol.s2 * (p_ref.ax2 - sol.theta2_0)).abs() <= FRAC_PI_2;
        let is_east = ha < 0.0;
        (on_branch_a == is_east, p_ref.ax1, p_ref.ax2)
    };

    // Counterweight-straight-down sky vector on Branch A is East horizon (HA=-6h) if Branch A
    // serves East, or West horizon (HA=+6h) if Branch A serves West.
    let (so, co) = lon_deg.to_radians().sin_cos();
    let u_cw = if branch_a_is_east {
        Vector3::new(-so, co, 0.0)
    } else {
        Vector3::new(so, -co, 0.0)
    };
    let m_cw = sol.rot.transpose() * u_cw;
    if m_cw.x.hypot(m_cw.y) < 1e-6 {
        return None;
    }
    let phi1_cw_down = m_cw.y.atan2(m_cw.x);
    let ax1_cw_down = ref_ax1 + wrap_pi(sol.s1 * phi1_cw_down - ref_ax1);

    // Determine whether the elevated celestial pole sits at +PI/2 or -PI/2 in mount `phi2`
    // from the fitted Axis-1 orientation (handles Rx(π) twins where `(rot * e_z).z < 0`).
    let axis1_z = (sol.rot * Vector3::z()).z;
    let pole_pos_phi2 = (axis1_z >= 0.0) == (lat_deg >= 0.0);
    let elev_pole_phi2 = if pole_pos_phi2 { FRAC_PI_2 } else { -FRAC_PI_2 };
    let ax2_elevated_pole = ref_ax2 + wrap_pi((sol.theta2_0 + sol.s2 * elev_pole_phi2) - ref_ax2);

    Some(GemMeridianState {
        branch_a_is_east,
        pole_pos_phi2,
        ax1_cw_down,
        ax2_elevated_pole,
    })
}

/// Fitted two-axis gimbal sky model supporting forward/inverse kinematics and rate tracking.
#[derive(Default)]
pub(super) struct GimbalSkyModel {
    samples: Vec<Sample>,
    solution: Option<FitSolution>,
    site: Option<(f64, f64)>,
    gem_meridian_flip: bool,
    gem_state: Option<GemMeridianState>,
}

impl GimbalSkyModel {
    pub(super) fn is_calibrated(&self) -> bool {
        self.solution.is_some()
    }

    pub(super) fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub(super) fn set_gem_meridian_flip(&mut self, enabled: bool) {
        self.gem_meridian_flip = enabled;
        self.refresh_gem_state();
    }

    pub(super) fn set_site(&mut self, lat_deg: f64, lon_deg: f64) {
        if lat_deg.is_finite() && lon_deg.is_finite() {
            self.site = Some((lat_deg, lon_deg));
            self.refresh_gem_state();
        }
    }

    pub(super) fn has_site(&self) -> bool {
        self.site.is_some()
    }

    fn refresh_gem_state(&mut self) {
        if !self.gem_meridian_flip || self.detected_alt_az() != Some(false) {
            self.gem_state = None;
            return;
        }
        let (Some(sol), Some((lat, lon))) = (self.solution.as_ref(), self.site) else {
            self.gem_state = None;
            return;
        };
        if let Some(updated) =
            compute_gem_state(sol, &self.samples, lat, lon, self.gem_state.as_ref())
        {
            self.gem_state = Some(updated);
        }
    }

    /// Infers physical Alt-Az (`Some(true)`) vs Equatorial (`Some(false)`) mode from the
    /// fitted mechanical Axis-1 orientation when both a fit and site coordinates exist.
    pub(super) fn detected_alt_az(&self) -> Option<bool> {
        let sol = self.solution?;
        let (lat, lon) = self.site?;
        let axis1 = sol.rot * Vector3::z();
        let aa_align = axis1.dot(&zenith_earth_vec(lat, lon)).abs();
        let eq_align = axis1.z.abs();
        Some(aa_align >= eq_align)
    }

    /// Clears all calibration samples and resets the fitted model (preserving site location).
    pub(super) fn reset(&mut self) {
        self.samples.clear();
        self.solution = None;
        self.gem_state = None;
    }

    /// Records a plate-solved `(ax1, ax2) <-> (ra_jnow, dec_jnow)` observation and refits when
    /// at least 3 non-collinear samples exist. Returns `true` if calibrated after the update.
    pub(super) fn add_observation(
        &mut self,
        ax1: f64,
        ax2: f64,
        ra_jnow_deg: f64,
        dec_jnow_deg: f64,
        time: DateTime<Utc>,
        ts: &Timescale,
    ) -> bool {
        self.add_observation_inner(ax1, ax2, ra_jnow_deg, dec_jnow_deg, time, ts);
        self.solution.is_some()
    }

    /// Returns `true` if the observation was accepted into `self.samples` (not rejected as an outlier).
    fn add_observation_inner(
        &mut self,
        ax1: f64,
        ax2: f64,
        ra_jnow_deg: f64,
        dec_jnow_deg: f64,
        time: DateTime<Utc>,
        ts: &Timescale,
    ) -> bool {
        if !(ax1.is_finite()
            && ax2.is_finite()
            && ra_jnow_deg.is_finite()
            && dec_jnow_deg.is_finite())
        {
            return false;
        }
        let u_earth = jnow_to_earth_vec(ra_jnow_deg, dec_jnow_deg, time, ts);
        let new_sample = Sample { ax1, ax2, u_earth };

        let mut candidate = self.samples.clone();
        if let Some(existing) = candidate.iter_mut().find(|s| {
            let d1 = wrap_pi(ax1 - s.ax1).abs();
            let d2 = wrap_pi(ax2 - s.ax2).abs();
            d1.hypot(d2) < MIN_ENCODER_SEPARATION_RAD
        }) {
            *existing = new_sample;
        } else {
            if candidate.len() >= MAX_SAMPLES {
                let evict = choose_eviction_index(&candidate, new_sample);
                candidate.remove(evict);
            }
            candidate.push(new_sample);
        }

        if has_non_collinear_geometry(&candidate) {
            let fitted = if max_sky_span_rad(&candidate) >= WIDE_BASELINE_SPAN_RAD {
                solve_samples(&candidate)
            } else if let Some((lat, lon)) = self.site {
                solve_constrained_priors(&candidate, lat, lon).or_else(|| solve_samples(&candidate))
            } else {
                solve_samples(&candidate)
            };

            if let Some(sol) = fitted {
                let aligned = match self.solution {
                    Some(ref prev) => align_hemisphere(sol, prev),
                    None => sol,
                };
                self.samples = candidate;
                self.solution = Some(aligned);
                self.refresh_gem_state();
                return true;
            }
            if self.solution.is_none() {
                // Keep accumulating until a valid 3-point geometry fits cleanly.
                self.samples = candidate;
                return true;
            }
            // Calibrated model rejected outlier sample: do not poison `self.samples` or `self.solution`.
            return false;
        }

        if let Some(sol) = self.solution {
            // Candidate lost 2D area (e.g. an in-place update shrunk a tight 3-point triangle);
            // preserve `self.samples` and still accept valid solves within the RMS sanity ceiling.
            let pred_u = sol.rot * mount_vec(ax1, ax2, sol.theta2_0, sol.s1, sol.s2);
            let err_rad = pred_u.dot(&u_earth).clamp(-1.0, 1.0).acos();
            return err_rad <= MAX_FIT_RMS_RAD;
        }
        self.samples = candidate;
        true
    }

    /// Applies an exact single-point local sync rotation when already calibrated,
    /// ensuring the mount agrees with the latest plate solve to sub-arcsecond accuracy.
    /// Rejects divergent outlier solves without corrupting the fitted model.
    pub(super) fn sync_local(
        &mut self,
        ax1: f64,
        ax2: f64,
        ra_jnow_deg: f64,
        dec_jnow_deg: f64,
        time: DateTime<Utc>,
        ts: &Timescale,
    ) -> bool {
        if !self.add_observation_inner(ax1, ax2, ra_jnow_deg, dec_jnow_deg, time, ts) {
            return false;
        }
        let Some(mut sol) = self.solution else {
            return false;
        };
        let target_u = jnow_to_earth_vec(ra_jnow_deg, dec_jnow_deg, time, ts);
        let pred_u = sol.rot * mount_vec(ax1, ax2, sol.theta2_0, sol.s1, sol.s2);
        let axis = pred_u.cross(&target_u);
        let sin_a = axis.norm();
        let cos_a = pred_u.dot(&target_u).clamp(-1.0, 1.0);
        let rot_params = if sin_a > 1e-12 {
            Some((axis / sin_a, sin_a.atan2(cos_a)))
        } else if cos_a < 0.0 {
            Some((basis_from_z(pred_u).column(0).into_owned(), PI))
        } else {
            None
        };
        if let Some((k, angle)) = rot_params {
            let k_mat = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
            let delta_r =
                Matrix3::identity() + k_mat * angle.sin() + k_mat * k_mat * (1.0 - angle.cos());
            sol.rot = delta_r * sol.rot;
            self.solution = Some(sol);
            self.refresh_gem_state();
        }
        true
    }

    /// Forward kinematics: converts raw encoder angles `(ax1, ax2)` to JNow `(ra_deg, dec_deg)`.
    pub(super) fn axes_to_jnow(
        &self,
        ax1: f64,
        ax2: f64,
        time: DateTime<Utc>,
        ts: &Timescale,
    ) -> Option<(f64, f64)> {
        let sol = self.solution?;
        let u = sol.rot * mount_vec(ax1, ax2, sol.theta2_0, sol.s1, sol.s2);
        Some(earth_vec_to_jnow(u, time, ts))
    }

    fn jnow_to_axes_inner(
        &self,
        ra_jnow_deg: f64,
        dec_jnow_deg: f64,
        cur_ax1: f64,
        cur_ax2: f64,
        time: DateTime<Utc>,
        ts: &Timescale,
        allow_meridian_flip: bool,
    ) -> Option<(f64, f64)> {
        if !(ra_jnow_deg.is_finite()
            && dec_jnow_deg.is_finite()
            && cur_ax1.is_finite()
            && cur_ax2.is_finite())
        {
            return None;
        }
        let sol = self.solution?;
        let u = jnow_to_earth_vec(ra_jnow_deg, dec_jnow_deg, time, ts);
        let m = sol.rot.transpose() * u;
        let phi2_a = m.z.clamp(-1.0, 1.0).asin();
        if !(phi2_a.abs() <= MAX_AXIS2_ELEVATION_RAD) {
            return None;
        }
        let phi1_a = m.y.atan2(m.x);

        let cur_on_branch_a = wrap_pi(sol.s2 * (cur_ax2 - sol.theta2_0)).abs() <= FRAC_PI_2;
        let (phi1, phi2, gem_flip_anchor) = match (self.gem_state, self.site) {
            (Some(gem), Some((_, lon))) => {
                let (want_a, anchor) = if allow_meridian_flip {
                    let ha = ha_earth_rad(u, lon);
                    let near_meridian = ha.abs() < MERIDIAN_GOTO_HYSTERESIS_RAD
                        || (PI - ha.abs()) < MERIDIAN_GOTO_HYSTERESIS_RAD;
                    let want_a = if near_meridian {
                        cur_on_branch_a
                    } else {
                        (ha < 0.0) == gem.branch_a_is_east
                    };
                    (want_a, Some(gem))
                } else {
                    (cur_on_branch_a, None)
                };
                if want_a {
                    (phi1_a, phi2_a, anchor)
                } else {
                    let phi2_b = if gem.pole_pos_phi2 {
                        PI - phi2_a
                    } else {
                        -PI - phi2_a
                    };
                    (wrap_pi(phi1_a + PI), phi2_b, anchor)
                }
            }
            _ => (phi1_a, phi2_a, None),
        };

        if let Some(gem) = gem_flip_anchor {
            // Align both anchors to the current TAU revolution so full-turn manual slews or
            // [0, 2π) startup offsets never cause a 360° unwind or underground Dec flip.
            let ax1_cw = gem.ax1_cw_down + ((cur_ax1 - gem.ax1_cw_down) / TAU).round() * TAU;
            let ax2_pole =
                gem.ax2_elevated_pole + ((cur_ax2 - gem.ax2_elevated_pole) / TAU).round() * TAU;
            let target_ax1 = ax1_cw + wrap_pi(sol.s1 * phi1 - ax1_cw);
            let pole_offset = if gem.pole_pos_phi2 {
                phi2 - FRAC_PI_2
            } else {
                phi2 + FRAC_PI_2
            };
            let target_ax2 = ax2_pole + sol.s2 * pole_offset;
            return Some((target_ax1, target_ax2));
        }

        let target_ax1 = cur_ax1 + wrap_pi(sol.s1 * phi1 - cur_ax1);
        let target_ax2 = cur_ax2 + wrap_pi((sol.theta2_0 + sol.s2 * phi2) - cur_ax2);
        Some((target_ax1, target_ax2))
    }

    /// Inverse kinematics for GoTo: enforces the `±85°` gimbal cone and (for Class B GEMs)
    /// selects the counterweight-down pier side with safe over-pole / CW-down unwrapping.
    pub(super) fn jnow_to_axes(
        &self,
        ra_jnow_deg: f64,
        dec_jnow_deg: f64,
        cur_ax1: f64,
        cur_ax2: f64,
        time: DateTime<Utc>,
        ts: &Timescale,
    ) -> Option<(f64, f64)> {
        self.jnow_to_axes_inner(ra_jnow_deg, dec_jnow_deg, cur_ax1, cur_ax2, time, ts, true)
    }

    /// Computes target axes at `time` AND instantaneous dual-axis tracking rates `(rad/s)`
    /// on the current pier side, halting cleanly at the `85°` gimbal cone or `10°` GEM meridian limit.
    pub(super) fn tracking_target_and_rates(
        &self,
        ra_jnow_deg: f64,
        dec_jnow_deg: f64,
        cur_ax1: f64,
        cur_ax2: f64,
        time: DateTime<Utc>,
        ts: &Timescale,
    ) -> Result<((f64, f64), (f64, f64)), TrackingHaltReason> {
        let (a1_now, a2_now) = self
            .jnow_to_axes_inner(ra_jnow_deg, dec_jnow_deg, cur_ax1, cur_ax2, time, ts, false)
            .ok_or(TrackingHaltReason::GimbalCone)?;

        if let Some(gem) = self.gem_state {
            let cw_up_angle = wrap_pi(a1_now - gem.ax1_cw_down).abs();
            if cw_up_angle > FRAC_PI_2 + MAX_MERIDIAN_PAST_RAD {
                return Err(TrackingHaltReason::MeridianLimit);
            }
        }

        let next_time = time + TimeDelta::seconds(1);
        let (a1_next, a2_next) = self
            .jnow_to_axes_inner(
                ra_jnow_deg,
                dec_jnow_deg,
                a1_now,
                a2_now,
                next_time,
                ts,
                false,
            )
            .ok_or(TrackingHaltReason::GimbalCone)?;
        Ok((
            (a1_now, a2_now),
            (wrap_pi(a1_next - a1_now), wrap_pi(a2_next - a2_now)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mount::angular_separation_deg;
    use nalgebra::Rotation3;

    fn simulate_mount_sky(
        true_rot: &Matrix3<f64>,
        true_th2_0: f64,
        true_s1: f64,
        true_s2: f64,
        ax1: f64,
        ax2: f64,
        time: DateTime<Utc>,
        ts: &Timescale,
    ) -> (f64, f64) {
        let u = true_rot * mount_vec(ax1, ax2, true_th2_0, true_s1, true_s2);
        earth_vec_to_jnow(u, time, ts)
    }

    #[test]
    fn test_tilted_alt_az_calibration_wrap_safety_and_tracking() {
        let ts = Timescale::default();
        let t0 = Utc::now();
        let true_rot = *Rotation3::from_euler_angles(0.65, -0.42, 1.35).matrix();
        // Test with Axis 2 near 350° (+6.11 rad, typical Celestron AUX [0, 2π) startup!)
        let true_th2_0 = 5.4;
        let true_s1 = -1.0;
        let true_s2 = 1.0;

        let mut model = GimbalSkyModel::default();
        assert!(!model.is_calibrated());

        let step = 4.0_f64.to_radians();
        let start_ax1 = 5.8;
        let start_ax2 = 6.1;
        for (ax1, ax2) in [
            (start_ax1, start_ax2),
            (start_ax1 + step, start_ax2),
            (start_ax1 + step, start_ax2 + step),
        ] {
            let (ra, dec) =
                simulate_mount_sky(&true_rot, true_th2_0, true_s1, true_s2, ax1, ax2, t0, &ts);
            model.add_observation(ax1, ax2, ra, dec, t0, &ts);
        }
        assert!(model.is_calibrated());

        // Verify target_ax2 unwraps around cur_ax2 (no -2π jump!).
        let (target_ra, target_dec) = simulate_mount_sky(
            &true_rot,
            true_th2_0,
            true_s1,
            true_s2,
            start_ax1 + 0.2,
            start_ax2 + 0.1,
            t0,
            &ts,
        );
        let (cmd_ax1, cmd_ax2) = model
            .jnow_to_axes(target_ra, target_dec, start_ax1, start_ax2, t0, &ts)
            .unwrap();
        assert!(
            (cmd_ax2 - (start_ax2 + 0.1)).abs() < 0.01,
            "Axis 2 wrapped instead of unwrapping around cur_ax2: {}",
            cmd_ax2
        );

        // Verify all-sky pointing accuracy and 10s dual-axis tracking.
        let ((_, _), (r1, r2)) = model
            .tracking_target_and_rates(target_ra, target_dec, cmd_ax1, cmd_ax2, t0, &ts)
            .unwrap();
        let t10 = t0 + TimeDelta::seconds(10);
        let (tracked_ra, tracked_dec) = simulate_mount_sky(
            &true_rot,
            true_th2_0,
            true_s1,
            true_s2,
            cmd_ax1 + r1 * 10.0,
            cmd_ax2 + r2 * 10.0,
            t10,
            &ts,
        );
        let track_err = angular_separation_deg(target_ra, target_dec, tracked_ra, tracked_dec);
        assert!(
            track_err < 0.005,
            "10s tracking drift too large: {}°",
            track_err
        );
    }

    #[test]
    fn test_equatorial_mount_nan_rejection_and_gimbal_cap() {
        let ts = Timescale::default();
        let t0 = Utc::now();
        let true_rot = *Rotation3::from_euler_angles(0.05, -0.04, 0.2).matrix();
        let mut model = GimbalSkyModel::default();

        let step = 5.0_f64.to_radians();
        for (ax1, ax2) in [(0.2, 0.6), (0.2 + step, 0.6), (0.2 + step, 0.6 + step)] {
            let (ra, dec) = simulate_mount_sky(&true_rot, 0.0, 1.0, 1.0, ax1, ax2, t0, &ts);
            model.add_observation(ax1, ax2, ra, dec, t0, &ts);
        }
        assert!(model.is_calibrated());

        assert!(
            model
                .jnow_to_axes(f64::NAN, 10.0, 0.2, 0.6, t0, &ts)
                .is_none()
        );
        let (pole_ra, pole_dec) = simulate_mount_sky(
            &true_rot,
            0.0,
            1.0,
            1.0,
            0.0,
            89.0_f64.to_radians(),
            t0,
            &ts,
        );
        assert!(
            model
                .jnow_to_axes(pole_ra, pole_dec, 0.2, 0.6, t0, &ts)
                .is_none()
        );
    }

    #[test]
    fn test_small_probe_detects_altaz_vs_eq_with_backlash_and_upgrades_on_wide_goto() {
        let ts = Timescale::default();
        let t0 = Utc::now();
        let (lat, lon) = (37.77, -122.42);
        let step = 5.0_f64.to_radians();
        let backlash = (15.0 / 60.0_f64).to_radians(); // 15 arcmin gear backlash!

        // 1) Alt-Az mount tilted by 3° with 15' backlash on the 5° probe.
        let tilt = *Rotation3::from_euler_angles(3.0_f64.to_radians(), 0.0, 0.0).matrix();
        let altaz_rot = tilt * basis_from_z(zenith_earth_vec(lat, lon));
        let mut aa_model = GimbalSkyModel::default();
        aa_model.set_site(lat, lon);

        let probe_pts = [
            (0.8, 0.65, 0.0, 0.0),
            (0.8 + step, 0.65, -backlash, 0.0),
            (0.8 + step, 0.65 + step, -backlash, -backlash),
        ];
        for (enc1, enc2, err1, err2) in probe_pts {
            let (ra, dec) = simulate_mount_sky(
                &altaz_rot,
                0.0,
                -1.0,
                1.0,
                enc1 + err1,
                enc2 + err2,
                t0,
                &ts,
            );
            aa_model.add_observation(enc1, enc2, ra, dec, t0, &ts);
        }
        assert!(aa_model.is_calibrated());
        assert_eq!(aa_model.detected_alt_az(), Some(true));

        // Coarse GoTo 45° away lands within ~4° (no ±phi2 flip!), then upgrades to full SO(3).
        let (wide_ra, wide_dec) = simulate_mount_sky(&altaz_rot, 0.0, -1.0, 1.0, 1.7, 0.9, t0, &ts);
        let (cmd1, cmd2) = aa_model
            .jnow_to_axes(wide_ra, wide_dec, 0.8 + step, 0.65 + step, t0, &ts)
            .unwrap();
        let (landed_ra, landed_dec) =
            simulate_mount_sky(&altaz_rot, 0.0, -1.0, 1.0, cmd1, cmd2, t0, &ts);
        assert!(
            angular_separation_deg(wide_ra, wide_dec, landed_ra, landed_dec) < 5.0,
            "Coarse GoTo from 5° probe with 15' backlash + 3° tilt missed badly"
        );

        // Clean wide-baseline post-slew solves upgrade to full unconstrained SO(3) solving out the 3° tilt.
        for (a1, a2) in [(1.7, 0.9), (2.4, 0.5)] {
            let (ra, dec) = simulate_mount_sky(&altaz_rot, 0.0, -1.0, 1.0, a1, a2, t0, &ts);
            aa_model.add_observation(a1, a2, ra, dec, t0, &ts);
        }
        assert_eq!(aa_model.detected_alt_az(), Some(true));
        let (test_ra, test_dec) =
            simulate_mount_sky(&altaz_rot, 0.0, -1.0, 1.0, 2.0, 0.75, t0, &ts);
        let (u1, u2) = aa_model
            .jnow_to_axes(test_ra, test_dec, 2.4, 0.5, t0, &ts)
            .unwrap();
        let (check_ra, check_dec) = simulate_mount_sky(&altaz_rot, 0.0, -1.0, 1.0, u1, u2, t0, &ts);
        assert!(
            angular_separation_deg(test_ra, test_dec, check_ra, check_dec) < 0.5,
            "Wide-baseline SO(3) upgrade failed to compensate 3° tripod tilt"
        );

        // 2) Equatorial mount (same site!) with 2° polar error + 15' backlash -> detected_alt_az == Some(false).
        let eq_rot =
            *Rotation3::from_euler_angles(2.0_f64.to_radians(), -1.5_f64.to_radians(), 0.7)
                .matrix();
        let mut eq_model = GimbalSkyModel::default();
        eq_model.set_site(lat, lon);
        for (enc1, enc2, err1, err2) in [
            (0.4, 0.5, 0.0, 0.0),
            (0.4 + step, 0.5, -backlash, 0.0),
            (0.4 + step, 0.5 + step, -backlash, -backlash),
        ] {
            let (ra, dec) =
                simulate_mount_sky(&eq_rot, 0.0, 1.0, 1.0, enc1 + err1, enc2 + err2, t0, &ts);
            eq_model.add_observation(enc1, enc2, ra, dec, t0, &ts);
        }
        assert!(eq_model.is_calibrated());
        assert_eq!(eq_model.detected_alt_az(), Some(false));

        // 3) Visiting >6 targets in a <15° cluster must NOT evict the wide-baseline anchor,
        //    and a corrupt outlier sync_local must be rejected without rotating the model.
        for i in 0..7 {
            let a1 = 2.0 + (i as f64) * 0.04;
            let a2 = 0.7 + ((i % 3) as f64) * 0.04;
            let (ra, dec) = simulate_mount_sky(&altaz_rot, 0.0, -1.0, 1.0, a1, a2, t0, &ts);
            assert!(aa_model.sync_local(a1, a2, ra, dec, t0, &ts));
        }
        // Corrupt outlier 15° off must return false and leave model pointing accurate!
        let (true_ra, true_dec) =
            simulate_mount_sky(&altaz_rot, 0.0, -1.0, 1.0, 2.0, 0.75, t0, &ts);
        assert!(!aa_model.sync_local(
            2.0,
            0.75,
            (true_ra + 15.0).rem_euclid(360.0),
            true_dec,
            t0,
            &ts
        ));
        let (after_ra, after_dec) = aa_model.axes_to_jnow(2.0, 0.75, t0, &ts).unwrap();
        assert!(
            angular_separation_deg(true_ra, true_dec, after_ra, after_dec) < 0.5,
            "Outlier sync or >6 clustered targets corrupted calibrated SO(3) model"
        );
    }

    #[test]
    fn test_eq_home_position_at_polaris_away_and_over_pole() {
        let ts = Timescale::default();
        let t0 = Utc::now();
        let (lat, lon) = (37.77, -122.42);
        let ax2_step = 9.0_f64.to_radians();
        let ax1_step = 20.0_f64.to_radians();
        let backlash = (15.0 / 60.0_f64).to_radians();

        // Test BOTH Axis-2 signs (s2 = +1 crosses OVER the North Celestial Pole, s2 = -1 steps AWAY)
        // across realistic polar alignment errors where sample 0 starts on the unflipped side of NCP:
        // (1) Polaris (89.2°, 0.8° from pole) with 0.4° polar error; (2) 87.0° (3° from pole) with 1.8° error.
        let cases = [
            (
                89.2_f64.to_radians(),
                0.3_f64.to_radians(),
                -0.25_f64.to_radians(),
            ),
            (
                87.0_f64.to_radians(),
                1.5_f64.to_radians(),
                -1.0_f64.to_radians(),
            ),
        ];

        for &(start_mag, rx, ry) in &cases {
            let eq_rot = *Rotation3::from_euler_angles(rx, ry, 0.4).matrix();
            for &s2 in &[1.0_f64, -1.0] {
                let start_ax2 = s2 * start_mag;
                let mut model = GimbalSkyModel::default();
                model.set_site(lat, lon);
                for (enc1, enc2, e1, e2) in [
                    (0.3, start_ax2, 0.0, 0.0),
                    (0.3, start_ax2 + ax2_step, 0.0, -backlash),
                    (0.3 + ax1_step, start_ax2 + ax2_step, -backlash, -backlash),
                ] {
                    let (ra, dec) =
                        simulate_mount_sky(&eq_rot, 0.0, 1.0, s2, enc1 + e1, enc2 + e2, t0, &ts);
                    model.add_observation(enc1, enc2, ra, dec, t0, &ts);
                }
                assert!(
                    model.is_calibrated(),
                    "Polaris probe failed to calibrate for s2={}",
                    s2
                );
                assert_eq!(model.detected_alt_az(), Some(false));

                // Coarse GoTo from near the pole must stay in the unflipped hemisphere,
                // then a single post-slew solve at Dec=+20° upgrades to full SO(3) < 0.5°!
                let (tgt_ra, tgt_dec) = simulate_mount_sky(
                    &eq_rot,
                    0.0,
                    1.0,
                    s2,
                    1.4,
                    s2 * 20.0_f64.to_radians(),
                    t0,
                    &ts,
                );
                let (cmd1, cmd2) = model
                    .jnow_to_axes(
                        tgt_ra,
                        tgt_dec,
                        0.3 + ax1_step,
                        start_ax2 + ax2_step,
                        t0,
                        &ts,
                    )
                    .unwrap();
                assert!(
                    wrap_pi(cmd2 - s2 * 20.0_f64.to_radians()).abs() < 16.0_f64.to_radians(),
                    "Polaris probe selected 180°-flipped twin (cmd2={:.1}°) for s2={}",
                    cmd2.to_degrees(),
                    s2
                );
                let (land_ra, land_dec) =
                    simulate_mount_sky(&eq_rot, 0.0, 1.0, s2, cmd1, cmd2, t0, &ts);
                let coarse_err = angular_separation_deg(tgt_ra, tgt_dec, land_ra, land_dec);
                assert!(
                    coarse_err < 16.0,
                    "Polaris-calibrated EQ coarse GoTo missed badly for s2={}: {:.2}°",
                    s2,
                    coarse_err
                );

                // Post-slew plate solve at the coarse landing spot upgrades to full SO(3) -> fine re-GoTo!
                assert!(model.sync_local(cmd1, cmd2, land_ra, land_dec, t0, &ts));
                let (fine1, fine2) = model
                    .jnow_to_axes(tgt_ra, tgt_dec, cmd1, cmd2, t0, &ts)
                    .unwrap();
                assert!(
                    wrap_pi(fine2 - s2 * 20.0_f64.to_radians()).abs() < 1.0_f64.to_radians(),
                    "Post-slew fine GoTo flipped hemisphere (fine2={:.1}°) for s2={}",
                    fine2.to_degrees(),
                    s2
                );
                let (fine_ra, fine_dec) =
                    simulate_mount_sky(&eq_rot, 0.0, 1.0, s2, fine1, fine2, t0, &ts);
                let fine_err = angular_separation_deg(tgt_ra, tgt_dec, fine_ra, fine_dec);
                assert!(
                    fine_err < 0.5,
                    "Post-slew fine re-GoTo failed to center for s2={}: {:.2}°",
                    s2,
                    fine_err
                );
            }
        }
    }

    #[test]
    fn test_gem_meridian_flip_goto_and_tracking_safety_limit() {
        let ts = Timescale::default();
        let t0 = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let (lat, lon) = (38.0, 0.0);
        let lst0_deg = (ts.from_datetime(t0).gast() * 15.0 + lon).rem_euclid(360.0);
        let eq_rot = Matrix3::identity();

        let mut model = GimbalSkyModel::default();
        model.set_site(lat, lon);
        model.set_gem_meridian_flip(true);

        // Calibrate in the East (HA = -3h -> RA = LST + 45°), counterweight down!
        let east_ax1 = 45.0_f64.to_radians();
        let east_ax2 = 30.0_f64.to_radians();
        let step = 5.0_f64.to_radians();
        for (a1, a2) in [
            (east_ax1, east_ax2),
            (east_ax1, east_ax2 + step),
            (east_ax1 + step, east_ax2 + step),
        ] {
            let (ra, dec) = simulate_mount_sky(&eq_rot, 0.0, 1.0, 1.0, a1, a2, t0, &ts);
            model.add_observation(a1, a2, ra, dec, t0, &ts);
        }
        assert!(model.is_calibrated());
        assert_eq!(model.detected_alt_az(), Some(false));

        // 1) GoTo West Southern-Dec target (HA = +3h, Dec = -20°): MUST meridian-flip UP over +90°
        //    to ax2 = +200° (NOT -160° underground!) and keep RA within CW-down ±90°.
        let west_ra = (lst0_deg - 45.0).rem_euclid(360.0);
        let (w1, w2) = model
            .jnow_to_axes(west_ra, -20.0, east_ax1, east_ax2, t0, &ts)
            .unwrap();
        assert!(
            (w2.to_degrees() - 200.0).abs() < 0.5,
            "NH GEM West Dec=-20° flip went underground instead of over North Pole: {:.1}°",
            w2.to_degrees()
        );
        let (check_ra, check_dec) = simulate_mount_sky(&eq_rot, 0.0, 1.0, 1.0, w1, w2, t0, &ts);
        assert!(angular_separation_deg(west_ra, -20.0, check_ra, check_dec) < 0.1);

        // 2) Fine-centering hysteresis within ±2° of meridian stays on current pier side.
        let near_meridian_ra = (lst0_deg - 1.0).rem_euclid(360.0); // HA = +1° (inside ±2° deadband)
        let (_, hm2) = model
            .jnow_to_axes(near_meridian_ra, 30.0, east_ax1, east_ax2, t0, &ts)
            .unwrap();
        assert!(
            (hm2.to_degrees() - 30.0).abs() < 0.5,
            "Meridian hysteresis failed; unexpectedly flipped at HA=+1° (ax2={:.1}°)",
            hm2.to_degrees()
        );

        // 3) Tracking across meridian stays on East pier side up to +10°, then halts at +11°!
        let track_ra = (lst0_deg + 1.0).rem_euclid(360.0); // starts at HA = -1° (East)
        let (tr1, tr2) = model
            .jnow_to_axes(track_ra, 30.0, east_ax1, east_ax2, t0, &ts)
            .unwrap();
        // 24 minutes later -> HA = +5° (past meridian, still within +10° limit -> OK!)
        let t_5deg = t0 + TimeDelta::minutes(24);
        assert!(
            model
                .tracking_target_and_rates(track_ra, 30.0, tr1, tr2, t_5deg, &ts)
                .is_ok()
        );
        // 48 minutes later -> HA = +11° (> +10° meridian safety limit -> MeridianLimit halt!)
        let t_11deg = t0 + TimeDelta::minutes(48);
        assert_eq!(
            model
                .tracking_target_and_rates(track_ra, 30.0, tr1, tr2, t_11deg, &ts)
                .unwrap_err(),
            TrackingHaltReason::MeridianLimit
        );

        // 4) Re-GoTo at HA = +11° flips cleanly to West pier side and resumes tracking!
        let (flip1, flip2) = model
            .jnow_to_axes(track_ra, 30.0, tr1, tr2, t_11deg, &ts)
            .unwrap();
        assert!((flip2.to_degrees() - 150.0).abs() < 0.5);
        assert!(
            model
                .tracking_target_and_rates(track_ra, 30.0, flip1, flip2, t_11deg, &ts)
                .is_ok()
        );

        // 5) Southern Hemisphere GEM (lat = -35°) flips Northern-Dec target UP over South Pole (-90° -> -200°)!
        let mut sh_model = GimbalSkyModel::default();
        sh_model.set_site(-35.0, lon);
        sh_model.set_gem_meridian_flip(true);
        for (a1, a2) in [
            (east_ax1, -30.0_f64.to_radians()),
            (east_ax1, -25.0_f64.to_radians()),
            (east_ax1 + step, -25.0_f64.to_radians()),
        ] {
            let (ra, dec) = simulate_mount_sky(&eq_rot, 0.0, 1.0, 1.0, a1, a2, t0, &ts);
            sh_model.add_observation(a1, a2, ra, dec, t0, &ts);
        }
        let (_, sh_w2) = sh_model
            .jnow_to_axes(west_ra, 20.0, east_ax1, -30.0_f64.to_radians(), t0, &ts)
            .unwrap();
        assert!(
            (sh_w2.to_degrees() - (-200.0)).abs() < 0.5,
            "SH GEM West Dec=+20° flip went underground instead of over South Pole: {:.1}°",
            sh_w2.to_degrees()
        );

        // 6) Full-turn (+360°) revolution offset must stay in the current turn (no -360° unwind
        //    or underground Dec flip when cur_ax2 is near +340° = -20° in [0, 2π)).
        let (rev1, rev2) = model
            .jnow_to_axes(
                west_ra,
                -20.0,
                east_ax1 + TAU,
                340.0_f64.to_radians(),
                t0,
                &ts,
            )
            .unwrap();
        assert!(
            (rev1 - (w1 + TAU)).abs() < 0.02,
            "Full-turn RA unwind detected: {:.1}° vs expected {:.1}°",
            rev1.to_degrees(),
            (w1 + TAU).to_degrees()
        );
        assert!(
            (rev2.to_degrees() - 560.0).abs() < 0.5,
            "Dec=340° flip went underground instead of 340°->450°->560° over North Pole: {:.1}°",
            rev2.to_degrees()
        );

        // 7) Rx(π) twin (`(rot * e_z).z < 0`, e.g. fitted before set_site) still flips over
        //    the elevated North Pole, and align_hemisphere preserves all 4 twins.
        let mut rx_model = GimbalSkyModel::default();
        rx_model.set_gem_meridian_flip(true);
        let rx_rot = rx_pi();
        for (a1, a2) in [
            (east_ax1, east_ax2),
            (east_ax1, east_ax2 + step),
            (east_ax1 + step, east_ax2 + step),
        ] {
            let (ra, dec) = simulate_mount_sky(&rx_rot, 0.0, -1.0, -1.0, a1, a2, t0, &ts);
            rx_model.add_observation(a1, a2, ra, dec, t0, &ts);
        }
        rx_model.set_site(lat, lon);
        let (rx_w1, rx_w2) = rx_model
            .jnow_to_axes(west_ra, -20.0, east_ax1, east_ax2, t0, &ts)
            .unwrap();
        let (rx_ra, rx_dec) = simulate_mount_sky(&rx_rot, 0.0, -1.0, -1.0, rx_w1, rx_w2, t0, &ts);
        assert!(angular_separation_deg(west_ra, -20.0, rx_ra, rx_dec) < 0.1);
        assert!(
            (rx_w2.to_degrees() - 200.0).abs() < 0.5,
            "Rx(π) twin flipped underground instead of over North Pole: {:.1}°",
            rx_w2.to_degrees()
        );

        // 8) Lower-culmination hysteresis (HA ≈ +179°, circumpolar Dec=+70°) stays on current
        //    pier side instead of oscillating 180° between coarse and fine passes.
        let lower_merid_ra = (lst0_deg - 179.0).rem_euclid(360.0);
        let (_, lc2) = model
            .jnow_to_axes(lower_merid_ra, 70.0, east_ax1, east_ax2, t0, &ts)
            .unwrap();
        assert!(
            lc2.to_degrees() < 90.0,
            "Lower-culmination hysteresis should keep current East pier side, got ax2={:.1}°",
            lc2.to_degrees()
        );
    }
}
