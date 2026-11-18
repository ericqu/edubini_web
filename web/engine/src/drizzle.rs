//! Drizzle kernels: the same algorithm as rust/lib.rs (the Python extension),
//! without PyO3 and rayon (WebAssembly runs it single-threaded).
//! See rust/lib.rs for the full description.
//!
//! The per-output-pixel computation is factored out so that the outlier pass
//! can *subtract* the contribution of a few rejected input pixels from a few
//! output pixels ("undrizzle"), which by linearity equals re-drizzling every
//! frame with those pixels' weights set to zero, at a tiny fraction of the cost.

use crate::img::{apply, M3};

const MAX_CHANNELS: usize = 4;

#[inline(always)]
fn overlap(c: f64, half: f64, p: f64) -> f64 {
    ((c + half).min(p + 0.5) - (c - half).max(p - 0.5)).max(0.0)
}

/// Input data shared by all output pixels of one frame.
pub struct Src<'a> {
    pub frame: &'a [f32],
    pub weight: &'a [f32],
    pub wi: usize,
    pub hi: usize,
    pub c: usize,
    pub half: f64,
}

type Px = (f64, [f64; MAX_CHANNELS]);

/// Homography kernel for output pixel (xo, yo).
#[inline(always)]
fn gather_h(s: &Src, xo: f64, yo: f64, h: &M3, h_inv: &M3, rx: f64, ry: f64) -> Px {
    let mut sw = 0.0f64;
    let mut sv = [0.0f64; MAX_CHANNELS];
    let (xc, yc) = apply(h_inv, xo, yo);
    if !xc.is_finite() || !yc.is_finite() {
        return (sw, sv);
    }
    let x0 = ((xc - rx).ceil() as isize).max(0);
    let x1 = ((xc + rx).floor() as isize).min(s.wi as isize - 1);
    let y0 = ((yc - ry).ceil() as isize).max(0);
    let y1 = ((yc + ry).floor() as isize).min(s.hi as isize - 1);
    if x0 > x1 || y0 > y1 {
        return (sw, sv);
    }
    for y in y0..=y1 {
        let row = y as usize * s.wi;
        for x in x0..=x1 {
            let idx = row + x as usize;
            let wv = s.weight[idx];
            if wv <= 0.0 {
                continue;
            }
            let (u, v) = apply(h, x as f64, y as f64);
            let a = overlap(u, s.half, xo) * overlap(v, s.half, yo);
            if a <= 0.0 {
                continue;
            }
            let w = a * wv as f64;
            sw += w;
            let px = &s.frame[idx * s.c..idx * s.c + s.c];
            for k in 0..s.c {
                sv[k] += w * px[k] as f64;
            }
        }
    }
    (sw, sv)
}

/// Dense-map kernel for output pixel (i, j); `at(i, j)` gives the input-frame
/// coordinates of output pixel centres (only (i, j) and its 4 neighbours are used).
#[inline(always)]
fn gather_map(s: &Src, i: usize, j: usize, wo: usize, ho: usize, at: &impl Fn(usize, usize) -> (f64, f64)) -> Px {
    let mut sw = 0.0f64;
    let mut sv = [0.0f64; MAX_CHANNELS];
    let (xo, yo) = (j as f64, i as f64);
    let reach = s.half + 0.5;
    let (xc, yc) = at(i, j);
    if !xc.is_finite() || !yc.is_finite() {
        return (sw, sv);
    }
    let (iu, id) = (i.saturating_sub(1), (i + 1).min(ho - 1));
    let (jl, jr) = (j.saturating_sub(1), (j + 1).min(wo - 1));
    let (pl, pr) = (at(i, jl), at(i, jr));
    let (pu, pd) = (at(iu, j), at(id, j));
    let dj = (jr - jl) as f64;
    let di = (id - iu) as f64;
    let (a, b) = ((pr.0 - pl.0) / dj, (pd.0 - pu.0) / di);
    let (cc, d) = ((pr.1 - pl.1) / dj, (pd.1 - pu.1) / di);
    let det = a * d - b * cc;
    if !det.is_finite() || det.abs() < 1e-12 {
        return (sw, sv);
    }
    let (fa, fb, fc, fd) = (d / det, -b / det, -cc / det, a / det);
    let rx = (a.abs() + b.abs()) * reach + 1e-9;
    let ry = (cc.abs() + d.abs()) * reach + 1e-9;
    let x0 = ((xc - rx).ceil() as isize).max(0);
    let x1 = ((xc + rx).floor() as isize).min(s.wi as isize - 1);
    let y0 = ((yc - ry).ceil() as isize).max(0);
    let y1 = ((yc + ry).floor() as isize).min(s.hi as isize - 1);
    if x0 > x1 || y0 > y1 {
        return (sw, sv);
    }
    for y in y0..=y1 {
        let row = y as usize * s.wi;
        let ey = y as f64 - yc;
        for x in x0..=x1 {
            let idx = row + x as usize;
            let wv = s.weight[idx];
            if wv <= 0.0 {
                continue;
            }
            let ex = x as f64 - xc;
            let u = xo + fa * ex + fb * ey;
            let v = yo + fc * ex + fd * ey;
            let ov = overlap(u, s.half, xo) * overlap(v, s.half, yo);
            if ov <= 0.0 {
                continue;
            }
            let w = ov * wv as f64;
            sw += w;
            let px = &s.frame[idx * s.c..idx * s.c + s.c];
            for k in 0..s.c {
                sv[k] += w * px[k] as f64;
            }
        }
    }
    (sw, sv)
}

#[inline(always)]
fn store(acc: &mut [f32], wacc: &mut [f32], o: usize, c: usize, px: Px, sign: f32) {
    let (sw, sv) = px;
    if sw > 0.0 {
        wacc[o] += sign * sw as f32;
        for k in 0..c {
            acc[o * c + k] += sign * sv[k] as f32;
        }
    }
}

fn reach_h(h_inv: &M3, half: f64) -> (f64, f64) {
    let reach = half + 0.5;
    let projective = h_inv[6] != 0.0 || h_inv[7] != 0.0;
    let s = h_inv[8];
    let rx = (h_inv[0].abs() + h_inv[1].abs()) / s.abs() * reach + if projective { 1.0 } else { 1e-9 };
    let ry = (h_inv[3].abs() + h_inv[4].abs()) / s.abs() * reach + if projective { 1.0 } else { 1e-9 };
    (rx, ry)
}

/// Add one frame (hi, wi, c) into the accumulators of output rows `rows`
/// (acc: (rows, wo, c), wacc: (rows, wo); the full output has `ho` rows).
/// `h` maps input pixel coords to output pixel coords, `h_inv` is its inverse.
#[allow(clippy::too_many_arguments)]
pub fn drizzle_add(
    acc: &mut [f32], wacc: &mut [f32], wo: usize, rows: std::ops::Range<usize>, c: usize,
    frame: &[f32], weight: &[f32], wi: usize, hi: usize,
    h: &M3, h_inv: &M3, drop_size: f64,
) {
    let nr = rows.len();
    assert!(c > 0 && c <= MAX_CHANNELS && acc.len() == wo * nr * c && wacc.len() == wo * nr);
    assert!(frame.len() == wi * hi * c && weight.len() == wi * hi && drop_size > 0.0);
    let s = Src { frame, weight, wi, hi, c, half: 0.5 * drop_size };
    let (rx, ry) = reach_h(h_inv, s.half);
    for i in rows.clone() {
        for j in 0..wo {
            let px = gather_h(&s, j as f64, i as f64, h, h_inv, rx, ry);
            store(acc, wacc, (i - rows.start) * wo + j, c, px, 1.0);
        }
    }
}

/// Same through a dense map: input-frame coordinates of the output pixel
/// centres of rows map_r0.. (it must cover rows.start-1 ..= rows.end, clipped
/// to the image, for the finite-difference Jacobian). Non-rigid alignment.
#[allow(clippy::too_many_arguments)]
pub fn drizzle_add_map(
    acc: &mut [f32], wacc: &mut [f32], wo: usize, ho: usize, rows: std::ops::Range<usize>, c: usize,
    frame: &[f32], weight: &[f32], wi: usize, hi: usize,
    map: &[f32], map_r0: usize, drop_size: f64,
) {
    let nr = rows.len();
    assert!(c > 0 && c <= MAX_CHANNELS && acc.len() == wo * nr * c && wacc.len() == wo * nr);
    assert!(frame.len() == wi * hi * c && weight.len() == wi * hi);
    assert!(map_r0 <= rows.start.saturating_sub(1) && map.len() >= wo * ((rows.end + 1).min(ho) - map_r0) * 2);
    assert!(ho >= 2 && wo >= 2 && drop_size > 0.0);
    let s = Src { frame, weight, wi, hi, c, half: 0.5 * drop_size };
    let at = |i: usize, j: usize| -> (f64, f64) {
        let k = ((i - map_r0) * wo + j) * 2;
        (map[k] as f64, map[k + 1] as f64)
    };
    for i in rows.clone() {
        for j in 0..wo {
            let px = gather_map(&s, i, j, wo, ho, &at);
            store(acc, wacc, (i - rows.start) * wo + j, c, px, 1.0);
        }
    }
}

/// Subtract, at the output pixels `targets`, what `drizzle_add` added from the
/// input pixels with nonzero `weight` (pass the rejected pixels' weights).
#[allow(clippy::too_many_arguments)]
pub fn undrizzle_h(
    acc: &mut [f32], wacc: &mut [f32], wo: usize, c: usize,
    frame: &[f32], weight: &[f32], wi: usize, hi: usize,
    h: &M3, h_inv: &M3, drop_size: f64, targets: &[usize],
) {
    let s = Src { frame, weight, wi, hi, c, half: 0.5 * drop_size };
    let (rx, ry) = reach_h(h_inv, s.half);
    for &o in targets {
        let px = gather_h(&s, (o % wo) as f64, (o / wo) as f64, h, h_inv, rx, ry);
        store(acc, wacc, o, c, px, -1.0);
    }
}

/// Map version of `undrizzle_h`; `at(i, j)` must reproduce the map that was
/// used by `drizzle_add_map`.
#[allow(clippy::too_many_arguments)]
pub fn undrizzle_map(
    acc: &mut [f32], wacc: &mut [f32], wo: usize, ho: usize, c: usize,
    frame: &[f32], weight: &[f32], wi: usize, hi: usize,
    at: &impl Fn(usize, usize) -> (f64, f64), drop_size: f64, targets: &[usize],
) {
    let s = Src { frame, weight, wi, hi, c, half: 0.5 * drop_size };
    for &o in targets {
        let px = gather_map(&s, o / wo, o % wo, wo, ho, at);
        store(acc, wacc, o, c, px, -1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::img::{inv, EYE};

    #[test]
    fn identity_reproduces_input() {
        let (w, h, c) = (40, 30, 3);
        let f: Vec<f32> = (0..w * h * c).map(|i| ((i * 37) % 251) as f32).collect();
        let mut acc = vec![0f32; f.len()];
        let mut wacc = vec![0f32; w * h];
        drizzle_add(&mut acc, &mut wacc, w, 0..h, c, &f, &vec![1.0; w * h], w, h, &EYE, &EYE, 1.0);
        for i in 0..w * h {
            assert!((wacc[i] - 1.0).abs() < 1e-6);
            for k in 0..c {
                assert!((acc[i * c + k] - f[i * c + k]).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn half_pixel_shift_lands_on_one_output_pixel() {
        let mut f = vec![0f32; 100];
        f[55] = 100.0;
        let s = 2.0;
        let hm: M3 = [s, 0.0, 0.5 + s * 0.25, 0.0, s, 0.5 + s * 0.25, 0.0, 0.0, 1.0];
        let mut acc = vec![0f32; 400];
        let mut wacc = vec![0f32; 400];
        drizzle_add(&mut acc, &mut wacc, 20, 0..20, 1, &f, &vec![1.0; 100], 10, 10, &hm, &inv(&hm), 1.0);
        assert!((acc[11 * 20 + 11] - 100.0).abs() < 1e-3);
        assert!((acc.iter().sum::<f32>() - 100.0).abs() < 1e-3);
    }

    /// Undrizzling some pixels must equal drizzling with their weight set to 0.
    #[test]
    fn undrizzle_equals_zero_weight() {
        let (wi, hi, s) = (30usize, 20usize, 2usize);
        let (wo, ho) = (wi * s, hi * s);
        let f: Vec<f32> = (0..wi * hi).map(|i| ((i * 53) % 241) as f32).collect();
        let th: f64 = 0.02;
        // frame -> output
        let hm: M3 = [s as f64 * th.cos(), -(s as f64) * th.sin(), 1.3, s as f64 * th.sin(), s as f64 * th.cos(), 0.7, 0.0, 0.0, 1.0];
        let hi_m = inv(&hm);
        let mut bad = vec![0f32; wi * hi];
        for &k in &[45usize, 46, 200, 333, 501] {
            bad[k] = 1.0;
        }
        let good: Vec<f32> = bad.iter().map(|b| 1.0 - b).collect();
        let at = |i: usize, j: usize| apply(&hi_m, j as f64, i as f64);
        let mut map = vec![0f32; wo * ho * 2];
        for i in 0..ho {
            for j in 0..wo {
                let (x, y) = at(i, j);
                map[(i * wo + j) * 2] = x as f32;
                map[(i * wo + j) * 2 + 1] = y as f32;
            }
        }
        let all: Vec<usize> = (0..wo * ho).collect();
        for use_map in [false, true] {
            let (mut a1, mut w1) = (vec![0f32; wo * ho], vec![0f32; wo * ho]);
            let (mut a2, mut w2) = (vec![0f32; wo * ho], vec![0f32; wo * ho]);
            if use_map {
                drizzle_add_map(&mut a1, &mut w1, wo, ho, 0..ho, 1, &f, &vec![1.0; wi * hi], wi, hi, &map, 0, 1.4);
                let atf = |i: usize, j: usize| {
                    let k = (i * wo + j) * 2;
                    (map[k] as f64, map[k + 1] as f64)
                };
                undrizzle_map(&mut a1, &mut w1, wo, ho, 1, &f, &bad, wi, hi, &atf, 1.4, &all);
                drizzle_add_map(&mut a2, &mut w2, wo, ho, 0..ho, 1, &f, &good, wi, hi, &map, 0, 1.4);
            } else {
                drizzle_add(&mut a1, &mut w1, wo, 0..ho, 1, &f, &vec![1.0; wi * hi], wi, hi, &hm, &hi_m, 1.4);
                undrizzle_h(&mut a1, &mut w1, wo, 1, &f, &bad, wi, hi, &hm, &hi_m, 1.4, &all);
                drizzle_add(&mut a2, &mut w2, wo, 0..ho, 1, &f, &good, wi, hi, &hm, &hi_m, 1.4);
            }
            for o in 0..wo * ho {
                assert!((a1[o] - a2[o]).abs() < 1e-2 && (w1[o] - w2[o]).abs() < 1e-5, "map={use_map} o={o}");
            }
        }
    }
}
