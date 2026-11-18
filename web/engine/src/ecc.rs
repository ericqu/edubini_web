//! Enhanced Correlation Coefficient registration (Evangelidis & Psarakis 2008),
//! the same forward-additive algorithm as cv2.findTransformECC, for the
//! translation and euclidean motion models, plus the coarse-to-fine registrar
//! of superres.py.
//!
//! All the per-iteration quantities (means, norms, Hessian, projections) are
//! accumulated in one fused pass over the template, without temporary images.

use crate::fft::phase_correlate;
use crate::img::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Motion {
    Translation,
    Euclidean,
}

/// An image prepared as ECC input: smoothed values and their gradients.
pub struct Prep {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f32>,
    pub gx: Vec<f32>,
    pub gy: Vec<f32>,
}

pub fn prep_image(img: &Img) -> Prep {
    let b = blur5(img);
    let (gx, gy) = grad(&b);
    Prep { w: img.w, h: img.h, v: b.d, gx, gy }
}

#[derive(Clone, Copy)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn full(w: usize, h: usize) -> Self {
        Rect { x: 0, y: 0, w, h }
    }
}

fn solve_sym(h: &[[f64; 3]; 3], b: &[f64; 3], n: usize) -> Option<[f64; 3]> {
    if n == 2 {
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        if !(det.abs() > 1e-300) {
            return None;
        }
        return Some([(h[1][1] * b[0] - h[0][1] * b[1]) / det, (h[0][0] * b[1] - h[1][0] * b[0]) / det, 0.0]);
    }
    let m: M3 = [h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2]];
    let i = inv(&m);
    if !i.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some([
        i[0] * b[0] + i[1] * b[1] + i[2] * b[2],
        i[3] * b[0] + i[4] * b[1] + i[5] * b[2],
        i[6] * b[0] + i[7] * b[1] + i[8] * b[2],
    ])
}

/// Sums over the valid template pixels for one ECC iteration (values may be
/// shifted by constants: every formula in `ecc` is shift-invariant).
#[derive(Default)]
struct Stats {
    n: f64,
    si: f64,
    sii: f64,
    st: f64,
    stt: f64,
    sti: f64,
    sj: [f64; 3],
    sjj: [[f64; 3]; 3],
    sji: [f64; 3],
    sjt: [f64; 3],
}

/// General path (any 2x3 warp; used for the euclidean model).
fn stats_general(t: &Img, tr: Rect, im: &Prep, ir: Rect, np: usize, warp: &[f64; 6]) -> Stats {
    let [a, b, tx, c, d, ty] = *warp;
    let (umax, vmax) = ((ir.w - 1) as f64, (ir.h - 1) as f64);
    let mut s = Stats::default();
    for y in 0..tr.h {
        let trow = &t.d[(tr.y + y) * t.w + tr.x..][..tr.w];
        let yf = y as f64;
        let (ub, vb) = (b * yf + tx, d * yf + ty);
        for (x, &tv) in trow.iter().enumerate() {
            let xf = x as f64;
            let u = a * xf + ub;
            let v = c * xf + vb;
            if !(u >= 0.0 && v >= 0.0 && u <= umax && v <= vmax) {
                continue;
            }
            let x0 = (u as usize).min(ir.w - 2);
            let y0 = (v as usize).min(ir.h - 2);
            let fx = (u - x0 as f64) as f32;
            let fy = (v - y0 as f64) as f32;
            let i0 = (ir.y + y0) * im.w + ir.x + x0;
            let i1 = i0 + im.w;
            let bl = |p: &[f32]| {
                let top = p[i0] + fx * (p[i0 + 1] - p[i0]);
                let bot = p[i1] + fx * (p[i1 + 1] - p[i1]);
                (top + fy * (bot - top)) as f64
            };
            let iv = bl(&im.v);
            let gx = bl(&im.gx);
            let gy = bl(&im.gy);
            let tv = tv as f64;
            let j = if np == 3 {
                let hx = -(xf * c) - yf * a;
                let hy = xf * a - yf * c;
                [gx * hx + gy * hy, gx, gy]
            } else {
                [gx, gy, 0.0]
            };
            s.n += 1.0;
            s.si += iv;
            s.sii += iv * iv;
            s.st += tv;
            s.stt += tv * tv;
            s.sti += tv * iv;
            for k in 0..np {
                s.sj[k] += j[k];
                s.sji[k] += j[k] * iv;
                s.sjt[k] += j[k] * tv;
                for l in k..np {
                    s.sjj[k][l] += j[k] * j[l];
                }
            }
        }
    }
    s
}

/// Fast path for a pure translation: the bilinear weights are the same for
/// every pixel, so rows are interpolated once and accumulated in f32 lanes
/// (vectorised). Values are centred by `ct` / `ci` to keep f32 sums accurate.
#[allow(clippy::too_many_arguments)]
fn stats_translation(t: &Img, tr: Rect, im: &Prep, ir: Rect, tx: f64, ty: f64, ct: f32, ci: f32,
                     rows: &mut [Vec<f32>; 3]) -> Stats {
    let mut s = Stats::default();
    let (oxf, oyf) = (tx.floor(), ty.floor());
    let (fx, fy) = ((tx - oxf) as f32, (ty - oyf) as f32);
    let (ox, oy) = (oxf as isize, oyf as isize);
    // template pixels whose 2x2 neighbourhood lies inside the image window
    let xlo = (-ox).max(0);
    let xhi = (tr.w as isize - 1).min(ir.w as isize - 2 - ox);
    let ylo = (-oy).max(0);
    let yhi = (tr.h as isize - 1).min(ir.h as isize - 2 - oy);
    if xlo > xhi || ylo > yhi {
        return s;
    }
    let nx = (xhi - xlo + 1) as usize;
    let (w00, w01, w10, w11) = ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy);
    const L: usize = 8;
    for y in ylo..=yhi {
        let trow = &t.d[(tr.y + y as usize) * t.w + tr.x + xlo as usize..][..nx];
        let b0 = (ir.y as isize + y + oy) as usize * im.w + (ir.x as isize + xlo + ox) as usize;
        let b1 = b0 + im.w;
        for (out, p, c) in [(0, &im.v, ci), (1, &im.gx, 0.0), (2, &im.gy, 0.0)] {
            let r = &mut rows[out];
            r.clear();
            let (r0, r1) = (&p[b0..b0 + nx + 1], &p[b1..b1 + nx + 1]);
            r.extend(
                r0[..nx].iter().zip(&r0[1..]).zip(r1[..nx].iter().zip(&r1[1..]))
                    .map(|((&a, &b), (&c2, &d))| w00 * a + w01 * b + w10 * c2 + w11 * d - c),
            );
        }
        let mut acc = [[0f32; L]; 14];
        let (ir_, gx_, gy_) = (&rows[0], &rows[1], &rows[2]);
        let mut add = |l: usize, tv: f32, iv: f32, gx: f32, gy: f32| {
            let tv = tv - ct;
            acc[0][l] += iv;
            acc[1][l] += iv * iv;
            acc[2][l] += tv;
            acc[3][l] += tv * tv;
            acc[4][l] += tv * iv;
            acc[5][l] += gx;
            acc[6][l] += gy;
            acc[7][l] += gx * gx;
            acc[8][l] += gx * gy;
            acc[9][l] += gy * gy;
            acc[10][l] += gx * iv;
            acc[11][l] += gy * iv;
            acc[12][l] += gx * tv;
            acc[13][l] += gy * tv;
        };
        let full = nx / L * L;
        for (((tc, ic), xc), yc) in trow[..full].chunks_exact(L).zip(ir_[..full].chunks_exact(L))
            .zip(gx_[..full].chunks_exact(L)).zip(gy_[..full].chunks_exact(L))
        {
            for l in 0..L {
                add(l, tc[l], ic[l], xc[l], yc[l]);
            }
        }
        for k in full..nx {
            add(0, trow[k], ir_[k], gx_[k], gy_[k]);
        }
        let q: Vec<f64> = acc.iter().map(|a| a.iter().map(|&v| v as f64).sum()).collect();
        s.n += nx as f64;
        s.si += q[0];
        s.sii += q[1];
        s.st += q[2];
        s.stt += q[3];
        s.sti += q[4];
        s.sj[0] += q[5];
        s.sj[1] += q[6];
        s.sjj[0][0] += q[7];
        s.sjj[0][1] += q[8];
        s.sjj[1][1] += q[9];
        s.sji[0] += q[10];
        s.sji[1] += q[11];
        s.sjt[0] += q[12];
        s.sjt[1] += q[13];
    }
    s
}

fn window_mean(d: &[f32], stride: usize, r: Rect) -> f32 {
    let mut s = 0f64;
    for y in r.y..r.y + r.h {
        s += d[y * stride + r.x..y * stride + r.x + r.w].iter().map(|&v| v as f64).sum::<f64>();
    }
    (s / (r.w * r.h) as f64) as f32
}

/// ECC of template `t` (already smoothed; window `tr`) against image `im`
/// (window `ir`). `warp` (2x3, row-major) maps template-window coordinates to
/// image-window coordinates and is refined in place. Returns the final
/// correlation coefficient, or None where OpenCV would raise.
#[allow(clippy::too_many_arguments)]
pub fn ecc(t: &Img, tr: Rect, im: &Prep, ir: Rect, motion: Motion, warp: &mut [f64; 6], iters: usize, eps: f64) -> Option<f64> {
    if ir.w < 2 || ir.h < 2 {
        return None;
    }
    let np = if motion == Motion::Euclidean { 3 } else { 2 };
    let (ct, ci) = if motion == Motion::Translation {
        (window_mean(&t.d, t.w, tr), window_mean(&im.v, im.w, ir))
    } else {
        (0.0, 0.0)
    };
    let mut rows = [Vec::with_capacity(tr.w), Vec::with_capacity(tr.w), Vec::with_capacity(tr.w)];
    let mut rho = -1.0f64;
    let mut last_rho = -eps;
    let mut it = 0;
    while it < iters && (rho - last_rho).abs() >= eps {
        it += 1;
        let mut s = match motion {
            Motion::Translation => stats_translation(t, tr, im, ir, warp[2], warp[5], ct, ci, &mut rows),
            Motion::Euclidean => stats_general(t, tr, im, ir, np, warp),
        };
        if s.n < 16.0 {
            return None;
        }
        for k in 0..np {
            for l in 0..k {
                s.sjj[k][l] = s.sjj[l][k];
            }
        }
        let (mi, mt) = (s.si / s.n, s.st / s.n);
        let img_n2 = s.sii - s.n * mi * mi;
        let tmp_n2 = s.stt - s.n * mt * mt;
        let corr = s.sti - s.n * mi * mt;
        if !(img_n2 > 0.0 && tmp_n2 > 0.0) {
            return None;
        }
        last_rho = rho;
        rho = corr / (img_n2 * tmp_n2).sqrt();
        if rho.is_nan() {
            return None;
        }
        let mut ip = [0f64; 3];
        let mut tp = [0f64; 3];
        for k in 0..np {
            ip[k] = s.sji[k] - mi * s.sj[k];
            tp[k] = s.sjt[k] - mt * s.sj[k];
        }
        let iph = solve_sym(&s.sjj, &ip, np)?;
        let dot = |p: &[f64; 3], q: &[f64; 3]| (0..np).map(|k| p[k] * q[k]).sum::<f64>();
        let lam_n = img_n2 - dot(&ip, &iph);
        let lam_d = corr - dot(&tp, &iph);
        if lam_d <= 0.0 {
            return None; // "the correlation is going to be minimized"
        }
        let lam = lam_n / lam_d;
        let mut ep = [0f64; 3];
        for k in 0..np {
            ep[k] = lam * tp[k] - ip[k];
        }
        let dp = solve_sym(&s.sjj, &ep, np)?;
        match motion {
            Motion::Translation => {
                warp[2] += dp[0];
                warp[5] += dp[1];
            }
            Motion::Euclidean => {
                let th = warp[3].clamp(-1.0, 1.0).asin() + dp[0];
                warp[2] += dp[1];
                warp[5] += dp[2];
                let (sn, co) = th.sin_cos();
                warp[0] = co;
                warp[4] = co;
                warp[3] = sn;
                warp[1] = -sn;
            }
        }
        if !warp.iter().all(|v| v.is_finite()) {
            return None;
        }
    }
    Some(rho)
}

/// Coarse-to-fine ECC registration against a fixed reference (superres.Registrar).
pub struct Registrar {
    pub motion: Motion,
    pub levels: usize,
    pub raw: Vec<Img>,  // reference pyramid
    pub blur: Vec<Img>, // its ECC-smoothed version
}

fn pyramid(img: &Img, levels: usize) -> Vec<Img> {
    let mut p = vec![img.clone()];
    for _ in 1..levels {
        let next = pyr_down(p.last().unwrap());
        p.push(next);
    }
    p
}

impl Registrar {
    pub fn new(reference: Img, motion: Motion, levels: usize) -> Self {
        let raw = pyramid(&reference, levels);
        let blur = raw.iter().map(blur5).collect();
        Registrar { motion, levels, raw, blur }
    }

    fn ecc_pyr(&self, pyr: &[Prep], init: &M3, levels: usize) -> Option<(M3, f64)> {
        let mut w = *init;
        let mut cc = 0.0;
        for lvl in (0..levels).rev() {
            let f = 0.5f64.powi(lvl as i32);
            let wl = rescale(&w, f);
            let mut p = [wl[0], wl[1], wl[2], wl[3], wl[4], wl[5]];
            let iters = if lvl == 0 { 50 } else { 100 };
            let t = &self.blur[lvl];
            let im = &pyr[lvl];
            cc = ecc(t, Rect::full(t.w, t.h), im, Rect::full(im.w, im.h), self.motion, &mut p, iters, 1e-6)?;
            w = rescale(&[p[0], p[1], p[2], p[3], p[4], p[5], 0.0, 0.0, 1.0], 1.0 / f);
        }
        Some((w, cc))
    }

    /// Warp mapping reference coords -> img coords, and the ECC correlation.
    pub fn register(&self, img: &Img, init: &M3, levels: Option<usize>) -> Option<(M3, f64)> {
        let levels = levels.map_or(self.levels, |l| l.min(self.levels)).max(1);
        let raw = pyramid(img, levels);
        let prep: Vec<Prep> = raw.iter().map(prep_image).collect();
        if let Some(r) = self.ecc_pyr(&prep, init, levels) {
            return Some(r);
        }
        // Fallback: initialise translation from phase correlation and retry.
        let (dx, dy, _) = phase_correlate(&self.raw[0], img, None);
        let guess = translation(dx, dy);
        let raw = pyramid(img, self.levels);
        let prep: Vec<Prep> = raw.iter().map(prep_image).collect();
        self.ecc_pyr(&prep, &guess, self.levels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(w: usize, h: usize, m: &M3) -> Img {
        // smooth random-ish texture evaluated through the inverse of m
        let mi = inv(m);
        let mut img = Img::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (u, v) = apply(&mi, x as f64, y as f64);
                img.d[y * w + x] = (0.5
                    + 0.2 * (u * 0.21).sin() * (v * 0.13).cos()
                    + 0.15 * (u * 0.07 + v * 0.05).sin()
                    + 0.1 * ((u - v) * 0.33).cos()) as f32;
            }
        }
        img
    }

    #[test]
    fn ecc_recovers_euclidean_motion() {
        let (w, h) = (320, 240);
        let th: f64 = 0.01;
        let truth: M3 = [th.cos(), -th.sin(), 2.3, th.sin(), th.cos(), -1.7, 0.0, 0.0, 1.0];
        let a = scene(w, h, &EYE);
        let b = scene(w, h, &truth); // b(truth x) = a(x): warp ref -> frame = truth
        let reg = Registrar::new(a, Motion::Euclidean, 3);
        let (wp, cc) = reg.register(&b, &EYE, None).expect("ecc");
        assert!(cc > 0.99, "cc {cc}");
        for k in [0, 1, 2, 3, 4, 5] {
            assert!((wp[k] - truth[k]).abs() < 0.02, "{k}: {} vs {}", wp[k], truth[k]);
        }
    }
}
