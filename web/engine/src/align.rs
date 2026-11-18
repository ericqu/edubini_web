//! Alignment = global warp W plus an optional smooth local displacement field
//! (superres.Alignment / superres._local_field).

use crate::ecc::{ecc, prep_image, Motion, Rect};
use crate::img::*;

/// Displacement field sampled on a coarse tile grid.
#[derive(Clone)]
pub struct Field {
    pub nx: usize,
    pub ny: usize,
    pub d: Vec<[f32; 2]>,
    pub origin: f64,
    pub step: f64,
}

impl Field {
    /// Bilinear interpolation, clamped at the grid edges (BORDER_REPLICATE).
    #[inline(always)]
    pub fn at(&self, rx: f64, ry: f64) -> (f64, f64) {
        let gx = ((rx - self.origin) / self.step).clamp(0.0, (self.nx - 1) as f64);
        let gy = ((ry - self.origin) / self.step).clamp(0.0, (self.ny - 1) as f64);
        let x0 = (gx as usize).min(self.nx.saturating_sub(2));
        let y0 = (gy as usize).min(self.ny.saturating_sub(2));
        let x1 = (x0 + 1).min(self.nx - 1);
        let y1 = (y0 + 1).min(self.ny - 1);
        let (fx, fy) = (gx - x0 as f64, gy - y0 as f64);
        let g = |x: usize, y: usize, k: usize| self.d[y * self.nx + x][k] as f64;
        let mut out = [0f64; 2];
        for k in 0..2 {
            let top = g(x0, y0, k) + fx * (g(x1, y0, k) - g(x0, y0, k));
            let bot = g(x0, y1, k) + fx * (g(x1, y1, k) - g(x0, y1, k));
            out[k] = top + fy * (bot - top);
        }
        (out[0], out[1])
    }
}

/// Maps reference coords r to frame coords: W(r + d(r)).
#[derive(Clone)]
pub struct Alignment {
    pub w: M3,
    pub w_inv: M3,
    pub field: Option<Field>,
}

impl Alignment {
    pub fn rigid(w: M3) -> Self {
        Alignment { w, w_inv: inv(&w), field: None }
    }

    #[inline(always)]
    pub fn ref_to_frame(&self, rx: f64, ry: f64) -> (f64, f64) {
        match &self.field {
            None => apply(&self.w, rx, ry),
            Some(f) => {
                let (dx, dy) = f.at(rx, ry);
                apply(&self.w, rx + dx, ry + dy)
            }
        }
    }

    #[inline(always)]
    pub fn frame_to_ref(&self, fx: f64, fy: f64) -> (f64, f64) {
        let (yx, yy) = apply(&self.w_inv, fx, fy);
        match &self.field {
            None => (yx, yy),
            Some(f) => {
                let (mut rx, mut ry) = (yx, yy);
                for _ in 0..3 {
                    let (dx, dy) = f.at(rx, ry);
                    rx = yx - dx;
                    ry = yy - dy;
                }
                (rx, ry)
            }
        }
    }

    /// Frame resampled onto the reference grid (bilinear, zero outside), and the
    /// mask of reference pixels whose nearest frame pixel exists.
    pub fn warp_to_ref(&self, g: &Img) -> (Img, Vec<u8>) {
        let (w, h) = (g.w, g.h);
        let mut out = Img::new(w, h);
        let mut valid = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (u, v) = self.ref_to_frame(x as f64, y as f64);
                let (val, _) = sample_zero(&g.d, w, h, u, v);
                out.d[y * w + x] = val;
                let (ur, vr) = (u.round(), v.round());
                valid[y * w + x] = (ur >= 0.0 && vr >= 0.0 && ur <= (w - 1) as f64 && vr <= (h - 1) as f64) as u8;
            }
        }
        (out, valid)
    }

    /// Add the frame resampled onto the reference grid (and its coverage) to sums.
    pub fn accumulate(&self, g: &Img, acc: &mut [f64], cnt: &mut [f64]) {
        let (w, h) = (g.w, g.h);
        for y in 0..h {
            for x in 0..w {
                let (u, v) = self.ref_to_frame(x as f64, y as f64);
                let (val, s) = sample_zero(&g.d, w, h, u, v);
                acc[y * w + x] += val as f64;
                cnt[y * w + x] += s as f64;
            }
        }
    }

    /// Frame coordinates of the output pixel centres of rows r0..r1, as
    /// (r1 - r0, wo, 2) f32 (the drizzle map).
    pub fn output_map(&self, wo: usize, r0: usize, r1: usize, scale: usize, map: &mut Vec<f32>) {
        map.resize(wo * (r1 - r0) * 2, 0.0);
        let s = scale as f64;
        let off = (s - 1.0) / 2.0;
        for v in r0..r1 {
            for u in 0..wo {
                let (fx, fy) = self.ref_to_frame((u as f64 - off) / s, (v as f64 - off) / s);
                let k = ((v - r0) * wo + u) * 2;
                map[k] = fx as f32;
                map[k + 1] = fy as f32;
            }
        }
    }

    /// Flat f64 encoding (to pass alignments between WebAssembly instances):
    /// W (9), nx, ny, origin, step, then nx*ny*2 field values (nx = ny = 0: rigid).
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = self.w.to_vec();
        match &self.field {
            None => v.extend([0.0, 0.0, 0.0, 0.0]),
            Some(f) => {
                v.extend([f.nx as f64, f.ny as f64, f.origin, f.step]);
                v.extend(f.d.iter().flat_map(|p| [p[0] as f64, p[1] as f64]));
            }
        }
        v
    }

    pub fn from_slice(v: &[f64]) -> Option<Self> {
        if v.len() < 13 {
            return None;
        }
        let w: M3 = v[..9].try_into().ok()?;
        let (nx, ny) = (v[9] as usize, v[10] as usize);
        let mut al = Alignment::rigid(w);
        if nx > 0 && ny > 0 {
            if v.len() < 13 + 2 * nx * ny {
                return None;
            }
            let d = (0..nx * ny).map(|k| [v[13 + 2 * k] as f32, v[14 + 2 * k] as f32]).collect();
            al.field = Some(Field { nx, ny, d, origin: v[11], step: v[12] });
        }
        Some(al)
    }
}

/// Smooth residual displacement of `aligned` (a frame already warped onto the
/// reference grid) relative to the reference, from per-tile translation ECC.
/// `ref_blur` is the ECC-smoothed reference. Returns (field, reliable tiles).
pub fn local_field(ref_raw: &Img, ref_blur: &Img, aligned: &Img, valid: &[u8], tile: usize) -> Option<(Field, usize)> {
    let (w, h) = (ref_raw.w, ref_raw.h);
    if w < tile || h < tile || tile < 8 {
        return None;
    }
    let step = tile / 2;
    let ys: Vec<usize> = (0..=h - tile).step_by(step).collect();
    let xs: Vec<usize> = (0..=w - tile).step_by(step).collect();
    let (ny, nx) = (ys.len(), xs.len());
    let ap = prep_image(aligned);
    let mut d = vec![[0f32; 2]; nx * ny];
    let mut conf = vec![0f32; nx * ny];
    let mut crop = Vec::with_capacity(tile * tile);
    for (ky, &y) in ys.iter().enumerate() {
        for (kx, &x) in xs.iter().enumerate() {
            let mut nv = 0usize;
            crop.clear();
            for yy in y..y + tile {
                nv += valid[yy * w + x..yy * w + x + tile].iter().map(|&v| v as usize).sum::<usize>();
                crop.extend_from_slice(&ref_raw.d[yy * w + x..yy * w + x + tile]);
            }
            if (nv as f64) < 0.98 * (tile * tile) as f64 {
                continue;
            }
            if mean_std(&crop).1 < 0.006 {
                continue; // featureless: no information
            }
            let r = Rect { x, y, w: tile, h: tile };
            let mut p = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            let Some(cc) = ecc(ref_blur, r, &ap, r, Motion::Translation, &mut p, 50, 1e-5) else {
                continue;
            };
            if cc < 0.5 || p[2].abs().max(p[5].abs()) > tile as f64 / 4.0 {
                continue;
            }
            d[ky * nx + kx] = [p[2] as f32, p[5] as f32];
            conf[ky * nx + kx] = cc as f32;
        }
    }
    // Drop tiles that disagree with their 3x3 neighbourhood by > 1 px.
    let ok: Vec<f32> = conf.iter().map(|&c| (c > 0.0) as u8 as f32).collect();
    let mut bad = vec![false; nx * ny];
    for ky in 0..ny {
        for kx in 0..nx {
            let (mut den, mut n0, mut n1) = (0f32, 0f32, 0f32);
            for yy in ky.saturating_sub(1)..(ky + 2).min(ny) {
                for xx in kx.saturating_sub(1)..(kx + 2).min(nx) {
                    let i = yy * nx + xx;
                    den += ok[i];
                    n0 += d[i][0] * ok[i];
                    n1 += d[i][1] * ok[i];
                }
            }
            let i = ky * nx + kx;
            let (m0, m1) = if den > 0.0 { (n0 / den, n1 / den) } else { (0.0, 0.0) };
            if ((d[i][0] - m0).powi(2) + (d[i][1] - m1).powi(2)).sqrt() > 1.0 {
                bad[i] = true;
            }
        }
    }
    for i in 0..nx * ny {
        if bad[i] {
            conf[i] = 0.0;
        }
    }
    let n_good = conf.iter().filter(|&&c| c > 0.0).count();
    // Normalised-convolution smoothing (fills featureless tiles, keeps the field
    // smooth at the tile scale).
    let k = gaussian_kernel(1.0);
    let wgt: Vec<f32> = conf.iter().map(|c| c * c).collect();
    let wsm = sep_filter(&wgt, nx, ny, 1, &k, Border::Replicate);
    let mut field = vec![[0f32; 2]; nx * ny];
    for c in 0..2 {
        let num: Vec<f32> = (0..nx * ny).map(|i| d[i][c] * wgt[i]).collect();
        let nsm = sep_filter(&num, nx, ny, 1, &k, Border::Replicate);
        for i in 0..nx * ny {
            field[i][c] = if wsm[i] > 1e-6 { nsm[i] / wsm[i] } else { 0.0 };
        }
    }
    Some((Field { nx, ny, d: field, origin: (tile as f64 - 1.0) / 2.0, step: step as f64 }, n_good))
}
