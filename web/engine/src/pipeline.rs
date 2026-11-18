//! The super-resolution pipeline of superres.enhance + deconv.wiener_deconvolve.
//!
//! Frames are added one at a time while recording (`add_frame`): duplicates are
//! dropped and the cheap scan (sharpness, chained phase-correlation trajectory)
//! is done immediately, so only registration and drizzle remain afterwards.

use crate::align::{local_field, Alignment};
use crate::deconv::wiener_deconvolve;
use crate::drizzle::{drizzle_add, drizzle_add_map, undrizzle_h, undrizzle_map};
use crate::ecc::{Motion, Registrar};
use crate::fft::PhaseRef;
use crate::img::*;

#[derive(Clone)]
pub struct Params {
    pub scale: usize,
    pub pixfrac: f64,
    pub motion: Motion,
    pub local: bool,
    /// Also refine pass 1 (which only builds the low-noise reference) locally.
    pub local_pass1: bool,
    pub tile: usize,
    pub reject: bool,
    pub reject_sigma: f64,
    pub keep: f64,
    pub min_overlap: f64,
    pub border: usize,
    pub deconvolve: bool,
    pub keep_raw: bool,
    /// Distinct-frame row defining the output view (None = automatic).
    pub reference: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            scale: 2,
            pixfrac: 0.7,
            motion: Motion::Euclidean,
            local: true,
            local_pass1: true,
            tile: 96,
            reject: true,
            reject_sigma: 4.0,
            keep: 1.0,
            min_overlap: 0.5,
            border: 2,
            deconvolve: true,
            keep_raw: false,
            reference: None,
        }
    }
}

/// Progress stages reported to the caller.
pub const ST_REGISTER: u32 = 1;
pub const ST_DRIZZLE: u32 = 2;
pub const ST_REJECT: u32 = 3;
pub const ST_DECONV: u32 = 4;

pub struct Output {
    pub w: usize,
    pub h: usize,
    pub image: Vec<f32>,       // (h, w, 3) RGB, 0..255
    pub raw: Option<Vec<f32>>, // before deconvolution (keep_raw)
    pub diag: String,          // JSON
}

pub struct Engine {
    pub w: usize,
    pub h: usize,
    frames: Vec<Vec<u8>>, // distinct frames, RGB u8
    sharp: Vec<f64>,
    pos: Vec<[f64; 2]>,
    n_total: usize,
    n_dup: usize,
    prev: Option<Img>,
    prev_phase: Option<PhaseRef>,
    small_win: Vec<f32>,
    p: [f64; 2],
    pub dedupe: f64,
}

fn sharpness(g: &Img) -> f64 {
    let g1 = gaussian_blur(g, 1.0);
    let g2 = gaussian_blur(g, 2.0);
    let g4 = gaussian_blur(g, 4.0);
    let var = |a: &[f32], b: &[f32]| {
        let n = a.len() as f64;
        let m = a.iter().zip(b).map(|(x, y)| (x - y) as f64).sum::<f64>() / n;
        a.iter().zip(b).map(|(x, y)| ((x - y) as f64 - m).powi(2)).sum::<f64>() / n
    };
    var(&g1.d, &g2.d) / (var(&g2.d, &g4.d) + 1e-12)
}

/// Input-pixel weights: 0 on the outer `border` pixels, then a cosine ramp.
fn border_mask(w: usize, h: usize, border: usize, feather: usize) -> Vec<f32> {
    let ramp = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let d = i.min(n - 1 - i) as f64 - border as f64 + 1.0;
                let t = (d / feather.max(1) as f64).clamp(0.0, 1.0);
                (0.5 - 0.5 * (std::f64::consts::PI * t).cos()).clamp(0.0, 1.0) as f32
            })
            .collect()
    };
    let (rx, ry) = (ramp(w), ramp(h));
    let mut m = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            m[y * w + x] = ry[y] * rx[x];
        }
    }
    m
}

/// (acc_a + acc_b) / (w_a + w_b), with poorly covered pixels blended into a
/// normalised-convolution fill.
fn finalize(acc_a: &[f32], acc_b: &[f32], wa: &[f32], wb: &[f32], wo: usize, ho: usize, scale: usize) -> Vec<f32> {
    let acc: Vec<f32> = acc_a.iter().zip(acc_b).map(|(a, b)| a + b).collect();
    let wacc: Vec<f32> = wa.iter().zip(wb).map(|(a, b)| a + b).collect();
    let k = gaussian_kernel((0.5 * scale as f64).max(1.0));
    let gw = sep_filter(&wacc, wo, ho, 1, &k, Border::Reflect101);
    let mut pos: Vec<f32> = wacc.iter().copied().filter(|&v| v > 0.0).collect();
    let refw = if pos.is_empty() { 1.0 } else { median(&mut pos) };
    drop(pos);
    let ga = sep_filter(&acc, wo, ho, 3, &k, Border::Reflect101);
    let mut out = acc;
    for i in 0..wo * ho {
        let wv = wacc[i];
        let t = (wv / (0.25 * refw)).clamp(0.0, 1.0);
        for c in 0..3 {
            let direct = if wv > 0.0 { out[i * 3 + c] / wv } else { 0.0 };
            let fill = if gw[i] > 0.0 { ga[i * 3 + c] / gw[i] } else { 0.0 };
            out[i * 3 + c] = t * direct + (1.0 - t) * fill;
        }
    }
    out
}

fn to_f32(rgb: &[u8], out: &mut Vec<f32>) {
    out.clear();
    out.extend(rgb.iter().map(|&v| v as f32));
}

/// Frame -> output homography of a rigid alignment.
fn frame_to_output(al: &Alignment, scale: usize) -> M3 {
    let s = scale as f64;
    let sm: M3 = [s, 0.0, (s - 1.0) / 2.0, 0.0, s, (s - 1.0) / 2.0, 0.0, 0.0, 1.0];
    let mut hm = mul(&sm, &al.w_inv);
    let n = hm[8];
    hm.iter_mut().for_each(|v| *v /= n);
    hm
}

/// Drizzle accumulators for output rows r0..r1 (the whole image, or one
/// worker's share of it), split into the two independent half-stacks.
pub struct Band {
    pub r0: usize,
    pub r1: usize,
    w: usize,
    h: usize,
    wo: usize,
    ho: usize,
    scale: usize,
    drop: f64,
    pub acc_a: Vec<f32>,
    pub acc_b: Vec<f32>,
    pub wa: Vec<f32>,
    pub wb: Vec<f32>,
    bmask: Vec<f32>,
    map: Vec<f32>,
    fbuf: Vec<f32>,
}

impl Band {
    pub fn new(w: usize, h: usize, prm: &Params, r0: usize, r1: usize) -> Self {
        let (wo, ho) = (w * prm.scale, h * prm.scale);
        let (r0, r1) = (r0.min(ho), r1.min(ho).max(r0.min(ho)));
        let n = wo * (r1 - r0);
        Band {
            r0, r1, w, h, wo, ho, scale: prm.scale, drop: prm.pixfrac * prm.scale as f64,
            acc_a: vec![0.0; n * 3], acc_b: vec![0.0; n * 3], wa: vec![0.0; n], wb: vec![0.0; n],
            bmask: border_mask(w, h, prm.border, 24), map: vec![], fbuf: vec![],
        }
    }

    /// Drizzle one frame (RGB u8) into this band.
    pub fn add(&mut self, rgb: &[u8], al: &Alignment, half_b: bool) {
        if self.r0 == self.r1 {
            return;
        }
        to_f32(rgb, &mut self.fbuf);
        let (acc, wacc) = if half_b { (&mut self.acc_b, &mut self.wb) } else { (&mut self.acc_a, &mut self.wa) };
        let (w, h, wo, ho) = (self.w, self.h, self.wo, self.ho);
        match &al.field {
            None => {
                let hm = frame_to_output(al, self.scale);
                drizzle_add(acc, wacc, wo, self.r0..self.r1, 3, &self.fbuf, &self.bmask, w, h, &hm, &inv(&hm), self.drop);
            }
            Some(_) => {
                let (m0, m1) = (self.r0.saturating_sub(1), (self.r1 + 1).min(ho));
                al.output_map(wo, m0, m1, self.scale, &mut self.map);
                drizzle_add_map(acc, wacc, wo, ho, self.r0..self.r1, 3, &self.fbuf, &self.bmask, w, h, &self.map, m0, self.drop);
            }
        }
    }
}

/// Remove the contribution of the input pixels with nonzero `wbad` (listed in
/// `bad`) from the accumulators: the same as having drizzled the frame with
/// their weight set to zero.
#[allow(clippy::too_many_arguments)]
fn undrizzle_frame(acc: &mut [f32], wacc: &mut [f32], f: &[f32], wbad: &[f32], bad: &[usize], al: &Alignment,
                   w: usize, h: usize, scale: usize, drop: f64, mark: &mut [u8], targets: &mut Vec<usize>) {
    let (wo, ho) = (w * scale, h * scale);
    let s = scale as f64;
    let off = (s - 1.0) / 2.0;
    let hm = frame_to_output(al, scale);
    // Output pixels whose gather box can contain a rejected input pixel.
    let r = 0.5 * drop + 0.5 + 1.5;
    targets.clear();
    for &p in bad {
        let (x, y) = ((p % w) as f64, (p / w) as f64);
        let (uo, vo) = match &al.field {
            None => apply(&hm, x, y),
            Some(_) => {
                let (rx, ry) = al.frame_to_ref(x, y);
                (rx * s + off, ry * s + off)
            }
        };
        let i0 = (vo - r).ceil().max(0.0) as usize;
        let i1 = ((vo + r).floor().max(-1.0) as isize).min(ho as isize - 1);
        let j0 = (uo - r).ceil().max(0.0) as usize;
        let j1 = ((uo + r).floor().max(-1.0) as isize).min(wo as isize - 1);
        for i in i0 as isize..=i1 {
            for j in j0 as isize..=j1 {
                let o = i as usize * wo + j as usize;
                if mark[o] == 0 {
                    mark[o] = 1;
                    targets.push(o);
                }
            }
        }
    }
    match &al.field {
        None => undrizzle_h(acc, wacc, wo, 3, f, wbad, w, h, &hm, &inv(&hm), drop, targets),
        Some(_) => {
            // reproduce Alignment::output_map exactly (including the f32 rounding)
            let at = |i: usize, j: usize| {
                let (x, y) = al.ref_to_frame((j as f64 - off) / s, (i as f64 - off) / s);
                (x as f32 as f64, y as f32 as f64)
            };
            undrizzle_map(acc, wacc, wo, ho, 3, f, wbad, w, h, &at, drop, targets);
        }
    }
    for &o in targets.iter() {
        mark[o] = 0;
    }
}

/// Output-grid coordinates of every input pixel centre, computed exactly on a
/// coarse grid and interpolated (only used to look up the model for blotting).
fn blot_coords(al: &Alignment, w: usize, h: usize, scale: usize, step: usize, bx: &mut Vec<f32>, by: &mut Vec<f32>) {
    let s = scale as f64;
    let off = (s - 1.0) / 2.0;
    let nx = (w - 1).div_ceil(step) + 1;
    let ny = (h - 1).div_ceil(step) + 1;
    let mut gx = vec![0f64; nx * ny];
    let mut gy = vec![0f64; nx * ny];
    for ky in 0..ny {
        for kx in 0..nx {
            let (rx, ry) = al.frame_to_ref((kx * step) as f64, (ky * step) as f64);
            gx[ky * nx + kx] = rx * s + off;
            gy[ky * nx + kx] = ry * s + off;
        }
    }
    bx.resize(w * h, 0.0);
    by.resize(w * h, 0.0);
    let inv_step = 1.0 / step as f64;
    for y in 0..h {
        let ky = y / step;
        let fy = (y - ky * step) as f64 * inv_step;
        let ky1 = (ky + 1).min(ny - 1);
        for x in 0..w {
            let kx = x / step;
            let fx = (x - kx * step) as f64 * inv_step;
            let kx1 = (kx + 1).min(nx - 1);
            let lerp = |g: &[f64]| {
                let top = g[ky * nx + kx] + fx * (g[ky * nx + kx1] - g[ky * nx + kx]);
                let bot = g[ky1 * nx + kx] + fx * (g[ky1 * nx + kx1] - g[ky1 * nx + kx]);
                top + fy * (bot - top)
            };
            bx[y * w + x] = lerp(&gx) as f32;
            by[y * w + x] = lerp(&gy) as f32;
        }
    }
}

/// Median of non-negative values from a 1/64-level histogram (exact enough for
/// a noise estimate, O(n)).
fn hist_median(vals: impl Iterator<Item = f32>) -> f32 {
    const SCALE: f32 = 64.0;
    const NB: usize = 256 * SCALE as usize;
    let mut hist = vec![0u32; NB];
    let mut n = 0u64;
    for v in vals {
        hist[((v * SCALE) as usize).min(NB - 1)] += 1;
        n += 1;
    }
    let half = n.div_ceil(2);
    let mut acc = 0u64;
    for (b, &c) in hist.iter().enumerate() {
        acc += c as u64;
        if acc >= half {
            return (b as f32 + 0.5) / SCALE;
        }
    }
    0.0
}

/// Global ECC, then optionally the local displacement field.
fn align(reg: &Registrar, g: &Img, init: &M3, local: bool, tile: usize, levels: Option<usize>) -> Option<(Alignment, f64, usize)> {
    let (wm, cc) = reg.register(g, init, levels)?;
    let al = Alignment::rigid(wm);
    if !local {
        return Some((al, cc, 0));
    }
    let (aligned, valid) = al.warp_to_ref(g);
    match local_field(&reg.raw[0], &reg.blur[0], &aligned, &valid, tile) {
        Some((field, n)) => Some((Alignment { field: Some(field), ..al }, cc, n)),
        None => Some((al, cc, 0)),
    }
}

/// Fraction of 4x4 sub-pixel phase cells hit by the frames.
fn phase_coverage(warps: &[M3], w: usize, h: usize) -> f64 {
    let bins = 4usize;
    let (cx, cy) = ((w / 2) as f64, (h / 2) as f64);
    let ph: Vec<[f64; 2]> = warps
        .iter()
        .map(|wp| {
            let (x, y) = apply(&inv(wp), cx, cy);
            [x - x.floor(), y - y.floor()]
        })
        .collect();
    if ph.is_empty() {
        return 0.0;
    }
    let tau = 2.0 * std::f64::consts::PI;
    let mut mean = [0f64; 2];
    for k in 0..2 {
        let s: f64 = ph.iter().map(|p| (tau * p[k]).sin()).sum();
        let c: f64 = ph.iter().map(|p| (tau * p[k]).cos()).sum();
        mean[k] = s.atan2(c) / tau;
    }
    let mut hit = vec![false; bins * bins];
    for p in &ph {
        let idx = |k: usize| -> usize {
            let c = (p[k] - mean[k] + 0.5 / bins as f64).rem_euclid(1.0);
            ((c * bins as f64) as usize).min(bins - 1)
        };
        hit[idx(1) * bins + idx(0)] = true;
    }
    hit.iter().filter(|&&b| b).count() as f64 / (bins * bins) as f64
}

impl Engine {
    pub fn new(w: usize, h: usize) -> Self {
        Engine {
            w, h, frames: vec![], sharp: vec![], pos: vec![], n_total: 0, n_dup: 0,
            prev: None, prev_phase: None, small_win: vec![], p: [0.0; 2], dedupe: 0.3,
        }
    }

    pub fn frames(&self) -> usize {
        self.frames.len()
    }

    pub fn frame_rgb(&self, row: usize) -> Option<&[u8]> {
        self.frames.get(row).map(|f| f.as_slice())
    }

    /// Add one frame (interleaved u8, `stride` = 3 for RGB or 4 for RGBA).
    /// Returns false if it duplicates the previous distinct frame.
    pub fn add_frame(&mut self, px: &[u8], stride: usize) -> bool {
        let (w, h) = (self.w, self.h);
        assert!(px.len() >= w * h * stride);
        self.n_total += 1;
        let g = gray_u8(px, w, h, stride);
        if let Some(prev) = &self.prev {
            let mad = g.d.iter().zip(&prev.d).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / (w * h) as f64;
            if mad * 255.0 < self.dedupe {
                self.n_dup += 1;
                return false;
            }
        }
        let small = down2(&g);
        match &self.prev_phase {
            None => self.small_win = hanning(small.w, small.h),
            Some(pr) => {
                let (dx, dy, _) = pr.correlate(&small);
                self.p[0] += 2.0 * dx;
                self.p[1] += 2.0 * dy;
            }
        }
        self.prev_phase = Some(PhaseRef::new(&small, Some(&self.small_win)));
        self.sharp.push(sharpness(&g));
        self.pos.push(self.p);
        let mut rgb = Vec::with_capacity(w * h * 3);
        for p in px.chunks_exact(stride).take(w * h) {
            rgb.extend_from_slice(&p[..3]);
        }
        self.frames.push(rgb);
        self.prev = Some(g);
        true
    }

    fn overlap(&self, row: usize) -> Vec<f64> {
        let (w, h) = (self.w as f64, self.h as f64);
        self.pos
            .iter()
            .map(|p| {
                let dx = (p[0] - self.pos[row][0]).abs();
                let dy = (p[1] - self.pos[row][1]).abs();
                (1.0 - dx / w).clamp(0.0, 1.0) * (1.0 - dy / h).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Reference row (among the sharpest quarter, the one shared by the most
    /// sharpness-weighted frames) and the rows to stack.
    fn select(&self, prm: &Params) -> (usize, Vec<usize>) {
        let n = self.sharp.len();
        let med = median_f64(&self.sharp).max(1e-12);
        let rank: Vec<f64> = self.sharp.iter().map(|s| s / med).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| self.sharp[b].partial_cmp(&self.sharp[a]).unwrap_or(std::cmp::Ordering::Equal));
        let cand = &order[..(n / 4).max(1)];
        let mut best = (cand[0], f64::MIN);
        for &r in cand {
            let ov = self.overlap(r);
            let s: f64 = ov.iter().zip(&rank).map(|(&o, &k)| if o >= prm.min_overlap { o * k } else { 0.0 }).sum();
            if s > best.1 {
                best = (r, s);
            }
        }
        let ref_row = prm.reference.map_or(best.0, |r| r.min(n - 1));
        let ov = self.overlap(ref_row);
        let mut rows: Vec<usize> = (0..n).filter(|&i| i != ref_row && ov[i] >= prm.min_overlap).collect();
        if prm.keep < 1.0 && !rows.is_empty() {
            let n_keep = ((prm.keep * rows.len() as f64).round() as usize).max(1);
            rows.sort_by(|&a, &b| self.sharp[b].partial_cmp(&self.sharp[a]).unwrap_or(std::cmp::Ordering::Equal));
            rows.truncate(n_keep);
        }
        rows.push(ref_row);
        rows.sort_unstable();
        (ref_row, rows)
    }

    pub fn gray(&self, row: usize) -> Img {
        gray_u8(&self.frames[row], self.w, self.h, 3)
    }

    /// The whole pipeline in this thread (the web app splits the same steps
    /// over several workers, see web/worker.js).
    pub fn process(&self, prm: &Params, prog: &mut dyn FnMut(u32, usize, usize)) -> Result<Output, String> {
        let mut job = Job::new(self, prm.clone())?;
        // Pass 1: register to the reference frame, build the stacked mean
        let p1 = job.pass1_ref(self);
        let chosen = job.chosen.clone();
        for (k, &i) in chosen.iter().enumerate() {
            prog(ST_REGISTER, k, chosen.len());
            let res = if i == job.ref_row {
                Some((Alignment::rigid(EYE), 1.0))
            } else {
                align_pass1(&p1, &self.gray(i), job.traj(i), &job.prm).map(|(a, cc, _)| (a, cc))
            };
            job.p1_add(self, i, res);
        }
        drop(p1);
        // Pass 2: re-register against the stacked reference and drizzle
        let reg = Registrar::new(job.p1_finish(self), job.prm.motion, job.levels);
        let mut band = Band::new(self.w, self.h, &job.prm, 0, job.ho);
        let rows = job.rows.clone();
        for (k, &i) in rows.iter().enumerate() {
            prog(ST_DRIZZLE, k, rows.len());
            let res = align_pass2(&reg, &self.gray(i), &job.init_w(i), &job.prm).map(|(a, _, t)| (a, t));
            job.p2_set(i, res);
            band.add(&self.frames[i], job.alignment(i), job.half_b(i));
        }
        drop(reg);
        job.set_acc(band);
        job.p2_finish();
        // Pass 3 + noise + deconvolution
        job.p3(self, prog);
        Ok(job.finish(self, prog))
    }
}

/// Pass-1 registration context: the reference frame and its phase-correlation spectrum.
pub struct Pass1Ref {
    reg: Registrar,
    phase: PhaseRef,
}

impl Pass1Ref {
    pub fn new(ref_gray: Img, motion: Motion, levels: usize) -> Self {
        let win = hanning(ref_gray.w, ref_gray.h);
        let phase = PhaseRef::new(&ref_gray, Some(&win));
        Pass1Ref { reg: Registrar::new(ref_gray, motion, levels), phase }
    }
}

/// Pass-1 alignment of one frame: direct phase correlation against the
/// reference (the chained scan trajectory `traj` is the sanity check /
/// fallback), then ECC. Returns (alignment, cc, reliable tiles).
pub fn align_pass1(r: &Pass1Ref, g: &Img, traj: [f64; 2], prm: &Params) -> Option<(Alignment, f64, usize)> {
    let (dx, dy, resp) = r.phase.correlate(g);
    let direct_ok = resp > 0.03 && (dx - traj[0]).hypot(dy - traj[1]) < 0.15 * g.w.min(g.h) as f64;
    let init = if direct_ok { translation(dx, dy) } else { translation(traj[0], traj[1]) };
    align(&r.reg, g, &init, prm.local && prm.local_pass1, prm.tile, None)
}

/// Pass-2 alignment against the stacked reference, starting from the pass-1 warp.
pub fn align_pass2(reg: &Registrar, g: &Img, init: &M3, prm: &Params) -> Option<(Alignment, f64, usize)> {
    align(reg, g, init, prm.local, prm.tile, Some(2))
}

/// State of one super-resolution run between its steps.
pub struct Job {
    pub prm: Params,
    w: usize,
    h: usize,
    pub ref_row: usize,
    pub chosen: Vec<usize>,
    pub levels: usize,
    traj: Vec<[f64; 2]>,
    aligns: Vec<Option<Alignment>>,
    ccs: Vec<(usize, f64)>,
    mean_acc: Vec<f64>,
    mean_cnt: Vec<f64>,
    n_fail: usize,
    n_low: usize,
    med_cc: f64,
    pub rows: Vec<usize>,
    in_b: Vec<bool>,
    tiles: Vec<f64>,
    pub wo: usize,
    pub ho: usize,
    pub acc_a: Vec<f32>,
    pub acc_b: Vec<f32>,
    pub wa: Vec<f32>,
    pub wb: Vec<f32>,
    model: Vec<f32>,
    rejected: f64,
}

impl Job {
    pub fn new(e: &Engine, prm: Params) -> Result<Job, String> {
        let n = e.frames.len();
        if n < 2 {
            return Err("need at least 2 distinct frames (was the camera moving?)".into());
        }
        let (w, h) = (e.w, e.h);
        let (ref_row, chosen) = e.select(&prm);
        let levels = (((w.min(h) as f64) / 90.0).log2() as i64 + 1).max(1) as usize;
        let traj = (0..n).map(|i| [e.pos[i][0] - e.pos[ref_row][0], e.pos[i][1] - e.pos[ref_row][1]]).collect();
        Ok(Job {
            w, h, ref_row, chosen, levels, traj,
            aligns: vec![None; n], ccs: vec![], mean_acc: vec![0.0; w * h], mean_cnt: vec![0.0; w * h],
            n_fail: 0, n_low: 0, med_cc: 0.0, rows: vec![], in_b: vec![false; n], tiles: vec![],
            wo: w * prm.scale, ho: h * prm.scale, acc_a: vec![], acc_b: vec![], wa: vec![], wb: vec![],
            model: vec![], rejected: 0.0, prm,
        })
    }

    pub fn traj(&self, row: usize) -> [f64; 2] {
        self.traj[row]
    }

    pub fn ref_gray(&self, e: &Engine) -> Img {
        e.gray(self.ref_row)
    }

    pub fn pass1_ref(&self, e: &Engine) -> Pass1Ref {
        Pass1Ref::new(self.ref_gray(e), self.prm.motion, self.levels)
    }

    /// Record the pass-1 alignment of `row` (None = registration failed).
    pub fn p1_add(&mut self, e: &Engine, row: usize, res: Option<(Alignment, f64)>) {
        match res {
            None => self.n_fail += 1,
            Some((al, cc)) => {
                al.accumulate(&e.gray(row), &mut self.mean_acc, &mut self.mean_cnt);
                self.aligns[row] = Some(al);
                self.ccs.push((row, cc));
            }
        }
    }

    /// Reject low-correlation frames; returns the stacked reference image.
    pub fn p1_finish(&mut self, e: &Engine) -> Img {
        let vals: Vec<f64> = self.ccs.iter().map(|c| c.1).collect();
        self.med_cc = median_f64(&vals);
        let dev: Vec<f64> = vals.iter().map(|v| (v - self.med_cc).abs()).collect();
        let mad = 1.4826 * median_f64(&dev) + 1e-4;
        for &(i, v) in &self.ccs {
            if v < self.med_cc - 5.0 * mad && v < 0.98 * self.med_cc && i != self.ref_row {
                self.aligns[i] = None;
                self.n_low += 1;
            }
        }
        let ref_gray = self.ref_gray(e);
        let mut ref_img = Img::new(self.w, self.h);
        for i in 0..self.w * self.h {
            ref_img.d[i] = if self.mean_cnt[i] > 0.5 {
                (self.mean_acc[i] / self.mean_cnt[i].max(1e-9)) as f32
            } else {
                ref_gray.d[i]
            };
        }
        self.mean_acc = vec![];
        self.mean_cnt = vec![];
        // frames in time order; all but the reference split into two halves
        self.rows = (0..self.aligns.len()).filter(|&i| self.aligns[i].is_some()).collect();
        let others: Vec<usize> = self.rows.iter().copied().filter(|&i| i != self.ref_row).collect();
        for &i in &others[others.len() / 2..] {
            self.in_b[i] = true;
        }
        ref_img
    }

    pub fn init_w(&self, row: usize) -> M3 {
        self.aligns[row].as_ref().map_or(EYE, |a| a.w)
    }

    /// Record the pass-2 alignment (None = keep the pass-1 one).
    pub fn p2_set(&mut self, row: usize, res: Option<(Alignment, usize)>) {
        if let Some((al, tiles)) = res {
            self.aligns[row] = Some(al);
            self.tiles.push(tiles as f64);
        }
    }

    pub fn alignment(&self, row: usize) -> &Alignment {
        self.aligns[row].as_ref().expect("row has no alignment")
    }

    pub fn half_b(&self, row: usize) -> bool {
        self.in_b[row]
    }

    /// Full-size accumulators (zeroed), to be filled by the workers' bands.
    pub fn alloc_acc(&mut self) {
        let n = self.wo * self.ho;
        self.acc_a = vec![0.0; n * 3];
        self.acc_b = vec![0.0; n * 3];
        self.wa = vec![0.0; n];
        self.wb = vec![0.0; n];
    }

    pub fn set_acc(&mut self, band: Band) {
        assert!(band.r0 == 0 && band.r1 == self.ho);
        self.acc_a = band.acc_a;
        self.acc_b = band.acc_b;
        self.wa = band.wa;
        self.wb = band.wb;
    }

    pub fn p2_finish(&mut self) {
        self.model = finalize(&self.acc_a, &self.acc_b, &self.wa, &self.wb, self.wo, self.ho, self.prm.scale);
    }

    /// Pass 3: outlier rejection by blotting the model back into each frame.
    /// Rejected pixels are subtracted from the pass-2 sums (undrizzle), which
    /// equals re-drizzling every frame without them.
    pub fn p3(&mut self, e: &Engine, prog: &mut dyn FnMut(u32, usize, usize)) {
        if !self.prm.reject {
            return;
        }
        let (w, h, wo, ho, s) = (self.w, self.h, self.wo, self.ho, self.prm.scale);
        let drop_size = self.prm.pixfrac * s as f64;
        let bmask = border_mask(w, h, self.prm.border, 24);
        let sig = (s as f64 / 12f64.sqrt()).max(0.3);
        let blurred = sep_filter(&self.model, wo, ho, 3, &gaussian_kernel(sig), Border::Reflect101);
        self.model = vec![];
        let (mut n_rej, mut n_pix) = (0usize, 0usize);
        let mut resid = vec![0f32; w * h];
        let mut valid = vec![false; w * h];
        let mut gb = vec![0f32; w * h];
        let (mut bx, mut by) = (vec![], vec![]);
        let mut mark = vec![0u8; wo * ho];
        let mut targets = vec![];
        let mut fbuf = vec![];
        for (k, &i) in self.rows.iter().enumerate() {
            prog(ST_REJECT, k, self.rows.len());
            let al = self.aligns[i].as_ref().unwrap();
            let fr = &e.frames[i];
            blot_coords(al, w, h, s, 8, &mut bx, &mut by);
            let mut px = [0f32; 3];
            for j in 0..w * h {
                let (x, y) = (bx[j] as f64, by[j] as f64);
                sample_clamped(&blurred, wo, ho, 3, x, y, &mut px);
                let f = &fr[j * 3..j * 3 + 3];
                resid[j] = (f[0] as f32 - px[0]).abs().max((f[1] as f32 - px[1]).abs()).max((f[2] as f32 - px[2]).abs());
                valid[j] = bmask[j] > 0.0 && x >= 0.0 && y >= 0.0 && x <= (wo - 1) as f64 && y <= (ho - 1) as f64;
                gb[j] = (px[0] + px[1] + px[2]) / 3.0;
            }
            let nv = valid.iter().filter(|&&v| v).count();
            if nv == 0 {
                continue;
            }
            n_pix += nv;
            let sigma = 1.4826 * hist_median((0..w * h).filter(|&j| valid[j]).map(|j| resid[j])) + 1e-3;
            let dil = morph3(&gb, w, h, true);
            let ero = morph3(&gb, w, h, false);
            let thr = self.prm.reject_sigma as f32 * sigma;
            let bad: Vec<f32> = (0..w * h)
                .map(|j| (valid[j] && resid[j] > thr + 0.5 * (dil[j] - ero[j])) as u8 as f32)
                .collect();
            let bad = morph3(&bad, w, h, true);
            let mut wbad = vec![0f32; w * h];
            let mut list = vec![];
            for j in 0..w * h {
                if bad[j] > 0.0 {
                    n_rej += 1;
                    if bmask[j] > 0.0 {
                        wbad[j] = bmask[j];
                        list.push(j);
                    }
                }
            }
            if list.is_empty() {
                continue;
            }
            to_f32(fr, &mut fbuf);
            let (acc, wacc) = if self.in_b[i] { (&mut self.acc_b, &mut self.wb) } else { (&mut self.acc_a, &mut self.wa) };
            undrizzle_frame(acc, wacc, &fbuf, &wbad, &list, al, w, h, s, drop_size, &mut mark, &mut targets);
        }
        self.rejected = n_rej as f64 / n_pix.max(1) as f64;
        // clamp tiny negative weights left by f32 cancellation
        for v in self.wa.iter_mut().chain(self.wb.iter_mut()) {
            if *v < 1e-6 {
                *v = 0.0;
            }
        }
        self.model = finalize(&self.acc_a, &self.acc_b, &self.wa, &self.wb, wo, ho, s);
    }

    /// Noise from the two half-stacks, deconvolution, diagnostics.
    pub fn finish(mut self, e: &Engine, prog: &mut dyn FnMut(u32, usize, usize)) -> Output {
        let (w, h, wo, ho) = (self.w, self.h, self.wo, self.ho);
        let mut noise = None;
        if self.wb.iter().any(|&v| v > 0.0) {
            let mut nz = vec![0f32; wo * ho * 3];
            for i in 0..wo * ho {
                let (x, y) = (self.wa[i], self.wb[i]);
                if x > 0.0 && y > 0.0 {
                    let k = (x * y).sqrt() / (x + y);
                    for c in 0..3 {
                        nz[i * 3 + c] = (self.acc_a[i * 3 + c] / x - self.acc_b[i * 3 + c] / y) * k;
                    }
                }
            }
            noise = Some(nz);
        }
        self.acc_a = vec![];
        self.acc_b = vec![];
        let warps: Vec<M3> = self.rows.iter().map(|&i| self.aligns[i].as_ref().unwrap().w).collect();
        let coverage = phase_coverage(&warps, w, h);
        let mut dinfo = String::from("null");
        let model = std::mem::take(&mut self.model);
        let (image, raw) = match (self.prm.deconvolve, noise) {
            (true, Some(nz)) => {
                prog(ST_DECONV, 0, 1);
                let (out, info) = wiener_deconvolve(&model, &nz, wo, ho, None, 10.0, 1.0, 64);
                dinfo = format!(
                    "{{\"sigma\":{:.3},\"snr1_freq\":{:.4},\"peak_gain\":{:.2}}}",
                    info.sigma, info.snr1_freq, info.peak_gain
                );
                (out, if self.prm.keep_raw { Some(model) } else { None })
            }
            _ => (model, None),
        };
        let tiles_med = if self.tiles.is_empty() { 0.0 } else { median_f64(&self.tiles) };
        let diag = format!(
            "{{\"frames_total\":{},\"duplicates\":{},\"distinct\":{},\"used\":{},\"failed\":{},\"low_cc\":{},\
             \"reference\":{},\"phase_coverage\":{:.4},\"median_cc\":{:.5},\"rejected_fraction\":{:.6},\
             \"local_tiles\":{},\"width\":{},\"height\":{},\"deconv\":{}}}",
            e.n_total, e.n_dup, e.frames.len(), self.rows.len(), self.n_fail, self.n_low, self.ref_row, coverage,
            self.med_cc, self.rejected, tiles_med, wo, ho, dinfo
        );
        Output { w: wo, h: ho, image, raw, diag }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::Field;

    /// Two half bands (as two workers would compute them) must equal one full band.
    #[test]
    fn bands_add_up() {
        let (w, h) = (64usize, 48usize);
        let prm = Params::default();
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 37) % 251) as u8).collect();
        let th: f64 = 0.01;
        let mut al = Alignment::rigid([th.cos(), -th.sin(), 0.7, th.sin(), th.cos(), -0.4, 0.0, 0.0, 1.0]);
        let d = (0..4 * 3).map(|k| [0.1 * (k % 4) as f32, -0.05 * (k / 4) as f32]).collect();
        al.field = Some(Field { nx: 4, ny: 3, d, origin: 7.5, step: 8.0 });
        let al = Alignment::from_slice(&al.to_vec()).unwrap();
        let ho = h * prm.scale;
        let mut full = Band::new(w, h, &prm, 0, ho);
        full.add(&rgb, &al, false);
        let mut top = Band::new(w, h, &prm, 0, ho / 2);
        let mut bot = Band::new(w, h, &prm, ho / 2, ho);
        top.add(&rgb, &al, false);
        bot.add(&rgb, &al, false);
        let joined: Vec<f32> = top.wa.iter().chain(&bot.wa).copied().collect();
        assert!(full.wa.iter().sum::<f32>() > 1000.0);
        for (a, b) in full.wa.iter().zip(&joined) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
