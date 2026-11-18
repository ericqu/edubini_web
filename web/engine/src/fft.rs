//! 2-D FFT (rustfft rows + transposed columns) and phase correlation.

use crate::img::Img;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

pub type C32 = Complex32;

fn transpose(src: &[C32], dst: &mut [C32], w: usize, h: usize) {
    const B: usize = 32;
    for y0 in (0..h).step_by(B) {
        for x0 in (0..w).step_by(B) {
            for y in y0..(y0 + B).min(h) {
                for x in x0..(x0 + B).min(w) {
                    dst[x * h + y] = src[y * w + x];
                }
            }
        }
    }
}

/// In-place 2-D FFT of a row-major (h, w) complex buffer. Unnormalised both ways.
pub fn fft2(buf: &mut [C32], w: usize, h: usize, inverse: bool) {
    let mut planner = FftPlanner::<f32>::new();
    let plan = |p: &mut FftPlanner<f32>, n| if inverse { p.plan_fft_inverse(n) } else { p.plan_fft_forward(n) };
    plan(&mut planner, w).process(buf);
    let mut t = vec![C32::default(); w * h];
    transpose(buf, &mut t, w, h);
    plan(&mut planner, h).process(&mut t);
    transpose(&t, buf, h, w);
}

/// Index of -k for an n-point FFT.
#[inline(always)]
pub fn neg(k: usize, n: usize) -> usize {
    if k == 0 {
        0
    } else {
        n - k
    }
}

/// numpy.fft.fftfreq(n)[k]
#[inline(always)]
pub fn fftfreq(k: usize, n: usize) -> f64 {
    if k < (n + 1) / 2 {
        k as f64 / n as f64
    } else {
        (k as f64 - n as f64) / n as f64
    }
}

/// Phase correlation against a fixed reference (its spectrum is cached).
/// Semantics of cv2.phaseCorrelate(a, b): returns (dx, dy, response) with
/// b(x + d) ~ a(x), i.e. the content of b is shifted by d relative to a.
pub struct PhaseRef {
    w: usize,
    h: usize,
    spec: Vec<C32>,
    win: Option<Vec<f32>>,
}

impl PhaseRef {
    pub fn new(a: &Img, win: Option<&[f32]>) -> Self {
        let win = win.map(|w| w.to_vec());
        let spec = Self::spectrum(a, win.as_deref());
        PhaseRef { w: a.w, h: a.h, spec, win }
    }

    fn spectrum(a: &Img, win: Option<&[f32]>) -> Vec<C32> {
        let mut buf: Vec<C32> = match win {
            Some(wn) => a.d.iter().zip(wn).map(|(&v, &k)| C32::new(v * k, 0.0)).collect(),
            None => a.d.iter().map(|&v| C32::new(v, 0.0)).collect(),
        };
        fft2(&mut buf, a.w, a.h, false);
        buf
    }

    pub fn correlate(&self, b: &Img) -> (f64, f64, f64) {
        assert!(b.w == self.w && b.h == self.h);
        let (w, h) = (self.w, self.h);
        let mut p = Self::spectrum(b, self.win.as_deref());
        for (q, a) in p.iter_mut().zip(&self.spec) {
            let c = a.conj() * *q;
            let m = c.norm();
            *q = if m > 1e-20 { c / m } else { C32::default() };
        }
        fft2(&mut p, w, h, true);
        let scale = 1.0 / (w * h) as f32;
        let mut best = (0usize, f32::MIN);
        for (i, v) in p.iter().enumerate() {
            if v.re > best.1 {
                best = (i, v.re);
            }
        }
        let (px, py) = ((best.0 % w) as isize, (best.0 / w) as isize);
        // 5x5 weighted centroid around the peak (with wrap-around)
        let (mut s, mut sx, mut sy) = (0f64, 0f64, 0f64);
        for dy in -2isize..=2 {
            for dx in -2isize..=2 {
                let x = (px + dx).rem_euclid(w as isize) as usize;
                let y = (py + dy).rem_euclid(h as isize) as usize;
                let v = (p[y * w + x].re * scale) as f64;
                s += v;
                sx += v * dx as f64;
                sy += v * dy as f64;
            }
        }
        let (cx, cy) = if s.abs() > 1e-12 { (sx / s, sy / s) } else { (0.0, 0.0) };
        let mut dx = px as f64 + cx;
        let mut dy = py as f64 + cy;
        if dx > w as f64 / 2.0 {
            dx -= w as f64;
        }
        if dy > h as f64 / 2.0 {
            dy -= h as f64;
        }
        (dx, dy, s)
    }
}

pub fn phase_correlate(a: &Img, b: &Img, win: Option<&[f32]>) -> (f64, f64, f64) {
    PhaseRef::new(a, win).correlate(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blurred random texture, content shifted by integer (ox, oy) with wrap-around.
    fn texture(w: usize, h: usize, ox: isize, oy: isize) -> Img {
        let mut base = Img::new(w, h);
        let mut s = 12345u64;
        for v in base.d.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (s >> 40) as f32 / (1u64 << 24) as f32;
        }
        let base = crate::img::gaussian_blur(&base, 1.5);
        let mut img = Img::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let sx = (x as isize - ox).rem_euclid(w as isize) as usize;
                let sy = (y as isize - oy).rem_euclid(h as isize) as usize;
                img.d[y * w + x] = base.d[sy * w + sx];
            }
        }
        img
    }

    #[test]
    fn fft_roundtrip() {
        let (w, h) = (12, 10);
        let orig: Vec<C32> = (0..w * h).map(|i| C32::new(i as f32, (i % 7) as f32)).collect();
        let mut b = orig.clone();
        fft2(&mut b, w, h, false);
        fft2(&mut b, w, h, true);
        for (a, b) in orig.iter().zip(&b) {
            assert!((a - b / (w * h) as f32).norm() < 1e-3);
        }
    }

    #[test]
    fn phase_correlation_finds_shift() {
        let a = texture(160, 120, 0, 0);
        let b = texture(160, 120, 3, -5); // content moved by (+3, -5)
        let win = crate::img::hanning(160, 120);
        let (dx, dy, r) = phase_correlate(&a, &b, Some(&win));
        assert!((dx - 3.0).abs() < 0.3 && (dy + 5.0).abs() < 0.3, "{dx} {dy}");
        assert!(r > 0.1);
    }
}
