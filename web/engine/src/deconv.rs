//! Wiener deconvolution with the SNR measured from the data (port of deconv.py).
//! Images are interleaved RGB f32, 0..255.

use crate::fft::{fft2, fftfreq, neg, C32};
use crate::img::{bidx, Border};

const NBINS: usize = 100;

pub struct Spectrum {
    pub freqs: Vec<f64>,
    pub power: Vec<f64>,
    pub noise_power: Vec<f64>,
    pub wiener: Vec<f64>,
}

pub struct DeconvInfo {
    pub sigma: f64,
    pub snr1_freq: f64,
    pub peak_gain: f64,
}

fn luma(img: &[f32]) -> Vec<f32> {
    img.chunks_exact(3).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect()
}

fn tukey(n: usize, frac: f64) -> Vec<f64> {
    let mut x = vec![1.0; n];
    let m = ((frac * n as f64 / 2.0) as usize).max(1);
    for k in 0..m.min(n) {
        let r = 0.5 * (1.0 - (std::f64::consts::PI * k as f64 / m as f64).cos());
        x[k] = r;
        x[n - 1 - k] = r;
    }
    x
}

#[inline(always)]
fn radial(kx: usize, ky: usize, w: usize, h: usize) -> f64 {
    fftfreq(kx, w).hypot(fftfreq(ky, h))
}

/// Radially averaged power spectra of image and noise (luma), and W = 1 - Pn/Pm.
pub fn measure_spectra(image: &[f32], noise: &[f32], w: usize, h: usize) -> Spectrum {
    let (m, n) = (luma(image), luma(noise));
    let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
    let (mm, mn) = (mean(&m), mean(&n));
    let (ty, tx) = (tukey(h, 0.1), tukey(w, 0.1));
    // Both real signals in one complex FFT: z = m + i n.
    let mut z: Vec<C32> = (0..w * h)
        .map(|i| {
            let wn = ty[i / w] * tx[i % w];
            C32::new(((m[i] as f64 - mm) * wn) as f32, ((n[i] as f64 - mn) * wn) as f32)
        })
        .collect();
    fft2(&mut z, w, h, false);
    let mut pm = vec![0f64; NBINS];
    let mut pn = vec![0f64; NBINS];
    let mut cnt = vec![0f64; NBINS];
    for ky in 0..h {
        for kx in 0..w {
            let a = z[ky * w + kx];
            let b = z[neg(ky, h) * w + neg(kx, w)].conj();
            let mk = (a + b) * 0.5; // spectrum of m
            let nk = (a - b) * 0.5; // i * spectrum of n (same magnitude)
            let r = radial(kx, ky, w, h);
            let idx = ((r * 2.0 * NBINS as f64) as usize).min(NBINS - 1);
            pm[idx] += mk.norm_sqr() as f64;
            pn[idx] += nk.norm_sqr() as f64;
            cnt[idx] += 1.0;
        }
    }
    let mut wv = vec![0f64; NBINS];
    for k in 0..NBINS {
        let c = cnt[k].max(1.0);
        pm[k] /= c;
        pn[k] /= c;
        let mut v = (1.0 - pn[k] / pm[k].max(1e-30)).clamp(0.0, 1.0);
        if v < 0.1 {
            v = 0.0;
        }
        wv[k] = if k > 0 { v.min(wv[k - 1]) } else { v };
    }
    let freqs = (0..NBINS).map(|k| (k as f64 + 0.5) * 0.5 / NBINS as f64).collect();
    Spectrum { freqs, power: pm, noise_power: pn, wiener: wv }
}

/// Weighted fit log Ps = a + b log f + c f^2 where SNR > 1.
fn fit_signal_model(s: &Spectrum, f_min: f64) -> Option<[f64; 3]> {
    let mut ata = [[0f64; 3]; 3];
    let mut aty = [0f64; 3];
    let mut n = 0;
    for k in 0..s.freqs.len() {
        let f = s.freqs[k];
        let ps = s.power[k] - s.noise_power[k];
        if !(f > f_min && s.wiener[k] > 0.5 && ps > 0.0) {
            continue;
        }
        n += 1;
        let wt2 = s.wiener[k] * s.wiener[k];
        let row = [1.0, f.ln(), f * f];
        let y = ps.ln();
        for i in 0..3 {
            aty[i] += wt2 * row[i] * y;
            for j in 0..3 {
                ata[i][j] += wt2 * row[i] * row[j];
            }
        }
    }
    if n < 5 {
        return None;
    }
    let m = [ata[0][0], ata[0][1], ata[0][2], ata[1][0], ata[1][1], ata[1][2], ata[2][0], ata[2][1], ata[2][2]];
    let i = crate::img::inv(&m);
    if !i.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some([
        i[0] * aty[0] + i[1] * aty[1] + i[2] * aty[2],
        i[3] * aty[0] + i[4] * aty[1] + i[5] * aty[2],
        i[6] * aty[0] + i[7] * aty[1] + i[8] * aty[2],
    ])
}

fn fit_psf_sigma(s: &Spectrum) -> f64 {
    match fit_signal_model(s, 0.01) {
        Some(c) if c[2] < 0.0 => (-c[2]).sqrt() / (2.0 * std::f64::consts::PI),
        _ => 0.0,
    }
}

fn smooth_wiener(s: &Spectrum) -> Vec<f64> {
    let f_min = 0.01;
    let Some(c) = fit_signal_model(s, f_min) else {
        return s.wiener.clone();
    };
    let nb = s.freqs.len();
    let log_pn: Vec<f64> = s.noise_power.iter().map(|&p| p.max(1e-30).ln()).collect();
    let k = 7isize;
    let mut out = vec![0f64; nb];
    for i in 0..nb {
        let mut acc = 0.0;
        for j in -(k / 2)..=(k / 2) {
            acc += log_pn[(i as isize + j).clamp(0, nb as isize - 1) as usize];
        }
        let pn = (acc / k as f64).exp();
        let f = s.freqs[i].max(1e-6);
        let ps = (c[0] + c[1] * f.ln() + c[2] * f * f).exp();
        let snr = ps / pn;
        let mut w = snr / (1.0 + snr);
        if s.freqs[i] <= f_min {
            w = 1.0;
        }
        out[i] = if i > 0 { w.min(out[i - 1]) } else { w };
    }
    out
}

/// Deconvolve `image` (h, w, 3) using its noise realisation. sigma: Gaussian PSF
/// in pixels (None = fitted). Returns (deconvolved image, info).
pub fn wiener_deconvolve(image: &[f32], noise: &[f32], w: usize, h: usize, sigma: Option<f64>,
                         max_gain: f64, strength: f64, pad: usize) -> (Vec<f32>, DeconvInfo) {
    let spec = measure_spectra(image, noise, w, h);
    let sigma = sigma.unwrap_or_else(|| fit_psf_sigma(&spec));
    let sw = smooth_wiener(&spec);
    let (pw, ph) = (w + 2 * pad, h + 2 * pad);
    // Filter G(r): radial, real and even, so two real channels can share one
    // complex FFT (z = R + iG -> filtered R' + iG').
    let step = 0.5 / NBINS as f64;
    let interp = |r: f64| -> f64 {
        let t = r / step - 0.5;
        if t <= 0.0 {
            sw[0]
        } else if t > (NBINS - 1) as f64 {
            0.0
        } else {
            let i = (t as usize).min(NBINS - 2);
            let f = t - i as f64;
            sw[i] + f * (sw[i + 1] - sw[i])
        }
    };
    let mut gfilt = vec![0f32; pw * ph];
    let mut peak = 0f64;
    let two_pi2 = 2.0 * std::f64::consts::PI.powi(2);
    for ky in 0..ph {
        for kx in 0..pw {
            let r = radial(kx, ky, pw, ph);
            let wv = interp(r);
            let hv = (-two_pi2 * sigma * sigma * r * r).exp();
            let mut g = wv / hv.max(1e-12);
            g = max_gain * (g / max_gain).tanh();
            g = 1.0 + strength * (g - 1.0);
            peak = peak.max(g);
            gfilt[ky * pw + kx] = g as f32;
        }
    }
    let mut out = vec![0f32; w * h * 3];
    let pairs: [(usize, Option<usize>); 2] = [(0, Some(1)), (2, None)];
    let mut z = vec![C32::default(); pw * ph];
    for (c_re, c_im) in pairs {
        for py in 0..ph {
            let sy = bidx(py as isize - pad as isize, h, Border::Reflect101);
            for px in 0..pw {
                let sx = bidx(px as isize - pad as isize, w, Border::Reflect101);
                let p = &image[(sy * w + sx) * 3..];
                z[py * pw + px] = C32::new(p[c_re], c_im.map_or(0.0, |c| p[c]));
            }
        }
        fft2(&mut z, pw, ph, false);
        for (v, &g) in z.iter_mut().zip(&gfilt) {
            *v *= g;
        }
        fft2(&mut z, pw, ph, true);
        let norm = 1.0 / (pw * ph) as f32;
        for y in 0..h {
            for x in 0..w {
                let v = z[(y + pad) * pw + x + pad] * norm;
                out[(y * w + x) * 3 + c_re] = v.re;
                if let Some(c) = c_im {
                    out[(y * w + x) * 3 + c] = v.im;
                }
            }
        }
    }
    let snr1 = (0..NBINS).rev().find(|&k| spec.wiener[k] >= 0.5).map_or(0.0, |k| spec.freqs[k]);
    (out, DeconvInfo { sigma, snr1_freq: snr1, peak_gain: peak })
}
