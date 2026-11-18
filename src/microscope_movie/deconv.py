"""Wiener deconvolution with the signal-to-noise ratio measured from the data.

The stacked image M has much less noise than any single frame, so detail that
the optics attenuated (but did not destroy) can be boosted back. How far is set
by the measured SNR, not by a guess:

* `noise` is a noise realisation of M (difference of two independent
  half-stacks, see `superres.Result.noise`). Its power spectrum Pn(f) is the
  noise in M; the power spectrum of M itself is Pm(f) = Ps(f) + Pn(f).
* The Wiener filter for a blur H is   G = (1/H) * Ps / (Ps + Pn)
                                         = (1/H) * W(f),  W = 1 - Pn/Pm,
  so only the blur H needs a model. W is measured directly and is ~1 where the
  signal dominates and ~0 where only noise is left, which is what keeps the
  deconvolution from amplifying noise.
* H is modelled as a Gaussian PSF (optics + pixel aperture + drizzle drop). Its
  width is fitted from the signal spectrum: log Ps = a + b log f - 4 pi^2 sigma^2 f^2
  (a power-law scene blurred by a Gaussian), or given explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SpectrumInfo:
    freqs: np.ndarray  # radial bin centres, cycles per pixel
    power: np.ndarray  # Pm, image power spectrum
    noise_power: np.ndarray  # Pn
    wiener: np.ndarray  # W = Ps / (Ps + Pn), cleaned


def _luma(img: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.reshape(img.shape[:2])


def _tukey2d(h: int, w: int, frac: float = 0.1) -> np.ndarray:
    def tukey(n):
        x = np.ones(n)
        m = max(1, int(frac * n / 2))
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        x[:m], x[-m:] = ramp, ramp[::-1]
        return x

    return np.outer(tukey(h), tukey(w))


def _radial_freq(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fx**2 + fy**2)


def measure_spectra(image: np.ndarray, noise: np.ndarray, nbins: int = 100) -> SpectrumInfo:
    """Radially averaged power spectra of the image and its noise, and the
    resulting Wiener weight W(f) (forced non-increasing, small values zeroed)."""
    m, n = _luma(image), _luma(noise)
    h, w = m.shape
    win = _tukey2d(h, w)
    pm = np.abs(np.fft.fft2((m - m.mean()) * win)) ** 2
    pn = np.abs(np.fft.fft2((n - n.mean()) * win)) ** 2
    r = _radial_freq(h, w)
    edges = np.linspace(0, 0.5, nbins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    cnt = np.bincount(idx.ravel(), minlength=nbins)
    Pm = np.bincount(idx.ravel(), pm.ravel(), minlength=nbins) / np.maximum(cnt, 1)
    Pn = np.bincount(idx.ravel(), pn.ravel(), minlength=nbins) / np.maximum(cnt, 1)
    W = np.clip(1 - Pn / np.maximum(Pm, 1e-30), 0, 1)
    W[W < 0.1] = 0  # below SNR ~0.1 the estimate is just estimation noise
    W = np.minimum.accumulate(W)  # SNR does not come back at higher frequency
    return SpectrumInfo((edges[:-1] + edges[1:]) / 2, Pm, Pn, W)


def fit_signal_model(spec: SpectrumInfo, f_min: float = 0.01):
    """Fit log Ps = a + b log f + c f^2 where SNR > 1. Returns (a, b, c) or None."""
    f = spec.freqs
    ps = spec.power - spec.noise_power
    sel = (f > f_min) & (spec.wiener > 0.5) & (ps > 0)
    if sel.sum() < 5:
        return None
    A = np.column_stack([np.ones(sel.sum()), np.log(f[sel]), f[sel] ** 2])
    wts = spec.wiener[sel]  # trust high-SNR bins more
    coef, *_ = np.linalg.lstsq(A * wts[:, None], np.log(ps[sel]) * wts, rcond=None)
    return coef


def fit_psf_sigma(spec: SpectrumInfo, f_min: float = 0.01) -> float:
    """Gaussian PSF sigma (pixels) fitted to the signal spectrum where SNR > 1."""
    coef = fit_signal_model(spec, f_min)
    if coef is None or coef[2] >= 0:
        return 0.0
    return float(np.sqrt(-coef[2]) / (2 * np.pi))


def smooth_wiener(spec: SpectrumInfo, f_min: float = 0.01) -> np.ndarray:
    """W(f) = SNR/(1+SNR) from the fitted signal model and a smoothed noise
    spectrum. Unlike the raw measurement it decays smoothly, so the filter has
    no sharp cut-off (which would ring)."""
    coef = fit_signal_model(spec, f_min)
    if coef is None:
        return spec.wiener
    f = np.maximum(spec.freqs, 1e-6)
    ps_model = np.exp(coef[0] + coef[1] * np.log(f) + coef[2] * f**2)
    log_pn = np.log(np.maximum(spec.noise_power, 1e-30))
    k = 7
    pn = np.exp(np.convolve(np.pad(log_pn, k // 2, mode="edge"), np.ones(k) / k, mode="valid"))
    snr = ps_model / pn
    w = snr / (1 + snr)
    w[spec.freqs <= f_min] = 1.0
    return np.minimum.accumulate(w)


def wiener_deconvolve(image: np.ndarray, noise: np.ndarray, sigma: float | None = None,
                      max_gain: float = 10.0, strength: float = 1.0, pad: int = 64):
    """Deconvolve `image` (H, W[, C]) using the noise realisation `noise`.

    sigma     Gaussian PSF sigma in pixels of `image`; None = fit from the data
    max_gain  cap on the filter gain at any frequency (safety valve)
    strength  0 = no change, 1 = full Wiener filter
    Returns (deconvolved image, info dict).
    """
    spec = measure_spectra(image, noise)
    fitted = fit_psf_sigma(spec)
    sigma = fitted if sigma is None else sigma

    img = np.ascontiguousarray(image, np.float32)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    h, w, c = img.shape
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    if padded.ndim == 2:
        padded = padded[..., None]
    ph, pw = padded.shape[:2]
    r = _radial_freq(ph, pw)
    W = np.interp(r, spec.freqs, smooth_wiener(spec), right=0.0)
    H = np.exp(-2 * np.pi**2 * sigma**2 * r**2)
    G = W / np.maximum(H, 1e-12)
    G = max_gain * np.tanh(G / max_gain)  # soft cap: no kink in the filter
    G = 1 + strength * (G - 1)
    out = np.empty_like(padded)
    for k in range(c):
        out[..., k] = np.real(np.fft.ifft2(np.fft.fft2(padded[..., k]) * G))
    out = out[pad : pad + h, pad : pad + w]
    if squeeze:
        out = out[..., 0]
    f_snr1 = spec.freqs[np.nonzero(spec.wiener >= 0.5)[0][-1]] if (spec.wiener >= 0.5).any() else 0.0
    info = dict(sigma=sigma, sigma_fitted=fitted, snr1_freq=float(f_snr1),
                peak_gain=float(G.max()), spectrum=spec)
    return out, info
