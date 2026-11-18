"""Measure whether (and how much) multi-frame super-resolution helps a video.

Two complementary measurements:

1. Single-frame spectrum vs temporal noise (`spectrum_snr`): per spatial
   frequency, the power in one frame divided by the power of the frame-to-frame
   noise. Ratios ~1 mean the optics deliver nothing at that frequency. Super-
   resolution needs real signal up to the pixel Nyquist limit (0.5 cycles/px),
   otherwise nothing is aliased and there is nothing to unfold.

2. Fourier Ring Correlation between two independent reconstructions
   (`frc_halves`): the selected frames are split into the first and second half
   of the movie (seconds apart, so noise, compression artifacts and sub-pixel
   phases are independent), each half is reconstructed on the same reference
   grid, and the correlation of the two results is measured per frequency ring.
   Detail that reproduces in both halves is real; the frequency where FRC drops
   below 1/7 is the achieved resolution. Comparing the drizzle FRC with the
   plain aligned average shows what super-resolution actually added.

   (Don't split into even/odd frames: adjacent frames share sub-pixel phase and
   H.264 compression noise, which makes everything look reproducible.)

CLI: `microscope-analyze movie.mov --out-dir results`
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .deconv import wiener_deconvolve
from .superres import _gray, _phase_correlate, enhance, save_image, scan_frames, select_frames, video_source

FRC_THRESHOLD = 1 / 7


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #


def _radial_power(g: np.ndarray, nbins: int) -> np.ndarray:
    n = g.shape[0]
    g = (g - g.mean()) * np.outer(np.hanning(n), np.hanning(n))
    p = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    y, x = np.indices(p.shape)
    r = np.hypot(x - n // 2, y - n // 2) / n
    edges = np.linspace(0, 0.5, nbins + 1)
    return np.array([p[(r >= edges[k]) & (r < edges[k + 1])].mean() for k in range(nbins)])


def spectrum_snr(path: str, nbins: int = 10, patch: int = 256, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Signal/noise power ratio of a single frame per frequency band (cycles/px).

    Noise is estimated from the difference between the frame and the next
    non-duplicate frame, aligned by phase correlation, divided by sqrt(2).
    Returns (band centre frequencies, median ratio over a few patches).
    """
    src = video_source(path)
    a = b = None
    for i, f in enumerate(src()):
        if i < start:
            continue
        g = _gray(f) * 255
        if a is None:
            a = g
        elif np.abs(g - a).mean() > 0.5:
            b = g
            break
    if b is None:
        raise ValueError("need two distinct frames")
    (dx, dy), _ = _phase_correlate(a, b)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    b = cv2.warpAffine(b, m, a.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    noise = (a - b) / np.sqrt(2)
    h, w = a.shape
    ratios = []
    for fy in (0.3, 0.5, 0.7):
        for fx in (0.3, 0.5, 0.7):
            y0 = int(np.clip(fy * h - patch / 2, 0, h - patch))
            x0 = int(np.clip(fx * w - patch / 2, 0, w - patch))
            sl = np.s_[y0 : y0 + patch, x0 : x0 + patch]
            ratios.append(_radial_power(a[sl], nbins) / (_radial_power(noise[sl], nbins) + 1e-12))
    freqs = (np.arange(nbins) + 0.5) * 0.5 / nbins
    return freqs, np.median(ratios, axis=0)


# --------------------------------------------------------------------------- #
# Fourier ring correlation
# --------------------------------------------------------------------------- #


def frc(a: np.ndarray, b: np.ndarray, nbins: int = 20, size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fourier ring correlation of two images (central square crop, Hann window)."""
    if a.ndim == 3:
        a = cv2.cvtColor(np.ascontiguousarray(a, np.float32), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(np.ascontiguousarray(b, np.float32), cv2.COLOR_BGR2GRAY)
    h, w = a.shape
    n = size or min(h, w, 1024)
    y0, x0 = (h - n) // 2, (w - n) // 2
    win = np.outer(np.hanning(n), np.hanning(n))
    ca, cb = a[y0 : y0 + n, x0 : x0 + n], b[y0 : y0 + n, x0 : x0 + n]
    fa = np.fft.fftshift(np.fft.fft2((ca - ca.mean()) * win))
    fb = np.fft.fftshift(np.fft.fft2((cb - cb.mean()) * win))
    y, x = np.indices(fa.shape)
    r = np.hypot(x - n // 2, y - n // 2) / n
    edges = np.linspace(0, 0.5, nbins + 1)
    out = []
    for k in range(nbins):
        s = (r >= edges[k]) & (r < edges[k + 1])
        num = np.real(np.sum(fa[s] * np.conj(fb[s])))
        den = np.sqrt(np.sum(np.abs(fa[s]) ** 2) * np.sum(np.abs(fb[s]) ** 2)) + 1e-30
        out.append(num / den)
    return (np.arange(nbins) + 0.5) * 0.5 / nbins, np.array(out)


def resolution(freqs: np.ndarray, curve: np.ndarray, threshold: float = FRC_THRESHOLD) -> float:
    """First frequency where the FRC drops below threshold (linear interpolation)."""
    below = np.nonzero(curve < threshold)[0]
    if len(below) == 0:
        return float(freqs[-1])
    k = below[0]
    if k == 0:
        return float(freqs[0])
    f0, f1, c0, c1 = freqs[k - 1], freqs[k], curve[k - 1], curve[k]
    return float(f0 + (c0 - threshold) / (c0 - c1) * (f1 - f0))


@dataclass
class FRCResult:
    freqs: np.ndarray  # cycles per OUTPUT pixel
    drizzle: np.ndarray
    average: np.ndarray
    scale: int
    reference: int
    halves: tuple[list[int], list[int]]
    images: dict  # 'driz_a', 'driz_b', 'avg_a', 'avg_b'

    @property
    def input_nyquist(self) -> float:
        """Input-pixel Nyquist limit expressed in cycles per output pixel."""
        return 0.5 / self.scale


def frc_halves(path: str, keep: float = 1.0, reference: int | None = None, verbose: bool = True,
               **enhance_kwargs) -> FRCResult:
    """Reconstruct the first and second half of the selected frames independently
    (same reference grid) and measure their Fourier ring correlation."""
    src = video_source(path)
    scan = scan_frames(src)
    ref, chosen = select_frames(scan, keep=keep, reference=reference,
                                min_overlap=enhance_kwargs.get("min_overlap", 0.5))
    others = [i for i in chosen if i != ref]
    half_a, half_b = others[: len(others) // 2], others[len(others) // 2 :]
    scale = enhance_kwargs.get("scale", 2)
    imgs = {}
    for name, half in (("a", half_a), ("b", half_b)):
        if verbose:
            print(f"Reconstructing half {name.upper()}: frames {half[0]}..{half[-1]} ({len(half)} + reference {ref})")
        r = enhance(src, keep=1.0, reference=ref, subset=set(half), scan=scan, verbose=False, **enhance_kwargs)
        imgs["driz_" + name], imgs["avg_" + name] = r.image, r.baseline
    freqs, fd = frc(imgs["driz_a"], imgs["driz_b"])
    _, fa = frc(imgs["avg_a"], imgs["avg_b"])
    return FRCResult(freqs, fd, fa, scale, ref, (half_a, half_b), imgs)


# --------------------------------------------------------------------------- #
# Visual comparison
# --------------------------------------------------------------------------- #


def comparison_crops(path: str, reference: int, images: dict[str, np.ndarray], scale: int,
                     out_path: str, crop: int = 400, n: int = 3) -> None:
    """Save a grid: rows = crops at a few positions with the most detail,
    columns = reference frame (bicubic) + each given image."""
    frame = None
    for i, f in enumerate(video_source(path)()):
        if i == reference:
            frame = f
            break
    h, w = frame.shape[:2]
    cols = {"reference frame (bicubic)": cv2.resize(frame.astype(np.float32), (w * scale, h * scale),
                                                     interpolation=cv2.INTER_CUBIC)}
    cols.update(images)
    # pick the crops with the most local detail in the first given image
    detail = cv2.cvtColor(np.ascontiguousarray(next(iter(images.values())), np.float32), cv2.COLOR_BGR2GRAY)
    detail = cv2.blur(np.abs(cv2.Laplacian(detail, cv2.CV_32F)), (crop, crop))
    H, W = detail.shape
    m = crop // 2 + 8
    detail[:m], detail[-m:], detail[:, :m], detail[:, -m:] = 0, 0, 0, 0
    rows = []
    for _ in range(n):
        y, x = np.unravel_index(np.argmax(detail), detail.shape)
        y0, x0 = y - crop // 2, x - crop // 2
        detail[max(0, y0 - crop // 2) : y0 + crop + crop // 2, max(0, x0 - crop // 2) : x0 + crop + crop // 2] = 0
        tiles = []
        for label, img in cols.items():
            t = np.clip(img[y0 : y0 + crop, x0 : x0 + crop], 0, 255).astype(np.uint8).copy()
            cv2.putText(t, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    cv2.imwrite(out_path, np.vstack(rows))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _fmt_row(label: str, values, fmt="{:6.2f}") -> str:
    return f"{label:<22s}" + " ".join(fmt.format(v) for v in values)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse how much multi-frame super-resolution can/does help a video.")
    p.add_argument("input_video")
    p.add_argument("--out-dir", default="results", help="Where to write images and CSV (default results/)")
    p.add_argument("--keep", type=float, default=1.0, help="Fraction of sharpest frames to use (default 1)")
    p.add_argument("--reference", type=int, default=None, help="Reference frame (default automatic)")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--pixfrac", type=float, default=0.7)
    p.add_argument("--no-frc", action="store_true", help="Only the (fast) spectrum analysis")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.input_video))[0])

    # 1. single-frame spectrum vs noise
    freqs, snr = spectrum_snr(args.input_video)
    print("\n1) Single-frame signal/noise power per spatial frequency (cycles per input pixel)")
    print(_fmt_row("frequency", freqs, "{:6.3f}"))
    print(_fmt_row("signal/noise", snr, "{:6.1f}"))
    hi = snr[freqs > 0.3].mean()
    print(f"   mean S/N above 0.3 cycles/px: {hi:.2f}  "
          + ("-> real detail near Nyquist: aliasing possible, SR can help"
             if hi > 2 else "-> little/no detail near Nyquist: little aliasing to unfold"))

    # 2. scan: frame quality and motion
    scan = scan_frames(video_source(args.input_video))
    rel = scan.sharpness / np.median(scan.sharpness)
    travel = scan.position.max(axis=0) - scan.position.min(axis=0)
    print(f"\n2) {scan.n_total} frames, {scan.n_dup} duplicates; travel {travel[0]:.0f} x {travel[1]:.0f} px; "
          f"sharpness (rel. to median) p10={np.percentile(rel, 10):.2f} p90={np.percentile(rel, 90):.2f}")

    if args.no_frc:
        return

    # 3. FRC of independent half reconstructions
    r = frc_halves(args.input_video, keep=args.keep, reference=args.reference,
                   scale=args.scale, pixfrac=args.pixfrac)
    print(f"\n3) Fourier ring correlation, first vs second half of the movie (reference frame {r.reference})")
    print("   frequency in cycles per OUTPUT pixel; input-pixel Nyquist = "
          f"{r.input_nyquist:.3f}; reproducible detail if FRC > {FRC_THRESHOLD:.3f}")
    print(_fmt_row("frequency", r.freqs, "{:6.3f}"))
    print(_fmt_row("FRC drizzle", r.drizzle))
    print(_fmt_row("FRC aligned average", r.average))
    rd, ra = resolution(r.freqs, r.drizzle), resolution(r.freqs, r.average)
    print(f"   resolution (FRC={FRC_THRESHOLD:.3f}): drizzle {rd:.3f} c/px (period {1 / rd / r.scale:.2f} input px), "
          f"average {ra:.3f} c/px (period {1 / ra / r.scale:.2f} input px)")
    beyond = r.freqs > r.input_nyquist
    print(f"   mean FRC beyond input Nyquist: drizzle {r.drizzle[beyond].mean():.2f}, "
          f"average {r.average[beyond].mean():.2f} (average cannot hold real detail there)")

    np.savetxt(stem + "_frc.csv", np.column_stack([r.freqs, r.drizzle, r.average]), delimiter=",",
               header="freq_cycles_per_output_px,frc_drizzle,frc_average", comments="")
    driz = (r.images["driz_a"] + r.images["driz_b"]) / 2
    avg = (r.images["avg_a"] + r.images["avg_b"]) / 2
    # Deconvolution (note: FRC cannot judge it, a linear filter scales both halves alike)
    decon, info = wiener_deconvolve(driz, (r.images["driz_a"] - r.images["driz_b"]) / 2)
    print(f"\n4) Wiener deconvolution: fitted PSF sigma {info['sigma']:.2f} output px; "
          f"stack SNR > 1 up to {info['snr1_freq']:.3f} cycles/output px; peak gain {info['peak_gain']:.1f}")
    comparison_crops(args.input_video, r.reference,
                     {"aligned average": avg, "drizzle": driz, "drizzle + deconv": decon},
                     args.scale, stem + "_crops.png")
    save_image(stem + "_drizzle.png", driz)
    save_image(stem + "_deconv.png", decon)
    print(f"\nWrote {stem}_frc.csv, {stem}_crops.png, {stem}_drizzle.png, {stem}_deconv.png")


if __name__ == "__main__":
    main()
