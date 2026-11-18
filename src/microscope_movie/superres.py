"""Multi-frame super-resolution for microscope videos (Drizzle).

Pipeline
--------
0. Scan: drop duplicate frames (no new samples), measure each frame's sharpness,
   chain phase correlation into a coarse motion trajectory. Pick the reference
   frame (the output view) and the frames to stack: those overlapping the
   reference view, optionally only the sharpest ones ("lucky imaging").
1. Register each selected frame to the reference: coarse-to-fine ECC (OpenCV)
   initialised from the trajectory, then optional local refinement (per-tile
   displacement field for rolling shutter / non-flat specimens).
2. Average the aligned frames into a low-noise reference, re-register against
   it, and drizzle every frame onto a grid `scale` times finer (Rust kernel).
3. Optional robust pass: "blot" the drizzled model back into each frame,
   reject pixels that disagree by more than k*sigma (compression glitches,
   flickering speculars, moving debris) and drizzle again without them.
4. Fill poorly covered output pixels by normalised convolution, sharpen, save.

The frames are streamed from the source on every pass, so memory use does not
grow with the length of the video.
"""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np

if __package__:
    from . import _drizzle
else:  # run as a script: `uv run src/microscope_movie/superres.py ...`
    from microscope_movie import _drizzle

FrameSource = Callable[[], Iterable[np.ndarray]]

MOTION_MODELS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


def video_source(path: str, max_frames: int | None = None) -> FrameSource:
    """Return a callable that yields the BGR uint8 frames of a video, restartable."""

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file {path}")
    cap.release()

    def frames():
        cap = cv2.VideoCapture(str(path))
        try:
            n = 0
            while max_frames is None or n < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
                n += 1
        finally:
            cap.release()

    return frames


@dataclass
class Result:
    image: np.ndarray  # float32, (H*s, W*s, C), 0..255 scale
    baseline: np.ndarray  # aligned plain average, bicubic-upscaled to same size
    weight: np.ndarray  # drizzle weight map (coverage)
    warps: dict[int, np.ndarray]  # frame index -> 3x3 warp (reference -> frame)
    diagnostics: dict = field(default_factory=dict)
    # Noise realisation of `image`: difference of two independent half-stacks
    # (first / second half of the movie), scaled to the noise of the full stack.
    # Used to measure the per-frequency SNR for Wiener deconvolution.
    noise: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3:
        frame = frame[..., 0]
    return frame.astype(np.float32) / 255.0


def _phase_correlate(a: np.ndarray, b: np.ndarray, window: np.ndarray | None = None):
    """cv2.phaseCorrelate on copies: OpenCV 5.0.0 multiplies the window into its
    inputs in place, which silently corrupts a reused reference image."""
    if window is None:
        return cv2.phaseCorrelate(a.copy(), b.copy())
    return cv2.phaseCorrelate(a.copy(), b.copy(), window)


def _pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _rescale_warp(w: np.ndarray, f: float) -> np.ndarray:
    """Express a 3x3 warp in coordinates scaled by f (pyrDown: x_coarse = x/2)."""
    d = np.diag([f, f, 1.0])
    return d @ w @ np.linalg.inv(d)


class Registrar:
    def __init__(self, reference: np.ndarray, motion: str, levels: int):
        self.mode = MOTION_MODELS[motion]
        self.levels = levels
        self.ref_pyr = _pyramid(reference, levels)

    def _ecc(self, img_pyr, init: np.ndarray, levels: int) -> tuple[np.ndarray, float]:
        w = init.copy()
        cc = 0.0
        for lvl in reversed(range(levels)):
            f = 0.5**lvl
            wl = _rescale_warp(w, f)
            warp = wl.astype(np.float32) if self.mode == cv2.MOTION_HOMOGRAPHY else wl[:2].astype(np.float32)
            iters = 50 if lvl == 0 else 100
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
            cc, warp = cv2.findTransformECC(self.ref_pyr[lvl], img_pyr[lvl], warp, self.mode, crit, None, 5)
            wl = np.eye(3)
            wl[: warp.shape[0]] = warp
            w = _rescale_warp(wl, 1.0 / f)
        return w, float(cc)

    def register(self, img: np.ndarray, init: np.ndarray, levels: int | None = None):
        """Warp mapping reference coords -> img coords, and the ECC correlation.

        Returns (None, 0.0) if registration fails.
        """
        levels = self.levels if levels is None else min(levels, self.levels)
        img_pyr = _pyramid(img, levels)
        try:
            return self._ecc(img_pyr, init, levels)
        except cv2.error:
            pass
        # Fallback: initialise translation from phase correlation and retry.
        (dx, dy), _ = _phase_correlate(self.ref_pyr[0], img_pyr[0])
        guess = np.eye(3)
        guess[0, 2], guess[1, 2] = dx, dy
        try:
            return self._ecc(img_pyr, guess, self.levels)
        except cv2.error:
            return None, 0.0


def _phase_coverage(warps: dict[int, np.ndarray], shape, bins: int = 4) -> tuple[float, np.ndarray]:
    """Fraction of (bins x bins) sub-pixel phase cells hit by the frames.

    Phase = fractional part of where the frame's centre pixel lands in the
    reference grid. Super-resolution needs these spread over the unit cell.
    """
    h, w = shape
    c = np.array([w // 2, h // 2, 1.0])
    phases = []
    for wp in warps.values():
        r = np.linalg.inv(wp) @ c
        r = r[:2] / r[2]
        phases.append(r - np.floor(r))
    phases = np.array(phases)
    # Centre the histogram on the circular mean phase so that registration jitter
    # around a single phase (no dither) is not split across the 0/1 wrap.
    ang = 2 * np.pi * phases
    mean = np.arctan2(np.sin(ang).mean(axis=0), np.cos(ang).mean(axis=0)) / (2 * np.pi)
    centred = np.mod(phases - mean + 0.5 / bins, 1.0)
    idx = np.minimum((centred * bins).astype(int), bins - 1)
    hist = np.zeros((bins, bins), int)
    np.add.at(hist, (idx[:, 1], idx[:, 0]), 1)
    return float((hist > 0).mean()), phases


# --------------------------------------------------------------------------- #
# Local (non-rigid) alignment
# --------------------------------------------------------------------------- #


def _apply_h(h: np.ndarray, x: np.ndarray, y: np.ndarray):
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w


@dataclass
class Alignment:
    """Maps reference coords r to frame coords: W(r + d(r)).

    W is the global warp from ECC; d is a smooth local displacement field sampled
    on a coarse tile grid (None = rigid). The field corrects what a global model
    cannot: rolling-shutter skew, a non-flat specimen, lens distortion.
    """

    W: np.ndarray
    field: np.ndarray | None = None  # (ny, nx, 2) float32
    origin: float = 0.0  # reference coordinate of grid node 0
    step: float = 1.0  # reference pixels between grid nodes

    def disp(self, rx, ry):
        if self.field is None:
            return 0.0, 0.0
        gx = ((rx - self.origin) / self.step).astype(np.float32)
        gy = ((ry - self.origin) / self.step).astype(np.float32)
        f = cv2.remap(self.field, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return f[..., 0], f[..., 1]

    def ref_to_frame(self, rx, ry):
        dx, dy = self.disp(rx, ry)
        return _apply_h(self.W, rx + dx, ry + dy)

    def frame_to_ref(self, fx, fy, iters: int = 3):
        yx, yy = _apply_h(np.linalg.inv(self.W), fx, fy)
        rx, ry = yx, yy
        for _ in range(iters):  # solve r + d(r) = y by fixed-point iteration
            dx, dy = self.disp(rx, ry)
            rx, ry = yx - dx, yy - dy
        return rx, ry

    def warp_to_ref(self, img: np.ndarray, shape, interp=cv2.INTER_LINEAR) -> np.ndarray:
        """Resample a frame (or mask) onto the reference grid."""
        if self.field is None:
            return cv2.warpPerspective(img, self.W, shape[::-1], flags=interp | cv2.WARP_INVERSE_MAP)
        ry, rx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
        fx, fy = self.ref_to_frame(rx, ry)
        return cv2.remap(img, fx.astype(np.float32), fy.astype(np.float32), interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def output_map(self, out_shape, scale: int) -> np.ndarray:
        """(Ho, Wo, 2) frame coordinates of every output pixel centre."""
        v, u = np.mgrid[0 : out_shape[0], 0 : out_shape[1]].astype(np.float32)
        off = (scale - 1) / 2
        fx, fy = self.ref_to_frame((u - off) / scale, (v - off) / scale)
        return np.ascontiguousarray(np.stack([fx, fy], axis=-1), dtype=np.float32)

    def blot_map(self, frame_shape, scale: int):
        """Output-grid coordinates of every input pixel centre (for blotting)."""
        fy, fx = np.mgrid[0 : frame_shape[0], 0 : frame_shape[1]].astype(np.float32)
        rx, ry = self.frame_to_ref(fx, fy)
        off = (scale - 1) / 2
        return (rx * scale + off).astype(np.float32), (ry * scale + off).astype(np.float32)


def _local_field(ref: np.ndarray, aligned: np.ndarray, valid: np.ndarray, tile: int, pool):
    """Smooth residual displacement of `aligned` (a frame already globally warped
    onto the reference grid) relative to `ref`, from per-tile sub-pixel ECC.

    Returns (field, origin, step, n_good) where field[k, l] is the displacement d
    with aligned(r + d) ~ ref(r) at tile centre (k, l).
    """
    h, w = ref.shape
    step = tile // 2
    ys = list(range(0, h - tile + 1, step))
    xs = list(range(0, w - tile + 1, step))
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)

    def one(yx):
        y, x = yx
        if valid[y : y + tile, x : x + tile].mean() < 0.98:
            return None
        t = ref[y : y + tile, x : x + tile]
        if t.std() < 0.006:  # featureless (e.g. blank background): no information
            return None
        try:
            cc, wp = cv2.findTransformECC(t, aligned[y : y + tile, x : x + tile],
                                          np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION,
                                          crit, None, 5)
        except cv2.error:
            return None
        if cc < 0.5 or max(abs(wp[0, 2]), abs(wp[1, 2])) > tile / 4:
            return None
        return wp[0, 2], wp[1, 2], cc

    res = list(pool.map(one, [(y, x) for y in ys for x in xs]))
    d = np.zeros((len(ys), len(xs), 2), np.float32)
    conf = np.zeros((len(ys), len(xs)), np.float32)
    for k, r in enumerate(res):
        if r is not None:
            d[k // len(xs), k % len(xs)] = r[:2]
            conf[k // len(xs), k % len(xs)] = r[2]
    # Drop tiles that disagree with their neighbours by > 1 px (bad matches).
    ok = (conf > 0).astype(np.float32)
    den = cv2.blur(ok, (3, 3), borderType=cv2.BORDER_CONSTANT)
    nb = np.zeros_like(d)
    for c in range(2):
        num = cv2.blur(d[..., c] * ok, (3, 3), borderType=cv2.BORDER_CONSTANT)
        nb[..., c] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    conf[np.linalg.norm(d - nb, axis=2) > 1.0] = 0
    n_good = int((conf > 0).sum())
    # Normalised-convolution smoothing fills featureless tiles from neighbours and
    # keeps the field smooth at the tile scale: it corrects geometry but cannot
    # "explain away" the fine aliased detail we want to recover.
    wgt = conf**2
    wsm = cv2.GaussianBlur(wgt, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
    field = np.zeros_like(d)
    for c in range(2):
        num = cv2.GaussianBlur(d[..., c] * wgt, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
        field[..., c] = np.divide(num, wsm, out=np.zeros_like(num), where=wsm > 1e-6)
    return field, (tile - 1) / 2.0, float(step), n_good


# --------------------------------------------------------------------------- #
# Drizzle helpers
# --------------------------------------------------------------------------- #


def _scale_matrix(s: int) -> np.ndarray:
    # OpenCV pixel-centre convention: u + 0.5 = s * (x + 0.5)
    return np.array([[s, 0, (s - 1) / 2], [0, s, (s - 1) / 2], [0, 0, 1.0]])


def _border_mask(shape, border: int, feather: int = 24) -> np.ndarray:
    """Input-pixel weights: 0 on the outer `border` pixels, then a cosine ramp
    over `feather` pixels. Feathering avoids seams in the stack where a frame's
    edge ends (invisible at first, but deconvolution would amplify them)."""
    def ramp(n):
        d = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float32) - border + 1
        return np.clip(0.5 - 0.5 * np.cos(np.pi * np.clip(d / max(feather, 1), 0, 1)), 0, 1)

    h, w = shape
    return np.outer(ramp(h), ramp(w)).astype(np.float32)


def _finalize(acc: np.ndarray, wacc: np.ndarray, scale: int) -> np.ndarray:
    """acc / wacc, with low-coverage pixels blended into a normalised-convolution fill."""
    w = wacc[..., None]
    direct = np.divide(acc, w, out=np.zeros_like(acc), where=w > 0)
    sigma = max(1.0, 0.5 * scale)
    ga = cv2.GaussianBlur(acc, (0, 0), sigma).reshape(acc.shape)
    gw = cv2.GaussianBlur(wacc, (0, 0), sigma)[..., None]
    fill = np.divide(ga, gw, out=np.zeros_like(acc), where=gw > 0)
    ref = np.median(wacc[wacc > 0]) if np.any(wacc > 0) else 1.0
    t = np.clip(wacc / (0.25 * ref), 0, 1)[..., None]
    return t * direct + (1 - t) * fill


def _progress(label: str, i: int, total: int | None):
    if total:
        sys.stderr.write(f"\r  {label}: {i}/{total}")
    else:
        sys.stderr.write(f"\r  {label}: {i}")
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def _as_float(frame: np.ndarray) -> np.ndarray:
    f = np.ascontiguousarray(frame, dtype=np.float32)
    return f[..., None] if f.ndim == 2 else f


def _align(registrar: Registrar, g, init, local: bool, tile: int, pool, levels=None):
    """Global ECC, then (optionally) a local displacement field. (Alignment|None, cc, n_tiles)"""
    W, cc = registrar.register(g, init, levels)
    if W is None:
        return None, 0.0, 0
    al = Alignment(W)
    if not local:
        return al, cc, 0
    shape = g.shape
    aligned = al.warp_to_ref(g, shape)
    valid = al.warp_to_ref(np.ones_like(g), shape, cv2.INTER_NEAREST) > 0
    field, origin, step, n = _local_field(registrar.ref_pyr[0], aligned, valid, tile, pool)
    return Alignment(W, field, origin, step), cc, n


def _drizzle_frame(acc, wacc, f, weight, al: Alignment, scale: int, drop: float):
    if al.field is None:
        H = _scale_matrix(scale) @ np.linalg.inv(al.W)  # frame coords -> output coords
        H /= H[2, 2]
        _drizzle.drizzle_add(acc, wacc, f, weight, H.ravel().tolist(), np.linalg.inv(H).ravel().tolist(), drop)
    else:
        _drizzle.drizzle_add_map(acc, wacc, f, weight, al.output_map(acc.shape[:2], scale), drop)


def _sharpness(g: np.ndarray) -> float:
    """Blur metric: fine-band (sigma 1-2) over mid-band (sigma 2-4) energy.

    The ratio measures how fast the spectrum falls off (defocus, motion blur)
    and is largely independent of how much texture happens to be in view.
    """
    g1 = cv2.GaussianBlur(g, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(g, (0, 0), 2.0)
    g4 = cv2.GaussianBlur(g, (0, 0), 4.0)
    return float(np.var(g1 - g2) / (np.var(g2 - g4) + 1e-12))


@dataclass
class FrameScan:
    index: np.ndarray  # frame numbers of the non-duplicate frames
    sharpness: np.ndarray
    position: np.ndarray  # (n, 2) cumulative content shift in px (coarse trajectory)
    shape: tuple[int, int]
    n_total: int
    n_dup: int

    def overlap(self, row: int) -> np.ndarray:
        """Approximate overlap fraction of every frame with frame `row`."""
        h, w = self.shape
        d = np.abs(self.position - self.position[row])
        return np.clip(1 - d[:, 0] / w, 0, 1) * np.clip(1 - d[:, 1] / h, 0, 1)


def scan_frames(source: FrameSource, dedupe_threshold: float = 0.3, progress=None) -> FrameScan:
    """One cheap pass: drop duplicates, measure sharpness, chain phase correlation
    between consecutive frames into a coarse trajectory."""
    idx, sharp, pos = [], [], []
    prev = prev_small = win = None
    p = np.zeros(2)
    n_total = n_dup = 0
    shape = None
    for i, frame in enumerate(source()):
        n_total += 1
        if progress:
            progress(i + 1)
        g = _gray(frame)
        if prev is not None and np.mean(np.abs(g - prev)) * 255 < dedupe_threshold:
            n_dup += 1
            continue
        small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if win is None:
            shape = g.shape
            win = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
        else:
            (dx, dy), _ = _phase_correlate(prev_small, small, win)
            p = p + 2 * np.array([dx, dy])
        idx.append(i)
        sharp.append(_sharpness(g))
        pos.append(p.copy())
        prev, prev_small = g, small
    if not idx:
        raise ValueError("Video contains no frames.")
    return FrameScan(np.array(idx), np.array(sharp), np.array(pos), shape, n_total, n_dup)


def select_frames(scan: FrameScan, keep: float = 1.0, reference: int | None = None,
                  min_overlap: float = 0.5, subset=None) -> tuple[int, list[int]]:
    """Choose the reference frame and the frames to stack.

    reference   frame number to use as the output view; None = automatic: among
                the sharpest quarter of frames, the one whose view is shared by
                the most (sharpness-weighted) other frames
    keep        fraction of the overlapping frames to keep, sharpest first
    min_overlap frames overlapping the reference view less than this are skipped
    subset      optional set of frame numbers allowed (the reference is always used)
    """
    n = len(scan.index)
    rank = scan.sharpness / np.median(scan.sharpness)
    if reference is None:
        cand = np.argsort(-scan.sharpness)[: max(1, n // 4)]
        scores = [np.sum(scan.overlap(r) * (scan.overlap(r) >= min_overlap) * rank) for r in cand]
        ref_row = int(cand[int(np.argmax(scores))])
    else:
        rows = np.nonzero(scan.index == reference)[0]
        if len(rows) == 0:
            # duplicate or out of range: take the nearest kept frame
            rows = [int(np.argmin(np.abs(scan.index - reference)))]
        ref_row = int(rows[0])
    ok = scan.overlap(ref_row) >= min_overlap
    if subset is not None:
        ok &= np.isin(scan.index, list(subset))
    ok[ref_row] = False
    rows = np.nonzero(ok)[0]
    if keep < 1.0 and len(rows):
        n_keep = max(1, int(round(keep * len(rows))))
        rows = rows[np.argsort(-scan.sharpness[rows])[:n_keep]]
    chosen = sorted([int(scan.index[ref_row])] + [int(scan.index[r]) for r in rows])
    return int(scan.index[ref_row]), chosen


def _get_frame(source: FrameSource, index: int) -> np.ndarray:
    for i, frame in enumerate(source()):
        if i == index:
            return frame
    raise IndexError(f"frame {index} not found")


def enhance(
    source: FrameSource,
    scale: int = 2,
    pixfrac: float = 0.7,
    motion: str = "euclidean",
    local: bool = True,
    tile: int = 96,
    reject: bool = True,
    reject_sigma: float = 4.0,
    keep: float = 1.0,
    reference: int | None = None,
    min_overlap: float = 0.5,
    subset=None,
    dedupe_threshold: float = 0.3,
    border: int = 2,
    total_frames: int | None = None,
    verbose: bool = True,
    scan: FrameScan | None = None,
) -> Result:
    """Multi-frame super-resolution of the frames produced by `source`.

    scale            output grid is `scale` times finer than the input
    pixfrac          drop size as a fraction of the input pixel (0 < pixfrac <= 1);
                     smaller = sharper but needs more, better-dithered frames
    motion           global model: translation | euclidean | affine | homography
    local            refine the global model with a smooth per-tile displacement
                     field (handles rolling shutter, non-flat specimens)
    tile             tile size in pixels for the local refinement
    reject           second drizzle pass rejecting outlier pixels
    reject_sigma     rejection threshold in units of per-frame noise sigma
    keep             fraction of frames to stack, sharpest first (lucky imaging)
    reference        frame number defining the output view (None = automatic)
    min_overlap      skip frames overlapping the reference view less than this
    subset           optional set of frame numbers allowed (used by analysis)
    dedupe_threshold frames whose mean abs difference to the previous frame is
                     below this (8-bit levels) are treated as duplicates
    border           input pixels ignored at the frame edges
    scan             reuse a previous `scan_frames` result (skips pass 0)
    """
    if motion not in MOTION_MODELS:
        raise ValueError(f"motion must be one of {list(MOTION_MODELS)}")
    if not 0 < pixfrac <= 1:
        raise ValueError("pixfrac must be in (0, 1]")
    if not 0 < keep <= 1:
        raise ValueError("keep must be in (0, 1]")
    if scale < 1:
        raise ValueError("scale must be >= 1")
    log = print if verbose else (lambda *a, **k: None)
    prog = _progress if verbose else (lambda *a: None)
    pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    n_passes = 3 if reject else 2
    try:
        # ---- Pass 0: scan (duplicates, sharpness, coarse trajectory) + select ----
        if scan is None:
            log("Pass 0: scanning frames (duplicates, sharpness, coarse motion)...")
            scan = scan_frames(source, dedupe_threshold, lambda i: prog("scan", i, total_frames))
            if verbose:
                sys.stderr.write("\n")
        n_total, n_dup, shape = scan.n_total, scan.n_dup, scan.shape
        ref_idx, chosen = select_frames(scan, keep, reference, min_overlap, subset)
        row_of = {int(k): r for r, k in enumerate(scan.index)}
        ref_row = row_of[ref_idx]
        sharp_rel = scan.sharpness / np.median(scan.sharpness)
        log(f"  {n_total} frames, {n_dup} duplicates; reference frame {ref_idx} "
            f"(sharpness {sharp_rel[ref_row]:.2f}x median); stacking {len(chosen)} frames"
            f"{f' (sharpest {100 * keep:.0f}%)' if keep < 1 else ''}")
        chosen_set = set(chosen)

        # ---- Pass 1: register selected frames to the reference, build mean -------
        log(f"Pass 1/{n_passes}: registering frames to reference frame {ref_idx}"
            f"{' (global + local)' if local else ''}...")
        ref_gray = _gray(_get_frame(source, ref_idx))
        levels = max(1, int(math.log2(min(shape) / 90)) + 1)
        registrar = Registrar(ref_gray, motion, levels)
        win = cv2.createHanningWindow(shape[::-1], cv2.CV_32F)
        mean_acc = np.zeros(shape, np.float64)
        mean_cnt = np.zeros(shape, np.float64)
        aligns: dict[int, Alignment] = {}
        ccs: dict[int, float] = {}
        n_fail = 0
        for i, frame in enumerate(source()):
            if i not in chosen_set:
                continue
            prog("pass 1", i + 1, total_frames)
            g = _gray(frame)
            if i == ref_idx:
                al, cc = Alignment(np.eye(3)), 1.0
            else:
                # Direct phase correlation against the reference (no drift); the
                # chained scan trajectory (which drifts) is a sanity check/fallback.
                traj = scan.position[row_of[i]] - scan.position[ref_row]
                (dx, dy), resp = _phase_correlate(ref_gray, g, win)
                init = np.eye(3)
                direct_ok = resp > 0.03 and np.hypot(dx - traj[0], dy - traj[1]) < 0.15 * min(shape)
                init[:2, 2] = (dx, dy) if direct_ok else traj
                al, cc, _ = _align(registrar, g, init, local, tile, pool)
                if al is None:
                    n_fail += 1
                    continue
            aligns[i], ccs[i] = al, cc
            mean_acc += al.warp_to_ref(g, shape)
            mean_cnt += al.warp_to_ref(np.ones_like(g), shape)
        if verbose:
            sys.stderr.write("\n")
        if n_fail:
            log(f"  {n_fail} frames failed to register")

        # Reject frames whose correlation is anomalously low (blur, big glitches).
        cc_arr = np.array(list(ccs.values()))
        med = np.median(cc_arr)
        mad = 1.4826 * np.median(np.abs(cc_arr - med)) + 1e-4
        low = [k for k, v in ccs.items() if v < med - 5 * mad and v < 0.98 * med]
        for k in low:
            aligns.pop(k)
        if low:
            log(f"  {len(low)} frames rejected for low correlation (median cc={med:.4f})")

        ref_img = np.where(mean_cnt > 0.5, mean_acc / np.maximum(mean_cnt, 1e-9), registrar.ref_pyr[0])
        ref_img = ref_img.astype(np.float32)

        # ---- Pass 2: re-register against the low-noise reference + drizzle -------
        log(f"Pass 2/{n_passes}: re-registering against stacked reference and drizzling at {scale}x...")
        registrar = Registrar(ref_img, motion, registrar.levels)
        h, w_in = shape
        drop = pixfrac * scale
        bmask = _border_mask(shape, border)
        acc = wacc = acc_b = wacc_b = None
        base_acc = base_cnt = None
        tiles_used = []
        # Frames (except the reference) split in time into two independent halves.
        others = sorted(k for k in aligns if k != ref_idx)
        half_b = set(others[len(others) // 2 :])
        for i, frame in enumerate(source()):
            if i not in aligns:
                continue
            prog("pass 2", i + 1, total_frames)
            al, _, n_tiles = _align(registrar, _gray(frame), aligns[i].W, local, tile, pool, levels=2)
            if al is not None:
                aligns[i] = al
                tiles_used.append(n_tiles)
            al = aligns[i]
            f = _as_float(frame)
            if acc is None:
                acc = np.zeros((h * scale, w_in * scale, f.shape[2]), np.float32)
                wacc = np.zeros(acc.shape[:2], np.float32)
                acc_b, wacc_b = np.zeros_like(acc), np.zeros_like(wacc)
                base_acc = np.zeros(f.shape, np.float64)
                base_cnt = np.zeros(shape, np.float64)
            _drizzle_frame(*((acc_b, wacc_b) if i in half_b else (acc, wacc)), f, bmask, al, scale, drop)
            base_acc += al.warp_to_ref(f * bmask[..., None], shape).reshape(f.shape)
            base_cnt += al.warp_to_ref(bmask, shape)
        if verbose:
            sys.stderr.write("\n")
        if local and tiles_used:
            log(f"  local alignment: median {int(np.median(tiles_used))} reliable tiles per frame")
        model = _finalize(acc + acc_b, wacc + wacc_b, scale)

        # ---- Pass 3: outlier rejection by blotting the model back ----------------
        rejected_frac = 0.0
        if reject:
            log(f"Pass 3/3: rejecting outliers (> {reject_sigma} sigma) and re-drizzling...")
            blurred = cv2.GaussianBlur(model, (0, 0), max(0.3, scale / math.sqrt(12))).reshape(model.shape)
            acc2, wacc2 = np.zeros_like(acc), np.zeros_like(wacc)
            acc2_b, wacc2_b = np.zeros_like(acc), np.zeros_like(wacc)
            n_rej = n_pix = 0
            kernel = np.ones((3, 3), np.uint8)
            for i, frame in enumerate(source()):
                if i not in aligns:
                    continue
                prog("pass 3", i + 1, total_frames)
                al = aligns[i]
                f = _as_float(frame)
                bx, by = al.blot_map(shape, scale)
                blot = cv2.remap(blurred, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                blot = blot.reshape(f.shape)
                resid = np.abs(f - blot).max(axis=2)
                valid = (bmask > 0) & (bx >= 0) & (by >= 0) & (bx <= acc.shape[1] - 1) & (by <= acc.shape[0] - 1)
                if not valid.any():
                    continue
                sigma = 1.4826 * np.median(resid[valid]) + 1e-3
                # Registration/interpolation errors scale with the local gradient, so
                # allow more slack on edges (same idea as driz_cr's derivative term).
                gb = blot.mean(axis=2)
                deriv = cv2.dilate(gb, kernel) - cv2.erode(gb, kernel)
                bad = (resid > reject_sigma * sigma + 0.5 * deriv) & valid
                bad = cv2.dilate(bad.astype(np.uint8), kernel).astype(bool)
                n_rej += int(bad.sum())
                n_pix += int(valid.sum())
                _drizzle_frame(*((acc2_b, wacc2_b) if i in half_b else (acc2, wacc2)), f,
                               (bmask * ~bad).astype(np.float32), al, scale, drop)
            if verbose:
                sys.stderr.write("\n")
            rejected_frac = n_rej / max(n_pix, 1)
            log(f"  rejected {100 * rejected_frac:.3f}% of input pixels")
            acc, wacc, acc_b, wacc_b = acc2, wacc2, acc2_b, wacc2_b
            model = _finalize(acc + acc_b, wacc + wacc_b, scale)
    finally:
        pool.shutdown()

    # Noise of the full stack from the two half-stacks: for M = (wA*A + wB*B)/(wA+wB)
    # with independent noise, (A - B) * sqrt(wA*wB)/(wA+wB) has the same noise level.
    noise = None
    if wacc_b is not None and wacc_b.any():
        both = (wacc > 0) & (wacc_b > 0)
        da = np.divide(acc, wacc[..., None], out=np.zeros_like(acc), where=wacc[..., None] > 0)
        db = np.divide(acc_b, wacc_b[..., None], out=np.zeros_like(acc), where=wacc_b[..., None] > 0)
        k = np.where(both, np.sqrt(wacc * wacc_b) / np.maximum(wacc + wacc_b, 1e-9), 0)[..., None]
        noise = ((da - db) * k).astype(np.float32)
    wacc = wacc + wacc_b

    baseline = np.divide(
        base_acc, base_cnt[..., None], out=np.zeros_like(base_acc), where=base_cnt[..., None] > 0
    ).astype(np.float32)
    baseline = cv2.resize(baseline, (w_in * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    if baseline.ndim == 2:
        baseline = baseline[..., None]

    # ---- Diagnostics -------------------------------------------------------------
    warps = {k: a.W for k, a in aligns.items()}
    coverage, phases = _phase_coverage(warps, shape)
    inner = wacc[2 * scale * border : -2 * scale * border or None, 2 * scale * border : -2 * scale * border or None]
    weight_cv = float(inner.std() / max(inner.mean(), 1e-9))
    shifts = np.array([np.linalg.inv(wp)[:2, 2] for wp in warps.values()])
    diag = dict(
        frames_read=n_total,
        frames_used=len(warps),
        duplicates=n_dup,
        phase_coverage=coverage,
        phase_spread=phases.std(axis=0).tolist(),
        shift_range_px=(shifts.max(axis=0) - shifts.min(axis=0)).tolist(),
        weight_cv=weight_cv,
        rejected_fraction=rejected_frac,
        median_cc=float(med),
        reference=ref_idx,
        frames=sorted(aligns),
    )
    log(
        f"Used {len(warps)} frames; motion range dx={diag['shift_range_px'][0]:.2f}px "
        f"dy={diag['shift_range_px'][1]:.2f}px; sub-pixel phase coverage {100 * coverage:.0f}% "
        f"(4x4 cells); weight-map CV {weight_cv:.2f}"
    )
    if coverage < 0.5:
        log(
            "  WARNING: frames sample very few sub-pixel phases, so there is little aliased\n"
            "  detail to unfold. Expect mostly noise reduction, not new resolution. Nudge the\n"
            "  stage or let the camera drift slightly while recording to add dither."
        )
    return Result(image=model, baseline=baseline, weight=wacc, warps=warps, diagnostics=diag, noise=noise)


def unsharp(img: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return img + amount * (img - blur)


def save_image(path: str, img: np.ndarray) -> None:
    """Save a 0..255 float image. PNG/TIFF are written as 16-bit to keep the
    extra bit depth gained by stacking; other formats as 8-bit."""
    img = np.squeeze(img)
    if path.lower().endswith((".png", ".tif", ".tiff")):
        out = np.clip(img * 257.0 + 0.5, 0, 65535).astype(np.uint16)
    else:
        out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    if not cv2.imwrite(path, out):
        raise IOError(f"Could not write {path}")


if __name__ == "__main__":
    from microscope_movie import main

    main()
