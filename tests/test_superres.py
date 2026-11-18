import cv2
import numpy as np
import pytest

from microscope_movie import _drizzle, enhance

F = 4  # ground-truth oversampling relative to the simulated camera


def ground_truth(h=160, w=200, seed=0):
    """Sharp scene at F x the camera resolution: Siemens star, gratings, dots."""
    rng = np.random.default_rng(seed)
    H, W = h * F, w * F
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.full((H, W), 200, np.float32)
    cx, cy, r = W * 0.3, H * 0.5, H * 0.42
    ang = np.arctan2(yy - cy, xx - cx)
    star = (np.sin(36 * ang) > 0) & ((xx - cx) ** 2 + (yy - cy) ** 2 < r**2)
    img[star] = 40
    # gratings with periods just above/below the camera Nyquist limit (2 camera px)
    for k, period in enumerate([1.6, 2.2, 3.0]):
        x0 = int(W * 0.62)
        y0 = int(H * (0.08 + 0.3 * k))
        sl = np.s_[y0 : y0 + H // 5, x0 : x0 + W // 3]
        img[sl] = np.where(np.sin(2 * np.pi * xx[sl] / (period * F)) > 0, 60, 220)
    for _ in range(60):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 7)), 20, -1)
    return img


def make_frames(gt, n=40, noise=2.0, seed=1, dither=True):
    rng = np.random.default_rng(seed)
    H, W = gt.shape
    frames, shifts = [], []
    for i in range(n):
        if i == 0 or not dither:
            dx = dy = th = 0.0
        else:
            dx, dy = rng.uniform(-2 * F, 2 * F, 2)  # up to +-2 camera px
            th = rng.uniform(-0.3, 0.3)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), th, 1.0)
        M[:, 2] += (dx, dy)
        warped = cv2.warpAffine(gt, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        lo = cv2.resize(warped, (W // F, H // F), interpolation=cv2.INTER_AREA)  # pixel aperture
        lo = lo + rng.normal(0, noise, lo.shape)
        frames.append(np.clip(np.round(lo), 0, 255).astype(np.uint8))
        shifts.append((dx / F, dy / F))
    return frames, shifts


def psnr(a, b, crop=12):
    a = np.squeeze(a)[crop:-crop, crop:-crop].astype(np.float64)
    b = np.squeeze(b)[crop:-crop, crop:-crop].astype(np.float64)
    return 10 * np.log10(255**2 / np.mean((a - b) ** 2))


def test_kernel_identity():
    """scale 1, pixfrac 1, identity warp must reproduce the input exactly."""
    rng = np.random.default_rng(0)
    f = rng.uniform(0, 255, (30, 40, 3)).astype(np.float32)
    acc = np.zeros_like(f)
    w = np.zeros(f.shape[:2], np.float32)
    eye = np.eye(3).ravel().tolist()
    _drizzle.drizzle_add(acc, w, f, np.ones(f.shape[:2], np.float32), eye, eye, 1.0)
    np.testing.assert_allclose(acc / w[..., None], f, rtol=1e-5)
    np.testing.assert_allclose(w, 1.0, rtol=1e-6)


def test_kernel_flux_and_shift():
    """A half-pixel shift at scale 2 lands each drop exactly on one output pixel."""
    f = np.zeros((10, 10, 1), np.float32)
    f[5, 5] = 100
    s = 2
    S = np.array([[s, 0, 0.5], [0, s, 0.5], [0, 0, 1.0]])
    T = np.array([[1, 0, 0.25], [0, 1, 0.25], [0, 0, 1.0]])
    H = S @ T  # centre (5,5) -> 2 * 5.25 + 0.5 = 11
    acc = np.zeros((20, 20, 1), np.float32)
    w = np.zeros((20, 20), np.float32)
    _drizzle.drizzle_add(acc, w, f, np.ones((10, 10), np.float32), H.ravel().tolist(),
                         np.linalg.inv(H).ravel().tolist(), 1.0)
    assert acc[11, 11, 0] == pytest.approx(100)
    assert acc.sum() == pytest.approx(100)


@pytest.fixture(scope="module")
def synthetic():
    gt = ground_truth()
    frames, shifts = make_frames(gt)
    return gt, frames, shifts


def test_registration_accuracy(synthetic):
    gt, frames, shifts = synthetic
    res = enhance(lambda: iter(frames), reference=0, scale=2, reject=False, verbose=False)
    for i, (dx, dy) in enumerate(shifts):
        # warp maps reference -> frame; frame content moved by (dx, dy) camera px
        H, W = frames[0].shape
        c = np.array([W / 2 - 0.5 / F * 0, H / 2, 1.0])
        p = res.warps[i] @ c
        # the true mapping: frame(x) = scene(x - shift) about the same centre
        assert p[0] / p[2] - c[0] == pytest.approx(dx, abs=0.05)
        assert p[1] / p[2] - c[1] == pytest.approx(dy, abs=0.05)


def test_superres_beats_aligned_average(synthetic):
    gt, frames, _ = synthetic
    H, W = gt.shape
    truth2x = cv2.resize(gt, (W * 2 // F, H * 2 // F), interpolation=cv2.INTER_AREA)
    res = enhance(lambda: iter(frames), reference=0, scale=2, pixfrac=0.6, verbose=False)
    single = cv2.resize(frames[0].astype(np.float32), truth2x.shape[::-1], interpolation=cv2.INTER_CUBIC)

    p_single = psnr(single, truth2x)
    p_base = psnr(res.baseline, truth2x)
    p_sr = psnr(res.image, truth2x)
    print(f"PSNR single={p_single:.2f} aligned-average={p_base:.2f} drizzle={p_sr:.2f}")
    assert res.diagnostics["phase_coverage"] > 0.8
    # Averaging aliased frames does not remove aliasing (it can even lose to a
    # single frame); unfolding it onto the finer grid must win clearly.
    assert p_sr > p_base + 1.0
    assert p_sr > p_single + 1.0


def test_no_dither_is_reported():
    gt = ground_truth()
    frames, _ = make_frames(gt, n=10, dither=False, seed=3)
    res = enhance(lambda: iter(frames), reference=0, scale=2, verbose=False)
    assert res.diagnostics["phase_coverage"] <= 1 / 16


def test_map_kernel_matches_homography_kernel():
    """For an affine warp the dense-map kernel must equal the exact homography kernel."""
    rng = np.random.default_rng(5)
    f = rng.uniform(0, 255, (40, 50, 3)).astype(np.float32)
    wt = np.ones(f.shape[:2], np.float32)
    s = 2
    A = np.array([[0.999, -0.02, 1.3], [0.02, 0.999, -0.7], [0, 0, 1.0]])  # frame -> ref
    H = np.array([[s, 0, 0.5], [0, s, 0.5], [0, 0, 1.0]]) @ A  # frame -> output
    Hi = np.linalg.inv(H)
    out = (80, 100)
    a1 = np.zeros(out + (3,), np.float32); w1 = np.zeros(out, np.float32)
    _drizzle.drizzle_add(a1, w1, f, wt, H.ravel().tolist(), Hi.ravel().tolist(), 1.4)
    v, u = np.mgrid[0:out[0], 0:out[1]].astype(np.float64)
    den = Hi[2, 0] * u + Hi[2, 1] * v + Hi[2, 2]
    m = np.stack([(Hi[0, 0] * u + Hi[0, 1] * v + Hi[0, 2]) / den,
                  (Hi[1, 0] * u + Hi[1, 1] * v + Hi[1, 2]) / den], axis=-1).astype(np.float32)
    a2 = np.zeros_like(a1); w2 = np.zeros_like(w1)
    _drizzle.drizzle_add_map(a2, w2, f, wt, m, 1.4)
    np.testing.assert_allclose(w2, w1, atol=1e-3)
    np.testing.assert_allclose(a2, a1, atol=0.3)


def test_local_alignment_fixes_nonrigid_motion():
    """Rolling-shutter-like skew (shift varies with row) defeats a global model;
    the local field must recover it."""
    gt = ground_truth(h=200, w=240, seed=2)
    H, W = gt.shape
    rng = np.random.default_rng(4)
    frames = []
    for i in range(24):
        dx, dy = (0.0, 0.0) if i == 0 else rng.uniform(-2 * F, 2 * F, 2)
        skew = 0.0 if i == 0 else rng.uniform(-0.03, 0.03)  # up to +-1.5 camera px top-to-bottom
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        mx = xx - dx - skew * (yy - H / 2) * (yy / H)  # quadratic in y: not affine
        my = yy - dy
        warped = cv2.remap(gt, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        lo = cv2.resize(warped, (W // F, H // F), interpolation=cv2.INTER_AREA) + rng.normal(0, 2, (H // F, W // F))
        frames.append(np.clip(np.round(lo), 0, 255).astype(np.uint8))
    truth2x = cv2.resize(gt, (W * 2 // F, H * 2 // F), interpolation=cv2.INTER_AREA)
    rigid = enhance(lambda: iter(frames), reference=0, scale=2, local=False, verbose=False)
    loc = enhance(lambda: iter(frames), reference=0, scale=2, local=True, tile=48, verbose=False)
    p_rigid, p_local = psnr(rigid.image, truth2x), psnr(loc.image, truth2x)
    print(f"PSNR rigid={p_rigid:.2f} local={p_local:.2f}")
    assert p_local > p_rigid + 0.5


def test_phase_correlate_wrapper_does_not_mutate():
    """OpenCV 5.0.0's phaseCorrelate multiplies the window into its inputs in place."""
    from microscope_movie.superres import _phase_correlate

    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.random((240, 320)).astype(np.float32), (0, 0), 2)
    b = np.roll(a, (0, 3), axis=(0, 1))
    a0, b0 = a.copy(), b.copy()
    win = cv2.createHanningWindow((320, 240), cv2.CV_32F)
    for _ in range(3):
        (dx, dy), _ = _phase_correlate(a, b, win)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)
    assert dx == pytest.approx(3, abs=0.2)


def test_frame_selection_keeps_sharpest():
    from microscope_movie.superres import scan_frames, select_frames

    gt = ground_truth()
    frames, _ = make_frames(gt, n=12)
    for k in (3, 5, 7):  # blur some frames
        frames[k] = cv2.GaussianBlur(frames[k], (0, 0), 1.5)
    scan = scan_frames(lambda: iter(frames))
    ref, chosen = select_frames(scan, keep=0.7, reference=0)
    assert ref == 0 and 0 in chosen
    assert not {3, 5, 7} & set(chosen)
    ref_auto, _ = select_frames(scan)
    assert ref_auto not in (3, 5, 7)


def test_deconvolution_recovers_optical_blur():
    """Optically blurred scene: Wiener deconvolution with the noise measured from
    the two half-stacks must get clearly closer to the sharp ground truth."""
    from microscope_movie.deconv import wiener_deconvolve

    gt = ground_truth(h=200, w=240, seed=6)
    frames, _ = make_frames(cv2.GaussianBlur(gt, (0, 0), 0.8 * F), n=40, seed=7)
    H, W = gt.shape
    truth2x = cv2.resize(gt, (W * 2 // F, H * 2 // F), interpolation=cv2.INTER_AREA)
    res = enhance(lambda: iter(frames), reference=0, verbose=False)
    assert res.noise is not None and res.noise.shape == res.image.shape
    out, info = wiener_deconvolve(res.image, res.noise)
    p0, p1 = psnr(res.image, truth2x), psnr(out, truth2x)
    print(f"PSNR drizzle={p0:.2f} deconvolved={p1:.2f} sigma={info['sigma']:.2f}")
    assert 1.0 < info["sigma"] < 3.0
    assert p1 > p0 + 1.5
