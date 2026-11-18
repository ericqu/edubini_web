"""Check the Rust/WebAssembly engine against the Python pipeline.

    uv run python web/engine/compare.py            # synthetic scenes (ground truth)
    uv run python web/engine/compare.py --video leaf_direct.mkv --frames 70

Synthetic: the scenes of tests/test_superres.py; prints PSNR against the ground
truth for Python and Rust. Video: PSNR of Rust vs Python on real frames, and
crops of both side by side in results/compare/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(ROOT, "web", "engine")
BIN = os.path.join(ENGINE, "target", "release", "sr_raw")
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from microscope_movie import enhance, wiener_deconvolve  # noqa: E402


def to_rgb(f):
    return np.repeat(f[..., None], 3, axis=2) if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def run_rust(frames_bgr, tmp, extra=()):
    h, w = frames_bgr[0].shape[:2]
    inp, out = os.path.join(tmp, "in.raw"), os.path.join(tmp, "out.raw")
    with open(inp, "wb") as fh:
        fh.write(b"MMF1" + np.array([w, h, len(frames_bgr)], "<u4").tobytes())
        for f in frames_bgr:
            fh.write(np.ascontiguousarray(to_rgb(f), np.uint8).tobytes())
    t0 = time.time()
    p = subprocess.run([BIN, inp, out, *extra], capture_output=True, text=True, check=True)
    dt = time.time() - t0
    with open(out, "rb") as fh:
        buf = fh.read()
    assert buf[:4] == b"MMO1"
    ow, oh = np.frombuffer(buf[4:12], "<u4")
    n = int(ow) * int(oh) * 3
    a = np.frombuffer(buf[12:], "<f4")
    final = a[:n].reshape(oh, ow, 3)
    raw = a[n : 2 * n].reshape(oh, ow, 3)
    return final, raw, json.loads(p.stdout), p.stderr.strip(), dt


def gray_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def psnr(a, b, crop=12):
    a = np.squeeze(a)[crop:-crop, crop:-crop].astype(np.float64)
    b = np.squeeze(b)[crop:-crop, crop:-crop].astype(np.float64)
    return 10 * np.log10(255**2 / np.mean((a - b) ** 2))


def synthetic(tmp):
    from test_superres import F, ground_truth, make_frames

    cases = []
    gt = ground_truth()
    frames, _ = make_frames(gt)
    cases.append(("aliased scene, 40 frames", gt, frames))
    gt2 = ground_truth(h=200, w=240, seed=6)
    frames2, _ = make_frames(cv2.GaussianBlur(gt2, (0, 0), 0.8 * F), n=40, seed=7)
    cases.append(("optically blurred scene", gt2, frames2))
    for name, gt, frames in cases:
        H, W = gt.shape
        truth = cv2.resize(gt, (W * 2 // F, H * 2 // F), interpolation=cv2.INTER_AREA)
        t0 = time.time()
        res = enhance(lambda: iter(frames), reference=0, scale=2, verbose=False)
        py_dec, info = wiener_deconvolve(res.image, res.noise)
        t_py = time.time() - t0
        final, raw, diag, timing, t_rs = run_rust(frames, tmp, ["--reference", "0"])
        print(f"\n{name}:  python {t_py:.1f}s, rust {t_rs:.1f}s ({timing})")
        print(f"  drizzle     PSNR  python {psnr(res.image, truth):6.2f}   rust {psnr(gray_of(raw), truth):6.2f}")
        print(f"  deconvolved PSNR  python {psnr(py_dec, truth):6.2f}   rust {psnr(gray_of(final), truth):6.2f}")
        print(f"  psf sigma         python {info['sigma']:6.2f}   rust {diag['deconv']['sigma']:6.2f}")
        print(f"  phase coverage    python {res.diagnostics['phase_coverage']:6.2f}   rust {diag['phase_coverage']:6.2f}")
        print(f"  rust vs python (deconvolved) PSNR {psnr(gray_of(final), py_dec):.2f}")


def video(path, n, tmp, extra=()):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    print(f"{path}: {len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]}")
    t0 = time.time()
    res = enhance(lambda: iter(frames), scale=2, verbose=False)
    py_dec, info = wiener_deconvolve(res.image, res.noise)
    t_py = time.time() - t0
    final, raw, diag, timing, t_rs = run_rust(frames, tmp, extra)
    py_rgb = cv2.cvtColor(np.clip(py_dec, 0, 255).astype(np.float32), cv2.COLOR_BGR2RGB)
    py_raw = cv2.cvtColor(res.image.astype(np.float32), cv2.COLOR_BGR2RGB)
    print(f"  python {t_py:.1f}s   rust {t_rs:.1f}s ({timing})")
    print(f"  python: used {res.diagnostics['frames_used']}, coverage {res.diagnostics['phase_coverage']:.2f}, "
          f"reference {res.diagnostics['reference']}, sigma {info['sigma']:.2f}")
    print(f"  rust:   {json.dumps(diag)}")
    print(f"  rust vs python PSNR: drizzle {psnr(raw, py_raw):.2f} dB, deconvolved {psnr(np.clip(final, 0, 255), py_rgb):.2f} dB")
    out_dir = os.path.join(ROOT, "results", "compare")
    os.makedirs(out_dir, exist_ok=True)
    h, w = final.shape[:2]
    y, x, s = h // 2 - 200, w // 2 - 300, 400
    crop = lambda im: np.clip(im[y : y + s, x : x + 600], 0, 255).astype(np.uint8)
    single = cv2.resize(to_rgb(frames[diag["reference"]]), (w, h), interpolation=cv2.INTER_CUBIC)
    strip = np.concatenate([crop(single), crop(py_rgb), crop(final)], axis=1)
    cv2.imwrite(os.path.join(out_dir, "single_python_rust.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"  crops (single frame | python | rust): {out_dir}/single_python_rust.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video")
    p.add_argument("--frames", type=int, default=70)
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--rust-args", default="", help="extra sr_raw options, e.g. --no-local-pass1")
    a = p.parse_args()
    if not a.no_build:
        subprocess.run(["cargo", "build", "--release", "--bin", "sr_raw"], cwd=ENGINE, check=True)
    tmp = os.path.join(ROOT, "results", "compare")
    os.makedirs(tmp, exist_ok=True)
    if a.video:
        video(a.video, a.frames, tmp, a.rust_args.split())
    else:
        synthetic(tmp)


if __name__ == "__main__":
    main()
