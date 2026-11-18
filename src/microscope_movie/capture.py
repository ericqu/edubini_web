"""Record frames straight from a USB microscope (macOS / AVFoundation).

QuickTime re-encodes the camera stream to H.264 (lossy, and it copies blocks
between frames, so their noise is no longer independent). This module talks
to the camera directly: it selects the exact native video mode, receives the
frames the camera delivers (with timestamps), and writes them losslessly:

    microscope-capture list                       # cameras and their modes
    microscope-capture probe                      # which mode has the most real detail?
    microscope-capture photo leaf.png             # a still (optionally a stacked burst)
    microscope-capture record leaf.mkv -s 20      # lossless FFV1 video for microscope-movie

The camera is chosen by name (default "HD video", the Edubini microscope).
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import AVFoundation as AVF
    import CoreMedia as CM
    import libdispatch
    import objc
    import Quartz
    from Foundation import NSObject
except ImportError:  # pragma: no cover - non-macOS
    AVF = None

DEFAULT_CAMERA = "HD video"


# --------------------------------------------------------------------------- #
# AVFoundation camera
# --------------------------------------------------------------------------- #


def _fourcc(code: int) -> str:
    return code.to_bytes(4, "big").decode(errors="replace")


def list_cameras():
    """[(device, [(width, height, pixfmt, max_fps, format_obj), ...]), ...]"""
    if AVF is None:
        raise RuntimeError("camera capture needs macOS with pyobjc-framework-AVFoundation")
    names = ["AVCaptureDeviceTypeExternal", "AVCaptureDeviceTypeExternalUnknown",
             "AVCaptureDeviceTypeBuiltInWideAngleCamera", "AVCaptureDeviceTypeContinuityCamera"]
    types = [getattr(AVF, n) for n in names if hasattr(AVF, n)]
    sess = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types, AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified)
    out = []
    for dev in sess.devices():
        modes = []
        for f in dev.formats():
            desc = f.formatDescription()
            d = CM.CMVideoFormatDescriptionGetDimensions(desc)
            fps = max((r.maxFrameRate() for r in f.videoSupportedFrameRateRanges()), default=0)
            modes.append((d.width, d.height, _fourcc(CM.CMFormatDescriptionGetMediaSubType(desc)), fps, f))
        out.append((dev, modes))
    return out


def find_camera(name: str = DEFAULT_CAMERA):
    for dev, modes in list_cameras():
        if name.lower() in dev.localizedName().lower():
            return dev, modes
    found = ", ".join(repr(d.localizedName().strip()) for d, _ in list_cameras())
    raise RuntimeError(f"no camera matching {name!r} (found: {found}). Is the microscope plugged in?")


def ensure_access(timeout: float = 60.0) -> None:
    status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    if status == 3:  # authorized
        return
    if status in (1, 2):
        raise PermissionError(
            "Camera access is denied for this app. Enable it in System Settings > "
            "Privacy & Security > Camera (for VS Code / Terminal), then restart that app.")
    done = threading.Event()
    result = {}

    def handler(granted):
        result["ok"] = bool(granted)
        done.set()

    print("Requesting camera access: please click 'Allow' in the macOS dialog...", file=sys.stderr)
    AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVF.AVMediaTypeVideo, handler)
    if not done.wait(timeout) or not result.get("ok"):
        raise PermissionError("Camera access was not granted.")


if AVF is not None:

    class _FrameDelegate(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
        def initWithSink_(self, sink):
            self = objc.super(_FrameDelegate, self).init()
            if self is None:
                return None
            self.sink = sink
            self.dropped = 0
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sbuf, connection):
            img = CM.CMSampleBufferGetImageBuffer(sbuf)
            if img is None:
                return
            Quartz.CVPixelBufferLockBaseAddress(img, 1)  # read-only
            try:
                h = Quartz.CVPixelBufferGetHeight(img)
                w = Quartz.CVPixelBufferGetWidth(img)
                bpr = Quartz.CVPixelBufferGetBytesPerRow(img)
                base = Quartz.CVPixelBufferGetBaseAddress(img)
                arr = np.frombuffer(base.as_buffer(bpr * h), np.uint8).reshape(h, bpr)
                frame = arr[:, : w * 4].reshape(h, w, 4)[..., :3].copy()  # BGRA -> BGR
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(img, 1)
            t = CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sbuf))
            self.sink(t, frame)

        def captureOutput_didDropSampleBuffer_fromConnection_(self, output, sbuf, connection):
            self.dropped += 1


class Camera:
    """Context manager delivering (timestamp, BGR uint8 frame) at a native mode."""

    def __init__(self, name: str = DEFAULT_CAMERA, size: tuple[int, int] | None = None, fps: float = 30):
        ensure_access()
        self.device, modes = find_camera(name)
        if size is None:  # largest mode
            mode = max(modes, key=lambda m: m[0] * m[1])
        else:
            matches = [m for m in modes if (m[0], m[1]) == tuple(size)]
            if not matches:
                avail = ", ".join(f"{m[0]}x{m[1]}" for m in modes)
                raise ValueError(f"mode {size[0]}x{size[1]} not supported (available: {avail})")
            mode = matches[0]
        self.size = (mode[0], mode[1])
        self.format = mode[4]
        self.fps = min(fps, mode[3])
        self.queue: queue.Queue = queue.Queue(maxsize=600)
        self.lost = 0

    def _sink(self, t, frame):
        try:
            self.queue.put_nowait((t, frame))
        except queue.Full:
            self.lost += 1

    def __enter__(self):
        self.session = AVF.AVCaptureSession.alloc().init()
        inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(self.device, None)
        if inp is None:
            raise RuntimeError(f"cannot open camera: {err}")
        self.delegate = _FrameDelegate.alloc().initWithSink_(self._sink)
        self.dq = libdispatch.dispatch_queue_create(b"microscope.capture", None)
        out = AVF.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        out.setSampleBufferDelegate_queue_(self.delegate, self.dq)
        self.session.addInput_(inp)
        self.session.addOutput_(out)
        # This camera only keeps a format set *after* the session is running;
        # set before, the session preset silently resets it to 1920x1080.
        self.session.startRunning()
        ok, err = self.device.lockForConfiguration_(None)
        if not ok:
            raise RuntimeError(f"cannot configure camera: {err}")
        try:
            self.device.setActiveFormat_(self.format)
        finally:
            self.device.unlockForConfiguration()
        return self

    def __exit__(self, *exc):
        self.session.stopRunning()

    def frames(self, seconds: float | None = None, warmup: float = 0.0):
        """Yield (t, frame). Skips `warmup` seconds (auto exposure settling)."""
        t_start = time.monotonic()
        while True:
            try:
                t, f = self.queue.get(timeout=3)
            except queue.Empty:
                raise RuntimeError("no frames from camera for 3 s")
            if (f.shape[1], f.shape[0]) != self.size:  # frames from before the mode switch
                if time.monotonic() - t_start > 5:
                    raise RuntimeError(f"camera delivers {f.shape[1]}x{f.shape[0]}, not {self.size[0]}x{self.size[1]}")
                continue
            el = time.monotonic() - t_start
            if el < warmup:
                continue
            yield t, f
            if seconds is not None and el >= warmup + seconds:
                return


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


def _gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


def unique_frames(frames, threshold: float = 0.3):
    """Drop repeated frames (mean abs difference < threshold levels)."""
    out = []
    for t, f in frames:
        if not out or np.mean(np.abs(_gray(f) - _gray(out[-1][1]))) >= threshold:
            out.append((t, f))
    return out


def detail_snr(frames, nbins: int = 20, region: float = 0.5):
    """Per-frequency (cycles/px) power ratio frame / temporal noise, for a still
    scene. Noise from differences of consecutive distinct frames, averaged."""
    g = [_gray(f) for _, f in frames]
    h, w = g[0].shape
    n = int(min(h, w) * region) // 2 * 2
    y0, x0 = (h - n) // 2, (w - n) // 2
    win = np.outer(np.hanning(n), np.hanning(n))
    yy, xx = np.indices((n, n))
    r = np.hypot(xx - n // 2, yy - n // 2) / n
    edges = np.linspace(0, 0.5, nbins + 1)

    def radial(img):
        c = img[y0 : y0 + n, x0 : x0 + n]
        p = np.abs(np.fft.fftshift(np.fft.fft2((c - c.mean()) * win))) ** 2
        return np.array([p[(r >= edges[k]) & (r < edges[k + 1])].mean() for k in range(nbins)])

    ps = np.mean([radial(x) for x in g], axis=0)
    pn = np.mean([radial((a - b) / np.sqrt(2)) for a, b in zip(g[:-1], g[1:])], axis=0)
    return (edges[:-1] + edges[1:]) / 2, ps / np.maximum(pn, 1e-12)


def detail_limit(freqs, ratio, threshold: float = 2.0) -> float:
    """Frequency where frame power falls to `threshold` x noise power (per-frame
    SNR = 1), linearly interpolated; scanning down from low frequencies."""
    below = np.nonzero(ratio < threshold)[0]
    if len(below) == 0:
        return float(freqs[-1])
    k = below[0]
    if k == 0:
        return 0.0
    f0, f1, r0, r1 = freqs[k - 1], freqs[k], np.log(ratio[k - 1]), np.log(ratio[k])
    return float(f0 + (r0 - np.log(threshold)) / (r0 - r1) * (f1 - f0))


def relative_scale(img, ref) -> tuple[float, float] | None:
    """How many `ref` pixels one `img` pixel spans, for two views of the same
    (centred, unrotated) scene from different camera modes: multi-scale template
    search of img's centre in ref. Returns (scale, correlation)."""
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a, b = g(img), g(ref)

    def score(s):
        r = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        n = int(min(r.shape[0], r.shape[1], b.shape[0], b.shape[1]) * 0.45)
        if n < 32:
            return -1.0
        cy, cx = r.shape[0] // 2, r.shape[1] // 2
        tmpl = r[cy - n // 2 : cy + n // 2, cx - n // 2 : cx + n // 2]
        return float(cv2.minMaxLoc(cv2.matchTemplate(b, tmpl, cv2.TM_CCOEFF_NORMED))[1])

    coarse = np.exp(np.linspace(np.log(0.4), np.log(2.5), 60))
    s0 = max(coarse, key=score)
    fine = np.linspace(s0 * 0.96, s0 * 1.04, 17)
    best = max(fine, key=score)
    c = score(best)
    return (float(best), c) if c > 0.5 else None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_list(args):
    for dev, modes in list_cameras():
        print(f"== {dev.localizedName().strip()}  ({dev.modelID()})")
        for w, h, pf, fps, _ in modes:
            print(f"   {w}x{h}  {pf!r}  up to {fps:.0f} fps")


def cmd_probe(args):
    _, modes = find_camera(args.camera)
    sizes = sorted({(m[0], m[1]) for m in modes}, key=lambda s: s[0] * s[1])
    os.makedirs(args.out_dir, exist_ok=True)
    print("Keep the microscope still and focused on a textured area.\n")
    rows = []
    stills = {}
    for size in sizes:
        try:
            with Camera(args.camera, size) as cam:
                got = list(cam.frames(seconds=args.seconds, warmup=1.5))
        except RuntimeError as e:
            print(f"  {size[0]}x{size[1]}: unusable ({e})")
            continue
        uniq = unique_frames(got)
        dt = got[-1][0] - got[0][0] if len(got) > 1 else 1
        freqs, ratio = detail_snr(uniq)
        lim = detail_limit(freqs, ratio)
        path = os.path.join(args.out_dir, f"mode_{size[0]}x{size[1]}.png")
        cv2.imwrite(path, uniq[-1][1])
        stills[size] = uniq[-1][1]
        rows.append(dict(size=size, fps=(len(got) - 1) / dt, ufps=(len(uniq) - 1) / dt, ratio=ratio, limit=lim))
        print(f"  {size[0]}x{size[1]}: {(len(got) - 1) / dt:5.1f} fps delivered, "
              f"{(len(uniq) - 1) / dt:5.1f} distinct; detail up to {lim:.3f} cycles/px")
    if not rows:
        raise RuntimeError("no usable mode")
    ref_size = max((r["size"] for r in rows), key=lambda s: s[0] * s[1])
    for r in rows:
        m = (1.0, 1.0) if r["size"] == ref_size else relative_scale(stills[r["size"]], stills[ref_size])
        r["scale"] = m[0] if m else None
    ref_name = f"{ref_size[0]}x{ref_size[1]}"
    print(f"\nDetail compared in common units: pixels of the {ref_name} mode (scale from template matching)")
    print(f"{'mode':>10} {'fps':>5} {'px scale':>8} {'detail c/px':>11} {'finest period':>13} {'= ' + ref_name + ' px':>14}")
    for r in rows:
        w, h = r["size"]
        s = r["scale"]
        period = 1 / r["limit"] if r["limit"] else float("inf")
        common = period * s if s else float("nan")
        r["common"] = common
        print(f"{w:>5}x{h:<4} {r['ufps']:5.1f} {s if s else float('nan'):8.2f} {r['limit']:11.3f} "
              f"{period:11.1f}px {common:12.1f}px")
    print("\nS/N power per frequency band (cycles/px of each mode):")
    print("  " + " " * 10 + " ".join(f"{f:5.3f}" for f in freqs))
    for r in rows:
        print(f"  {r['size'][0]}x{r['size'][1]:<5}" + " ".join(f"{v:5.1f}" for v in r["ratio"]))
    known = [r for r in rows if r["scale"]]
    best = min(known, key=lambda r: r["common"])
    print(f"\nFinest real detail: {best['size'][0]}x{best['size'][1]} (period {best['common']:.1f} {ref_name} px, "
          f"{best['ufps']:.0f} fps). Stills saved in {args.out_dir}/")


def _parse_size(s):
    if s is None:
        return None
    w, h = s.lower().split("x")
    return int(w), int(h)


def cmd_photo(args):
    with Camera(args.camera, _parse_size(args.mode)) as cam:
        got = [f for _, f in unique_frames(cam.frames(seconds=args.burst / max(cam.fps, 1) + 0.5, warmup=1.5))]
    got = got[-args.burst :]
    if len(got) == 1:
        img = got[0]
    else:  # align by phase correlation and average: a low-noise still
        ref = _gray(got[0])
        acc = np.zeros(got[0].shape, np.float64)
        for f in got:
            (dx, dy), _ = cv2.phaseCorrelate(ref.copy(), _gray(f))
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            acc += cv2.warpAffine(f.astype(np.float32), m, f.shape[1::-1],
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        img = acc / len(got)
    if args.output.lower().endswith((".png", ".tif", ".tiff")) and len(got) > 1:
        out = np.clip(img * 257 + 0.5, 0, 65535).astype(np.uint16)  # keep stacked bit depth
    else:
        out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(args.output, out)
    print(f"Saved {img.shape[1]}x{img.shape[0]} still ({len(got)} frame{'s' if len(got) > 1 else ''}) to {args.output}")


def cmd_record(args):
    size = _parse_size(args.mode)
    stop = threading.Event()
    with Camera(args.camera, size) as cam:
        w, h = cam.size
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"FFV1"), cam.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("cannot open FFV1 writer (use a .mkv or .avi output)")
        q: queue.Queue = queue.Queue(maxsize=300)

        def write():
            while True:
                f = q.get()
                if f is None:
                    return
                writer.write(f)

        th = threading.Thread(target=write, daemon=True)
        th.start()
        n = 0
        t0 = time.monotonic()
        stamps = []
        print(f"Recording {w}x{h} to {args.output} "
              + (f"for {args.seconds:.0f} s" if args.seconds else "(Ctrl+C to stop)") + "...")
        try:
            for t, f in cam.frames(seconds=args.seconds, warmup=1.0):
                q.put(f)
                n += 1
                stamps.append(t)
                if args.preview:
                    cv2.imshow("microscope (q to stop)", f)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                sys.stderr.write(f"\r  {n} frames, {time.monotonic() - t0:5.1f} s")
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            q.put(None)
            th.join()
            writer.release()
            if args.preview:
                cv2.destroyAllWindows()
    fps = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0.0
    print(f"\nSaved {n} frames ({fps:.1f} fps, camera timestamps) losslessly to {args.output}")
    if cam.lost or cam.delegate.dropped:
        print(f"  warning: {cam.lost + cam.delegate.dropped} frames dropped (disk or CPU too slow)")


def main() -> None:
    p = argparse.ArgumentParser(description="Capture frames directly from the USB microscope (lossless).")
    p.add_argument("--camera", default=DEFAULT_CAMERA, help=f"camera name substring (default {DEFAULT_CAMERA!r})")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list cameras and native modes")
    pp = sub.add_parser("probe", help="measure real detail and frame rate of every mode")
    pp.add_argument("--seconds", type=float, default=2.5)
    pp.add_argument("--out-dir", default="results/probe")
    ph = sub.add_parser("photo", help="save a still image")
    ph.add_argument("output")
    ph.add_argument("--mode", help="WxH native mode (default: largest)")
    ph.add_argument("--burst", type=int, default=1, help="average this many frames (aligned) into one low-noise still")
    pr = sub.add_parser("record", help="record a lossless FFV1 video (.mkv)")
    pr.add_argument("output")
    pr.add_argument("--mode", help="WxH native mode (default: largest)")
    pr.add_argument("-s", "--seconds", type=float, default=None, help="duration (default: until Ctrl+C)")
    pr.add_argument("--preview", action="store_true", help="show a live preview window (q to stop)")
    args = p.parse_args()
    {"list": cmd_list, "probe": cmd_probe, "photo": cmd_photo, "record": cmd_record}[args.cmd](args)


if __name__ == "__main__":
    main()
