"""Microscope movie enhancement: multi-frame super-resolution (Drizzle)."""

import argparse
import os

import cv2

from .deconv import wiener_deconvolve
from .superres import enhance, save_image, unsharp, video_source

__all__ = ["enhance", "video_source", "save_image", "wiener_deconvolve", "main"]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fuse all frames of a microscope video into one super-resolved image "
        "(sub-pixel registration + Drizzle)."
    )
    p.add_argument("input_video", help="Input video file (e.g. .mov)")
    p.add_argument("output_image", nargs="?",
                   help="Output image; .png/.tif are saved 16-bit, .jpg 8-bit "
                   "(default: <input>_superres.png)")
    p.add_argument("-o", "--output", dest="output_opt", metavar="PATH",
                   help="Output image (alternative to the positional argument)")
    p.add_argument("--scale", type=int, default=2, help="Upsampling factor (default 2)")
    p.add_argument(
        "--pixfrac", type=float, default=0.7,
        help="Drop size as fraction of an input pixel, (0,1]. Smaller = sharper, needs more dither (default 0.7)",
    )
    p.add_argument(
        "--motion", choices=["translation", "euclidean", "affine", "homography"], default="euclidean",
        help="Registration motion model (default euclidean)",
    )
    p.add_argument("--local", action=argparse.BooleanOptionalAction, default=True,
                   help="Refine alignment with a per-tile displacement field (default on; "
                   "--no-local for rigid-only, faster)")
    p.add_argument("--tile", type=int, default=96, help="Tile size for local alignment (default 96)")
    p.add_argument("--keep", type=float, default=1.0,
                   help="Fraction of frames to stack, sharpest first, e.g. 0.6 (default 1 = all)")
    p.add_argument("--reference", type=int, default=None,
                   help="Frame number that defines the output view (default: automatic, "
                   "a sharp frame shared by many others)")
    p.add_argument("--min-overlap", type=float, default=0.5,
                   help="Skip frames overlapping the reference view less than this (default 0.5)")
    p.add_argument("--reject", action=argparse.BooleanOptionalAction, default=True,
                   help="Outlier rejection pass (default on)")
    p.add_argument("--reject-sigma", type=float, default=4.0, help="Outlier threshold in sigma (default 4)")
    p.add_argument("--deconvolve", action=argparse.BooleanOptionalAction, default=True,
                   help="Wiener deconvolution using the noise measured from two independent "
                   "half-stacks (default on)")
    p.add_argument("--psf-sigma", type=float, default=None,
                   help="Gaussian blur sigma in OUTPUT pixels for deconvolution (default: fitted)")
    p.add_argument("--deconv-strength", type=float, default=1.0,
                   help="0 = none .. 1 = full Wiener filter (default 1)")
    p.add_argument("--sharpen", type=float, default=None,
                   help="Unsharp-mask amount, 0 to disable (default 0 with deconvolution, else 0.5)")
    p.add_argument("--save-raw", metavar="PATH",
                   help="Also save the drizzled image before deconvolution/sharpening")
    p.add_argument("--max-frames", type=int, default=None, help="Only use the first N frames")
    p.add_argument("--baseline", metavar="PATH",
                   help="Also save the aligned plain average, bicubic-upscaled, for comparison")
    args = p.parse_args()
    if args.output_image and args.output_opt:
        p.error("give the output either positionally or with -o, not both")
    args.output_image = (
        args.output_opt or args.output_image or os.path.splitext(args.input_video)[0] + "_superres.png"
    )

    cap = cv2.VideoCapture(args.input_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()
    if args.max_frames:
        total = min(total or args.max_frames, args.max_frames)

    res = enhance(
        video_source(args.input_video, args.max_frames),
        scale=args.scale,
        pixfrac=args.pixfrac,
        motion=args.motion,
        local=args.local,
        tile=args.tile,
        keep=args.keep,
        reference=args.reference,
        min_overlap=args.min_overlap,
        reject=args.reject,
        reject_sigma=args.reject_sigma,
        total_frames=total,
    )
    sharpen = args.sharpen if args.sharpen is not None else (0.0 if args.deconvolve else 0.5)
    sigma = max(1.0, 0.75 * args.scale)
    final = res.image
    if args.deconvolve and res.noise is not None:
        final, info = wiener_deconvolve(res.image, res.noise, sigma=args.psf_sigma,
                                        strength=args.deconv_strength)
        print(f"Deconvolution: PSF sigma {info['sigma']:.2f} output px "
              f"({'fitted' if args.psf_sigma is None else f'given; fit was {info['sigma_fitted']:.2f}'}), "
              f"SNR > 1 up to {info['snr1_freq']:.3f} cycles/output px, peak gain {info['peak_gain']:.1f}")
    elif args.deconvolve:
        print("Deconvolution skipped: needs at least 3 frames to measure the noise")
    if args.save_raw:
        save_image(args.save_raw, res.image)
        print(f"Saved drizzled image before deconvolution to: {args.save_raw}")
    save_image(args.output_image, unsharp(final, sharpen, sigma))
    print(f"Saved super-resolved image ({res.image.shape[1]}x{res.image.shape[0]}) to: {args.output_image}")
    if args.baseline:
        save_image(args.baseline, unsharp(res.baseline, sharpen, sigma))
        print(f"Saved baseline (aligned average, bicubic) to: {args.baseline}")

