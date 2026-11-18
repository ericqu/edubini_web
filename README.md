# microscope-movie

Fuse the frames of a microscope video into one image using multi-frame
super-resolution (sub-pixel registration + Drizzle).

```sh
uv sync                     # builds the Rust Drizzle kernel (needs cargo)
uv run microscope-movie input.mov -o out.png --baseline avg.png
```

How it works:

1. Drop duplicated frames (phone encoders often repeat frames).
2. Register every frame with coarse-to-fine ECC (OpenCV), then re-register
   against the stacked average for better sub-pixel accuracy.
3. Drizzle each frame onto a `--scale`× finer grid (`rust/lib.rs`): each input
   pixel becomes a drop of `--pixfrac` × its size, added with exact overlap
   weights. Frames with different sub-pixel offsets fill in different fine-grid
   positions, which recovers detail that is aliased in each single frame.
4. Blot the result back into each frame, reject outlier pixels and drizzle again.
5. Deconvolve (Wiener). The frames are drizzled into two independent half-stacks
   (first / second half of the movie); their difference is a pure noise sample,
   which gives the stack's noise spectrum. The filter boosts each frequency by
   1/blur only as far as the measured signal-to-noise allows, so detail the
   optics attenuated comes back without amplifying noise. The blur (Gaussian
   PSF) is fitted from the spectrum, or set with `--psf-sigma`.
6. Save; `.png`/`.tif` are 16-bit so the bit depth gained by stacking is kept.

Super-resolution requires **sub-pixel dither**: frames must be offset from each
other by varying fractions of a pixel. The tool prints the "sub-pixel phase
coverage". If it is low (a perfectly rigid setup), you only get noise and
compression-artifact reduction. Then use `--pixfrac 1`, and when recording let
the stage drift slowly or nudge it gently. Registration handles motion of many
pixels; the fractional part is what matters.

Frame selection: the reference frame (the output view) is chosen automatically
as a sharp frame shared by many others (`--reference N` to force one); frames
overlapping it less than `--min-overlap` are skipped; `--keep 0.6` stacks only
the sharpest 60% (useful when parts of the clip are motion-blurred).

## Recording from the microscope (macOS)

QuickTime re-encodes the camera stream to H.264; `microscope-capture` reads the
camera directly (AVFoundation) at a chosen native mode and saves losslessly:

```sh
uv run microscope-capture list                               # cameras and modes
uv run microscope-capture probe                              # real detail + fps of every mode
uv run microscope-capture photo leaf.png --burst 10          # still; burst = aligned average
uv run microscope-capture record leaf.mkv --mode 1280x720 -s 20 [--preview]   # lossless FFV1
uv run microscope-movie leaf.mkv -o leaf_sr.png
```

Edubini microscope ("HD video", USB 2.0, MJPEG): 1280x720 (23 fps), 1280x960
(19 fps, cropped view) and 1920x1080 (12 fps) all resolve the same real detail;
1080p adds empty pixels. 640x480 and 800x600 deliver no frames after a mode switch.
macOS asks once for camera permission.

## iPad app (web app, `web/`)

A Safari web app for the microscope plugged into an iPad (iPadOS 17+, USB-C):
live view, **Photo**, and **Super photo** (records ~3 s, then runs this
pipeline on the iPad). Photos are kept in the app's own gallery (timestamped);
"Save / share" sends them to Photos or Files. No App Store, nothing uploaded.

The pipeline is ported to Rust (`web/engine/`, no OpenCV) and compiled to
WebAssembly. It runs in a worker plus helper workers that share the frame
registration and the drizzle (each drizzles one band of the output rows).
Rejected outliers are subtracted from the drizzle sums ("undrizzle") instead of
re-drizzling every frame. Pass 1 uses global alignment only.

```sh
web/build.sh                                   # build web/engine.wasm (cargo, wasm32 target)
uv run python web/serve.py                     # HTTPS on the LAN (self-signed); open the printed URL
uv run python web/engine/compare.py            # Rust vs Python on the synthetic scenes
uv run python web/engine/compare.py --video leaf_direct.mkv
uv run --with websockets python web/test_e2e.py leaf_direct.mkv   # headless Chrome, fake camera
(cd web/engine && cargo test --release && node parallel_node.mjs ../../results/compare/in.raw)
```

The microscope only delivers frames when 1280x720 is requested explicitly
(Safari's default 640x480 gives none); the app does that and retries.
`?helpers=0` runs everything in one worker. To deploy, serve `web/` (minus
`engine/`) from any static HTTPS host and bump `CACHE` in `web/sw.js`.

## Analysing a video

```sh
uv run microscope-analyze movie.mov --out-dir results
```

prints (1) single-frame signal/noise per spatial frequency: super-resolution
needs real detail near 0.5 cycles/px; (2) motion and sharpness statistics;
(3) Fourier ring correlation between independent reconstructions of the first and
second half of the movie: the frequency where it drops below 1/7 is the
resolution actually achieved, for Drizzle and for a plain aligned average.
It writes `<name>_frc.csv`, `<name>_crops.png` and `<name>_drizzle.png`.

Options: `--scale`, `--pixfrac`, `--motion {translation,euclidean,affine,homography}`, `--no-local`, `--tile`, `--keep`, `--reference`, `--min-overlap`,
`--no-reject`, `--reject-sigma`, `--no-deconvolve`, `--psf-sigma`, `--deconv-strength`,
`--save-raw PATH`, `--sharpen AMOUNT`, `--max-frames N`.

Tests (synthetic aliased scene with known ground truth): `uv run pytest`.
