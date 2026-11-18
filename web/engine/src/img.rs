//! Small image toolbox: 3x3 matrices, gray images, separable filters with
//! OpenCV border conventions, pyramids, sampling, morphology.
//!
//! Coordinates follow OpenCV: pixel (x, y) has its centre at integer (x, y).

pub type M3 = [f64; 9];
pub const EYE: M3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub fn mul(a: &M3, b: &M3) -> M3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    r
}

pub fn inv(m: &M3) -> M3 {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let co = [
        e * i - f * h,
        -(d * i - f * g),
        d * h - e * g,
        -(b * i - c * h),
        a * i - c * g,
        -(a * h - b * g),
        b * f - c * e,
        -(a * f - c * d),
        a * e - b * d,
    ];
    let det = a * co[0] + b * co[1] + c * co[2];
    [
        co[0] / det, co[3] / det, co[6] / det,
        co[1] / det, co[4] / det, co[7] / det,
        co[2] / det, co[5] / det, co[8] / det,
    ]
}

#[inline(always)]
pub fn apply(h: &M3, x: f64, y: f64) -> (f64, f64) {
    let w = h[6] * x + h[7] * y + h[8];
    ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
}

pub fn translation(dx: f64, dy: f64) -> M3 {
    [1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0]
}

/// Express a warp in coordinates scaled by f (pyrDown: x_coarse = x / 2): D W D^-1.
pub fn rescale(w: &M3, f: f64) -> M3 {
    [w[0], w[1], w[2] * f, w[3], w[4], w[5] * f, w[6] / f, w[7] / f, w[8]]
}

#[derive(Clone)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Img {
    pub fn new(w: usize, h: usize) -> Self {
        Img { w, h, d: vec![0.0; w * h] }
    }
    #[inline(always)]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.d[y * self.w + x]
    }
}

/// Gray level in [0, 1] (OpenCV BGR2GRAY weights) of an interleaved RGB(A) u8 image.
pub fn gray_u8(px: &[u8], w: usize, h: usize, stride: usize) -> Img {
    let mut g = Img::new(w, h);
    for (o, p) in g.d.iter_mut().zip(px.chunks_exact(stride)) {
        *o = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) * (1.0 / 255.0);
    }
    g
}

#[derive(Clone, Copy, PartialEq)]
pub enum Border {
    Reflect101,
    Replicate,
}

#[inline(always)]
pub fn bidx(i: isize, n: usize, b: Border) -> usize {
    let ni = n as isize;
    if n == 1 {
        return 0;
    }
    match b {
        Border::Replicate => i.clamp(0, ni - 1) as usize,
        Border::Reflect101 => {
            let mut i = i;
            loop {
                if i < 0 {
                    i = -i;
                } else if i >= ni {
                    i = 2 * ni - 2 - i;
                } else {
                    return i as usize;
                }
            }
        }
    }
}

/// Gaussian kernel with OpenCV's size rule for float images (ksize = round(8 sigma + 1) | 1).
pub fn gaussian_kernel(sigma: f64) -> Vec<f32> {
    let n = ((sigma * 8.0 + 1.0).round() as usize) | 1;
    let r = (n / 2) as f64;
    let mut k: Vec<f64> = (0..n).map(|i| (-(i as f64 - r).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k.into_iter().map(|v| v as f32).collect()
}

/// Separable filter (same 1-D kernel along x and y) of an interleaved image with c channels.
pub fn sep_filter(src: &[f32], w: usize, h: usize, c: usize, k: &[f32], border: Border) -> Vec<f32> {
    let r = k.len() / 2;
    let rowlen = w * c;
    let mut tmp = vec![0f32; rowlen * h];
    let mut pad = vec![0f32; (w + 2 * r) * c];
    for y in 0..h {
        let row = &src[y * rowlen..(y + 1) * rowlen];
        for xp in 0..w + 2 * r {
            let sx = bidx(xp as isize - r as isize, w, border);
            pad[xp * c..xp * c + c].copy_from_slice(&row[sx * c..sx * c + c]);
        }
        let out = &mut tmp[y * rowlen..(y + 1) * rowlen];
        for (j, &kj) in k.iter().enumerate() {
            let p = &pad[j * c..j * c + rowlen];
            for (o, &v) in out.iter_mut().zip(p) {
                *o += kj * v;
            }
        }
    }
    let mut dst = vec![0f32; rowlen * h];
    for y in 0..h {
        let out = &mut dst[y * rowlen..(y + 1) * rowlen];
        for (j, &kj) in k.iter().enumerate() {
            let sy = bidx(y as isize + j as isize - r as isize, h, border);
            let p = &tmp[sy * rowlen..(sy + 1) * rowlen];
            for (o, &v) in out.iter_mut().zip(p) {
                *o += kj * v;
            }
        }
    }
    dst
}

pub fn gaussian_blur(img: &Img, sigma: f64) -> Img {
    Img { w: img.w, h: img.h, d: sep_filter(&img.d, img.w, img.h, 1, &gaussian_kernel(sigma), Border::Reflect101) }
}

/// The 5-tap binomial kernel used by pyrDown and by ECC's pre-smoothing (gaussFiltSize 5).
pub const K5: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];

pub fn blur5(img: &Img) -> Img {
    Img { w: img.w, h: img.h, d: sep_filter(&img.d, img.w, img.h, 1, &K5, Border::Reflect101) }
}

/// cv2.pyrDown: 5x5 binomial blur, keep even rows/columns.
pub fn pyr_down(img: &Img) -> Img {
    let b = blur5(img);
    let (w2, h2) = ((img.w + 1) / 2, (img.h + 1) / 2);
    let mut out = Img::new(w2, h2);
    for y in 0..h2 {
        for x in 0..w2 {
            out.d[y * w2 + x] = b.d[2 * y * img.w + 2 * x];
        }
    }
    out
}

/// Half-size by 2x2 averaging (cv2.resize INTER_AREA with fx = fy = 0.5).
pub fn down2(img: &Img) -> Img {
    let (w2, h2) = (img.w / 2, img.h / 2);
    let mut out = Img::new(w2, h2);
    for y in 0..h2 {
        let r0 = &img.d[2 * y * img.w..];
        let r1 = &img.d[(2 * y + 1) * img.w..];
        for x in 0..w2 {
            out.d[y * w2 + x] = 0.25 * (r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1]);
        }
    }
    out
}

/// Central-difference gradients (filter2D with [-0.5, 0, 0.5], BORDER_REFLECT_101).
pub fn grad(img: &Img) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (img.w, img.h);
    let mut gx = vec![0f32; w * h];
    let mut gy = vec![0f32; w * h];
    for y in 0..h {
        let yu = bidx(y as isize - 1, h, Border::Reflect101);
        let yd = bidx(y as isize + 1, h, Border::Reflect101);
        for x in 0..w {
            let xl = bidx(x as isize - 1, w, Border::Reflect101);
            let xr = bidx(x as isize + 1, w, Border::Reflect101);
            gx[y * w + x] = 0.5 * (img.d[y * w + xr] - img.d[y * w + xl]);
            gy[y * w + x] = 0.5 * (img.d[yd * w + x] - img.d[yu * w + x]);
        }
    }
    (gx, gy)
}

/// cv2.createHanningWindow, row-major.
pub fn hanning(w: usize, h: usize) -> Vec<f32> {
    let f = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|j| if n > 1 { 0.5 * (1.0 - (2.0 * std::f64::consts::PI * j as f64 / (n - 1) as f64).cos()) } else { 1.0 })
            .collect()
    };
    let (wc, wr) = (f(w), f(h));
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = (wr[y] * wc[x]) as f32;
        }
    }
    out
}

/// Bilinear sample of interleaved c-channel data, coordinates clamped to the image
/// (= BORDER_REPLICATE). Writes c values into `out`.
#[inline(always)]
pub fn sample_clamped(d: &[f32], w: usize, h: usize, c: usize, x: f64, y: f64, out: &mut [f32]) {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let x0 = (x as usize).min(w.saturating_sub(2));
    let y0 = (y as usize).min(h.saturating_sub(2));
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let (i00, i01, i10, i11) = ((y0 * w + x0) * c, (y0 * w + x1) * c, (y1 * w + x0) * c, (y1 * w + x1) * c);
    for k in 0..c {
        let top = d[i00 + k] + fx * (d[i01 + k] - d[i00 + k]);
        let bot = d[i10 + k] + fx * (d[i11 + k] - d[i10 + k]);
        out[k] = top + fy * (bot - top);
    }
}

/// Bilinear sample of a gray image with zero outside (BORDER_CONSTANT 0).
/// Returns (value, weight): weight is what a warped image of ones would give.
#[inline(always)]
pub fn sample_zero(d: &[f32], w: usize, h: usize, x: f64, y: f64) -> (f32, f32) {
    let xf = x.floor();
    let yf = y.floor();
    let (x0, y0) = (xf as isize, yf as isize);
    let fx = (x - xf) as f32;
    let fy = (y - yf) as f32;
    let (wi, hi) = (w as isize, h as isize);
    let mut v = 0f32;
    let mut s = 0f32;
    for (dy, wy) in [(0isize, 1.0 - fy), (1, fy)] {
        let yy = y0 + dy;
        if yy < 0 || yy >= hi || wy == 0.0 {
            continue;
        }
        for (dx, wx) in [(0isize, 1.0 - fx), (1, fx)] {
            let xx = x0 + dx;
            if xx < 0 || xx >= wi || wx == 0.0 {
                continue;
            }
            let wt = wx * wy;
            v += wt * d[yy as usize * w + xx as usize];
            s += wt;
        }
    }
    (v, s)
}

/// 3x3 max (dilate) or min (erode); out-of-image neighbours are ignored.
pub fn morph3(d: &[f32], w: usize, h: usize, max: bool) -> Vec<f32> {
    let pick = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
    // rows then columns (separable for max/min)
    let mut t = vec![0f32; w * h];
    for y in 0..h {
        let r = &d[y * w..(y + 1) * w];
        for x in 0..w {
            let mut v = r[x];
            if x > 0 {
                v = pick(v, r[x - 1]);
            }
            if x + 1 < w {
                v = pick(v, r[x + 1]);
            }
            t[y * w + x] = v;
        }
    }
    let mut o = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut v = t[y * w + x];
            if y > 0 {
                v = pick(v, t[(y - 1) * w + x]);
            }
            if y + 1 < h {
                v = pick(v, t[(y + 1) * w + x]);
            }
            o[y * w + x] = v;
        }
    }
    o
}

pub fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    *m
}

pub fn median_f64(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

pub fn mean_std(d: &[f32]) -> (f64, f64) {
    let n = d.len().max(1) as f64;
    let m = d.iter().map(|&v| v as f64).sum::<f64>() / n;
    let v = d.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / n;
    (m, v.sqrt())
}
