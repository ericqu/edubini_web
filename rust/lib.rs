//! Drizzle (variable-pixel linear reconstruction) kernel.
//!
//! Every input pixel is shrunk to a square "drop" of side `drop_size` (in output
//! pixels), its centre is mapped onto the fine output grid through a homography,
//! and its value is added to each output pixel it overlaps, weighted by the exact
//! overlap area. Because the frames sample the scene at different sub-pixel
//! phases, the drops land on different fine-grid positions and the aliased
//! high-frequency content is "unfolded" onto the finer grid.
//!
//! The kernel is written in *gather* form: each output pixel looks up the input
//! pixels whose drops can touch it. Output rows are therefore independent and are
//! processed in parallel without locks or per-thread accumulators.
//!
//! Coordinates follow OpenCV: pixel (x, y) has its centre at integer (x, y) and
//! covers [x-0.5, x+0.5] x [y-0.5, y+0.5].

use numpy::{
    PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const MAX_CHANNELS: usize = 4;

#[inline(always)]
fn project(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let w = h[6] * x + h[7] * y + h[8];
    (
        (h[0] * x + h[1] * y + h[2]) / w,
        (h[3] * x + h[4] * y + h[5]) / w,
    )
}

/// Length of the overlap of [c - half, c + half] with the unit pixel centred on p.
#[inline(always)]
fn overlap(c: f64, half: f64, p: f64) -> f64 {
    ((c + half).min(p + 0.5) - (c - half).max(p - 0.5)).max(0.0)
}

/// Add one frame into the drizzle accumulators (in place).
///
/// acc      (Ho, Wo, C) float32  running sum of weight * value
/// wacc     (Ho, Wo)    float32  running sum of weights
/// frame    (H, W, C)   float32  input frame
/// weight   (H, W)      float32  per-input-pixel weight (0 = masked / rejected)
/// h        row-major 3x3 homography: input pixel coords -> output pixel coords
/// h_inv    its inverse
/// drop_size  side of the drop square, in output pixels (pixfrac * scale)
#[pyfunction]
#[pyo3(signature = (acc, wacc, frame, weight, h, h_inv, drop_size))]
#[allow(clippy::too_many_arguments)]
fn drizzle_add(
    py: Python<'_>,
    mut acc: PyReadwriteArray3<f32>,
    mut wacc: PyReadwriteArray2<f32>,
    frame: PyReadonlyArray3<f32>,
    weight: PyReadonlyArray2<f32>,
    h: [f64; 9],
    h_inv: [f64; 9],
    drop_size: f64,
) -> PyResult<()> {
    let (ho, wo, c) = {
        let s = acc.shape();
        (s[0], s[1], s[2])
    };
    let (hi, wi) = {
        let s = frame.shape();
        if s[2] != c {
            return Err(PyValueError::new_err("frame and acc channel counts differ"));
        }
        (s[0], s[1])
    };
    if c == 0 || c > MAX_CHANNELS {
        return Err(PyValueError::new_err("channel count must be 1..=4"));
    }
    if wacc.shape() != [ho, wo] {
        return Err(PyValueError::new_err("wacc shape must equal acc.shape[:2]"));
    }
    if weight.shape() != [hi, wi] {
        return Err(PyValueError::new_err("weight shape must equal frame.shape[:2]"));
    }
    if !(drop_size > 0.0) {
        return Err(PyValueError::new_err("drop_size must be > 0"));
    }
    let contiguous = |_| PyValueError::new_err("arrays must be C-contiguous");
    let acc_s = acc.as_slice_mut().map_err(contiguous)?;
    let wacc_s = wacc.as_slice_mut().map_err(contiguous)?;
    let frame_s = frame.as_slice().map_err(contiguous)?;
    let weight_s = weight.as_slice().map_err(contiguous)?;

    let half = 0.5 * drop_size;
    // An input pixel can touch output pixel (j, i) only if its drop centre lies
    // within `reach` (L-inf, output px) of (j, i). Map that box back to input
    // space with the inverse Jacobian (exact for affine maps; a 1 px margin
    // covers the perspective term of a homography).
    let reach = half + 0.5;
    let projective = h_inv[6] != 0.0 || h_inv[7] != 0.0;
    let s = h_inv[8];
    let rx = (h_inv[0].abs() + h_inv[1].abs()) / s.abs() * reach + if projective { 1.0 } else { 1e-9 };
    let ry = (h_inv[3].abs() + h_inv[4].abs()) / s.abs() * reach + if projective { 1.0 } else { 1e-9 };
    let (wi_i, hi_i) = (wi as isize, hi as isize);

    py.detach(|| {
        acc_s
            .par_chunks_mut(wo * c)
            .zip(wacc_s.par_chunks_mut(wo))
            .enumerate()
            .for_each(|(i, (acc_row, w_row))| {
                let yo = i as f64;
                for j in 0..wo {
                    let xo = j as f64;
                    let (xc, yc) = project(&h_inv, xo, yo);
                    if !xc.is_finite() || !yc.is_finite() {
                        continue;
                    }
                    let x0 = ((xc - rx).ceil() as isize).max(0);
                    let x1 = ((xc + rx).floor() as isize).min(wi_i - 1);
                    let y0 = ((yc - ry).ceil() as isize).max(0);
                    let y1 = ((yc + ry).floor() as isize).min(hi_i - 1);
                    if x0 > x1 || y0 > y1 {
                        continue;
                    }
                    let mut sw = 0.0f64;
                    let mut sv = [0.0f64; MAX_CHANNELS];
                    for y in y0..=y1 {
                        let row = y as usize * wi;
                        for x in x0..=x1 {
                            let idx = row + x as usize;
                            let wv = weight_s[idx];
                            if wv <= 0.0 {
                                continue;
                            }
                            let (u, v) = project(&h, x as f64, y as f64);
                            let a = overlap(u, half, xo) * overlap(v, half, yo);
                            if a <= 0.0 {
                                continue;
                            }
                            let w = a * wv as f64;
                            sw += w;
                            let px = &frame_s[idx * c..idx * c + c];
                            for k in 0..c {
                                sv[k] += w * px[k] as f64;
                            }
                        }
                    }
                    if sw > 0.0 {
                        w_row[j] += sw as f32;
                        let out = &mut acc_row[j * c..j * c + c];
                        for k in 0..c {
                            out[k] += sv[k] as f32;
                        }
                    }
                }
            });
    });
    Ok(())
}

/// Add one frame into the drizzle accumulators through a dense coordinate map.
///
/// Same as `drizzle_add`, but the geometry is an arbitrary smooth warp given as
/// `map` (Ho, Wo, 2): the input-frame coordinates (x, y) of every output pixel
/// centre. This supports non-rigid (locally refined) alignment. Around each
/// output pixel the warp is linearised with the map's finite-difference
/// Jacobian, which is exact for affine warps and accurate for smooth ones.
#[pyfunction]
#[pyo3(signature = (acc, wacc, frame, weight, map, drop_size))]
fn drizzle_add_map(
    py: Python<'_>,
    mut acc: PyReadwriteArray3<f32>,
    mut wacc: PyReadwriteArray2<f32>,
    frame: PyReadonlyArray3<f32>,
    weight: PyReadonlyArray2<f32>,
    map: PyReadonlyArray3<f32>,
    drop_size: f64,
) -> PyResult<()> {
    let (ho, wo, c) = {
        let s = acc.shape();
        (s[0], s[1], s[2])
    };
    let (hi, wi) = {
        let s = frame.shape();
        if s[2] != c {
            return Err(PyValueError::new_err("frame and acc channel counts differ"));
        }
        (s[0], s[1])
    };
    if c == 0 || c > MAX_CHANNELS {
        return Err(PyValueError::new_err("channel count must be 1..=4"));
    }
    if wacc.shape() != [ho, wo] {
        return Err(PyValueError::new_err("wacc shape must equal acc.shape[:2]"));
    }
    if weight.shape() != [hi, wi] {
        return Err(PyValueError::new_err("weight shape must equal frame.shape[:2]"));
    }
    if map.shape() != [ho, wo, 2] {
        return Err(PyValueError::new_err("map shape must be (Ho, Wo, 2)"));
    }
    if ho < 2 || wo < 2 {
        return Err(PyValueError::new_err("output must be at least 2x2"));
    }
    if !(drop_size > 0.0) {
        return Err(PyValueError::new_err("drop_size must be > 0"));
    }
    let contiguous = |_| PyValueError::new_err("arrays must be C-contiguous");
    let acc_s = acc.as_slice_mut().map_err(contiguous)?;
    let wacc_s = wacc.as_slice_mut().map_err(contiguous)?;
    let frame_s = frame.as_slice().map_err(contiguous)?;
    let weight_s = weight.as_slice().map_err(contiguous)?;
    let map_s = map.as_slice().map_err(contiguous)?;

    let half = 0.5 * drop_size;
    let reach = half + 0.5;
    let (wi_i, hi_i) = (wi as isize, hi as isize);
    let at = |i: usize, j: usize| -> (f64, f64) {
        let k = (i * wo + j) * 2;
        (map_s[k] as f64, map_s[k + 1] as f64)
    };

    py.detach(|| {
        acc_s
            .par_chunks_mut(wo * c)
            .zip(wacc_s.par_chunks_mut(wo))
            .enumerate()
            .for_each(|(i, (acc_row, w_row))| {
                let yo = i as f64;
                let (iu, id) = (i.saturating_sub(1), (i + 1).min(ho - 1));
                for j in 0..wo {
                    let xo = j as f64;
                    let (xc, yc) = at(i, j);
                    if !xc.is_finite() || !yc.is_finite() {
                        continue;
                    }
                    // Jacobian of output -> input (columns: d/dj, d/di).
                    let (jl, jr) = (j.saturating_sub(1), (j + 1).min(wo - 1));
                    let (pl, pr) = (at(i, jl), at(i, jr));
                    let (pu, pd) = (at(iu, j), at(id, j));
                    let dj = (jr - jl) as f64;
                    let di = (id - iu) as f64;
                    let (a, b) = ((pr.0 - pl.0) / dj, (pd.0 - pu.0) / di);
                    let (cc, d) = ((pr.1 - pl.1) / dj, (pd.1 - pu.1) / di);
                    let det = a * d - b * cc;
                    if !det.is_finite() || det.abs() < 1e-12 {
                        continue;
                    }
                    // Forward Jacobian (input -> output) = inverse.
                    let (fa, fb, fc, fd) = (d / det, -b / det, -cc / det, a / det);
                    let rx = (a.abs() + b.abs()) * reach + 1e-9;
                    let ry = (cc.abs() + d.abs()) * reach + 1e-9;
                    let x0 = ((xc - rx).ceil() as isize).max(0);
                    let x1 = ((xc + rx).floor() as isize).min(wi_i - 1);
                    let y0 = ((yc - ry).ceil() as isize).max(0);
                    let y1 = ((yc + ry).floor() as isize).min(hi_i - 1);
                    if x0 > x1 || y0 > y1 {
                        continue;
                    }
                    let mut sw = 0.0f64;
                    let mut sv = [0.0f64; MAX_CHANNELS];
                    for y in y0..=y1 {
                        let row = y as usize * wi;
                        let ey = y as f64 - yc;
                        for x in x0..=x1 {
                            let idx = row + x as usize;
                            let wv = weight_s[idx];
                            if wv <= 0.0 {
                                continue;
                            }
                            let ex = x as f64 - xc;
                            let u = xo + fa * ex + fb * ey;
                            let v = yo + fc * ex + fd * ey;
                            let ov = overlap(u, half, xo) * overlap(v, half, yo);
                            if ov <= 0.0 {
                                continue;
                            }
                            let w = ov * wv as f64;
                            sw += w;
                            let px = &frame_s[idx * c..idx * c + c];
                            for k in 0..c {
                                sv[k] += w * px[k] as f64;
                            }
                        }
                    }
                    if sw > 0.0 {
                        w_row[j] += sw as f32;
                        let out = &mut acc_row[j * c..j * c + c];
                        for k in 0..c {
                            out[k] += sv[k] as f32;
                        }
                    }
                }
            });
    });
    Ok(())
}

#[pymodule]
fn _drizzle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(drizzle_add, m)?)?;
    m.add_function(wrap_pyfunction!(drizzle_add_map, m)?)?;
    Ok(())
}
