//! Microscope super-resolution engine (port of the Python pipeline) with a plain
//! C ABI for WebAssembly; see web/worker.js for the JavaScript side.
//!
//! Exports (all sizes in bytes / pixels):
//!   mm_alloc(n) -> ptr, mm_free(ptr, n)       scratch memory for the caller
//!   mm_begin(w, h)                             start a new capture
//!   mm_add_frame(ptr_rgba) -> 1 kept / 0 dup   one RGBA frame of w*h*4 bytes
//!   mm_frames() -> distinct frames so far
//!   mm_process(scale, pixfrac, local, reject, deconvolve) -> 0 ok / 1 error
//!   mm_result_ptr/_w/_h                        RGBA u8 result
//!   mm_text_ptr/_len                           diagnostics JSON or error message
//!   mm_frame_rgba(row, dst)                    copy a stored frame (the reference) as RGBA
//!
//! The same pipeline split into steps, so that a coordinating worker can hand
//! registration and drizzle to helper workers (each its own instance):
//!   coordinator: mm_plan, mm_ref_gray, mm_frame_ptr, mm_p1_add, mm_p1_finish,
//!                mm_init_w, mm_p2_set, mm_alignment, mm_half_b, mm_acc_alloc,
//!                mm_acc_ptr, mm_p2_finish, mm_p3, mm_finish, mm_buf_ptr/_len
//!   helper:      mm_h_init, mm_h_ref, mm_h_align, mm_h_out_ptr/_len,
//!                mm_h_band, mm_h_drizzle, mm_h_band_ptr/_len
//! Imports (module "env"): js_progress(stage, done, total), js_log(ptr, len).

pub mod align;
pub mod deconv;
pub mod drizzle;
pub mod ecc;
pub mod fft;
pub mod img;
pub mod pipeline;

use align::Alignment;
use ecc::Registrar;
use img::{gray_u8, Img, EYE};
use pipeline::{align_pass1, align_pass2, Band, Engine, Job, Output, Params, Pass1Ref};
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
mod host {
    #[link(wasm_import_module = "env")]
    extern "C" {
        pub fn js_progress(stage: u32, done: u32, total: u32);
        pub fn js_log(ptr: *const u8, len: usize);
    }
}

fn progress(stage: u32, done: usize, total: usize) {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        host::js_progress(stage, done as u32, total as u32)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (stage, done, total);
}

pub fn log(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        host::js_log(msg.as_ptr(), msg.len())
    };
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{msg}");
}

struct Helper {
    w: usize,
    h: usize,
    prm: Params,
    p1: Option<Pass1Ref>,
    reg2: Option<Registrar>,
    band: Option<Band>,
    out: Vec<f64>,
}

struct State {
    engine: Option<Engine>,
    job: Option<Job>,
    result: Vec<u8>,
    rw: usize,
    rh: usize,
    text: String,
    buf: Vec<f64>,
    fbuf: Vec<f32>,
    helper: Option<Helper>,
}

thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State {
        engine: None, job: None, result: Vec::new(), rw: 0, rh: 0, text: String::new(),
        buf: Vec::new(), fbuf: Vec::new(), helper: None,
    }) };
}

fn params(scale: u32, pixfrac: f32, local: u32, reject: u32, deconvolve: u32) -> Params {
    Params {
        scale: scale.max(1) as usize,
        pixfrac: pixfrac as f64,
        local: local != 0,
        reject: reject != 0,
        deconvolve: deconvolve != 0,
        // pass 1 only builds the reference image: global alignment is enough
        // there (52+ dB from the full Python result) and saves ~20% of the time
        local_pass1: false,
        ..Params::default()
    }
}

fn store_output(s: &mut State, out: Output) {
    let mut rgba = Vec::with_capacity(out.w * out.h * 4);
    for p in out.image.chunks_exact(3) {
        for &v in p {
            rgba.push((v + 0.5).clamp(0.0, 255.0) as u8);
        }
        rgba.push(255);
    }
    s.result = rgba;
    s.rw = out.w;
    s.rh = out.h;
    s.text = out.diag;
}

/// Run `f` with the engine and the current job.
fn with_job<R>(f: impl FnOnce(&Engine, &mut Job, &mut State2) -> R) -> R {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let st = &mut *s;
        let e = st.engine.as_ref().expect("mm_begin first");
        let j = st.job.as_mut().expect("mm_plan first");
        let mut out = State2 { buf: &mut st.buf, fbuf: &mut st.fbuf };
        f(e, j, &mut out)
    })
}

struct State2<'a> {
    buf: &'a mut Vec<f64>,
    fbuf: &'a mut Vec<f32>,
}

unsafe fn slice_f64<'a>(p: *const f64, n: u32) -> &'a [f64] {
    std::slice::from_raw_parts(p, n as usize)
}

#[no_mangle]
pub extern "C" fn mm_alloc(n: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(n);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `p` must come from `mm_alloc(n)` with the same `n`.
#[no_mangle]
pub unsafe extern "C" fn mm_free(p: *mut u8, n: usize) {
    drop(Vec::from_raw_parts(p, 0, n));
}

#[no_mangle]
pub extern "C" fn mm_begin(w: u32, h: u32) {
    std::panic::set_hook(Box::new(|info| log(&format!("engine panic: {info}"))));
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.engine = Some(Engine::new(w as usize, h as usize));
        s.result = Vec::new();
        s.text.clear();
    });
}

/// # Safety
/// `p` must point to w*h*4 readable bytes (w, h from `mm_begin`).
#[no_mangle]
pub unsafe extern "C" fn mm_add_frame(p: *const u8) -> u32 {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let e = s.engine.as_mut().expect("mm_begin first");
        let px = std::slice::from_raw_parts(p, e.w * e.h * 4);
        e.add_frame(px, 4) as u32
    })
}

#[no_mangle]
pub extern "C" fn mm_frames() -> u32 {
    STATE.with(|s| s.borrow().engine.as_ref().map_or(0, |e| e.frames() as u32))
}

#[no_mangle]
pub extern "C" fn mm_process(scale: u32, pixfrac: f32, local: u32, reject: u32, deconvolve: u32) -> u32 {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let prm = params(scale, pixfrac, local, reject, deconvolve);
        let res = s.engine.as_ref().expect("mm_begin first").process(&prm, &mut progress);
        match res {
            Ok(out) => {
                store_output(&mut s, out);
                0
            }
            Err(e) => {
                s.text = e;
                1
            }
        }
    })
}

/// Copy distinct frame `row` as RGBA into `dst` (w*h*4 bytes); 0 if no such frame.
///
/// # Safety
/// `dst` must point to w*h*4 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn mm_frame_rgba(row: u32, dst: *mut u8) -> u32 {
    STATE.with(|s| {
        let s = s.borrow();
        let Some(e) = s.engine.as_ref() else { return 0 };
        let Some(rgb) = e.frame_rgb(row as usize) else { return 0 };
        let out = std::slice::from_raw_parts_mut(dst, e.w * e.h * 4);
        for (o, p) in out.chunks_exact_mut(4).zip(rgb.chunks_exact(3)) {
            o[..3].copy_from_slice(p);
            o[3] = 255;
        }
        1
    })
}

/// Free the frames and the result (call after reading the result).
#[no_mangle]
pub extern "C" fn mm_reset() {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.engine = None;
        s.job = None;
        s.result = Vec::new();
        s.fbuf = Vec::new();
    });
}

#[no_mangle]
pub extern "C" fn mm_result_ptr() -> *const u8 {
    STATE.with(|s| s.borrow().result.as_ptr())
}
#[no_mangle]
pub extern "C" fn mm_result_w() -> u32 {
    STATE.with(|s| s.borrow().rw as u32)
}
#[no_mangle]
pub extern "C" fn mm_result_h() -> u32 {
    STATE.with(|s| s.borrow().rh as u32)
}
#[no_mangle]
pub extern "C" fn mm_text_ptr() -> *const u8 {
    STATE.with(|s| s.borrow().text.as_ptr())
}
#[no_mangle]
pub extern "C" fn mm_text_len() -> u32 {
    STATE.with(|s| s.borrow().text.len() as u32)
}

// --------------------------------------------------------------------------
// Step-wise pipeline (coordinator side)
// --------------------------------------------------------------------------

/// Plan a run: choose the reference and the frames. Returns the number of
/// chosen frames (or -1, message in mm_text). mm_buf then holds
/// [ref_row, levels, out_w, out_h, n, (row, traj_x, traj_y) * n].
#[no_mangle]
pub extern "C" fn mm_plan(scale: u32, pixfrac: f32, local: u32, reject: u32, deconvolve: u32) -> i32 {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let st = &mut *s;
        let e = st.engine.as_ref().expect("mm_begin first");
        match Job::new(e, params(scale, pixfrac, local, reject, deconvolve)) {
            Err(m) => {
                st.text = m;
                -1
            }
            Ok(job) => {
                let n = job.chosen.len();
                st.buf = vec![job.ref_row as f64, job.levels as f64, job.wo as f64, job.ho as f64, n as f64];
                for &r in &job.chosen {
                    let t = job.traj(r);
                    st.buf.extend([r as f64, t[0], t[1]]);
                }
                st.job = Some(job);
                n as i32
            }
        }
    })
}

#[no_mangle]
pub extern "C" fn mm_buf_ptr() -> *const f64 {
    STATE.with(|s| s.borrow().buf.as_ptr())
}
#[no_mangle]
pub extern "C" fn mm_buf_len() -> u32 {
    STATE.with(|s| s.borrow().buf.len() as u32)
}

/// Gray reference frame (w*h f32) for the helpers' pass 1.
#[no_mangle]
pub extern "C" fn mm_ref_gray() -> *const f32 {
    with_job(|e, j, o| {
        *o.fbuf = j.ref_gray(e).d;
        o.fbuf.as_ptr()
    })
}

/// Stored distinct frame `row`, RGB u8 (w*h*3 bytes).
#[no_mangle]
pub extern "C" fn mm_frame_ptr(row: u32) -> *const u8 {
    STATE.with(|s| s.borrow().engine.as_ref().and_then(|e| e.frame_rgb(row as usize)).map_or(std::ptr::null(), |f| f.as_ptr()))
}

/// Pass-1 result for `row` (ok = 0: registration failed). The reference row
/// takes no alignment (identity).
///
/// # Safety
/// `al` must point to `n` f64 (an encoded alignment) when ok != 0.
#[no_mangle]
pub unsafe extern "C" fn mm_p1_add(row: u32, ok: u32, cc: f64, al: *const f64, n: u32) {
    with_job(|e, j, _| {
        let row = row as usize;
        let res = if row == j.ref_row {
            Some((Alignment::rigid(EYE), 1.0))
        } else if ok != 0 {
            Alignment::from_slice(slice_f64(al, n)).map(|a| (a, cc))
        } else {
            None
        };
        j.p1_add(e, row, res);
    })
}

/// End of pass 1: returns the stacked reference (w*h f32); mm_buf = [n, rows...].
#[no_mangle]
pub extern "C" fn mm_p1_finish() -> *const f32 {
    with_job(|e, j, o| {
        *o.fbuf = j.p1_finish(e).d;
        *o.buf = std::iter::once(j.rows.len() as f64).chain(j.rows.iter().map(|&r| r as f64)).collect();
        o.fbuf.as_ptr()
    })
}

/// Pass-1 warp of `row` (9 f64) as the pass-2 starting point.
#[no_mangle]
pub extern "C" fn mm_init_w(row: u32) -> *const f64 {
    with_job(|_, j, o| {
        *o.buf = j.init_w(row as usize).to_vec();
        o.buf.as_ptr()
    })
}

/// Pass-2 result for `row` (ok = 0: keep the pass-1 alignment).
///
/// # Safety
/// `al` must point to `n` f64 when ok != 0.
#[no_mangle]
pub unsafe extern "C" fn mm_p2_set(row: u32, ok: u32, tiles: u32, al: *const f64, n: u32) {
    with_job(|_, j, _| {
        let res = if ok != 0 { Alignment::from_slice(slice_f64(al, n)).map(|a| (a, tiles as usize)) } else { None };
        j.p2_set(row as usize, res);
    })
}

/// Encoded final alignment of `row` into mm_buf; returns its length.
#[no_mangle]
pub extern "C" fn mm_alignment(row: u32) -> u32 {
    with_job(|_, j, o| {
        *o.buf = j.alignment(row as usize).to_vec();
        o.buf.len() as u32
    })
}

#[no_mangle]
pub extern "C" fn mm_half_b(row: u32) -> u32 {
    with_job(|_, j, _| j.half_b(row as usize) as u32)
}

#[no_mangle]
pub extern "C" fn mm_acc_alloc() {
    with_job(|_, j, _| j.alloc_acc())
}

/// Full-size accumulators to fill from the bands: 0 acc_a, 1 acc_b (RGB), 2 wa, 3 wb.
#[no_mangle]
pub extern "C" fn mm_acc_ptr(k: u32) -> *mut f32 {
    with_job(|_, j, _| match k {
        0 => j.acc_a.as_mut_ptr(),
        1 => j.acc_b.as_mut_ptr(),
        2 => j.wa.as_mut_ptr(),
        _ => j.wb.as_mut_ptr(),
    })
}

#[no_mangle]
pub extern "C" fn mm_p2_finish() {
    with_job(|_, j, _| j.p2_finish())
}

#[no_mangle]
pub extern "C" fn mm_p3() {
    with_job(|e, j, _| j.p3(e, &mut progress))
}

/// Noise, deconvolution and the RGBA result (mm_result_*, mm_text). Returns 0.
#[no_mangle]
pub extern "C" fn mm_finish() -> u32 {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let job = s.job.take().expect("mm_plan first");
        let out = job.finish(s.engine.as_ref().expect("mm_begin first"), &mut progress);
        store_output(&mut s, out);
        0
    })
}

// --------------------------------------------------------------------------
// Helper side: registration and one band of the drizzle
// --------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn mm_h_init(w: u32, h: u32, scale: u32, pixfrac: f32, local: u32) {
    std::panic::set_hook(Box::new(|info| log(&format!("helper panic: {info}"))));
    STATE.with(|s| {
        s.borrow_mut().helper = Some(Helper {
            w: w as usize, h: h as usize, prm: params(scale, pixfrac, local, 1, 1),
            p1: None, reg2: None, band: None, out: vec![],
        });
    });
}

fn with_helper<R>(f: impl FnOnce(&mut Helper) -> R) -> R {
    STATE.with(|s| f(s.borrow_mut().helper.as_mut().expect("mm_h_init first")))
}

/// Set the reference (w*h f32 gray) for pass 1 (phase correlation + ECC
/// against one frame) or pass 2 (ECC against the stacked reference).
///
/// # Safety
/// `gray` must point to w*h f32.
#[no_mangle]
pub unsafe extern "C" fn mm_h_ref(gray: *const f32, levels: u32, pass: u32) {
    with_helper(|hp| {
        let img = Img { w: hp.w, h: hp.h, d: std::slice::from_raw_parts(gray, hp.w * hp.h).to_vec() };
        hp.p1 = None;
        hp.reg2 = None;
        if pass == 1 {
            hp.p1 = Some(Pass1Ref::new(img, hp.prm.motion, levels as usize));
        } else {
            hp.reg2 = Some(Registrar::new(img, hp.prm.motion, levels as usize));
        }
    })
}

/// Register one RGB frame. Pass 1 uses (traj_x, traj_y), pass 2 the 9-f64
/// warp `init`. Returns 1 on success; mm_h_out = [cc, tiles, alignment...].
///
/// # Safety
/// `rgb` must point to w*h*3 bytes, `init` to 9 f64 (pass 2).
#[no_mangle]
pub unsafe extern "C" fn mm_h_align(rgb: *const u8, traj_x: f64, traj_y: f64, init: *const f64) -> u32 {
    with_helper(|hp| {
        let g = gray_u8(std::slice::from_raw_parts(rgb, hp.w * hp.h * 3), hp.w, hp.h, 3);
        let res = if let Some(p1) = &hp.p1 {
            align_pass1(p1, &g, [traj_x, traj_y], &hp.prm)
        } else if let Some(reg) = &hp.reg2 {
            let w: img::M3 = slice_f64(init, 9).try_into().unwrap();
            align_pass2(reg, &g, &w, &hp.prm)
        } else {
            None
        };
        match res {
            Some((al, cc, tiles)) => {
                hp.out = vec![cc, tiles as f64];
                hp.out.extend(al.to_vec());
                1
            }
            None => {
                hp.out.clear();
                0
            }
        }
    })
}

#[no_mangle]
pub extern "C" fn mm_h_out_ptr() -> *const f64 {
    with_helper(|hp| hp.out.as_ptr())
}
#[no_mangle]
pub extern "C" fn mm_h_out_len() -> u32 {
    with_helper(|hp| hp.out.len() as u32)
}

/// Own the drizzle accumulators of output rows r0..r1.
#[no_mangle]
pub extern "C" fn mm_h_band(r0: u32, r1: u32) {
    with_helper(|hp| hp.band = Some(Band::new(hp.w, hp.h, &hp.prm, r0 as usize, r1 as usize)))
}

/// Drizzle one RGB frame into this helper's band.
///
/// # Safety
/// `rgb` must point to w*h*3 bytes and `al` to `n` f64.
#[no_mangle]
pub unsafe extern "C" fn mm_h_drizzle(rgb: *const u8, al: *const f64, n: u32, half_b: u32) {
    with_helper(|hp| {
        let al = Alignment::from_slice(slice_f64(al, n)).expect("bad alignment");
        let px = std::slice::from_raw_parts(rgb, hp.w * hp.h * 3);
        hp.band.as_mut().expect("mm_h_band first").add(px, &al, half_b != 0);
    })
}

/// Band accumulators: 0 acc_a, 1 acc_b, 2 wa, 3 wb (pointer and f32 count).
#[no_mangle]
pub extern "C" fn mm_h_band_ptr(k: u32) -> *const f32 {
    with_helper(|hp| {
        let b = hp.band.as_ref().expect("mm_h_band first");
        [&b.acc_a, &b.acc_b, &b.wa, &b.wb][k.min(3) as usize].as_ptr()
    })
}
#[no_mangle]
pub extern "C" fn mm_h_band_len(k: u32) -> u32 {
    with_helper(|hp| {
        let b = hp.band.as_ref().expect("mm_h_band first");
        [&b.acc_a, &b.acc_b, &b.wa, &b.wb][k.min(3) as usize].len() as u32
    })
}

/// Free the helper's reference and band.
#[no_mangle]
pub extern "C" fn mm_h_reset() {
    STATE.with(|s| s.borrow_mut().helper = None);
}
