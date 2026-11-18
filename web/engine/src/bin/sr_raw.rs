//! Native test harness: run the engine on raw frames written by compare.py.
//!
//!   sr_raw IN.raw OUT.raw [--no-local] [--no-reject] [--no-deconv] [--pixfrac F] [--scale S]
//!
//! IN:  b"MMF1", u32 w, h, n (little endian), then n frames of w*h*3 RGB u8
//! OUT: b"MMO1", u32 w, h, then (h, w, 3) f32 final image, then the same size
//!      f32 image before deconvolution (zeros if none). Diagnostics JSON on stdout.

use microscope_engine::pipeline::{Engine, Params};
use std::io::{Read, Write};
use std::time::Instant;

fn u32_at(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut prm = Params { keep_raw: true, ..Params::default() };
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--no-local" => prm.local = false,
            "--no-local-pass1" => prm.local_pass1 = false,
            "--no-reject" => prm.reject = false,
            "--no-deconv" => prm.deconvolve = false,
            "--pixfrac" => {
                i += 1;
                prm.pixfrac = args[i].parse().unwrap()
            }
            "--reference" => {
                i += 1;
                prm.reference = Some(args[i].parse().unwrap())
            }
            "--scale" => {
                i += 1;
                prm.scale = args[i].parse().unwrap()
            }
            a => panic!("unknown option {a}"),
        }
        i += 1;
    }
    let mut buf = vec![];
    std::fs::File::open(&args[1]).unwrap().read_to_end(&mut buf).unwrap();
    assert_eq!(&buf[..4], b"MMF1");
    let (w, h, n) = (u32_at(&buf, 4), u32_at(&buf, 8), u32_at(&buf, 12));
    let t0 = Instant::now();
    let mut e = Engine::new(w, h);
    for k in 0..n {
        let off = 16 + k * w * h * 3;
        e.add_frame(&buf[off..off + w * h * 3], 3);
    }
    let t_scan = t0.elapsed().as_secs_f64();
    let mut last = (0, Instant::now());
    let mut times = vec![];
    let out = e
        .process(&prm, &mut |st, _, _| {
            if st != last.0 {
                times.push((last.0, last.1.elapsed().as_secs_f64()));
                last = (st, Instant::now());
            }
        })
        .unwrap();
    times.push((last.0, last.1.elapsed().as_secs_f64()));
    eprintln!("scan {t_scan:.2}s; stages {times:?}; total {:.2}s", t0.elapsed().as_secs_f64());
    let mut f = std::fs::File::create(&args[2]).unwrap();
    f.write_all(b"MMO1").unwrap();
    f.write_all(&(out.w as u32).to_le_bytes()).unwrap();
    f.write_all(&(out.h as u32).to_le_bytes()).unwrap();
    let raw = out.raw.unwrap_or_else(|| vec![0.0; out.image.len()]);
    for img in [&out.image, &raw] {
        let bytes: Vec<u8> = img.iter().flat_map(|v| v.to_le_bytes()).collect();
        f.write_all(&bytes).unwrap();
    }
    println!("{}", out.diag);
}
