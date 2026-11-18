//! Timing of the pipeline's building blocks on raw frames (see compare.py).
use microscope_engine::align::{local_field, Alignment};
use microscope_engine::drizzle::drizzle_add_map;
use microscope_engine::ecc::{Motion, Registrar};
use microscope_engine::img::*;
use std::io::Read;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let mut buf = vec![];
    std::fs::File::open(&path).unwrap().read_to_end(&mut buf).unwrap();
    let u = |i: usize| u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
    let (w, h, n) = (u(4), u(8), u(12));
    let frame = |k: usize| &buf[16 + k * w * h * 3..16 + (k + 1) * w * h * 3];
    let g0 = gray_u8(frame(0), w, h, 3);
    let reg = Registrar::new(g0.clone(), Motion::Euclidean, 4);
    let t = Instant::now();
    let mut als = vec![];
    for k in 1..n.min(11) {
        let g = gray_u8(frame(k), w, h, 3);
        als.push((reg.register(&g, &EYE, None).unwrap().0, g));
    }
    let m = als.len() as f64;
    println!("global ECC 4 levels: {:.1} ms/frame", t.elapsed().as_secs_f64() * 1e3 / m);
    let t = Instant::now();
    for (wm, g) in &als {
        reg.register(g, wm, Some(2)).unwrap();
    }
    println!("global ECC 2 levels from previous: {:.1} ms/frame", t.elapsed().as_secs_f64() * 1e3 / m);
    let t = Instant::now();
    let mut warped = vec![];
    for (wm, g) in &als {
        warped.push(Alignment::rigid(*wm).warp_to_ref(g));
    }
    println!("warp_to_ref: {:.1} ms/frame", t.elapsed().as_secs_f64() * 1e3 / m);
    let t = Instant::now();
    let mut fields = vec![];
    for (a, v) in &warped {
        fields.push(local_field(&reg.raw[0], &reg.blur[0], a, v, 96).unwrap());
    }
    println!("local_field: {:.1} ms/frame ({} good tiles)", t.elapsed().as_secs_f64() * 1e3 / m, fields[0].1);
    let al = Alignment { field: Some(fields[0].0.clone()), ..Alignment::rigid(als[0].0) };
    let (wo, ho) = (2 * w, 2 * h);
    let mut map = vec![];
    let t = Instant::now();
    al.output_map(wo, 0, ho, 2, &mut map);
    println!("output_map: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    let f: Vec<f32> = frame(1).iter().map(|&v| v as f32).collect();
    let wt = vec![1f32; w * h];
    let mut acc = vec![0f32; wo * ho * 3];
    let mut wacc = vec![0f32; wo * ho];
    let t = Instant::now();
    drizzle_add_map(&mut acc, &mut wacc, wo, ho, 0..ho, 3, &f, &wt, w, h, &map, 0, 1.4);
    println!("drizzle_add_map: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let mut s = 0.0;
    for y in 0..h {
        for x in 0..w {
            let (rx, ry) = al.frame_to_ref(x as f64, y as f64);
            s += rx + ry;
        }
    }
    println!("frame_to_ref (blot coords): {:.1} ms ({s:.0})", t.elapsed().as_secs_f64() * 1e3);
}
