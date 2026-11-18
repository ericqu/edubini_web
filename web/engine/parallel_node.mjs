// Check the step-wise/helper exports against mm_process on the same frames
// (same steps as web/worker.js, run synchronously in one thread):
//   node parallel_node.mjs ../../results/compare/in.raw [helpers]
import fs from "node:fs";

const wasm = fs.readFileSync(new URL("../engine.wasm", import.meta.url));
const buf = fs.readFileSync(process.argv[2]);
const K = +(process.argv[3] || 4);
const dv = new DataView(buf.buffer, buf.byteOffset);
const [w, h, nf] = [dv.getUint32(4, true), dv.getUint32(8, true), Math.min(dv.getUint32(12, true), 40)];

async function instance() {
  let mem;
  const { instance } = await WebAssembly.instantiate(wasm, {
    env: { js_progress() {}, js_log(p, l) { console.error(new TextDecoder().decode(new Uint8Array(mem.buffer, p, l))); } },
  });
  mem = instance.exports.memory;
  return instance.exports;
}
const alloc = (ex, bytes) => { const p = ex.mm_alloc(bytes + 8); return p + ((8 - (p % 8)) % 8); };

function loadFrames(ex) {
  ex.mm_begin(w, h);
  const fp = alloc(ex, w * h * 4);
  for (let k = 0; k < nf; k++) {
    const src = buf.subarray(16 + k * w * h * 3, 16 + (k + 1) * w * h * 3);
    const dst = new Uint8Array(ex.memory.buffer, fp, w * h * 4);
    for (let i = 0, j = 0; i < w * h * 3; i += 3, j += 4) { dst[j] = src[i]; dst[j + 1] = src[i + 1]; dst[j + 2] = src[i + 2]; dst[j + 3] = 255; }
    ex.mm_add_frame(fp);
  }
}
const result = (ex) => new Uint8Array(ex.memory.buffer, ex.mm_result_ptr(), ex.mm_result_w() * ex.mm_result_h() * 4).slice();
const diag = (ex) => new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ex.mm_text_ptr(), ex.mm_text_len()));

// reference: single-threaded
const a = await instance();
loadFrames(a);
let t0 = performance.now();
a.mm_process(2, 0.7, 1, 1, 1);
console.log(`single: ${((performance.now() - t0) / 1000).toFixed(1)} s`, diag(a));
const ref = result(a);

// step-wise with K helper instances
const e = await instance();
loadFrames(e);
const hs = [];
for (let k = 0; k < K; k++) hs.push(await instance());
t0 = performance.now();
const f64 = (ex, p, n) => new Float64Array(ex.memory.buffer, p, n).slice();
const f32 = (ex, p, n) => new Float32Array(ex.memory.buffer, p, n).slice();
const n = e.mm_plan(2, 0.7, 1, 1, 1);
const plan = f64(e, e.mm_buf_ptr(), e.mm_buf_len());
const [refRow, levels, wo, ho] = plan;
const scratch = alloc(e, 8 * 8192);
const H = hs.map((x, k) => {
  const r0 = Math.floor((ho * k) / K), r1 = Math.floor((ho * (k + 1)) / K);
  x.mm_h_init(w, h, 2, 0.7, 1);
  x.mm_h_band(r0, r1);
  return { x, r0, r1, rgb: alloc(x, w * h * 3), gray: alloc(x, w * h * 4), al: alloc(x, 8 * 8192) };
});
const frame = (row) => new Uint8Array(e.memory.buffer, e.mm_frame_ptr(row), w * h * 3).slice();
const setRef = (gray, pass) => H.forEach((hp) => { new Float32Array(hp.x.memory.buffer, hp.gray, w * h).set(gray); hp.x.mm_h_ref(hp.gray, levels, pass); });
setRef(f32(e, e.mm_ref_gray(), w * h), 1);
e.mm_p1_add(refRow, 1, 1, 0, 0);
for (let k = 0; k < n; k++) {
  const [row, tx, ty] = [plan[5 + 3 * k], plan[6 + 3 * k], plan[7 + 3 * k]];
  if (row === refRow) continue;
  const hp = H[k % K];
  new Uint8Array(hp.x.memory.buffer, hp.rgb, w * h * 3).set(frame(row));
  const ok = hp.x.mm_h_align(hp.rgb, tx, ty, hp.al);
  const out = f64(hp.x, hp.x.mm_h_out_ptr(), hp.x.mm_h_out_len());
  new Float64Array(e.memory.buffer, scratch, out.length - 2).set(out.subarray(2));
  e.mm_p1_add(row, ok, out[0] || 0, scratch, Math.max(0, out.length - 2));
}
setRef(f32(e, e.mm_p1_finish(), w * h), 2);
const rb = f64(e, e.mm_buf_ptr(), e.mm_buf_len());
const rows = Array.from(rb.subarray(1, 1 + rb[0]));
rows.forEach((row, k) => {
  const hp = H[k % K];
  new Uint8Array(hp.x.memory.buffer, hp.rgb, w * h * 3).set(frame(row));
  new Float64Array(hp.x.memory.buffer, hp.al, 9).set(f64(e, e.mm_init_w(row), 9));
  const ok = hp.x.mm_h_align(hp.rgb, 0, 0, hp.al);
  const out = f64(hp.x, hp.x.mm_h_out_ptr(), hp.x.mm_h_out_len());
  new Float64Array(e.memory.buffer, scratch, out.length - 2).set(out.subarray(2));
  e.mm_p2_set(row, ok, out[1] || 0, scratch, Math.max(0, out.length - 2));
  const alLen = e.mm_alignment(row);
  const al = f64(e, e.mm_buf_ptr(), alLen);
  const halfB = e.mm_half_b(row);
  if (process.env.DEBUG && k < 2) console.log("row", row, "ok", ok, "out[0..3]", Array.from(out.subarray(0, 4)), "al", al.length, Array.from(al.subarray(0, 14)).map((v) => +v.toFixed(4)));
  const rgb = frame(row);
  for (const q of H) {
    new Uint8Array(q.x.memory.buffer, q.rgb, w * h * 3).set(rgb);
    new Float64Array(q.x.memory.buffer, q.al, al.length).set(al);
    q.x.mm_h_drizzle(q.rgb, q.al, al.length, halfB);
    if (process.env.DEBUG && k < 2) {
      const wa = new Float32Array(q.x.memory.buffer, q.x.mm_h_band_ptr(halfB ? 3 : 2), q.x.mm_h_band_len(2));
      let sm = 0; for (const v of wa) sm += v;
      const rgbSum = new Uint8Array(q.x.memory.buffer, q.rgb, w * h * 3).reduce((a, b) => a + b, 0);
      console.log(`after drizzle row ${row} helper ${q.r0}: w sum ${sm}, rgb sum ${rgbSum}, al[0..2]`, Array.from(new Float64Array(q.x.memory.buffer, q.al, 3)));
    }
  }
});
e.mm_acc_alloc();
const per = [3, 3, 1, 1];
for (const hp of H) {
  for (let j = 0; j < 4; j++) {
    const part = f32(hp.x, hp.x.mm_h_band_ptr(j), hp.x.mm_h_band_len(j));
    if (process.env.DEBUG) { let s = 0; for (const v of part) s += v; console.log(`band ${hp.r0}-${hp.r1} part ${j}: len ${part.length} sum ${s.toExponential(3)}`); }
    new Float32Array(e.memory.buffer, e.mm_acc_ptr(j) + hp.r0 * wo * per[j] * 4, part.length).set(part);
  }
}
e.mm_p2_finish();
e.mm_p3();
e.mm_finish();
console.log(`steps x${K}: ${((performance.now() - t0) / 1000).toFixed(1)} s (sequential here)`, diag(e));
const out = result(e);
let se = 0;
for (let i = 0; i < out.length; i++) if (i % 4 !== 3) se += (out[i] - ref[i]) ** 2;
const mse = se / (out.length * 0.75);
console.log(`PSNR steps vs single: ${(10 * Math.log10(255 ** 2 / mse)).toFixed(2)} dB`);
