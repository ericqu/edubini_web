// Run the WebAssembly engine under Node on raw frames written by compare.py:
//   node run_node.mjs ../../results/compare/in.raw
import fs from "node:fs";
const wasmPath = new URL("./target/wasm32-unknown-unknown/release/microscope_engine.wasm", import.meta.url);
const buf = fs.readFileSync(process.argv[2]);
const dv = new DataView(buf.buffer, buf.byteOffset);
const [w, h, n] = [dv.getUint32(4, true), dv.getUint32(8, true), dv.getUint32(12, true)];
let mem;
const stages = {};
let last = [0, performance.now()];
const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath), {
  env: {
    js_progress(stage) {
      if (stage !== last[0]) { stages[last[0]] = ((performance.now() - last[1]) / 1000).toFixed(2); last = [stage, performance.now()]; }
    },
    js_log(p, len) { console.error(new TextDecoder().decode(new Uint8Array(mem.buffer, p, len))); },
  },
});
const e = instance.exports; mem = e.memory;
const t0 = performance.now();
e.mm_begin(w, h);
const fp = e.mm_alloc(w * h * 4);
for (let k = 0; k < n; k++) {
  const src = buf.subarray(16 + k * w * h * 3, 16 + (k + 1) * w * h * 3);
  const dst = new Uint8Array(mem.buffer, fp, w * h * 4);
  for (let i = 0, j = 0; i < w * h * 3; i += 3, j += 4) { dst[j] = src[i]; dst[j + 1] = src[i + 1]; dst[j + 2] = src[i + 2]; dst[j + 3] = 255; }
  e.mm_add_frame(fp);
}
const tScan = (performance.now() - t0) / 1000;
last = [0, performance.now()];
const rc = e.mm_process(2, 0.7, 1, 1, 1);
stages[last[0]] = ((performance.now() - last[1]) / 1000).toFixed(2);
const text = new TextDecoder().decode(new Uint8Array(mem.buffer, e.mm_text_ptr(), e.mm_text_len()));
console.log(rc === 0 ? "ok" : "error", text);
console.log(`scan ${tScan.toFixed(2)}s, stages`, stages, `total ${((performance.now() - t0) / 1000).toFixed(1)}s, wasm memory ${(mem.buffer.byteLength / 2 ** 20).toFixed(0)} MB`);
