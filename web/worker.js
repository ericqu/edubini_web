// Engine worker: keeps the recorded frames and runs the super-resolution
// pipeline (engine.wasm). When the page provides helper workers, registration
// and drizzle are spread over them (each helper drizzles one horizontal band
// of the output); otherwise everything runs here.
//
// main -> worker: {type: "helpers", ports} | {type: "begin", w, h} | {type: "frame", buf} | {type: "process", opts}
// worker -> main: {type: "ready"} | {type: "added", kept, n} | {type: "progress", stage, done, total}
//                 {type: "result", w, h, rgba, diag, refW, refH, ref} | {type: "error", message} | {type: "log", text}

let ex = null;
let w = 0, h = 0, framePtr = 0, scratchPtr = 0;
const SCRATCH_F64 = 8192;
const decoder = new TextDecoder();
let helpers = [];

const text = (ptr, len) => decoder.decode(new Uint8Array(ex.memory.buffer, ptr, len).slice());
const progress = (stage, done, total) => postMessage({ type: "progress", stage, done, total });

const ready = (async () => {
  const bytes = await (await fetch("engine.wasm", { cache: "no-cache" })).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: {
      js_progress: progress,
      js_log: (ptr, len) => postMessage({ type: "log", text: text(ptr, len) }),
    },
  });
  ex = instance.exports;
  postMessage({ type: "ready" });
})();

// ---------------------------------------------------------------- helpers
let nextId = 1;
function setHelpers(ports) {
  helpers = ports.map((port) => {
    const hp = { port, inflight: 0, pending: new Map(), onDrizzled: null };
    port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "drizzled") { hp.onDrizzled && hp.onDrizzled(m.row); return; }
      const p = hp.pending.get(m.id);
      if (!p) {
        if (m.type === "error") { // e.g. a failed drizzle: fail everything waiting on this helper
          hp.failed = m.message;
          for (const q of hp.pending.values()) q.reject(new Error("helper: " + m.message));
          hp.pending.clear();
        }
        return;
      }
      hp.pending.delete(m.id);
      if (m.type === "error") p.reject(new Error("helper: " + m.message)); else p.resolve(m);
    };
    return hp;
  });
}
function request(hp, msg, transfer = []) {
  const id = nextId++;
  if (hp.failed) return Promise.reject(new Error("helper: " + hp.failed));
  return new Promise((resolve, reject) => {
    hp.pending.set(id, { resolve, reject });
    hp.port.postMessage({ ...msg, id }, transfer);
  });
}
/** Run `items` on the helpers (at most 2 in flight each); onResult(item, reply) runs here in order of completion. */
function runPool(items, makeMsg, onResult) {
  return new Promise((resolve, reject) => {
    let next = 0, outstanding = 0, failed = false;
    if (!items.length) return resolve();
    const pump = () => {
      for (const hp of helpers) {
        while (!failed && hp.inflight < 2 && next < items.length) {
          const item = items[next++];
          const [msg, transfer] = makeMsg(item);
          hp.inflight++;
          outstanding++;
          request(hp, msg, transfer).then((r) => {
            hp.inflight--;
            outstanding--;
            onResult(item, r);
            if (next >= items.length && outstanding === 0) resolve(); else pump();
          }).catch((e) => { failed = true; reject(e); });
        }
      }
    };
    pump();
  });
}

// ---------------------------------------------------------------- memory helpers
const f64 = (ptr, n) => new Float64Array(ex.memory.buffer, ptr, n).slice();
const f32 = (ptr, n) => new Float32Array(ex.memory.buffer, ptr, n).slice();
const frameCopy = (row) => new Uint8Array(ex.memory.buffer, ex.mm_frame_ptr(row), w * h * 3).slice();
function scratch(arr) {
  new Float64Array(ex.memory.buffer, scratchPtr, arr.length).set(arr);
  return scratchPtr;
}
function alloc(bytes, align) {
  const p = ex.mm_alloc(bytes + align);
  return p + ((align - (p % align)) % align);
}

// ---------------------------------------------------------------- processing
async function processParallel(o) {
  const n = ex.mm_plan(o.scale, o.pixfrac, o.local ? 1 : 0, o.reject ? 1 : 0, o.deconvolve ? 1 : 0);
  if (n < 0) throw new Error(text(ex.mm_text_ptr(), ex.mm_text_len()));
  const plan = f64(ex.mm_buf_ptr(), ex.mm_buf_len());
  const [refRow, levels, wo, ho] = plan;
  const chosen = [];
  for (let k = 0; k < n; k++) chosen.push({ row: plan[5 + 3 * k], tx: plan[6 + 3 * k], ty: plan[7 + 3 * k] });
  const K = helpers.length;
  helpers.forEach((hp, k) => {
    const r0 = Math.floor((ho * k) / K), r1 = Math.floor((ho * (k + 1)) / K);
    hp.port.postMessage({ type: "init", w, h, scale: o.scale, pixfrac: o.pixfrac, local: o.local, r0, r1 });
    hp.band = [r0, r1];
  });

  // Pass 1: register every frame against the reference frame
  const refGray = f32(ex.mm_ref_gray(), w * h);
  helpers.forEach((hp) => hp.port.postMessage({ type: "ref", gray: refGray, levels, pass: 1 }));
  ex.mm_p1_add(refRow, 1, 1.0, 0, 0);
  const others = chosen.filter((c) => c.row !== refRow);
  let done = 0;
  progress(1, 0, others.length);
  await runPool(others, (c) => {
    const rgb = frameCopy(c.row);
    return [{ type: "align", rgb, tx: c.tx, ty: c.ty }, [rgb.buffer]];
  }, (c, r) => {
    if (r.ok) ex.mm_p1_add(c.row, 1, r.out[0], scratch(r.out.subarray(2)), r.out.length - 2);
    else ex.mm_p1_add(c.row, 0, 0, 0, 0);
    progress(1, ++done, others.length);
  });

  // Pass 2: re-register against the stacked reference; every helper drizzles
  // each frame into its band
  const refImg = f32(ex.mm_p1_finish(), w * h);
  const rowsBuf = f64(ex.mm_buf_ptr(), ex.mm_buf_len());
  const rows = Array.from(rowsBuf.subarray(1, 1 + rowsBuf[0]));
  helpers.forEach((hp) => hp.port.postMessage({ type: "ref", gray: refImg, levels, pass: 2 }));
  let acks = 0;
  const total = rows.length * K;
  progress(2, 0, total);
  helpers.forEach((hp) => (hp.onDrizzled = () => progress(2, ++acks, total)));
  await runPool(rows, (row) => {
    const rgb = frameCopy(row);
    return [{ type: "align", rgb, init: f64(ex.mm_init_w(row), 9) }, [rgb.buffer]];
  }, (row, r) => {
    if (r.ok) ex.mm_p2_set(row, 1, r.out[1], scratch(r.out.subarray(2)), r.out.length - 2);
    else ex.mm_p2_set(row, 0, 0, 0, 0);
    const alLen = ex.mm_alignment(row); // (re)fills mm_buf: call before mm_buf_ptr
    const al = f64(ex.mm_buf_ptr(), alLen);
    const msg = { type: "drizzle", row, rgb: frameCopy(row), al, halfB: ex.mm_half_b(row) };
    helpers.forEach((hp) => hp.port.postMessage(msg)); // cloned for each helper
  });

  // Gather the bands (each helper answers after its queued drizzles)
  const bands = await Promise.all(helpers.map((hp) => request(hp, { type: "band" })));
  ex.mm_acc_alloc();
  bands.forEach((b, k) => {
    const [r0] = helpers[k].band;
    const per = [3, 3, 1, 1];
    b.parts.forEach((part, j) => {
      new Float32Array(ex.memory.buffer, ex.mm_acc_ptr(j) + r0 * wo * per[j] * 4, part.length).set(part);
    });
  });
  helpers.forEach((hp) => hp.port.postMessage({ type: "reset" }));
  ex.mm_p2_finish();
  ex.mm_p3();
  ex.mm_finish();
}

async function handle(m) {
  if (m.type === "helpers") {
    setHelpers(m.ports);
  } else if (m.type === "begin") {
    if (m.w !== w || m.h !== h || !framePtr) { // scratch buffers, kept across runs
      framePtr = alloc(m.w * m.h * 4, 8);
      if (!scratchPtr) scratchPtr = alloc(SCRATCH_F64 * 8, 8);
    }
    w = m.w; h = m.h;
    helpers.forEach((hp) => { hp.failed = null; hp.inflight = 0; hp.pending.clear(); });
    ex.mm_begin(w, h);
  } else if (m.type === "frame") {
    new Uint8Array(ex.memory.buffer, framePtr, w * h * 4).set(new Uint8Array(m.buf));
    const kept = ex.mm_add_frame(framePtr);
    postMessage({ type: "added", kept, n: ex.mm_frames() });
  } else if (m.type === "process") {
    const o = Object.assign({ scale: 2, pixfrac: 0.7, local: true, reject: true, deconvolve: true }, m.opts);
    const t0 = performance.now();
    if (helpers.length) {
      await processParallel(o);
    } else if (ex.mm_process(o.scale, o.pixfrac, o.local ? 1 : 0, o.reject ? 1 : 0, o.deconvolve ? 1 : 0) !== 0) {
      throw new Error(text(ex.mm_text_ptr(), ex.mm_text_len()));
    }
    const diag = JSON.parse(text(ex.mm_text_ptr(), ex.mm_text_len()));
    diag.seconds = (performance.now() - t0) / 1000;
    diag.helpers = helpers.length;
    const rw = ex.mm_result_w(), rh = ex.mm_result_h();
    const rgba = new Uint8Array(ex.memory.buffer, ex.mm_result_ptr(), rw * rh * 4).slice().buffer;
    let ref = null;
    if (ex.mm_frame_rgba(diag.reference, framePtr)) {
      ref = new Uint8Array(ex.memory.buffer, framePtr, w * h * 4).slice().buffer;
    }
    ex.mm_reset();
    diag.wasm_mb = Math.round(ex.memory.buffer.byteLength / 2 ** 20);
    postMessage({ type: "result", w: rw, h: rh, rgba, diag, refW: w, refH: h, ref }, ref ? [rgba, ref] : [rgba]);
  }
}

// Handle messages strictly in order (the first ones may arrive before the
// engine has loaded).
let chain = ready;
onmessage = (e) => {
  const m = e.data;
  chain = chain.then(() => handle(m)).catch((err) => {
    if (ex) ex.mm_reset();
    postMessage({ type: "error", message: String((err && err.message) || err) });
  });
};
