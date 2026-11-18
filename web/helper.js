// Helper worker: registers frames and drizzles one horizontal band of the
// output for the engine worker (its own engine.wasm instance). It talks to the
// engine worker over a MessagePort handed over by the page.
//
// engine -> helper: init | ref | align (reply "aligned") | drizzle (ack "drizzled") | band (reply "band") | reset

let ex = null;
let w = 0, h = 0, rgbPtr = 0, grayPtr = 0, alPtr = 0;
const AL_CAP = 8192; // f64 slots for an encoded alignment

const ready = (async () => {
  const bytes = await (await fetch("engine.wasm", { cache: "no-cache" })).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: {
      js_progress: () => {},
      js_log: (ptr, len) => console.log("[helper]", new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ptr, len).slice())),
    },
  });
  ex = instance.exports;
})();

function alloc(bytes, align) {
  const p = ex.mm_alloc(bytes + align);
  return p + ((align - (p % align)) % align);
}
const putU8 = (src) => new Uint8Array(ex.memory.buffer, rgbPtr, w * h * 3).set(src);
const putF64 = (src) => new Float64Array(ex.memory.buffer, alPtr, src.length).set(src);

function handle(m, port) {
  switch (m.type) {
    case "init":
      if (m.w !== w || m.h !== h || !rgbPtr) { // scratch buffers, kept across runs
        rgbPtr = alloc(m.w * m.h * 3, 8);
        grayPtr = alloc(m.w * m.h * 4, 8);
        if (!alPtr) alPtr = alloc(AL_CAP * 8, 8);
      }
      w = m.w; h = m.h;
      ex.mm_h_init(w, h, m.scale, m.pixfrac, m.local ? 1 : 0);
      ex.mm_h_band(m.r0, m.r1);
      break;
    case "ref":
      new Float32Array(ex.memory.buffer, grayPtr, w * h).set(m.gray);
      ex.mm_h_ref(grayPtr, m.levels, m.pass);
      break;
    case "align": {
      putU8(m.rgb);
      if (m.init) putF64(m.init);
      const ok = ex.mm_h_align(rgbPtr, m.tx || 0, m.ty || 0, alPtr);
      const out = new Float64Array(ex.memory.buffer, ex.mm_h_out_ptr(), ex.mm_h_out_len()).slice();
      port.postMessage({ type: "aligned", id: m.id, ok, out }, [out.buffer]);
      break;
    }
    case "drizzle":
      putU8(m.rgb);
      putF64(m.al);
      ex.mm_h_drizzle(rgbPtr, alPtr, m.al.length, m.halfB);
      port.postMessage({ type: "drizzled", row: m.row });
      break;
    case "band": {
      const parts = [0, 1, 2, 3].map((k) => new Float32Array(ex.memory.buffer, ex.mm_h_band_ptr(k), ex.mm_h_band_len(k)).slice());
      port.postMessage({ type: "band", id: m.id, parts }, parts.map((p) => p.buffer));
      break;
    }
    case "reset":
      ex.mm_h_reset();
      break;
  }
}

onmessage = (e) => {
  if (e.data.type !== "port") return;
  const port = e.data.port;
  let chain = ready;
  port.onmessage = (ev) => {
    const m = ev.data;
    chain = chain.then(() => handle(m, port)).catch((err) => {
      port.postMessage({ type: "error", id: m.id, message: String((err && err.message) || err) });
    });
  };
};
