"use strict";
// Microscope web app: live view, photo, and "super photo" (multi-frame
// super-resolution by engine.wasm in worker.js). Photos are kept in IndexedDB.

const VERSION = "0.1.0";
const WANT_W = 1280, WANT_H = 720; // the microscope only delivers frames in an explicitly requested mode
const $ = (id) => document.getElementById(id);
const video = $("live");

// ----------------------------------------------------------------- settings
const settings = Object.assign({ camId: null, duration: 5, details: false }, (() => {
  try { return JSON.parse(localStorage.getItem("settings") || "{}"); } catch { return {}; }
})());
settings.duration = Math.min(8, Math.max(4, Math.round(settings.duration))); // recording: 4..8 s
// Frames kept per super photo: longer recordings keep every 2nd/3rd frame, so
// memory and processing time stay what the iPads handled at 3 s (~70 frames).
const MAX_FRAMES = 90;
let camFps = 23;
function saveSettings() {
  try { localStorage.setItem("settings", JSON.stringify(settings)); } catch { /* private mode */ }
}

// ----------------------------------------------------------------- UI helpers
function setStatus(text, cls = "") {
  $("status").className = cls;
  $("status").querySelector("span").textContent = text;
}
let toastTimer = null;
function toast(text, ms = 2500) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), ms);
}
function message(title, text, buttons = [{ label: "OK" }]) {
  $("msgTitle").textContent = title;
  $("msgText").textContent = text;
  const row = $("msgButtons");
  row.innerHTML = "";
  for (const b of buttons) {
    const el = document.createElement("button");
    el.className = "btn" + (b.primary ? " primary" : "");
    el.textContent = b.label;
    el.onclick = () => { $("msg").hidden = true; b.action && b.action(); };
    row.appendChild(el);
  }
  $("msg").hidden = false;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function stamp(d = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}
function niceDate(d) {
  return new Date(d).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

// ----------------------------------------------------------------- storage
const dbp = new Promise((resolve, reject) => {
  const r = indexedDB.open("microscope", 1);
  r.onupgradeneeded = () => r.result.createObjectStore("photos", { keyPath: "id" });
  r.onsuccess = () => resolve(r.result);
  r.onerror = () => reject(r.error);
});
async function tx(mode, fn) {
  const db = await dbp;
  return new Promise((resolve, reject) => {
    const t = db.transaction("photos", mode);
    const req = fn(t.objectStore("photos"));
    t.oncomplete = () => resolve(req && req.result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || new Error("storage aborted (disk full?)"));
  });
}
const putPhoto = (rec) => tx("readwrite", (s) => s.put(rec));
const delPhoto = (id) => tx("readwrite", (s) => s.delete(id));
const allPhotos = async () => ((await tx("readonly", (s) => s.getAll())) || []).sort((a, b) => b.id - a.id);
if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});

// ----------------------------------------------------------------- camera
let stream = null, camReady = false, opening = false, busy = false;

async function cameras() {
  return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
}
const looksBuiltIn = (label) => /front|back|facetime|face time|ultra|wide|tele|truedepth|desk view|built-in|integrated/i.test(label);

function stopCamera() {
  camReady = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  updateButtons();
}

/** Resolves true when the next video frame arrives within `ms`. */
function nextFrame(ms) {
  return new Promise((resolve) => {
    let done = false;
    const timer = setTimeout(() => { done = true; resolve(false); }, ms);
    video.requestVideoFrameCallback(() => { if (!done) { clearTimeout(timer); resolve(true); } });
  });
}

async function measureFps(ms = 1500) {
  let n = 0, t0 = null, t1 = null;
  const end = performance.now() + ms;
  await new Promise((resolve) => {
    const cb = (now) => {
      if (t0 === null) t0 = now; else n++;
      t1 = now;
      if (performance.now() < end && stream) video.requestVideoFrameCallback(cb); else resolve();
    };
    video.requestVideoFrameCallback(cb);
  });
  return n > 0 ? (n * 1000) / (t1 - t0) : 0;
}

async function openCamera() {
  if (opening) return;
  opening = true;
  stopCamera();
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser cannot use cameras here (the page must be opened over https).");
    }
    setStatus("Starting camera…");
    let cams = await cameras();
    if (!cams.some((c) => c.label)) { // labels appear only once camera access is granted
      const s = await navigator.mediaDevices.getUserMedia({ video: true });
      s.getTracks().forEach((t) => t.stop());
      cams = await cameras();
    }
    fillCameraList(cams);
    const cam = cams.find((c) => c.deviceId === settings.camId) || cams.find((c) => !looksBuiltIn(c.label));
    if (!cam) {
      setStatus("No microscope", "bad");
      message("Plug in the microscope", "Connect the microscope to the iPad's USB-C port. It will start automatically.",
        [{ label: "Try again", primary: true, action: openCamera }]);
      return;
    }
    for (let attempt = 1; attempt <= 5; attempt++) {
      // The microscope delivers no frames in its default mode: ask for 1280x720
      // exactly; fall back to "ideal" in case a different camera is used.
      const exact = attempt <= 3;
      const size = exact ? { width: { exact: WANT_W }, height: { exact: WANT_H } } : { width: { ideal: WANT_W }, height: { ideal: WANT_H } };
      setStatus(attempt === 1 ? "Starting microscope…" : `Starting microscope (try ${attempt})…`);
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: cam.deviceId }, ...size } });
      } catch (e) {
        stream = null;
        if (e.name === "NotAllowedError") throw e;
        await sleep(400);
        continue;
      }
      video.srcObject = stream;
      try { await video.play(); } catch { /* autoplay is muted, normally fine */ }
      if (await nextFrame(3500)) {
        const track = stream.getVideoTracks()[0];
        track.onended = () => { if (!busy) { stopCamera(); setStatus("Microscope disconnected", "bad"); } };
        camReady = true;
        updateButtons();
        setStatus(`Microscope ready · ${video.videoWidth}×${video.videoHeight}`, "ok");
        measureFps().then((fps) => {
          if (fps > 1) camFps = fps;
          report("camera", { fps, size: `${video.videoWidth}x${video.videoHeight}`, label: track.label });
          if (camReady) setStatus(`Microscope ready · ${video.videoWidth}×${video.videoHeight} · ${fps.toFixed(0)} fps`, "ok");
        });
        return;
      }
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      await sleep(400);
    }
    setStatus("Microscope not responding", "bad");
    message("The microscope is not sending pictures", "Unplug it, plug it back in, and tap Try again.",
      [{ label: "Try again", primary: true, action: openCamera }]);
  } catch (e) {
    setStatus("Camera error", "bad");
    const denied = e && e.name === "NotAllowedError";
    message(denied ? "Camera access is off" : "Camera problem",
      denied ? "Allow camera access for this app: Settings › Apps › Safari › Camera (or tap Try again)." : String(e.message || e),
      [{ label: "Try again", primary: true, action: openCamera }]);
  } finally {
    opening = false;
  }
}

let devTimer = null;
navigator.mediaDevices && navigator.mediaDevices.addEventListener("devicechange", () => {
  clearTimeout(devTimer);
  devTimer = setTimeout(() => {
    if (busy) return;
    const live = stream && stream.getVideoTracks().some((t) => t.readyState === "live");
    if (!live) { $("msg").hidden = true; openCamera(); }
    else cameras().then(fillCameraList);
  }, 800);
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible" || busy) return;
  const live = stream && stream.getVideoTracks().some((t) => t.readyState === "live");
  if (!live) openCamera();
});

function updateButtons() {
  $("btnPhoto").disabled = !camReady || busy;
  $("btnSuper").disabled = !camReady || busy;
}

// ----------------------------------------------------------------- capture helpers
const grabCanvas = $("grab");
const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });
function grab() {
  const w = video.videoWidth, h = video.videoHeight;
  if (grabCanvas.width !== w || grabCanvas.height !== h) { grabCanvas.width = w; grabCanvas.height = h; }
  grabCtx.drawImage(video, 0, 0, w, h);
  return grabCtx.getImageData(0, 0, w, h);
}
function canvasOf(imageData) {
  const c = document.createElement("canvas");
  c.width = imageData.width;
  c.height = imageData.height;
  c.getContext("2d").putImageData(imageData, 0, 0);
  return c;
}
const toBlob = (canvas, type = "image/png", q) => new Promise((resolve, reject) =>
  canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("could not encode the image"))), type, q));
async function thumbOf(canvas) {
  const t = document.createElement("canvas");
  t.width = 400;
  t.height = Math.round((400 * canvas.height) / canvas.width);
  t.getContext("2d").drawImage(canvas, 0, 0, t.width, t.height);
  return toBlob(t, "image/jpeg", 0.8);
}
function flash() {
  const f = document.createElement("div");
  f.style.cssText = "position:fixed;inset:0;background:#fff;opacity:.7;transition:opacity .35s;pointer-events:none";
  document.body.appendChild(f);
  requestAnimationFrame(() => requestAnimationFrame(() => (f.style.opacity = "0")));
  setTimeout(() => f.remove(), 400);
}

// ----------------------------------------------------------------- photo
async function takePhoto() {
  if (!camReady || busy) return;
  busy = true;
  updateButtons();
  try {
    flash();
    const c = canvasOf(grab());
    const now = new Date();
    const rec = {
      id: now.getTime(), kind: "photo", created: now.toISOString(), w: c.width, h: c.height,
      name: `microscope-${stamp(now)}-photo.png`, blob: await toBlob(c), thumb: await thumbOf(c), ref: null, diag: null,
    };
    await putPhoto(rec);
    refreshLastThumb();
    toast("Photo saved in the gallery");
  } catch (e) {
    message("Could not save the photo", String(e.message || e));
  } finally {
    busy = false;
    updateButtons();
  }
}

// ----------------------------------------------------------------- super photo
let worker = null;
let helpers = [];
let job = null;

// Helper workers share the registration and drizzle (one per spare core,
// at most 4; "?helpers=N" in the URL overrides, 0 = everything in one worker).
function helperCount() {
  const q = new URLSearchParams(location.search).get("helpers");
  if (q !== null) return Math.max(0, Math.min(8, +q || 0));
  return Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 3) - 1));
}

function getWorker() {
  if (worker) return worker;
  worker = new Worker("worker.js");
  worker.onmessage = (e) => job && job.onMessage(e.data);
  worker.onerror = (e) => job && job.fail(new Error(e.message || "engine crashed"));
  const ports = [];
  for (let k = 0; k < helperCount(); k++) {
    const hw = new Worker("helper.js");
    const ch = new MessageChannel();
    hw.postMessage({ type: "port", port: ch.port2 }, [ch.port2]);
    hw.onerror = (e) => job && job.fail(new Error(e.message || "helper crashed"));
    ports.push(ch.port1);
    helpers.push(hw);
  }
  worker.postMessage({ type: "helpers", ports }, ports);
  return worker;
}
function killWorker() {
  if (worker) worker.terminate();
  helpers.forEach((hw) => hw.terminate());
  worker = null;
  helpers = [];
}

// progress bar: frames still being analysed, then the engine's stages
const STAGES = { 1: ["Lining up the frames", 0.12, 0.4], 2: ["Combining the frames", 0.4, 0.8], 3: ["Removing glitches", 0.8, 0.93], 4: ["Sharpening", 0.93, 1.0] };
function setBusy(stage, frac) {
  $("busyStage").textContent = stage;
  $("busyBar").style.width = `${Math.round(100 * Math.min(1, Math.max(0, frac)))}%`;
}

async function superPhoto() {
  if (!camReady || busy) return;
  busy = true;
  updateButtons();
  const w = video.videoWidth, h = video.videoHeight;
  const thisJob = {
    cancelled: false, sent: 0, added: 0, kept: 0,
    onMessage(m) {
      if (m.type === "added") {
        this.added++;
        this.kept = m.n;
        if (this.capturing === false && !this.processing) setBusy(`Looking at the frames (${this.added}/${this.sent})`, 0.12 * this.added / this.sent);
      } else if (m.type === "progress") {
        this.processing = true;
        const [label, a, b] = STAGES[m.stage] || ["Working", 0, 1];
        setBusy(label, a + (b - a) * (m.total ? m.done / m.total : 0));
      } else if (m.type === "result") this.done(m);
      else if (m.type === "error") this.fail(new Error(m.message));
      else if (m.type === "log") console.log("[engine]", m.text);
    },
  };
  const result = new Promise((resolve, reject) => { thisJob.done = resolve; thisJob.fail = reject; });
  job = thisJob;
  try {
    const wk = getWorker();
    wk.postMessage({ type: "begin", w, h });

    // 1. record: every new camera frame goes to the engine
    $("hold").hidden = false;
    thisJob.capturing = true;
    const arc = $("ringArc");
    const dur = settings.duration * 1000;
    const stride = Math.max(1, Math.ceil((settings.duration * camFps) / MAX_FRAMES));
    let seen = 0;
    await new Promise((resolve, reject) => {
      let t0 = null;
      let watchdog = setTimeout(() => reject(new Error("The microscope stopped sending pictures.")), 3000);
      const step = (now) => {
        if (thisJob.cancelled) return resolve();
        clearTimeout(watchdog);
        watchdog = setTimeout(() => reject(new Error("The microscope stopped sending pictures.")), 3000);
        if (t0 === null) t0 = now;
        if (seen++ % stride === 0) {
          const img = grab();
          wk.postMessage({ type: "frame", buf: img.data.buffer }, [img.data.buffer]);
          thisJob.sent++;
        }
        const el = now - t0;
        arc.style.strokeDashoffset = String(314.16 * (1 - Math.min(1, el / dur)));
        if (el >= dur) { clearTimeout(watchdog); resolve(); } else video.requestVideoFrameCallback(step);
      };
      video.requestVideoFrameCallback(step);
    });
    thisJob.capturing = false;
    $("hold").hidden = true;
    arc.style.strokeDashoffset = "314.16";
    if (thisJob.cancelled) throw new Error("cancelled");

    // 2. process
    setBusy(`Looking at the frames (${thisJob.added}/${thisJob.sent})`, 0);
    $("busy").hidden = false;
    wk.postMessage({ type: "process", opts: {} });
    const res = await result;

    // 3. encode and store
    setBusy("Saving", 1);
    const c = canvasOf(new ImageData(new Uint8ClampedArray(res.rgba), res.w, res.h));
    const now = new Date();
    let ref = null;
    if (res.ref) {
      const rc = canvasOf(new ImageData(new Uint8ClampedArray(res.ref), res.refW, res.refH));
      ref = await toBlob(rc);
      rc.width = rc.height = 0;
    }
    const rec = {
      id: now.getTime(), kind: "super", created: now.toISOString(), w: res.w, h: res.h,
      name: `microscope-${stamp(now)}-super.png`, blob: await toBlob(c), thumb: await thumbOf(c), ref,
      diag: Object.assign(res.diag, { sent: thisJob.sent, record_s: settings.duration, stride }),
    };
    c.width = c.height = 0;
    await putPhoto(rec);
    refreshLastThumb();
    $("busy").hidden = true;
    openViewer(rec, true);
    report("super", { diag: rec.diag, fps: camFps });
  } catch (e) {
    $("hold").hidden = true;
    $("busy").hidden = true;
    if (!thisJob.cancelled) {
      killWorker(); // start fresh next time
      const msg = String(e.message || e);
      report("super-error", { message: msg, sent: thisJob.sent });
      message("The super photo did not work", /distinct|moving/.test(msg)
        ? "The pictures were all identical. Check that the microscope is sending a live image." : msg);
    }
  } finally {
    job = null;
    busy = false;
    updateButtons();
  }
}

$("busyCancel").onclick = () => {
  if (!job) return;
  job.cancelled = true;
  killWorker(); // the engine cannot be interrupted mid-computation
  job.fail(new Error("cancelled"));
  $("busy").hidden = true;
};

// ----------------------------------------------------------------- viewer
const vimg = $("vimg"), vscroll = $("vscroll");
let viewing = null, vurls = [], zoom = 1, showBefore = false, fromGallery = false;

function fitWidth() {
  if (!viewing) return 0;
  const sw = vscroll.clientWidth, sh = vscroll.clientHeight;
  return Math.min(sw / viewing.w, sh / viewing.h) * viewing.w;
}
function applyZoom(z, cx, cy) {
  const old = vimg.getBoundingClientRect();
  const fx = old.width ? ((cx ?? innerWidth / 2) - old.left) / old.width : 0.5;
  const fy = old.height ? ((cy ?? innerHeight / 2) - old.top) / old.height : 0.5;
  zoom = Math.min(8, Math.max(1, z));
  const wpx = fitWidth() * zoom;
  vimg.style.width = `${wpx}px`;
  vimg.style.height = `${(wpx * viewing.h) / viewing.w}px`;
  // keep the touched point under the finger
  const nw = wpx, nh = (wpx * viewing.h) / viewing.w;
  vscroll.scrollLeft = fx * nw - (cx ?? innerWidth / 2);
  vscroll.scrollTop = fy * nh - (cy ?? innerHeight / 2);
}
function noteFor(rec) {
  if (rec.kind !== "super" || !rec.diag) return "";
  const d = rec.diag;
  const parts = [];
  if (d.used < 0.6 * d.distinct) parts.push("Some frames were skipped because the microscope moved too much.");
  if (d.phase_coverage < 0.5) parts.push("Tip: the picture hardly moved. A tiny wiggle while recording gives more detail.");
  if (settings.details) {
    parts.push(`${d.used} of ${d.distinct} frames · sub-pixel coverage ${(100 * d.phase_coverage).toFixed(0)}% · ` +
      `blur σ ${d.deconv ? d.deconv.sigma.toFixed(2) : "–"} · ${d.seconds.toFixed(1)} s · ${d.wasm_mb} MB`);
  }
  return parts.join("\n");
}
function setViewerImage() {
  vurls.forEach(URL.revokeObjectURL);
  vurls = [];
  const blob = showBefore && viewing.ref ? viewing.ref : viewing.blob;
  const u = URL.createObjectURL(blob);
  vurls.push(u);
  vimg.src = u;
  const b = $("badge");
  b.hidden = !viewing.ref;
  b.textContent = showBefore ? "BEFORE · one frame" : "AFTER · super photo";
  $("vbefore").classList.toggle("on", showBefore);
}
function openViewer(rec, fresh = false) {
  viewing = rec;
  showBefore = false;
  fromGallery = !$("gallery").hidden;
  $("viewer").hidden = false;
  $("vtitle").innerHTML = "";
  const t = document.createElement("b");
  t.textContent = rec.kind === "super" ? "Super photo" : "Photo";
  $("vtitle").append(t, `${niceDate(rec.created)} · ${rec.w}×${rec.h}`);
  $("vbefore").hidden = !rec.ref;
  const note = noteFor(rec);
  $("vnote").textContent = note;
  $("vnote").hidden = !note;
  if (note && !settings.details) setTimeout(() => { if (viewing === rec) $("vnote").hidden = true; }, 6000);
  setViewerImage();
  requestAnimationFrame(() => applyZoom(1));
  if (fresh) toast("Saved in the gallery");
}
function closeViewer() {
  $("viewer").hidden = true;
  vurls.forEach(URL.revokeObjectURL);
  vurls = [];
  vimg.removeAttribute("src");
  viewing = null;
  if (fromGallery) openGallery();
}
$("vclose").onclick = closeViewer;
$("vzoom").onclick = () => applyZoom(zoom >= 4 ? 1 : zoom * 2);
vimg.addEventListener("dblclick", (e) => applyZoom(zoom > 1 ? 1 : 3, e.clientX, e.clientY));
let pinchStart = 1;
vscroll.addEventListener("gesturestart", (e) => { e.preventDefault(); pinchStart = zoom; });
vscroll.addEventListener("gesturechange", (e) => { e.preventDefault(); applyZoom(pinchStart * e.scale, e.clientX, e.clientY); });
addEventListener("resize", () => viewing && applyZoom(zoom));
$("vbefore").onclick = () => { showBefore = !showBefore; setViewerImage(); };
$("vshare").onclick = async () => {
  if (!viewing) return;
  const file = new File([viewing.blob], viewing.name, { type: "image/png" });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file] }); } catch (e) { if (e.name !== "AbortError") toast("Could not share: " + e.message); }
  } else {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(file);
    a.download = viewing.name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }
};
let delArmed = null;
$("vdelete").onclick = async () => {
  const btn = $("vdelete");
  if (delArmed !== viewing.id) {
    delArmed = viewing.id;
    btn.textContent = "Tap again to delete";
    btn.classList.add("danger");
    setTimeout(() => { delArmed = null; btn.textContent = "Delete"; btn.classList.remove("danger"); }, 3000);
    return;
  }
  delArmed = null;
  btn.textContent = "Delete";
  btn.classList.remove("danger");
  await delPhoto(viewing.id);
  refreshLastThumb();
  closeViewer();
};

// ----------------------------------------------------------------- gallery
let gurls = [];
async function openGallery() {
  const grid = $("grid");
  gurls.forEach(URL.revokeObjectURL);
  gurls = [];
  grid.innerHTML = "";
  const list = await allPhotos();
  if (!list.length) {
    const p = document.createElement("p");
    p.id = "gempty";
    p.textContent = "No photos yet. Take one with the Photo or Super photo button.";
    grid.appendChild(p);
  }
  for (const rec of list) {
    const tile = document.createElement("button");
    tile.className = "tile";
    const img = document.createElement("img");
    const u = URL.createObjectURL(rec.thumb);
    gurls.push(u);
    img.src = u;
    const badge = document.createElement("em");
    badge.textContent = rec.kind === "super" ? "SUPER" : "PHOTO";
    if (rec.kind === "super") badge.className = "super";
    const label = document.createElement("span");
    label.textContent = niceDate(rec.created);
    tile.append(img, badge, label);
    tile.onclick = () => { openViewer(rec); };
    grid.appendChild(tile);
  }
  $("gallery").hidden = false;
}
$("btnGallery").onclick = openGallery;
$("gclose").onclick = () => { $("gallery").hidden = true; gurls.forEach(URL.revokeObjectURL); gurls = []; };
let lastUrl = null;
async function refreshLastThumb() {
  const list = await allPhotos();
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = list.length ? URL.createObjectURL(list[0].thumb) : null;
  $("lastThumb").style.backgroundImage = lastUrl ? `url(${lastUrl})` : "";
}

// ----------------------------------------------------------------- settings
function fillCameraList(cams) {
  const sel = $("camSel");
  sel.innerHTML = "";
  for (const c of cams) {
    const o = document.createElement("option");
    o.value = c.deviceId;
    o.textContent = c.label || "Camera";
    sel.appendChild(o);
  }
  const active = stream && stream.getVideoTracks()[0];
  const id = active ? active.getSettings().deviceId : settings.camId;
  if (id) sel.value = id;
}
function showDuration() { $("durLabel").textContent = `${settings.duration} s`; }
$("gear").onclick = async () => {
  $("dur").value = settings.duration;
  $("details").checked = settings.details;
  $("ver").textContent = `Version ${VERSION}`;
  showDuration();
  try { fillCameraList(await cameras()); } catch { /* ignore */ }
  $("settings").hidden = false;
};
$("dur").oninput = () => { settings.duration = +$("dur").value; showDuration(); saveSettings(); };
$("details").onchange = () => { settings.details = $("details").checked; saveSettings(); };
$("camSel").onchange = () => { settings.camId = $("camSel").value; saveSettings(); $("settings").hidden = true; openCamera(); };
$("setReload").onclick = () => { $("settings").hidden = true; openCamera(); };
$("setClose").onclick = () => { $("settings").hidden = true; };

// ----------------------------------------------------------------- dev reports
// Only when served by web/serve.py (port 8443 on the Mac): send what device
// this is and how fast it processed, to results/devices.jsonl. Never on GitHub.
const DEV_REPORTS = location.port === "8443";
function report(kind, data = {}) {
  if (!DEV_REPORTS) return;
  const body = JSON.stringify({
    kind, ...data,
    device: {
      ua: navigator.userAgent, cores: navigator.hardwareConcurrency, touch: navigator.maxTouchPoints,
      screen: `${screen.width}x${screen.height}@${devicePixelRatio}`, standalone: matchMedia("(display-mode: standalone)").matches,
    },
  });
  fetch("log", { method: "POST", body, headers: { "Content-Type": "application/json" } }).catch(() => {});
}

// ----------------------------------------------------------------- start
$("btnPhoto").onclick = takePhoto;
$("btnSuper").onclick = superPhoto;
refreshLastThumb();
openCamera();
getWorker(); // load the engine early
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => { /* e.g. untrusted dev certificate */ });
window.__app = { superPhoto, takePhoto, allPhotos, settings, get job() { return job; } }; // for automated tests
