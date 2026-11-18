// Offline support: the app shell is cached; files are served from the cache
// and refreshed in the background (a new version is used on the next launch).
// Bump CACHE when deploying a new version.
const CACHE = "microscope-v1";
const FILES = ["./", "index.html", "app.js", "worker.js", "helper.js", "engine.wasm", "manifest.webmanifest",
  "icons/icon-180.png", "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  e.respondWith(caches.open(CACHE).then(async (cache) => {
    const hit = await cache.match(req, { ignoreSearch: true });
    const refresh = fetch(req).then((res) => {
      if (res.ok) cache.put(req, res.clone());
      return res;
    });
    if (hit) {
      e.waitUntil(refresh.catch(() => {}));
      return hit;
    }
    return refresh;
  }));
});
