// Service worker: ilova qobig'ini (HTML/CSS/JS) va modellarni keshlaydi.
// API so'rovlari HECH QACHON keshlanmaydi (shaxsiy ma'lumotlar).
const VERSION = "v1";
const SHELL = `shell-${VERSION}`;
const SHELL_FILES = ["/", "/dashboard", "/login", "/signup", "/account", "/register", "/users", "/accounts", "/settings",
  "/css/style.css", "/js/common.js", "/js/ml.js", "/js/ml-core.js", "/manifest.webmanifest", "/icons/icon-192.png",
  "/vendor/ort/ort.wasm.min.mjs", "/vendor/ort/ort-wasm-simd-threaded.mjs"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith("shell-") && k !== SHELL).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/")) return; // tarmoqqa
  if (url.origin === location.origin && (url.pathname.startsWith("/models/") || url.pathname.startsWith("/vendor/"))) {
    // katta, o'zgarmas fayllar: avval kesh
    e.respondWith(caches.open("ml-models-v1").then(async (c) => (await c.match(e.request)) ||
      fetch(e.request).then((r) => { if (r.ok) c.put(e.request, r.clone()); return r; })));
    return;
  }
  if (url.origin === location.origin) {
    // sahifalar va skriptlar: avval tarmoq (yangilanish darhol ko'rinsin), oflayn bo'lsa kesh
    e.respondWith(fetch(e.request).then((r) => {
      if (r.ok) { const copy = r.clone(); caches.open(SHELL).then((c) => c.put(e.request, copy)); }
      return r;
    }).catch(() => caches.match(e.request, { ignoreSearch: true })));
  }
});
