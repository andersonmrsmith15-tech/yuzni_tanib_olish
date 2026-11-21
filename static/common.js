// Umumiy yordamchi funksiyalar: navigatsiya, API, kamera, chizish.
const NAV = [
  ["/", "Kamera"],
  ["/register", "Ro'yxatdan o'tkazish"],
  ["/users", "Foydalanuvchilar"],
  ["/dashboard", "Dashboard"],
  ["/accounts", "Hisoblar"],
  ["/settings", "Sozlamalar"],
];
const ROLE_BADGE = { admin: "acc", director: "warn", user: "" };

// Joriy hisob (rol bo'yicha sahifa ichidagi elementlarni yashirish uchun)
let _me = null;
function getMe() {
  if (!_me) _me = api("/api/me").catch(() => null);
  return _me;
}

async function renderNav() {
  if (document.body.dataset.nonav !== undefined) return;
  const me = await getMe();
  if (!me) return;
  const nav = document.createElement("nav");
  nav.className = "top";
  nav.innerHTML = `<div class="brand"><span class="dot"></span>Yuz va tana tanib olish</div>` +
    NAV.filter(([href]) => me.pages.includes(href))
      .map(([href, t]) => `<a class="tab ${location.pathname === href ? "active" : ""}" href="${href}">${t}</a>`).join("") +
    `<div class="row" style="margin-left:auto;gap:8px">
       <a class="tab ${location.pathname === "/account" ? "active" : ""}" href="/account" title="Mening hisobim">${esc(me.full_name)}</a>
       <span class="badge ${ROLE_BADGE[me.role]}">${esc(me.role_label)}</span>
       <button id="logout" style="padding:5px 10px">Chiqish</button></div>`;
  document.body.prepend(nav);
  nav.querySelector("#logout").onclick = async () => { await fetch("/api/auth/logout", { method: "POST" }); location.href = "/login"; };
  document.querySelectorAll("[data-role]").forEach((el) => {
    if (!el.dataset.role.split(",").includes(me.role)) el.remove();
  });
}

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  if (r.status === 401 && !url.startsWith("/api/auth/")) { location.href = "/login?next=" + encodeURIComponent(location.pathname); throw new Error("Tizimga kiring"); }
  let data = null;
  try { data = await r.json(); } catch { data = null; }
  if (!r.ok && !(data && "ok" in data)) throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
  return data;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("uz-UZ", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}
function fmtDate(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString("uz-UZ") + " " + d.toLocaleTimeString("uz-UZ", { hour: "2-digit", minute: "2-digit", hour12: false });
}
function ago(ts) {
  if (!ts) return "hech qachon";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "hozirgina";
  if (s < 3600) return `${Math.floor(s / 60)} daqiqa oldin`;
  if (s < 86400) return `${Math.floor(s / 3600)} soat oldin`;
  return `${Math.floor(s / 86400)} kun oldin`;
}
function initials(name) {
  return (name || "?").split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("");
}
function avatar(u, cls = "avatar") {
  return u.thumb ? `<img class="${cls}" src="${u.thumb}" alt="">` : `<div class="${cls}">${esc(initials(u.name))}</div>`;
}

async function startCamera(video) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Brauzer kamerani qo'llab-quvvatlamaydi");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

const _grab = document.createElement("canvas");
function grabFrame(video, maxW = 960, quality = 0.85) {
  const s = Math.min(1, maxW / video.videoWidth);
  _grab.width = Math.round(video.videoWidth * s);
  _grab.height = Math.round(video.videoHeight * s);
  _grab.getContext("2d").drawImage(video, 0, 0, _grab.width, _grab.height);
  return new Promise((res) => _grab.toBlob(res, "image/jpeg", quality));
}

// Natijalarni video ustidagi canvasga chizish (object-fit: contain hisobga olinadi)
function drawResults(canvas, video, res) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!res || !res.size) return;
  const [W, H] = res.size;
  const s = Math.min(rect.width / W, rect.height / H);
  const ox = (rect.width - W * s) / 2, oy = (rect.height - H * s) / 2;
  const css = getComputedStyle(document.documentElement);
  const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  // Yorliq: 1-qator ism (katta, qalin), 2-qator o'xshashlik va atributlar (kichikroq)
  const box = (b, color, title, sub, dash) => {
    const [x, y, w, h] = b.map((v) => v * s);
    ctx.setLineDash(dash || []);
    ctx.lineWidth = 3; ctx.strokeStyle = color;
    ctx.strokeRect(ox + x, oy + y, w, h);
    ctx.setLineDash([]);
    if (!title) return;
    const big = Math.round(Math.max(20, Math.min(30, rect.width / 30)));
    const small = Math.round(big * 0.58);
    ctx.font = `700 ${big}px ${FONT}`; const tw = ctx.measureText(title).width;
    ctx.font = `500 ${small}px ${FONT}`; const sw = sub ? ctx.measureText(sub).width : 0;
    const pad = 8, lh = big + pad * 1.2 + (sub ? small + 4 : 0);
    const bw = Math.max(tw, sw) + pad * 2;
    let lx = Math.min(ox + x, rect.width - bw - 2); lx = Math.max(2, lx);
    const ly = oy + y - lh - 4 < 0 ? oy + y + h + 4 : oy + y - lh - 4;
    ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(lx, ly, bw, lh, 6) : ctx.rect(lx, ly, bw, lh); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.textBaseline = "top";
    ctx.font = `700 ${big}px ${FONT}`; ctx.fillText(title, lx + pad, ly + pad * 0.6);
    if (sub) { ctx.font = `500 ${small}px ${FONT}`; ctx.globalAlpha = 0.95; ctx.fillText(sub, lx + pad, ly + pad * 0.6 + big + 4); ctx.globalAlpha = 1; }
    ctx.textBaseline = "alphabetic";
  };
  for (const b of res.bodies || []) {
    if (b.via === "body") box(b.box, "#7c3aed", b.name, `tana bo'yicha (taxminiy) · ${(b.score * 100).toFixed(0)}%`, [6, 5]);
    else box(b.box, b.name ? "rgba(124,58,237,.7)" : "rgba(160,160,170,.8)", "", "", [6, 5]);
  }
  for (const f of res.faces || []) {
    const color = f.name ? (f.learned ? "#d97706" : "#16a34a") : "#dc2626";
    const parts = [`${(f.score * 100).toFixed(0)}%`];
    if (f.learned) parts.push("o'rgandi");
    for (const a of f.attrs?.present || []) parts.push(`${a.icon} ${a.label}`);
    box(f.box, color, f.name || "Noma'lum", parts.join(" · "));
  }
}

function attrChips(attrs) {
  const list = attrs?.present || [];
  return list.map((a) => `<span class="chip" title="${Math.round(a.p * 100)}%">${a.icon} ${esc(a.label)}</span>`).join("");
}

document.addEventListener("DOMContentLoaded", renderNav);

// Atributlar ro'yxati (kalit -> nom, belgi)
let _attrMeta = null;
async function attrMeta() {
  if (!_attrMeta) {
    try { const r = await api("/api/attributes"); _attrMeta = Object.fromEntries(r.items.map((a) => [a.key, a])); }
    catch { _attrMeta = {}; }
  }
  return _attrMeta;
}
function keyChips(keys, meta) {
  return (keys || []).map((k) => meta[k] ? `<span class="chip">${meta[k].icon} ${esc(meta[k].label)}</span>` : "").join("");
}
