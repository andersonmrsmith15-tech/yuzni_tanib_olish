// Brauzerdagi neyron tarmoqlar (onnxruntime-web, WASM). Kadrlar serverga yuborilmaydi:
// serverga faqat embedding vektorlari va (o'rganish/ro'yxat uchun) kichik yuz/tana kesimlari boradi.
import * as ort from "/vendor/ort/ort.wasm.min.mjs";
import * as core from "/js/ml-core.js";

ort.env.wasm.wasmPaths = "/vendor/ort/";
ort.env.logLevel = "error";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const CLIP_REV = "6b7c523703ae26e39be4b03774966f8571c7f523";
export const MODELS = {
  yunet: { url: "/models/face_detection_yunet_dyn.onnx", mb: 0.2, label: "Yuz detektori" },
  sface: { url: "/models/face_recognition_sface_2021dec.onnx", mb: 39, label: "Yuzni tanish" },
  yolox: { url: "/models/object_detection_yolox_2022nov.onnx", mb: 36, label: "Odam detektori" },
  reid: { url: "/models/person_reid_youtu_uint8.onnx", mb: 27, label: "Tanani tanish" },
  clip: { url: `https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/${CLIP_REV}/onnx/vision_model_quantized.onnx`, mb: 89, label: "Atributlar (CLIP)" },
};
export const GROUPS = { face: ["yunet", "sface"], body: ["yolox", "reid"], attrs: ["clip"] };
const CACHE = "ml-models-v1";
export const isMobile = matchMedia("(pointer: coarse)").matches;

const sessions = {};
const loading = {};
let prompts = null;

async function fetchCached(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch { /* Cache API yo'q */ }
  const r = await fetch(url, { mode: "cors" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = +r.headers.get("content-length") || 0;
  const reader = r.body.getReader();
  const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    onProgress?.(got, total);
  }
  const buf = new Uint8Array(got); let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  try { await cache?.put(url, new Response(buf, { headers: { "Content-Type": "application/octet-stream" } })); } catch { /* joy yetmasa ham ishlayveradi */ }
  return buf;
}

// names: ["yunet","sface",...]; onProgress(name, fraction 0..1, label)
export async function load(names, onProgress) {
  await Promise.all(names.map((n) => {
    if (sessions[n]) return null;
    if (!loading[n]) {
      loading[n] = (async () => {
        const bytes = await fetchCached(MODELS[n].url, (g, t) => onProgress?.(n, t ? g / t : 0, MODELS[n].label));
        sessions[n] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
        onProgress?.(n, 1, MODELS[n].label);
      })().catch((e) => { delete loading[n]; throw e; });
    }
    return loading[n];
  }));
  if (names.includes("clip") && !prompts) {
    const p = await (await fetch("/models/attr_prompts.json")).json();
    prompts = { ...p, pos: p.pos.map((v) => Float32Array.from(v)), neg: p.neg.map((v) => Float32Array.from(v)) };
  }
}
export const ready = (n) => !!sessions[n];

// ORT sessiyalari parallel chaqirilmasin: hamma ishlar bitta navbatda
let queue = Promise.resolve();
function exclusive(fn) { const p = queue.then(fn, fn); queue = p.catch(() => {}); return p; }
const T = (data, dims) => new ort.Tensor("float32", data, dims);

// ---------------- kadr ----------------
// manba: <video>, <img>, ImageBitmap yoki <canvas>. maxSide dan katta bo'lsa kichraytiriladi.
export function frameFrom(src, maxSide = 960) {
  const w0 = src.videoWidth || src.naturalWidth || src.width, h0 = src.videoHeight || src.naturalHeight || src.height;
  const s = Math.min(1, maxSide / Math.max(w0, h0));
  const w = Math.round(w0 * s), h = Math.round(h0 * s);
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  return { data: img.data, width: w, height: h, canvas, scale: s };
}

// ---------------- modellar ----------------
export const detectFaces = (img, opts) => exclusive(async () => {
  const yi = core.yunetInput(img);
  const o = await sessions.yunet.run({ input: T(yi.tensor, yi.dims) });
  return core.decodeYunet(Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v.data])), yi.PW, opts);
});

export const faceEmbedding = (img, face) => exclusive(async () => {
  const o = await sessions.sface.run({ data: T(core.sfaceInput(img, face.kps), [1, 3, 112, 112]) });
  return core.l2norm(o.fc1.data);
});

export const detectBodies = (img) => exclusive(async () => {
  const xi = core.yoloxInput(img);
  const o = await sessions.yolox.run({ images: T(xi.tensor, [1, 3, 640, 640]) });
  return core.decodeYolox(o.output.data, xi.ratio, img.width, img.height);
});

export const bodyEmbedding = (img, box) => exclusive(async () => {
  const o = await sessions.reid.run({ input: T(core.reidInput(img, box), [1, 3, 256, 128]) });
  return core.l2norm(o.output.data);
});

export const attributes = (img, face) => exclusive(async () => {
  const run = async (kind) => core.l2norm((await sessions.clip.run({ pixel_values: T(core.clipInput(img, face.box, kind), [1, 3, 224, 224]) })).image_embeds.data);
  return core.attributeProbs(await run("head"), await run("face"), prompts);
});

export const attrMetaLocal = () => prompts;

// ---------------- yordamchilar ----------------
export function b64(f32) {
  const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

// Kesimni JPEG (base64, "data:" prefikssiz) qiladi
export function cropJpeg(img, box, pad = 0, maxSize = 160, quality = 0.85) {
  const [x, y, w, h] = box;
  const px = w * pad, py = h * pad;
  const x0 = Math.max(0, Math.round(x - px)), y0 = Math.max(0, Math.round(y - py));
  const x1 = Math.min(img.width, Math.round(x + w + px)), y1 = Math.min(img.height, Math.round(y + h + py));
  const cw = x1 - x0, ch = y1 - y0;
  if (cw < 4 || ch < 4) return null;
  const s = Math.min(1, maxSize / Math.max(cw, ch));
  const c = document.createElement("canvas");
  c.width = Math.round(cw * s); c.height = Math.round(ch * s);
  c.getContext("2d").drawImage(img.canvas, x0, y0, cw, ch, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", quality).split(",")[1];
}

export const linkFacesToBodies = core.linkFacesToBodies;
export const iou = core.iou;

// Ro'yxatdan o'tkazish / rasm yuklash: rasmdagi asosiy odam (eng katta yuz va uning tanasi)
export async function analyzeSubject(img, { body = true } = {}) {
  const warnings = [];
  const faces = await detectFaces(img);
  if (faces.length > 1) warnings.push("rasmda bir nechta yuz bor - eng kattasi olindi");
  const face = faces.length ? faces.reduce((a, b) => (a.box[2] * a.box[3] >= b.box[2] * b.box[3] ? a : b)) : null;
  let bodyBox = null;
  if (body && ready("yolox") && ready("reid")) {
    const bodies = await detectBodies(img);
    if (face) { core.linkFacesToBodies([face], bodies); bodyBox = bodies.find((b) => b.faces.length)?.box || null; }
    else if (bodies.length) bodyBox = bodies.reduce((a, b) => (a.box[2] * a.box[3] >= b.box[2] * b.box[3] ? a : b)).box;
  }
  const item = { warnings };
  if (face) item.face = { emb: b64(await faceEmbedding(img, face)), crop: cropJpeg(img, face.box, 0.25, 192) };
  if (bodyBox) item.body = { emb: b64(await bodyEmbedding(img, bodyBox)), crop: cropJpeg(img, bodyBox, 0, 256) };
  return { item, face, bodyBox };
}

// Fayldan (yuklangan rasm) kadr
export async function frameFromFile(file, maxSide = 1600) {
  const bmp = await createImageBitmap(file);
  try { return frameFrom(bmp, maxSide); } finally { bmp.close?.(); }
}
