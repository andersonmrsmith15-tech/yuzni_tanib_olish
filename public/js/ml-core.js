// ML yadrosi: brauzerda ham, Node.js testlarida ham ishlaydi (DOM ishlatilmaydi).
// Rasm formati: { data: Uint8ClampedArray (RGBA), width, height } — ImageData bilan bir xil.
// Har bir qadam OpenCV (cv2.FaceDetectorYN / FaceRecognizerSF) bilan raqamlar darajasida solishtirilgan:
// tests/ml_parity.mjs ga qarang.

// ---------------- umumiy ----------------
export function l2norm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / s;
  return out;
}

export function nms(boxes, scores, thr) {
  // boxes: [x, y, w, h]
  const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
  const keep = [];
  const removed = new Uint8Array(boxes.length);
  for (const i of order) {
    if (removed[i]) continue;
    keep.push(i);
    const [ax, ay, aw, ah] = boxes[i];
    for (const j of order) {
      if (removed[j] || j === i) continue;
      const [bx, by, bw, bh] = boxes[j];
      const ix = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx));
      const iy = Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by));
      const inter = ix * iy;
      if (inter / (aw * ah + bw * bh - inter + 1e-9) >= thr) removed[j] = 1;
    }
  }
  return keep;
}

// Bilinear warpAffine (teskari xarita, chegaradan tashqari = fill). Natija: RGB Uint8Array (h*w*3)
export function warpAffineRGB(img, M, w, h, fill = 0) {
  const [a, b, c, d, e, f] = M; // dst = [a b c; d e f] * src
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  const ic = -(ia * c + ib * f), iff = -(id * c + ie * f);
  const W = img.width, H = img.height, src = img.data;
  const out = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const sx = ia * x + ib * y + ic, sy = id * x + ie * y + iff;
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      let r = 0, g = 0, bl = 0;
      for (let dy = 0; dy < 2; dy++) {
        const yi = y0 + dy, wy = dy ? fy : 1 - fy;
        for (let dx = 0; dx < 2; dx++) {
          const xi = x0 + dx, wgt = (dx ? fx : 1 - fx) * wy;
          if (xi >= 0 && xi < W && yi >= 0 && yi < H) {
            const p = (yi * W + xi) * 4;
            r += src[p] * wgt; g += src[p + 1] * wgt; bl += src[p + 2] * wgt;
          } else { r += fill * wgt; g += fill * wgt; bl += fill * wgt; }
        }
      }
      const o = (y * w + x) * 3;
      out[o] = Math.min(255, Math.max(0, Math.round(r)));
      out[o + 1] = Math.min(255, Math.max(0, Math.round(g)));
      out[o + 2] = Math.min(255, Math.max(0, Math.round(bl)));
    }
  }
  return out;
}

// cv2.resize(INTER_LINEAR) ga mos: yarim-piksel markazlar, chegarada qiymat takrorlanadi
export function resizeRGB(img, x0, y0, cw, ch, w, h) {
  const W = img.width, H = img.height, src = img.data;
  const out = new Uint8Array(w * h * 3);
  const sxs = cw / w, sys = ch / h;
  for (let y = 0; y < h; y++) {
    let sy = (y + 0.5) * sys - 0.5; if (sy < 0) sy = 0;
    let yi = Math.floor(sy); const fy = sy - yi;
    const yA = Math.min(H - 1, y0 + yi), yB = Math.min(H - 1, y0 + Math.min(yi + 1, ch - 1));
    for (let x = 0; x < w; x++) {
      let sx = (x + 0.5) * sxs - 0.5; if (sx < 0) sx = 0;
      let xi = Math.floor(sx); const fx = sx - xi;
      const xA = Math.min(W - 1, x0 + xi), xB = Math.min(W - 1, x0 + Math.min(xi + 1, cw - 1));
      const o = (y * w + x) * 3;
      for (let k = 0; k < 3; k++) {
        const v = (src[(yA * W + xA) * 4 + k] * (1 - fx) + src[(yA * W + xB) * 4 + k] * fx) * (1 - fy) +
                  (src[(yB * W + xA) * 4 + k] * (1 - fx) + src[(yB * W + xB) * 4 + k] * fx) * fy;
        out[o + k] = Math.min(255, Math.max(0, Math.round(v)));
      }
    }
  }
  return out;
}

// RGB Uint8 (HWC) -> Float32 NCHW. order: "rgb" | "bgr"; mean/std ixtiyoriy (0..1 shkalada)
export function toCHW(rgb, w, h, { order = "rgb", scale = 1, mean = null, std = null } = {}) {
  const n = w * h, out = new Float32Array(3 * n);
  const idx = order === "bgr" ? [2, 1, 0] : [0, 1, 2];
  for (let c = 0; c < 3; c++) {
    const sc = idx[c], m = mean ? mean[c] : 0, s = std ? std[c] : 1;
    for (let i = 0; i < n; i++) out[c * n + i] = (rgb[i * 3 + sc] * scale - m) / s;
  }
  return out;
}

// ---------------- YuNet (yuz detektori) ----------------
// OpenCV kabi: rasm asl o'lchamda, o'ng/pastdan 32 ga karrali qilib nol bilan to'ldiriladi.
export function yunetInput(img) {
  const PW = Math.ceil(img.width / 32) * 32, PH = Math.ceil(img.height / 32) * 32;
  const n = PW * PH, out = new Float32Array(3 * n), src = img.data, W = img.width;
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < W; x++) {
      const p = (y * W + x) * 4, i = y * PW + x;
      out[i] = src[p + 2]; out[n + i] = src[p + 1]; out[2 * n + i] = src[p]; // BGR
    }
  }
  return { tensor: out, dims: [1, 3, PH, PW], PW, PH };
}

export function decodeYunet(out, PW, { scoreThr = 0.8, nmsThr = 0.3, scale = 1 } = {}) {
  const dets = [];
  for (const st of [8, 16, 32]) {
    const cols = PW / st, cls = out[`cls_${st}`], obj = out[`obj_${st}`], bb = out[`bbox_${st}`], kp = out[`kps_${st}`];
    for (let i = 0; i < cls.length; i++) {
      const s = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
      if (s <= scoreThr) continue;
      const r = Math.floor(i / cols), c = i % cols;
      const cx = (c + bb[i * 4]) * st, cy = (r + bb[i * 4 + 1]) * st;
      const w = Math.exp(bb[i * 4 + 2]) * st, h = Math.exp(bb[i * 4 + 3]) * st;
      const kps = [];
      for (let k = 0; k < 5; k++) kps.push((kp[i * 10 + 2 * k] + c) * st / scale, (kp[i * 10 + 2 * k + 1] + r) * st / scale);
      dets.push({ box: [(cx - w / 2) / scale, (cy - h / 2) / scale, w / scale, h / scale], kps, score: s });
    }
  }
  const keep = nms(dets.map((d) => d.box), dets.map((d) => d.score), nmsThr);
  return keep.map((i) => dets[i]);
}

// ---------------- SFace (yuz embedding) ----------------
const ARCFACE_DST = [38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041];

// Eng kichik kvadratlar bo'yicha o'xshashlik o'zgartirishi (aylantirish + masshtab + siljish, aks ettirishsiz)
export function similarityTransform(src, dst = ARCFACE_DST) {
  const n = src.length / 2;
  let mx = 0, my = 0, nx = 0, ny = 0;
  for (let i = 0; i < n; i++) { mx += src[2 * i]; my += src[2 * i + 1]; nx += dst[2 * i]; ny += dst[2 * i + 1]; }
  mx /= n; my /= n; nx /= n; ny /= n;
  let sxx = 0, a = 0, b = 0;
  for (let i = 0; i < n; i++) {
    const px = src[2 * i] - mx, py = src[2 * i + 1] - my, qx = dst[2 * i] - nx, qy = dst[2 * i + 1] - ny;
    sxx += px * px + py * py; a += px * qx + py * qy; b += px * qy - py * qx;
  }
  a /= sxx; b /= sxx;
  return [a, -b, nx - (a * mx - b * my), b, a, ny - (b * mx + a * my)];
}

export function sfaceInput(img, kps) {
  const M = similarityTransform(kps);
  const rgb = warpAffineRGB(img, M, 112, 112, 0);
  return toCHW(rgb, 112, 112, { order: "rgb" });
}

// ---------------- YOLOX (odam detektori) ----------------
export function yoloxInput(img) {
  const size = 640, r = Math.min(size / img.height, size / img.width);
  const nw = Math.trunc(img.width * r), nh = Math.trunc(img.height * r);
  const small = resizeRGB(img, 0, 0, img.width, img.height, nw, nh);
  const n = size * size, out = new Float32Array(3 * n).fill(114);
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
    const s = (y * nw + x) * 3, i = y * size + x;
    out[i] = small[s + 2]; out[n + i] = small[s + 1]; out[2 * n + i] = small[s]; // BGR
  }
  return { tensor: out, ratio: r };
}

export function decodeYolox(out, ratio, imgW, imgH, { conf = 0.5, nmsThr = 0.45 } = {}) {
  const boxes = [], scores = [];
  let k = 0;
  for (const st of [8, 16, 32]) {
    const g = 640 / st;
    for (let gy = 0; gy < g; gy++) for (let gx = 0; gx < g; gx++, k++) {
      const o = k * 85, score = out[o + 4] * out[o + 5]; // obj * P(person)
      if (score <= conf) continue;
      const cx = (out[o] + gx) * st, cy = (out[o + 1] + gy) * st;
      const w = Math.exp(out[o + 2]) * st, h = Math.exp(out[o + 3]) * st;
      boxes.push([(cx - w / 2) / ratio, (cy - h / 2) / ratio, w / ratio, h / ratio]);
      scores.push(score);
    }
  }
  const res = [];
  for (const i of nms(boxes, scores, nmsThr)) {
    let [x, y, w, h] = boxes[i];
    const x0 = Math.max(0, Math.trunc(x)), y0 = Math.max(0, Math.trunc(y));
    const x1 = Math.min(imgW, Math.trunc(x + w)), y1 = Math.min(imgH, Math.trunc(y + h));
    if (x1 - x0 < 20 || y1 - y0 < 40) continue;
    res.push({ box: [x0, y0, x1 - x0, y1 - y0], score: scores[i] });
  }
  return res;
}

// ---------------- ReID (tana embedding) ----------------
export function reidInput(img, box) {
  const [x, y, w, h] = box;
  const rgb = resizeRGB(img, x, y, w, h, 128, 256);
  return toCHW(rgb, 128, 256, { order: "rgb", scale: 1 / 255, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] });
}

// ---------------- CLIP atributlari ----------------
function squareCropAffine(box, scale, cyRel, outSize) {
  const [x, y, w, h] = box;
  const side = Math.max(w, h) * scale, cx = x + w / 2, cy = y + h * cyRel;
  const x0 = Math.trunc(cx - side / 2), y0 = Math.trunc(cy - side / 2);
  const s = outSize / Math.trunc(side);
  // kesimni outSize ga cho'zish: dst = s*(src - x0) (yarim-piksel markazlar bilan)
  return [s, 0, -x0 * s + 0.5 * s - 0.5, 0, s, -y0 * s + 0.5 * s - 0.5];
}
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073], CLIP_STD = [0.26862954, 0.26130258, 0.27577711];
export function clipInput(img, box, kind) {
  const M = kind === "face" ? squareCropAffine(box, 1.3, 0.5, 224) : squareCropAffine(box, 2.0, 0.35, 224);
  const rgb = warpAffineRGB(img, M, 224, 224, 127);
  return toCHW(rgb, 224, 224, { order: "rgb", scale: 1 / 255, mean: CLIP_MEAN, std: CLIP_STD });
}

// prompts: { keys, crops: ["head"|"face"], pos: [[512]], neg: [[512]] }
export function attributeProbs(headEmb, faceEmb, prompts) {
  const out = {};
  prompts.keys.forEach((k, i) => {
    const e = prompts.crops[i] === "face" ? faceEmb : headEmb;
    let d = 0;
    for (let j = 0; j < e.length; j++) d += e[j] * (prompts.pos[i][j] - prompts.neg[i][j]);
    out[k] = 1 / (1 + Math.exp(-100 * d));
  });
  return out;
}

// Yuzni tanaga bog'lash (yuz markazi tananing yuqori 65% qismida)
export function linkFacesToBodies(faces, bodies) {
  bodies.forEach((b) => (b.faces = []));
  faces.forEach((f, fi) => {
    const [fx, fy, fw, fh] = f.box, cx = fx + fw / 2, cy = fy + fh / 2;
    let best = -1, area = Infinity;
    bodies.forEach((b, bi) => {
      const [bx, by, bw, bh] = b.box;
      if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh * 0.65 && bw * bh < area) { best = bi; area = bw * bh; }
    });
    if (best >= 0) bodies[best].faces.push(fi);
  });
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9);
}
