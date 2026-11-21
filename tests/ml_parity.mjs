// Brauzer ML yadrosi (public/js/ml-core.js) eski OpenCV tizimi bilan bir xil natija berishini tekshiradi.
// Ishga tushirish: cd tests && npm install && npm test   (avval: python tests/make_reference.py ...)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import * as core from "../public/js/ml-core.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const MODELS = path.join(here, "..", "public", "models");
const CLIP = process.env.CLIP_VISION || path.join(here, "..", "models", "clip", "vision_model_quantized.onnx");
const ref = JSON.parse(fs.readFileSync(path.join(here, "ref", "reference.json")));
const prompts = JSON.parse(fs.readFileSync(path.join(MODELS, "attr_prompts.json")));

const sess = async (f) => ort.InferenceSession.create(f);
const yunet = await sess(path.join(MODELS, "face_detection_yunet_dyn.onnx"));
const sface = await sess(path.join(MODELS, "face_recognition_sface_2021dec.onnx"));
const yolox = await sess(path.join(MODELS, "object_detection_yolox_2022nov.onnx"));
const reid = await sess(path.join(MODELS, "person_reid_youtu_uint8.onnx"));
const clip = fs.existsSync(CLIP) ? await sess(CLIP) : null;

const T = (data, dims) => new ort.Tensor("float32", data, dims);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? "OK  " : "FAIL"} ${msg}`); if (!ok) fail++; };

for (const c of ref) {
  const img = { data: new Uint8ClampedArray(fs.readFileSync(path.join(here, "ref", c.name + ".rgba"))), width: c.w, height: c.h };
  // yuz
  const yi = core.yunetInput(img);
  const yo = await yunet.run({ input: T(yi.tensor, yi.dims) });
  const out = Object.fromEntries(Object.entries(yo).map(([k, v]) => [k, v.data]));
  const faces = core.decodeYunet(out, yi.PW);
  check(faces.length === c.faces.length, `${c.name}: yuzlar soni ${faces.length} (OpenCV: ${c.faces.length})`);
  for (const rf of c.faces) {
    const f = faces.reduce((b, x) => (core.iou(x.box, rf.box) > core.iou(b.box, rf.box) ? x : b), faces[0]);
    const kd = Math.max(...f.kps.map((v, i) => Math.abs(v - rf.kps[i])));
    const e = core.l2norm((await sface.run({ data: T(core.sfaceInput(img, f.kps), [1, 3, 112, 112]) })).fc1.data);
    const sim = dot(e, rf.emb);
    check(sim > 0.995, `${c.name}: yuz embedding o'xshashligi ${sim.toFixed(4)}, landmark farqi ${kd.toFixed(3)}px`);
    if (clip) {
      const run = async (kind) => core.l2norm((await clip.run({ pixel_values: T(core.clipInput(img, f.box, kind), [1, 3, 224, 224]) })).image_embeds.data);
      const p = core.attributeProbs(await run("head"), await run("face"), prompts);
      const worst = Math.max(...prompts.keys.map((k) => Math.abs(p[k] - rf.attrs[k])));
      const flips = prompts.keys.filter((k, i) => (p[k] >= prompts.defaults[i]) !== (rf.attrs[k] >= prompts.defaults[i]));
      check(flips.length === 0, `${c.name}: atributlar, eng katta ehtimol farqi ${worst.toFixed(3)}${flips.length ? ", farq qilganlar: " + flips : ""}`);
    }
  }
  // tana
  const xi = core.yoloxInput(img);
  const xo = (await yolox.run({ images: T(xi.tensor, [1, 3, 640, 640]) })).output.data;
  const bodies = core.decodeYolox(xo, xi.ratio, c.w, c.h);
  check(bodies.length === c.bodies.length, `${c.name}: tanalar soni ${bodies.length} (OpenCV: ${c.bodies.length})`);
  for (const rb of c.bodies) {
    const b = bodies.reduce((best, x) => (core.iou(x.box, rb.box) > core.iou(best.box, rb.box) ? x : best), bodies[0]);
    if (!b) continue;
    const e = core.l2norm((await reid.run({ input: T(core.reidInput(img, b.box), [1, 3, 256, 128]) })).output.data);
    const sim = dot(e, rb.emb);
    check(sim > 0.97, `${c.name}: tana embedding o'xshashligi ${sim.toFixed(4)} (int8 model), quti IoU ${core.iou(b.box, rb.box).toFixed(3)}`);
  }
}
console.log(fail ? `\n${fail} ta tekshiruv o'tmadi` : "\nHammasi o'tdi");
process.exit(fail ? 1 : 0);
