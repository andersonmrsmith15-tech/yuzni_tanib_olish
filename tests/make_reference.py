"""Eski (OpenCV/Python) tizim natijalarini tests/ref/ ga yozadi: JS pipeline shular bilan solishtiriladi.
Ishlatish: legacy/.venv emas, loyiha .venv:  python tests/make_reference.py rasm1.jpg rasm2.jpg ...
"""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "legacy"))
from app.models import Models          # noqa: E402
from app.attributes import AttributeDetector  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "ref"); os.makedirs(OUT, exist_ok=True)
m, a = Models(), AttributeDetector()
cases = []
for f in sys.argv[1:]:
    img = cv2.imread(f); name = os.path.splitext(os.path.basename(f))[0]
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA); rgba.tofile(os.path.join(OUT, name + ".rgba"))
    faces, bodies = m.analyze(img)
    probs = a.predict(img, [x.box for x in faces]) if faces else np.zeros((0, 9))
    cases.append({"name": name, "w": img.shape[1], "h": img.shape[0],
        "faces": [{"box": [float(v) for v in x.raw[:4]], "kps": [float(v) for v in x.raw[4:14]], "score": float(x.raw[14]),
                   "emb": x.emb.tolist(), "attrs": dict(zip(a.keys, map(float, p)))} for x, p in zip(faces, probs)],
        "bodies": [{"box": list(map(int, b.box)), "score": b.score, "emb": b.emb.tolist()} for b in bodies]})
json.dump(cases, open(os.path.join(OUT, "reference.json"), "w"))
# CLIP matn embeddinglari brauzer uchun (atribut tavsiflari o'zgarsa qayta yarating)
from app.attributes import ATTRS  # noqa: E402
json.dump({"keys": a.keys, "crops": [x[6] for x in ATTRS], "labels": [x[1] for x in ATTRS], "icons": [x[2] for x in ATTRS],
           "defaults": [x[5] for x in ATTRS], "pos": np.round(a.pos, 6).tolist(), "neg": np.round(a.neg, 6).tolist()},
          open(os.path.join(os.path.dirname(__file__), "..", "public", "models", "attr_prompts.json"), "w"))
print("ok", len(cases))
