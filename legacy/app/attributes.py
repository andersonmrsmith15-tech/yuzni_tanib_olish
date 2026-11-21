"""Tashqi ko'rinish atributlari: ko'zoynak, kepka, soqol va boshqalar.

CLIP (ViT-B/32) modeli "zero-shot" usulida ishlatiladi: bosh qismining rasmi har bir atribut uchun
"bor" va "yo'q" matn tavsiflari bilan solishtiriladi. Qo'shimcha o'qitish shart emas, yangi atribut
qo'shish uchun ATTRS ro'yxatiga yangi qator qo'shish kifoya.

Matn embeddinglari bir marta hisoblanib, models/clip/prompts_cache.npz ga yoziladi.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

import cv2
import numpy as np

from .models import MODEL_DIR

CLIP_DIR = os.path.join(MODEL_DIR, "clip")

# (kalit, o'zbekcha nom, belgisi, ["bor" tavsiflari], ["yo'q" tavsiflari], chegara, kesim)
# kesim: "head" - bosh kiyim va quloqlar sig'adigan keng kesim; "face" - faqat yuz (soqol/mo'ylov uchun,
# keng kesimda soqol juda kichik ko'rinib, aniqlik pasayadi).
# Chegaralar admin "Sozlamalar" sahifasida o'zgartiriladi va bazada saqlanadi.
ATTRS = [
    ("glasses", "Ko'zoynak", "👓",
     ["a photo of a person wearing eyeglasses", "a face with glasses on"],
     ["a photo of a person without glasses", "a face with no glasses"], 0.60, "head"),
    ("sunglasses", "Qora ko'zoynak", "🕶️",
     ["a photo of a person wearing dark sunglasses"],
     ["a photo of a person with visible eyes", "a person wearing clear eyeglasses"], 0.65, "head"),
    ("hat", "Kepka / bosh kiyim", "🧢",
     ["a photo of a person wearing a cap", "a photo of a person wearing a hat", "a person wearing a beanie"],
     ["a photo of a person with a bare head", "a person with visible hair and no hat"], 0.60, "head"),
    ("beard", "Soqol", "🧔",
     # 11 ta portretda sinalgan: soqollilar >= 0.76, soqolsizlar <= 0.45 -> chegara 0.60
     ["a photo of a person with a beard on the chin and cheeks", "a bearded man",
      "a man with a thick beard", "a man with a short beard"],
     ["a photo of a person with a smooth shaved chin", "a clean-shaven man",
      "a woman's face", "a man with no facial hair"], 0.60, "face"),
    ("mustache", "Mo'ylov", "👨",
     ["a photo of a man with a mustache", "a man with a moustache and no beard", "a bearded man with a mustache"],
     ["a clean-shaven man", "a photo of a man with no facial hair", "a woman's face"], 0.65, "face"),
    ("mask", "Niqob", "😷",
     ["a photo of a person wearing a face mask", "a person with a medical mask covering mouth and nose"],
     ["a photo of a person with an uncovered face"], 0.65, "head"),
    ("headphones", "Quloqchin", "🎧",
     ["a person wearing over-ear headphones on the head", "a person wearing a headset with a microphone"],
     ["a person with bare ears and no headphones", "a portrait photo of a person"], 0.70, "head"),
    ("headscarf", "Ro'mol", "🧕",
     ["a photo of a woman wearing a headscarf", "a person wearing a hijab"],
     ["a photo of a person with uncovered hair"], 0.75, "head"),
    ("earrings", "Sirg'a", "💎",
     ["a close-up photo of a person wearing earrings"],
     ["a photo of a person without earrings"], 0.78, "head"),
]

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


class AttributeDetector:
    def __init__(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.vision = ort.InferenceSession(os.path.join(CLIP_DIR, "vision_model_quantized.onnx"), opts,
                                           providers=["CPUExecutionProvider"])
        self.lock = threading.Lock()
        self.pos, self.neg = self._text_embeddings()  # (A, 512) har biri
        self.keys = [a[0] for a in ATTRS]
        self.thr = np.array([a[5] for a in ATTRS], np.float32)

    # ---------------- matn ----------------
    def _text_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        prompts = [(a[3], a[4]) for a in ATTRS]
        h = hashlib.sha1(json.dumps(prompts).encode()).hexdigest()[:16]
        cache = os.path.join(CLIP_DIR, "prompts_cache.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            if str(z["hash"]) == h:
                return z["pos"], z["neg"]
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(CLIP_DIR, "tokenizer.json"))
        text = ort.InferenceSession(os.path.join(CLIP_DIR, "text_model_quantized.onnx"),
                                    providers=["CPUExecutionProvider"])

        def embed(sentences: list[str]) -> np.ndarray:
            vecs = []
            for s in sentences:
                ids = np.array([tok.encode(s).ids], np.int64)
                v = text.run(None, {"input_ids": ids})[0][0]
                vecs.append(v / np.linalg.norm(v))
            m = np.mean(vecs, axis=0)
            return m / np.linalg.norm(m)

        pos = np.stack([embed(p) for p, _ in prompts]).astype(np.float32)
        neg = np.stack([embed(n) for _, n in prompts]).astype(np.float32)
        np.savez(cache, pos=pos, neg=neg, hash=h)
        return pos, neg

    # ---------------- rasm ----------------
    @staticmethod
    def _square_crop(img: np.ndarray, box, scale: float, cy_rel: float) -> np.ndarray:
        x, y, w, h = box
        H, W = img.shape[:2]
        cx, cy = x + w / 2, y + h * cy_rel
        side = max(w, h) * scale
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        x1, y1 = int(cx + side / 2), int(cy + side / 2)
        crop = np.full((y1 - y0, x1 - x0, 3), 127, np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        return crop

    def head_crop(self, img: np.ndarray, box) -> np.ndarray:
        """Bosh kiyim va quloqlar ham sig'adigan keng kesim."""
        return self._square_crop(img, box, 2.0, 0.35)

    def face_crop(self, img: np.ndarray, box) -> np.ndarray:
        """Faqat yuz (iyak va yonoqlar bilan) - soqol/mo'ylov uchun."""
        return self._square_crop(img, box, 1.3, 0.5)

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        batch = []
        for c in crops:
            c = cv2.resize(c, (224, 224), interpolation=cv2.INTER_CUBIC)
            rgb = c[:, :, ::-1].astype(np.float32) / 255.0
            batch.append(np.transpose((rgb - MEAN) / STD, (2, 0, 1)))
        with self.lock:
            emb = self.vision.run(None, {"pixel_values": np.stack(batch).astype(np.float32)})[0]
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def predict(self, img: np.ndarray, boxes: list) -> np.ndarray:
        """(N, A) - har bir yuz uchun har bir atribut 'bor' bo'lish ehtimoli."""
        n = len(boxes)
        if not n:
            return np.zeros((0, len(ATTRS)), np.float32)
        emb = self._embed([self.head_crop(img, b) for b in boxes] + [self.face_crop(img, b) for b in boxes])
        head, face = emb[:n], emb[n:]
        out = np.empty((n, len(ATTRS)), np.float32)
        for k, a in enumerate(ATTRS):
            e = face if a[6] == "face" else head
            d = 100.0 * (e @ self.pos[k] - e @ self.neg[k])  # CLIP logit masshtabi
            out[:, k] = 1.0 / (1.0 + np.exp(-d))
        return out

    def set_thresholds(self, thr: dict) -> None:
        for i, k in enumerate(self.keys):
            if k in thr:
                self.thr[i] = float(np.clip(float(thr[k]), 0.05, 0.99))

    def thresholds(self) -> dict:
        return {k: round(float(t), 3) for k, t in zip(self.keys, self.thr)}

    @staticmethod
    def defaults() -> dict:
        return {a[0]: a[5] for a in ATTRS}

    def to_json(self, probs: np.ndarray) -> dict:
        """Bitta yuz uchun: {"present": [...], "all": {kalit: ehtimol}}."""
        present = []
        for i, (key, label, icon, *_rest) in enumerate(ATTRS):
            if probs[i] >= self.thr[i]:
                present.append({"key": key, "label": label, "icon": icon, "p": round(float(probs[i]), 2)})
        # qora ko'zoynak bo'lsa, oddiy ko'zoynakni takrorlamaymiz
        keys = {p["key"] for p in present}
        if "sunglasses" in keys:
            present = [p for p in present if p["key"] != "glasses"]
        return {"present": present, "all": {k: round(float(p), 2) for k, p in zip(self.keys, probs)}}
