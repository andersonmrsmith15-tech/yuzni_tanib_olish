"""Neyron tarmoq modellari: yuz detektori, yuz embedding, odam (tana) detektori, tana embedding.

Hammasi OpenCV (cv2.dnn) orqali ishlaydi, qo'shimcha og'ir kutubxonalar kerak emas.
Modellar: OpenCV Zoo (Apache-2.0 / MIT litsenziyalari).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

FACE_DET_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_REC_MODEL = "face_recognition_sface_2021dec.onnx"
PERSON_DET_MODEL = "object_detection_yolox_2022nov.onnx"
BODY_REID_MODEL = "person_reid_youtu_2021nov.onnx"


def _path(name: str) -> str:
    p = os.path.join(MODEL_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Model topilmadi: {p}\n'python download_models.py' buyrug'ini ishga tushiring."
        )
    return p


def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h
    score: float
    raw: np.ndarray  # YuNet natijasi (15 ta landmark qiymati bilan)
    emb: np.ndarray | None = None


@dataclass
class Body:
    box: tuple[int, int, int, int]  # x, y, w, h
    score: float
    emb: np.ndarray | None = None
    faces: list[int] = field(default_factory=list)


class Models:
    """Barcha modellar bitta joyda. cv2.dnn thread-safe emas, shuning uchun lock ishlatiladi."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.face_det = cv2.FaceDetectorYN.create(_path(FACE_DET_MODEL), "", (320, 320), 0.8, 0.3, 5000)
        self.face_rec = cv2.FaceRecognizerSF.create(_path(FACE_REC_MODEL), "")
        self.person_det = cv2.dnn.readNet(_path(PERSON_DET_MODEL))
        self.body_reid = cv2.dnn.readNet(_path(BODY_REID_MODEL))

    # ---------------- Yuz ----------------
    def detect_faces(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        self.face_det.setInputSize((w, h))
        _, res = self.face_det.detect(img)
        faces: list[Face] = []
        if res is None:
            return faces
        for row in res:
            x, y, bw, bh = row[:4]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
            if x1 - x0 < 10 or y1 - y0 < 10:
                continue
            faces.append(Face((x0, y0, x1 - x0, y1 - y0), float(row[14]), row))
        return faces

    def face_embedding(self, img: np.ndarray, face: Face) -> np.ndarray:
        aligned = self.face_rec.alignCrop(img, face.raw)
        feat = self.face_rec.feature(aligned)
        return l2norm(feat)

    # ---------------- Tana ----------------
    def detect_persons(self, img: np.ndarray, conf: float = 0.5) -> list[Body]:
        """YOLOX (COCO) bilan faqat 'person' sinfini aniqlash."""
        size = 640
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(h * r), int(w * r)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = np.transpose(padded, (2, 0, 1))[np.newaxis].astype(np.float32)
        self.person_det.setInput(blob)
        out = self.person_det.forward()[0]  # (8400, 85)

        grids, strides = [], []
        for s in (8, 16, 32):
            g = size // s
            xv, yv = np.meshgrid(np.arange(g), np.arange(g))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((g * g, 1), s))
        grids = np.concatenate(grids).astype(np.float32)
        strides = np.concatenate(strides).astype(np.float32)

        scores = out[:, 4] * out[:, 5]  # obj * P(person)
        keep = scores > conf
        if not np.any(keep):
            return []
        o, gr, st, sc = out[keep], grids[keep], strides[keep], scores[keep]
        cxy = (o[:, :2] + gr) * st
        wh = np.exp(o[:, 2:4]) * st
        x0y0 = (cxy - wh / 2) / r
        x1y1 = (cxy + wh / 2) / r
        boxes = np.concatenate([x0y0, x1y1], 1)
        idx = _nms(boxes, sc, 0.45)

        bodies: list[Body] = []
        for i in idx:
            x0, y0, x1, y1 = boxes[i]
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w, int(x1)), min(h, int(y1))
            if x1 - x0 < 20 or y1 - y0 < 40:
                continue
            bodies.append(Body((x0, y0, x1 - x0, y1 - y0), float(sc[i])))
        return bodies

    def body_embedding(self, img: np.ndarray, body: Body) -> np.ndarray:
        x, y, w, h = body.box
        crop = img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 256))
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis].astype(np.float32)
        self.body_reid.setInput(blob)
        return l2norm(self.body_reid.forward())

    # ---------------- Hammasi birga ----------------
    def analyze(self, img: np.ndarray, with_body: bool = True) -> tuple[list[Face], list[Body]]:
        """Kadrdagi barcha yuz va tanalarni topib, embeddinglarini hisoblaydi."""
        with self.lock:
            faces = self.detect_faces(img)
            for f in faces:
                f.emb = self.face_embedding(img, f)
            bodies: list[Body] = []
            if with_body:
                bodies = self.detect_persons(img)
                for b in bodies:
                    b.emb = self.body_embedding(img, b)
            _link_faces_to_bodies(faces, bodies)
        return faces, bodies


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < thr]
    return keep


def _link_faces_to_bodies(faces: list[Face], bodies: list[Body]) -> None:
    """Har bir yuzni u tegishli bo'lgan tanaga bog'laydi (yuz markazi tananing yuqori qismida)."""
    for fi, f in enumerate(faces):
        fx, fy, fw, fh = f.box
        cx, cy = fx + fw / 2, fy + fh / 2
        best, best_area = None, None
        for bi, b in enumerate(bodies):
            bx, by, bw, bh = b.box
            if bx <= cx <= bx + bw and by <= cy <= by + bh * 0.65:
                area = bw * bh
                if best_area is None or area < best_area:
                    best, best_area = bi, area
        if best is not None:
            bodies[best].faces.append(fi)
