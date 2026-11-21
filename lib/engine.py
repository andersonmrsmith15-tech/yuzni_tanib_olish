"""Tanib olish va doimiy o'rganish (server tomoni).

Modellar brauzerda ishlaydi va serverga faqat embedding vektorlari keladi. Server ularni galereya bilan
solishtiradi va o'rganish qoidalarini qo'llaydi. Qoidalar eski lokal versiya bilan bir xil:
  * ishonchli tanilgan (o'xshashlik >= face_learn, boshqalardan farq >= face_margin, yuz aniq va katta)
    va galereyadagilardan yetarlicha farq qiladigan (< face_novelty) yuz yangi namuna sifatida o'rganiladi;
  * yuz orqali ishonchli tanilgan odamning tanasi tana galereyasiga qo'shiladi;
  * galereya to'lsa, eng ko'p takrorlanadigan avtomatik yuz namunasi / eng eski tana namunasi o'chiriladi.

O'rganish ikki bosqichda: /api/recognize "o'rganish mumkin" deb imzolangan token qaytaradi, brauzer esa
/api/learn ga shu token bilan rasm kesimini yuboradi. Token embeddingga bog'langan, shuning uchun mijoz
ixtiyoriy vektor qo'sha olmaydi.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from . import people
from .db import db
from .people import gallery, log, now

ATTR_META = json.load(open(os.path.join(os.path.dirname(__file__), "attr_meta.json"), encoding="utf-8"))["items"]
ATTR_KEYS = {a["key"] for a in ATTR_META}


@dataclass
class Config:
    face_match: float = 0.40
    face_learn: float = 0.55
    face_margin: float = 0.10
    face_novelty: float = 0.85
    face_min_size: int = 60
    face_min_det: float = 0.90
    max_face_samples: int = 80
    body_match: float = 0.70
    body_novelty: float = 0.92
    max_body_samples: int = 40
    learn_cooldown: float = 2.0
    log_interval: float = 20.0
    unknown_log_interval: float = 10.0


CONFIG_KEYS = ("face_match", "face_learn", "body_match", "face_novelty", "log_interval")
CONFIG_LABELS = {
    "face_match": "Yuz bo'yicha tanish chegarasi",
    "face_learn": "Avto-o'rganish uchun minimal o'xshashlik",
    "face_novelty": "Yangi ko'rinish chegarasi (shundan past bo'lsa o'rganadi)",
    "body_match": "Tana bo'yicha tanish chegarasi",
    "log_interval": "Bir odamni jurnalga yozish oralig'i (soniya)",
}


# ---------------- sozlamalar ----------------
def get_setting(key, default=None):
    r = db.one("SELECT value FROM settings WHERE key=%s", (key,))
    return r["value"] if r else default


def set_setting(key, value) -> None:
    from psycopg.types.json import Jsonb
    db.run("INSERT INTO settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
           (key, Jsonb(value)))


def config() -> Config:
    cfg = Config()
    for k, v in (get_setting("config", {}) or {}).items():
        if k in CONFIG_KEYS:
            setattr(cfg, k, type(getattr(cfg, k))(v))
    return cfg


def attr_thresholds() -> dict:
    saved = get_setting("attr_thresholds", {}) or {}
    return {a["key"]: float(saved.get(a["key"], a["default"])) for a in ATTR_META}


# ---------------- vektorlar ----------------
def decode_emb(s: str, dim: int) -> np.ndarray:
    """Brauzerdan base64 (Float32Array baytlari) ko'rinishida keladi."""
    v = np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float32)
    if v.shape != (dim,) or not np.all(np.isfinite(v)):
        raise ValueError("noto'g'ri embedding")
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError("bo'sh embedding")
    return v / n


def decode_jpeg(s: str | None, limit: int = 200_000) -> bytes | None:
    if not s:
        return None
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    b = base64.b64decode(s)
    if len(b) > limit or not b.startswith(b"\xff\xd8"):
        raise ValueError("rasm kesimi noto'g'ri yoki juda katta")
    return b


def match(emb: np.ndarray, kind: str, exclude: str | None = None):
    mat, owner, _ = gallery.of(kind)
    if not len(mat):
        return None, 0.0, 0.0
    sims = mat @ emb
    best: dict[str, float] = {}
    for uid, s in zip(owner, sims):
        if uid != exclude and s > best.get(uid, -1.0):
            best[uid] = float(s)
    if not best:
        return None, 0.0, 0.0
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0], ranked[0][1], (ranked[1][1] if len(ranked) > 1 else 0.0)


# ---------------- o'rganish tokeni ----------------
def _secret() -> bytes:
    return os.environ.get("SECRET_KEY", "").encode() or b"dev-secret"


def _token(uid: str, kind: str, emb: np.ndarray, ts: int) -> str:
    h = hashlib.sha256(np.round(emb, 5).astype(np.float32).tobytes()).hexdigest()
    return hmac.new(_secret(), f"{uid}|{kind}|{ts}|{h}".encode(), hashlib.sha256).hexdigest()[:32]


def _novel(uid: str, kind: str, emb: np.ndarray, cfg: Config) -> bool:
    mat, _ = gallery.user_matrix(kind, uid)
    thr = cfg.face_novelty if kind == "face" else cfg.body_novelty
    return not len(mat) or float(np.max(mat @ emb)) < thr


def _cooled(last: float, cfg: Config) -> bool:
    return now() - (last or 0) >= cfg.learn_cooldown


# ---------------- jonli tanib olish ----------------
def recognize(payload: dict, allow_learn: bool) -> dict:
    t0 = time.time()
    cfg = config()
    gallery.refresh()
    faces_in = payload.get("faces", [])[:8]
    bodies_in = payload.get("bodies", [])[:8]
    users = {}
    want = set()

    faces_out, learn = [], []
    face_user = []
    for i, f in enumerate(faces_in):
        emb = decode_emb(f["emb"], people.FACE_DIM)
        uid, score, second = match(emb, "face")
        uid = uid if score >= cfg.face_match else None
        face_user.append((uid, score))
        faces_out.append({"user_id": uid, "name": gallery.names.get(uid), "score": round(score, 3), "learned": False})
        if uid:
            want.add(uid)
        f["_emb"], f["_m"] = emb, (uid, score, second)

    if want:
        users = {r["id"]: r for r in db.all(
            "SELECT id, name, last_learn_face, last_learn_body, last_logged FROM users WHERE id = ANY(%s)", (list(want),))}

    if allow_learn:
        for i, f in enumerate(faces_in):
            uid, score, second = f["_m"]
            if not uid or uid not in users:
                continue
            ok = (score >= cfg.face_learn and score - second >= cfg.face_margin
                  and float(f.get("det", 0)) >= cfg.face_min_det and float(f.get("size", 0)) >= cfg.face_min_size)
            if ok and _cooled(users[uid]["last_learn_face"], cfg) and _novel(uid, "face", f["_emb"], cfg):
                ts = int(now())
                learn.append({"kind": "face", "index": i, "user_id": uid, "ts": ts, "token": _token(uid, "face", f["_emb"], ts)})

    bodies_out = []
    for j, b in enumerate(bodies_in):
        emb = decode_emb(b["emb"], people.BODY_DIM)
        linked = [face_user[i] for i in b.get("faces", []) if 0 <= i < len(face_user) and face_user[i][0]]
        uid, score, via = None, 0.0, None
        if linked:
            uid, score = max(linked, key=lambda t: t[1])
            via = "face"
            if (allow_learn and score >= cfg.face_learn and len(b.get("faces", [])) == 1 and uid in users
                    and _cooled(users[uid]["last_learn_body"], cfg) and _novel(uid, "body", emb, cfg)):
                ts = int(now())
                learn.append({"kind": "body", "index": j, "user_id": uid, "ts": ts, "token": _token(uid, "body", emb, ts)})
        elif not b.get("faces"):
            m_uid, m_score, m_second = match(emb, "body")
            if m_score >= cfg.body_match and m_score - m_second >= 0.03:
                uid, score, via = m_uid, m_score, "body"
        bodies_out.append({"user_id": uid, "name": gallery.names.get(uid), "score": round(score, 3), "via": via})

    _log_sightings(faces_in, faces_out, bodies_out, users, cfg)
    return {"faces": faces_out, "bodies": bodies_out, "learn": learn, "ms": int((time.time() - t0) * 1000)}


def _log_sightings(faces_in, faces_out, bodies_out, users, cfg) -> None:
    t = now()
    best: dict[str, dict] = {}
    for i, r in enumerate(faces_out):
        if r["user_id"]:
            attrs = [k for k in faces_in[i].get("attrs", []) if k in ATTR_KEYS] if "attrs" in faces_in[i] else None
            cand = {"score": r["score"], "via": "face", "attrs": attrs}
            if r["user_id"] not in best or cand["score"] > best[r["user_id"]]["score"]:
                best[r["user_id"]] = cand
    for r in bodies_out:
        if r["user_id"] and r["user_id"] not in best:
            best[r["user_id"]] = {"score": r["score"], "via": r["via"] or "body", "attrs": None}
            if r["user_id"] not in users:
                row = db.one("SELECT id, name, last_logged FROM users WHERE id=%s", (r["user_id"],))
                if row:
                    users[row["id"]] = row
    for uid, r in best.items():
        u = users.get(uid)
        if not u or t - (u["last_logged"] or 0) < cfg.log_interval:
            continue
        # shartli UPDATE: bir vaqtda ikki so'rov kelsa ham faqat bittasi jurnalga yozadi
        from psycopg.types.json import Jsonb
        n = db.run("""UPDATE users SET last_logged=%s, last_seen=%s, seen_count=seen_count+1,
                      last_attrs=COALESCE(%s, last_attrs) WHERE id=%s AND last_logged < %s""",
                   (t, t, Jsonb(r["attrs"]) if r["attrs"] is not None else None, uid, t - cfg.log_interval))
        if n:
            log("recognized", uid, u["name"], via=r["via"], score=r["score"], attrs=r["attrs"])
    unknown = [r for r in faces_out if not r["user_id"]]
    if unknown:
        n = db.run("UPDATE counters SET value=%s WHERE name='last_unknown_log' AND value < %s",
                   (t, t - cfg.unknown_log_interval))
        if n:
            log("unknown", None, None, via="face", score=max(r["score"] for r in unknown),
                text=f"{len(unknown)} ta noma'lum yuz")


def learn(payload: dict) -> dict:
    cfg = config()
    uid, kind, ts = payload.get("user_id"), payload.get("kind"), int(payload.get("ts", 0))
    if kind not in ("face", "body") or abs(now() - ts) > 120:
        return {"ok": False, "error": "token eskirgan"}
    emb = decode_emb(payload["emb"], people.FACE_DIM if kind == "face" else people.BODY_DIM)
    if not hmac.compare_digest(_token(uid, kind, emb, ts), str(payload.get("token", ""))):
        return {"ok": False, "error": "token noto'g'ri"}
    gallery.refresh()
    col = "last_learn_face" if kind == "face" else "last_learn_body"
    # cooldown ni atomar band qilamiz (parallel so'rovlar ikki marta o'rgatmasin)
    u = db.one(f"UPDATE users SET {col}=%s WHERE id=%s AND {col} < %s RETURNING name", (now(), uid, now() - cfg.learn_cooldown))
    if not u or not _novel(uid, kind, emb, cfg):
        return {"ok": True, "learned": False}
    people.add_sample(uid, kind, emb, "auto", decode_jpeg(payload.get("crop")))
    gallery.refresh()
    _prune(uid, kind, cfg)
    log("learn", uid, u["name"], via=kind,
        text=f"{u['name']}: yangi {'yuz' if kind == 'face' else 'tana'} ko'rinishi o'rganildi")
    return {"ok": True, "learned": True}


def _prune(uid: str, kind: str, cfg: Config) -> None:
    limit = cfg.max_face_samples if kind == "face" else cfg.max_body_samples
    mat, meta = gallery.user_matrix(kind, uid)
    removed = False
    while len(meta) > limit:
        auto = [i for i, m in enumerate(meta) if m["source"] == "auto"]
        if not auto:
            break
        if kind == "body":
            victim = min(auto, key=lambda i: meta[i]["ts"])
        else:
            sims = mat @ mat.T
            np.fill_diagonal(sims, 0)
            red = sims.mean(axis=1)
            victim = max(auto, key=lambda i: red[i])
        db.run("DELETE FROM samples WHERE id=%s", (meta[victim]["id"],))
        mat = np.delete(mat, victim, axis=0)
        meta.pop(victim)
        removed = True
    if removed:
        people.bump_version()


# ---------------- ro'yxatdan o'tkazish / rasm qo'shish ----------------
def _parse_items(items: list) -> list[dict]:
    """Brauzer har bir rasm uchun {face: {emb, crop}|null, body: {emb, crop}|null, warnings: []} yuboradi."""
    out = []
    for it in items[:60]:
        p = {"warnings": [str(w)[:200] for w in it.get("warnings", [])][:5]}
        if it.get("face"):
            p["face"] = (decode_emb(it["face"]["emb"], people.FACE_DIM), decode_jpeg(it["face"].get("crop")))
        if it.get("body"):
            p["body"] = (decode_emb(it["body"]["emb"], people.BODY_DIM), decode_jpeg(it["body"].get("crop")))
        out.append(p)
    return out


def add_images(uid: str, items: list[dict], source: str) -> dict:
    cfg = config()
    gallery.refresh()
    stats = {"face_added": 0, "body_added": 0, "skipped": 0, "warnings": [], "conflicts": 0}
    own = gallery.user_matrix("face", uid)[0]
    name = gallery.names.get(uid, "")
    added_faces: list[np.ndarray] = []

    def work(cur):
        nonlocal own
        for n, it in enumerate(items, 1):
            stats["warnings"] += [f"{n}-rasm: {w}" for w in it["warnings"]]
            if "face" not in it and "body" not in it:
                stats["skipped"] += 1
                stats["warnings"].append(f"{n}-rasm: odam topilmadi")
                continue
            if "face" in it:
                emb, jpeg = it["face"]
                gal = np.vstack([own] + added_faces) if (len(own) or added_faces) else own
                if len(gal):
                    o_uid, o_score, _ = match(emb, "face", exclude=uid)
                    mine = float(np.max(gal @ emb))
                    if o_uid and o_score >= cfg.face_learn and o_score > mine + 0.1:
                        stats["conflicts"] += 1
                        stats["warnings"].append(f"{n}-rasm: bu yuz ko'proq '{gallery.names.get(o_uid)}'ga o'xshaydi - qo'shilmadi")
                        continue
                    if mine < cfg.face_match - 0.1:
                        stats["conflicts"] += 1
                        stats["warnings"].append(f"{n}-rasm: yuz '{name}'ga o'xshamaydi (o'xshashlik {mine:.2f}) - qo'shilmadi")
                        continue
                people.add_sample(uid, "face", emb, source, jpeg, cur=cur)
                added_faces.append(emb[None])
                stats["face_added"] += 1
            if "body" in it:
                emb, jpeg = it["body"]
                people.add_sample(uid, "body", emb, source, jpeg, cur=cur)
                stats["body_added"] += 1

    db.tx(work)
    if source == "upload":
        log("upload", uid, name, text=f"{name}: {stats['face_added']} yuz, {stats['body_added']} tana namunasi yuklandi")
    return stats


def register(form: dict, items_raw: list, source: str, force: bool) -> tuple[dict, int]:
    cfg = config()
    gallery.refresh()
    items = _parse_items(items_raw)
    faces = [it["face"][0] for it in items if "face" in it]
    min_faces = 3 if source == "register" else 1
    if len(faces) < min_faces:
        return {"ok": False, "error": f"Yuz yetarlicha aniqlanmadi ({len(faces)} ta, kamida {min_faces} ta kerak). "
                                      "Yorug'roq joyda, kameraga yaqinroq turing."}, 422
    # takroriy ro'yxatni tekshirish
    scores: dict[str, list[float]] = {}
    for e in faces:
        u, s, _ = match(e, "face")
        if u:
            scores.setdefault(u, []).append(s)
    dup, best = None, 0.0
    for u, sc in scores.items():
        avg = float(np.mean(sc)) * len(sc) / len(faces)
        if avg > best:
            dup, best = u, avg
    if dup and best >= cfg.face_learn and not force:
        nm = gallery.names.get(dup)
        return {"ok": False, "duplicate": {"id": dup, "name": nm, "score": round(best, 3)},
                "error": f"Bu odam allaqachon '{nm}' nomi bilan ro'yxatda bo'lishi mumkin (o'xshashlik {best:.2f})."}, 409
    user = people.create_user(form.get("name", ""), form.get("phone", ""), form.get("department", ""), form.get("note", ""))
    gallery.refresh()
    stats = add_images(user["id"], items, source)
    log("register", user["id"], user["name"],
        text=f"{user['name']} ro'yxatdan o'tdi ({stats['face_added']} yuz, {stats['body_added']} tana namunasi)")
    return {"ok": True, "user": people.get_user(user["id"]), "stats": stats}, 200


def settings_payload() -> dict:
    cfg, defaults, thr = config(), Config(), attr_thresholds()
    return {
        "attrs_enabled": True,
        "attrs": [{**{k: a[k] for k in ("key", "label", "icon", "default")}, "threshold": thr[a["key"]]} for a in ATTR_META],
        "config": [{"key": k, "label": CONFIG_LABELS[k], "value": getattr(cfg, k), "default": getattr(defaults, k)}
                   for k in CONFIG_KEYS],
        "allow_signup": bool(get_setting("allow_signup", True)),
    }


def save_settings(d: dict) -> None:
    if "attrs" in d:
        thr = attr_thresholds()
        for k, v in d["attrs"].items():
            if k in thr:
                thr[k] = float(min(0.99, max(0.05, float(v))))
        set_setting("attr_thresholds", thr)
    if "config" in d:
        saved = get_setting("config", {}) or {}
        base = Config()
        for k, v in d["config"].items():
            if k in CONFIG_KEYS:
                saved[k] = type(getattr(base, k))(v)
        set_setting("config", saved)
    if "allow_signup" in d:
        set_setting("allow_signup", bool(d["allow_signup"]))


def config_dict() -> dict:
    return dataclasses.asdict(config())
