"""Barcha ma'lumotlar SQLite ma'lumotlar bazasida saqlanadi: data/biometrics.db

Jadvallar:
  users    - foydalanuvchilar (ism, telefon, lavozim/bo'lim, izoh, yaratilgan vaqt)
  samples  - yuz/tana namunalari: embedding vektori (BLOB) + rasm kesimi (JPEG BLOB)
  events   - jurnal: tanib olish, noma'lum odam, avtomatik o'rganish, ro'yxatdan o'tish, o'chirish

Tezlik uchun embeddinglar xotirada ham saqlanadi (keshi), lekin manba - baza.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import ROOT

DB_PATH = os.environ.get("BIOMETRICS_DB", os.path.join(ROOT, "data", "biometrics.db"))

FACE_DIM = 128
BODY_DIM = 768

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    phone       TEXT,
    department  TEXT,
    note        TEXT,
    created     REAL NOT NULL,
    last_seen   REAL,
    seen_count  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS samples (
    id       TEXT PRIMARY KEY,
    user_id  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind     TEXT NOT NULL CHECK (kind IN ('face','body')),
    source   TEXT NOT NULL CHECK (source IN ('register','upload','auto')),
    ts       REAL NOT NULL,
    emb      BLOB NOT NULL,
    image    BLOB
);
CREATE INDEX IF NOT EXISTS idx_samples_user ON samples(user_id, kind);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    type     TEXT NOT NULL,   -- recognized | unknown | learn | register | delete | upload
    user_id  TEXT,
    user_name TEXT,
    via      TEXT,            -- face | body
    score    REAL,
    text     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, ts);
"""


@dataclass
class Sample:
    id: str
    kind: str  # "face" | "body"
    source: str  # "register" | "upload" | "auto"
    ts: float
    has_image: bool = False

    @property
    def anchor(self) -> bool:
        """Qo'lda qo'shilgan namunalar (ro'yxatdan o'tish / yuklash) avtomatik o'chirilmaydi."""
        return self.source != "auto"


@dataclass
class User:
    id: str
    name: str
    created: float
    phone: str = ""
    department: str = ""
    note: str = ""
    face_meta: list[Sample] = field(default_factory=list)
    body_meta: list[Sample] = field(default_factory=list)
    face: np.ndarray = field(default_factory=lambda: np.zeros((0, FACE_DIM), np.float32))
    body: np.ndarray = field(default_factory=lambda: np.zeros((0, BODY_DIM), np.float32))
    last_seen: float | None = None
    seen_count: int = 0
    last_attrs: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        def count(meta, src):
            return sum(1 for s in meta if s.source == src)

        thumb = next((s.id for s in self.face_meta if s.has_image and s.source != "auto"), None) or \
            next((s.id for s in self.face_meta if s.has_image), None)
        return {
            "id": self.id, "name": self.name, "phone": self.phone, "department": self.department,
            "note": self.note, "created": self.created, "last_seen": self.last_seen,
            "seen_count": self.seen_count,
            "face_samples": len(self.face_meta), "body_samples": len(self.body_meta),
            "face_auto": count(self.face_meta, "auto"), "body_auto": count(self.body_meta, "auto"),
            "face_upload": count(self.face_meta, "upload") + count(self.body_meta, "upload"),
            "thumb": f"/api/samples/{thumb}/image" if thumb else None,
            "last_attrs": self.last_attrs,
        }


class Store:
    def __init__(self, path: str = DB_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()
        self.users: dict[str, User] = {}
        self._load()

    def _migrate(self) -> None:
        """Eski bazaga yangi ustunlarni qo'shadi (ma'lumot yo'qolmaydi)."""
        cols = lambda t: {r[1] for r in self.db.execute(f"PRAGMA table_info({t})")}  # noqa: E731
        if "attrs" not in cols("events"):
            self.db.execute("ALTER TABLE events ADD COLUMN attrs TEXT")  # JSON: ["glasses", "hat"]
        if "last_attrs" not in cols("users"):
            self.db.execute("ALTER TABLE users ADD COLUMN last_attrs TEXT")
        self.db.commit()

    def _load(self) -> None:
        with self.lock:
            for r in self.db.execute("SELECT * FROM users ORDER BY created"):
                self.users[r["id"]] = User(
                    id=r["id"], name=r["name"], created=r["created"], phone=r["phone"] or "",
                    department=r["department"] or "", note=r["note"] or "",
                    last_seen=r["last_seen"], seen_count=r["seen_count"],
                    last_attrs=json.loads(r["last_attrs"]) if r["last_attrs"] else [])
            faces: dict[str, list] = {}
            bodies: dict[str, list] = {}
            q = "SELECT id, user_id, kind, source, ts, emb, image IS NOT NULL AS has_img FROM samples ORDER BY ts"
            for r in self.db.execute(q):
                u = self.users.get(r["user_id"])
                if not u:
                    continue
                s = Sample(r["id"], r["kind"], r["source"], r["ts"], bool(r["has_img"]))
                vec = np.frombuffer(r["emb"], dtype=np.float32)
                (faces if s.kind == "face" else bodies).setdefault(u.id, []).append((s, vec))
            for uid, u in self.users.items():
                if uid in faces:
                    u.face_meta = [s for s, _ in faces[uid]]
                    u.face = np.vstack([v for _, v in faces[uid]])
                if uid in bodies:
                    u.body_meta = [s for s, _ in bodies[uid]]
                    u.body = np.vstack([v for _, v in bodies[uid]])

    # ---------------- foydalanuvchilar ----------------
    def create_user(self, name: str, phone: str = "", department: str = "", note: str = "") -> User:
        with self.lock:
            u = User(id=uuid.uuid4().hex[:12], name=name.strip(), created=time.time(),
                     phone=phone.strip(), department=department.strip(), note=note.strip())
            self.db.execute(
                "INSERT INTO users (id, name, phone, department, note, created) VALUES (?,?,?,?,?,?)",
                (u.id, u.name, u.phone, u.department, u.note, u.created))
            self.db.commit()
            self.users[u.id] = u
            return u

    def update_user(self, uid: str, **fields) -> User | None:
        allowed = {"name", "phone", "department", "note"}
        with self.lock:
            u = self.users.get(uid)
            if not u:
                return None
            for k, v in fields.items():
                if k in allowed and v is not None:
                    setattr(u, k, str(v).strip())
            self.db.execute("UPDATE users SET name=?, phone=?, department=?, note=? WHERE id=?",
                            (u.name, u.phone, u.department, u.note, uid))
            self.db.commit()
            return u

    def delete_user(self, uid: str) -> bool:
        with self.lock:
            u = self.users.pop(uid, None)
            if not u:
                return False
            self.db.execute("DELETE FROM samples WHERE user_id=?", (uid,))
            self.db.execute("DELETE FROM users WHERE id=?", (uid,))
            self.db.commit()
            self.log("delete", None, u.name, text=f"{u.name} o'chirildi")
            return True

    def save(self, u: User) -> None:
        """Faqat 'ko'rilgan' statistikasini yozadi (namunalar darhol yoziladi)."""
        with self.lock:
            self.db.execute("UPDATE users SET last_seen=?, seen_count=?, last_attrs=? WHERE id=?",
                            (u.last_seen, u.seen_count, json.dumps(u.last_attrs), u.id))
            self.db.commit()

    # ---------------- namunalar ----------------
    def add_sample(self, u: User, kind: str, emb: np.ndarray, source: str, crop: np.ndarray | None) -> Sample:
        with self.lock:
            sid = uuid.uuid4().hex[:12]
            img_bytes = None
            if crop is not None and crop.size:
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                img_bytes = buf.tobytes() if ok else None
            vec = np.asarray(emb, np.float32).reshape(-1)
            s = Sample(sid, kind, source, time.time(), img_bytes is not None)
            self.db.execute(
                "INSERT INTO samples (id, user_id, kind, source, ts, emb, image) VALUES (?,?,?,?,?,?,?)",
                (sid, u.id, kind, source, s.ts, vec.tobytes(), img_bytes))
            self.db.commit()
            if kind == "face":
                u.face = np.vstack([u.face, vec[None]])
                u.face_meta.append(s)
            else:
                u.body = np.vstack([u.body, vec[None]])
                u.body_meta.append(s)
            return s

    def remove_sample(self, u: User, kind: str, index: int) -> None:
        with self.lock:
            meta = u.face_meta if kind == "face" else u.body_meta
            s = meta.pop(index)
            if kind == "face":
                u.face = np.delete(u.face, index, axis=0)
            else:
                u.body = np.delete(u.body, index, axis=0)
            self.db.execute("DELETE FROM samples WHERE id=?", (s.id,))
            self.db.commit()

    def remove_sample_by_id(self, uid: str, sid: str) -> bool:
        with self.lock:
            u = self.users.get(uid)
            if not u:
                return False
            for kind, meta in (("face", u.face_meta), ("body", u.body_meta)):
                for i, s in enumerate(meta):
                    if s.id == sid:
                        self.remove_sample(u, kind, i)
                        return True
            return False

    def sample_image(self, sid: str) -> bytes | None:
        with self.lock:
            r = self.db.execute("SELECT image FROM samples WHERE id=?", (sid,)).fetchone()
            return r["image"] if r else None

    def user_samples(self, uid: str) -> list[dict]:
        u = self.users.get(uid)
        if not u:
            return []
        return [{"id": s.id, "kind": s.kind, "source": s.source, "ts": s.ts,
                 "image": f"/api/samples/{s.id}/image" if s.has_image else None}
                for s in sorted(u.face_meta + u.body_meta, key=lambda s: s.ts, reverse=True)]

    def gallery(self, kind: str) -> tuple[np.ndarray, list[str]]:
        """Barcha foydalanuvchilarning vektorlari bitta matritsada + har bir qator egasi."""
        with self.lock:
            mats, owners = [], []
            for u in self.users.values():
                m = u.face if kind == "face" else u.body
                if len(m):
                    mats.append(m)
                    owners.extend([u.id] * len(m))
            dim = FACE_DIM if kind == "face" else BODY_DIM
            return (np.vstack(mats) if mats else np.zeros((0, dim), np.float32)), owners

    # ---------------- jurnal ----------------
    def log(self, type_: str, user_id: str | None, user_name: str | None, via: str | None = None,
            score: float | None = None, text: str | None = None, attrs: list[str] | None = None) -> None:
        with self.lock:
            self.db.execute(
                "INSERT INTO events (ts, type, user_id, user_name, via, score, text, attrs) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), type_, user_id, user_name, via, score, text,
                 json.dumps(attrs) if attrs is not None else None))
            self.db.commit()

    def recent_events(self, limit: int = 50, types: tuple[str, ...] | None = None, since: float = 0) -> list[dict]:
        with self.lock:
            q = "SELECT * FROM events WHERE ts > ?"
            args: list = [since]
            if types:
                q += f" AND type IN ({','.join('?' * len(types))})"
                args += list(types)
            q += " ORDER BY ts DESC LIMIT ?"
            args.append(limit)
            rows = [dict(r) for r in self.db.execute(q, args)]
        for r in rows:
            r["attrs"] = json.loads(r["attrs"]) if r.get("attrs") else None
        return rows

    def attr_stats(self, uid: str | None = None, since: float = 0) -> dict:
        """Har bir atribut necha foiz holatda ko'ringan (faqat yuz bo'yicha tanilgan hodisalar)."""
        q = "SELECT attrs FROM events WHERE type='recognized' AND attrs IS NOT NULL AND ts>=?"
        args: list = [since]
        if uid:
            q += " AND user_id=?"
            args.append(uid)
        with self.lock:
            rows = [json.loads(r[0]) for r in self.db.execute(q, args)]
        counts: dict[str, int] = {}
        for keys in rows:
            for k in keys:
                counts[k] = counts.get(k, 0) + 1
        return {"total": len(rows), "counts": counts}

    # ---------------- dashboard statistikasi ----------------
    def stats(self) -> dict:
        now = time.time()
        lt = time.localtime(now)
        day_start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        with self.lock:
            one = lambda q, *a: self.db.execute(q, a).fetchone()[0]  # noqa: E731
            totals = {
                "users": len(self.users),
                "face_samples": one("SELECT COUNT(*) FROM samples WHERE kind='face'"),
                "body_samples": one("SELECT COUNT(*) FROM samples WHERE kind='body'"),
                "auto_samples": one("SELECT COUNT(*) FROM samples WHERE source='auto'"),
                "recognized_today": one("SELECT COUNT(*) FROM events WHERE type='recognized' AND ts>=?", day_start),
                "unique_today": one("SELECT COUNT(DISTINCT user_id) FROM events WHERE type='recognized' AND ts>=?",
                                    day_start),
                "unknown_today": one("SELECT COUNT(*) FROM events WHERE type='unknown' AND ts>=?", day_start),
                "learned_today": one("SELECT COUNT(*) FROM events WHERE type='learn' AND ts>=?", day_start),
                "registered_week": one("SELECT COUNT(*) FROM users WHERE created>=?", now - 7 * 86400),
                "db_size_mb": round(sum(os.path.getsize(p) for p in (self.path, self.path + "-wal") if os.path.exists(p)) / 1e6, 2),
            }
            # oxirgi 14 kun, kunlik
            days = []
            for i in range(13, -1, -1):
                d0 = day_start - i * 86400
                d1 = d0 + 86400
                row = self.db.execute(
                    "SELECT SUM(type='recognized'), SUM(type='unknown'), SUM(type='learn') "
                    "FROM events WHERE ts>=? AND ts<?", (d0, d1)).fetchone()
                days.append({"day": time.strftime("%m-%d", time.localtime(d0)),
                             "recognized": row[0] or 0, "unknown": row[1] or 0, "learned": row[2] or 0})
            # bugun, soatlik
            hours = [0] * 24
            for r in self.db.execute("SELECT ts FROM events WHERE type='recognized' AND ts>=?", (day_start,)):
                hours[time.localtime(r["ts"]).tm_hour] += 1
            via = {r[0] or "face": r[1] for r in self.db.execute(
                "SELECT via, COUNT(*) FROM events WHERE type='recognized' AND ts>=? GROUP BY via",
                (now - 30 * 86400,))}
            top = [dict(r) for r in self.db.execute(
                "SELECT user_id, user_name, COUNT(*) AS n, MAX(ts) AS last FROM events "
                "WHERE type='recognized' AND ts>=? AND user_id IS NOT NULL "
                "GROUP BY user_id ORDER BY n DESC LIMIT 8", (now - 30 * 86400,))]
        return {"totals": totals, "days": days, "hours": hours, "via": via, "top": top,
                "users": [u.summary() for u in self.users.values()],
                "attrs_today": self.attr_stats(since=day_start)}
