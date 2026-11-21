"""Tanib olinadigan odamlar, ularning namunalari, jurnal va Dashboard statistikasi (Postgres)."""
from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from psycopg.types.json import Jsonb

from .db import db

FACE_DIM, BODY_DIM = 128, 768
TZ = ZoneInfo(os.environ.get("APP_TZ", "Asia/Tashkent"))


def now() -> float:
    return time.time()


# ---------------- galereya keshi ----------------
class Gallery:
    """Barcha embeddinglar xotirada. Baza o'zgarganda (gallery_version) qayta yuklanadi."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.version = None
        self.names: dict[str, str] = {}
        self.face = np.zeros((0, FACE_DIM), np.float32)
        self.face_owner = np.array([], dtype=object)
        self.face_meta: list[dict] = []
        self.body = np.zeros((0, BODY_DIM), np.float32)
        self.body_owner = np.array([], dtype=object)
        self.body_meta: list[dict] = []

    def refresh(self) -> "Gallery":
        v = db.val("SELECT value FROM counters WHERE name='gallery_version'")
        if v == self.version:
            return self
        with self.lock:
            rows = db.all("SELECT id, user_id, kind, source, ts, emb FROM samples ORDER BY ts")
            names = {r["id"]: r["name"] for r in db.all("SELECT id, name FROM users")}
            f = [r for r in rows if r["kind"] == "face" and r["user_id"] in names]
            b = [r for r in rows if r["kind"] == "body" and r["user_id"] in names]
            vec = lambda rs, d: (np.stack([np.frombuffer(r["emb"], np.float32) for r in rs])  # noqa: E731
                                 if rs else np.zeros((0, d), np.float32))
            self.face, self.body = vec(f, FACE_DIM), vec(b, BODY_DIM)
            self.face_owner = np.array([r["user_id"] for r in f], dtype=object)
            self.body_owner = np.array([r["user_id"] for r in b], dtype=object)
            meta = lambda rs: [{"id": r["id"], "source": r["source"], "ts": r["ts"]} for r in rs]  # noqa: E731
            self.face_meta, self.body_meta = meta(f), meta(b)
            self.names, self.version = names, v
        return self

    def of(self, kind: str):
        return (self.face, self.face_owner, self.face_meta) if kind == "face" else \
               (self.body, self.body_owner, self.body_meta)

    def user_matrix(self, kind: str, uid: str):
        mat, owner, meta = self.of(kind)
        idx = np.where(owner == uid)[0]
        return mat[idx], [meta[i] for i in idx]


gallery = Gallery()


def bump_version(cur=None) -> None:
    sql = "UPDATE counters SET value = value + 1 WHERE name='gallery_version'"
    cur.execute(sql) if cur is not None else db.run(sql)


# ---------------- odamlar ----------------
def user_summary(r: dict, counts: dict | None = None) -> dict:
    c = counts or {}
    return {
        "id": r["id"], "name": r["name"], "phone": r["phone"], "department": r["department"], "note": r["note"],
        "created": r["created"], "last_seen": r["last_seen"], "seen_count": r["seen_count"],
        "last_attrs": r["last_attrs"] or [],
        "face_samples": c.get("face", 0), "body_samples": c.get("body", 0),
        "face_auto": c.get("face_auto", 0), "body_auto": c.get("body_auto", 0),
        "face_upload": c.get("upload", 0),
        "thumb": f"/api/samples/{c['thumb']}/image" if c.get("thumb") else None,
    }


def _counts(uid: str | None = None) -> dict[str, dict]:
    where = "WHERE user_id=%s" if uid else ""
    rows = db.all(f"""
        SELECT user_id,
               COUNT(*) FILTER (WHERE kind='face') AS face,
               COUNT(*) FILTER (WHERE kind='body') AS body,
               COUNT(*) FILTER (WHERE kind='face' AND source='auto') AS face_auto,
               COUNT(*) FILTER (WHERE kind='body' AND source='auto') AS body_auto,
               COUNT(*) FILTER (WHERE source='upload') AS upload,
               COALESCE((ARRAY_AGG(id ORDER BY (source='auto'), ts) FILTER (WHERE kind='face' AND image IS NOT NULL))[1], NULL) AS thumb
        FROM samples {where} GROUP BY user_id""", (uid,) if uid else ())
    return {r["user_id"]: r for r in rows}


def list_users() -> list[dict]:
    counts = _counts()
    return [user_summary(r, counts.get(r["id"])) for r in db.all("SELECT * FROM users ORDER BY created DESC")]


def get_user(uid: str) -> dict | None:
    r = db.one("SELECT * FROM users WHERE id=%s", (uid,))
    return user_summary(r, _counts(uid).get(uid)) if r else None


def create_user(name: str, phone="", department="", note="") -> dict:
    uid = uuid.uuid4().hex[:12]
    db.run("INSERT INTO users (id, name, phone, department, note, created) VALUES (%s,%s,%s,%s,%s,%s)",
           (uid, name.strip(), phone.strip(), department.strip(), note.strip(), now()))
    bump_version()
    return get_user(uid)


def update_user(uid: str, fields: dict) -> dict | None:
    allowed = {k: str(v).strip() for k, v in fields.items() if k in ("name", "phone", "department", "note") and v is not None}
    if allowed:
        sets = ", ".join(f"{k}=%s" for k in allowed)
        db.run(f"UPDATE users SET {sets} WHERE id=%s", (*allowed.values(), uid))
        if "name" in allowed:
            bump_version()
    return get_user(uid)


def delete_user(uid: str) -> bool:
    r = db.one("DELETE FROM users WHERE id=%s RETURNING name", (uid,))
    if not r:
        return False
    bump_version()
    log("delete", None, r["name"], text=f"{r['name']} o'chirildi")
    return True


# ---------------- namunalar ----------------
def add_sample(uid: str, kind: str, emb: np.ndarray, source: str, jpeg: bytes | None, cur=None) -> str:
    sid = uuid.uuid4().hex[:12]
    args = (sid, uid, kind, source, now(), np.asarray(emb, np.float32).reshape(-1).tobytes(), jpeg)
    sql = "INSERT INTO samples (id, user_id, kind, source, ts, emb, image) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    if cur is not None:
        cur.execute(sql, args)
        bump_version(cur)
    else:
        db.run(sql, args)
        bump_version()
    return sid


def delete_sample(uid: str, sid: str) -> bool:
    n = db.run("DELETE FROM samples WHERE id=%s AND user_id=%s", (sid, uid))
    if n:
        bump_version()
    return bool(n)


def sample_image(sid: str) -> bytes | None:
    r = db.one("SELECT image FROM samples WHERE id=%s", (sid,))
    return bytes(r["image"]) if r and r["image"] else None


def user_samples(uid: str) -> list[dict]:
    rows = db.all("SELECT id, kind, source, ts, image IS NOT NULL AS has_img FROM samples WHERE user_id=%s ORDER BY ts DESC",
                  (uid,))
    return [{"id": r["id"], "kind": r["kind"], "source": r["source"], "ts": r["ts"],
             "image": f"/api/samples/{r['id']}/image" if r["has_img"] else None} for r in rows]


# ---------------- jurnal ----------------
def log(type_: str, user_id, user_name, via=None, score=None, text=None, attrs=None, cur=None) -> None:
    sql = "INSERT INTO events (ts, type, user_id, user_name, via, score, text, attrs) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
    args = (now(), type_, user_id, user_name, via, score, text, Jsonb(attrs) if attrs is not None else None)
    cur.execute(sql, args) if cur is not None else db.run(sql, args)


def recent_events(limit=50, types=None, user_id=None) -> list[dict]:
    q, args = "SELECT ts, type, user_id, user_name, via, score, text, attrs FROM events WHERE TRUE", []
    if types:
        q += " AND type = ANY(%s)"
        args.append(list(types))
    if user_id:
        q += " AND user_id=%s"
        args.append(user_id)
    q += " ORDER BY ts DESC LIMIT %s"
    args.append(limit)
    return db.all(q, args)


def attr_stats(uid: str | None = None, since: float = 0) -> dict:
    q = "SELECT attrs FROM events WHERE type='recognized' AND attrs IS NOT NULL AND ts >= %s"
    args: list = [since]
    if uid:
        q += " AND user_id=%s"
        args.append(uid)
    rows = db.all(q, args)
    counts: dict[str, int] = {}
    for r in rows:
        for k in r["attrs"]:
            counts[k] = counts.get(k, 0) + 1
    return {"total": len(rows), "counts": counts}


def day_start(t: float | None = None) -> float:
    d = datetime.fromtimestamp(t or now(), TZ)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def stats() -> dict:
    t, d0 = now(), day_start()
    tot = db.one("""
        SELECT (SELECT COUNT(*) FROM users) AS users,
               (SELECT COUNT(*) FROM samples WHERE kind='face') AS face_samples,
               (SELECT COUNT(*) FROM samples WHERE kind='body') AS body_samples,
               (SELECT COUNT(*) FROM samples WHERE source='auto') AS auto_samples,
               (SELECT COUNT(*) FROM events WHERE type='recognized' AND ts >= %(d0)s) AS recognized_today,
               (SELECT COUNT(DISTINCT user_id) FROM events WHERE type='recognized' AND ts >= %(d0)s) AS unique_today,
               (SELECT COUNT(*) FROM events WHERE type='unknown' AND ts >= %(d0)s) AS unknown_today,
               (SELECT COUNT(*) FROM events WHERE type='learn' AND ts >= %(d0)s) AS learned_today,
               (SELECT COUNT(*) FROM users WHERE created >= %(w)s) AS registered_week,
               ROUND(pg_database_size(current_database()) / 1e6, 1) AS db_size_mb""", {"d0": d0, "w": t - 7 * 86400})
    tot["db_size_mb"] = float(tot["db_size_mb"])

    first = datetime.fromtimestamp(d0, TZ) - timedelta(days=13)
    bounds = [(first + timedelta(days=i)).timestamp() for i in range(15)]
    rows = db.all("""
        SELECT width_bucket(ts, %s::float8[]) AS b,
               COUNT(*) FILTER (WHERE type='recognized') AS recognized,
               COUNT(*) FILTER (WHERE type='unknown') AS unknown,
               COUNT(*) FILTER (WHERE type='learn') AS learned
        FROM events WHERE ts >= %s AND ts < %s GROUP BY 1""", (bounds, bounds[0], bounds[-1]))
    by = {r["b"]: r for r in rows}
    days = [{"day": datetime.fromtimestamp(bounds[i], TZ).strftime("%m-%d"),
             "recognized": by.get(i + 1, {}).get("recognized", 0), "unknown": by.get(i + 1, {}).get("unknown", 0),
             "learned": by.get(i + 1, {}).get("learned", 0)} for i in range(14)]

    hours = [0] * 24
    for r in db.all("SELECT ts FROM events WHERE type='recognized' AND ts >= %s", (d0,)):
        hours[datetime.fromtimestamp(r["ts"], TZ).hour] += 1
    via = {(r["via"] or "face"): r["n"] for r in db.all(
        "SELECT via, COUNT(*) AS n FROM events WHERE type='recognized' AND ts >= %s GROUP BY via", (t - 30 * 86400,))}
    top = db.all("""SELECT user_id, MAX(user_name) AS user_name, COUNT(*) AS n, MAX(ts) AS last FROM events
                    WHERE type='recognized' AND ts >= %s AND user_id IS NOT NULL
                    GROUP BY user_id ORDER BY n DESC LIMIT 8""", (t - 30 * 86400,))
    return {"totals": tot, "days": days, "hours": hours, "via": via, "top": top,
            "users": list_users(), "attrs_today": attr_stats(since=d0)}
