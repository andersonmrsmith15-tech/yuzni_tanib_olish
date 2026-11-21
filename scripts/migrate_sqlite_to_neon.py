"""Eski lokal SQLite bazasini (data/biometrics.db) Neon Postgres ga ko'chiradi.

    python scripts/migrate_sqlite_to_neon.py [yo'l/biometrics.db]

Qayta ishga tushirish xavfsiz: mavjud id lar o'tkazib yuboriladi (ON CONFLICT DO NOTHING).
Yuz embeddinglari brauzerdagi pipeline bilan bir xil (tests/ml_parity.mjs), shuning uchun o'zgarishsiz ko'chiriladi.
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from lib.db import ROOT  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "biometrics.db")
if not os.path.exists(src):
    sys.exit(f"Topilmadi: {src}")
s = sqlite3.connect(src)
s.row_factory = sqlite3.Row
tables = {r[0] for r in s.execute("SELECT name FROM sqlite_master WHERE type='table'")}
cols = lambda t: {r[1] for r in s.execute(f"PRAGMA table_info({t})")}  # noqa: E731

url = os.environ.get("DATABASE_URL_UNPOOLED") or os.environ["DATABASE_URL"]
n = {}
with psycopg.connect(url) as pg, pg.cursor() as cur:
    ucols = cols("users")
    for r in s.execute("SELECT * FROM users"):
        la = json.loads(r["last_attrs"]) if "last_attrs" in ucols and r["last_attrs"] else []
        cur.execute("""INSERT INTO users (id, name, phone, department, note, created, last_seen, seen_count, last_attrs)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING""",
                    (r["id"], r["name"], r["phone"] or "", r["department"] or "", r["note"] or "", r["created"],
                     r["last_seen"], r["seen_count"] or 0, Jsonb(la)))
        n["users"] = n.get("users", 0) + cur.rowcount
    for r in s.execute("SELECT * FROM samples"):
        cur.execute("""INSERT INTO samples (id, user_id, kind, source, ts, emb, image) VALUES (%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (id) DO NOTHING""",
                    (r["id"], r["user_id"], r["kind"], r["source"], r["ts"], bytes(r["emb"]),
                     bytes(r["image"]) if r["image"] else None))
        n["samples"] = n.get("samples", 0) + cur.rowcount
    ecols = cols("events")
    if not cur.execute("SELECT 1 FROM events LIMIT 1").fetchone():  # jurnal faqat bo'sh bazaga ko'chiriladi
        for r in s.execute("SELECT * FROM events ORDER BY ts"):
            at = json.loads(r["attrs"]) if "attrs" in ecols and r["attrs"] else None
            cur.execute("INSERT INTO events (ts, type, user_id, user_name, via, score, text, attrs) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                        (r["ts"], r["type"], r["user_id"], r["user_name"], r["via"], r["score"], r["text"],
                         Jsonb(at) if at is not None else None))
            n["events"] = n.get("events", 0) + 1
    if "accounts" in tables:
        for r in s.execute("SELECT * FROM accounts"):
            cur.execute("""INSERT INTO accounts (id, username, full_name, password_hash, role, active, created, last_login, pw_version)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (r["id"], r["username"], r["full_name"], r["password_hash"], r["role"], bool(r["active"]),
                         r["created"], r["last_login"], r["pw_version"]))
            n["accounts"] = n.get("accounts", 0) + cur.rowcount
    if "settings" in tables:
        for r in s.execute("SELECT key, value FROM settings"):
            cur.execute("INSERT INTO settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
                        (r["key"], Jsonb(json.loads(r["value"]))))
    cur.execute("UPDATE counters SET value = value + 1 WHERE name='gallery_version'")
    pg.commit()
print("Ko'chirildi:", n or "yangi ma'lumot yo'q")
