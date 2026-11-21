"""Tizimga kirish hisoblari, rollar va sozlamalar (shu SQLite bazada).

Rollar:
  admin    - barcha imkoniyatlar (ro'yxatdan o'tkazish, foydalanuvchilar, hisoblar, sozlamalar, dashboard, kamera)
  director - faqat Dashboard
  user     - faqat Kamera

Eng birinchi ro'yxatdan o'tgan hisob avtomatik admin bo'ladi.
"""
from __future__ import annotations

import json
import re
import time
import uuid

from werkzeug.security import check_password_hash, generate_password_hash

from .store import Store

ROLES = {"admin": "Admin", "director": "Direktor", "user": "Foydalanuvchi"}
HOME = {"admin": "/", "director": "/dashboard", "user": "/"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name     TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('admin','director','user')),
    active        INTEGER NOT NULL DEFAULT 1,
    created       REAL NOT NULL,
    last_login    REAL,
    pw_version    INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.]{3,32}$")
MIN_PASSWORD = 8


class AuthError(Exception):
    pass


class Accounts:
    def __init__(self, store: Store) -> None:
        self.s = store
        with store.lock:
            store.db.executescript(SCHEMA)
            store.db.commit()
        self._fails: dict[str, list[float]] = {}

    @property
    def db(self):
        return self.s.db

    # ---------------- yordamchi ----------------
    @staticmethod
    def public(r) -> dict:
        return {"id": r["id"], "username": r["username"], "full_name": r["full_name"], "role": r["role"],
                "role_label": ROLES[r["role"]], "active": bool(r["active"]), "created": r["created"],
                "last_login": r["last_login"]}

    def get(self, aid: str):
        with self.s.lock:
            return self.db.execute("SELECT * FROM accounts WHERE id=?", (aid,)).fetchone()

    def count(self) -> int:
        with self.s.lock:
            return self.db.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]

    def _active_admins(self) -> int:
        with self.s.lock:
            return self.db.execute("SELECT COUNT(*) FROM accounts WHERE role='admin' AND active=1").fetchone()[0]

    def list(self) -> list[dict]:
        with self.s.lock:
            return [self.public(r) for r in self.db.execute("SELECT * FROM accounts ORDER BY created")]

    # ---------------- sign up / sign in ----------------
    def signup(self, username: str, full_name: str, password: str, role: str | None = None) -> dict:
        username, full_name = (username or "").strip(), (full_name or "").strip()
        if not USERNAME_RE.match(username):
            raise AuthError("Login 3-32 belgidan iborat bo'lsin: lotin harflari, raqamlar, '_' yoki '.'")
        if not full_name:
            raise AuthError("Ism familiyani kiriting")
        if len(password or "") < MIN_PASSWORD:
            raise AuthError(f"Parol kamida {MIN_PASSWORD} belgidan iborat bo'lsin")
        with self.s.lock:
            first = self.count() == 0
            role = "admin" if first else (role if role in ROLES else "user")
            if self.db.execute("SELECT 1 FROM accounts WHERE username=?", (username,)).fetchone():
                raise AuthError("Bu login band. Boshqasini tanlang")
            aid = uuid.uuid4().hex[:12]
            self.db.execute(
                "INSERT INTO accounts (id, username, full_name, password_hash, role, created) VALUES (?,?,?,?,?,?)",
                (aid, username, full_name, generate_password_hash(password), role, time.time()))
            self.db.commit()
        self.s.log("account", None, full_name, text=f"Yangi hisob: {username} ({ROLES[role]})")
        return self.public(self.get(aid))

    def signin(self, username: str, password: str, ip: str) -> dict:
        key = f"{ip}|{(username or '').lower()}"
        now = time.time()
        fails = [t for t in self._fails.get(key, []) if now - t < 300]
        if len(fails) >= 5:
            wait = int(300 - (now - fails[0]))
            raise AuthError(f"Juda ko'p noto'g'ri urinish. {wait // 60 + 1} daqiqadan keyin qayta urinib ko'ring")
        with self.s.lock:
            r = self.db.execute("SELECT * FROM accounts WHERE username=?", ((username or "").strip(),)).fetchone()
        if not r or not check_password_hash(r["password_hash"], password or ""):
            fails.append(now)
            self._fails[key] = fails
            raise AuthError("Login yoki parol noto'g'ri")
        if not r["active"]:
            raise AuthError("Hisobingiz o'chirib qo'yilgan. Admin bilan bog'laning")
        self._fails.pop(key, None)
        with self.s.lock:
            self.db.execute("UPDATE accounts SET last_login=? WHERE id=?", (now, r["id"]))
            self.db.commit()
        return {**self.public(r), "pw_version": r["pw_version"]}

    # ---------------- admin amallari ----------------
    def update(self, aid: str, actor_id: str, role: str | None = None, active: bool | None = None,
               full_name: str | None = None) -> dict:
        r = self.get(aid)
        if not r:
            raise AuthError("Hisob topilmadi")
        if role is not None and role not in ROLES:
            raise AuthError("Noto'g'ri rol")
        losing_admin = r["role"] == "admin" and r["active"] and (
            (role is not None and role != "admin") or active is False)
        if losing_admin and self._active_admins() <= 1:
            raise AuthError("Kamida bitta faol admin qolishi kerak")
        if aid == actor_id and (active is False or (role is not None and role != r["role"])):
            raise AuthError("O'z hisobingizning rolini yoki holatini o'zgartira olmaysiz")
        with self.s.lock:
            self.db.execute(
                "UPDATE accounts SET role=COALESCE(?, role), active=COALESCE(?, active), "
                "full_name=COALESCE(?, full_name) WHERE id=?",
                (role, None if active is None else int(active), (full_name or "").strip() or None, aid))
            self.db.commit()
        return self.public(self.get(aid))

    def delete(self, aid: str, actor_id: str) -> None:
        r = self.get(aid)
        if not r:
            raise AuthError("Hisob topilmadi")
        if aid == actor_id:
            raise AuthError("O'z hisobingizni o'chira olmaysiz")
        if r["role"] == "admin" and r["active"] and self._active_admins() <= 1:
            raise AuthError("Oxirgi adminni o'chirib bo'lmaydi")
        with self.s.lock:
            self.db.execute("DELETE FROM accounts WHERE id=?", (aid,))
            self.db.commit()

    def set_password(self, aid: str, new: str, old: str | None = None) -> None:
        r = self.get(aid)
        if not r:
            raise AuthError("Hisob topilmadi")
        if old is not None and not check_password_hash(r["password_hash"], old):
            raise AuthError("Joriy parol noto'g'ri")
        if len(new or "") < MIN_PASSWORD:
            raise AuthError(f"Parol kamida {MIN_PASSWORD} belgidan iborat bo'lsin")
        with self.s.lock:
            # pw_version oshadi -> boshqa qurilmalardagi eski sessiyalar bekor bo'ladi
            self.db.execute("UPDATE accounts SET password_hash=?, pw_version=pw_version+1 WHERE id=?",
                            (generate_password_hash(new), aid))
            self.db.commit()

    # ---------------- sozlamalar ----------------
    def get_setting(self, key: str, default=None):
        with self.s.lock:
            r = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_setting(self, key: str, value) -> None:
        with self.s.lock:
            self.db.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
            self.db.commit()
