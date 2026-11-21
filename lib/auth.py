"""Tizim hisoblari va rollar (Postgres).

  admin    - barcha imkoniyatlar
  director - faqat Dashboard
  user     - faqat Kamera
Eng birinchi ochilgan hisob avtomatik admin bo'ladi.
"""
from __future__ import annotations

import re
import time
import uuid

import psycopg
from werkzeug.security import check_password_hash, generate_password_hash

from .db import db
from .people import log

ROLES = {"admin": "Admin", "director": "Direktor", "user": "Foydalanuvchi"}
HOME = {"admin": "/", "director": "/dashboard", "user": "/"}
PAGES = {
    "/": {"admin", "user"},
    "/dashboard": {"admin", "director"},
    "/register": {"admin"}, "/users": {"admin"}, "/accounts": {"admin"}, "/settings": {"admin"},
    "/account": set(ROLES),
}
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.]{3,32}$")
MIN_PASSWORD = 8
MAX_FAILS, FAIL_WINDOW = 5, 300


class AuthError(Exception):
    pass


def public(r: dict) -> dict:
    return {"id": r["id"], "username": r["username"], "full_name": r["full_name"], "role": r["role"],
            "role_label": ROLES[r["role"]], "active": bool(r["active"]), "created": r["created"],
            "last_login": r["last_login"]}


def get(aid: str) -> dict | None:
    return db.one("SELECT * FROM accounts WHERE id=%s", (aid,))


def count() -> int:
    return db.val("SELECT COUNT(*) FROM accounts")


def _active_admins() -> int:
    return db.val("SELECT COUNT(*) FROM accounts WHERE role='admin' AND active")


def list_accounts() -> list[dict]:
    return [public(r) for r in db.all("SELECT * FROM accounts ORDER BY created")]


def signup(username: str, full_name: str, password: str, role: str | None = None) -> dict:
    username, full_name = (username or "").strip(), (full_name or "").strip()
    if not USERNAME_RE.match(username):
        raise AuthError("Login 3-32 belgidan iborat bo'lsin: lotin harflari, raqamlar, '_' yoki '.'")
    if not full_name:
        raise AuthError("Ism familiyani kiriting")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"Parol kamida {MIN_PASSWORD} belgidan iborat bo'lsin")
    aid = uuid.uuid4().hex[:12]
    pw = generate_password_hash(password)

    def work(cur):
        # birinchi hisob = admin; parallel sign up lar uchun jadvalni qisqa muddat qulflaymiz
        cur.execute("LOCK TABLE accounts IN SHARE ROW EXCLUSIVE MODE")
        cur.execute("SELECT COUNT(*) AS n FROM accounts")
        first = cur.fetchone()["n"] == 0
        r = "admin" if first else (role if role in ROLES else "user")
        cur.execute("INSERT INTO accounts (id, username, full_name, password_hash, role, created) VALUES (%s,%s,%s,%s,%s,%s)",
                    (aid, username, full_name, pw, r, time.time()))
        return r

    try:
        r = db.tx(work)
    except psycopg.errors.UniqueViolation:
        raise AuthError("Bu login band. Boshqasini tanlang") from None
    log("account", None, full_name, text=f"Yangi hisob: {username} ({ROLES[r]})")
    return public(get(aid))


def signin(username: str, password: str, ip: str) -> dict:
    key = f"{ip}|{(username or '').strip().lower()}"
    t = time.time()
    db.run("DELETE FROM login_attempts WHERE ts < %s", (t - FAIL_WINDOW,))
    fails = db.all("SELECT ts FROM login_attempts WHERE key=%s ORDER BY ts", (key,))
    if len(fails) >= MAX_FAILS:
        wait = int(FAIL_WINDOW - (t - fails[0]["ts"]))
        raise AuthError(f"Juda ko'p noto'g'ri urinish. {wait // 60 + 1} daqiqadan keyin qayta urinib ko'ring")
    r = db.one("SELECT * FROM accounts WHERE lower(username)=lower(%s)", ((username or "").strip(),))
    if not r or not check_password_hash(r["password_hash"], password or ""):
        db.run("INSERT INTO login_attempts (key, ts) VALUES (%s, %s)", (key, t))
        raise AuthError("Login yoki parol noto'g'ri")
    if not r["active"]:
        raise AuthError("Hisobingiz o'chirib qo'yilgan. Admin bilan bog'laning")
    db.run("DELETE FROM login_attempts WHERE key=%s", (key,))
    db.run("UPDATE accounts SET last_login=%s WHERE id=%s", (t, r["id"]))
    return {**public(r), "pw_version": r["pw_version"]}


def update(aid: str, actor_id: str, role=None, active=None, full_name=None) -> dict:
    r = get(aid)
    if not r:
        raise AuthError("Hisob topilmadi")
    if role is not None and role not in ROLES:
        raise AuthError("Noto'g'ri rol")
    if aid == actor_id and (active is False or (role is not None and role != r["role"])):
        raise AuthError("O'z hisobingizning rolini yoki holatini o'zgartira olmaysiz")
    losing_admin = r["role"] == "admin" and r["active"] and ((role is not None and role != "admin") or active is False)
    if losing_admin and _active_admins() <= 1:
        raise AuthError("Kamida bitta faol admin qolishi kerak")
    db.run("UPDATE accounts SET role=COALESCE(%s, role), active=COALESCE(%s, active), "
           "full_name=COALESCE(%s, full_name) WHERE id=%s",
           (role, active, (full_name or "").strip() or None, aid))
    return public(get(aid))


def delete(aid: str, actor_id: str) -> None:
    r = get(aid)
    if not r:
        raise AuthError("Hisob topilmadi")
    if aid == actor_id:
        raise AuthError("O'z hisobingizni o'chira olmaysiz")
    if r["role"] == "admin" and r["active"] and _active_admins() <= 1:
        raise AuthError("Oxirgi adminni o'chirib bo'lmaydi")
    db.run("DELETE FROM accounts WHERE id=%s", (aid,))


def set_password(aid: str, new: str, old: str | None = None) -> None:
    r = get(aid)
    if not r:
        raise AuthError("Hisob topilmadi")
    if old is not None and not check_password_hash(r["password_hash"], old):
        raise AuthError("Joriy parol noto'g'ri")
    if len(new or "") < MIN_PASSWORD:
        raise AuthError(f"Parol kamida {MIN_PASSWORD} belgidan iborat bo'lsin")
    # pw_version oshadi -> boshqa qurilmalardagi sessiyalar bekor bo'ladi
    db.run("UPDATE accounts SET password_hash=%s, pw_version=pw_version+1 WHERE id=%s", (generate_password_hash(new), aid))
