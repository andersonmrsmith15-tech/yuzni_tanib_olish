"""Yuz va tanani tanib olish tizimi - veb server.

Ishga tushirish:  .venv/bin/python server.py   ->  http://127.0.0.1:5050
"""
from __future__ import annotations

import dataclasses
import os
import secrets
from datetime import timedelta

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, redirect, request, send_from_directory, session

from app.attributes import ATTRS, AttributeDetector
from app.auth import HOME, ROLES, Accounts, AuthError
from app.engine import Config, Engine
from app.models import Models
from app.store import Store

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # ko'p rasm yuklash uchun

print("Modellar yuklanmoqda...")
models = Models()
store = Store()
try:
    attrs = AttributeDetector()
except Exception as e:  # CLIP modeli bo'lmasa tizim atributsiz ishlayveradi
    print(f"[ogohlantirish] Atributlar o'chirildi: {e}")
    attrs = None
engine = Engine(models, store, Config(), attrs=attrs)
accounts = Accounts(store)

# Saqlangan sozlamalarni qo'llash (admin "Sozlamalar" sahifasida o'zgartiradi)
CONFIG_KEYS = ("face_match", "face_learn", "body_match", "face_novelty", "log_interval")
for k, v in (accounts.get_setting("config", {}) or {}).items():
    if k in CONFIG_KEYS:
        setattr(engine.cfg, k, type(getattr(engine.cfg, k))(v))
if attrs:
    attrs.set_thresholds(accounts.get_setting("attr_thresholds", {}) or {})


def _secret_key() -> bytes:
    path = os.path.join(ROOT, "data", "secret.key")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(secrets.token_bytes(32))
        os.chmod(path, 0o600)
    with open(path, "rb") as f:
        return f.read()


app.config.update(SECRET_KEY=_secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(days=7))
print(f"Tayyor. Foydalanuvchilar: {len(store.users)}, hisoblar: {accounts.count()}")
if accounts.count() == 0:
    print("Hali hisob yo'q: /signup sahifasida birinchi bo'lib ro'yxatdan o'tgan odam ADMIN bo'ladi.")

# ---------------- ruxsatlar ----------------
ALL = set(ROLES)
ADMIN = {"admin"}
PUBLIC = {"/login", "/signup", "/api/auth/login", "/api/auth/signup", "/api/auth/status"}
PAGES = {
    "/": {"admin", "user"},
    "/dashboard": {"admin", "director"},
    "/register": ADMIN, "/users": ADMIN, "/accounts": ADMIN, "/settings": ADMIN,
    "/account": ALL,
}


def required_roles(path: str, method: str) -> set[str] | None:
    """None -> ochiq. Ro'yxatda yo'q yo'llar faqat admin uchun (xavfsiz standart)."""
    if path in PUBLIC or path.startswith("/static/"):
        return None
    if path in PAGES:
        return PAGES[path]
    if path == "/api/recognize":
        return {"admin", "user"}
    if path in ("/api/me", "/api/attributes", "/api/auth/logout", "/api/auth/password"):
        return ALL
    if path in ("/api/stats", "/api/events") and method == "GET":
        return {"admin", "director"}
    if path.startswith("/api/samples/") and method == "GET":
        return {"admin", "director"}
    return ADMIN


def current_account():
    aid = session.get("aid")
    if not aid:
        return None
    r = accounts.get(aid)
    if not r or not r["active"] or r["pw_version"] != session.get("pv"):
        session.clear()
        return None
    return r


@app.before_request
def gate():
    need = required_roles(request.path, request.method)
    if need is None:
        return None
    acc = current_account()
    is_api = request.path.startswith("/api/")
    if acc is None:
        if is_api:
            return jsonify({"ok": False, "error": "Tizimga kiring"}), 401
        return redirect(f"/login?next={request.path}")
    request.account = acc
    if acc["role"] not in need:
        if is_api:
            return jsonify({"ok": False, "error": "Bu amal uchun ruxsatingiz yo'q"}), 403
        return redirect(HOME[acc["role"]])
    return None


def decode(file_storage) -> np.ndarray | None:
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    m = max(h, w)
    if m > 1600:  # juda katta rasmlarni kichraytiramiz
        s = 1600 / m
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def read_images(field: str = "images") -> tuple[list[np.ndarray], int]:
    imgs, bad = [], 0
    for f in request.files.getlist(field):
        img = decode(f)
        if img is None:
            bad += 1
        else:
            imgs.append(img)
    return imgs, bad


def flag(name: str, default: bool = True) -> bool:
    v = request.values.get(name)
    return default if v is None else v in ("1", "true", "on", "yes")


# ---------------- sahifalar ----------------
@app.get("/")
def page_live():
    return send_from_directory(STATIC, "index.html")


@app.get("/register")
def page_register():
    return send_from_directory(STATIC, "register.html")


@app.get("/users")
def page_users():
    return send_from_directory(STATIC, "users.html")


@app.get("/dashboard")
def page_dashboard():
    return send_from_directory(STATIC, "dashboard.html")


@app.get("/accounts")
def page_accounts():
    return send_from_directory(STATIC, "accounts.html")


@app.get("/settings")
def page_settings():
    return send_from_directory(STATIC, "settings.html")


@app.get("/account")
def page_account():
    return send_from_directory(STATIC, "account.html")


@app.get("/login")
def page_login():
    if current_account():
        return redirect(HOME[current_account()["role"]])
    return send_from_directory(STATIC, "login.html")


@app.get("/signup")
def page_signup():
    return send_from_directory(STATIC, "signup.html")


# ---------------- autentifikatsiya ----------------
def signup_open() -> bool:
    return accounts.count() == 0 or bool(accounts.get_setting("allow_signup", True))


def _login(acc: dict, remember: bool) -> None:
    session.clear()
    session["aid"], session["pv"] = acc["id"], acc["pw_version"]
    session.permanent = remember


@app.get("/api/auth/status")
def api_auth_status():
    return jsonify({"signup_open": signup_open(), "first": accounts.count() == 0})


@app.post("/api/auth/signup")
def api_signup():
    if not signup_open():
        return jsonify({"ok": False, "error": "Ro'yxatdan o'tish admin tomonidan yopilgan"}), 403
    d = request.get_json(force=True) or {}
    if d.get("password") != d.get("password2"):
        return jsonify({"ok": False, "error": "Parollar mos emas"}), 400
    try:
        acc = accounts.signup(d.get("username"), d.get("full_name"), d.get("password"))
        acc = accounts.signin(d.get("username"), d.get("password"), request.remote_addr or "")
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    _login(acc, remember=False)
    acc.pop("pw_version", None)
    return jsonify({"ok": True, "account": acc, "home": HOME[acc["role"]]})


@app.post("/api/auth/login")
def api_login():
    d = request.get_json(force=True) or {}
    try:
        acc = accounts.signin(d.get("username"), d.get("password"), request.remote_addr or "")
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 401
    _login(acc, remember=bool(d.get("remember")))
    store.log("login", None, acc["full_name"], text=f"{acc['full_name']} ({acc['role_label']}) tizimga kirdi")
    nxt = d.get("next") or ""
    home = nxt if nxt.startswith("/") and not nxt.startswith("//") and acc["role"] in PAGES.get(nxt, set()) \
        else HOME[acc["role"]]
    acc.pop("pw_version", None)
    return jsonify({"ok": True, "account": acc, "home": home})


@app.post("/api/auth/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/me")
def api_me():
    acc = accounts.public(request.account)
    acc["pages"] = [p for p, roles in PAGES.items() if acc["role"] in roles]
    return jsonify(acc)


@app.post("/api/auth/password")
def api_password():
    d = request.get_json(force=True) or {}
    if d.get("new") != d.get("new2"):
        return jsonify({"ok": False, "error": "Yangi parollar mos emas"}), 400
    try:
        accounts.set_password(request.account["id"], d.get("new"), old=d.get("old") or "")
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    _login(accounts.get(request.account["id"]), remember=session.permanent)
    return jsonify({"ok": True})


# ---------------- hisoblar (admin) ----------------
@app.get("/api/accounts")
def api_accounts():
    return jsonify({"accounts": accounts.list(), "roles": ROLES, "me": request.account["id"]})


@app.post("/api/accounts")
def api_account_create():
    d = request.get_json(force=True) or {}
    try:
        acc = accounts.signup(d.get("username"), d.get("full_name"), d.get("password"), role=d.get("role"))
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "account": acc})


@app.patch("/api/accounts/<aid>")
def api_account_update(aid):
    d = request.get_json(force=True) or {}
    try:
        if d.get("password"):
            accounts.set_password(aid, d["password"])
        acc = accounts.update(aid, request.account["id"], role=d.get("role"), active=d.get("active"),
                              full_name=d.get("full_name"))
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "account": acc})


@app.delete("/api/accounts/<aid>")
def api_account_delete(aid):
    try:
        accounts.delete(aid, request.account["id"])
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True})


# ---------------- sozlamalar (admin) ----------------
CONFIG_LABELS = {
    "face_match": "Yuz bo'yicha tanish chegarasi",
    "face_learn": "Avto-o'rganish uchun minimal o'xshashlik",
    "face_novelty": "Yangi ko'rinish chegarasi (shundan past bo'lsa o'rganadi)",
    "body_match": "Tana bo'yicha tanish chegarasi",
    "log_interval": "Bir odamni jurnalga yozish oralig'i (soniya)",
}


@app.get("/api/settings")
def api_settings():
    defaults = Config()
    return jsonify({
        "attrs_enabled": attrs is not None,
        "attrs": [{"key": a[0], "label": a[1], "icon": a[2], "threshold": attrs.thresholds()[a[0]] if attrs else a[5],
                   "default": a[5]} for a in ATTRS],
        "config": [{"key": k, "label": CONFIG_LABELS[k], "value": getattr(engine.cfg, k),
                    "default": getattr(defaults, k)} for k in CONFIG_KEYS],
        "allow_signup": bool(accounts.get_setting("allow_signup", True)),
    })


@app.post("/api/settings")
def api_settings_save():
    d = request.get_json(force=True) or {}
    if "attrs" in d and attrs:
        attrs.set_thresholds(d["attrs"])
        accounts.set_setting("attr_thresholds", attrs.thresholds())
    if "config" in d:
        saved = accounts.get_setting("config", {}) or {}
        for k, v in d["config"].items():
            if k in CONFIG_KEYS:
                setattr(engine.cfg, k, type(getattr(engine.cfg, k))(v))
                saved[k] = getattr(engine.cfg, k)
        accounts.set_setting("config", saved)
    if "allow_signup" in d:
        accounts.set_setting("allow_signup", bool(d["allow_signup"]))
    return api_settings()


# ---------------- API ----------------
@app.post("/api/recognize")
def api_recognize():
    f = request.files.get("frame")
    if not f:
        abort(400, "frame yo'q")
    img = decode(f)
    if img is None:
        abort(400, "rasm o'qilmadi")
    is_admin = request.account["role"] == "admin"
    # Oddiy foydalanuvchi o'rganishni o'chira olmaydi; diagnostika (barcha ehtimollar) faqat admin uchun
    res = engine.recognize(img, learn=flag("learn") if is_admin else True, with_body=flag("body"),
                           with_attrs=flag("attrs"))
    if not is_admin:
        for f in res["faces"]:
            if f.get("attrs"):
                f["attrs"].pop("all", None)
    res["size"] = [img.shape[1], img.shape[0]]
    return jsonify(res)


@app.get("/api/users")
def api_users():
    users = sorted(store.users.values(), key=lambda u: u.created, reverse=True)
    return jsonify([u.summary() for u in users])


@app.post("/api/users")
def api_register():
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Ism kiritilmagan"}), 400
    imgs, bad = read_images()
    if not imgs:
        return jsonify({"ok": False, "error": "Rasm yuborilmadi"}), 400
    source = "upload" if request.form.get("source") == "upload" else "register"
    profile = {k: request.form.get(k, "") for k in ("phone", "department", "note")}
    res = engine.register(name, imgs, source, force=flag("force", False), profile=profile)
    if bad and res.get("stats"):
        res["stats"]["warnings"].append(f"{bad} ta fayl rasm sifatida o'qilmadi")
    return jsonify(res), (200 if res["ok"] else 409 if res.get("duplicate") else 422)


@app.get("/api/users/<uid>")
def api_user(uid):
    u = store.users.get(uid) or abort(404)
    events = [e for e in store.recent_events(500) if e["user_id"] == uid][:40]
    return jsonify({**u.summary(), "samples": store.user_samples(uid), "events": events,
                    "attr_stats": store.attr_stats(uid)})


@app.get("/api/attributes")
def api_attributes():
    return jsonify({"enabled": attrs is not None,
                    "items": [{"key": a[0], "label": a[1], "icon": a[2]} for a in ATTRS]})


@app.patch("/api/users/<uid>")
def api_user_update(uid):
    data = request.get_json(force=True) or {}
    if "name" in data and not str(data["name"]).strip():
        return jsonify({"ok": False, "error": "Ism bo'sh bo'lishi mumkin emas"}), 400
    u = store.update_user(uid, **data) or abort(404)
    return jsonify({"ok": True, "user": u.summary()})


@app.delete("/api/users/<uid>")
def api_user_delete(uid):
    return jsonify({"ok": store.delete_user(uid)})


@app.post("/api/users/<uid>/images")
def api_user_images(uid):
    u = store.users.get(uid) or abort(404)
    imgs, bad = read_images()
    if not imgs:
        return jsonify({"ok": False, "error": "Rasm yuborilmadi"}), 400
    source = "register" if request.form.get("source") == "register" else "upload"
    stats = engine.add_images(u, imgs, source)
    if bad:
        stats["warnings"].append(f"{bad} ta fayl rasm sifatida o'qilmadi")
    return jsonify({"ok": True, "stats": stats, "user": u.summary()})


@app.delete("/api/users/<uid>/samples/<sid>")
def api_sample_delete(uid, sid):
    return jsonify({"ok": store.remove_sample_by_id(uid, sid)})


@app.get("/api/samples/<sid>/image")
def api_sample_image(sid):
    data = store.sample_image(sid) or abort(404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/stats")
def api_stats():
    return jsonify(store.stats())


@app.get("/api/events")
def api_events():
    limit = min(int(request.args.get("limit", 50)), 500)
    types = tuple(t for t in request.args.get("types", "").split(",") if t) or None
    return jsonify(store.recent_events(limit, types))


@app.get("/api/config")
def api_config():
    return jsonify(dataclasses.asdict(engine.cfg))


@app.post("/api/config")
def api_config_set():
    data = request.get_json(force=True) or {}
    for k, v in data.items():
        if hasattr(engine.cfg, k):
            cur = getattr(engine.cfg, k)
            setattr(engine.cfg, k, type(cur)(v))
    return jsonify(dataclasses.asdict(engine.cfg))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Brauzerda oching: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
