"""Vercel serverless funksiyasi (Flask). Barcha /api/* so'rovlari shu yerga keladi.

Lokal sinov:  .venv/bin/python api/index.py   ->  http://127.0.0.1:5050
(lokalda public/ papkasidagi sahifalarni ham shu server beradi; Vercelda ularni Vercel o'zi beradi)
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, request, send_from_directory, session  # noqa: E402

from lib import auth, engine, people  # noqa: E402
from lib.auth import HOME, PAGES, ROLES, AuthError  # noqa: E402
from lib.db import ROOT  # noqa: E402

app = Flask(__name__)
ON_VERCEL = bool(os.environ.get("VERCEL"))
REQUIRED_ENV = ("DATABASE_URL", "SECRET_KEY")


def missing_env() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.environ.get(k)]


# SECRET_KEY bo'lmasa server yiqilmasin: aniq xato xabari qaytaramiz (pastdagi gate ga qarang)
secret = os.environ.get("SECRET_KEY") or ("" if ON_VERCEL else "local-dev-only")
app.config.update(
    SECRET_KEY=secret or "missing",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=ON_VERCEL,
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    MAX_CONTENT_LENGTH=4_400_000,  # Vercel so'rov hajmi chegarasi ~4.5 MB
)

ALL, ADMIN = set(ROLES), {"admin"}
PUBLIC_API = {"/api/auth/login", "/api/auth/signup", "/api/auth/status", "/api/health"}


def required_roles(path: str, method: str):
    if path in PUBLIC_API or not path.startswith("/api/"):
        return None  # sahifalar statik; ularni common.js /api/me orqali tekshiradi
    if path in ("/api/recognize", "/api/learn"):
        return {"admin", "user"}
    if path in ("/api/me", "/api/attributes", "/api/auth/logout", "/api/auth/password"):
        return ALL
    if path in ("/api/stats", "/api/events") and method == "GET":
        return {"admin", "director"}
    if path.startswith("/api/samples/") and method == "GET":
        return {"admin", "director"}
    return ADMIN  # ro'yxatda yo'q yo'llar faqat admin uchun


def current_account():
    aid = session.get("aid")
    if not aid:
        return None
    r = auth.get(aid)
    if not r or not r["active"] or r["pw_version"] != session.get("pv"):
        session.clear()
        return None
    return r


@app.before_request
def gate():
    if request.path.startswith("/api/") and request.path != "/api/health":
        miss = [k for k in missing_env() if ON_VERCEL or k != "SECRET_KEY"]
        if miss:
            return jsonify({"ok": False, "error": "Server sozlanmagan: Vercel → Project → Settings → Environment Variables "
                                                  f"bo'limiga {', '.join(miss)} qo'shing va qayta deploy qiling."}), 500
    need = required_roles(request.path, request.method)
    if need is None:
        return None
    acc = current_account()
    if acc is None:
        return jsonify({"ok": False, "error": "Tizimga kiring"}), 401
    request.account = acc
    if acc["role"] not in need:
        return jsonify({"ok": False, "error": "Bu amal uchun ruxsatingiz yo'q"}), 403
    return None


@app.errorhandler(ValueError)
def bad_value(e):
    return jsonify({"ok": False, "error": f"Noto'g'ri ma'lumot: {e}"}), 400


@app.errorhandler(Exception)
def server_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({"ok": False, "error": e.description or e.name}), e.code
    app.logger.exception("server xatosi")
    msg = str(e).split("\n")[0][:300]
    if "password" in msg.lower():
        msg = "ma'lumotlar bazasiga ulanib bo'lmadi (DATABASE_URL ni tekshiring)"
    return jsonify({"ok": False, "error": f"Server xatosi ({type(e).__name__}): {msg}"}), 500


@app.errorhandler(413)
def too_big(_e):
    return jsonify({"ok": False, "error": "So'rov juda katta. Rasmlarni kamroq yuboring."}), 413


def body() -> dict:
    return request.get_json(force=True, silent=True) or {}


# ---------------- autentifikatsiya ----------------
def signup_open() -> bool:
    return auth.count() == 0 or bool(engine.get_setting("allow_signup", True))


def _login(acc: dict, remember: bool) -> None:
    session.clear()
    session["aid"], session["pv"] = acc["id"], acc["pw_version"]
    session.permanent = remember


@app.get("/api/health")
def api_health():
    """Diagnostika: sozlamalar va bazaga ulanish (maxfiy qiymatlarni ko'rsatmaydi)."""
    res = {"vercel": ON_VERCEL, "env": {k: bool(os.environ.get(k)) for k in REQUIRED_ENV}}
    try:
        res["tables"] = people.db.val("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' "
                                      "AND table_name IN ('users','samples','events','accounts','settings','counters','login_attempts')")
        res["db"] = "ok" if res["tables"] == 7 else "jadvallar yetishmaydi: python scripts/migrate.py ni ishga tushiring"
        res["accounts"] = people.db.val("SELECT COUNT(*) FROM accounts") if res["tables"] == 7 else None
    except Exception as e:  # noqa: BLE001
        res["db"] = f"ulanib bo'lmadi: {type(e).__name__}"
    res["ok"] = res["db"] == "ok" and all(res["env"].values())
    return jsonify(res), (200 if res["ok"] else 500)


@app.get("/api/auth/status")
def api_auth_status():
    n = auth.count()
    return jsonify({"signup_open": n == 0 or bool(engine.get_setting("allow_signup", True)), "first": n == 0})


@app.post("/api/auth/signup")
def api_signup():
    if not signup_open():
        return jsonify({"ok": False, "error": "Ro'yxatdan o'tish admin tomonidan yopilgan"}), 403
    d = body()
    if d.get("password") != d.get("password2"):
        return jsonify({"ok": False, "error": "Parollar mos emas"}), 400
    try:
        auth.signup(d.get("username"), d.get("full_name"), d.get("password"))
        acc = auth.signin(d.get("username"), d.get("password"), request.remote_addr or "")
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    _login(acc, remember=False)
    acc.pop("pw_version", None)
    return jsonify({"ok": True, "account": acc, "home": HOME[acc["role"]]})


@app.post("/api/auth/login")
def api_login():
    d = body()
    ip = (request.headers.get("x-forwarded-for") or request.remote_addr or "").split(",")[0].strip()
    try:
        acc = auth.signin(d.get("username"), d.get("password"), ip)
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 401
    _login(acc, remember=bool(d.get("remember")))
    people.log("login", None, acc["full_name"], text=f"{acc['full_name']} ({acc['role_label']}) tizimga kirdi")
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
    acc = auth.public(request.account)
    acc["pages"] = [p for p, roles in PAGES.items() if acc["role"] in roles]
    acc["home"] = HOME[acc["role"]]
    return jsonify(acc)


@app.post("/api/auth/password")
def api_password():
    d = body()
    if d.get("new") != d.get("new2"):
        return jsonify({"ok": False, "error": "Yangi parollar mos emas"}), 400
    try:
        auth.set_password(request.account["id"], d.get("new"), old=d.get("old") or "")
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    _login(auth.get(request.account["id"]), remember=session.permanent)
    return jsonify({"ok": True})


# ---------------- hisoblar (admin) ----------------
@app.get("/api/accounts")
def api_accounts():
    return jsonify({"accounts": auth.list_accounts(), "roles": ROLES, "me": request.account["id"]})


@app.post("/api/accounts")
def api_account_create():
    d = body()
    try:
        acc = auth.signup(d.get("username"), d.get("full_name"), d.get("password"), role=d.get("role"))
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "account": acc})


@app.patch("/api/accounts/<aid>")
def api_account_update(aid):
    d = body()
    try:
        if d.get("password"):
            auth.set_password(aid, d["password"])
        acc = auth.update(aid, request.account["id"], role=d.get("role"), active=d.get("active"), full_name=d.get("full_name"))
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "account": acc})


@app.delete("/api/accounts/<aid>")
def api_account_delete(aid):
    try:
        auth.delete(aid, request.account["id"])
    except AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True})


# ---------------- sozlamalar ----------------
@app.get("/api/settings")
def api_settings():
    return jsonify(engine.settings_payload())


@app.post("/api/settings")
def api_settings_save():
    engine.save_settings(body())
    return jsonify(engine.settings_payload())


@app.get("/api/attributes")
def api_attributes():
    thr = engine.attr_thresholds()
    return jsonify({"enabled": True, "items": [{**a, "threshold": thr[a["key"]]} for a in engine.ATTR_META]})


# ---------------- tanib olish ----------------
@app.post("/api/recognize")
def api_recognize():
    is_admin = request.account["role"] == "admin"
    d = body()
    # oddiy foydalanuvchi o'rganishni o'chira olmaydi
    allow_learn = bool(d.get("learn", True)) if is_admin else True
    res = engine.recognize(d, allow_learn)
    res["events"] = people.recent_events(8, types=("learn",))
    return jsonify(res)


@app.post("/api/learn")
def api_learn():
    return jsonify(engine.learn(body()))


# ---------------- foydalanuvchilar (admin) ----------------
@app.get("/api/users")
def api_users():
    return jsonify(people.list_users())


@app.post("/api/users")
def api_register():
    d = body()
    name = (d.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Ism kiritilmagan"}), 400
    if not d.get("items"):
        return jsonify({"ok": False, "error": "Rasm yuborilmadi"}), 400
    source = "upload" if d.get("source") == "upload" else "register"
    res, code = engine.register(d, d["items"], source, force=bool(d.get("force")))
    return jsonify(res), code


@app.get("/api/users/<uid>")
def api_user(uid):
    u = people.get_user(uid)
    if not u:
        return jsonify({"ok": False, "error": "topilmadi"}), 404
    return jsonify({**u, "samples": people.user_samples(uid), "events": people.recent_events(40, user_id=uid),
                    "attr_stats": people.attr_stats(uid)})


@app.patch("/api/users/<uid>")
def api_user_update(uid):
    d = body()
    if "name" in d and not str(d["name"]).strip():
        return jsonify({"ok": False, "error": "Ism bo'sh bo'lishi mumkin emas"}), 400
    u = people.update_user(uid, d)
    return (jsonify({"ok": True, "user": u}) if u else (jsonify({"ok": False, "error": "topilmadi"}), 404))


@app.delete("/api/users/<uid>")
def api_user_delete(uid):
    return jsonify({"ok": people.delete_user(uid)})


@app.post("/api/users/<uid>/images")
def api_user_images(uid):
    if not people.get_user(uid):
        return jsonify({"ok": False, "error": "topilmadi"}), 404
    d = body()
    if not d.get("items"):
        return jsonify({"ok": False, "error": "Rasm yuborilmadi"}), 400
    source = "register" if d.get("source") == "register" else "upload"
    stats = engine.add_images(uid, engine._parse_items(d["items"]), source)
    return jsonify({"ok": True, "stats": stats, "user": people.get_user(uid)})


@app.delete("/api/users/<uid>/samples/<sid>")
def api_sample_delete(uid, sid):
    return jsonify({"ok": people.delete_sample(uid, sid)})


@app.get("/api/samples/<sid>/image")
def api_sample_image(sid):
    data = people.sample_image(sid)
    if not data:
        return Response(status=404)
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "private, max-age=86400", "Cross-Origin-Resource-Policy": "same-origin"})


@app.get("/api/stats")
def api_stats():
    return jsonify(people.stats())


@app.get("/api/events")
def api_events():
    limit = min(int(request.args.get("limit", 50)), 500)
    types = tuple(t for t in request.args.get("types", "").split(",") if t) or None
    return jsonify(people.recent_events(limit, types))


@app.get("/api/config")
def api_config():
    return jsonify(engine.config_dict())


# ---------------- faqat lokal: statik sahifalar ----------------
if not os.environ.get("VERCEL"):
    PUBLIC_DIR = os.path.join(ROOT, "public")

    @app.after_request
    def isolation_headers(resp):
        # vercel.json dagi sarlavhalar bilan bir xil: ko'p oqimli WASM uchun kerak
        resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        resp.headers.setdefault("Cross-Origin-Embedder-Policy", "require-corp")
        return resp

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def static_pages(path):
        if path.startswith("api/"):
            return jsonify({"ok": False, "error": "topilmadi"}), 404
        full = os.path.join(PUBLIC_DIR, path)
        if path and os.path.isfile(full):
            return send_from_directory(PUBLIC_DIR, path)
        page = (path or "index").rstrip("/") + ".html"
        if os.path.isfile(os.path.join(PUBLIC_DIR, page)):
            return send_from_directory(PUBLIC_DIR, page)
        return Response("Topilmadi", status=404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Lokal server: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
