"""API ning to'liq sinovi (Flask test client + haqiqiy Postgres).

FAQAT TEST BAZASIDA ishga tushiring (hisoblar va odamlar yaratadi):
    TEST_DATABASE_URL=postgresql://...test-branch... python tests/api_e2e.py
Embeddinglar tests/ref/reference.json dan olinadi (python tests/make_reference.py bilan yaratiladi).
"""
import base64
import json
import os
import sys

import numpy as np

url = os.environ.get("TEST_DATABASE_URL")
if not url:
    sys.exit("TEST_DATABASE_URL kerak (asosiy bazada ishga tushirmang)")
os.environ["DATABASE_URL"] = url
os.environ["DATABASE_URL_UNPOOLED"] = url.replace("-pooler", "")
os.environ["SECRET_KEY"] = "test-secret"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import psycopg  # noqa: E402

with psycopg.connect(os.environ["DATABASE_URL_UNPOOLED"], autocommit=True) as c:
    c.execute(open(os.path.join(ROOT, "lib", "schema.sql")).read())
    c.execute("DELETE FROM accounts; DELETE FROM login_attempts; DELETE FROM settings")

from api.index import app  # noqa: E402

ref = {c["name"]: c for c in json.load(open(os.path.join(ROOT, "tests", "ref", "reference.json")))}
b64 = lambda v: base64.b64encode(np.asarray(v, "<f4").tobytes()).decode()  # noqa: E731


def jpeg(name):
    c = ref[name]
    rgba = np.fromfile(os.path.join(ROOT, "tests", "ref", name + ".rgba"), np.uint8).reshape(c["h"], c["w"], 4)
    return base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)[:160, :160])[1]).decode()


def item(name):
    c = ref[name]
    it = {"face": {"emb": b64(c["faces"][0]["emb"]), "crop": jpeg(name)}, "warnings": []}
    if c["bodies"]:
        it["body"] = {"emb": b64(c["bodies"][0]["emb"]), "crop": jpeg(name)}
    return it


def near(v, cos, seed=0):
    """v ga berilgan kosinus o'xshashlikdagi tasodifiy vektor (yangi 'ko'rinish' simulyatsiyasi)."""
    v = np.asarray(v, np.float32)
    r = np.random.default_rng(seed).standard_normal(len(v)).astype(np.float32)
    r -= (r @ v) * v
    r /= np.linalg.norm(r)
    return cos * v + np.sqrt(1 - cos ** 2) * r


fails = 0


def check(ok, msg):
    global fails
    print(("OK   " if ok else "FAIL ") + msg)
    fails += 0 if ok else 1


A, D, U, X = app.test_client(), app.test_client(), app.test_client(), app.test_client()
J = lambda cl, m, p, d=None: getattr(cl, m)(p, json=d)  # noqa: E731

check(J(X, "get", "/api/auth/status").json["first"], "bo'sh bazada birinchi hisob kutilmoqda")
r = J(A, "post", "/api/auth/signup", {"full_name": "Test Admin", "username": "tadmin", "password": "adminpass1", "password2": "adminpass1"})
check(r.json.get("account", {}).get("role") == "admin", "birinchi sign up -> admin")
r = J(U, "post", "/api/auth/signup", {"full_name": "Test User", "username": "tuser", "password": "userpass1", "password2": "userpass1"})
check(r.json.get("account", {}).get("role") == "user", "ikkinchi sign up -> user")
r = J(A, "post", "/api/accounts", {"full_name": "Direktor", "username": "tdir", "password": "dirpass12", "role": "director"})
check(r.json.get("ok"), "admin direktor hisobini yaratdi")
check(J(D, "post", "/api/auth/login", {"username": "tdir", "password": "dirpass12"}).json.get("home") == "/dashboard", "direktor login -> /dashboard")
check(J(U, "post", "/api/auth/signup", {"full_name": "x", "username": "tuser", "password": "userpass1", "password2": "userpass1"}).status_code == 400, "band login rad etildi")

for who, cl, exp in [("admin", A, {"/api/stats": 200, "/api/users": 200, "/api/accounts": 200, "/api/settings": 200}),
                     ("direktor", D, {"/api/stats": 200, "/api/users": 403, "/api/accounts": 403, "/api/settings": 403}),
                     ("user", U, {"/api/stats": 403, "/api/users": 403, "/api/accounts": 403, "/api/attributes": 200}),
                     ("anonim", X, {"/api/stats": 401, "/api/me": 401})]:
    got = {p: J(cl, "get", p).status_code for p in exp}
    check(got == exp, f"{who} ruxsatlari: {got}")
check(J(D, "post", "/api/recognize", {"faces": []}).status_code == 403, "direktor kamerani ishlata olmaydi")

# ro'yxatdan o'tkazish
r = J(A, "post", "/api/users", {"name": "Lena", "source": "upload", "items": [item("lena")]})
check(r.status_code == 200 and r.json["stats"]["face_added"] == 1, f"Lena ro'yxatdan o'tdi: {r.json.get('stats')}")
lena_id = r.json["user"]["id"]
r = J(A, "post", "/api/users", {"name": "Lena 2", "source": "upload", "items": [item("lena")]})
check(r.status_code == 409 and r.json["duplicate"]["id"] == lena_id, "takroriy ro'yxat aniqlandi")
r = J(A, "post", f"/api/users/{lena_id}/images", {"items": [item("messi5")]})
check(r.json["stats"]["face_added"] == 0 and r.json["stats"]["conflicts"] == 1, "boshqa odam rasmi Lenaga qo'shilmadi")
r = J(A, "post", "/api/users", {"name": "Kam", "source": "register", "items": [item("messi5")]})
check(r.status_code == 422, "kamera orqali ro'yxatda 3 tadan kam yuz rad etildi")

# tanib olish (oddiy user)
e_lena = ref["lena"]["faces"][0]["emb"]
r = J(U, "post", "/api/recognize", {"faces": [{"emb": b64(e_lena), "det": 0.95, "size": 150, "attrs": ["hat"]}], "bodies": []})
check(r.json["faces"][0]["name"] == "Lena", f"Lena tanildi ({r.json['faces'][0]['score']}), {r.json['ms']} ms")
r = J(U, "post", "/api/recognize", {"faces": [{"emb": b64(ref["messi5"]["faces"][0]["emb"]), "det": 0.95, "size": 40}], "bodies": []})
check(r.json["faces"][0]["user_id"] is None, "Messi noma'lum")

# o'rganish: yangi ko'rinish (o'xshashlik 0.72)
v = near(e_lena, 0.72)
r = J(U, "post", "/api/recognize", {"faces": [{"emb": b64(v), "det": 0.95, "size": 150}], "bodies": []})
L = r.json["learn"]
check(len(L) == 1 and L[0]["kind"] == "face", f"server o'rganishga ruxsat berdi: {L}")
bad = dict(L[0], token="0" * 32)
check(J(U, "post", "/api/learn", {**bad, "emb": b64(v), "crop": jpeg("lena")}).json.get("ok") is False, "qalbaki token rad etildi")
check(J(U, "post", "/api/learn", {**L[0], "emb": b64(near(e_lena, 0.7, 1)), "crop": jpeg("lena")}).json.get("ok") is False,
      "token boshqa embedding bilan ishlamaydi")
r = J(U, "post", "/api/learn", {**L[0], "emb": b64(v), "crop": jpeg("lena")})
check(r.json.get("learned") is True, "yangi yuz ko'rinishi o'rganildi")
check(J(U, "post", "/api/learn", {**L[0], "emb": b64(v), "crop": jpeg("lena")}).json.get("learned") is False, "takroriy o'rganish yo'q (cooldown)")
r = J(U, "post", "/api/recognize", {"faces": [{"emb": b64(near(v / np.linalg.norm(v), 0.95, 5)), "det": 0.95, "size": 150}], "bodies": []})
check(r.json["learn"] == [], "o'rganilgan ko'rinishga yaqin kadr qayta o'rganilmaydi")

# tana: yuz bilan bog'langan tana -> o'rganish; yuzsiz tana -> tana bo'yicha taxmin
eb = ref["lena"]["bodies"][0]["emb"]
import time  # noqa: E402
time.sleep(2.1)
r = J(U, "post", "/api/recognize", {"faces": [{"emb": b64(e_lena), "det": 0.95, "size": 150}],
                                     "bodies": [{"emb": b64(near(eb, 0.8)), "faces": [0]}]})
check(r.json["bodies"][0]["name"] == "Lena" and any(x["kind"] == "body" for x in r.json["learn"]), "tana yuz orqali bog'landi va o'rganishga taklif qilindi")
r = J(U, "post", "/api/recognize", {"faces": [], "bodies": [{"emb": b64(near(eb, 0.95, 3)), "faces": []}]})
check(r.json["bodies"][0]["via"] == "body" and r.json["bodies"][0]["name"] == "Lena", f"yuzsiz tana bo'yicha tanildi ({r.json['bodies'][0]['score']})")

# jurnal va statistika
st = J(D, "get", "/api/stats").json
check(st["totals"]["recognized_today"] >= 1 and st["totals"]["learned_today"] >= 1 and st["totals"]["unknown_today"] >= 1,
      f"statistika: {({k: st['totals'][k] for k in ('users', 'recognized_today', 'unknown_today', 'learned_today')})}")
check(st["attrs_today"]["counts"].get("hat", 0) >= 1, "atribut (kepka) jurnalga yozildi")
u = J(A, "get", f"/api/users/{lena_id}").json
check(u["last_attrs"] == ["hat"] and u["attr_stats"]["counts"].get("hat") == 1, "Lena profilida atribut statistikasi")
sid = u["samples"][0]["id"]
img = A.get(f"/api/samples/{sid}/image")
check(img.status_code == 200 and img.data[:2] == b"\xff\xd8", "namuna rasmi bazadan o'qildi")
check(U.get(f"/api/samples/{sid}/image").status_code == 403, "oddiy user namuna rasmini ko'ra olmaydi")

# sozlamalar va parol
J(A, "post", "/api/settings", {"attrs": {"beard": 0.7}, "allow_signup": False})
check(J(U, "get", "/api/attributes").json["items"][3]["threshold"] == 0.7, "soqol chegarasi saqlandi va userga ham yetib bordi")
check(J(X, "post", "/api/auth/signup", {"full_name": "y", "username": "yyy", "password": "12345678", "password2": "12345678"}).status_code == 403,
      "Sign up yopilgan")
for _ in range(5):
    J(X, "post", "/api/auth/login", {"username": "tuser", "password": "xato"})
check("ko'p" in J(X, "post", "/api/auth/login", {"username": "tuser", "password": "userpass1"}).json["error"], "5 xatodan keyin blok")
uid = [a["id"] for a in J(A, "get", "/api/accounts").json["accounts"] if a["username"] == "tuser"][0]
J(A, "patch", f"/api/accounts/{uid}", {"password": "yangiparol1"})
check(J(U, "get", "/api/me").status_code == 401, "parol almashganda eski sessiya bekor bo'ldi")
me = J(A, "get", "/api/me").json
check(J(A, "patch", f"/api/accounts/{me['id']}", {"role": "user"}).status_code == 400, "admin o'zini pasaytira olmaydi")
check(J(A, "delete", f"/api/users/{lena_id}").json["ok"], "Lena o'chirildi")

print(f"\n{fails} ta xato" if fails else "\nHammasi o'tdi")
sys.exit(1 if fails else 0)
