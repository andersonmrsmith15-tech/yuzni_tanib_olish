"""Tanib olish va doimiy o'rganish mantig'i.

Qanday o'rganadi:
  * Har bir foydalanuvchida yuz va tana namunalari "galereyasi" bor.
  * Kamerada odam ishonchli tanilganda (yuz o'xshashligi yuqori va boshqa odamlardan aniq farq bor),
    yangi kadr galereyadagi namunalardan YETARLICHA FARQ QILSA (boshqa burchak, yorug'lik, ko'zoynak,
    soch turmagi va h.k.) - u yangi namuna sifatida qo'shiladi. Shu tariqa tizim odamni har xil
    sharoitdagi rasmlari orqali o'rganib boradi.
  * Yuz orqali tanilgan odamning tanasi (kiyimi, qomati) ham tana galereyasiga qo'shiladi, shuning uchun
    keyin yuzi ko'rinmasa ham tana bo'yicha taxminan taniladi.
  * Galereya to'lib qolsa, eng ko'p takrorlanadigan (boshqalarga eng o'xshash) avtomatik namuna
    o'chiriladi - xilma-xillik saqlanadi. Qo'lda qo'shilgan namunalar o'chirilmaydi.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .models import Body, Face, Models
from .store import Store, User


@dataclass
class Config:
    # SFace kosinus o'xshashligi (rasmiy chegarasi 0.363)
    face_match: float = 0.40  # shundan yuqori -> taniladi
    face_learn: float = 0.55  # shundan yuqori -> avtomatik o'rganish mumkin
    face_margin: float = 0.10  # 1-o'rin va 2-o'rin orasidagi minimal farq (o'rganish uchun)
    face_novelty: float = 0.85  # galereyadagi eng yaqin namunadan past bo'lsa -> "yangi ko'rinish"
    face_min_size: int = 60  # o'rganish uchun yuzning minimal o'lchami (piksel)
    face_min_det: float = 0.90  # o'rganish uchun detektor ishonchi
    max_face_samples: int = 80

    # Tana (ReID) kosinus o'xshashligi - kiyim o'zgarsa pasayadi, shuning uchun "taxminiy"
    body_match: float = 0.70
    body_novelty: float = 0.92
    max_body_samples: int = 40

    learn_cooldown: float = 2.0  # bir odam uchun ikki o'rganish orasidagi minimal vaqt (s)
    attr_smoothing: float = 0.35  # yangi kadr ulushi (0..1): kichik -> barqarorroq
    attr_min_face: int = 40  # atribut aniqlash uchun minimal yuz o'lchami
    attr_max_faces: int = 4  # bir kadrda nechta yuz uchun atribut hisoblanadi
    log_interval: float = 20.0  # bir odamni jurnalga necha soniyada bir yozish
    unknown_log_interval: float = 10.0


@dataclass
class Match:
    user_id: str | None
    score: float
    second: float


def _crop(img: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    x, y, w, h = box
    H, W = img.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return img[y0:y1, x0:x1].copy()


class Engine:
    def __init__(self, models: Models, store: Store, cfg: Config | None = None, attrs=None) -> None:
        self.m = models
        self.s = store
        self.cfg = cfg or Config()
        self.attrs = attrs  # AttributeDetector yoki None (model yo'q bo'lsa)
        self._attr_state: dict[str, tuple[np.ndarray, float]] = {}  # silliqlash uchun
        self._last_learn: dict[str, float] = {}
        self._last_log: dict[str, float] = {}  # jurnalni har kadrda to'ldirib yubormaslik uchun

    # ---------------- solishtirish ----------------
    def _match(self, emb: np.ndarray, kind: str) -> Match:
        mat, owners = self.s.gallery(kind)
        if not len(mat):
            return Match(None, 0.0, 0.0)
        sims = mat @ emb
        best: dict[str, float] = {}
        for uid, sim in zip(owners, sims):
            if sim > best.get(uid, -1.0):
                best[uid] = float(sim)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        uid, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return Match(uid, score, second)

    # ---------------- jonli tanib olish ----------------
    def recognize(self, img: np.ndarray, learn: bool = True, with_body: bool = True,
                  with_attrs: bool = True) -> dict:
        t0 = time.time()
        faces, bodies = self.m.analyze(img, with_body=with_body)
        cfg = self.cfg
        out_faces, out_bodies = [], []
        face_user: dict[int, tuple[str | None, float]] = {}

        learned_flags = []
        for i, f in enumerate(faces):
            m = self._match(f.emb, "face")
            uid = m.user_id if m.score >= cfg.face_match else None
            face_user[i] = (uid, m.score)
            learned = False
            if learn and uid and self._can_learn_face(f, m):
                learned = self._learn(uid, "face", f.emb, img, f.box, pad=0.25)
            learned_flags.append(learned)
        attrs = self._face_attributes(img, faces, face_user) if with_attrs else [None] * len(faces)
        for i, f in enumerate(faces):
            uid, score = face_user[i]
            j = self._face_json(f, uid, score, learned_flags[i])
            j["attrs"] = attrs[i]
            out_faces.append(j)

        for b in bodies:
            uid, score, via = None, 0.0, None
            linked = [face_user[i] for i in b.faces if face_user[i][0]]
            if linked:
                uid, score = max(linked, key=lambda t: t[1])
                via = "face"
                # Tanani faqat yuz ISHONCHLI tanilganda o'rganamiz (aks holda begona kiyimni o'rganib qo'yishi mumkin)
                if learn and score >= cfg.face_learn and len(b.faces) == 1:
                    self._learn(uid, "body", b.emb, img, b.box, pad=0.0)
            elif not b.faces:
                # Yuz ko'rinmayapti (orqasi o'girilgan, uzoqda) -> tana bo'yicha taxmin
                m = self._match(b.emb, "body")
                if m.score >= cfg.body_match and m.score - m.second >= 0.03:
                    uid, score, via = m.user_id, m.score, "body"
            out_bodies.append(self._body_json(b, uid, score, via))

        self._log_sightings(out_faces, out_bodies)
        events = self.s.recent_events(8, types=("learn",))
        return {"faces": out_faces, "bodies": out_bodies, "ms": int((time.time() - t0) * 1000),
                "events": events}

    def _face_attributes(self, img: np.ndarray, faces: list[Face], face_user: dict) -> list[dict | None]:
        """Ko'zoynak, kepka, soqol va h.k. Bir necha kadr bo'yicha silliqlanadi (natija sakramasligi uchun)."""
        out: list[dict | None] = [None] * len(faces)
        if not self.attrs or not faces:
            return out
        order = sorted(range(len(faces)), key=lambda i: -faces[i].box[2] * faces[i].box[3])
        order = [i for i in order if min(faces[i].box[2:]) >= self.cfg.attr_min_face][: self.cfg.attr_max_faces]
        if not order:
            return out
        probs = self.attrs.predict(img, [faces[i].box for i in order])
        now = time.time()
        self._attr_state = {k: v for k, v in self._attr_state.items() if now - v[1] < 5}
        a = self.cfg.attr_smoothing
        for j, i in enumerate(order):
            x, y, w, h = faces[i].box
            key = face_user[i][0] or f"pos:{int(x + w / 2) // 80}:{int(y + h / 2) // 80}"
            prev = self._attr_state.get(key)
            p = probs[j] if prev is None else (1 - a) * prev[0] + a * probs[j]
            self._attr_state[key] = (p, now)
            out[i] = self.attrs.to_json(p)
        return out

    def _log_sightings(self, out_faces: list[dict], out_bodies: list[dict]) -> None:
        now = time.time()
        best: dict[str, dict] = {}
        for r in out_faces + out_bodies:
            if r["user_id"] and (r["user_id"] not in best or r["score"] > best[r["user_id"]]["score"]):
                best[r["user_id"]] = r
        for uid, r in best.items():
            u = self.s.users.get(uid)
            if not u:
                continue
            if now - self._last_log.get(uid, 0) >= self.cfg.log_interval:
                self._last_log[uid] = now
                keys = [a["key"] for a in r["attrs"]["present"]] if r.get("attrs") else None
                u.seen_count += 1
                u.last_seen = now
                if keys is not None:
                    u.last_attrs = keys
                self.s.save(u)
                self.s.log("recognized", uid, u.name, via=r.get("via") or "face", score=r["score"], attrs=keys)
            else:
                u.last_seen = now
                if r.get("attrs"):
                    u.last_attrs = [a["key"] for a in r["attrs"]["present"]]
        unknown = [f for f in out_faces if not f["user_id"]]
        if unknown and now - self._last_log.get("__unknown__", 0) >= self.cfg.unknown_log_interval:
            self._last_log["__unknown__"] = now
            self.s.log("unknown", None, None, via="face", score=max(f["score"] for f in unknown),
                       text=f"{len(unknown)} ta noma'lum yuz")

    def _can_learn_face(self, f: Face, m: Match) -> bool:
        c = self.cfg
        return (
            m.score >= c.face_learn
            and m.score - m.second >= c.face_margin
            and f.score >= c.face_min_det
            and min(f.box[2], f.box[3]) >= c.face_min_size
        )

    def _learn(self, uid: str, kind: str, emb: np.ndarray, img, box, pad: float) -> bool:
        c = self.cfg
        now = time.time()
        key = f"{uid}:{kind}"
        if now - self._last_learn.get(key, 0) < c.learn_cooldown:
            return False
        u = self.s.users.get(uid)
        if not u:
            return False
        gal = u.face if kind == "face" else u.body
        novelty_thr = c.face_novelty if kind == "face" else c.body_novelty
        if len(gal) and float(np.max(gal @ emb)) >= novelty_thr:
            return False  # bunaqa ko'rinish allaqachon bor
        self._last_learn[key] = now
        self.s.add_sample(u, kind, emb, "auto", _crop(img, box, pad))
        self._prune(u, kind)
        self.s.save(u)
        self.s.log("learn", u.id, u.name, via=kind,
                   text=f"{u.name}: yangi {'yuz' if kind == 'face' else 'tana'} ko'rinishi o'rganildi")
        return True

    def _prune(self, u: User, kind: str) -> None:
        c = self.cfg
        limit = c.max_face_samples if kind == "face" else c.max_body_samples
        meta = u.face_meta if kind == "face" else u.body_meta
        while len(meta) > limit:
            gal = u.face if kind == "face" else u.body
            auto_idx = [i for i, s in enumerate(meta) if not s.anchor]
            if not auto_idx:
                break
            if kind == "body":
                # kiyim o'zgarib turadi -> eng eski avtomatik tana namunasini o'chiramiz
                victim = min(auto_idx, key=lambda i: meta[i].ts)
            else:
                sims = gal @ gal.T
                np.fill_diagonal(sims, 0)
                redundancy = sims.mean(axis=1)
                victim = max(auto_idx, key=lambda i: redundancy[i])
            self.s.remove_sample(u, kind, victim)

    # ---------------- ro'yxatdan o'tkazish / rasm qo'shish ----------------
    def _main_subject(self, img: np.ndarray) -> tuple[Face | None, Body | None, list[str]]:
        faces, bodies = self.m.analyze(img, with_body=True)
        warn = []
        face = max(faces, key=lambda f: f.box[2] * f.box[3]) if faces else None
        if len(faces) > 1:
            warn.append("rasmda bir nechta yuz bor - eng kattasi olindi")
        body = None
        if face is not None:
            fi = faces.index(face)
            body = next((b for b in bodies if fi in b.faces), None)
        elif bodies:
            body = max(bodies, key=lambda b: b.box[2] * b.box[3])
        return face, body, warn

    def find_duplicate(self, embs: list[np.ndarray], exclude: str | None = None) -> tuple[User | None, float]:
        if not embs:
            return None, 0.0
        scores: dict[str, list[float]] = {}
        for e in embs:
            m = self._match(e, "face")
            if m.user_id and m.user_id != exclude:
                scores.setdefault(m.user_id, []).append(m.score)
        best_uid, best = None, 0.0
        for uid, sc in scores.items():
            avg = float(np.mean(sc)) * len(sc) / len(embs)
            if avg > best:
                best_uid, best = uid, avg
        return (self.s.users.get(best_uid) if best_uid else None), best

    def add_images(self, u: User, images: list[np.ndarray], source: str) -> dict:
        stats = {"face_added": 0, "body_added": 0, "skipped": 0, "warnings": [], "conflicts": 0}
        for n, img in enumerate(images, 1):
            face, body, warn = self._main_subject(img)
            stats["warnings"] += [f"{n}-rasm: {w}" for w in warn]
            if face is None and body is None:
                stats["skipped"] += 1
                stats["warnings"].append(f"{n}-rasm: odam topilmadi")
                continue
            if face is not None:
                # boshqa odamning rasmini adashib qo'shishdan himoya
                if len(u.face):
                    m = self._match(face.emb, "face")
                    own = float(np.max(u.face @ face.emb))
                    if m.user_id and m.user_id != u.id and m.score >= self.cfg.face_learn and m.score > own + 0.1:
                        other = self.s.users[m.user_id].name
                        stats["conflicts"] += 1
                        stats["warnings"].append(f"{n}-rasm: bu yuz ko'proq '{other}'ga o'xshaydi - qo'shilmadi")
                        continue
                    if own < self.cfg.face_match - 0.1:
                        stats["conflicts"] += 1
                        stats["warnings"].append(f"{n}-rasm: yuz '{u.name}'ga o'xshamaydi "
                                                 f"(o'xshashlik {own:.2f}) - qo'shilmadi")
                        continue
                self.s.add_sample(u, "face", face.emb, source, _crop(img, face.box, 0.25))
                stats["face_added"] += 1
            if body is not None:
                self.s.add_sample(u, "body", body.emb, source, _crop(img, body.box))
                stats["body_added"] += 1
        if source == "upload":
            self.s.log("upload", u.id, u.name,
                       text=f"{u.name}: {stats['face_added']} yuz, {stats['body_added']} tana namunasi yuklandi")
        return stats

    def register(self, name: str, images: list[np.ndarray], source: str, force: bool = False,
                 profile: dict | None = None) -> dict:
        # Avval yuzlarni hisoblab, takroriy ro'yxatni tekshiramiz
        face_embs = []
        for img in images:
            face, _, _ = self._main_subject(img)
            if face is not None:
                face_embs.append(face.emb)
        min_faces = 3 if source == "register" else 1
        if len(face_embs) < min_faces:
            return {"ok": False, "error": f"Yuz yetarlicha aniqlanmadi ({len(face_embs)} ta, kamida {min_faces} ta kerak). "
                                          "Yorug'roq joyda, kameraga yaqinroq turing."}
        dup, score = self.find_duplicate(face_embs)
        if dup and score >= self.cfg.face_learn and not force:
            return {"ok": False, "duplicate": {"id": dup.id, "name": dup.name, "score": round(score, 3)},
                    "error": f"Bu odam allaqachon '{dup.name}' nomi bilan ro'yxatda bo'lishi mumkin "
                             f"(o'xshashlik {score:.2f})."}
        p = profile or {}
        u = self.s.create_user(name, p.get("phone", ""), p.get("department", ""), p.get("note", ""))
        stats = self.add_images(u, images, source)
        self.s.log("register", u.id, u.name, text=f"{u.name} ro'yxatdan o'tdi ({stats['face_added']} yuz, "
                                                   f"{stats['body_added']} tana namunasi)")
        return {"ok": True, "user": u.summary(), "stats": stats}

    # ---------------- JSON ----------------
    def _name(self, uid: str | None) -> str | None:
        u = self.s.users.get(uid) if uid else None
        return u.name if u else None

    def _face_json(self, f: Face, uid, score, learned) -> dict:
        return {"box": list(f.box), "det": round(f.score, 3), "user_id": uid, "name": self._name(uid),
                "score": round(float(score), 3), "learned": learned}

    def _body_json(self, b: Body, uid, score, via) -> dict:
        return {"box": list(b.box), "det": round(b.score, 3), "user_id": uid, "name": self._name(uid),
                "score": round(float(score), 3), "via": via}
