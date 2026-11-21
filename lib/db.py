"""Postgres (Neon) ulanishi. Serverless funksiya "iliq" turganda ulanish qayta ishlatiladi."""
from __future__ import annotations

import os
import threading

import psycopg
from psycopg.rows import dict_row

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env() -> None:
    """Lokal ishga tushirishda .env ni o'qiydi (Vercelda o'zgaruvchilar paneldan keladi)."""
    path = os.path.join(ROOT, ".env")
    if os.environ.get("VERCEL") or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()

_lock = threading.RLock()
_conn: psycopg.Connection | None = None


def _connect() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL o'rnatilmagan")
    # prepare_threshold=None: PgBouncer (Neon pooler) bilan xavfsiz
    return psycopg.connect(url, autocommit=True, row_factory=dict_row, prepare_threshold=None, connect_timeout=15)


class DB:
    """Oddiy yordamchi: db.all(), db.one(), db.val(), db.run(), db.tx()."""

    def _c(self) -> psycopg.Connection:
        global _conn
        if _conn is None or _conn.closed:
            _conn = _connect()
        return _conn

    def _exec(self, fn):
        with _lock:
            try:
                return fn(self._c())
            except psycopg.OperationalError:
                # Neon compute uxlab qolgan yoki ulanish uzilgan bo'lishi mumkin -> bir marta qayta urinamiz
                global _conn
                try:
                    if _conn is not None:
                        _conn.close()
                finally:
                    _conn = None
                return fn(self._c())

    def all(self, sql: str, args=()) -> list[dict]:
        return self._exec(lambda c: c.execute(sql, args).fetchall())

    def one(self, sql: str, args=()) -> dict | None:
        return self._exec(lambda c: c.execute(sql, args).fetchone())

    def val(self, sql: str, args=()):
        r = self.one(sql, args)
        return None if r is None else next(iter(r.values()))

    def run(self, sql: str, args=()) -> int:
        return self._exec(lambda c: c.execute(sql, args).rowcount)

    def tx(self, fn):
        """fn(cursor) ni bitta tranzaksiyada bajaradi."""
        def go(c):
            with c.transaction():
                with c.cursor() as cur:
                    return fn(cur)
        return self._exec(go)


db = DB()
