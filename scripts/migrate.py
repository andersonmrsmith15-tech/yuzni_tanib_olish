"""lib/schema.sql ni Neon bazasiga qo'llaydi (to'g'ridan-to'g'ri ulanish orqali).

    python scripts/migrate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg  # noqa: E402

from lib.db import ROOT  # noqa: E402  (.env ni yuklaydi)

url = os.environ.get("DATABASE_URL_UNPOOLED") or os.environ["DATABASE_URL"]
with open(os.path.join(ROOT, "lib", "schema.sql"), encoding="utf-8") as f:
    sql = f.read()
with psycopg.connect(url, autocommit=True) as c:
    c.execute(sql)
    tables = [r[0] for r in c.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1")]
print("Sxema qo'llandi. Jadvallar:", ", ".join(tables))
