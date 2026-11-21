-- Yuz va tana tanib olish tizimi: Postgres (Neon) sxemasi.
-- Qo'llash: python scripts/migrate.py   (DATABASE_URL_UNPOOLED - to'g'ridan-to'g'ri ulanish ishlatiladi)
-- Har bir buyruq qayta ishga tushirilsa ham xavfsiz (IF NOT EXISTS).

-- Tanib olinadigan odamlar
CREATE TABLE IF NOT EXISTS users (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    phone            TEXT NOT NULL DEFAULT '',
    department       TEXT NOT NULL DEFAULT '',
    note             TEXT NOT NULL DEFAULT '',
    created          DOUBLE PRECISION NOT NULL,
    last_seen        DOUBLE PRECISION,
    seen_count       INTEGER NOT NULL DEFAULT 0,
    last_attrs       JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_learn_face  DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_learn_body  DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_logged      DOUBLE PRECISION NOT NULL DEFAULT 0
);

-- Yuz (128 float32) va tana (768 float32) namunalari + JPEG kesim
CREATE TABLE IF NOT EXISTS samples (
    id       TEXT PRIMARY KEY,
    user_id  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind     TEXT NOT NULL CHECK (kind IN ('face', 'body')),
    source   TEXT NOT NULL CHECK (source IN ('register', 'upload', 'auto')),
    ts       DOUBLE PRECISION NOT NULL,
    emb      BYTEA NOT NULL,
    image    BYTEA
);
CREATE INDEX IF NOT EXISTS idx_samples_user ON samples (user_id, kind);

-- Jurnal: recognized | unknown | learn | register | upload | delete | login | account
CREATE TABLE IF NOT EXISTS events (
    id         BIGSERIAL PRIMARY KEY,
    ts         DOUBLE PRECISION NOT NULL,
    type       TEXT NOT NULL,
    user_id    TEXT,
    user_name  TEXT,
    via        TEXT,
    score      DOUBLE PRECISION,
    text       TEXT,
    attrs      JSONB
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts);
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events (type, ts);
CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events (user_id, ts);

-- Tizimga kirish hisoblari
CREATE TABLE IF NOT EXISTS accounts (
    id             TEXT PRIMARY KEY,
    username       TEXT NOT NULL,
    full_name      TEXT NOT NULL,
    password_hash  TEXT NOT NULL,
    role           TEXT NOT NULL CHECK (role IN ('admin', 'director', 'user')),
    active         BOOLEAN NOT NULL DEFAULT TRUE,
    created        DOUBLE PRECISION NOT NULL,
    last_login     DOUBLE PRECISION,
    pw_version     INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_accounts_username ON accounts (lower(username));

-- Noto'g'ri kirish urinishlari (serverless: xotirada emas, bazada)
CREATE TABLE IF NOT EXISTS login_attempts (
    key  TEXT NOT NULL,
    ts   DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_attempts ON login_attempts (key, ts);

-- Admin sozlamalari (chegaralar, Sign up holati)
CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  JSONB NOT NULL
);

-- Hisoblagichlar: galereya versiyasi (keshni yangilash uchun) va h.k.
CREATE TABLE IF NOT EXISTS counters (
    name   TEXT PRIMARY KEY,
    value  DOUBLE PRECISION NOT NULL DEFAULT 0
);
INSERT INTO counters (name, value) VALUES ('gallery_version', 1), ('last_unknown_log', 0)
ON CONFLICT (name) DO NOTHING;
