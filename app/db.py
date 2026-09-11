# -*- coding: utf-8 -*-
"""SQLite 持久化：项目 / 版本快照 / 校准。"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "instance" / "darkroom.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    image_path      TEXT NOT NULL,
    image_w         INTEGER NOT NULL,
    image_h         INTEGER NOT NULL,
    magnification   REAL NOT NULL DEFAULT 1,
    aperture        TEXT NOT NULL DEFAULT 'f/8',
    base_exposure   REAL NOT NULL DEFAULT 10,
    paper_grade     INTEGER NOT NULL DEFAULT 2,
    negative_invert INTEGER NOT NULL DEFAULT 0,
    calibration     TEXT NOT NULL DEFAULT '{}',
    plan            TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    plan        TEXT NOT NULL,
    calibration TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_versions_project ON versions(project_id);
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ---------- projects ----------

def create_project(name, image_path, w, h, base_exposure=10.0,
                   magnification=1.0, aperture="f/8", paper_grade=2):
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO projects (name, image_path, image_w, image_h,
                                     base_exposure, magnification, aperture,
                                     paper_grade, plan, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)""",
            (name, str(image_path), w, h, base_exposure,
             magnification, aperture, paper_grade, now(), now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_projects():
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, name, image_w, image_h, magnification, aperture, "
            "base_exposure, paper_grade, updated_at FROM projects ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_project(pid):
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def update_project(pid, **fields):
    allowed = {
        "name", "magnification", "aperture", "base_exposure", "paper_grade",
        "negative_invert", "calibration", "plan",
    }
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k}=?")
            vals.append(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
    if not sets:
        return
    sets.append("updated_at=?")
    vals.append(now())
    vals.append(pid)
    conn = get_db()
    try:
        conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()
    finally:
        conn.close()


def delete_project(pid):
    conn = get_db()
    try:
        conn.execute("DELETE FROM projects WHERE id=?", (pid,))
        conn.commit()
    finally:
        conn.close()


# ---------- versions ----------

def save_version(pid, label, note, plan, calibration):
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO versions (project_id, label, note, plan, calibration, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (pid, label, note,
             json.dumps(plan, ensure_ascii=False),
             json.dumps(calibration, ensure_ascii=False), now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_versions(pid):
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, label, note, created_at FROM versions "
            "WHERE project_id=? ORDER BY id DESC", (pid,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_version(vid):
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def delete_version(vid):
    conn = get_db()
    try:
        conn.execute("DELETE FROM versions WHERE id=?", (vid,))
        conn.commit()
    finally:
        conn.close()
