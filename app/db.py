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

-- 实体遮挡板模块 ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS mask_settings (
    project_id  INTEGER PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    paper_w     REAL NOT NULL DEFAULT 203,
    paper_h     REAL NOT NULL DEFAULT 254,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mask_calibrations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    disc_diameter REAL NOT NULL,              -- 已知圆片实际直径(mm)
    points        TEXT NOT NULL DEFAULT '[]', -- [{h,pd,band}] 高度/投影直径/过渡带宽(mm)
    fit           TEXT NOT NULL DEFAULT '{}', -- 拟合结果 {L,beta,w0,rms,...}
    tool_version  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mask_tools (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    region_id      TEXT,
    name           TEXT NOT NULL,
    calibration_id INTEGER REFERENCES mask_calibrations(id) ON DELETE SET NULL,
    spec           TEXT NOT NULL DEFAULT '{}',
    result         TEXT NOT NULL DEFAULT '{}',
    tool_version   TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mask_tools_project ON mask_tools(project_id);

CREATE TABLE IF NOT EXISTS mask_tool_versions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_id              INTEGER NOT NULL REFERENCES mask_tools(id) ON DELETE CASCADE,
    label                TEXT NOT NULL,
    note                 TEXT NOT NULL DEFAULT '',
    spec                 TEXT NOT NULL,
    calibration_snapshot TEXT NOT NULL DEFAULT '{}',
    result               TEXT NOT NULL DEFAULT '{}',
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mask_tv_tool ON mask_tool_versions(tool_id);
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


# ---------- 实体遮挡板：设置 ----------

def get_mask_settings(pid):
    conn = get_db()
    try:
        r = conn.execute(
            "SELECT * FROM mask_settings WHERE project_id=?", (pid,)).fetchone()
        if r:
            return dict(r)
        return {"project_id": pid, "paper_w": 203.0, "paper_h": 254.0}
    finally:
        conn.close()


def update_mask_settings(pid, paper_w, paper_h):
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO mask_settings (project_id, paper_w, paper_h, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(project_id) DO UPDATE SET
                 paper_w=excluded.paper_w, paper_h=excluded.paper_h,
                 updated_at=excluded.updated_at""",
            (pid, paper_w, paper_h, now()))
        conn.commit()
    finally:
        conn.close()


# ---------- 实体遮挡板：校准 ----------

def list_mask_calibrations():
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, name, disc_diameter, fit, tool_version, created_at, updated_at "
            "FROM mask_calibrations ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_mask_calibration(cid):
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM mask_calibrations WHERE id=?",
                         (cid,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def create_mask_calibration(name, disc_diameter, points, fit, tool_version):
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO mask_calibrations
                 (name, disc_diameter, points, fit, tool_version, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, disc_diameter,
             json.dumps(points, ensure_ascii=False),
             json.dumps(fit, ensure_ascii=False), tool_version, now(), now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_mask_calibration(cid, **fields):
    allowed = {"name", "disc_diameter", "points", "fit"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k}=?")
            vals.append(json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (dict, list)) else v)
    if not sets:
        return
    sets.append("updated_at=?")
    vals.extend([now(), cid])
    conn = get_db()
    try:
        conn.execute(
            f"UPDATE mask_calibrations SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()
    finally:
        conn.close()


def delete_mask_calibration(cid):
    conn = get_db()
    try:
        conn.execute("DELETE FROM mask_calibrations WHERE id=?", (cid,))
        conn.commit()
    finally:
        conn.close()


# ---------- 实体遮挡板：工具 ----------

def list_mask_tools(pid):
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, project_id, region_id, name, calibration_id, spec, "
            "result, tool_version, created_at, updated_at "
            "FROM mask_tools WHERE project_id=? ORDER BY id", (pid,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_mask_tool(tid):
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM mask_tools WHERE id=?",
                         (tid,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def create_mask_tool(pid, region_id, name, calibration_id, spec, result,
                     tool_version):
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO mask_tools
                 (project_id, region_id, name, calibration_id, spec, result,
                  tool_version, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pid, region_id, name, calibration_id,
             json.dumps(spec, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False),
             tool_version, now(), now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_mask_tool(tid, **fields):
    allowed = {"region_id", "name", "calibration_id", "spec", "result"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k}=?")
            vals.append(json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (dict, list)) else v)
    if not sets:
        return
    sets.append("updated_at=?")
    vals.extend([now(), tid])
    conn = get_db()
    try:
        conn.execute(
            f"UPDATE mask_tools SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()
    finally:
        conn.close()


def delete_mask_tool(tid):
    conn = get_db()
    try:
        conn.execute("DELETE FROM mask_tools WHERE id=?", (tid,))
        conn.commit()
    finally:
        conn.close()


# ---------- 实体遮挡板：工具版本 ----------

def list_mask_tool_versions(tid):
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, tool_id, label, note, created_at FROM mask_tool_versions "
            "WHERE tool_id=? ORDER BY id DESC", (tid,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_mask_tool_version(vid):
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM mask_tool_versions WHERE id=?",
                         (vid,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def save_mask_tool_version(tid, label, note, spec, calibration_snapshot, result):
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO mask_tool_versions
                 (tool_id, label, note, spec, calibration_snapshot, result, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (tid, label, note,
             json.dumps(spec, ensure_ascii=False),
             json.dumps(calibration_snapshot, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False), now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def delete_mask_tool_version(vid):
    conn = get_db()
    try:
        conn.execute("DELETE FROM mask_tool_versions WHERE id=?", (vid,))
        conn.commit()
    finally:
        conn.close()
