# -*- coding: utf-8 -*-
"""Flask 路由：项目、校准、引擎计算、版本、操作单导出。全部本地离线。"""
import base64
import io
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
from flask import (Flask, Response, abort, jsonify, redirect, render_template,
                   request, send_from_directory, url_for)

from . import db, engine, imageutil

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent / "instance" / "images"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

db.init_db()


def b64(data, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def load_plan(p):
    try:
        return json.loads(p.get("plan") or "{}")
    except json.JSONDecodeError:
        return {}


def load_cal(p):
    try:
        cal = json.loads(p.get("calibration") or "{}")
    except json.JSONDecodeError:
        cal = {}
    return engine.normalize_calibration(cal, p.get("paper_grade", 2),
                                        p.get("base_exposure", 10))


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- pages

@app.route("/")
def index():
    return render_template("index.html", projects=db.list_projects())


@app.route("/projects/<int:pid>")
def project_page(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    return render_template("project.html", project=p)


# ---------------------------------------------------------------- project API

@app.route("/api/projects", methods=["POST"])
def api_create():
    f = request.files.get("image")
    name = (request.form.get("name") or "未命名方案").strip()
    if not f:
        return jsonify({"error": "缺少底片扫描图"}), 400
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fname = uuid.uuid4().hex + ".png"
    fpath = UPLOAD_DIR / fname
    # 先经 Pillow 打开验证，再存为 PNG
    tmp = UPLOAD_DIR / ("tmp_" + fname)
    f.save(str(tmp))
    try:
        img = imageutil.load_image(str(tmp))
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        img.save(str(fpath))
        w, h = img.size
    finally:
        tmp.exists() and tmp.unlink()
    pid = db.create_project(name, f"/images/{fname}", w, h,
                            base_exposure=float(request.form.get("base_exposure") or 10))
    return jsonify({"id": pid, "redirect": url_for("project_page", pid=pid)})


@app.route("/api/projects/<int:pid>")
def api_get(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    p["plan"] = load_plan(p)
    p["calibration"] = load_cal(p)
    return jsonify(p)


@app.route("/api/projects/<int:pid>", methods=["PUT"])
def api_update(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    data = request.get_json(force=True)
    fields = {}
    for k in ("name", "magnification", "aperture", "base_exposure",
              "paper_grade", "negative_invert", "calibration", "plan"):
        if k in data:
            fields[k] = data[k]
    db.update_project(pid, **fields)
    return jsonify({"ok": True})


@app.route("/api/projects/<int:pid>", methods=["DELETE"])
def api_delete(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    # 删除图片文件
    try:
        fpath = UPLOAD_DIR / Path(p["image_path"]).name
        if fpath.exists():
            fpath.unlink()
    except OSError:
        pass
    db.delete_project(pid)
    return jsonify({"ok": True})


@app.route("/images/<path:fname>")
def serve_image(fname):
    return send_from_directory(str(UPLOAD_DIR), fname)


# ---------------------------------------------------------------- calibration

@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    data = request.get_json(force=True)
    result = engine.fit_curve(
        data.get("points", []),
        grade=int(data.get("paper_grade", 2)),
        base_exposure=float(data.get("base_exposure", 10)),
    )
    out = {
        "params": result["params"],
        "points": data.get("points", []),
        "fitted": result["fitted"],
        "range": result["range"],
        "rms": result["rms"],
        "n": result["n"],
        "calibrated": result["n"] >= 2 and result["rms"] is not None,
    }
    return jsonify(out)


# ---------------------------------------------------------------- compute

def _run_compute(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    plan = load_plan(p)
    plan.setdefault("base_exposure", p["base_exposure"])
    cal = load_cal(p)
    _, _, gray0 = imageutil.base_map_from_file(
        UPLOAD_DIR / Path(p["image_path"]).name,
        bool(p["negative_invert"]))
    return p, plan, cal, engine.compute(gray0, plan, cal)


@app.route("/api/projects/<int:pid>/compute")
def api_compute(pid):
    # 该端点供导出/校验使用；日常预览在浏览器 JS 端完成。
    p, plan, cal, res = _run_compute(pid)
    H, W = res["gray"].shape
    step = max(1, round(max(H, W) / 480))
    gray = res["gray"][::step, ::step]
    return jsonify({
        "gray": np.round(gray).astype(np.uint8).tolist(),
        "warnings": res["warnings"],
        "timeline_end": res["timeline_end"],
        "intervals": res["intervals"],
    })


# ---------------------------------------------------------------- versions

@app.route("/api/projects/<int:pid>/versions")
def api_versions(pid):
    return jsonify(db.list_versions(pid))


@app.route("/api/projects/<int:pid>/versions", methods=["POST"])
def api_save_version(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    data = request.get_json(force=True)
    vid = db.save_version(
        pid, (data.get("label") or "版本").strip(),
        (data.get("note") or "").strip(), load_plan(p), load_cal(p))
    return jsonify({"id": vid})


@app.route("/api/versions/<int:vid>", methods=["DELETE"])
def api_delete_version(vid):
    db.delete_version(vid)
    return jsonify({"ok": True})


@app.route("/api/versions/<int:vid>/restore", methods=["POST"])
def api_restore_version(vid):
    v = db.get_version(vid)
    if not v:
        abort(404)
    db.update_project(v["project_id"],
                      plan=json.loads(v["plan"]),
                      calibration=json.loads(v["calibration"]))
    return jsonify({"ok": True, "project_id": v["project_id"]})


# ---------------------------------------------------------------- export

def _stopwatch_nodes(plan, intervals, total):
    """生成秒表节点：每一步开始/结束的累计时刻 + 动作。"""
    base = max(float(plan.get("base_exposure", 10)), 0.01)
    nodes = [(0.0, "开始基础曝光（镜头下全部区域）")]
    acts = []
    regions = {r["id"]: r for r in plan.get("regions", [])}
    for s in plan.get("steps", []):
        reg = regions.get(s.get("region_id"), {})
        nm = reg.get("name", "未命名区域")
        if s.get("type") == "dodge":
            dur = base * float(s.get("ratio", 0))
            t0 = float(s.get("start", 0))
            tool_sz = reg.get("size", 0) or 0
            tool_desc = f"{tool_sz:.0f}px 工具，" if reg.get("kind") == "brush" else "多边形板，"
            acts.append((t0, t0 + dur,
                         f"遮挡：{nm}（{tool_desc}持续 {dur:.1f}s）"))
        else:
            extra = base * (2.0 ** float(s.get("stops", 0)) - 1.0)
            t0 = float(s.get("start", base))
            acts.append((t0, t0 + extra, f"加光：{nm}（+{s.get('stops',0):g} 档，持续 {extra:.1f}s）"))
    for a, b, text in sorted(acts, key=lambda z: z[0]):
        nodes.append((round(a, 1), text + " —— 开始"))
        nodes.append((round(b, 1), text + " —— 结束"))
    nodes.append((round(total, 1), "结束 / 移开相纸"))
    nodes.sort(key=lambda z: z[0])
    return nodes


@app.route("/projects/<int:pid>/print-sheet")
def print_sheet(pid):
    p, plan, cal, res = _run_compute(pid)
    regions = {r["id"]: r for r in plan.get("regions", [])}

    thumbs = []
    overlay_covs = []
    for i, s in enumerate(plan.get("steps", [])):
        rid = s.get("region_id")
        reg = regions.get(rid)
        cov = res["masks"].get(rid)
        if reg is None or cov is None:
            continue
        kind = "遮挡" if s["type"] == "dodge" else "加光"
        label = f"{kind} · {reg.get('name','区域')}"
        thumbs.append({
            "label": label,
            "img": b64(imageutil.mask_thumb(cov, label)),
            "feather": reg.get("feather", 0),
            "size": reg.get("size", 0),
            "kind": reg.get("kind"),
        })
        overlay_covs.append(cov)

    result_img = b64(imageutil.result_png(
        res["gray"], res["base_map"], overlays=overlay_covs))
    base_img = b64(imageutil.result_png(res["base_map"], res["base_map"]))

    # 阶梯曝光数据
    points = cal.get("points", [])
    fitted = cal.get("fitted", [])

    nodes = _stopwatch_nodes(plan, res["intervals"], res["timeline_end"])
    base_exp = max(float(plan.get("base_exposure", 10)), 0.01)
    rows = []
    for s in plan.get("steps", []):
        reg = regions.get(s.get("region_id"), {})
        if s["type"] == "dodge":
            dur = base_exp * float(s.get("ratio", 0))
            rows.append({"n": len(rows) + 1, "type": "遮挡",
                         "region": reg.get("name", "?"),
                         "start": float(s.get("start", 0)), "dur": dur,
                         "param": f"遮挡 {float(s.get('ratio',0))*100:.0f}%",
                         "feather": reg.get("feather", 0),
                         "size": reg.get("size", 0),
                         "end": float(s.get("start", 0)) + dur})
        else:
            extra = base_exp * (2.0 ** float(s.get("stops", 0)) - 1.0)
            t0 = float(s.get("start", base_exp))
            rows.append({"n": len(rows) + 1, "type": "加光",
                         "region": reg.get("name", "?"),
                         "start": t0, "dur": extra,
                         "param": f"+{s.get('stops',0):g} 档",
                         "feather": reg.get("feather", 0),
                         "size": reg.get("size", 0),
                         "end": t0 + extra})

    return render_template(
        "print_sheet.html", p=p, plan=plan, cal=cal,
        points=points, fitted=fitted,
        warnings=res["warnings"], thumbs=thumbs,
        result_img=result_img, base_img=base_img,
        nodes=nodes, rows=rows, total=res["timeline_end"],
        now_ts=db.now())


@app.route("/api/projects/<int:pid>/export.json")
def export_json(pid):
    p, plan, cal, res = _run_compute(pid)
    payload = {
        "project": {k: p[k] for k in
                    ("id", "name", "magnification", "aperture",
                     "base_exposure", "paper_grade", "image_w", "image_h")},
        "plan": plan,
        "calibration": cal,
        "stopwatch": [{"t": t, "action": a}
                      for t, a in _stopwatch_nodes(plan, res["intervals"],
                                                   res["timeline_end"])],
        "warnings": res["warnings"],
        "exported_at": db.now(),
    }
    resp = Response(json.dumps(payload, ensure_ascii=False, indent=2),
                    mimetype="application/json")
    resp.headers["Content-Disposition"] = (
        f'attachment; filename="darkroom-plan-{pid}.json"')
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
