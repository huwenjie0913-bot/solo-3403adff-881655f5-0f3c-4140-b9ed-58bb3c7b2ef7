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

from . import db, engine, imageutil, maskboard

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
    pid = db.create_project(
        name, f"/images/{fname}", w, h,
        base_exposure=float(request.form.get("base_exposure") or 10),
        magnification=float(request.form.get("magnification") or 1),
        aperture=(request.form.get("aperture") or "f/8").strip() or "f/8",
        paper_grade=int(request.form.get("paper_grade", 2)))
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
                         "kind": reg.get("kind"),
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
                         "kind": reg.get("kind"),
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


# ---------------------------------------------------------------- 实体遮挡板

def _load_json(s, default):
    try:
        return json.loads(s or "") or default
    except (json.JSONDecodeError, TypeError):
        return default


def _mask_target(plan, region_id):
    reg = next((r for r in plan.get("regions", [])
                if r.get("id") == region_id), None)
    return reg


@app.route("/projects/<int:pid>/maskboard")
def maskboard_page(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    settings = db.get_mask_settings(pid)
    cals = []
    for r in db.list_mask_calibrations():
        row = dict(r)
        row["fit"] = _load_json(row.get("fit"), {})
        cals.append(row)
    return render_template("maskboard.html", project=p,
                           settings=settings, calibrations=cals,
                           tool_version=maskboard.TOOL_VERSION)


@app.route("/api/projects/<int:pid>/mask-regions")
def api_mask_regions(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    plan = load_plan(p)
    out = []
    for r in plan.get("regions", []):
        target = maskboard.extract_target_loops(r, p)
        out.append({
            "id": r["id"], "name": r.get("name", "区域"),
            "kind": r.get("kind", "brush"),
            "feather": r.get("feather", 0),
            "outer": target["outer"], "holes": target["holes"],
            "area_mm2": target["area"], "multiple": target["multiple"],
            "n_points": len(r.get("points", [])),
            "n_strokes": len(r.get("strokes", [])),
        })
    return jsonify({"paper_mm": maskboard.paper_size_mm(p), "regions": out})


@app.route("/api/mask-calibrations")
def api_mask_cal_list():
    out = []
    for r in db.list_mask_calibrations():
        row = dict(r)
        row["fit"] = _load_json(row.get("fit"), {})
        out.append(row)
    return jsonify({"calibrations": out})


@app.route("/api/mask-calibrations/<int:cid>")
def api_mask_cal_get(cid):
    row = db.get_mask_calibration(cid)
    if not row:
        abort(404)
    row["points"] = _load_json(row["points"], [])
    row["fit"] = _load_json(row["fit"], {})
    return jsonify(row)


@app.route("/api/mask-calibrations", methods=["POST"])
def api_mask_cal_create():
    data = request.get_json(force=True)
    disc = float(data.get("disc_diameter") or 0)
    if disc <= 0:
        return jsonify({"error": "圆片直径必须大于 0"}), 400
    pts = data.get("points", [])
    fit = maskboard.fit_calibration(pts, disc)
    cid = db.create_mask_calibration(
        (data.get("name") or "圆片校准").strip(), disc, pts, fit,
        maskboard.TOOL_VERSION)
    return jsonify({"id": cid, "fit": fit})


@app.route("/api/mask-calibrations/<int:cid>", methods=["PUT"])
def api_mask_cal_update(cid):
    row = db.get_mask_calibration(cid)
    if not row:
        abort(404)
    data = request.get_json(force=True)
    fields = {}
    if "name" in data:
        fields["name"] = data["name"].strip() or "圆片校准"
    if "disc_diameter" in data:
        disc = float(data["disc_diameter"])
        if disc <= 0:
            return jsonify({"error": "圆片直径必须大于 0"}), 400
        fields["disc_diameter"] = disc
    if "points" in data:
        fields["points"] = data["points"]
    disc = fields.get("disc_diameter", row["disc_diameter"])
    pts = fields.get("points", _load_json(row["points"], []))
    fields["fit"] = maskboard.fit_calibration(pts, disc)
    db.update_mask_calibration(cid, **fields)
    return jsonify({"ok": True, "fit": fields["fit"]})


@app.route("/api/mask-calibrations/<int:cid>", methods=["DELETE"])
def api_mask_cal_delete(cid):
    db.delete_mask_calibration(cid)
    return jsonify({"ok": True})


@app.route("/api/mask-fit", methods=["POST"])
def api_mask_fit_preview():
    data = request.get_json(force=True)
    fit = maskboard.fit_calibration(data.get("points", []),
                                    float(data.get("disc_diameter", 0)))
    pred = None
    if fit["calibrated"] and data.get("height") is not None:
        pred = maskboard.predict(fit, float(data["height"]))
    return jsonify({"fit": fit, "predict": pred})


def _recompute_tool(spec, project, region, settings, fit=None,
                    force_backcalc=False):
    """按 spec 重算投影/问题。spec.outer 存在（已反算或已手改）时直接投影，
    否则先按高度/羽化从目标反算；force_backcalc=True 强制重新反算。"""
    target = maskboard.extract_target_loops(region, project)
    pred = maskboard.predict(fit, spec["height"])
    if pred is None:
        pred = {"scale": 1.0, "band": 0.0, "scale_err": 0.0,
                "band_err": 0.0, "in_range": False}
    if force_backcalc or not spec.get("outer"):
        calc = maskboard.back_calc(target, pred,
                                   float(spec.get("feather_comp", 0.35)))
    else:
        calc = _calc_from_spec(spec, pred)
    issues = maskboard.validate(
        spec, calc, target, project,
        settings["paper_w"], settings["paper_h"], fit)
    return target, pred, calc, issues


@app.route("/api/projects/<int:pid>/mask-tools")
def api_mask_tools(pid):
    rows = db.list_mask_tools(pid)
    for r in rows:
        r["spec"] = _load_json(r["spec"], {})
        r["result"] = _load_json(r["result"], {})
    return jsonify({"tools": rows})


@app.route("/api/projects/<int:pid>/mask-tools", methods=["POST"])
def api_mask_tool_create(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    data = request.get_json(force=True)
    plan = load_plan(p)
    region = _mask_target(plan, data.get("region_id"))
    if region is None:
        return jsonify({"error": "请选择关联区域"}), 400
    cal = db.get_mask_calibration(int(data.get("calibration_id") or 0))
    if not cal:
        return jsonify({"error": "请先选择投影校准"}), 400
    fit = _load_json(cal["fit"], {})
    settings = db.get_mask_settings(pid)
    h = float(data.get("height") or fit.get("h_min") or 10)
    spec = {
        "region_id": region["id"],
        "region_name": region.get("name", "区域"),
        "height": h,
        "sheet_w": float(data.get("sheet_w") or settings["paper_w"]),
        "sheet_h": float(data.get("sheet_h") or settings["paper_h"]),
        "min_bridge": float(data.get("min_bridge") or 5),
        "feather_comp": float(data.get("feather_comp", 0.35)),
        "outer": None, "holes": [],          # 首次由目标反算生成
        "handle": None,
    }
    target, pred, calc, issues = _recompute_tool(spec, p, region, settings, fit)
    spec["outer"] = calc["board"]["outer"]
    spec["holes"] = calc["board"]["holes"]
    # 默认手柄：沿 +x 方向，自动找接入点
    anchor = maskboard.handle_anchor_suggest(calc["board"]["outer"], 0)
    spec["handle"] = {"ax": anchor["x"], "ay": anchor["y"], "dir": 0,
                      "length": maskboard.HANDLE_L, "width": maskboard.HANDLE_W}
    _, _, calc, issues = _recompute_tool(spec, p, region, settings, fit)
    result = _tool_result(target, pred, calc, issues, fit)
    tid = db.create_mask_tool(
        pid, region["id"], (data.get("name") or
                            f"遮挡板·{region.get('name','区域')}").strip(),
        cal["id"], spec, result, maskboard.TOOL_VERSION)
    return jsonify({"id": tid, "spec": spec, "result": result})


def _tool_result(target, pred, calc, issues, fit):
    return {
        "target": target, "predict": pred, "board": calc["board"],
        "projected": calc["projected"],
        "error_mm": calc["error_mm"], "error_max_mm": calc["error_max_mm"],
        "scale": calc["scale"], "bbox": calc["bbox"],
        "feather_grow_mm": calc["feather_grow_mm"],
        "issues": issues,
        "fit_summary": {"beta": fit.get("beta"), "L": fit.get("L"),
                        "proj_rms": fit.get("proj_rms"),
                        "h_min": fit.get("h_min"), "h_max": fit.get("h_max")},
    }


@app.route("/api/mask-tools/<int:tid>", methods=["PUT"])
def api_mask_tool_update(tid):
    t = db.get_mask_tool(tid)
    if not t:
        abort(404)
    p = db.get_project(t["project_id"])
    plan = load_plan(p)
    spec = _load_json(t["spec"], {})
    data = request.get_json(force=True)
    if "spec" in data:
        spec.update(data["spec"])
    cal = db.get_mask_calibration(t["calibration_id"])
    fit = _load_json(cal["fit"], {}) if cal else {}
    settings = db.get_mask_settings(t["project_id"])
    region = _mask_target(plan, spec.get("region_id") or t.get("region_id"))
    if region is None:
        return jsonify({"error": "关联区域已不存在"}), 400
    target, pred, calc, issues = _recompute_tool(spec, p, region, settings, fit)
    result = _tool_result(target, pred, calc, issues, fit)
    fields = {"spec": spec, "result": result}
    if "name" in data:
        fields["name"] = data["name"].strip() or t["name"]
    db.update_mask_tool(tid, **fields)
    return jsonify({"ok": True, "spec": spec, "result": result})


def _calc_from_spec(spec, pred):
    """用户编辑过轮廓后，用 spec.outer 直接投影，不重新反算。"""
    board = {"outer": spec["outer"], "holes": spec.get("holes", [])}
    s = pred["scale"]
    center = maskboard.centroid(board["outer"])
    projected = {
        "outer": maskboard.scale_loop(board["outer"], s, center[0], center[1]),
        "holes": [maskboard.scale_loop(h, s, *maskboard.centroid(h))
                  for h in board["holes"]],
    }
    bb = maskboard.bbox_of([board["outer"]] + board["holes"])
    e = max(maskboard.ERROR_TOL_MM, abs(pred.get("scale_err", 0)) / s * 20)
    return {"board": board, "projected": projected,
            "error_mm": e, "error_max_mm": e, "scale": s, "bbox": bb,
            "feather_grow_mm": 0.0,
            "center": {"x": center[0], "y": center[1]}}


@app.route("/api/mask-tools/<int:tid>", methods=["DELETE"])
def api_mask_tool_delete(tid):
    db.delete_mask_tool(tid)
    return jsonify({"ok": True})


@app.route("/api/mask-tools/<int:tid>/recalc", methods=["POST"])
def api_mask_tool_recalc(tid):
    """按当前高度/校准重新反算（放弃轮廓手改）。"""
    t = db.get_mask_tool(tid)
    if not t:
        abort(404)
    p = db.get_project(t["project_id"])
    plan = load_plan(p)
    spec = _load_json(t["spec"], {})
    data = request.get_json(silent=True) or {}
    if "height" in data:
        spec["height"] = float(data["height"])
    if "min_bridge" in data:
        spec["min_bridge"] = float(data["min_bridge"])
    if "feather_comp" in data:
        spec["feather_comp"] = float(data["feather_comp"])
    if "sheet_w" in data:
        spec["sheet_w"] = float(data["sheet_w"])
    if "sheet_h" in data:
        spec["sheet_h"] = float(data["sheet_h"])
    cal = db.get_mask_calibration(t["calibration_id"])
    fit = _load_json(cal["fit"], {}) if cal else {}
    settings = db.get_mask_settings(t["project_id"])
    region = _mask_target(plan, spec.get("region_id") or t.get("region_id"))
    # 强制按当前高度/羽化重新反算（放弃轮廓手改），再写回轮廓与默认手柄
    target, pred, calc, issues = _recompute_tool(
        spec, p, region, settings, fit, force_backcalc=True)
    spec["outer"] = calc["board"]["outer"]
    spec["holes"] = calc["board"]["holes"]
    if spec.get("handle"):
        anchor = maskboard.handle_anchor_suggest(
            spec["outer"], spec["handle"].get("dir", 0))
        spec["handle"]["ax"] = anchor["x"]
        spec["handle"]["ay"] = anchor["y"]
    _, _, calc, issues = _recompute_tool(spec, p, region, settings, fit)
    result = _tool_result(target, pred, calc, issues, fit)
    db.update_mask_tool(tid, spec=spec, result=result)
    return jsonify({"ok": True, "spec": spec, "result": result})


@app.route("/api/projects/<int:pid>/mask-settings", methods=["PUT"])
def api_mask_settings(pid):
    if not db.get_project(pid):
        abort(404)
    data = request.get_json(force=True)
    db.update_mask_settings(pid, float(data["paper_w"]),
                            float(data["paper_h"]))
    return jsonify({"ok": True})


@app.route("/api/mask-tools/<int:tid>/versions")
def api_mask_tool_versions(tid):
    rows = db.list_mask_tool_versions(tid)
    return jsonify({"versions": rows})


@app.route("/api/mask-tools/<int:tid>/versions", methods=["POST"])
def api_mask_tool_version_save(tid):
    t = db.get_mask_tool(tid)
    if not t:
        abort(404)
    data = request.get_json(force=True)
    spec = _load_json(t["spec"], {})
    result = _load_json(t["result"], {})
    cal = db.get_mask_calibration(t["calibration_id"])
    cal_snap = _load_json(cal["fit"], {}) if cal else {}
    vid = db.save_mask_tool_version(
        tid, (data.get("label") or "遮挡板版本").strip(),
        (data.get("note") or "").strip(), spec, cal_snap, result)
    return jsonify({"id": vid})


@app.route("/api/mask-tool-versions/<int:vid>/restore", methods=["POST"])
def api_mask_tool_version_restore(vid):
    v = db.get_mask_tool_version(vid)
    if not v:
        abort(404)
    t = db.get_mask_tool(v["tool_id"])
    db.update_mask_tool(v["tool_id"],
                        spec=_load_json(v["spec"], {}),
                        result=_load_json(v["result"], {}))
    return jsonify({"ok": True, "tool_id": v["tool_id"]})


@app.route("/api/mask-tool-versions/<int:vid>", methods=["DELETE"])
def api_mask_tool_version_delete(vid):
    db.delete_mask_tool_version(vid)
    return jsonify({"ok": True})


@app.route("/api/mask-tools/compare")
def api_mask_compare():
    va = db.get_mask_tool_version(int(request.args.get("a", 0)))
    vb = db.get_mask_tool_version(int(request.args.get("b", 0)))
    if not va or not vb:
        abort(404)
    return jsonify(maskboard.compare_versions(
        {"spec": _load_json(va["spec"], {}),
         "result": _load_json(va["result"], {}),
         "calibration_snapshot": _load_json(va["calibration_snapshot"], {})},
        {"spec": _load_json(vb["spec"], {}),
         "result": _load_json(vb["result"], {}),
         "calibration_snapshot": _load_json(vb["calibration_snapshot"], {})}))


@app.route("/projects/<int:pid>/maskboard.svg")
def maskboard_svg(pid):
    p = db.get_project(pid)
    if not p:
        abort(404)
    settings = db.get_mask_settings(pid)
    sheet_w, sheet_h = settings["paper_w"], settings["paper_h"]
    id_param = request.args.get("tools")
    tools_all = db.list_mask_tools(pid)
    if id_param:
        wanted = {int(x) for x in id_param.split(",") if x.strip().isdigit()}
        tools_all = [t for t in tools_all if t["id"] in wanted]
    docs, nest_items = [], []
    for t in tools_all:
        spec = _load_json(t["spec"], {})
        result = _load_json(t["result"], {})
        bb = result.get("bbox") or maskboard.bbox_of(
            [spec.get("outer", [])] + spec.get("holes", []))
        hspec = spec.get("handle")
        if hspec and bb:
            hp = maskboard.handle_polygon(
                {"x": hspec["ax"], "y": hspec["ay"]}, hspec["dir"],
                hspec.get("length", maskboard.HANDLE_L),
                hspec.get("width", maskboard.HANDLE_W))
            hbb = maskboard.bbox_of([hp])
            bb = {"x": min(bb["x"], hbb["x"]), "y": min(bb["y"], hbb["y"]),
                  "w": max(bb["x"] + bb["w"], hbb["x"] + hbb["w"]) -
                       min(bb["x"], hbb["x"]),
                  "h": max(bb["y"] + bb["h"], hbb["y"] + hbb["h"]) -
                       min(bb["y"], hbb["y"])}
        if not bb:
            continue
        nest_items.append({"id": t["id"], "name": t["name"], "bbox": bb})
        docs.append({"tool": t, "spec": spec, "result": result,
                     "issues": result.get("issues", [])})
    if not docs:
        abort(404)
    nest = maskboard.nest_tools(nest_items, sheet_w, sheet_h)
    svg = maskboard.export_svg(docs, nest, sheet_w, sheet_h,
                               project_name=p["name"])
    resp = Response(svg, mimetype="image/svg+xml")
    resp.headers["Content-Disposition"] = (
        f'attachment; filename="maskboard-{pid}.svg"')
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
