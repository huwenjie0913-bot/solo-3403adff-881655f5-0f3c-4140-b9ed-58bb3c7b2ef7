# -*- coding: utf-8 -*-
"""实体遮挡板模块端到端冒烟：
校准拟合 → 区域轮廓提取 → 反算 → 五类问题检测 → 版本比较 → SVG 导出。
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.app import app as flask_app  # noqa: E402
from app import db, maskboard  # noqa: E402

TMP = Path("/tmp/darkroom_mask")
TMP.mkdir(exist_ok=True)
db.DB_PATH = TMP / "mask.db"
flask_app.config["TESTING"] = True
import app.app as ap  # noqa: E402
ap.UPLOAD_DIR = TMP / "images"
db.init_db()


def make_png():
    rng = np.linspace(0, 255, 320, dtype=np.uint8)[None, :].repeat(240, 0)
    buf = io.BytesIO()
    Image.fromarray(rng, "L").convert("RGB").save(buf, "PNG")
    buf.seek(0)
    return buf


def main():
    c = flask_app.test_client()
    r = c.post("/api/projects", data={
        "name": "遮挡板测试", "base_exposure": "10",
        "magnification": "2", "image": (make_png(), "neg.png")},
        content_type="multipart/form-data")
    pid = r.get_json()["id"]
    pw, ph = maskboard.paper_size_mm(db.get_project(pid))
    assert (pw, ph) == (640.0, 480.0), (pw, ph)
    print("paper mm:", pw, ph)

    # ---- 区域：一个大多边形 + 一个小多边形（后者用来制造窄桥/手柄冲突）
    plan = {"base_exposure": 10, "regions": [
        {"id": "r1", "kind": "polygon", "name": "窗光",
         "points": [{"x": .2, "y": .2}, {"x": .8, "y": .2},
                    {"x": .8, "y": .7}, {"x": .2, "y": .7}],
         "feather": 0, "size": 80, "strokes": []},
        {"id": "r2", "kind": "polygon", "name": "角部",
         "points": [{"x": .05, "y": .05}, {"x": .12, "y": .05},
                    {"x": .12, "y": .12}, {"x": .05, "y": .12}],
         "feather": 0, "size": 80, "strokes": []}],
        "steps": []}
    c.put(f"/api/projects/{pid}", json={"plan": plan})

    # 页面
    assert c.get(f"/projects/{pid}/maskboard").status_code == 200
    regs = c.get(f"/api/projects/{pid}/mask-regions").get_json()
    r1 = next(x for x in regs["regions"] if x["id"] == "r1")
    assert len(r1["outer"]) == 4, r1["outer"]
    # 区域宽 = 0.6*640 = 384mm
    xs = [p["x"] for p in r1["outer"]]
    assert abs(max(xs) - min(xs) - 384.0) < 1e-6
    print("target contour mm:", r1["outer"][:2])

    # ---- 校准拟合：圆片 D=20，两高度 s=1.1/1.3，带 3+0.02h
    pts = [{"h": 100, "pd": 22.0, "band": 5.0},
           {"h": 300, "pd": 26.0, "band": 9.0}]
    r = c.post("/api/mask-fit", json={
        "disc_diameter": 20, "points": pts, "height": 200})
    j = r.get_json()
    f = j["fit"]
    assert f["calibrated"] and abs(f["beta"] - 0.001) < 1e-12, f["beta"]
    assert abs(j["predict"]["scale"] - 1.2) < 1e-9
    print("fit beta:", f["beta"], "L:", f["L"], "proj_rms:", f["proj_rms"],
          "band:", f["w0"], f["gamma"])

    # 存校准
    cid = c.post("/api/mask-calibrations", json={
        "name": "测试圆片", "disc_diameter": 20, "points": pts}).get_json()["id"]
    assert len(c.get("/api/mask-calibrations").get_json()["calibrations"]) == 1
    cal_get = c.get(f"/api/mask-calibrations/{cid}").get_json()
    assert len(cal_get["points"]) == 2

    # ---- 创建工具（高度 200，在校准 100..300 内）
    r = c.post(f"/api/projects/{pid}/mask-tools", json={
        "region_id": "r1", "calibration_id": cid, "height": 200,
        "min_bridge": 5, "sheet_w": 600, "sheet_h": 500})
    assert r.status_code == 200, r.data
    created = r.get_json()
    tid = created["id"]
    spec, result = created["spec"], created["result"]
    assert abs(result["scale"] - 1.2) < 1e-9
    kinds = [i["kind"] for i in result["issues"]]
    print("initial issues:", kinds)
    assert "self_intersect" not in kinds
    assert "sheet_overflow" not in kinds
    # 板轮廓必须比目标小（s>1）
    tbb = result["bbox"]
    assert tbb["w"] < 384.0, tbb

    # ---- 超校准范围
    r = c.post(f"/api/mask-tools/{tid}/recalc", json={"height": 900})
    kinds = [i["kind"] for i in r.get_json()["result"]["issues"]]
    assert "height_out_of_range" in kinds, kinds
    # 复位
    c.post(f"/api/mask-tools/{tid}/recalc", json={"height": 200})

    # ---- 手柄压住目标：朝向指向区域内部时应报错
    t = db.get_mask_tool(tid)
    import json
    spec2 = json.loads(t["spec"])
    # 把手柄接入点放到外环内部并朝板内
    cx = sum(p["x"] for p in spec2["outer"]) / len(spec2["outer"])
    cy = sum(p["y"] for p in spec2["outer"]) / len(spec2["outer"])
    spec2["handle"] = {"ax": cx, "ay": cy, "dir": 0,
                       "length": 10, "width": 10}
    r = c.put(f"/api/mask-tools/{tid}", json={"spec": spec2})
    kinds = [i["kind"] for i in r.get_json()["result"]["issues"]]
    assert "handle_over_target" in kinds, kinds
    print("handle-over-target OK")

    # ---- 轮廓自交：拖动顶点使四边形变蝶形
    spec2["outer"] = [
        {"x": 128 + 100, "y": 96 + 60}, {"x": 512 - 100, "y": 240 + 60},
        {"x": 512 - 100, "y": 96 + 60}, {"x": 128 + 100, "y": 240 + 60}]
    # 默认外接手柄（朝 +x 贴边）避免干扰
    anc = maskboard.handle_anchor_suggest(spec2["outer"], 0)
    spec2["handle"] = {"ax": anc["x"], "ay": anc["y"], "dir": 0,
                       "length": 60, "width": 18}
    r = c.put(f"/api/mask-tools/{tid}", json={"spec": spec2})
    kinds = [i["kind"] for i in r.get_json()["result"]["issues"]]
    assert "self_intersect" in kinds, kinds
    print("self-intersect OK")

    # ---- 排版越界：巨大板材尺寸需求
    spec2["outer"] = [
        {"x": 0, "y": 0}, {"x": 700, "y": 0},
        {"x": 700, "y": 300}, {"x": 0, "y": 300}]
    spec2["handle"] = None
    r = c.put(f"/api/mask-tools/{tid}", json={"spec": spec2})
    kinds = [i["kind"] for i in r.get_json()["result"]["issues"]]
    assert "sheet_overflow" in kinds, kinds
    print("sheet-overflow OK")

    # ---- 窄桥：两块对顶楔形（用第二个工具关联 r2，手工造窄颈）
    r2_tool = c.post(f"/api/projects/{pid}/mask-tools", json={
        "region_id": "r2", "calibration_id": cid, "height": 200,
        "sheet_w": 600, "sheet_h": 500})
    tid2 = r2_tool.get_json()["id"]
    t2 = db.get_mask_tool(tid2)
    s2 = json.loads(t2["spec"])
    # 两块矩形由一条 2mm 的窄颈相连（哑铃形简单多边形）
    s2["outer"] = [
        {"x": 20, "y": 20}, {"x": 220, "y": 20},
        {"x": 220, "y": 100}, {"x": 111, "y": 100},
        {"x": 111, "y": 61}, {"x": 109, "y": 61},
        {"x": 109, "y": 100}, {"x": 20, "y": 100}]
    s2["handle"] = None
    s2["min_bridge"] = 5
    r = c.put(f"/api/mask-tools/{tid2}", json={"spec": s2})
    kinds = [i["kind"] for i in r.get_json()["result"]["issues"]]
    assert "narrow_bridge" in kinds, kinds
    print("narrow-bridge OK")

    # ---- 版本：保存两版并比较
    # 恢复 tid 为正常轮廓
    c.post(f"/api/mask-tools/{tid}/recalc", json={"height": 100})
    c.post(f"/api/mask-tools/{tid}/versions", json={"label": "v-low"})
    c.post(f"/api/mask-tools/{tid}/recalc", json={"height": 300})
    v2 = c.post(f"/api/mask-tools/{tid}/versions",
                json={"label": "v-high"}).get_json()["id"]
    vs = c.get(f"/api/mask-tools/{tid}/versions").get_json()["versions"]
    assert len(vs) == 2
    diff = c.get(f"/api/mask-tools/compare?a={vs[1]['id']}&b={vs[0]['id']}").get_json()
    # 列表按 id 倒序：vs[0] 为新版（h=300），vs[1] 为旧版（h=100）
    assert diff["height_delta"] == 200, diff
    assert diff["proj_hausdorff_mm"] >= 0
    print("compare:", diff)
    # 回滚
    assert c.post(f"/api/mask-tool-versions/{v2}/restore").status_code == 200

    # ---- SVG 导出
    svg = c.get(f"/projects/{pid}/maskboard.svg").data.decode()
    assert svg.startswith("<svg") and "比例校验框" in svg
    assert "50.00mm" in svg and "裁" not in svg  # 关键标注在
    assert "maskboard-1.0" in svg
    assert f"#{1}" in svg
    print("svg bytes:", len(svg))

    # ---- 显式像素→毫米基准：录入打印影像 320×240mm（无视倍率 2），
    # 区域多边形 0.2..0.8 → 宽 192mm；SVG 须注明基准，且 1:1 画布宽=板材宽
    r = c.put(f"/api/projects/{pid}/mask-settings", json={
        "paper_w": 600, "paper_h": 500, "print_w": 320, "print_h": 240})
    j = r.get_json()
    assert j["scale_source"] == "print_size"
    assert abs(j["paper_mm"][0] - 320) < 1e-9 and abs(j["paper_mm"][1] - 240) < 1e-9
    assert abs(j["px_per_mm"][0] - 1.0) < 1e-9  # 320px / 320mm
    regs = c.get(f"/api/projects/{pid}/mask-regions").get_json()
    r1b = next(x for x in regs["regions"] if x["id"] == "r1")
    w_mm = max(p["x"] for p in r1b["outer"]) - min(p["x"] for p in r1b["outer"])
    assert abs(w_mm - 0.6 * 320) < 1e-6, w_mm   # r1: x 0.2..0.8 → 192mm
    # 重新反算 tid（r1），板宽 = 目标宽 / s(h=200)=1.2
    rr = c.post(f"/api/mask-tools/{tid}/recalc", json={"height": 200,
                "sheet_w": 600, "sheet_h": 500}).get_json()
    bb = rr["result"]["bbox"]
    # 板宽 = 目标宽 / s，再加羽化内缩补偿（h=200 时 band=7mm，
    # grow=0.35*7/1.2≈2.04mm，径向外扩近似）→ 约 164mm
    assert 160.0 < bb["w"] < 165.0, bb
    svg2 = c.get(f"/projects/{pid}/maskboard.svg?tools={tid}").data.decode()
    assert 'width="600.0mm" height="500.0mm"' in svg2
    assert "320.0×240.0mm" in svg2 and "显式打印尺寸" in svg2
    # 50mm 比例框边长必须是 50 个 SVG 用户单位（=50mm）
    import re as _re
    m = _re.search(r'<rect class="frame" x="8(?:\.0+)?" y="[\d.]+" '
                   r'width="(50(?:\.0+)?)" height="\1"', svg2)
    assert m, "比例校验框不是 50mm 见方"
    print("explicit baseline OK:", round(w_mm, 1),
          "mm target,", round(bb["w"], 1), "mm board")

    # 删除工具 / 校准（SET NULL 不应报错）
    assert c.delete(f"/api/mask-tools/{tid}").status_code == 200
    assert c.delete(f"/api/mask-calibrations/{cid}").status_code == 200
    assert c.delete(f"/api/projects/{pid}").status_code == 200
    print("MASKBOARD ALL OK")


if __name__ == "__main__":
    main()
