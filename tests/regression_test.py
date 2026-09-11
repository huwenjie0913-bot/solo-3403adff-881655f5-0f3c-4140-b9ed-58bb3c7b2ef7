# -*- coding: utf-8 -*-
"""三处已复核缺陷的回归测试：
1. 新建方案持久化放大倍率/光圈/反差号并能读回；
2. 加光档数换算统一为 2^stops（10s 基础 + 1 档 = 追加 10s）；
3. 打印操作单步骤行带 kind，画笔步骤显示直径而非走多边形分支。
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.app import app as flask_app  # noqa: E402
from app import db  # noqa: E402

TMP = Path("/tmp/darkroom_regress")
TMP.mkdir(exist_ok=True)
db.DB_PATH = TMP / "regress.db"
import app.app as ap  # noqa: E402
ap.UPLOAD_DIR = TMP / "images"
flask_app.config["TESTING"] = True
db.init_db()


def make_png():
    rng = np.linspace(0, 255, 320, dtype=np.uint8)[None, :].repeat(240, 0)
    buf = io.BytesIO()
    Image.fromarray(rng, "L").convert("RGB").save(buf, "PNG")
    buf.seek(0)
    return buf


def main():
    c = flask_app.test_client()

    # ---- 缺陷 1：建项目时的倍率/光圈/反差号持久化与读回
    r = c.post("/api/projects", data={
        "name": "回归方案", "base_exposure": "10",
        "magnification": "5", "aperture": "f/16", "paper_grade": "4",
        "image": (make_png(), "neg.png"),
    }, content_type="multipart/form-data")
    assert r.status_code == 200, r.data
    pid = r.get_json()["id"]

    p = c.get(f"/api/projects/{pid}").get_json()
    assert float(p["magnification"]) == 5.0, p["magnification"]
    assert p["aperture"] == "f/16", p["aperture"]
    assert int(p["paper_grade"]) == 4, p["paper_grade"]
    assert float(p["base_exposure"]) == 10.0
    # 工作台页面也应读回这些值
    html = c.get(f"/projects/{pid}").data.decode()
    assert 'value="f/16"' in html
    assert '<option value="4" selected>4 号</option>' in html
    assert re.search(r'id="f-mag"[^>]*value="5(\.0)?"', html), "放大倍率未读回"
    print("[1] 放大参数持久化与读回 OK")

    # ---- 编排：画笔 + 多边形两类区域，各挂一个加光步骤
    plan = {
        "base_exposure": 10,
        "regions": [
            {"id": "r1", "kind": "brush", "name": "角落画笔",
             "size": 120, "feather": 15,
             "strokes": [{"points": [{"x": .1, "y": .1}, {"x": .2, "y": .2}]}],
             "points": []},
            {"id": "r2", "kind": "polygon", "name": "天空",
             "size": 80, "feather": 0, "strokes": [],
             "points": [{"x": .1, "y": .1}, {"x": .9, "y": .1},
                        {"x": .9, "y": .4}, {"x": .1, "y": .4}]},
        ],
        "steps": [
            {"id": "s1", "type": "burn", "region_id": "r1",
             "start": 10, "stops": 1},   # 10s 基础 + 1 档 → 追加 10s
            {"id": "s2", "type": "burn", "region_id": "r2",
             "start": 20, "stops": 1},
        ],
    }
    assert c.put(f"/api/projects/{pid}", json={
        "magnification": 5, "aperture": "f/16", "base_exposure": 10,
        "paper_grade": 4, "plan": plan}).status_code == 200

    # ---- 缺陷 2：1 档 = 2×，追加时长必须等于基础曝光 10s（不是 90s）
    comp = c.get(f"/api/projects/{pid}/compute").get_json()
    intervals = {i[0]: i for i in comp["intervals"]}
    for sid in ("s1", "s2"):
        sid_, kind, a, b = intervals[sid]
        assert kind == "burn"
        assert abs(b - a - 10.0) < 1e-6, (sid, a, b)
    # 操作单分段行同样为 10.0s
    sheet = c.get(f"/projects/{pid}/print-sheet").data.decode()
    assert "10.0" in sheet
    # 90s 是旧 10^stops 换算的错误结果，不应出现在持续列
    assert ">90.0<" not in sheet and ">90<" not in sheet
    print("[2] 加光 1 档追加 10s（2× 换算）OK")

    # ---- 缺陷 3：画笔步骤行带 kind，显示直径；多边形走羽化分支
    # 画笔：工具列应为 "120px / 15px"，且不应落到"多边形 /"
    brush_cell = "120px / 15px"
    poly_cell = "多边形 / 0px 羽化"
    assert brush_cell in sheet, "画笔工具直径未正确渲染"
    assert poly_cell in sheet, "多边形工具类型未正确渲染"
    # 画笔行必须配直径单元格，多边形行必须配羽化单元格（按区域名定位 <tr>）
    rows_html = re.findall(r"<tr>\s*<td>\d+</td>.*?</tr>", sheet, re.S)
    cells = {}
    for tr in rows_html:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(tds) >= 8:
            cells[tds[2].strip()] = re.sub(r"<[^>]+>", "", tds[7]).strip()
    assert cells["角落画笔"] == "120px / 15px", cells
    assert cells["天空"] == "多边形 / 0px 羽化", cells
    print("[3] 打印步骤 kind/工具类型渲染 OK")

    # ---- 版本回滚与 JSON 导出流程不回归
    r = c.post(f"/api/projects/{pid}/versions",
               json={"label": "v1", "note": "回归"})
    vid = r.get_json()["id"]
    assert c.post(f"/api/versions/{vid}/restore").status_code == 200
    payload = c.get(f"/api/projects/{pid}/export.json").get_json()
    assert payload["project"]["aperture"] == "f/16"
    # 秒表节点中加光动作的持续时间也是 10s
    burn_nodes = [n for n in payload["stopwatch"] if "加光" in n["action"]]
    assert any("持续 10.0s" in n["action"] for n in burn_nodes), \
        [n["action"] for n in burn_nodes]
    print("[4] 版本 / JSON 导出 / 秒表节点未回归 OK")

    assert c.delete(f"/api/projects/{pid}").status_code == 200
    print("REGRESSION ALL OK")


if __name__ == "__main__":
    main()
