# -*- coding: utf-8 -*-
"""端到端冒烟测试：建项目→校准→编排→检测→版本→操作单。"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.app import app as flask_app  # noqa: E402
from app import db  # noqa: E402

# 使用临时数据库/图片目录，避免污染开发数据
TMP = Path("/tmp/darkroom_test")
TMP.mkdir(exist_ok=True)
db.DB_PATH = TMP / "test.db"
flask_app.config["TESTING"] = True
import app.app as ap  # noqa: E402
ap.UPLOAD_DIR = TMP / "images"
db.init_db()


def make_png():
    rng = np.linspace(0, 255, 320, dtype=np.uint8)[None, :].repeat(240, 0)
    img = Image.fromarray(rng, "L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf


def main():
    c = flask_app.test_client()
    # 首页
    assert c.get("/").status_code == 200

    # 建项目
    r = c.post("/api/projects", data={
        "name": "测试老巷", "base_exposure": "8",
        "image": (make_png(), "neg.png"),
    }, content_type="multipart/form-data")
    assert r.status_code == 200, r.data
    pid = r.get_json()["id"]
    print("project", pid)

    # 校准
    pts = [{"t": t, "gray": g} for t, g in
           [(2, 240), (4, 222), (8, 168), (16, 90), (32, 32), (64, 16)]]
    r = c.post("/api/calibrate", json={
        "points": pts, "paper_grade": 2, "base_exposure": 8})
    cal = r.get_json()
    assert cal["calibrated"] and cal["rms"] < 8, cal["rms"]
    print("calibrate rms", cal["rms"], "range", cal["range"])

    # 编排：一个多边形区域 + 遮挡 + 大幅加光（触发超范围/双工具/抵消）
    plan = {
        "base_exposure": 8,
        "regions": [
            {"id": "r1", "kind": "polygon", "name": "天空",
             "points": [{"x": .1, "y": .1}, {"x": .9, "y": .1},
                        {"x": .9, "y": .5}, {"x": .1, "y": .5}],
             "feather": 0, "size": 80, "strokes": []},
            {"id": "r2", "kind": "polygon", "name": "天空遮",
             "points": [{"x": .12, "y": .12}, {"x": .88, "y": .12},
                        {"x": .88, "y": .48}, {"x": .12, "y": .48}],
             "feather": 20, "size": 80, "strokes": []},
        ],
        "steps": [
            {"id": "s1", "type": "dodge", "region_id": "r1",
             "start": 0, "ratio": 0.8},
            {"id": "s2", "type": "burn", "region_id": "r2",
             "start": 2, "stops": 1},
            {"id": "s3", "type": "burn", "region_id": "r1",
             "start": 100, "stops": 3},
        ],
    }
    r = c.put(f"/api/projects/{pid}", json={
        "magnification": 4, "aperture": "f/11", "base_exposure": 8,
        "paper_grade": 2, "calibration": cal, "plan": plan})
    assert r.status_code == 200

    r = c.get(f"/api/projects/{pid}/compute")
    comp = r.get_json()
    kinds = [w["kind"] for w in comp["warnings"]]
    print("warnings:", kinds)
    assert "two_tools" in kinds
    assert "cancel" in kinds
    assert any(k.startswith("out_of_range") for k in kinds)

    # 版本
    r = c.post(f"/api/projects/{pid}/versions",
               json={"label": "v1", "note": "测试"})
    vid = r.get_json()["id"]
    assert len(c.get(f"/api/projects/{pid}/versions").get_json()) >= 1
    r = c.post(f"/api/versions/{vid}/restore")
    assert r.status_code == 200

    # 操作单
    r = c.get(f"/projects/{pid}/print-sheet")
    assert r.status_code == 200
    html = r.data.decode()
    assert "放大打印操作单" in html and "测试老巷" in html
    assert "data:image/png;base64" in html
    assert "遮挡" in html and "加光" in html
    print("print-sheet bytes:", len(html))

    # JSON 导出
    r = c.get(f"/api/projects/{pid}/export.json")
    payload = r.get_json()
    assert payload["stopwatch"][0]["t"] == 0
    assert any("遮挡" in n["action"] for n in payload["stopwatch"])
    print("stopwatch nodes:", [(n["t"], n["action"][:12]) for n in payload["stopwatch"]])

    # 删除
    assert c.delete(f"/api/projects/{pid}").status_code == 200
    print("ALL OK")


if __name__ == "__main__":
    main()
