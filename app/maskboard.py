# -*- coding: utf-8 -*-
"""实体遮挡板模块核心：投影校准、目标轮廓提取（毫米）、遮挡板反算、
问题检测、实际尺寸拼版、SVG 导出与版本偏差比较。

物理模型（点光源投影，放大机镜头）：
    已知直径 D 的圆片置于距相纸 h 处，投影直径
        pd = D * s(h),  s(h) = 1 + beta * h
    板边在相纸上的照度过渡带宽度（高斯等效全宽）
        band = w0 + gamma * h
板必须在镜头与相纸之间，s(h)>=1，故 beta>=0；h 越大投影越大、边缘越虚。
反算：先按 s(h) 把目标（相纸上的期望投影）缩小，再按板边过渡带对
相纸边缘的贡献 w0/s(h) 内缩（羽化补偿）。

单位：除栅格化外全部为毫米；归一化区域坐标通过底片扫描尺寸 * 放大倍率
换算到相纸（长边沿项目方向，等比映射，与前端 engine 的 0..1 约定一致）。
"""
import math

import numpy as np

TOOL_VERSION = "maskboard-1.0"

MIN_HEIGHT = 0.5          # mm，低于此视为压在相纸上
ERROR_TOL_MM = 3.0        # 比例校验框默认边长
SELF_INTERSECT_EPS = 1e-6


# ---------------------------------------------------------------- 校准

def fit_calibration(points, disc_diameter):
    """points: [{h, pd, band}]，至少两个不同高度。
    返回拟合结果（含每条点的残差与有效高度范围）。
    """
    D = float(disc_diameter)
    rows = []
    for p in points:
        h = float(p.get("h", 0))
        pd = p.get("pd")
        band = p.get("band")
        if h < MIN_HEIGHT or pd is None:
            continue
        rows.append((h, float(pd),
                     float(band) if band is not None else None))
    out = {"disc_diameter": D, "n": len(rows), "calibrated": False,
           "L": None, "beta": None, "w0": 0.0, "gamma": 0.0,
           "scale_rms": None, "band_rms": None,
           "h_min": None, "h_max": None,
           "residuals": [], "range_extrapolated": False}
    if len(rows) < 2:
        return out
    hs = np.array([r[0] for r in rows])
    if np.unique(hs).size < 2 or D <= 0:
        return out

    # s(h) = pd / D = 1 + beta*h  —— 一元最小二乘，带物理约束 beta>=0
    s = np.array([r[1] for r in rows]) / D
    A = np.vstack([np.ones_like(hs), hs]).T
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    beta = float(max(0.0, (coef[1] if len(coef) > 1 else 0.0)))
    s_pred = 1.0 + beta * hs
    scale_rms = float(np.sqrt(np.mean((s_pred - s) ** 2)))
    proj_rms = scale_rms * D   # 换算为投影直径毫米残差

    # 过渡带：band = w0 + gamma*h（仅当有两组带宽带数据）
    band_rows = [(h, b) for h, _, b in rows if b is not None]
    w0 = gamma = 0.0
    band_rms = None
    if len(band_rows) >= 2 and len({h for h, _ in band_rows}) >= 2:
        bh = np.array([b[0] for b in band_rows])
        bv = np.array([b[1] for b in band_rows])
        Ab = np.vstack([np.ones_like(bh), bh]).T
        cb, *_ = np.linalg.lstsq(Ab, bv, rcond=None)
        w0 = float(max(0.0, cb[0]))
        gamma = float(max(0.0, cb[1] if len(cb) > 1 else 0.0))
        bp = w0 + gamma * bh
        band_rms = float(np.sqrt(np.mean((bp - bv) ** 2)))

    # 以镜头距离 L 表达：beta = 1/(L-h) 的一阶形式不直接线性，
    # 这里存等效“投影焦距” L_eq = 1/beta（h<<L 时即镜头到相纸距离），
    # 仅用于界面显示物理量。
    L = float(1.0 / beta) if beta > 1e-9 else None

    residuals = []
    for (h, pd, band), sp in zip(rows, s_pred):
        residuals.append({"h": h, "pd": pd, "pd_pred": float(sp * D),
                          "scale": float(pd / D),
                          "band": band,
                          "band_pred": (w0 + gamma * h) if band is not None
                                       else None})
    out.update({
        "calibrated": True, "L": L, "beta": beta,
        "w0": w0, "gamma": gamma,
        "scale_rms": round(scale_rms, 6),
        "proj_rms": round(proj_rms, 4),
        "band_rms": round(band_rms, 4) if band_rms is not None else None,
        "h_min": float(hs.min()), "h_max": float(hs.max()),
        "residuals": residuals,
    })
    return out


def predict(fit, h):
    """返回 (scale, feather_band_mm, scale_err, band_err)。"""
    if not fit or not fit.get("calibrated"):
        return None
    h = float(h)
    beta = fit["beta"]
    scale = 1.0 + beta * h
    # 误差带：标定 RMS + 外推按到标定区间的距离线性放宽
    rms = fit.get("scale_rms") or 0.0
    margin = 0.0
    if h < fit["h_min"]:
        margin = 0.25 * rms * (fit["h_min"] - h) / max(1.0, fit["h_max"] - fit["h_min"])
    elif h > fit["h_max"]:
        margin = 0.25 * rms * (h - fit["h_max"]) / max(1.0, fit["h_max"] - fit["h_min"])
    scale_err = max(rms + margin, 1e-4)
    band = fit["w0"] + fit["gamma"] * h
    band_err = fit.get("band_rms") or 0.0
    return {"h": h, "scale": scale, "band": band,
            "scale_err": scale_err, "band_err": band_err,
            "in_range": fit["h_min"] - 1e-6 <= h <= fit["h_max"] + 1e-6}


# ---------------------------------------------------------------- 区域 → 毫米轮廓

def print_size_mm(project, settings=None):
    """像素→毫米的实体换算基准。

    优先使用项目遮挡板设置中显式录入的“打印影像尺寸”(mm)；
    未录入时回退到 扫描像素 × 放大倍率（旧行为）。
    返回 (print_w_mm, print_h_mm, px_per_mm_x, px_per_mm_y, source)。
    """
    img_w = max(1.0, float(project.get("image_w", 1)))
    img_h = max(1.0, float(project.get("image_h", 1)))
    pw = float((settings or {}).get("print_w") or 0)
    ph = float((settings or {}).get("print_h") or 0)
    source = "print_size"
    if pw <= 0 and ph <= 0:
        mag = max(0.01, float(project.get("magnification", 1)))
        pw, ph = img_w * mag, img_h * mag
        source = "magnification"
    elif pw <= 0:          # 只给高：按扫描宽高比补宽
        pw = ph * img_w / img_h
    elif ph <= 0:          # 只给宽：按比补高
        ph = pw * img_h / img_w
    return pw, ph, img_w / pw, img_h / ph, source


def paper_size_mm(project, settings=None):
    """打印影像在相纸上的实体尺寸 (mm)。"""
    pw, ph, _, _, _ = print_size_mm(project, settings)
    return pw, ph


def region_to_mm(region, project, settings=None):
    """归一化区域坐标 → 毫米坐标（板材坐标系，原点左下，y 向上）。"""
    pw, ph = paper_size_mm(project, settings)
    pts = []
    for p in region.get("points", []):
        pts.append({"x": float(p["x"]) * pw,
                    "y": (1.0 - float(p["y"])) * ph})
    return pts


def _raster_mm(region, project, settings=None, mm_per_cell=1.5):
    """把区域栅格化为相纸毫米网格上的覆盖率（复用 engine 的区域定义）。
    返回 (mask, (mm_x, mm_y))：每个栅格在 x/y 方向对应的毫米数。"""
    from . import engine
    pw, ph, pxmm_x, pxmm_y, _ = print_size_mm(project, settings)
    W = max(8, int(round(pw / mm_per_cell)))
    H = max(8, int(round(ph / mm_per_cell)))
    mm_x, mm_y = pw / W, ph / H
    # region 的 feather/size 以扫描像素给出；扫描 px → mm 用显式基准
    # （1/px_per_mm），再 → 栅格 px（/mm每格）。
    kx = 1.0 / pxmm_x / mm_per_cell
    ky = 1.0 / pxmm_y / mm_per_cell
    k = (kx + ky) / 2.0
    conv = region.copy()
    conv["feather"] = max(0.0, float(region.get("feather", 0))) * k
    if conv.get("kind") == "brush":
        conv["size"] = max(1.0, float(region.get("size", 80))) * k
        # engine.rasterize_region 的笔画点取 0..1 归一坐标（内部再乘 W/H）。
        # 其按段插值在长线段接缝附近可能留缝，这里按笔刷半径预密化
        # （段长与半径都换算到扫描像素）。
        img_diag = math.hypot(float(project.get("image_w", 1)),
                              float(project.get("image_h", 1)))
        r_px = max(1.0, float(region.get("size", 80))) * 0.5
        dense = []
        for st in region.get("strokes", []):
            pts = st.get("points", [])
            for a, b in zip(pts, pts[1:]):
                seg_px = math.hypot((b["x"] - a["x"]),
                                    (b["y"] - a["y"])) * img_diag
                n = max(1, int(seg_px / max(r_px * 0.5, 1.0)))
                for i in range(n):
                    t = i / n
                    dense.append({"x": a["x"] + (b["x"] - a["x"]) * t,
                                  "y": a["y"] + (b["y"] - a["y"]) * t})
            if pts:
                dense.append({"x": pts[-1]["x"], "y": pts[-1]["y"]})
        conv["strokes"] = [{"points": dense}]
        conv["points"] = []
    else:
        conv["points"] = [{"x": p["x"] * W, "y": p["y"] * H}
                          for p in region.get("points", [])]
    m = engine.rasterize_region(conv, W, H)
    return m, (mm_x, mm_y)


def extract_target_loops(region, project, settings=None, simplify_tol=1.2):
    """从区域定义提取相纸上的目标覆盖轮廓，返回
    {outer:[[x,y]...mm], holes:[[...]], area_mm2, multiple:bool,
     print_w_mm, print_h_mm, scale_source}。
    多边形直接取顶点；画笔先栅格化再边界跟踪成环并化简。
    """
    pw, ph, _, _, scale_source = print_size_mm(project, settings)
    base = {"print_w_mm": pw, "print_h_mm": ph, "scale_source": scale_source}
    if region.get("kind") != "brush":
        poly = region_to_mm(region, project, settings)
        if len(poly) < 3:
            return {"outer": [], "holes": [], "area": 0.0,
                    "multiple": False, **base}
        poly = _simplify([(p["x"], p["y"]) for p in poly], simplify_tol)
        return {"outer": _as_list(poly), "holes": [],
                "area": abs(_signed_area(poly)), "multiple": False, **base}

    m, (mm_x, mm_y) = _raster_mm(region, project, settings)
    bit = (m > 0.5)
    if not bit.any():
        return {"outer": [], "holes": [], "area": 0.0,
                "multiple": False, **base}
    loops_px = _trace_boundaries(bit)
    rings = []
    for loop in loops_px:
        # 栅格 y 向下、板材坐标 y 向上 → 翻转
        pts = [(x * mm_x, ph - y * mm_y) for x, y in loop]
        pts = _simplify(pts, simplify_tol)
        if len(pts) >= 3 and abs(_signed_area(pts)) > 4 * simplify_tol ** 2:
            rings.append(pts)
    # 面积最大的环为外环；其余被外环包含的为孔洞
    rings.sort(key=lambda r: abs(_signed_area(r)), reverse=True)
    if not rings:
        return {"outer": [], "holes": [], "area": 0.0,
                "multiple": False, **base}
    outer = rings[0]
    holes, extra = [], []
    for r in rings[1:]:
        if _point_in_polygon(r[0], outer):
            holes.append(r)
        else:
            extra.append(r)
    return {"outer": _as_list(outer), "holes": [_as_list(h) for h in holes],
            "area": abs(_signed_area(outer)),
            "multiple": bool(extra),
            "extra_outers": [_as_list(e) for e in extra], **base}


def _trace_boundaries(bit):
    """对二值掩膜做边界跟踪，返回全部外环/孔洞环。

    在“格边网格”上把每个填充像素视为单位方块。边界为有向格边：
    填充方块恒在边的右手侧（外环顺时针、孔洞逆时针）。在角点处
    按“先右转、再直行、再左转、最后回头”选择下一条边，即可沿
    闭合边界行走且不串环。
    """
    H, W = bit.shape

    def filled(x, y):
        return 0 <= x < W and 0 <= y < H and bool(bit[y, x])

    used = set()
    loops = []

    # 方向：0=+x  1=+y  2=-x  3=-y；边 (kind,x,y,d) 从角点 (x,y) 出发
    # d=0: ('H',x,y)；d=1: ('V',x,y)；d=2: ('H',x-1,y)；d=3: ('V',x,y-1)
    def edge_key(x, y, d):
        if d == 0:
            return ("H", x, y)
        if d == 1:
            return ("V", x, y)
        if d == 2:
            return ("H", x - 1, y)
        return ("V", x, y - 1)

    def is_boundary(x, y, d):
        # 起点角点 (x,y)，右手侧方块：填且另一侧空即边界。
        # d=0: 右=下；d=1(+y): 右=-x；d=2(-x): 右=上；d=3(-y): 右=+x
        if d == 0:
            return filled(x, y) and not filled(x, y - 1)
        if d == 1:
            return filled(x - 1, y) and not filled(x, y)
        if d == 2:
            return filled(x - 1, y - 1) and not filled(x - 1, y)
        return filled(x, y - 1) and not filled(x - 1, y - 1)

    def walk(sx, sy, sd):
        x, y, d = sx, sy, sd
        pts = []
        while True:
            key = edge_key(x, y, d)
            if key in used:
                break
            used.add(key)
            pts.append((float(x), float(y)))
            # 到终点角点
            if d == 0:
                x += 1
            elif d == 1:
                y += 1
            elif d == 2:
                x -= 1
            else:
                y -= 1
            nxt = None
            # 填充在右手侧：到角点后优先右转贴边，其次直行、左转、回头
            for nd in ((d - 1) % 4, d, (d + 1) % 4, (d + 2) % 4):
                if is_boundary(x, y, nd):
                    nxt = nd
                    break
            if nxt is None or (x, y, nxt) == (sx, sy, sd):
                break
            d = nxt
        return pts

    # 找起点：逐像素检查未用的朝外格边
    for y in range(H):
        for x in range(W):
            if not filled(x, y):
                # 孔洞起点：空像素 (x,y) 右侧/下侧贴着填充，且该填充边未走过。
                # d=1（向上）右手侧为右方像素 (x+1,y)：右填左空 → 孔洞内边界。
                if filled(x + 1, y) and edge_key(x, y, 1) not in used \
                        and is_boundary(x, y, 1):
                    lp = walk(x, y, 1)
                    if len(lp) >= 4:
                        loops.append(lp)
                continue
            # 上边 d=0：方块左上 (x,y) 向右，右手侧（下方）填充、上方空
            if edge_key(x, y, 0) not in used and is_boundary(x, y, 0):
                lp = walk(x, y, 0)
                if len(lp) >= 4:
                    loops.append(lp)
    return loops


# ---------------------------------------------------------------- 几何

def _as_list(loop):
    return [{"x": float(x), "y": float(y)} for x, y in loop]


def _signed_area(loop):
    s = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def _simplify(pts, tol):
    """Ramer–Douglas–Peucker。"""
    if len(pts) < 3:
        return pts

    def perp(p, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        den = math.hypot(dx, dy)
        if den < 1e-12:
            return math.hypot(p[0] - a[0], p[1] - a[1])
        return abs(dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / den

    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        dmax, imax = 0.0, -1
        for i in range(i0 + 1, i1):
            d = perp(pts[i], pts[i0], pts[i1])
            if d > dmax:
                dmax, imax = d, i
        if dmax > tol and imax > 0:
            keep[imax] = True
            stack.append((i0, imax))
            stack.append((imax, i1))
    return [pts[i] for i in range(len(pts)) if keep[i]]


def _centroid(loop):
    a2 = 0.0
    cx = cy = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = _xy(loop[i])
        x1, y1 = _xy(loop[(i + 1) % n])
        cr = x0 * y1 - x1 * y0
        a2 += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    if abs(a2) < 1e-9:
        xs = [_xy(p)[0] for p in loop]
        ys = [_xy(p)[1] for p in loop]
        return sum(xs) / n, sum(ys) / n
    return cx / (3 * a2), cy / (3 * a2)


def _xy(p):
    return (p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1])


def centroid(loop):
    """公开接口：返回多边形质心 (cx, cy)。"""
    return _centroid(loop)


def scale_loop(loop, s, cx=None, cy=None):
    if cx is None:
        cx, cy = _centroid(loop)
    out = []
    for p in loop:
        x, y = _xy(p)
        out.append({"x": cx + (x - cx) * s, "y": cy + (y - cy) * s})
    return out


def bbox_of(loops):
    xs, ys = [], []
    for lp in loops:
        for p in lp:
            x, y = _xy(p)
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs),
            "h": max(ys) - min(ys)}


def _point_in_polygon(p, poly):
    x, y = _xy(p)
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = _xy(poly[i])
        x1, y1 = _xy(poly[(i + 1) % n])
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xi > x:
                inside = not inside
    return inside


def _seg_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    a, b, c, d = map(_xy, (p1, p2, p3, p4))
    if max(min(a[0], b[0]), min(c[0], d[0])) > min(max(a[0], b[0]), max(c[0], d[0])):
        return False
    if max(min(a[1], b[1]), min(c[1], d[1])) > min(max(a[1], b[1]), max(c[1], d[1])):
        return False
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def _seg_bridge(p1, p2, p3, p4):
    """两线段的“颈宽”信息：相交返回 None；否则返回
    (最短距离, 中点 x, 中点 y, 颈评分)。
    颈评分综合方向平行度与沿向重叠：两近似平行且互相正对、最近点
    不只是端点接触时接近 1；尖角/端点贴近趋于 0。"""
    if _seg_intersect(p1, p2, p3, p4):
        return None
    a, b, c, d = map(_xy, (p1, p2, p3, p4))

    def foot(p, a0, b0):
        dx, dy = b0[0] - a0[0], b0[1] - a0[1]
        den = dx * dx + dy * dy
        if den < 1e-12:
            return 0.0, a0
        t = max(0.0, min(1.0, ((p[0] - a0[0]) * dx + (p[1] - a0[1]) * dy) / den))
        return t, (a0[0] + t * dx, a0[1] + t * dy)

    cands = []
    for p, e0, e1 in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
        t, q = foot(p, e0, e1)
        cands.append((math.hypot(p[0] - q[0], p[1] - q[1]), t, p, q))
    dist, t, p, q = min(cands, key=lambda z: z[0])
    if dist < 1e-9:
        return None
    ux, uy = (b[0] - a[0], b[1] - a[1])
    vx, vy = (d[0] - c[0], d[1] - c[1])
    lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
    par = abs(ux * vx + uy * vy) / max(lu * lv, 1e-12)
    # 沿公共方向的投影重叠率（以 ab 为基准投影 cd 两端）
    overlap = 0.0
    if lu > 1e-9:
        u0 = (ux * (c[0] - a[0]) + uy * (c[1] - a[1])) / (lu * lu)
        u1 = (ux * (d[0] - a[0]) + uy * (d[1] - a[1])) / (lu * lu)
        lo, hi = min(u0, u1), max(u0, u1)
        cover = max(0.0, min(1.0, hi) - max(0.0, lo))
        overlap = cover
    score = par * max(overlap, 0.2 * min(t, 1 - t) / 0.2)
    return dist, (p[0] + q[0]) / 2, (p[1] + q[1]) / 2, score


def loop_self_intersections(loop):
    """返回自交边对索引列表（相邻边共享顶点，跳过）。"""
    hits = []
    n = len(loop)
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (i == 0 and j == n - 1) or j == i + 1:
                continue
            c, d = loop[j], loop[(j + 1) % n]
            if _seg_intersect(a, b, c, d):
                hits.append((i, j))
    return hits


# ---------------------------------------------------------------- 遮挡板反算

def back_calc(target, pred, feather_comp=0.35):
    """由目标（相纸上期望投影）与校准预测反算板材切割轮廓。
    target: {outer, holes}（毫米）。pred: predict() 结果。
    feather_comp: 板边过渡带占相纸侧等效内缩比例（0.35 ≈ 半高宽的 0.7/2）。
    返回 {board, projected, err_band, bbox...}
    """
    s = pred["scale"]
    band = pred["band"]
    cx, cy = _centroid(target["outer"])

    # 板轮廓：先按投影缩放反算，再做羽化内缩（沿质心径向外扩/内缩）。
    # 投影时板边的过渡带会让“有效挡光边界”比几何投影边再向内吃掉
    # 约 feather_comp * band / s（板平面上），故板要做大一点。
    grow = feather_comp * band / s
    inv = 1.0 / s
    board_outer = scale_loop(target["outer"], inv, cx, cy)
    board_outer = _offset_radial(board_outer, cx, cy, grow)
    board_holes = []
    for h in target.get("holes", []):
        hx, hy = _centroid(h)
        bh = scale_loop(h, inv, hx, hy)
        # 孔洞是“不挡光”的内岛：同样做大（径向往孔洞外）补偿羽化
        bh = _offset_radial(bh, hx, hy, -grow)
        board_holes.append(bh)

    projected = {
        "outer": scale_loop(board_outer, s, cx, cy),
        "holes": [scale_loop(h, s, *_centroid(h)) for h in board_holes],
    }
    # 误差带（毫米，投影平面）：缩放残差 + 过渡带不确定度
    ds = pred["scale_err"]
    e = abs(ds / s) if s else 0.0
    band_e = abs(pred.get("band_err", 0.0)) * 0.5
    err = [max(ERROR_TOL_MM, 2.0 * e * max(abs(p["x"] - cx), abs(p["y"] - cy)))
           + band_e for p in projected["outer"]]
    bbox = bbox_of([board_outer] + board_holes)
    return {
        "board": {"outer": board_outer, "holes": board_holes},
        "projected": projected,
        "error_mm": float(min(err) if err else ERROR_TOL_MM),
        "error_max_mm": float(max(err) if err else ERROR_TOL_MM),
        "feather_grow_mm": grow,
        "scale": s,
        "bbox": bbox,
        "center": {"x": cx, "y": cy},
    }


def _offset_radial(loop, cx, cy, d):
    """沿质心方向把每个顶点外移 d（d<0 内移）。简单稳健，非严格等距偏置。"""
    out = []
    for p in loop:
        x, y = _xy(p)
        r = math.hypot(x - cx, y - cy)
        if r < 1e-9:
            out.append({"x": x, "y": y})
        else:
            k = (r + d) / r
            out.append({"x": cx + (x - cx) * k, "y": cy + (y - cy) * k})
    return out


# ---------------------------------------------------------------- 手柄

HANDLE_W = 18.0   # mm 默认手柄尺寸
HANDLE_L = 60.0


def handle_polygon(anchor, direction, length=HANDLE_L, width=HANDLE_W):
    """anchor: 板外环上的接入点 {x,y}；direction: 角度（度，0=右，逆时针）。
    返回矩形 4 顶点（毫米）。"""
    a = math.radians(direction)
    ux, uy = math.cos(a), math.sin(a)
    px, py = -uy, ux
    hw = width / 2.0
    pts = []
    for sx, sy in ((0, -1), (0, 1), (1, 1), (1, -1)):
        pts.append({"x": anchor["x"] + ux * length * sx + px * hw * sy,
                    "y": anchor["y"] + uy * length * sx + py * hw * sy})
    return pts


def handle_anchor_suggest(outer, direction):
    """在板外环上寻找朝 direction 方向最突出的边中点作为接入点。"""
    a = math.radians(direction)
    ux, uy = math.cos(a), math.sin(a)
    cx, cy = _centroid(outer)
    best, best_d = None, None
    n = len(outer)
    for i in range(n):
        p0, p1 = outer[i], outer[(i + 1) % n]
        mx, my = (p0["x"] + p1["x"]) / 2, (p0["y"] + p1["y"]) / 2
        d = (mx - cx) * ux + (my - cy) * uy
        if best_d is None or d > best_d:
            best_d, best = d, {"x": mx, "y": my}
    return best or outer[0]


# ---------------------------------------------------------------- 问题检测

def validate(spec, calc_result, target, project, paper_w, paper_h,
             fit=None, settings=None):
    """汇总遮挡板的全部问题，返回 issue 列表。
    issue: {kind, severity:'error'|'warn', message, where?}
    """
    issues = []
    h = float(spec.get("height", 0))

    # 1) 超校准范围
    if not fit or not fit.get("calibrated"):
        issues.append({"kind": "uncalibrated", "severity": "error",
                       "message": "尚未完成投影校准（至少两个高度）"})
    else:
        if h < fit["h_min"] or h > fit["h_max"]:
            issues.append({
                "kind": "height_out_of_range", "severity": "error",
                "message": f"高度 {h:g}mm 超出校准范围 "
                           f"{fit['h_min']:g}–{fit['h_max']:g}mm，"
                           "缩放/羽化均为外推值",
                "h": h, "h_min": fit["h_min"], "h_max": fit["h_max"]})
        if fit.get("proj_rms", 0) and fit["proj_rms"] > 1.5:
            issues.append({
                "kind": "calibration_rms", "severity": "warn",
                "message": f"校准投影残差 {fit['proj_rms']:.2f}mm 偏大，"
                           "建议复核圆片测量"})

    board = calc_result["board"]
    outer, holes = board["outer"], board["holes"]

    # 2) 轮廓自交（外环与孔洞）
    for name, loop in [("外轮廓", outer)] + [("孔洞", h_) for h_ in holes]:
        hits = loop_self_intersections(loop)
        if hits:
            issues.append({
                "kind": "self_intersect", "severity": "error",
                "message": f"{name}自交（{len(hits)} 处），"
                           "切割路径不可用，请拖开控制点",
                "edges": hits[:8], "which": name})

    # 3) 窄桥：外轮廓自距离、孔洞之间、孔洞到外边
    min_bridge = float(spec.get("min_bridge", 5))
    narrow = _narrow_bridges(outer, holes, min_bridge)
    for msg, pts in narrow:
        issues.append({"kind": "narrow_bridge", "severity": "error",
                       "message": msg, "at": pts})

    # 4) 手柄压住目标区
    handle = spec.get("handle")
    if handle:
        hp = handle_polygon({"x": handle["ax"], "y": handle["ay"]},
                            handle["dir"], handle.get("length", HANDLE_L),
                            handle.get("width", HANDLE_W))
        proj = calc_result["projected"]
        # 以目标区域（相纸上应被挡的范围）为准检查手柄侵入；先把手柄按同一
        # 投影中心缩放到相纸坐标。板边羽化外扩不参与该判定。
        c = calc_result.get("center") or _centroid(proj["outer"])
        s = calc_result["scale"]
        hp_handle = dict(handle)
        hp_handle["ax"] = c["x"] + (handle["ax"] - c["x"]) * s
        hp_handle["ay"] = c["y"] + (handle["ay"] - c["y"]) * s
        hp_handle["length"] = handle.get("length", HANDLE_L) * s
        hp_handle["width"] = handle.get("width", HANDLE_W) * s
        hit_pts = [
            {"x": c["x"] + (p["x"] - c["x"]) * s,
             "y": c["y"] + (p["y"] - c["y"]) * s}
            for p in _handle_intrusion(hp_handle, target["outer"],
                                       target.get("holes", []))]
        if hit_pts:
            issues.append({
                "kind": "handle_over_target", "severity": "error",
                "message": "手柄投影压在目标区域内，会误挡光线，"
                           "请改朝向/接入点或缩短手柄",
                "at": hit_pts[:4]})

    # 5) 排版越界：板材 + 手柄必须落在板材尺寸内
    sw = float(spec.get("sheet_w", paper_w))
    sh = float(spec.get("sheet_h", paper_h))
    all_cut = [outer] + holes
    if handle:
        all_cut = all_cut + [handle_polygon(
            {"x": handle["ax"], "y": handle["ay"]}, handle["dir"],
            handle.get("length", HANDLE_L), handle.get("width", HANDLE_W))]
    bb = bbox_of(all_cut)
    if bb and (bb["x"] < 0 or bb["y"] < 0 or
               bb["x"] + bb["w"] > sw + 1e-6 or
               bb["y"] + bb["h"] > sh + 1e-6):
        issues.append({
            "kind": "sheet_overflow", "severity": "error",
            "message": f"轮廓+手柄包围盒 {bb['w']:.0f}×{bb['h']:.0f}mm "
                       f"超出板材 {sw:g}×{sh:g}mm，或原点不在板内",
            "bbox": bb})

    # 6) 目标落在相纸外（关联区域本身异常）
    pw, ph = paper_size_mm(project, settings)
    tb = bbox_of([target["outer"]])
    if tb and (tb["x"] < 0 or tb["y"] < 0 or
               tb["x"] + tb["w"] > pw + 1e-6 or tb["y"] + tb["h"] > ph + 1e-6):
        issues.append({"kind": "target_off_paper", "severity": "warn",
                       "message": "目标区域超出相纸范围"})
    return issues


def _narrow_bridges(outer, holes, min_bridge):
    out = []

    def pair_check(lp_a, lp_b, label):
        na, nb = len(lp_a), len(lp_b)
        for i in range(na):
            a, b = lp_a[i], lp_a[(i + 1) % na]
            jstart = i + 2 if lp_a is lp_b else 0
            jend = nb - 1 if lp_a is lp_b else nb
            for j in range(jstart, jend):
                if lp_a is lp_b and (i == 0 and j == nb - 1):
                    continue
                c, d = lp_b[j], lp_b[(j + 1) % nb]
                info = _seg_bridge(a, b, c, d)
                if info and info[0] < min_bridge and info[3] >= 0.6:
                    dd, px, py, _ = info
                    # 近似平行且最近点落在段内部才是窄颈，
                    # 排除尖角/端点贴近造成的误报
                    out.append((f"{label}最小料宽 {dd:.1f}mm < "
                                f"{min_bridge:g}mm", [{"x": px, "y": py}]))
                    if len(out) >= 6:
                        return

    pair_check(outer, outer, "外轮廓")
    for k, h_ in enumerate(holes):
        pair_check(h_, h_, f"孔洞{k + 1}")
        pair_check(outer, h_, f"孔洞{k + 1}到外边")
    for k in range(len(holes)):
        for m in range(k + 1, len(holes)):
            pair_check(holes[k], holes[m], f"孔洞{k + 1}/{m + 1}之间")
    return out


def _handle_intrusion(handle, outer, holes, n_samples=11):
    """沿手柄中线从接入点向外采样；忽略靠近接入点的接缝（10% 长度），
    若其后过半采样点落在投影目标内（孔洞外），则判定手柄压住目标。"""
    a = math.radians(handle["dir"])
    ux, uy = math.cos(a), math.sin(a)
    length = handle.get("length", HANDLE_L)
    inside, hits = 0, []
    for k in range(n_samples):
        t = (k + 0.5) / n_samples
        if t < 0.12:
            continue
        p = {"x": handle["ax"] + ux * length * t,
             "y": handle["ay"] + uy * length * t}
        if _point_in_polygon(p, outer) and not any(
                _point_in_polygon(p, h) for h in holes):
            inside += 1
            hits.append(p)
    return hits if inside >= 5 else []


# ---------------------------------------------------------------- 拼版

def nest_tools(tools, sheet_w, sheet_h, gap=8.0, margin=10.0):
    """简单下台阶式（next-fit decreasing height）实际尺寸拼版。
    tools: [{id,name,bbox:{x,y,w,h}, rotate_ok}]（bbox 为板坐标系）。
    返回 {placements:[{tool_id,x,y,rot,cx,cy}], sheets:[{w,h}], fits:bool}
    """
    items = sorted(tools, key=lambda t: -(t["bbox"]["h"] * t["bbox"]["w"]))
    placements = {}
    cur_x = cur_y = margin
    row_h = 0.0
    fits = True
    for t in items:
        bb = t["bbox"]
        w, h = bb["w"], bb["h"]
        rot = False
        if t.get("rotate_ok", True) and w > h and w > sheet_w - 2 * margin:
            w, h = h, w
            rot = True
        if cur_x + w > sheet_w - margin and cur_x > margin:
            cur_x = margin
            cur_y += row_h + gap
            row_h = 0
        if cur_x + w > sheet_w - margin or cur_y + h > sheet_h - margin:
            fits = False
        # 仍给一个位置（越界时 SVG 会标红），保证可导出
        placements[t["id"]] = {
            "tool_id": t["id"], "x": cur_x, "y": cur_y,
            "rot": 90 if rot else 0,
            "ox": bb["x"], "oy": bb["y"], "w": w, "h": h}
        cur_x += w + gap
        row_h = max(row_h, h)
    return {"placements": placements,
            "sheets": [{"w": sheet_w, "h": sheet_h}], "fits": fits}


# ---------------------------------------------------------------- SVG

SVG_HEADER = """<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm"
 viewBox="0 0 {w} {h}" font-family="Helvetica, 'PingFang SC', sans-serif">
<rect x="0" y="0" width="{w}" height="{h}" fill="#fff"/>
"""


def svg_path(loop, ox=0.0, oy=0.0):
    d = []
    for i, p in enumerate(loop):
        x, y = _xy(p)
        d.append(("M" if i == 0 else "L") + f"{x - ox:.2f},{y - oy:.2f}")
    d.append("Z")
    return " ".join(d)


def export_svg(tool_docs, nest, sheet_w, sheet_h, project_name="",
               tool_version=TOOL_VERSION, baseline=None):
    """生成按实际尺寸（1 单位 = 1mm）的切割 SVG。"""
    pl = nest["placements"]
    out = [SVG_HEADER.format(w=round(sheet_w, 2), h=round(sheet_h, 2))]
    # 样式
    out.append("""<style>
 .cut{fill:none;stroke:#c00;stroke-width:.25;}
 .hole{fill:none;stroke:#c00;stroke-width:.25;stroke-dasharray:2 1;}
 .handle{fill:none;stroke:#1565c0;stroke-width:.25;}
 .proj{fill:none;stroke:#888;stroke-width:.15;stroke-dasharray:1.5 1.5;}
 .mark{stroke:#000;stroke-width:.2;}
 .label{font-size:3mm;fill:#000;}
 .sub{font-size:2mm;fill:#444;}
 .err{fill:none;stroke:#e80;stroke-width:.12;}
 .frame{fill:none;stroke:#000;stroke-width:.2;}
</style>
""")

    # 板材裁切线（四角裁切标记）
    cm = 5.0
    for cx0, cy0, dx, dy in ((0, 0, 1, 1), (sheet_w, 0, -1, 1),
                             (0, sheet_h, 1, -1), (sheet_w, sheet_h, -1, -1)):
        out.append(f'<line class="mark" x1="{cx0 + dx*cm:.2f}" y1="{cy0:.2f}"'
                   f' x2="{cx0 + dx*cm:.2f}" y2="{cy0 + dy*cm:.2f}"/>')
        out.append(f'<line class="mark" x1="{cx0:.2f}" y1="{cy0 + dy*cm:.2f}"'
                   f' x2="{cx0 + dx*cm:.2f}" y2="{cy0 + dy*cm:.2f}"/>')
    out.append(f'<rect class="frame" x="0" y="0" width="{sheet_w:g}" height="{sheet_h:g}"/>')

    for idx, doc in enumerate(tool_docs, 1):
        t = doc["tool"]
        spec = doc["spec"]
        res = doc["result"]
        p = pl.get(t["id"])
        if not p:
            continue
        gx, gy = p["x"] - p["ox"], p["y"] - p["oy"]
        rot = p["rot"]
        tr = (f'<g transform="translate({p["x"]:.2f},{p["y"]:.2f}) '
              f'rotate({rot}) translate({-p["ox"]:.2f},{-p["oy"]:.2f})">')
        out.append(tr)
        board = res["board"]
        # 预计投影（灰色虚线，供对位参考）与误差带
        proj = res["projected"]
        out.append(f'<path class="proj" d="{svg_path(proj["outer"])}"/>')
        for h_ in proj.get("holes", []):
            out.append(f'<path class="proj" d="{svg_path(h_)}"/>')
        # 误差带：沿投影外环径向偏移 ±e 的两条包络
        e = max(0.8, res.get("error_mm", ERROR_TOL_MM))
        cx, cy = _centroid(proj["outer"])
        out.append(f'<path class="err" d="{_band_path(proj["outer"], e, cx, cy, +1)}"/>')
        out.append(f'<path class="err" d="{_band_path(proj["outer"], e, cx, cy, -1)}"/>')
        # 切割线：外轮廓（红）、孔洞（红虚线）
        out.append(f'<path class="cut" d="{svg_path(board["outer"])}"/>')
        for h_ in board["holes"]:
            out.append(f'<path class="hole" d="{svg_path(h_)}"/>')
        # 手柄（蓝）与朝向箭头
        hspec = spec.get("handle")
        if hspec:
            hp = handle_polygon({"x": hspec["ax"], "y": hspec["ay"]},
                                hspec["dir"], hspec.get("length", HANDLE_L),
                                hspec.get("width", HANDLE_W))
            out.append(f'<path class="handle" d="{svg_path(hp)}"/>')
            tip = {"x": hspec["ax"] + math.cos(math.radians(hspec["dir"])) *
                            hspec.get("length", HANDLE_L),
                   "y": hspec["ay"] + math.sin(math.radians(hspec["dir"])) *
                            hspec.get("length", HANDLE_L)}
            out.append(f'<circle cx="{tip["x"]:.2f}" cy="{tip["y"]:.2f}" '
                       f'r="1.2" class="handle"/>')
        out.append('</g>')

        # 编号与信息（不随旋转，写在拼版坐标）
        lx, ly = p["x"], p["y"] - 2.0
        out.append(f'<text class="label" x="{lx:.2f}" y="{ly:.2f}">'
                   f'#{idx} {_esc(t["name"])}</text>')
        out.append(f'<text class="sub" x="{lx:.2f}" y="{ly + 2.6:.2f}">'
                   f'区域 {_esc(spec.get("region_name", "?"))} · '
                   f'h={spec.get("height"):g}mm · '
                   f's={res.get("scale", 1):.3f} · '
                   f'{"旋转90°" if rot else "不转"} · '
                   f'朝向 {(spec.get("handle") or {}).get("dir", 0):g}°</text>')
        if any(i["severity"] == "error" for i in doc.get("issues", [])):
            out.append(f'<text class="sub" x="{lx:.2f}" y="{ly + 5.2:.2f}" '
                       f'fill="#c00">含错误：导出仅供排查，勿直接切割</text>')

    # 比例校验框（50mm 标准方框 + 标注），置于板材左下角内
    bx, bl = 8.0, 50.0
    by = sheet_h - bx - bl     # 方框底沿距板材下边 8mm
    out.append(f'<rect class="frame" x="{bx}" y="{by:.2f}" width="{bl:.2f}" height="{bl:.2f}"/>')
    for k in range(1, 10):
        x = bx + bl * k / 10
        out.append(f'<line class="mark" x1="{x:.2f}" y1="{by + bl:.2f}" '
                   f'x2="{x:.2f}" y2="{by + bl - (2 if k % 5 else 4):.2f}"/>')
    out.append(f'<text class="sub" x="{bx}" y="{by - 3:.2f}">'
               f'比例校验框：此方框边长 50.00mm（输出比例 1:1，不缩放打印）</text>')

    if baseline:
        pxmm = (baseline["print_w_mm"] / max(1, baseline["image_w_px"]) +
                baseline["print_h_mm"] / max(1, baseline["image_h_px"])) / 2
        pw_label = (f'{baseline["print_w_mm"]:.1f}×{baseline["print_h_mm"]:.1f}mm '
                    f'@ {baseline["image_w_px"]}×{baseline["image_h_px"]}px')
        src_label = ('显式打印尺寸' if baseline.get("source") == "print_size"
                     else f'倍率 {baseline.get("magnification", 1):g}×')
        out.append(f'<text class="sub" x="{bx}" y="{by - 6.4:.2f}">'
                   f'打印影像基准 {pw_label} · 1px={pxmm:.4f}mm（{src_label}）</text>')

    out.append(f'<text class="sub" x="{sheet_w - 8:.2f}" y="{sheet_h - 8:.2f}" '
               f'text-anchor="end">{_esc(project_name)} · 实体遮挡板拼版 · '
               f'{tool_version}</text>')
    out.append("</svg>\n")
    return "".join(out)


def _band_path(loop, e, cx, cy, sign):
    """误差带包络：相对投影环沿质心径向偏移 ±e。"""
    pts = _offset_radial(loop, cx, cy, sign * e)
    return svg_path(pts)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------- 版本比较

def compare_versions(a, b):
    """比较两版工具的板轮廓与投影偏差（毫米）。
    a/b: {spec, result, calibration_snapshot?}。
    """
    ra, rb = a.get("result", {}), b.get("result", "")
    if isinstance(rb, str):
        import json
        rb = json.loads(b or "{}")
    if isinstance(ra, str):
        import json
        ra = json.loads(ra or "{}")

    def loops(r):
        return (r.get("board", {}).get("outer", []),
                r.get("projected", {}).get("outer", []))

    boa, pra = loops(ra)
    bob, prb = loops(rb)
    out = {
        "height_delta": float(b.get("spec", {}).get("height", 0)) -
                        float(a.get("spec", {}).get("height", 0)),
        "scale_delta": float(rb.get("scale", 1) - ra.get("scale", 1)),
    }
    if boa and bob:
        da = _loop_distance_stats(boa, bob)
        out["board_hausdorff_mm"] = da["hausdorff"]
        out["board_mean_mm"] = da["mean"]
    if pra and prb:
        dp = _loop_distance_stats(pra, prb)
        out["proj_hausdorff_mm"] = dp["hausdorff"]
        out["proj_mean_mm"] = dp["mean"]
    # 校准差异
    ca = a.get("calibration_snapshot") or {}
    cb = b.get("calibration_snapshot") or {}
    if isinstance(ca, str):
        import json
        ca = json.loads(ca or "{}")
    if isinstance(cb, str):
        import json
        cb = json.loads(cb or "{}")
    if ca.get("beta") is not None and cb.get("beta") is not None:
        out["beta_delta"] = cb["beta"] - ca["beta"]
    out["error_band_a_mm"] = ra.get("error_mm")
    out["error_band_b_mm"] = rb.get("error_mm")
    return out


def _loop_distance_stats(loop_a, loop_b):
    """有向 Hausdorff + 平均点-环距离（质心对齐后）。"""
    ca = _centroid(loop_a)
    cb = _centroid(loop_b)
    aa = [{"x": _xy(p)[0] - ca[0] + cb[0], "y": _xy(p)[1] - ca[1] + cb[1]}
          for p in loop_a]

    def directed(p, q):
        ds = []
        nq = len(q)
        for v in p:
            dm = min(_point_seg(v, q[i], q[(i + 1) % nq])
                     for i in range(nq))
            ds.append(dm)
        return max(ds), sum(ds) / len(ds)

    h1, m1 = directed(aa, loop_b)
    h2, m2 = directed(loop_b, aa)
    return {"hausdorff": round(max(h1, h2), 3),
            "mean": round((m1 + m2) / 2, 3)}


def _point_seg(p, a, b):
    p, a, b = _xy(p), _xy(a), _xy(b)
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    if den < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den))
    qx, qy = a[0] + t * dx, a[1] + t * dy
    return math.hypot(p[0] - qx, p[1] - qy)
