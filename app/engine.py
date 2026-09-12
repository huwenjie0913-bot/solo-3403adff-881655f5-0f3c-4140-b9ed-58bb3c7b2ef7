# -*- coding: utf-8 -*-
"""相纸响应曲线拟合、区域栅格化、累计曝光与问题检测。

响应模型（D=反射密度，G=0..255 灰度，D=log10(255/G)）：
    D(x) = Dmin + (Dmax-Dmin) / (1 + 10**(-k*(x-x0)))
其中 x = log10(E/E0)，E0 为基础曝光对应的曝光量，即 x0 取 0 时
基础曝光落在曲线中段。JS 端 engine.js 为同一份逻辑的镜像，
两端以同一份数据计算，结果应一致。
"""
import math

import numpy as np

X_LO, X_HI, LUT_N = -3.0, 7.0, 8192

# 反差号 -> 默认陡度 k（未校准时用）
GRADE_K = {0: 0.35, 1: 0.55, 2: 0.85, 3: 1.2, 4: 1.6, 5: 2.1}

EDGE_JUMP_LEVELS = 3.0   # 每像素参考分辨率(1000px)下的灰度跳变阈值
CANCEL_RATIO = 0.7       # 遮挡与加光互相抵消的覆盖比例阈值
EDGELEN = 10             # 边缘窄带宽度(px@1000)


# ---------------------------------------------------------------- curve

def logistic(x, Dmin, Dmax, x0, k):
    return Dmin + (Dmax - Dmin) / (1.0 + np.exp(-math.log(10) * k * (x - x0)))


def fit_curve(points, grade=2, base_exposure=10.0):
    """points: [{t: 秒, gray: 0..255 实测灰度}]。
    返回 {params, rms, range:[x_min,x_max], fitted:[{x,gray}], n}。
    采用带阻尼的 Gauss-Newton 最小二乘（残差为灰度）。
    """
    pts = [p for p in points if p.get("t", 0) > 0 and 0 <= p.get("gray", -1) <= 255]
    if len(pts) < 2:
        p = default_params(grade, base_exposure)
        return {"params": p, "rms": None, "range": [], "fitted": [], "n": len(pts)}

    t = np.array([p["t"] for p in pts], dtype=float)
    g = np.array([p["gray"] for p in pts], dtype=float)
    x = np.log10(np.maximum(t / max(base_exposure, 1e-6), 1e-9))

    k0 = GRADE_K.get(int(grade), 0.85)
    # 初值：Dmin 对应最长曝光（最暗、灰度最小），Dmax 对应最短曝光
    bright = np.sort(g)[-max(1, len(g) // 4):].mean()  # 亮（灰度大）
    dark = np.sort(g)[: max(1, len(g) // 4)].mean()    # 暗（灰度小）
    Dmin0 = math.log10(255.0 / max(bright, 4.0))
    Dmax0 = math.log10(255.0 / max(dark, 4.0))
    if Dmax0 < Dmin0 + 0.1:
        Dmax0 = Dmin0 + 1.0
    # x0：灰度中点 ((亮+暗)/2) 对应的 x（x 随 t 单调增）
    order = np.argsort(x)
    xs, gs = x[order], g[order]
    gmid = (bright + dark) / 2.0
    x00 = float(np.interp(gmid, gs, xs))
    if not math.isfinite(x00):
        x00 = 0.0

    theta = np.array([max(Dmin0, 0.0), min(max(Dmax0, 0.1), 3.2),
                      x00, max(k0, 0.05)], dtype=float)
    bounds = [(0.0, 1.2), (0.3, 4.0), (-2.5, 6.0), (0.05, 5.0)]
    lo = np.array([b[0] + 1e-4 for b in bounds])
    hi = np.array([b[1] - 1e-4 for b in bounds])
    L = math.log(10)

    def model(th, xx):
        return 255.0 * np.power(10.0, -logistic(xx, *th))

    def rms_of(th):
        return float(np.sqrt(np.mean((model(th, x) - g) ** 2)))

    rms_last = rms_of(theta)
    lam = 1e-5
    for _ in range(80):
        Dmin, Dmax, x0, k = theta
        z = L * k * (x - x0)
        sig = 1.0 / (1.0 + np.exp(-z))
        D = Dmin + (Dmax - Dmin) * sig
        ghat = 255.0 * np.power(10.0, -D)
        r = ghat - g
        J = np.empty((len(x), 4))
        J[:, 0] = -L * ghat * (1 - sig)
        J[:, 1] = -L * ghat * sig
        J[:, 2] = L * k * (Dmax - Dmin) * ghat * sig * (1 - sig)
        J[:, 3] = -L * (x - x0) * (Dmax - Dmin) * ghat * sig * (1 - sig)
        JtJ, Jtr = J.T @ J, J.T @ r
        # Levenberg–Marquardt 阻尼 + 回退线搜索
        best, best_rms = None, rms_last
        for lam_try in (lam, lam * 10, lam * 100, lam / 10 if lam > 1e-7 else lam):
            try:
                delta = np.linalg.solve(JtJ + lam_try * np.diag(np.diag(JtJ) + 1e-12), -Jtr)
            except np.linalg.LinAlgError:
                continue
            for step in (1.0, 0.5, 0.25):
                cand = np.clip(theta + delta * step, lo, hi)
                rms_c = rms_of(cand)
                if rms_c < best_rms - 1e-6:
                    best, best_rms = cand, rms_c
            if best is not None:
                lam = lam_try * 0.5
                break
        if best is None:
            lam *= 10
            if lam > 1e6:
                break
            continue
        if abs(rms_last - best_rms) < 1e-5:
            theta = best
            break
        theta, rms_last = best, best_rms

    params = {"Dmin": float(theta[0]), "Dmax": float(theta[1]),
              "x0": float(theta[2]), "k": float(theta[3])}
    rms = float(np.sqrt(np.mean((model(theta, x) - g) ** 2)))

    xx = np.linspace(X_LO, X_HI, 160)
    fitted = [{"x": float(a), "gray": float(max(0, min(255, model(theta, a))))}
              for a in xx]
    data_x = x[np.argsort(x)]
    return {
        "params": params,
        "rms": round(rms, 3),
        "range": [float(data_x[0]), float(data_x[-1])],
        "points": [{"x": float(a), "gray": float(b)} for a, b in zip(x, g)],
        "fitted": fitted,
        "n": len(pts),
    }


def default_params(grade=2, base_exposure=10.0):
    return {"Dmin": 0.06, "Dmax": 2.05, "x0": 0.0,
            "k": GRADE_K.get(int(grade), 0.85)}


def normalize_calibration(cal, grade=2, base_exposure=10.0):
    cal = cal or {}
    if cal.get("params"):
        return cal
    out = dict(cal)
    out["params"] = default_params(grade, base_exposure)
    out["calibrated"] = False
    return out


# ---------------------------------------------------------------- split-grade

# 分级滤镜（split-grade）：同一张相纸分两段曝光——先低反差滤镜（软，约 00 号）
# 后高反差滤镜（硬，约 5 号）。每段按各自拟合曲线产生"净密度贡献"
# （超出纸基 Dmin 的密度），总密度 = 纸基 + 两段净贡献之和。
SPLIT_GRADES = {"soft": 0, "hard": 5}      # 未校准时的默认反差号
SPLIT_T_MIN, SPLIT_T_MAX = 0.2, 600.0      # 目标求解的时长搜索界限(s)


def split_enabled(plan):
    sp = (plan or {}).get("split") or {}
    return bool(sp.get("enabled"))


def normalize_split(cal):
    """取出 calibration.split 的两条曲线并补默认参数。"""
    sc = (cal or {}).get("split") or {}
    return (normalize_calibration(sc.get("soft"), SPLIT_GRADES["soft"]),
            normalize_calibration(sc.get("hard"), SPLIT_GRADES["hard"]))


def step_filter(s):
    """步骤所属滤镜段。缺省按暗房习惯：遮挡护高光→软段，加光塑阴影→硬段。"""
    f = s.get("filter")
    if f in ("soft", "hard"):
        return f
    return "soft" if s.get("type") == "dodge" else "hard"


def split_times(plan):
    sp = (plan or {}).get("split") or {}
    t_soft = min(600.0, max(0.05, float(sp.get("t_soft", 10.0) or 10.0)))
    t_hard = min(600.0, max(0.05, float(sp.get("t_hard", 10.0) or 10.0)))
    return t_soft, t_hard


def split_layout(plan):
    """分级滤镜时间轴布局。
    软段从 0 开始，基础曝光 t_soft，段尾延伸至最迟的软步骤
    （加光追加会延长该滤镜的连续曝光时段）；换片节点 = 软段结束；
    硬段从换片点开始，同样为基础 t_hard 加步骤延伸。
    返回 (segments, intervals, swap_t)：
      segments  [(filter, seg0, seg1)]，intervals  [(step, filter, a, b)]。
    遮挡被钳在所属段的基础窗口内，加光钳到段首之后，
    因此每个步骤始终落在所属滤镜的连续曝光段内。"""
    t_soft, t_hard = split_times(plan)

    def mk(s, f, seg0, t_base):
        if s.get("type") == "dodge":
            dur = t_base * min(1.0, max(0.0, float(s.get("ratio", 0))))
            a = min(max(float(s.get("start", seg0)), seg0),
                    seg0 + t_base - dur)
            return (s, f, a, a + dur)
        dur = t_base * (2.0 ** float(s.get("stops", 0)) - 1.0)
        a = max(float(s.get("start", seg0 + t_base)), seg0)
        return (s, f, a, a + dur)

    intervals = []
    soft_end = t_soft
    for s in _steps(plan):
        if step_filter(s) != "soft":
            continue
        iv = mk(s, "soft", 0.0, t_soft)
        intervals.append(iv)
        soft_end = max(soft_end, iv[3])
    hard_end = soft_end + t_hard
    for s in _steps(plan):
        if step_filter(s) != "hard":
            continue
        iv = mk(s, "hard", soft_end, t_hard)
        intervals.append(iv)
        hard_end = max(hard_end, iv[3])
    segments = [("soft", 0.0, soft_end), ("hard", soft_end, hard_end)]
    return segments, intervals, soft_end


def split_segments(plan):
    """两个连续曝光段：(滤镜, 段首, 段尾)。换片节点 = 软段结束。"""
    return split_layout(plan)[0]


def split_intervals(plan):
    """展开 split 模式步骤区间为 (step, filter, a, b)，见 split_layout。"""
    return split_layout(plan)[1]


# ---------------------------------------------------------------- LUT

def build_lut(params):
    xs = np.linspace(X_LO, X_HI, LUT_N)
    D = logistic(xs, params["Dmin"], params["Dmax"], params["x0"], params["k"])
    gray = 255.0 * np.power(10.0, -D)
    return np.clip(gray, 0, 255).astype(np.float32)


def gray_for_logE(logE, lut):
    idx = np.clip((logE - X_LO) / (X_HI - X_LO) * (LUT_N - 1), 0, LUT_N - 1)
    return lut[idx.astype(np.int32) if hasattr(idx, "astype") else int(idx)]


# ---------------------------------------------------------------- masks

def rasterize_region(region, W, H):
    """把区域栅格化为 0..1 羽化覆盖率（以图像像素为坐标）。"""
    kind = region.get("kind", "brush")
    feather = max(0.0, float(region.get("feather", 0)))
    m = np.zeros((H, W), dtype=np.float32)

    if kind == "brush":
        size = max(1.0, float(region.get("size", 80)))
        r = size / 2.0
        yy, xx = np.mgrid[0:H, 0:W]
        for st in region.get("strokes", []):
            pts = st.get("points", [])
            for a, b in zip(pts, pts[1:]):
                x0, y0 = a["x"] * W, a["y"] * H
                x1, y1 = b["x"] * W, b["y"] * H
                steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / max(r, 1) * 0.5))
                for i in range(steps + 1):
                    t = i / steps
                    cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                    m[d2 <= r * r] = 1.0
    else:  # polygon
        from PIL import Image, ImageDraw
        poly = [(p["x"] * W, p["y"] * H) for p in region.get("points", [])]
        if len(poly) >= 3:
            img = Image.new("L", (W, H), 0)
            ImageDraw.Draw(img).polygon(poly, fill=255)
            m = (np.asarray(img, dtype=np.float32) / 255.0)

    if feather > 0:
        radius = feather / 2.0
        if radius >= 0.5:
            from PIL import Image, ImageFilter
            img = Image.fromarray((m * 255).astype(np.uint8))
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            m = np.asarray(img, dtype=np.float32) / 255.0
    return m


# ---------------------------------------------------------------- compute

def _steps(plan):
    return plan.get("steps", []) if isinstance(plan, dict) else []


def compute(base_gray, plan, cal, ref_size=1000):
    """累计曝光计算。
    base_gray: HxW float32，基础曝光下的灰度（由底片扫描的透光率映射）。
    返回 dict: gray, logE, masks{id}, base_map, out_hi/out_lo, contours, warnings。
    plan.split.enabled 时走分级滤镜双段合成（见 compute_split）。
    """
    if split_enabled(plan):
        return compute_split(base_gray, plan, cal, ref_size)
    H, W = base_gray.shape
    params = cal["params"]
    lut = build_lut(params)

    base_exposure = max(float(plan.get("base_exposure", 10.0)), 0.01)
    # 由基础曝光下每像素灰度反解 logistic 得到相对 log 曝光量 x_pix
    # （基础曝光的标准位置对应曲线 x0 附近；透光强处曝光更多）。
    Dbase = -np.log10(np.clip(base_gray, 1.0, 255.0) / 255.0)
    # logistic 反解:
    ratio = np.clip((Dbase - params["Dmin"]) /
                    (params["Dmax"] - params["Dmin"]), 1e-6, 1 - 1e-6)
    x_pix = params["x0"] - np.log10(1.0 / ratio - 1.0) / (math.log(10) * params["k"])

    # ---- 时间轴展开：区间列表 [a,b] ----
    steps = _steps(plan)
    intervals = []
    for s in steps:
        if s.get("type") == "dodge":
            t0 = float(s.get("start", 0))
            dur = base_exposure * float(s.get("ratio", 0))
            intervals.append((s, max(0.0, t0), t0 + dur))
        elif s.get("type") == "burn":
            t0 = float(s.get("start", base_exposure))
            dur = base_exposure * (2.0 ** float(s.get("stops", 0)) - 1.0)
            intervals.append((s, t0, t0 + dur))

    # ---- 累计：线性曝光能量（完整基础曝光为 1），遮挡减、加光加 ----
    logE = x_pix.astype(np.float64)
    mult = np.ones((H, W), dtype=np.float64)
    masks = {}
    scale = ref_size / max(H, W)
    for s, a, b in intervals:
        rid = s.get("region_id")
        region = _find_region(plan, rid)
        if region is None:
            continue
        if rid not in masks:
            masks[rid] = rasterize_region(region, W, H)
        cov = masks[rid]
        d = max(0.0, (b - a) / base_exposure)
        if s["type"] == "dodge":
            mult -= d * cov
        else:
            mult += d * cov
    logE = logE + np.log10(np.maximum(mult, 1e-6))

    gray = gray_for_logE(logE.astype(np.float32), lut).astype(np.float32)

    result = {
        "gray": gray,
        "logE": logE.astype(np.float32),
        "masks": masks,
        "base_map": base_gray.astype(np.float32),
        "intervals": [(s.get("id"), s.get("type"), round(a, 3), round(b, 3))
                      for s, a, b in intervals],
        "timeline_end": max([base_exposure] + [b for _, _, b in intervals]),
        "warnings": [],
    }

    result["warnings"] = detect_warnings(result, plan, cal, scale, intervals)
    result["contours"] = contours(gray, scale)
    return result


def _find_region(plan, rid):
    for r in plan.get("regions", []):
        if r.get("id") == rid:
            return r
    return None


# ---------------------------------------------------------------- split 合成

def _invert_x(base_gray, params):
    """由基础灰度反解每像素相对 log 曝光量（以 plan.base_exposure 为参考）。
    注意：单曲线模式沿用历史公式（除以 ln10·k，见 compute），旧方案行为不变；
    split 模式是新模型，这里使用 logistic 的精确反解（除以 k），
    使"软段单独曝光 base_ref 秒"恰好复现 base_gray。"""
    Dbase = -np.log10(np.clip(base_gray, 1.0, 255.0) / 255.0)
    ratio = np.clip((Dbase - params["Dmin"]) /
                    (params["Dmax"] - params["Dmin"]), 1e-6, 1 - 1e-6)
    return params["x0"] - np.log10(1.0 / ratio - 1.0) / params["k"]


def _net_density(logE, params):
    """净密度贡献：logistic(logE) − Dmin（直接求值，尾部不做 LUT 截断，
    高光透光区的 logE 可低至 −10 以下，LUT 截断会引入显著密度误差）。"""
    Dmin, Dmax, x0, k = (params["Dmin"], params["Dmax"],
                         params["x0"], params["k"])
    z = np.power(10.0, np.clip(-k * (logE - x0), -300, 300))
    return (Dmax - Dmin) / (1.0 + z)


def compute_split(base_gray, plan, cal, ref_size=1000):
    """分级滤镜双段合成。
    每像素：logE_soft/logE_hard 为两段各自的 log 曝光量（含局部步骤），
    总密度 = Dmin_ref + 净贡献_soft + 净贡献_hard，再换算灰度。
    同时输出 gray_soft/gray_hard 两个单段视图与换片时刻 swap_t。
    """
    H, W = base_gray.shape
    cal_soft, cal_hard = normalize_split(cal)
    ps, ph = cal_soft["params"], cal_hard["params"]
    dmin_ref = ps["Dmin"]
    t_soft, t_hard = split_times(plan)
    base_ref = max(float(plan.get("base_exposure", 10.0)), 0.01)

    # 底片透光率参数化：用软滤镜曲线反解（软段单独曝光 base_ref 秒时
    # 净密度恰为 base_gray 对应密度 − Dmin，保持与单曲线模式一致）
    x_pix = _invert_x(base_gray, ps).astype(np.float64)

    # 两路曝光能量（段基础曝光为 1，步骤按段基础时长归一增减）
    mult_s = np.ones((H, W), dtype=np.float64)
    mult_h = np.ones((H, W), dtype=np.float64)
    masks = {}
    scale = ref_size / max(H, W)
    segments, intervals, swap_t = split_layout(plan)
    for s, f, a, b in intervals:
        rid = s.get("region_id")
        region = _find_region(plan, rid)
        if region is None:
            continue
        if rid not in masks:
            masks[rid] = rasterize_region(region, W, H)
        cov = masks[rid]
        t_seg = t_soft if f == "soft" else t_hard
        d = max(0.0, (b - a) / t_seg)
        tgt = mult_s if f == "soft" else mult_h
        if s["type"] == "dodge":
            tgt -= d * cov
        else:
            tgt += d * cov

    logE_s = x_pix + np.log10(np.maximum(t_soft * mult_s, 1e-6) / base_ref)
    logE_h = x_pix + np.log10(np.maximum(t_hard * mult_h, 1e-6) / base_ref)
    net_s = _net_density(logE_s, ps)
    net_h = _net_density(logE_h, ph)

    def to_gray(dens):
        return np.clip(255.0 * np.power(10.0, -dens), 0, 255).astype(np.float32)

    gray = to_gray(dmin_ref + net_s + net_h)
    gray_soft = to_gray(dmin_ref + net_s)
    gray_hard = to_gray(dmin_ref + net_h)

    result = {
        "gray": gray,
        "gray_soft": gray_soft,
        "gray_hard": gray_hard,
        "logE": np.minimum(logE_s, logE_h).astype(np.float32),
        "logE_soft": logE_s.astype(np.float32),
        "logE_hard": logE_h.astype(np.float32),
        "masks": masks,
        "base_map": base_gray.astype(np.float32),
        "intervals": [(s.get("id"), s.get("type"), round(a, 3), round(b, 3), f)
                      for s, f, a, b in intervals],
        "segments": [(f, round(a, 3), round(b, 3))
                     for f, a, b in segments],
        "swap_t": swap_t,
        "timeline_end": segments[1][2],
        "warnings": [],
    }
    result["warnings"] = detect_warnings_split(
        result, plan, cal, cal_soft, cal_hard, scale, intervals)
    result["contours"] = contours(gray, scale)
    return result


# ---------------------------------------------------------------- warnings

def detect_warnings(result, plan, cal, scale, intervals):
    logE, gray = result["logE"], result["gray"]
    masks = result["masks"]
    H, W = gray.shape
    warns = []

    # 1) 超范围
    p = cal["params"]
    cr = cal.get("range")
    if cr:
        xmin, xmax = float(cr[0]), float(cr[1])
    else:
        # 未校准：用曲线近似线性段（5%..95%）
        L = math.log(10)

        def x_at(f):
            return p["x0"] + math.log10(f / (1 - f)) / (L * p["k"])
        xmin, xmax = x_at(0.03), x_at(0.97)

    out_hi = logE > xmax   # 曝光过量 -> 死黑高光（相纸上高光区域过暗）
    out_lo = logE < xmin   # 曝光不足 -> 无影阴影
    total = H * W
    for mask, kind, msg in ((out_hi, "over", "曝光超出响应范围：高光堵死（过曝）"),
                           (out_lo, "under", "曝光低于响应范围：阴影无影（欠曝）")):
        n = int(mask.sum())
        if n / total > 0.0005:
            ys, xs = np.where(mask)
            warns.append({
                "kind": "out_of_range:" + kind,
                "severity": "error",
                "message": msg + f"（{n/total*100:.1f}% 面积）",
                "bbox": [float(xs.min()) / W, float(ys.min()) / H,
                         float(xs.max()) / W, float(ys.max()) / H],
                "time": None,
            })

    # 2) 边缘跳变：|grad gray| 阈值，参考分辨率归一
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gmag = np.hypot(gx, gy) * scale  # 归一到每 1000px 宽
    band = np.zeros((H, W), dtype=bool)
    for cov in masks.values():
        edge = cov > 0.03
        # 以 EDGELEN 像素为半宽膨胀，得到工具边缘窄带
        if edge.any():
            from PIL import Image, ImageFilter
            k = EDGELEN * 2 + 1
            dil = np.asarray(
                Image.fromarray((edge * 255).astype(np.uint8))
                     .filter(ImageFilter.MaxFilter(k))
            ) > 0
            band |= dil
    jump = (gmag > EDGE_JUMP_LEVELS) & band
    n = int(jump.sum())
    if n / total > 1e-5:
        ys, xs = np.where(jump)
        warns.append({
            "kind": "edge_jump",
            "severity": "warn",
            "message": f"工具边缘曝光跳变 {EDGE_JUMP_LEVELS:.0f} 灰阶/像素以上，"
                       f"建议加大羽化（{n} 像素）",
            "bbox": [float(xs.min()) / W, float(ys.min()) / H,
                     float(xs.max()) / W, float(ys.max()) / H],
            "time": None,
        })

    # 3) 互相抵消 & 4) 双工具时段：按步骤对检查
    base_e = max(float(plan.get("base_exposure", 10)), 0.01)
    for i in range(len(intervals)):
        s1, a1, b1 = intervals[i]
        for j in range(i + 1, len(intervals)):
            s2, a2, b2 = intervals[j]
            ov_a, ov_b = max(a1, a2), min(b1, b2)
            r1 = _find_region(plan, s1.get("region_id"))
            r2 = _find_region(plan, s2.get("region_id"))
            if r1 is None or r2 is None:
                continue
            m1 = masks.get(s1["region_id"])
            m2 = masks.get(s2["region_id"])
            if m1 is None or m2 is None:
                continue
            inter = (m1 > 0.1) & (m2 > 0.1)
            union = (m1 > 0.1) | (m2 > 0.1)
            iou = float(inter.sum()) / max(int(union.sum()), 1)
            cover_min = float(inter.sum()) / max(
                int((m1 > 0.1).sum()), int((m2 > 0.1).sum()), 1)

            # 双工具：时间重叠且区域相交（任意两件手持工具同时工作）
            if ov_b - ov_a > 0.05 and iou > 0.02:
                warns.append({
                    "kind": "two_tools",
                    "severity": "warn",
                    "message": f"{r1.get('name','区域')} 与 {r2.get('name','区域')} "
                               f"在 {ov_a:.1f}–{ov_b:.1f}s 需同时移动两件工具",
                    "steps": [s1.get("id"), s2.get("id")],
                    "time": [round(ov_a, 2), round(ov_b, 2)],
                })

            # 互相抵消：一遮一加、区域高度重叠且曝光能量近似平衡
            # （遮挡减少 d1 倍基础曝光，加光增加 d2 倍，min/max 接近 1 即抵消）
            if {s1["type"], s2["type"]} == {"dodge", "burn"} and cover_min > CANCEL_RATIO:
                d1 = (b1 - a1) / base_e
                d2 = (b2 - a2) / base_e
                balance = min(d1, d2) / max(d1, d2)
                if balance > 0.5:
                    warns.append({
                        "kind": "cancel",
                        "severity": "warn",
                        "message": f"{r1.get('name','区域')} 的遮挡与 {r2.get('name','区域')} "
                                   f"的加光在同一区域互相抵消（覆盖 {cover_min*100:.0f}%，"
                                   f"能量差 {abs(d1-d2)/max(d1,d2)*100:.0f}%），请取舍",
                        "steps": [s1.get("id"), s2.get("id")],
                        "bbox": _bbox_of(inter),
                        "time": None,
                    })
    return warns


def _bbox_of(mask):
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()) / W, float(ys.min()) / H,
            float(xs.max()) / W, float(ys.max()) / H]


# ---------------------------------------------------------------- split 求解

def _logistic_scalar(x, p):
    return p["Dmin"] + (p["Dmax"] - p["Dmin"]) / \
        (1.0 + 10.0 ** (-p["k"] * (x - p["x0"])))


def _logistic_deriv(x, p):
    """dD/dx，x 为 log10 相对曝光量。"""
    z = 10.0 ** (-p["k"] * (x - p["x0"]))
    sig = 1.0 / (1.0 + z)
    return (p["Dmax"] - p["Dmin"]) * math.log(10) * p["k"] * sig * (1 - sig)


def split_target_density(gray_target, dmin_ref):
    """目标灰度 → 需要两段共同提供的净密度。"""
    g = min(255.0, max(1.0, float(gray_target)))
    return math.log10(255.0 / g) - dmin_ref


def solve_split_exposures(ps, ph, dmin_ref, targets, base_ref,
                          t_lo=SPLIT_T_MIN, t_hi=SPLIT_T_MAX,
                          t_soft0=None, t_hard0=None):
    """搜索两路基础曝光时长，使各目标点达到目标灰度。
    targets: [{x_pix, m_soft, m_hard, gray}]，每点一个方程：
        net_soft(x + log10(m_s·t_s/B)) + net_hard(x + log10(m_h·t_h/B))
            = D(gray) − Dmin_ref
    两点时用阻尼牛顿解 2×2 系统；单点时退化为单变量二分
    （另一路时长保持初值）。返回 {t_soft, t_hard, ok, residual,
    unreachable:[...]}。unreachable 列出超出可达范围的目标下标。
    """
    pts = []
    for t in targets:
        if t is None:
            continue
        pts.append({
            "a_s": float(t["x_pix"]) +
                   math.log10(max(float(t.get("m_soft", 1.0)), 1e-6) / base_ref),
            "a_h": float(t["x_pix"]) +
                   math.log10(max(float(t.get("m_hard", 1.0)), 1e-6) / base_ref),
            "dt": split_target_density(t.get("gray", 128), dmin_ref),
        })
    if not pts:
        return {"t_soft": t_soft0, "t_hard": t_hard0, "ok": False,
                "residual": None, "unreachable": []}

    # 可达范围：净密度 ∈ [0, 两路最大净贡献之和]（时长界限内）
    lo_u, hi_u = math.log10(t_lo), math.log10(t_hi)
    unreachable = []
    for i, q in enumerate(pts):
        net_max = (_logistic_scalar(q["a_s"] + hi_u, ps) - ps["Dmin"] +
                   _logistic_scalar(q["a_h"] + hi_u, ph) - ph["Dmin"])
        net_min = (_logistic_scalar(q["a_s"] + lo_u, ps) - ps["Dmin"] +
                   _logistic_scalar(q["a_h"] + lo_u, ph) - ph["Dmin"])
        if q["dt"] < net_min - 1e-9 or q["dt"] > net_max + 1e-9:
            unreachable.append(i)

    def resid(u, v):
        return [(_logistic_scalar(q["a_s"] + u, ps) - ps["Dmin"] +
                 _logistic_scalar(q["a_h"] + v, ph) - ph["Dmin"]) - q["dt"]
                for q in pts]

    u = math.log10(min(max(t_soft0 or 10.0, t_lo), t_hi))
    v = math.log10(min(max(t_hard0 or 10.0, t_lo), t_hi))
    ok = False
    r = resid(u, v)
    if len(pts) == 1:
        # 单目标：只调软路（高光主要由低反差段控制），硬路保持
        for _ in range(80):
            r = resid(u, v)[0]
            if abs(r) < 1e-5:
                ok = True
                break
            d = _logistic_deriv(pts[0]["a_s"] + u, ps)
            if d < 1e-9:
                break
            u = min(hi_u, max(lo_u, u - r / d))
        r = resid(u, v)
    else:
        for _ in range(80):
            r = resid(u, v)
            if max(abs(r[0]), abs(r[1])) < 1e-5:
                ok = True
                break
            j11 = _logistic_deriv(pts[0]["a_s"] + u, ps)
            j12 = _logistic_deriv(pts[0]["a_h"] + v, ph)
            j21 = _logistic_deriv(pts[1]["a_s"] + u, ps)
            j22 = _logistic_deriv(pts[1]["a_h"] + v, ph)
            det = j11 * j22 - j12 * j21
            if abs(det) < 1e-12:
                break
            du = (r[0] * j22 - r[1] * j12) / det
            dv = (j11 * r[1] - j21 * r[0]) / det
            # 阻尼：单步不超过 0.5 个数量级，越界则收缩
            for damp in (1.0, 0.5, 0.25, 0.1):
                nu = min(hi_u, max(lo_u, u - damp * du))
                nv = min(hi_u, max(lo_u, v - damp * dv))
                nr = resid(nu, nv)
                if max(abs(nr[0]), abs(nr[1])) < max(abs(r[0]), abs(r[1])):
                    u, v = nu, nv
                    break
            else:
                break
        r = resid(u, v)
        ok = max(abs(r[0]), abs(r[1])) < 1e-3

    return {"t_soft": round(10.0 ** u, 2), "t_hard": round(10.0 ** v, 2),
            "ok": ok and not unreachable,
            "residual": max(abs(x) for x in r),
            "unreachable": unreachable}


# ---------------------------------------------------------------- split 检测

def detect_warnings_split(result, plan, cal, cal_soft, cal_hard, scale,
                          intervals):
    """split 模式问题检测：沿用单曲线的范围/边缘/抵消/双工具检查
    （对两路 logE 分别判范围），另加：
      split_cal    校准点不足（n<2）
      split_target 目标灰度超出两条曲线的可达范围
      split_swap   局部曝光跨越换片节点
      split_seg    步骤落在其滤镜段之外
    """
    warns = []
    H, W = result["gray"].shape
    total = H * W
    t_swap = result["swap_t"]
    segs = dict((f, (a, b)) for f, a, b in result["segments"])

    # 1) 校准点不足 → 定位到校准面板
    for f, c, label in (("soft", cal_soft, "低反差"), ("hard", cal_hard, "高反差")):
        n = int(c.get("n") or len(c.get("points") or []))
        if n < 2:
            warns.append({
                "kind": "split_cal", "severity": "error",
                "message": f"{label}滤镜校准点不足（{n}/2）："
                           f"请录入阶梯曝光并拟合后再编排",
                "cal": f, "time": None,
            })

    # 2) 两路 logE 超各自校准范围
    for logE, c, label, tag in (
            (result["logE_soft"], cal_soft, "低反差段", "soft"),
            (result["logE_hard"], cal_hard, "高反差段", "hard")):
        cr = c.get("range")
        p = c["params"]
        if cr:
            xmin, xmax = float(cr[0]), float(cr[1])
        else:
            L = math.log(10)
            xmin = p["x0"] + math.log10(0.03 / 0.97) / (L * p["k"])
            xmax = p["x0"] + math.log10(0.97 / 0.03) / (L * p["k"])
        for mask, kind, msg in (
                (logE > xmax, "over", "曝光超出响应范围：高光堵死（过曝）"),
                (logE < xmin, "under", "曝光低于响应范围：阴影无影（欠曝）")):
            n = int(mask.sum())
            if n / total > 0.0005:
                ys, xs = np.where(mask)
                warns.append({
                    "kind": f"out_of_range:{kind}", "severity": "error",
                    "message": f"{label}：{msg}（{n/total*100:.1f}% 面积）",
                    "bbox": [float(xs.min()) / W, float(ys.min()) / H,
                             float(xs.max()) / W, float(ys.max()) / H],
                    "time": None,
                })

    # 3) 目标可达性（高光/阴影目标点）
    sp = plan.get("split") or {}
    tg = sp.get("targets") or {}
    base_ref = max(float(plan.get("base_exposure", 10.0)), 0.01)
    ps, ph = cal_soft["params"], cal_hard["params"]
    pts, labels = [], []
    for key, label in (("highlight", "高光"), ("shadow", "阴影")):
        t = tg.get(key)
        if t and t.get("gray") is not None:
            xi = min(W - 1, max(0, int(round(float(t["x"]) * W))))
            yi = min(H - 1, max(0, int(round(float(t["y"]) * H))))
            # 目标点处的两路能量系数（由当前步骤布局决定；
            # 与 compute_split 一致，按段基础时长归一）
            t_soft, t_hard = split_times(plan)
            m_s = m_h = 1.0
            for s, f, a, b in intervals:
                cov = result["masks"].get(s.get("region_id"))
                if cov is None:
                    continue
                t_seg = t_soft if f == "soft" else t_hard
                d = max(0.0, (b - a) / max(t_seg, 0.01)) * float(cov[yi, xi])
                if f == "soft":
                    m_s += -d if s["type"] == "dodge" else d
                else:
                    m_h += -d if s["type"] == "dodge" else d
            x_pix = float(_invert_x(
                result["base_map"][yi:yi + 1, xi:xi + 1], ps)[0, 0])
            pts.append({"x_pix": x_pix, "m_soft": max(m_s, 1e-6),
                        "m_hard": max(m_h, 1e-6), "gray": t["gray"]})
            labels.append((key, label, t))
    if pts:
        sol = solve_split_exposures(
            ps, ph, ps["Dmin"], pts, base_ref,
            t_soft0=sp.get("t_soft"), t_hard0=sp.get("t_hard"))
        for i in sol["unreachable"]:
            key, label, t = labels[i]
            warns.append({
                "kind": "split_target", "severity": "error",
                "message": f"{label}目标灰度 {t['gray']} 超出两条曲线的可达范围"
                           f"（{SPLIT_T_MIN:g}–{SPLIT_T_MAX:g}s 内无法达到），"
                           f"请调整目标或加长搜索时长",
                "point": [float(t["x"]), float(t["y"])], "which": key,
                "time": None,
            })
        if not sol["ok"] and not sol["unreachable"] and len(pts) >= 2:
            # 各自可达但联立无解（两目标相互牵制）
            key, label, t = labels[-1]
            warns.append({
                "kind": "split_target", "severity": "error",
                "message": "高光与阴影目标无法同时满足：两路时长搜索未收敛，"
                           "请放宽其中一个目标",
                "point": [float(t["x"]), float(t["y"])], "which": key,
                "time": None,
            })

    # 4) 跨换片 & 段外步骤
    regions = {r["id"]: r for r in plan.get("regions", [])}
    for s, f, a, b in intervals:
        reg = regions.get(s.get("region_id"), {})
        nm = reg.get("name", "区域")
        seg0, seg1 = segs[f]
        label = "低反差" if f == "soft" else "高反差"
        if a < t_swap < b:
            warns.append({
                "kind": "split_swap", "severity": "warn",
                "message": f"「{nm}」的{label}滤镜曝光跨越换片节点 "
                           f"{t_swap:.1f}s（{a:.1f}–{b:.1f}s）：换片期间该步骤"
                           f"无法连续执行，请缩短或改到另一段",
                "steps": [s.get("id")],
                "time": [round(a, 2), round(b, 2)],
            })
        elif b <= seg0 or a >= seg1:
            warns.append({
                "kind": "split_seg", "severity": "warn",
                "message": f"「{nm}」被指定为{label}滤镜，但其时段 "
                           f"{a:.1f}–{b:.1f}s 不在该滤镜段 "
                           f"{seg0:.1f}–{seg1:.1f}s 内",
                "steps": [s.get("id")],
                "time": [round(a, 2), round(b, 2)],
            })

    # 5) 边缘跳变 / 抵消 / 双工具：复用单曲线检测（基于合成灰度）
    pseudo = dict(result)
    pseudo["logE"] = result["logE"]
    t_soft, t_hard = split_times(plan)
    warns.extend(_detect_common(
        pseudo, plan, scale, intervals,
        seg_lens={"soft": t_soft, "hard": t_hard}))
    return warns


def _detect_common(result, plan, scale, intervals, seg_lens=None):
    """边缘跳变、互相抵消、双工具三类与模式无关的检测。
    intervals 为 (step, filter, a, b)；seg_lens 给出各滤镜段长，
    用于把步骤时长归一为基础曝光能量比例。"""
    gray = result["gray"]
    masks = result["masks"]
    H, W = gray.shape
    total = H * W
    warns = []
    seg_lens = seg_lens or {}

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gmag = np.hypot(gx, gy) * scale
    band = np.zeros((H, W), dtype=bool)
    for cov in masks.values():
        edge = cov > 0.03
        if edge.any():
            from PIL import Image, ImageFilter
            k = EDGELEN * 2 + 1
            dil = np.asarray(
                Image.fromarray((edge * 255).astype(np.uint8))
                     .filter(ImageFilter.MaxFilter(k))
            ) > 0
            band |= dil
    jump = (gmag > EDGE_JUMP_LEVELS) & band
    n = int(jump.sum())
    if n / total > 1e-5:
        ys, xs = np.where(jump)
        warns.append({
            "kind": "edge_jump",
            "severity": "warn",
            "message": f"工具边缘曝光跳变 {EDGE_JUMP_LEVELS:.0f} 灰阶/像素以上，"
                       f"建议加大羽化（{n} 像素）",
            "bbox": [float(xs.min()) / W, float(ys.min()) / H,
                     float(xs.max()) / W, float(ys.max()) / H],
            "time": None,
        })

    for i in range(len(intervals)):
        s1, a1, b1 = intervals[i][0], intervals[i][2], intervals[i][3]
        for j in range(i + 1, len(intervals)):
            s2, a2, b2 = intervals[j][0], intervals[j][2], intervals[j][3]
            ov_a, ov_b = max(a1, a2), min(b1, b2)
            r1 = _find_region(plan, s1.get("region_id"))
            r2 = _find_region(plan, s2.get("region_id"))
            if r1 is None or r2 is None:
                continue
            m1 = masks.get(s1["region_id"])
            m2 = masks.get(s2["region_id"])
            if m1 is None or m2 is None:
                continue
            inter = (m1 > 0.1) & (m2 > 0.1)
            union = (m1 > 0.1) | (m2 > 0.1)
            iou = float(inter.sum()) / max(int(union.sum()), 1)
            cover_min = float(inter.sum()) / max(
                int((m1 > 0.1).sum()), int((m2 > 0.1).sum()), 1)
            if ov_b - ov_a > 0.05 and iou > 0.02:
                warns.append({
                    "kind": "two_tools",
                    "severity": "warn",
                    "message": f"{r1.get('name','区域')} 与 {r2.get('name','区域')} "
                               f"在 {ov_a:.1f}–{ov_b:.1f}s 需同时移动两件工具",
                    "steps": [s1.get("id"), s2.get("id")],
                    "time": [round(ov_a, 2), round(ov_b, 2)],
                })
            if {s1["type"], s2["type"]} == {"dodge", "burn"} and \
                    cover_min > CANCEL_RATIO:
                f1, f2 = intervals[i][1], intervals[j][1]
                d1 = (b1 - a1) / max(seg_lens.get(f1, 1.0), 0.01)
                d2 = (b2 - a2) / max(seg_lens.get(f2, 1.0), 0.01)
                balance = min(d1, d2) / max(d1, d2)
                if balance > 0.5:
                    warns.append({
                        "kind": "cancel",
                        "severity": "warn",
                        "message": f"{r1.get('name','区域')} 的遮挡与 "
                                   f"{r2.get('name','区域')} 的加光在同一区域"
                                   f"互相抵消（覆盖 {cover_min*100:.0f}%，"
                                   f"能量差 {abs(d1-d2)/max(d1,d2)*100:.0f}%），"
                                   f"请取舍",
                        "steps": [s1.get("id"), s2.get("id")],
                        "bbox": _bbox_of(inter),
                        "time": None,
                    })
    return warns


# ---------------------------------------------------------------- contours

CONTOUR_LEVELS = list(range(32, 255, 32))  # 255-32n: 32,64,...,224


def contours(gray, scale):
    """Marching squares 提取等密度线，坐标归一化到 0..1。"""
    H, W = gray.shape
    tl, tr = gray[:-1, :-1], gray[:-1, 1:]
    bl, br = gray[1:, :-1], gray[1:, 1:]
    segs = []
    for level in CONTOUR_LEVELS:
        bit = (gray > level).astype(np.uint8)
        v = bit[:-1, :-1] | bit[:-1, 1:] << 1 | bit[1:, 1:] << 2 | bit[1:, :-1] << 3

        for case in range(1, 15):
            sel = v == case
            if not sel.any():
                continue
            yc = np.argwhere(sel)[:, 0].astype(np.float32)
            xc = np.argwhere(sel)[:, 1].astype(np.float32)
            corner = {"tl": tl[sel], "tr": tr[sel], "bl": bl[sel], "br": br[sel]}

            def edge_point(name):
                if name == "t":
                    t = _it(corner["tl"], corner["tr"], level)
                    return xc + t, yc
                if name == "r":
                    t = _it(corner["tr"], corner["br"], level)
                    return xc + 1, yc + t
                if name == "b":
                    t = _it(corner["bl"], corner["br"], level)
                    return xc + t, yc + 1
                t = _it(corner["tl"], corner["bl"], level)
                return xc, yc + t

            for e0, e1 in _MARCH[case]:
                x0, y0 = edge_point(e0)
                x1, y1 = edge_point(e1)
                for a, b_, c, d in zip(x0, y0, x1, y1):
                    segs.append((float(a) / W, float(b_) / H,
                                 float(c) / W, float(d) / H, level))
    return segs


def _it(v0, v1, level):
    return np.clip((level - v0) / np.where(v1 == v0, 1.0, v1 - v0), 0.0, 1.0)


_MARCH = {
    1: [("t", "l")], 2: [("t", "r")], 3: [("l", "r")],
    4: [("r", "b")], 5: [("t", "r"), ("b", "l")],
    6: [("t", "b")], 7: [("l", "b")], 8: [("l", "b")],
    9: [("t", "b")], 10: [("t", "l"), ("r", "b")],
    11: [("r", "b")], 12: [("l", "r")], 13: [("t", "r")],
    14: [("t", "l")],
}
