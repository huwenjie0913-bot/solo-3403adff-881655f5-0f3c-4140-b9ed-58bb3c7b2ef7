// engine.js —— 与 app/engine.py 同构的浏览器端计算：
// 响应曲线拟合、LUT、区域栅格化、累计曝光、问题检测、等密度线。

export const X_LO = -3, X_HI = 7, LUT_N = 8192;
export const GRADE_K = {0: 0.35, 1: 0.55, 2: 0.85, 3: 1.2, 4: 1.6, 5: 2.1};
const EDGE_JUMP = 3, CANCEL_RATIO = 0.7, EDGELEN = 10;
const MARCH = {
  1: [['t','l']], 2: [['t','r']], 3: [['l','r']], 4: [['r','b']],
  5: [['t','r'],['b','l']], 6: [['t','b']], 7: [['l','b']],
  8: [['l','b']], 9: [['t','b']], 10: [['t','l'],['r','b']],
  11: [['r','b']], 12: [['l','r']], 13: [['t','r']], 14: [['t','l']],
};
const LN10 = Math.log(10);

function logistic(x, p) {
  return p.Dmin + (p.Dmax - p.Dmin) / (1 + Math.exp(-LN10 * p.k * (x - p.x0)));
}

// ------------------------------------------------------------------ 拟合

export function fitCurve(points, grade = 2, baseExposure = 10) {
  const pts = points.filter(p => p.t > 0 && p.gray >= 0 && p.gray <= 255);
  if (pts.length < 2) {
    return {params: defaultParams(grade, baseExposure), rms: null,
            range: [], fitted: [], points: [], n: pts.length, calibrated: false};
  }
  const x = pts.map(p => Math.log10(Math.max(p.t / baseExposure, 1e-9)));
  const g = pts.map(p => p.gray);
  const sorted = x.map((v, i) => [v, g[i]]).sort((a, b) => a[0] - b[0]);

  const gAsc = [...g].sort((a, b) => a - b);
  const q = Math.max(1, Math.floor(g.length / 4));
  const bright = gAsc.slice(-q).reduce((a, b) => a + b, 0) / q;
  const dark = gAsc.slice(0, q).reduce((a, b) => a + b, 0) / q;
  let Dmin0 = Math.log10(255 / Math.max(bright, 4));
  let Dmax0 = Math.log10(255 / Math.max(dark, 4));
  if (Dmax0 < Dmin0 + 0.1) Dmax0 = Dmin0 + 1;
  const gmid = (bright + dark) / 2;
  let x00 = interp1(sorted.map(s => s[1]), sorted.map(s => s[0]), gmid);
  if (!isFinite(x00)) x00 = 0;

  let theta = [clamp(Dmin0, 0, 1.2), clamp(Dmax0, 0.3, 4), x00,
               GRADE_K[grade] ?? 0.85];
  const lo = [1e-4, 0.3 + 1e-4, -2.5 + 1e-4, 0.05 + 1e-4];
  const hi = [1.2 - 1e-4, 4 - 1e-4, 6 - 1e-4, 5 - 1e-4];

  const model = th => x.map(xv => 255 * 10 ** -logistic(xv,
        {Dmin: th[0], Dmax: th[1], x0: th[2], k: th[3]}));
  const rmsOf = th => Math.sqrt(mean(model(th).map((v, i) => (v - g[i]) ** 2)));

  let rmsLast = rmsOf(theta), lam = 1e-5;
  for (let it = 0; it < 80; it++) {
    const [Dmin, Dmax, x0, k] = theta;
    const sig = x.map(xv => 1 / (1 + Math.exp(-LN10 * k * (xv - x0))));
    const ghat = model(theta);
    const J = x.map((xv, i) => {
      const s = sig[i];
      return [-LN10 * ghat[i] * (1 - s),
              -LN10 * ghat[i] * s,
               LN10 * k * (Dmax - Dmin) * ghat[i] * s * (1 - s),
              -LN10 * (xv - x0) * (Dmax - Dmin) * ghat[i] * s * (1 - s)];
    });
    const r = ghat.map((v, i) => v - g[i]);
    let best = null, bestRms = rmsLast;
    for (const lamTry of [lam, lam * 10, lam * 100,
                          lam > 1e-7 ? lam / 10 : lam]) {
      let delta;
      try {
        delta = solve(J, r, lamTry);
      } catch (e) { continue; }
      for (const step of [1, 0.5, 0.25]) {
        const cand = theta.map((v, i) =>
          clamp(v + delta[i] * step, lo[i], hi[i]));
        const rc = rmsOf(cand);
        if (rc < bestRms - 1e-6) { best = cand; bestRms = rc; }
      }
      if (best) { lam = lamTry * 0.5; break; }
    }
    if (!best) { lam *= 10; if (lam > 1e6) break; continue; }
    if (Math.abs(rmsLast - bestRms) < 1e-5) { theta = best; break; }
    theta = best; rmsLast = bestRms;
  }

  const params = {Dmin: theta[0], Dmax: theta[1], x0: theta[2], k: theta[3]};
  const rms = rmsOf(theta);
  const xs = sorted.map(s => s[0]);
  const fitted = [];
  for (let i = 0; i < 160; i++) {
    const xv = X_LO + (X_HI - X_LO) * i / 159;
    fitted.push({x: xv, gray: clamp(255 * 10 ** -logistic(xv, params), 0, 255)});
  }
  return {params, rms, range: [xs[0], xs[xs.length - 1]], fitted,
          points: x.map((xv, i) => ({x: xv, gray: g[i]})),
          n: pts.length, calibrated: true};
}

export function defaultParams(grade = 2) {
  return {Dmin: 0.06, Dmax: 2.05, x0: 0, k: GRADE_K[grade] ?? 0.85};
}

// 小规模稠密线性方程（4x4，Gauss-Jordan），求解 (J'J+λdiag) δ = -J'r
function solve(J, r, lam) {
  const n = 4;
  const A = Array.from({length: n}, () => new Float64Array(n + 1));
  for (const row of J) {
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) A[a][b] += row[a] * row[b];
    }
  }
  // 右端项：-J^T r
  for (let i = 0; i < J.length; i++)
    for (let a = 0; a < n; a++) A[a][n] -= J[i][a] * r[i];
  for (let a = 0; a < n; a++)
    A[a][a] += lam * (A[a][a] + 1e-12);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r2 = col + 1; r2 < n; r2++)
      if (Math.abs(A[r2][col]) > Math.abs(A[piv][col])) piv = r2;
    [A[piv], A[col]] = [A[col], A[piv]];
    if (Math.abs(A[col][col]) < 1e-14) throw new Error('singular');
    for (let r2 = 0; r2 < n; r2++) {
      if (r2 === col) continue;
      const f = A[r2][col] / A[col][col];
      for (let c2 = col; c2 <= n; c2++) A[r2][c2] -= f * A[col][c2];
    }
  }
  return Array.from({length: n}, (_, i) => A[i][n] / A[i][i]);
}

// ------------------------------------------------------------------ 栅格化

// 在 ctx（W×H 离屏画布）上把区域画成 alpha 0..255，返回 ImageData alpha
export function rasterize(region, W, H) {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  const feather = +region.feather || 0;
  ctx.save();
  if (feather > 0.5) ctx.filter = `blur(${feather / 2}px)`;

  if (region.kind === 'brush') {
    const size = Math.max(1, +region.size || 80);
    const r = size / 2;
    ctx.fillStyle = '#fff';
    for (const st of region.strokes || []) {
      const pts = st.points || [];
      for (let i = 0; i < pts.length; i++) {
        const last = i === pts.length - 1;
        if (!last) {
          ctx.beginPath();
          ctx.lineWidth = size;
          ctx.lineCap = 'round';
          ctx.moveTo(pts[i].x * W, pts[i].y * H);
          ctx.lineTo(pts[i + 1].x * W, pts[i + 1].y * H);
          ctx.strokeStyle = '#fff';
          ctx.stroke();
        }
      }
      for (const pt of pts) {
        ctx.beginPath();
        ctx.arc(pt.x * W, pt.y * H, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  } else if ((region.points || []).length >= 3) {
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    region.points.forEach((p, i) => {
      const X = p.x * W, Y = p.y * H;
      i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    });
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
  const data = ctx.getImageData(0, 0, W, H).data;
  const out = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = data[i * 4] / 255;
  return out;
}

// ------------------------------------------------------------------ 计算

export function regionSignature(r) {
  return JSON.stringify([r.kind, r.size | 0, r.feather | 0,
                         r.points || [], r.strokes || []]);
}

export function intervalsOf(plan) {
  const base = Math.max(+plan.base_exposure || 10, 0.01);
  const out = [];
  for (const s of plan.steps || []) {
    if (s.type === 'dodge') {
      const a = Math.max(0, +s.start || 0);
      out.push({s, a, b: a + base * (+s.ratio || 0)});
    } else {
      const a = (s.start === null || s.start === undefined) ? base : +s.start;
      const d = base * (2 ** (+s.stops || 0) - 1);
      out.push({s, a, b: a + d});
    }
  }
  return out;
}

export function buildLUT(p) {
  const lut = new Float32Array(LUT_N);
  for (let i = 0; i < LUT_N; i++) {
    const xv = X_LO + (X_HI - X_LO) * i / (LUT_N - 1);
    lut[i] = clamp(255 * 10 ** -logistic(xv, p), 0, 255);
  }
  return lut;
}

export function compute(baseGray, W, H, plan, cal, maskCache = null, time = null) {
  const p = cal.params || defaultParams(2);
  const lut = buildLUT(p);
  const base = Math.max(+plan.base_exposure || 10, 0.01);

  // 由基础灰度反解 x_pix：完整基础曝光下每像素的 log10 相对曝光量
  const xPix = new Float64Array(W * H);
  const span = p.Dmax - p.Dmin;
  for (let i = 0; i < W * H; i++) {
    const D = -Math.log10(clamp(baseGray[i], 1, 255) / 255);
    const ratio = clamp((D - p.Dmin) / span, 1e-6, 1 - 1e-6);
    xPix[i] = p.x0 - Math.log10(1 / ratio - 1) / (LN10 * p.k);
  }

  const intervals = intervalsOf(plan);
  const masks = {};
  const scrub = time !== null && time !== undefined;
  // 曝光能量（以完整基础曝光为 1）：基础均匀 1（擦洗时按比例），
  // 遮挡窗口内无曝光（减去），加光窗口额外给光（加上）。
  const mult = new Float64Array(W * H);
  mult.fill(scrub ? Math.max(0, Math.min(time, base)) / base : 1);

  for (const {s, a, b} of intervals) {
    const rid = s.region_id;
    const reg = regionById(plan, rid);
    if (!reg) continue;
    let cov;
    if (maskCache && maskCache[rid] && maskCache[rid].sig === regionSignature(reg)) {
      cov = maskCache[rid].data;
    } else {
      cov = rasterize(reg, W, H);
      if (maskCache) maskCache[rid] = {sig: regionSignature(reg), data: cov};
    }
    masks[rid] = cov;

    let d;
    if (scrub) {
      const oa = Math.max(a, 0), ob = Math.min(b, time);
      d = Math.max(0, ob - oa) / base;
    } else {
      d = (b - a) / base;
    }
    if (s.type === 'dodge') {
      for (let i = 0; i < W * H; i++) mult[i] -= d * cov[i];
    } else {
      for (let i = 0; i < W * H; i++) mult[i] += d * cov[i];
    }
  }

  const logE = new Float64Array(W * H);
  const gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    logE[i] = xPix[i] + Math.log10(Math.max(mult[i], 1e-6));
    const idx = clamp((logE[i] - X_LO) / (X_HI - X_LO) * (LUT_N - 1), 0, LUT_N - 1);
    gray[i] = lut[idx | 0];
  }
  const end = Math.max(base, ...intervals.map(v => v.b));

  const warnings = scrub ? [] : detect(logE, gray, W, H, plan, cal, masks, intervals);
  const contours = extractContours(gray, W, H);
  return {gray, logE, masks, intervals, timelineEnd: end,
          warnings, contours, W, H};
}

function regionById(plan, id) {
  return (plan.regions || []).find(r => r.id === id);
}

// ------------------------------------------------------------------ 检测

function detect(logE, gray, W, H, plan, cal, masks, intervals) {
  const warns = [];
  const p = cal.params;
  let xmin, xmax;
  if (cal.range && cal.range.length === 2) {
    [xmin, xmax] = cal.range;
  } else {
    const xAt = f => p.x0 + Math.log10(f / (1 - f)) / (LN10 * p.k);
    xmin = xAt(0.03); xmax = xAt(0.97);
  }
  const total = W * H;
  let bboxOver = null, bboxUnder = null, nOver = 0, nUnder = 0;
  for (let i = 0; i < total; i++) {
    if (logE[i] > xmax) { nOver++; bboxOver = addBBox(bboxOver, i, W, H); }
    if (logE[i] < xmin) { nUnder++; bboxUnder = addBBox(bboxUnder, i, W, H); }
  }
  if (nOver / total > 0.0005)
    warns.push({kind: 'out_of_range:over', severity: 'error',
      message: `曝光超出响应范围：高光堵死（过曝，${(nOver/total*100).toFixed(1)}% 面积）`,
      bbox: bboxOver, time: null});
  if (nUnder / total > 0.0005)
    warns.push({kind: 'out_of_range:under', severity: 'error',
      message: `曝光低于响应范围：阴影无影（欠曝，${(nUnder/total*100).toFixed(1)}% 面积）`,
      bbox: bboxUnder, time: null});

  // 边缘跳变：在掩膜膨胀窄带内检测梯度
  const scale = 1000 / Math.max(W, H);
  const band = new Uint8Array(total);
  for (const rid in masks) dilateBand(masks[rid], W, H, band);
  let jumpBox = null, nJump = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (!band[i]) continue;
      const gx = gray[i + 1] - gray[i - 1];
      const gy = gray[i + W] - gray[i - W];
      if (Math.hypot(gx, gy) * scale > EDGE_JUMP) {
        nJump++; jumpBox = addBBox(jumpBox, i, W, H);
      }
    }
  }
  if (nJump / total > 1e-5)
    warns.push({kind: 'edge_jump', severity: 'warn',
      message: `工具边缘曝光跳变 ${EDGE_JUMP} 灰阶/像素以上，建议加大羽化（${nJump} 像素）`,
      bbox: jumpBox, time: null});

  // 步骤对
  const baseExp = Math.max(+plan.base_exposure || 10, 0.01);
  for (let i = 0; i < intervals.length; i++) {
    for (let j = i + 1; j < intervals.length; j++) {
      const A = intervals[i], B = intervals[j];
      const oa = Math.max(A.a, B.a), ob = Math.min(A.b, B.b);
      const r1 = regionById(plan, A.s.region_id);
      const r2 = regionById(plan, B.s.region_id);
      if (!r1 || !r2) continue;
      const m1 = masks[A.s.region_id], m2 = masks[B.s.region_id];
      if (!m1 || !m2) continue;
      let inter = 0, union = 0, area1 = 0, area2 = 0;
      let box = null;
      for (let k = 0; k < total; k++) {
        const in1 = m1[k] > 0.1, in2 = m2[k] > 0.1;
        if (in1) area1++;
        if (in2) area2++;
        if (in1 && in2) { inter++; box = addBBox(box, k, W, H); }
        if (in1 || in2) union++;
      }
      const iou = inter / Math.max(union, 1);
      const coverMin = inter / Math.max(area1, area2, 1);

      // 双工具：时间重叠且区域相交
      if (ob - oa > 0.05 && iou > 0.02)
        warns.push({kind: 'two_tools', severity: 'warn',
          message: `${r1.name || '区域'} 与 ${r2.name || '区域'} 在 ${oa.toFixed(1)}–${ob.toFixed(1)}s 需同时移动两件工具`,
          steps: [A.s.id, B.s.id], time: [round2(oa), round2(ob)]});

      // 互相抵消：一遮一加、高度重叠、能量近似平衡
      const types = new Set([A.s.type, B.s.type]);
      if (types.has('dodge') && types.has('burn') && coverMin > CANCEL_RATIO) {
        const d1 = (A.b - A.a) / baseExp, d2 = (B.b - B.a) / baseExp;
        const balance = Math.min(d1, d2) / Math.max(d1, d2);
        if (balance > 0.5)
          warns.push({kind: 'cancel', severity: 'warn',
            message: `${r1.name || '区域'} 的遮挡与 ${r2.name || '区域'} 的加光在同一区域互相抵消（覆盖 ${(coverMin*100)|0}%，能量差 ${(Math.abs(d1-d2)/Math.max(d1,d2)*100)|0}%），请取舍`,
            steps: [A.s.id, B.s.id], bbox: box});
      }
    }
  }
  return warns;
}

function dilateBand(cov, W, H, band) {
  // 先标记再按 EDGELEN 方形膨胀
  const seed = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) if (cov[i] > 0.03) seed[i] = 1;
  // 行列方向各做一次 max-filter（可分离，半径 EDGELEN）
  const tmp = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let cnt = 0;
    for (let x = 0; x < W + EDGELEN; x++) {
      if (x < W && seed[row + x]) cnt++;
      if (x - 2 * EDGELEN - 1 >= 0 && seed[row + x - 2 * EDGELEN - 1]) cnt--;
      const xr = x - EDGELEN;
      if (xr >= 0 && xr < W && cnt) tmp[row + xr] = 1;
    }
  }
  for (let x = 0; x < W; x++) {
    let cnt = 0;
    for (let y = 0; y < H + EDGELEN; y++) {
      if (y < H && tmp[y * W + x]) cnt++;
      if (y - 2 * EDGELEN - 1 >= 0 && tmp[(y - 2 * EDGELEN - 1) * W + x]) cnt--;
      const yr = y - EDGELEN;
      if (yr >= 0 && yr < H && cnt) band[yr * W + x] = 1;
    }
  }
}

function addBBox(b, i, W, H) {
  const x = (i % W) / W, y = (((i / W) | 0)) / H;
  if (!b) return [x, y, x, y];
  if (x < b[0]) b[0] = x;
  if (y < b[1]) b[1] = y;
  if (x > b[2]) b[2] = x;
  if (y > b[3]) b[3] = y;
  return b;
}

// ------------------------------------------------------------------ 等值线

const LEVELS = [32, 64, 96, 128, 160, 192, 224];
export function extractContours(gray, W, H) {
  const at = (x, y) => gray[y * W + x];
  const segs = [];
  for (const level of LEVELS) {
    for (let y = 0; y < H - 1; y++) {
      for (let x = 0; x < W - 1; x++) {
        const i = y * W + x;
        let v = 0;
        if (at(x, y) > level) v |= 1;
        if (at(x + 1, y) > level) v |= 2;
        if (at(x + 1, y + 1) > level) v |= 4;
        if (at(x, y + 1) > level) v |= 8;
        if (!v || v === 15) continue;
        const tl = at(x, y), tr = at(x + 1, y),
              br = at(x + 1, y + 1), bl = at(x, y + 1);
        const ep = name => {
          if (name === 't') return [x + it(tl, tr, level), y];
          if (name === 'r') return [x + 1, y + it(tr, br, level)];
          if (name === 'b') return [x + it(bl, br, level), y + 1];
          return [x, y + it(tl, bl, level)];
        };
        for (const [e0, e1] of MARCH[v]) {
          const [x0, y0] = ep(e0), [x1, y1] = ep(e1);
          segs.push(x0 / W, y0 / H, x1 / W, y1 / H, level);
        }
      }
    }
  }
  return segs;
}
function it(a, b, lv) { return clamp((lv - a) / (b === a ? 1 : b - a), 0, 1); }

// ------------------------------------------------------------------ utils

function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
function round2(v) { return Math.round(v * 100) / 100; }
function interp1(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}
