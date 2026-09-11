// maskboard.js —— 与 app/maskboard.py 同构的浏览器端几何：
// 校准拟合、缩放/反算、自交/窄桥/手柄/越界检测、拼版与偏差比较。
// 目标轮廓（尤其画笔区域）由服务端 /mask-regions 栅格提取，
// 前端只处理已矢量化的毫米多边形，保证拖动时即时重算。

export const TOOL_VERSION = 'maskboard-1.0';
export const HANDLE_W = 18, HANDLE_L = 60;
export const ERROR_TOL_MM = 3.0;

// ------------------------------------------------------------------ 校准

export function fitCalibration(points, discDiameter) {
  const D = +discDiameter;
  const rows = points
    .map(p => ({h: +p.h, pd: p.pd === '' || p.pd == null ? null : +p.pd,
                band: p.band === '' || p.band == null ? null : +p.band}))
    .filter(p => p.h >= 0.5 && p.pd !== null);
  const out = {disc_diameter: D, n: rows.length, calibrated: false,
               L: null, beta: null, w0: 0, gamma: 0,
               scale_rms: null, proj_rms: null, band_rms: null,
               h_min: null, h_max: null, residuals: []};
  if (rows.length < 2 || D <= 0) return out;
  const hs = rows.map(r => r.h);
  if (new Set(hs).size < 2) return out;

  // s = 1 + beta*h 最小二乘
  const n = rows.length;
  const sx = hs.reduce((a, b) => a + b, 0);
  const sxx = hs.reduce((a, b) => a + b * b, 0);
  const sv = rows.reduce((a, r) => a + r.pd / D, 0);
  const sxv = rows.reduce((a, r, i) => a + hs[i] * (r.pd / D), 0);
  const det = n * sxx - sx * sx;
  let c0 = det ? (sv * sxx - sxv * sx) / det : 1;
  let c1 = det ? (n * sxv - sx * sv) / det : 0;
  const beta = Math.max(0, c1);
  const sPred = hs.map(h => 1 + beta * h);
  const scaleRms = Math.sqrt(mean(rows.map((r, i) => (sPred[i] - r.pd / D) ** 2)));

  let w0 = 0, gamma = 0, bandRms = null;
  const br = rows.filter(r => r.band !== null);
  if (br.length >= 2 && new Set(br.map(r => r.h)).size >= 2) {
    const bx = br.map(r => r.h), bv = br.map(r => r.band);
    const bsx = bx.reduce((a, b) => a + b, 0);
    const bsxx = bx.reduce((a, b) => a + b * b, 0);
    const bsv = bv.reduce((a, b) => a + b, 0);
    const bsxv = bx.reduce((a, h, i) => a + h * bv[i], 0);
    const bd = br.length * bsxx - bsx * bsx;
    w0 = Math.max(0, bd ? (bsv * bsxx - bsxv * bsx) / bd : 0);
    gamma = Math.max(0, bd ? (br.length * bsxv - bsx * bsv) / bd : 0);
    bandRms = Math.sqrt(mean(br.map(r => (w0 + gamma * r.h - r.band) ** 2)));
  }

  out.calibrated = true;
  out.beta = beta;
  out.L = beta > 1e-9 ? 1 / beta : null;
  out.w0 = w0; out.gamma = gamma;
  out.scale_rms = +scaleRms.toFixed(6);
  out.proj_rms = +(scaleRms * D).toFixed(4);
  out.band_rms = bandRms === null ? null : +bandRms.toFixed(4);
  out.h_min = Math.min(...hs);
  out.h_max = Math.max(...hs);
  out.residuals = rows.map((r, i) => ({
    h: r.h, pd: r.pd, pd_pred: sPred[i] * D, scale: r.pd / D,
    band: r.band,
    band_pred: r.band === null ? null : w0 + gamma * r.h}));
  return out;
}

export function predict(fit, h) {
  if (!fit || !fit.calibrated) return null;
  h = +h;
  const scale = 1 + fit.beta * h;
  const rms = fit.scale_rms || 0;
  const span = Math.max(1, fit.h_max - fit.h_min);
  let margin = 0;
  if (h < fit.h_min) margin = 0.25 * rms * (fit.h_min - h) / span;
  else if (h > fit.h_max) margin = 0.25 * rms * (h - fit.h_max) / span;
  return {
    h, scale, band: fit.w0 + fit.gamma * h,
    scale_err: Math.max(rms + margin, 1e-4),
    band_err: fit.band_rms || 0,
    in_range: h >= fit.h_min - 1e-6 && h <= fit.h_max + 1e-6};
}

// ------------------------------------------------------------------ 几何

const xy = p => [p.x, p.y];

export function signedArea(loop) {
  let s = 0;
  for (let i = 0; i < loop.length; i++) {
    const [x0, y0] = xy(loop[i]);
    const [x1, y1] = xy(loop[(i + 1) % loop.length]);
    s += x0 * y1 - x1 * y0;
  }
  return s / 2;
}

export function centroid(loop) {
  let a2 = 0, cx = 0, cy = 0;
  for (let i = 0; i < loop.length; i++) {
    const [x0, y0] = xy(loop[i]);
    const [x1, y1] = xy(loop[(i + 1) % loop.length]);
    const cr = x0 * y1 - x1 * y0;
    a2 += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr;
  }
  if (Math.abs(a2) < 1e-9) {
    return {x: avg(loop.map(p => p.x)), y: avg(loop.map(p => p.y))};
  }
  return {x: cx / (3 * a2), y: cy / (3 * a2)};
}

function avg(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }

export function scaleLoop(loop, s, c) {
  c = c || centroid(loop);
  return loop.map(p => ({x: c.x + (p.x - c.x) * s, y: c.y + (p.y - c.y) * s}));
}

export function bboxOf(loops) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const lp of loops) for (const p of lp) {
    x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y);
  }
  if (!isFinite(x0)) return null;
  return {x: x0, y: y0, w: x1 - x0, h: y1 - y0};
}

function ccw(a, b, c) {
  return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]);
}

export function segIntersect(p1, p2, p3, p4) {
  const [a, b, c, d] = [p1, p2, p3, p4].map(xy);
  if (Math.max(Math.min(a[0], b[0]), Math.min(c[0], d[0])) >
      Math.min(Math.max(a[0], b[0]), Math.max(c[0], d[0]))) return false;
  if (Math.max(Math.min(a[1], b[1]), Math.min(c[1], d[1])) >
      Math.min(Math.max(a[1], b[1]), Math.max(c[1], d[1]))) return false;
  return ccw(a, c, d) !== ccw(b, c, d) && ccw(a, b, c) !== ccw(a, b, d);
}

export function pointSegDist(p, a, b) {
  const [P, A, B] = [p, a, b].map(xy);
  const dx = B[0] - A[0], dy = B[1] - A[1];
  const den = dx * dx + dy * dy;
  if (den < 1e-12) return Math.hypot(P[0] - A[0], P[1] - A[1]);
  const t = Math.max(0, Math.min(1, ((P[0] - A[0]) * dx + (P[1] - A[1]) * dy) / den));
  return Math.hypot(P[0] - (A[0] + t * dx), P[1] - (A[1] + t * dy));
}

// 颈宽：相交返回 null；否则 {dist, score}。score 综合平行度与沿向重叠，
// 正对平行墙接近 1；尖角/端点贴近趋于 0。
function segBridge(p1, p2, p3, p4) {
  if (segIntersect(p1, p2, p3, p4)) return null;
  const foot = (p, a, b) => {
    const dx = b.x - a.x, dy = b.y - a.y;
    const den = dx * dx + dy * dy;
    if (den < 1e-12) return [0, {x: a.x, y: a.y}];
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / den));
    return [t, {x: a.x + t * dx, y: a.y + t * dy}];
  };
  const cands = [];
  for (const [p, a, b] of [[p1, p3, p4], [p2, p3, p4], [p3, p1, p2], [p4, p1, p2]]) {
    const [t, q] = foot(p, a, b);
    cands.push({dist: Math.hypot(p.x - q.x, p.y - q.y), t});
  }
  const m = cands.reduce((u, v) => v.dist < u.dist ? v : u);
  if (m.dist < 1e-9) return null;
  const ux = p2.x - p1.x, uy = p2.y - p1.y;
  const vx = p4.x - p3.x, vy = p4.y - p3.y;
  const lu = Math.hypot(ux, uy), lv = Math.hypot(vx, vy);
  const par = Math.abs(ux * vx + uy * vy) / Math.max(lu * lv, 1e-12);
  let overlap = 0;
  if (lu > 1e-9) {
    const u0 = (ux * (p3.x - p1.x) + uy * (p3.y - p1.y)) / (lu * lu);
    const u1 = (ux * (p4.x - p1.x) + uy * (p4.y - p1.y)) / (lu * lu);
    overlap = Math.max(0, Math.min(1, Math.max(u0, u1)) - Math.max(0, Math.min(u0, u1)));
  }
  const score = par * Math.max(overlap, 0.2 * Math.min(m.t, 1 - m.t) / 0.2);
  return {dist: m.dist, par: score};
}

export function pointInPolygon(p, poly) {
  const [x, y] = xy(p);
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = xy(poly[i]), [xj, yj] = xy(poly[j]);
    if ((yi > y) !== (yj > y)) {
      const xx = xi + (y - yi) * (xj - xi) / (yj - yi);
      if (xx > x) inside = !inside;
    }
  }
  return inside;
}

export function selfIntersections(loop) {
  const hits = [], n = loop.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (j === i + 1 || (i === 0 && j === n - 1)) continue;
      if (segIntersect(loop[i], loop[(i + 1) % n],
                       loop[j], loop[(j + 1) % n])) hits.push([i, j]);
    }
  }
  return hits;
}

// ------------------------------------------------------------------ 反算

function offsetRadial(loop, c, d) {
  return loop.map(p => {
    const r = Math.hypot(p.x - c.x, p.y - c.y);
    if (r < 1e-9) return {...p};
    const k = (r + d) / r;
    return {x: c.x + (p.x - c.x) * k, y: c.y + (p.y - c.y) * k};
  });
}

export function backCalc(target, pred, featherComp = 0.35) {
  const s = pred.scale, band = pred.band;
  const c = centroid(target.outer);
  const grow = featherComp * band / s;
  const boardOuter = offsetRadial(scaleLoop(target.outer, 1 / s, c), c, grow);
  const boardHoles = (target.holes || []).map(h => {
    const hc = centroid(h);
    return offsetRadial(scaleLoop(h, 1 / s, hc), hc, -grow);
  });
  const projected = {
    outer: scaleLoop(boardOuter, s, c),
    holes: boardHoles.map(h => scaleLoop(h, s, centroid(h)))};
  const e = Math.max(ERROR_TOL_MM,
    2 * Math.abs(pred.scale_err / s) *
      Math.max(...projected.outer.map(p => Math.max(Math.abs(p.x - c.x), Math.abs(p.y - c.y)))) +
    Math.abs(pred.band_err) * 0.5);
  return {
    board: {outer: boardOuter, holes: boardHoles},
    projected, error_mm: e, error_max_mm: e,
    feather_grow_mm: grow, scale: s,
    bbox: bboxOf([boardOuter, ...boardHoles]), center: c};
}

/** 用户已手改轮廓后，按当前 spec 几何直接投影与检测。 */
export function calcFromSpec(spec, pred) {
  const board = {outer: spec.outer, holes: spec.holes || []};
  const s = pred.scale;
  const projected = {
    outer: scaleLoop(board.outer, s),
    holes: board.holes.map(h => scaleLoop(h, s))};
  const e = Math.max(ERROR_TOL_MM, Math.abs(pred.scale_err / s) * 20);
  return {board, projected, error_mm: e, error_max_mm: e, scale: s,
          bbox: bboxOf([board.outer, ...board.holes]),
          feather_grow_mm: 0};
}

// ------------------------------------------------------------------ 手柄

export function handlePolygon(anchor, dir, length = HANDLE_L, width = HANDLE_W) {
  const a = dir * Math.PI / 180;
  const ux = Math.cos(a), uy = Math.sin(a);
  const px = -uy, py = ux, hw = width / 2;
  const pts = [];
  for (const [sx, sy] of [[0, -1], [0, 1], [1, 1], [1, -1]]) {
    pts.push({x: anchor.x + ux * length * sx + px * hw * sy,
              y: anchor.y + uy * length * sx + py * hw * sy});
  }
  return pts;
}

export function handleAnchorSuggest(outer, dir) {
  const a = dir * Math.PI / 180;
  const ux = Math.cos(a), uy = Math.sin(a);
  const c = centroid(outer);
  let best = outer[0], bestD = -Infinity;
  for (let i = 0; i < outer.length; i++) {
    const p0 = outer[i], p1 = outer[(i + 1) % outer.length];
    const mx = (p0.x + p1.x) / 2, my = (p0.y + p1.y) / 2;
    const d = (mx - c.x) * ux + (my - c.y) * uy;
    if (d > bestD) { bestD = d; best = {x: mx, y: my}; }
  }
  return best;
}

// ------------------------------------------------------------------ 检测

function narrowBridges(outer, holes, minBridge) {
  const out = [];
  function pairCheck(A, B, label) {
    const same = A === B;
    for (let i = 0; i < A.length; i++) {
      const jStart = same ? i + 2 : 0;
      const jEnd = same ? A.length - 1 : B.length;
      for (let j = jStart; j < jEnd; j++) {
        if (same && i === 0 && j === A.length - 1) continue;
        const d = segBridge(A[i], A[(i + 1) % A.length],
                            B[j], B[(j + 1) % B.length]);
        if (d && d.dist > 0 && d.dist < minBridge && d.par >= 0.6) {
          out.push({kind: 'narrow_bridge', severity: 'error',
            message: `${label}最小料宽 ${d.dist.toFixed(1)}mm < ${minBridge}mm`,
            at: [{x: (A[i].x + B[j].x) / 2, y: (A[i].y + B[j].y) / 2}]});
          if (out.length >= 6) return;
        }
      }
      if (out.length >= 6) return;
    }
  }
  pairCheck(outer, outer, '外轮廓');
  holes.forEach((h, k) => pairCheck(h, h, `孔洞${k + 1}`));
  holes.forEach((h, k) => pairCheck(outer, h, `孔洞${k + 1}到外边`));
  for (let k = 0; k < holes.length; k++)
    for (let m = k + 1; m < holes.length; m++)
      pairCheck(holes[k], holes[m], `孔洞${k + 1}/${m + 1}之间`);
  return out;
}

function handleOverTarget(handle, projected) {
  // 沿手柄中线采样，忽略接入点附近 12% 的接缝；过半点落入投影目标内才算
  const a = handle.dir * Math.PI / 180;
  const ux = Math.cos(a), uy = Math.sin(a);
  const N = 11;
  let inside = 0;
  const hits = [];
  for (let k = 0; k < N; k++) {
    const t = (k + 0.5) / N;
    if (t < 0.12) continue;
    const p = {x: handle.ax + ux * handle.length * t,
               y: handle.ay + uy * handle.length * t};
    if (pointInPolygon(p, projected.outer) &&
        !projected.holes.some(h => pointInPolygon(p, h))) {
      inside++; hits.push(p);
    }
  }
  return inside >= 5 ? hits : [];
}

export function validate(spec, calc, target, fit, paperW, paperH) {
  const issues = [];
  const h = +spec.height;
  if (!fit || !fit.calibrated) {
    issues.push({kind: 'uncalibrated', severity: 'error',
                 message: '尚未完成投影校准（至少两个高度）'});
  } else {
    if (h < fit.h_min || h > fit.h_max)
      issues.push({kind: 'height_out_of_range', severity: 'error',
        message: `高度 ${h}mm 超出校准范围 ${fit.h_min}–${fit.h_max}mm，缩放/羽化为外推值`});
    if ((fit.proj_rms || 0) > 1.5)
      issues.push({kind: 'calibration_rms', severity: 'warn',
        message: `校准投影残差 ${fit.proj_rms.toFixed(2)}mm 偏大，建议复核测量`});
  }
  const {outer, holes} = calc.board;
  for (const [which, loop] of [['外轮廓', outer],
                               ...holes.map((hp, i) => [`孔洞${i + 1}`, hp])]) {
    const hits = selfIntersections(loop);
    if (hits.length)
      issues.push({kind: 'self_intersect', severity: 'error',
        message: `${which}自交（${hits.length} 处），切割路径不可用`,
        edges: hits.slice(0, 8), which});
  }
  issues.push(...narrowBridges(outer, holes, +spec.min_bridge));

  if (spec.handle) {
    // 手柄（板坐标）先按投影中心缩放到相纸坐标，再对目标区判定侵入
    const cb = centroid(calc.board.outer);
    const s = calc.scale;
    const hpHandle = {
      ...spec.handle,
      ax: cb.x + (spec.handle.ax - cb.x) * s,
      ay: cb.y + (spec.handle.ay - cb.y) * s,
      length: spec.handle.length * s, width: spec.handle.width * s};
    const hit = handleOverTarget(hpHandle, target);
    if (hit.length)
      issues.push({kind: 'handle_over_target', severity: 'error',
        message: '手柄投影压在目标区域内，会误挡光线，请改朝向/接入点',
        at: hit.slice(0, 4)});
  }

  const sw = +spec.sheet_w, sh = +spec.sheet_h;
  const allLoops = [outer, ...holes];
  if (spec.handle) allLoops.push(handlePolygon(
    {x: spec.handle.ax, y: spec.handle.ay}, spec.handle.dir,
    spec.handle.length, spec.handle.width));
  const bb = bboxOf(allLoops);
  if (bb && (bb.x < 0 || bb.y < 0 || bb.x + bb.w > sw + 1e-6 ||
             bb.y + bb.h > sh + 1e-6))
    issues.push({kind: 'sheet_overflow', severity: 'error',
      message: `轮廓+手柄 ${bb.w.toFixed(0)}×${bb.h.toFixed(0)}mm 超出板材 ${sw}×${sh}mm`,
      bbox: bb});

  const tb = bboxOf([target.outer]);
  if (tb && (tb.x < 0 || tb.y < 0 || tb.x + tb.w > paperW + 1e-6 ||
             tb.y + tb.h > paperH + 1e-6))
    issues.push({kind: 'target_off_paper', severity: 'warn',
                 message: '目标区域超出相纸范围'});
  return issues;
}

// ------------------------------------------------------------------ 拼版

export function nestTools(items, sheetW, sheetH, gap = 8, margin = 10) {
  items = [...items].sort((a, b) =>
    b.bbox.h * b.bbox.w - a.bbox.h * a.bbox.w);
  const placements = {};
  let x = margin, y = margin, rowH = 0, fits = true;
  for (const t of items) {
    let {w, h} = t.bbox, rot = false;
    if (t.rotate_ok !== false && w > h && w > sheetW - 2 * margin) {
      [w, h] = [h, w]; rot = true;
    }
    if (x + w > sheetW - margin && x > margin) {
      x = margin; y += rowH + gap; rowH = 0;
    }
    if (x + w > sheetW - margin || y + h > sheetH - margin) fits = false;
    placements[t.id] = {tool_id: t.id, x, y, rot: rot ? 90 : 0,
                        ox: t.bbox.x, oy: t.bbox.y, w, h};
    x += w + gap;
    rowH = Math.max(rowH, h);
  }
  return {placements, fits, sheets: [{w: sheetW, h: sheetH}]};
}

// ------------------------------------------------------------------ 比较

function pointLoopHausdorff(A, B) {
  function directed(P, Q) {
    let hi = 0, sum = 0;
    for (const v of P) {
      let dm = Infinity;
      for (let i = 0; i < Q.length; i++)
        dm = Math.min(dm, pointSegDist(v, Q[i], Q[(i + 1) % Q.length]));
      hi = Math.max(hi, dm); sum += dm;
    }
    return [hi, sum / P.length];
  }
  const ca = centroid(A), cb = centroid(B);
  const Ashift = A.map(p => ({x: p.x - ca.x + cb.x, y: p.y - ca.y + cb.y}));
  const [h1, m1] = directed(Ashift, B);
  const [h2, m2] = directed(B, Ashift);
  return {hausdorff: +Math.max(h1, h2).toFixed(3), mean: +((m1 + m2) / 2).toFixed(3)};
}

export function compareVersions(a, b) {
  const out = {
    height_delta: b.spec.height - a.spec.height,
    scale_delta: +(b.result.scale - a.result.scale).toFixed(5)};
  if (a.result.board?.outer?.length && b.result.board?.outer?.length) {
    const d = pointLoopHausdorff(a.result.board.outer, b.result.board.outer);
    out.board_hausdorff_mm = d.hausdorff;
    out.board_mean_mm = d.mean;
  }
  if (a.result.projected?.outer?.length && b.result.projected?.outer?.length) {
    const d = pointLoopHausdorff(a.result.projected.outer, b.result.projected.outer);
    out.proj_hausdorff_mm = d.hausdorff;
    out.proj_mean_mm = d.mean;
  }
  if (a.calibration_snapshot?.beta != null && b.calibration_snapshot?.beta != null)
    out.beta_delta = b.calibration_snapshot.beta - a.calibration_snapshot.beta;
  out.error_band_a_mm = a.result.error_mm;
  out.error_band_b_mm = b.result.error_mm;
  return out;
}
