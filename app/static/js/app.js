// app.js —— 工作台主控
import {compute, fitCurve, defaultParams, intervalsOf, regionSignature} from './engine.js';

const GRID = 480;                 // 计算栅格最长边
const PALETTE = ['#ff8c1a', '#50aaff', '#b4ff50', '#ff5ac8', '#b478ff', '#3cdcc8'];

export class App {
  constructor(pid) {
    this.pid = pid;
    this.plan = {regions: [], steps: [], base_exposure: 10};
    this.cal = {params: defaultParams(2), calibrated: false};
    this.meta = {};
    this.selRegion = null;
    this.selStep = null;
    this.tool = 'brush';
    this.viewMode = 'final';
    this.showContour = true;
    this.showMask = true;
    this.drawing = null;         // 当前笔画
    this.draft = null;           // 多边形草稿点
    this.scrubT = null;
    this.tmax = 30;
    this.maskCache = {};
    this.res = null;
    this.baseGray = null;
    this.dirty = false;

    this.$ = id => document.getElementById(id);
    this.bindEvents();
    this.init();
  }

  async init() {
    const p = await (await fetch(`/api/projects/${this.pid}`)).json();
    this.meta = p;
    this.plan = Object.assign({regions: [], steps: [], base_exposure: p.base_exposure},
                              p.plan || {});
    this.plan.base_exposure = +this.plan.base_exposure || p.base_exposure;
    this.cal = p.calibration && p.calibration.params ? p.calibration
               : {params: defaultParams(p.paper_grade), calibrated: false};
    this.$('proj-name').value = p.name;
    this.$('f-mag').value = p.magnification;
    this.$('f-apt').value = p.aperture;
    this.$('f-base').value = this.plan.base_exposure;
    this.$('f-grade').value = p.paper_grade;
    this.$('f-invert').value = p.negative_invert;
    this.calPoints = this.cal.points || [];
    this.syncCalTable();
    this.drawCurve();
    this.updateCalStatus();

    await this.loadImage(p.image_path, +p.negative_invert);
    this.layoutCanvases();
    new ResizeObserver(() => this.layoutCanvases()).observe(this.$('stage'));
    this.renderRegions();
    this.recompute(true);
    this.loadVersions();
    setInterval(() => this.autosave(), 1500);
    window.addEventListener('beforeunload', () => this.autosave(true));
  }

  // ------------------------------------------------------------ 图像载入
  async loadImage(src, invert) {
    const img = new Image();
    img.src = src;
    await img.decode();
    this.imgW0 = img.naturalWidth;
    this.imgH0 = img.naturalHeight;
    const s = GRID / Math.max(img.naturalWidth, img.naturalHeight);
    this.W = Math.max(2, Math.round(img.naturalWidth * s));
    this.H = Math.max(2, Math.round(img.naturalHeight * s));

    const c = document.createElement('canvas');
    c.width = this.W; c.height = this.H;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, this.W, this.H);
    const lum = ctx.getImageData(0, 0, this.W, this.H).data;
    this.baseGray = new Float32Array(this.W * this.H);
    for (let i = 0; i < this.W * this.H; i++) {
      const L = lum[i * 4] * 0.299 + lum[i * 4 + 1] * 0.587 + lum[i * 4 + 2] * 0.114;
      const T = invert ? L : 255 - L;
      this.baseGray[i] = 245 - (T / 255) * (245 - 12);
    }
    for (const id of ['cv-image', 'cv-mask', 'cv-contour', 'cv-draw', 'crosshair']) {
      const cv = this.$(id);
      cv.width = this.W; cv.height = this.H;
    }
  }

  layoutCanvases() {
    const stage = this.$('stage');
    const rw = stage.clientWidth, rh = stage.clientHeight;
    if (!this.W) return;
    const ar = this.W / this.H;
    let w = rw, h = w / ar;
    if (h > rh) { h = rh; w = h * ar; }
    for (const id of ['cv-image', 'cv-mask', 'cv-contour', 'cv-draw', 'crosshair']) {
      const cv = this.$(id);
      cv.style.width = w + 'px';
      cv.style.height = h + 'px';
      cv.style.left = (rw - w) / 2 + 'px';
      cv.style.top = (rh - h) / 2 + 'px';
    }
    this.contentRect = {w, h};
    this.drawOverlay();
  }

  toNorm(e) {
    const cv = this.$('cv-image');
    const rect = cv.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top) / rect.height,
    };
  }

  // ------------------------------------------------------------ 事件
  bindEvents() {
    // 顶栏
    this.$('btn-save').onclick = () => this.save(true);
    this.$('btn-version').onclick = () => this.saveVersion();
    this.$('btn-versions').onclick = () => this.loadVersions(true);
    this.$('proj-name').onchange = e => this.mark();

    // 放大参数
    this.$('f-mag').onchange = () => this.mark();
    this.$('f-apt').onchange = () => this.mark();
    this.$('f-base').onchange = e => this.applyBase(+e.target.value || 10);
    this.$('f-grade').onchange = e => {
      if (!this.cal.calibrated) this.cal.params = defaultParams(+e.target.value);
      this.recompute(); this.mark();
    };
    this.$('f-invert').onchange = async e => {
      await this.loadImage(this.meta.image_path, +e.target.value);
      this.layoutCanvases();
      this.recompute(true); this.mark();
    };

    // 校准
    this.$('cal-add').onclick = () => {
      const last = this.calPoints[this.calPoints.length - 1];
      this.calPoints.push({t: last ? last.t * 2 : 4, gray: 180});
      this.syncCalTable(); this.mark();
    };
    this.$('cal-fit').onclick = () => this.doFit();
    this.$('cal-clear').onclick = () => {
      this.calPoints = []; this.cal = {params: defaultParams(+this.$('f-grade').value),
        calibrated: false, points: []};
      this.syncCalTable(); this.drawCurve(); this.updateCalStatus();
      this.recompute(); this.mark();
    };
    this.$('eyedrop').onchange = e => {
      this.$('crosshair').style.cursor = e.target.checked ? 'crosshair' : '';
    };

    // 工具
    this.$('tool-brush').onclick = () => this.setTool('brush');
    this.$('tool-poly').onclick = () => this.setTool('poly');
    this.$('tool-size').oninput = e => {
      const r = this.region(this.selRegion);
      if (r) { r.size = +e.target.value; this.invalidate(r.id); this.scheduleDraw(); this.recompute(); this.mark(); }
    };
    this.$('tool-feather').oninput = e => {
      const r = this.region(this.selRegion);
      if (r) { r.feather = +e.target.value; this.invalidate(r.id); this.scheduleDraw(); this.recompute(); this.mark(); }
    };

    // 视图
    this.$('view-mode').onclick = e => {
      if (e.target.dataset.mode) {
        this.viewMode = e.target.dataset.mode;
        this.$('view-mode').querySelectorAll('button').forEach(b =>
          b.classList.toggle('on', b.dataset.mode === this.viewMode));
        this.scheduleDraw();
      }
    };
    this.$('show-contour').onchange = e => { this.showContour = e.target.checked; this.scheduleDraw(); };
    this.$('show-mask').onchange = e => { this.showMask = e.target.checked; this.scheduleDraw(); };

    // 舞台指针
    const stage = this.$('stage');
    const cross = this.$('crosshair');
    cross.style.pointerEvents = 'none';
    const drawLayer = this.$('cv-draw');
    drawLayer.style.pointerEvents = 'none';
    // 事件挂在 stage 上，通过目标判断是否点在控制条
    stage.addEventListener('pointerdown', e => this.onPointerDown(e));
    stage.addEventListener('pointermove', e => this.onPointerMove(e));
    stage.addEventListener('pointerup', e => this.onPointerUp(e));
    stage.addEventListener('dblclick', e => this.onDblClick(e));
    stage.addEventListener('keydown', e => {
      if (e.key === 'Escape') { this.draft = null; this.clearScrub(); this.scheduleDraw(); }
      if (e.key === 'Enter' && this.draft && this.draft.points.length >= 3) this.commitPolygon();
    });
    stage.tabIndex = 0;

    // 创建步骤
    this.$('create-dodge').onclick = () => this.createStep('dodge');
    this.$('create-burn').onclick = () => this.createStep('burn');

    // 时间轴
    this.$('timeline').addEventListener('pointerdown', e => this.tlDown(e));
    this.$('timeline').addEventListener('pointermove', e => this.tlMove(e));
    this.$('timeline').addEventListener('pointerup', () => this.tlUp());
    this.$('timeline').addEventListener('dblclick', () => this.clearScrub());
  }

  setTool(t) {
    this.tool = t;
    this.$('tool-brush').classList.toggle('on', t === 'brush');
    this.$('tool-poly').classList.toggle('on', t === 'poly');
  }

  // ------------------------------------------------------------ 舞台交互
  onPointerDown(e) {
    if (e.target.closest('.seg') || e.target.id === 'show-contour' ||
        e.target.id === 'show-mask' || e.target.tagName === 'LABEL') return;
    const p = this.toNorm(e);
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return;
    this.$('stage').focus();

    if (this.$('eyedrop').checked) {
      const g = Math.round(this.sampleGray(p));
      const last = this.calPoints[this.calPoints.length - 1];
      this.calPoints.push({t: last ? +(last.t * 1.5).toFixed(1) : 4, gray: g});
      this.syncCalTable();
      this.toast(`已取灰度 ${g}`);
      return;
    }

    if (this.tool === 'brush') {
      let r = this.region(this.selRegion);
      if (!r || r.kind !== 'brush') {
        r = this.addRegion('brush', `画笔区域 ${this.plan.regions.length + 1}`);
      }
      r.strokes = r.strokes || [];
      this.drawing = {region: r, points: [p]};
      r.strokes.push({points: this.drawing.points});
      this.scheduleDraw();
    } else {
      if (!this.draft) this.draft = {points: []};
      this.draft.points.push(p);
      this.scheduleDraw();
    }
  }

  onPointerMove(e) {
    const p = this.toNorm(e);
    // 光标圆圈
    this.drawCrosshair(p);
    if (this.drawing) {
      const pts = this.drawing.points;
      const last = pts[pts.length - 1];
      const sizeN = (+this.$('tool-size').value) /
        Math.max(this.imgW0, this.imgH0);
      if (Math.hypot(p.x - last.x, p.y - last.y) > sizeN * 0.12) {
        pts.push(p);
        this.scheduleDraw();
      }
    }
  }

  onPointerUp() {
    if (this.drawing) {
      this.invalidate(this.drawing.region.id);
      this.drawing = null;
      this.renderRegions();
      this.recompute();
      this.mark();
    }
  }

  onDblClick(e) {
    if (this.tool === 'poly' && this.draft && this.draft.points.length >= 3) {
      this.commitPolygon();
    }
  }

  commitPolygon() {
    const r = this.addRegion('polygon', `多边形区域 ${this.plan.regions.length + 1}`);
    r.points = this.draft.points;
    this.draft = null;
    this.invalidate(r.id);
    this.renderRegions();
    this.recompute();
    this.mark();
  }

  sampleGray(p) {
    const x = Math.min(this.W - 1, Math.max(0, Math.round(p.x * this.W)));
    const y = Math.min(this.H - 1, Math.max(0, Math.round(p.y * this.H)));
    return this.baseGray[y * this.W + x];
  }

  drawCrosshair(p) {
    const cv = this.$('crosshair');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, this.W, this.H);
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1 || this.$('eyedrop').checked) return;
    const sx = (+this.$('tool-size').value) * this.W / this.imgW0;
    const sy = (+this.$('tool-size').value) * this.H / this.imgH0;
    const rx = sx / 2, ry = sy / 2;
    const x = p.x * this.W, y = p.y * this.H;
    ctx.strokeStyle = 'rgba(255,255,255,.8)';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    if (this.tool === 'poly' && this.draft) {
      const a = this.draft.points[this.draft.points.length - 1];
      ctx.beginPath(); ctx.moveTo(a.x * this.W, a.y * this.H); ctx.lineTo(x, y);
      ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.stroke();
    }
  }

  // ------------------------------------------------------------ 区域/步骤
  addRegion(kind, name) {
    const id = 'r' + (Math.max(0, ...this.plan.regions.map(x => +x.id.slice(1))) + 1);
    const r = {
      id, name, kind,
      size: +this.$('tool-size').value,
      feather: +this.$('tool-feather').value,
      strokes: [], points: [],
    };
    this.plan.regions.push(r);
    this.selectRegion(r.id);
    return r;
  }

  region(id) { return this.plan.regions.find(r => r.id === id); }
  step(id) { return this.plan.steps.find(s => s.id === id); }
  invalidate(id) { delete this.maskCache[id]; }

  selectRegion(id) {
    this.selRegion = id;
    const r = this.region(id);
    if (r) {
      this.$('tool-size').value = r.size;
      this.$('tool-feather').value = r.feather;
      this.$('step-create').style.display = '';
    } else {
      this.$('step-create').style.display = 'none';
    }
    this.renderRegions();
    this.renderStepEditor();
    this.scheduleDraw();
  }

  deleteRegion(id) {
    this.plan.regions = this.plan.regions.filter(r => r.id !== id);
    this.plan.steps = this.plan.steps.filter(s => s.region_id !== id);
    delete this.maskCache[id];
    if (this.selRegion === id) this.selectRegion(null);
    else this.renderRegions();
    this.renderTimeline();
    this.recompute();
    this.mark();
  }

  createStep(type) {
    if (!this.region(this.selRegion)) return;
    const base = this.plan.base_exposure;
    const s = {
      id: 's' + (Math.max(0, ...this.plan.steps.map(x => +x.id.slice(1))) + 1),
      type, region_id: this.selRegion,
    };
    if (type === 'dodge') { s.start = 0; s.ratio = 0.4; }
    else { s.start = base; s.stops = 1; }
    this.plan.steps.push(s);
    this.selStep = s.id;
    this.clampSteps();
    this.renderStepEditor();
    this.recompute();
    this.mark();
  }

  selectStep(id) {
    this.selStep = id;
    const s = this.step(id);
    if (s) this.selectRegion(s.region_id);
    this.renderStepEditor();
    this.renderTimeline();
  }

  applyBase(v) {
    this.plan.base_exposure = Math.min(300, Math.max(0.5, v));
    this.$('f-base').value = this.plan.base_exposure;
    this.clampSteps();
    this.recompute();
    this.renderTimeline();
    this.mark();
  }

  clampSteps() {
    const base = this.plan.base_exposure;
    for (const s of this.plan.steps) {
      if (s.type === 'dodge') {
        s.ratio = Math.min(1, Math.max(0, +s.ratio));
        s.start = Math.min(base - base * s.ratio, Math.max(0, +s.start));
      }
    }
  }

  // ------------------------------------------------------------ 计算
  recompute(immediate = false) {
    const run = () => {
      if (!this.baseGray) return;
      this.res = compute(this.baseGray, this.W, this.H, this.plan,
                         this.cal, this.maskCache, this.scrubT);
      this.scheduleDraw();
      this.renderWarnings();
      this.renderTimeline();
    };
    if (immediate) { run(); return; }
    clearTimeout(this._rc);
    this._rc = setTimeout(run, 120);
  }

  scheduleDraw() {
    requestAnimationFrame(() => this.draw());
  }

  // ------------------------------------------------------------ 画布渲染
  draw() {
    if (!this.res) return;
    this.drawImageLayer();
    this.drawOverlay();
    this.drawContours();
    this.drawDraft();
  }

  grayToImageData(arr) {
    const img = new ImageData(this.W, this.H);
    for (let i = 0; i < this.W * this.H; i++) {
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = arr[i];
      img.data[i * 4 + 3] = 255;
    }
    return img;
  }

  drawImageLayer() {
    const cv = this.$('cv-image');
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(this.W, this.H);
    const {gray, masks} = this.res;
    for (let i = 0; i < this.W * this.H; i++) {
      if (this.viewMode === 'delta') {
        const d = gray[i] - this.baseGray[i];
        if (d < -0.5) { img.data[i*4] = 255; img.data[i*4+1] = Math.max(60, 150 + d * 0.8); img.data[i*4+2] = Math.max(30, 70 + d * 0.5); }
        else if (d > 0.5) { img.data[i*4] = 80; img.data[i*4+1] = Math.min(220, 160 + d * 0.4); img.data[i*4+2] = 255; }
        else { img.data[i*4] = img.data[i*4+1] = img.data[i*4+2] = 28; }
      } else if (this.viewMode === 'mask') {
        // 先铺暗色底，叠加覆盖最高的遮罩颜色
        let best = 0, col = null;
        for (const rid in masks) {
          if (masks[rid][i] > best) { best = masks[rid][i]; col = rid; }
        }
        if (col && best > 0.01) {
          const [r, g, b] = this.regionColor(col, true);
          img.data[i*4] = r; img.data[i*4+1] = g; img.data[i*4+2] = b;
        } else {
          img.data[i*4] = img.data[i*4+1] = img.data[i*4+2] = 18;
        }
      } else {
        const v = this.viewMode === 'base' ? this.baseGray[i] : gray[i];
        img.data[i*4] = img.data[i*4+1] = img.data[i*4+2] = v;
      }
      img.data[i*4+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  regionColor(rid, rgb = false) {
    const idx = this.plan.regions.findIndex(r => r.id === rid);
    const hex = PALETTE[Math.max(0, idx) % PALETTE.length];
    if (rgb) return [parseInt(hex.slice(1, 3), 16),
                     parseInt(hex.slice(3, 5), 16),
                     parseInt(hex.slice(5, 7), 16)];
    return hex;
  }

  drawOverlay() {
    const cv = this.$('cv-mask');
    if (!this.W) return;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, this.W, this.H);
    if (!this.showMask || !this.res || this.viewMode === 'mask') return;
    // 一帧内累加所有步骤的覆盖（按选中状态加权），再单次 putImageData
    const ar = new Float32Array(this.W * this.H);
    const ag = new Float32Array(this.W * this.H);
    const ab = new Float32Array(this.W * this.H);
    const aw = new Float32Array(this.W * this.H);
    for (const s of this.plan.steps) {
      const cov = this.res.masks[s.region_id];
      if (!cov) continue;
      const [r, g, b] = this.regionColor(s.region_id, true);
      const sel = this.selStep === s.id || this.selRegion === s.region_id;
      const strength = sel ? 0.6 : 0.3;
      for (let i = 0; i < this.W * this.H; i++) {
        const w = cov[i] * strength;
        ar[i] += r * w; ag[i] += g * w; ab[i] += b * w; aw[i] += w;
      }
    }
    const img = ctx.createImageData(this.W, this.H);
    for (let i = 0; i < this.W * this.H; i++) {
      if (aw[i] < 0.004) continue;
      img.data[i*4]   = ar[i] / aw[i];
      img.data[i*4+1] = ag[i] / aw[i];
      img.data[i*4+2] = ab[i] / aw[i];
      img.data[i*4+3] = Math.min(255, aw[i] * 255);
    }
    ctx.putImageData(img, 0, 0);
  }

  drawContours() {
    const cv = this.$('cv-contour');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, this.W, this.H);
    if (!this.showContour || !this.res) return;
    const seg = this.res.contours;
    ctx.lineWidth = 0.6;
    for (let i = 0; i < seg.length; i += 5) {
      const lv = seg[i + 4];
      ctx.strokeStyle = lv <= 96 ? 'rgba(120,255,160,.75)'
                      : lv >= 192 ? 'rgba(255,230,120,.75)'
                      : 'rgba(140,220,255,.55)';
      ctx.beginPath();
      ctx.moveTo(seg[i] * this.W, seg[i+1] * this.H);
      ctx.lineTo(seg[i+2] * this.W, seg[i+3] * this.H);
      ctx.stroke();
    }
  }

  drawDraft() {
    const cv = this.$('cv-draw');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, this.W, this.H);
    // 选中区域的轮廓提示
    const r = this.region(this.selRegion);
    if (r && this.res) {
      const cov = this.res.masks[r.id];
      if (cov) {
        const img = ctx.createImageData(this.W, this.H);
        for (let i = 0; i < this.W * this.H; i++) {
          if (cov[i] > 0.04 && cov[i] < 0.96) {
            img.data[i*4] = 255; img.data[i*4+1] = 255;
            img.data[i*4+2] = 255; img.data[i*4+3] = 110;
          }
        }
        ctx.putImageData(img, 0, 0);
      }
    }
    // 多边形草稿
    if (this.draft) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2;
      ctx.beginPath();
      this.draft.points.forEach((p, i) => i ? ctx.lineTo(p.x*this.W, p.y*this.H)
                                            : ctx.moveTo(p.x*this.W, p.y*this.H));
      ctx.stroke();
      for (const p of this.draft.points) {
        ctx.fillStyle = '#ff8c1a';
        ctx.beginPath(); ctx.arc(p.x*this.W, p.y*this.H, 3, 0, 7); ctx.fill();
      }
    }
  }

  // ------------------------------------------------------------ 侧栏列表
  renderRegions() {
    const box = this.$('region-list');
    if (!this.plan.regions.length) {
      box.innerHTML = '<p class="muted">在图上用画笔涂抹或画多边形来圈定区域。</p>';
      return;
    }
    box.innerHTML = '';
    this.plan.regions.forEach(r => {
      const div = document.createElement('div');
      div.className = 'item' + (r.id === this.selRegion ? ' sel' : '');
      div.innerHTML = `
        <div class="row" style="margin:0">
          <span class="dot" style="background:${this.regionColor(r.id)}"></span>
          <input class="rname" value="${r.name.replace(/"/g, '&quot;')}"
                 style="flex:1;min-width:0">
          <span class="tag">${r.kind === 'brush' ? '画笔' : '多边形'}</span>
        </div>
        <div class="tools">
          <button class="rsel">选择</button>
          <button class="rreset">清除笔迹</button>
          <button class="rdel danger">删除</button>
        </div>`;
      div.querySelector('.rname').onchange = e => { r.name = e.target.value; this.mark(); };
      div.querySelector('.rsel').onclick = () => this.selectRegion(r.id);
      div.querySelector('.rreset').onclick = () => {
        r.strokes = []; r.points = []; this.invalidate(r.id);
        this.plan.steps = this.plan.steps.filter(s => s.region_id !== r.id);
        this.recompute(); this.renderStepEditor(); this.mark();
      };
      div.querySelector('.rdel').onclick = () => this.deleteRegion(r.id);
      div.onclick = e => { if (e.target.tagName !== 'INPUT' &&
                            !e.target.closest('button')) this.selectRegion(r.id); };
      box.appendChild(div);
    });
  }

  renderStepEditor() {
    const box = this.$('step-editor');
    const s = this.step(this.selStep);
    if (!s) {
      box.innerHTML = '<span class="muted">在时间轴上点选一个遮挡/加光步骤进行编辑；' +
        '或先在图上画好区域，再点下方按钮生成步骤。</span>';
      return;
    }
    const base = this.plan.base_exposure;
    if (s.type === 'dodge') {
      const dur = base * s.ratio;
      box.innerHTML = `
        <div class="kv">类型<span style="color:var(--dodge)">遮挡（dodge）</span></div>
        <div class="kv">区域<select class="e-region"></select></div>
        <div class="kv">遮挡比例
          <input type="range" class="e-ratio" min="0" max="1" step="0.01" value="${s.ratio}">
        </div>
        <div class="kv">开始(s)
          <input type="number" class="e-start" step="0.1" min="0" max="${(base-dur).toFixed(1)}" value="${s.start}">
        </div>
        <div class="muted">在 ${(+s.start).toFixed(1)}s–${(+s.start+dur).toFixed(1)}s 挡住光线，
          等效减光约 ${(Math.log2(1/(1-s.ratio))).toFixed(2)} 档</div>`;
    } else {
      const extra = base * (2 ** s.stops - 1);
      box.innerHTML = `
        <div class="kv">类型<span style="color:var(--burn)">加光（burn）</span></div>
        <div class="kv">区域<select class="e-region"></select></div>
        <div class="kv">加光档数
          <input type="range" class="e-stops" min="0" max="3" step="0.05" value="${s.stops}">
        </div>
        <div class="kv">开始(s)
          <input type="number" class="e-start" step="0.1" value="${s.start}">
        </div>
        <div class="muted">${(+s.start).toFixed(1)}s 起追加 ${extra.toFixed(1)}s
          （共到 ${(+s.start+extra).toFixed(1)}s）</div>`;
    }
    const sel = box.querySelector('.e-region');
    for (const r of this.plan.regions) {
      const o = document.createElement('option');
      o.value = r.id; o.textContent = r.name;
      if (r.id === s.region_id) o.selected = true;
      sel.appendChild(o);
    }
    sel.onchange = () => {
      s.region_id = sel.value; this.selectRegion(s.region_id);
      this.recompute(); this.mark();
    };
    const ratio = box.querySelector('.e-ratio');
    if (ratio) ratio.oninput = () => {
      s.ratio = +ratio.value; this.clampSteps();
      this.renderTimeline(); this.recompute(); this.renderStepEditor(); this.mark();
    };
    const stops = box.querySelector('.e-stops');
    if (stops) stops.oninput = () => {
      s.stops = +stops.value;
      this.renderTimeline(); this.recompute(); this.renderStepEditor(); this.mark();
    };
    box.querySelector('.e-start').onchange = e => {
      s.start = Math.max(0, +e.target.value || 0);
      this.clampSteps();
      this.renderTimeline(); this.recompute(); this.renderStepEditor(); this.mark();
    };
    const del = document.createElement('button');
    del.className = 'danger'; del.textContent = '删除本步骤';
    del.style.marginTop = '6px';
    del.onclick = () => {
      this.plan.steps = this.plan.steps.filter(x => x.id !== s.id);
      this.selStep = null;
      this.renderStepEditor(); this.renderTimeline(); this.recompute(); this.mark();
    };
    box.appendChild(del);
  }

  // ------------------------------------------------------------ 问题检测
  renderWarnings() {
    const box = this.$('warn-list');
    const ws = this.res ? this.res.warnings : [];
    this.$('tl-summary').textContent = ws.length ? `· ${ws.length} 条提示` : '· 无问题';
    if (!ws.length) { box.innerHTML = '<span class="muted">未发现问题。</span>'; return; }
    box.innerHTML = '';
    ws.forEach((w, idx) => {
      const d = document.createElement('div');
      d.className = 'warn-item ' + (w.severity === 'error' ? 'error' : '');
      d.innerHTML = `<span class="tag ${w.severity}">${
        w.kind.startsWith('out_of_range') ? '超范围' :
        w.kind === 'edge_jump' ? '边缘跳变' :
        w.kind === 'cancel' ? '互相抵消' : '双工具'}</span> ${w.message}`;
      d.onclick = () => this.locateWarning(w);
      box.appendChild(d);
    });
  }

  locateWarning(w) {
    if (w.bbox) {
      const f = this.$('flash');
      const cr = this.contentRect;
      const [x0, y0, x1, y1] = w.bbox;
      f.style.left = (cr.w * x0 + (this.$('stage').clientWidth - cr.w) / 2) + 'px';
      f.style.top = (cr.h * y0 + (this.$('stage').clientHeight - cr.h) / 2) + 'px';
      f.style.width = Math.max(2, cr.w * (x1 - x0)) + 'px';
      f.style.height = Math.max(2, cr.h * (y1 - y0)) + 'px';
      f.style.display = 'block';
      f.classList.toggle('edge', w.severity !== 'error');
      clearTimeout(this._ft);
      this._ft = setTimeout(() => f.style.display = 'none', 2200);
    }
    if (w.steps && w.steps.length) this.selectStep(w.steps[0]);
    if (w.time) {
      this.setScrub(w.time[1]);
      this.renderTimeline();
      document.querySelectorAll('.tl-block').forEach(b => {
        if (w.steps.includes(b.dataset.sid)) {
          b.classList.add('flashing');
          setTimeout(() => b.classList.remove('flashing'), 1800);
        }
      });
    }
  }

  // ------------------------------------------------------------ 时间轴
  tX(t) {
    const lane = this.$('lane-base');
    return (t / this.tmax) * lane.clientWidth;
  }
  tFromX(px) {
    return px / this.$('lane-base').clientWidth * this.tmax;
  }

  renderTimeline() {
    if (!this.res) return;
    const base = this.plan.base_exposure;
    this.tmax = Math.max(base * 2.2, this.res.timelineEnd * 1.12, 15);
    // 基础块
    const lb = this.$('lane-base');
    lb.innerHTML = '';
    const bb = document.createElement('div');
    bb.className = 'tl-block tl-base';
    bb.style.left = '0px';
    bb.style.background = 'rgba(216,162,74,.5)';
    bb.style.width = this.tX(base) + 'px';
    bb.style.color = '#f3e3c0';
    bb.innerHTML = `基础曝光 ${base.toFixed(1)}s<span class="handle"></span>`;
    bb.dataset.kind = 'base';
    lb.appendChild(bb);

    const mkLane = (id, type) => {
      const lane = this.$(id);
      lane.innerHTML = '';
      for (const s of this.plan.steps.filter(x => x.type === type)) {
        const iv = intervalsOf({...this.plan, steps: [s]})[0];
        const reg = this.region(s.region_id);
        const el = document.createElement('div');
        el.className = `tl-block ${type}` + (s.id === this.selStep ? ' sel' : '');
        el.style.left = this.tX(iv.a) + 'px';
        el.style.width = Math.max(4, this.tX(iv.b) - this.tX(iv.a)) + 'px';
        el.dataset.sid = s.id;
        el.textContent = `${reg ? reg.name : '?'} ${
          type === 'dodge' ? '遮挡 ' + Math.round(s.ratio*100) + '%'
                           : '+' + s.stops + '档'}`;
        const h = document.createElement('span');
        h.className = 'handle';
        el.appendChild(h);
        lane.appendChild(el);
      }
    };
    mkLane('lane-dodge', 'dodge');
    mkLane('lane-burn', 'burn');

    // 刻度
    const ticks = this.$('tl-ticks');
    ticks.innerHTML = '';
    const step = this.tickStep();
    for (let t = 0; t <= this.tmax + 1e-6; t += step) {
      const el = document.createElement('div');
      el.className = 'tick';
      el.style.left = this.tX(t) + 'px';
      el.textContent = t.toFixed(step < 1 ? 1 : 0) + 's';
      ticks.appendChild(el);
    }
    this.$('base-end-label').textContent = base.toFixed(1);
    this.updateScrubCursor();
  }

  tickStep() {
    const target = this.tmax / 9;
    const pow = 10 ** Math.floor(Math.log10(target));
    for (const m of [1, 2, 2.5, 5, 10])
      if (m * pow >= target) return m * pow;
    return 10 * pow;
  }

  tlDown(e) {
    const block = e.target.closest('.tl-block');
    const rect = this.$('lane-base').getBoundingClientRect();
    const x0 = e.clientX - rect.left;
    if (block) {
      if (block.dataset.kind === 'base') {
        const onHandle = e.target.classList.contains('handle');
        this._drag = {
          kind: onHandle ? 'base-resize' : 'scrub-base',
          x0, startBase: this.plan.base_exposure,
        };
        if (!onHandle) this.setScrub(this.tFromX(x0));
      } else {
        const sid = block.dataset.sid;
        this.selectStep(sid);
        this._drag = {
          kind: e.target.classList.contains('handle') ? 'resize' : 'move',
          sid, x0, startStart: this.step(sid).start,
          startRatio: this.step(sid).ratio, startStops: this.step(sid).stops,
        };
      }
    } else if (x0 >= 0 && x0 <= rect.width) {
      this._drag = {kind: 'scrub', x0};
      this.setScrub(this.tFromX(x0));
      this.renderTimeline();
    }
  }

  tlMove(e) {
    if (!this._drag) return;
    const rect = this.$('lane-base').getBoundingClientRect();
    const x = e.clientX - rect.left;
    const d = this._drag;
    const dt = this.tFromX(x - d.x0);
    const base = this.plan.base_exposure;

    if (d.kind === 'scrub' || d.kind === 'scrub-base') {
      this.setScrub(Math.max(0, this.tFromX(x)));
      if (d.kind === 'scrub') this.renderTimeline();
      return;
    }
    if (d.kind === 'base-resize') {
      const nb = Math.min(300, Math.max(0.5, d.startBase + dt));
      this.$('f-base').value = nb.toFixed(1);
      this.plan.base_exposure = nb;
      this.clampSteps();
      this.renderTimeline();
      this.recompute();
      return;
    }

    const s = this.step(d.sid);
    if (d.kind === 'move') {
      if (s.type === 'dodge') {
        s.start = Math.min(base - base * s.ratio, Math.max(0, d.startStart + dt));
      } else {
        s.start = Math.max(0, d.startStart + dt);
      }
    } else if (d.kind === 'resize') {
      if (s.type === 'dodge') {
        // 拖右缘：新的遮挡时长 = 起始 + 鼠标相对位移
        const newDur = Math.max(0.2, base * d.startRatio + dt);
        s.ratio = Math.min(1, newDur / base);
      } else {
        const newExtra = Math.max(0, base * (2 ** d.startStops - 1) + dt);
        s.stops = Math.min(4, Math.max(0, Math.log2(1 + newExtra / base)));
      }
    }
    this.renderTimeline();
    this.recompute();
  }

  tlUp() {
    if (this._drag && (this._drag.kind === 'move' || this._drag.kind === 'resize')) {
      this.renderStepEditor();
      this.mark();
    }
    if (this._drag && this._drag.kind === 'base-resize') {
      this.$('f-base').value = this.plan.base_exposure;
      this.mark();
    }
    this._drag = null;
  }

  setScrub(t) {
    this.scrubT = Math.max(0, t);
    this.$('cursor-t').textContent = `秒表 t = ${this.scrubT.toFixed(1)}s（双击时间轴回到结束）`;
    this.recompute(true);
    this.updateScrubCursor();
  }

  clearScrub() {
    if (this.scrubT === null) return;
    this.scrubT = null;
    this.$('cursor-t').textContent = '';
    this.$('tl-cursor').style.display = 'none';
    this.recompute(true);
    this.renderTimeline();
  }

  updateScrubCursor() {
    const cur = this.$('tl-cursor');
    if (this.scrubT === null) { cur.style.display = 'none'; return; }
    cur.style.display = '';
    cur.style.left = this.tX(this.scrubT) + 'px';
  }

  // ------------------------------------------------------------ 校准曲线
  syncCalTable() {
    const tb = this.$('cal-table').querySelector('tbody');
    tb.innerHTML = '';
    this.calPoints.forEach((pt, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td><input type="number" step="0.1" min="0" value="${pt.t}"></td>
        <td><input type="number" step="1" min="0" max="255" value="${pt.gray}"></td>
        <td><button class="danger">×</button></td>`;
      const [tI, gI] = tr.querySelectorAll('input');
      tI.onchange = () => { pt.t = +tI.value || 0; this.drawCurve(); this.mark(); };
      gI.onchange = () => { pt.gray = Math.min(255, Math.max(0, +gI.value)); this.drawCurve(); this.mark(); };
      tr.querySelector('button').onclick = () => {
        this.calPoints.splice(i, 1); this.syncCalTable(); this.drawCurve(); this.mark();
      };
      tb.appendChild(tr);
    });
  }

  doFit() {
    const base = this.plan.base_exposure;
    const r = fitCurve(this.calPoints, +this.$('f-grade').value, base);
    this.cal = {
      params: r.params, points: this.calPoints, fitted: r.fitted,
      range: r.range, rms: r.rms, n: r.n,
      calibrated: r.n >= 2 && r.rms !== null,
    };
    this.drawCurve();
    this.updateCalStatus();
    this.recompute();
    this.mark();
    this.toast(r.calibrated ? `曲线已拟合，残差 RMS=${r.rms.toFixed(2)} 灰阶` : '数据点不足');
  }

  updateCalStatus() {
    const el = this.$('cal-status');
    const info = this.$('cal-info');
    if (this.cal.calibrated) {
      el.textContent = `已校准 · RMS ${this.cal.rms.toFixed(2)}`;
      el.style.color = 'var(--ok)';
      const p = this.cal.params;
      info.innerHTML = `Dmin=${p.Dmin.toFixed(2)} Dmax=${p.Dmax.toFixed(2)} ` +
        `x0=${p.x0.toFixed(2)} k=${p.k.toFixed(2)}<br>` +
        `<span class="muted">实测响应区间 ${(this.plan.base_exposure*10**this.cal.range[0]).toFixed(1)}–` +
        `${(this.plan.base_exposure*10**this.cal.range[1]).toFixed(1)}s（相对基础曝光）</span>`;
    } else {
      el.textContent = '未校准（按反差号默认曲线）';
      el.style.color = 'var(--warn)';
      info.textContent = '至少录入 2 个阶梯点后拟合；未校准时按反差号使用典型曲线，范围提示仅供参考。';
    }
  }

  drawCurve() {
    const cv = this.$('curve-canvas');
    const W = cv.width, H = cv.height;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    const base = this.plan.base_exposure;
    const tLo = base * 0.1, tHi = base * 32;
    const xOf = t => (Math.log10(t / tLo)) / Math.log10(tHi / tLo) * (W - 50) + 44;
    const yOf = g => H - 34 - g / 255 * (H - 60);

    // 网格
    ctx.strokeStyle = '#333845'; ctx.fillStyle = '#8a8f9c';
    ctx.font = '11px sans-serif'; ctx.lineWidth = 1;
    for (const g of [0, 64, 128, 192, 255]) {
      ctx.beginPath(); ctx.moveTo(44, yOf(g)); ctx.lineTo(W - 8, yOf(g)); ctx.stroke();
      ctx.fillText(String(g), 8, yOf(g) + 4);
    }
    let tt = tLo;
    while (tt <= tHi * 1.001) {
      ctx.beginPath();
      ctx.moveTo(xOf(tt), 20); ctx.lineTo(xOf(tt), H - 34); ctx.stroke();
      ctx.fillText(tt >= 10 ? tt.toFixed(0) : tt.toFixed(1), xOf(tt) - 8, H - 18);
      tt *= 2;
    }
    ctx.save();
    ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('相纸灰度（0=黑）', -60, 0);
    ctx.restore();
    ctx.fillText('曝光秒数（对数轴）', W / 2 - 40, H - 4);

    // 校准范围阴影
    if (this.cal.calibrated && this.cal.range) {
      ctx.fillStyle = 'rgba(111,208,138,.08)';
      ctx.fillRect(xOf(base * 10 ** this.cal.range[0]), 20,
        xOf(base * 10 ** this.cal.range[1]) - xOf(base * 10 ** this.cal.range[0]),
        H - 54);
    }

    // 曲线
    ctx.strokeStyle = '#d8a24a'; ctx.lineWidth = 2;
    ctx.beginPath();
    const useFitted = this.cal.fitted && this.cal.fitted.length;
    if (useFitted) {
      this.cal.fitted.forEach((q, i) => {
        const t = base * 10 ** q.x;
        if (t < tLo || t > tHi) return;
        const X = xOf(t), Y = yOf(q.gray);
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      });
    } else {
      // 默认曲线采样
      const p = this.cal.params;
      for (let i = 0; i <= 200; i++) {
        const t = tLo * (tHi / tLo) ** (i / 200);
        const x = Math.log10(t / base);
        const D = p.Dmin + (p.Dmax - p.Dmin) /
          (1 + 10 ** (-(x - p.x0) * p.k));
        const g = 255 * 10 ** -D;
        const X = xOf(t), Y = yOf(g);
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      }
    }
    ctx.stroke();

    // 实测点
    ctx.fillStyle = '#5fd08a';
    for (const pt of this.calPoints) {
      ctx.beginPath(); ctx.arc(xOf(pt.t), yOf(pt.gray), 4, 0, 7); ctx.fill();
    }
    // 基础曝光标记
    ctx.strokeStyle = '#7fb2ff'; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(xOf(base), 20); ctx.lineTo(xOf(base), H - 34); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#7fb2ff';
    ctx.fillText('基础', xOf(base) - 26, 14);
  }

  // ------------------------------------------------------------ 持久化
  dirtySet() { this.dirty = true; }
  mark() { this.dirty = true; }

  async autosave(force = false) {
    if (!this.dirty && !force) return;
    await this.save(false);
  }

  async save(manual) {
    if (!this.meta.id) return;
    const payload = {
      name: this.$('proj-name').value,
      magnification: +this.$('f-mag').value || 1,
      aperture: this.$('f-apt').value,
      base_exposure: +this.$('f-base').value,
      paper_grade: +this.$('f-grade').value,
      negative_invert: +this.$('f-invert').value,
      calibration: this.cal,
      plan: this.plan,
    };
    const r = await fetch(`/api/projects/${this.pid}`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (r.ok) {
      this.dirty = false;
      this.meta = {...this.meta, ...payload};
      if (manual) this.toast('已保存');
    }
  }

  async saveVersion() {
    const label = prompt('版本标签（如：试条第 2 轮）', `版本 ${new Date().toLocaleString()}`);
    if (!label) return;
    const note = prompt('备注（可留空）', '') || '';
    await this.save(false);
    await fetch(`/api/projects/${this.pid}/versions`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({label, note}),
    });
    this.toast('版本已保存');
    this.loadVersions();
  }

  async loadVersions(open = false) {
    const vs = await (await fetch(`/api/projects/${this.pid}/versions`)).json();
    const box = this.$('version-list');
    if (!vs.length) { box.innerHTML = '<span class="muted">暂无保存的版本。</span>'; return; }
    box.innerHTML = '';
    vs.forEach(v => {
      const d = document.createElement('div');
      d.className = 'item';
      d.innerHTML = `<div class="name">${v.label}</div>
        <div class="muted" style="font-size:11px">${v.created_at}${v.note ? ' · ' + v.note : ''}</div>
        <div class="tools"><button class="vrestore">回滚</button>
        <button class="vdel danger">删除</button></div>`;
      d.querySelector('.vrestore').onclick = async () => {
        if (!confirm(`回滚到「${v.label}」？当前未存版本的改动会丢失。`)) return;
        await fetch(`/api/versions/${v.id}/restore`, {method: 'POST'});
        location.reload();
      };
      d.querySelector('.vdel').onclick = async () => {
        if (!confirm('删除该版本？')) return;
        await fetch(`/api/versions/${v.id}`, {method: 'DELETE'});
        this.loadVersions();
      };
      box.appendChild(d);
    });
    if (open) this.toast(`${vs.length} 个版本`);
  }

  toast(msg) {
    const t = this.$('toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(this._tt);
    this._tt = setTimeout(() => t.classList.remove('show'), 1800);
  }
}
