// maskboard_editor.js —— 实体遮挡板制作台主控
import {
  TOOL_VERSION, HANDLE_L, HANDLE_W, fitCalibration, predict,
  backCalc, calcFromSpec, validate, centroid, bboxOf,
  handlePolygon, handleAnchorSuggest, scaleLoop, nestTools,
  compareVersions, pointInPolygon,
} from './maskboard.js';

const COL = {
  target: 'rgba(120,255,160,.9)',
  targetFill: 'rgba(120,255,160,.10)',
  project: 'rgba(255,194,77,.95)',
  err: 'rgba(255,140,26,.55)',
  board: '#ff5c5c',
  handle: '#4d9fff',
  sheet: 'rgba(255,255,255,.10)',
  paper: 'rgba(120,160,255,.10)',
};

export class MaskBoardApp {
  constructor(pid) {
    this.pid = pid;
    const init = JSON.parse(document.getElementById('mb-init').textContent);
    this.calibrations = (init.calibrations || []).map(c => ({
      ...c, fit: typeof c.fit === 'string' ? JSON.parse(c.fit || '{}') : c.fit}));
    this.toolVersion = init.tool_version || TOOL_VERSION;
    this.regions = [];
    this.paperMm = {w: 200, h: 250};
    this.scaleSource = 'magnification';
    this.pxPerMm = {x: 1, y: 1};
    this.tools = [];
    this.selTool = null;
    this.selLoop = null;        // 'outer' | 'hole:<i>' | 'handle'
    this.selVertex = null;      // {loop, i}
    this.drag = null;
    this.addHoleMode = false;
    this.draftHole = [];
    this.layers = {target: true, project: true, board: true, handle: true};
    this.fit = null;            // 当前正在编辑的校准拟合（未落库也可预览）
    this.points = [];           // 校准表行 {h,pd,band}
    this.disc = 20;
    this.activeCalId = null;
    this.$ = id => document.getElementById(id);
    this.bind();
    this.init();
  }

  async init() {
    const [rmeta, rtools] = await Promise.all([
      fetch(`/api/projects/${this.pid}/mask-regions`).then(r => r.json()),
      fetch(`/api/projects/${this.pid}/mask-tools`).then(r => r.json()),
    ]);
    this.paperMm = {w: rmeta.paper_mm[0], h: rmeta.paper_mm[1]};
    this.scaleSource = rmeta.scale_source || 'magnification';
    this.pxPerMm = {x: rmeta.px_per_mm?.[0] || 1, y: rmeta.px_per_mm?.[1] || 1};
    if (Array.isArray(rmeta.print_size)) {
      this.$('print-w').value = rmeta.print_size[0] || 0;
      this.$('print-h').value = rmeta.print_size[1] || 0;
    }
    if (Array.isArray(rmeta.sheet_mm)) {
      this.$('sh-w').value = rmeta.sheet_mm[0];
      this.$('sh-h').value = rmeta.sheet_mm[1];
    }
    this.regions = rmeta.regions;
    // 初始数据岛只带校准列表，补拉最新列表
    await this.loadCalibrations();
    this.tools = rtools.tools.map(t => ({...t,
      spec: typeof t.spec === 'string' ? JSON.parse(t.spec) : t.spec,
      result: typeof t.result === 'string' ? JSON.parse(t.result) : t.result}));
    this.updatePaperInfo();
    this.renderRegionSelect();
    this.renderCalSelect();
    this.renderCalList();
    this.renderToolList();
    this.layoutCanvas();
    new ResizeObserver(() => this.layoutCanvas()).observe(this.$('mb-stage'));
    this.draw();
    setInterval(() => this.autosave(), 4000);
  }

  // ------------------------------------------------------------ 基础 UI
  bind() {
    this.$('sh-w').onchange = () => this.sheetChange();
    this.$('sh-h').onchange = () => this.sheetChange();
    this.$('print-w').onchange = () => this.printSizeChange();
    this.$('print-h').onchange = () => this.printSizeChange();

    this.$('cal-disc').oninput = e => { this.disc = +e.target.value || 0; this.runFit(); };
    this.$('cal-add').onclick = () => {
      const last = this.points[this.points.length - 1];
      this.points.push({h: last ? +last.h + 20 : 20,
                        pd: last ? +last.pd * 1.2 : this.disc, band: ''});
      this.renderCalTable(); this.runFit();
    };
    this.$('cal-fit').onclick = () => this.runFit(true);
    this.$('cal-save').onclick = () => this.saveCalibration();
    this.$('cal-select').onchange = e => this.loadCalibration(+e.target.value);

    this.$('tool-create').onclick = () => this.createTool();
    this.$('mode-hole').onclick = () => this.toggleHoleMode();
    this.$('del-vertex').onclick = () => this.deleteSelectedVertex();
    this.$('del-hole').onclick = () => this.deleteSelectedHole();
    this.$('recalc-btn').onclick = () => this.recalc();

    for (const btn of this.$('mb-layers').querySelectorAll('button'))
      btn.onclick = () => {
        this.layers[btn.dataset.layer] = !this.layers[btn.dataset.layer];
        btn.classList.toggle('on', this.layers[btn.dataset.layer]);
        this.draw();
      };

    const stage = this.$('mb-stage');
    stage.addEventListener('pointerdown', e => this.onDown(e));
    stage.addEventListener('pointermove', e => this.onMove(e));
    window.addEventListener('pointerup', () => this.onUp());
    stage.addEventListener('dblclick', () => this.finishHoleDraft());
    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') { this.draftHole = []; this.addHoleMode = false;
        this.$('mode-hole').classList.remove('on'); this.draw(); }
      if (e.key === 'Enter' && this.draftHole.length >= 3) this.finishHoleDraft();
      if ((e.key === 'Delete' || e.key === 'Backspace') && this.selVertex)
        this.deleteSelectedVertex();
    });

    this.$('mb-save').onclick = () => { this.save(true); };
    this.$('tv-save').onclick = () => this.saveVersion();
    this.$('tv-compare').onclick = () => this.compareVersionsDialog();
  }

  toast(msg) {
    const t = this.$('toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(this._tt);
    this._tt = setTimeout(() => t.classList.remove('show'), 2000);
  }

  async loadCalibrations(selectId) {
    const j = await fetch('/api/mask-calibrations').then(r => r.json());
    this.calibrations = j.calibrations;
    this.renderCalSelect();
    this.renderCalList();
    if (selectId != null) {
      this.$('cal-select').value = selectId;
      const full = await fetch(`/api/mask-calibrations/${selectId}`).then(r => r.json());
      this.disc = full.disc_diameter;
      this.$('cal-disc').value = this.disc;
      this.points = full.points || [];
      this.fit = full.fit;
      this.renderCalTable();
      this.updateCalInfo();
    }
  }

  renderCalList() {
    const box = this.$('cal-list');
    if (!box) return;
    if (!this.calibrations.length) {
      box.innerHTML = '<p class="muted" style="font-size:11px;margin:4px 0">尚无已保存校准。</p>';
      return;
    }
    box.innerHTML = '';
    for (const c of this.calibrations) {
      const f = c.fit || {};
      const div = document.createElement('div');
      div.className = 'item' + (c.id === this.activeCalId ? ' sel' : '');
      div.innerHTML =
        `<div class="row" style="margin:0"><span class="name" style="font-size:12px">${esc(c.name)}</span></div>
         <div class="muted" style="font-size:11px">⌀${c.disc_diameter}mm ·
           ${f.calibrated ? 'β=' + (+f.beta).toFixed(5) +
             `，RMS ${(+f.proj_rms).toFixed(2)}mm，${f.h_min}–${f.h_max}mm`
             : '未拟合'}</div>
         <div class="tools">
           <button class="c-use">选用</button>
           <button class="c-del danger">删除</button></div>`;
      div.querySelector('.c-use').onclick = () => {
        this.$('cal-select').value = c.id;
        this.loadCalibration(c.id);
        this.renderCalList();
      };
      div.querySelector('.c-del').onclick = async () => {
        if (!confirm(`删除校准「${c.name}」？引用它的遮挡板将转为未绑定。`)) return;
        await fetch(`/api/mask-calibrations/${c.id}`, {method: 'DELETE'});
        if (this.activeCalId === c.id) {
          this.activeCalId = null;
          this.$('cal-select').value = '';
        }
        await this.loadCalibrations(this.activeCalId);
      };
      box.appendChild(div);
    }
  }

  renderRegionSelect() {
    const sel = this.$('region-select');
    sel.innerHTML = '';
    if (!this.regions.length)
      sel.innerHTML = '<option value="">（请先在编排台画区域）</option>';
    for (const r of this.regions) {
      const o = document.createElement('option');
      o.value = r.id;
      o.textContent = `${r.name} · ${r.kind === 'brush' ? '画笔' : '多边形'}` +
        ` · ${(r.area_mm2 / 100).toFixed(1)}cm²`;
      sel.appendChild(o);
    }
  }

  renderCalSelect() {
    const sel = this.$('cal-select');
    const cur = sel.value;
    sel.innerHTML = '<option value="">— 未选择（仅预览）—</option>';
    for (const c of this.calibrations) {
      const o = document.createElement('option');
      o.value = c.id;
      const f = c.fit || {};
      o.textContent = `${c.name}（⌀${c.disc_diameter}mm，${f.calibrated ?
        'β=' + (+f.beta).toFixed(5) : '未拟合'}）`;
      sel.appendChild(o);
    }
    sel.value = this.activeCalId || cur || '';
  }

  renderCalTable() {
    const tb = this.$('cal-table').querySelector('tbody');
    tb.innerHTML = '';
    this.points.forEach((p, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td><input type="number" step="0.1" class="c-h" value="${p.h}"></td>
         <td><input type="number" step="0.1" class="c-pd" value="${p.pd}"></td>
         <td><input type="number" step="0.1" class="c-band" value="${p.band}"></td>
         <td><button class="danger c-del">×</button></td>`;
      tr.querySelector('.c-h').oninput = e => { p.h = +e.target.value; this.runFit(); };
      tr.querySelector('.c-pd').oninput = e => { p.pd = +e.target.value; this.runFit(); };
      tr.querySelector('.c-band').oninput = e => {
        p.band = e.target.value === '' ? '' : +e.target.value; this.runFit(); };
      tr.querySelector('.c-del').onclick = () => {
        this.points.splice(i, 1); this.renderCalTable(); this.runFit(); };
      tb.appendChild(tr);
    });
  }

  async loadCalibration(cid) {
    this.activeCalId = cid || null;
    if (!cid) {
      this.points = []; this.fit = null;
      this.disc = +this.$('cal-disc').value || 20;
    } else {
      // 列表接口不带原始测点，回读详情
      const full = await fetch(`/api/mask-calibrations/${cid}`).then(r => r.json());
      this.disc = full.disc_diameter;
      this.$('cal-disc').value = this.disc;
      this.points = full.points || [];
      this.fit = full.fit;
    }
    this.renderCalTable();
    this.updateCalInfo();
    this.renderCalList();
    this.recomputeCurrent();
  }

  runFit(manual = false) {
    this.fit = fitCalibration(this.points, this.disc);
    this.updateCalInfo();
    this.recomputeCurrent();
    if (manual) this.toast(this.fit.calibrated
      ? `拟合完成：β=${(+this.fit.beta).toFixed(5)}，投影残差 ${this.fit.proj_rms.toFixed(2)}mm`
      : '至少需要两个不同高度的有效测点');
  }

  updateCalInfo() {
    const el = this.$('cal-info');
    const f = this.fit;
    if (!f || !f.calibrated) {
      el.innerHTML = '<span class="muted">至少 2 个高度；投影直径应大于圆片直径。</span>';
      return;
    }
    const h = this.tool() ? +this.tool().spec.height : null;
    let predLine = '';
    if (h !== null) {
      const pr = predict(f, h);
      predLine = `<br>当前 h=${h}mm → s=<strong>${pr.scale.toFixed(4)}</strong>，` +
        `过渡带 ${pr.band.toFixed(2)}mm，${pr.in_range ? '校准内' :
        '<span style="color:var(--err)">外推</span>'}`;
    }
    el.innerHTML =
      `β=${(+f.beta).toFixed(6)} mm⁻¹ · 等效镜头距离 L=${f.L ? f.L.toFixed(0) + 'mm' : '∞'}<br>` +
      `w₀=${f.w0.toFixed(2)}mm · γ=${(+f.gamma).toFixed(5)} · ` +
      `高度范围 ${f.h_min}–${f.h_max}mm<br>` +
      `投影直径 RMS=${f.proj_rms.toFixed(3)}mm` +
      (f.band_rms != null ? ` · 过渡带 RMS=${f.band_rms.toFixed(2)}mm` : '') +
      predLine;
  }

  async saveCalibration() {
    if (!this.fit || !this.fit.calibrated) { this.toast('先拟合到至少两个高度'); return; }
    const name = prompt('校准名称',
      `圆片⌀${this.disc} ${new Date().toLocaleDateString()}`);
    if (!name) return;
    if (this.activeCalId) {
      const r = await fetch(`/api/mask-calibrations/${this.activeCalId}`, {
        method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, disc_diameter: this.disc, points: this.points})});
      if (r.ok) { this.toast('校准已更新'); await this.loadCalibrations(this.activeCalId); }
    } else {
      const r = await fetch('/api/mask-calibrations', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, disc_diameter: this.disc, points: this.points})});
      const j = await r.json();
      if (j.id) { this.activeCalId = j.id; this.toast('校准已保存');
        await this.loadCalibrations(j.id); }
    }
    this.refreshToolCalibration();
  }

  refreshToolCalibration() {
    const t = this.tool();
    if (t && this.activeCalId) {
      t.calibration_id = this.activeCalId;
      this.mark();
    }
    this.recomputeCurrent();
  }

  // ------------------------------------------------------------ 工具
  tool() { return this.tools.find(t => t.id === this.selTool) || null; }

  renderToolList() {
    const box = this.$('tool-list');
    if (!this.tools.length) {
      box.innerHTML = '<p class="muted" style="font-size:11px">尚无遮挡板。</p>';
      return;
    }
    box.innerHTML = '';
    this.tools.forEach(t => {
      const errs = (t.result.issues || []).filter(i => i.severity === 'error').length;
      const div = document.createElement('div');
      div.className = 'item' + (t.id === this.selTool ? ' sel' : '');
      div.innerHTML =
        `<div class="name">${esc(t.name)} ${errs ? `<span class="tag error">${errs} 错</span>` :
          '<span class="tag" style="background:#22522f;color:#9fe0b3">可用</span>'}</div>
         <div class="muted" style="font-size:11px">区域 ${esc(t.spec.region_name || '?')}
           · h=${t.spec.height}mm · s=${(t.result.scale || 1).toFixed(3)}</div>
         <div class="tools">
           <button class="t-sel">编辑</button>
           <button class="t-del danger">删除</button></div>`;
      div.querySelector('.t-sel').onclick = () => this.selectTool(t.id);
      div.querySelector('.t-del').onclick = async () => {
        if (!confirm(`删除遮挡板「${t.name}」？`)) return;
        await fetch(`/api/mask-tools/${t.id}`, {method: 'DELETE'});
        this.tools = this.tools.filter(x => x.id !== t.id);
        if (this.selTool === t.id) { this.selTool = null; this.renderParams(); }
        this.renderToolList(); this.draw(); this.updateExportLink();
      };
      box.appendChild(div);
    });
  }

  async createTool() {
    const rid = this.$('region-select').value;
    if (!rid) { this.toast('请选择区域'); return; }
    if (!this.activeCalId) { this.toast('请先选择已保存的投影校准'); return; }
    const region = this.regions.find(r => r.id === rid);
    if (!region || !region.outer || region.outer.length < 3) {
      this.toast('该区域没有可提取的轮廓'); return;
    }
    const r = await fetch(`/api/projects/${this.pid}/mask-tools`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        region_id: rid, calibration_id: this.activeCalId,
        height: this.fit && this.fit.calibrated ? this.fit.h_min : 10,
        min_bridge: 5,
        sheet_w: +this.$('sh-w').value, sheet_h: +this.$('sh-h').value})});
    const j = await r.json();
    if (!r.ok) { this.toast(j.error || '创建失败'); return; }
    this.tools.push({id: j.id, project_id: this.pid, region_id: rid,
      name: `遮挡板·${region.name}`, calibration_id: this.activeCalId,
      spec: j.spec, result: j.result, tool_version: TOOL_VERSION});
    this.selTool = j.id;
    this.selLoop = 'outer';
    this.renderToolList(); this.renderParams(); this.draw();
    this.updateExportLink();
    this.loadVersions();
    this.mark();
  }

  selectTool(id) {
    this.selTool = id;
    this.selLoop = 'outer';
    this.selVertex = null;
    this.draftHole = [];
    this.renderToolList(); this.renderParams(); this.draw();
    this.loadVersions();
  }

  updatePaperInfo() {
    const src = this.scaleSource === 'print_size' ? '显式打印尺寸' : '扫描像素×放大倍率';
    this.$('paper-info').textContent =
      `打印影像 ${this.paperMm.w.toFixed(1)}×${this.paperMm.h.toFixed(1)}mm · ` +
      `1px≈${(1 / this.pxPerMm.x).toFixed(3)}mm（${src}）。板材可大于影像；原点在板材左下角。`;
  }

  async printSizeChange() {
    const printW = Math.max(0, +this.$('print-w').value || 0);
    const printH = Math.max(0, +this.$('print-h').value || 0);
    const r = await fetch(`/api/projects/${this.pid}/mask-settings`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        paper_w: +this.$('sh-w').value, paper_h: +this.$('sh-h').value,
        print_w: printW, print_h: printH})});
    const j = await r.json();
    this.paperMm = {w: j.paper_mm[0], h: j.paper_mm[1]};
    this.scaleSource = j.scale_source;
    this.pxPerMm = {x: j.px_per_mm[0], y: j.px_per_mm[1]};
    this.updatePaperInfo();
    // 基准变了：目标轮廓必须按新毫米基准重新拉取，再对所有板重算
    await this.reloadRegionsAndTools();
    this.toast('实体尺度基准已更新，轮廓已按新基准重算');
  }

  async reloadRegionsAndTools() {
    const meta = await fetch(`/api/projects/${this.pid}/mask-regions`)
      .then(r => r.json());
    this.regions = meta.regions;
    this.paperMm = {w: meta.paper_mm[0], h: meta.paper_mm[1]};
    this.scaleSource = meta.scale_source;
    this.pxPerMm = {x: meta.px_per_mm[0], y: meta.px_per_mm[1]};
    this.renderRegionSelect();
    // 基准变了：所有板按新基准重新反算
    for (const tt of this.tools) {
      await this.recalcServer(tt, {height: tt.spec.height});
    }
    this.draw();
    this.renderToolList();
    this.renderParams();
  }

  async recalcServer(t, body) {
    const r = await fetch(`/api/mask-tools/${t.id}/recalc`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)});
    if (!r.ok) return;
    const j = await r.json();
    t.spec = j.spec;
    t.result = {...t.result, ...j.result, target: this.regions.find(
      x => x.id === (t.spec.region_id || t.region_id))};
    this.renderIssues();
  }

  sheetChange() {
    const t = this.tool();
    if (t) {
      t.spec.sheet_w = +this.$('sh-w').value;
      t.spec.sheet_h = +this.$('sh-h').value;
      this.recomputeCurrent(); this.mark();
    }
    fetch(`/api/projects/${this.pid}/mask-settings`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({paper_w: +this.$('sh-w').value,
                            paper_h: +this.$('sh-h').value,
                            print_w: +this.$('print-w').value || 0,
                            print_h: +this.$('print-h').value || 0})});
  }

  renderParams() {
    const box = this.$('tool-params');
    const t = this.tool();
    if (!t) { box.innerHTML = '<span class="muted">先创建或选择一块遮挡板。</span>';
      this.$('issue-list').innerHTML = '—'; this.$('issue-summary').textContent = ''; return; }
    const s = t.spec, r = t.result;
    const bb = r.bbox || bboxOf([s.outer]);
    box.innerHTML = `
      <div class="kv">名称<input id="p-name" value="${esc(t.name)}"></div>
      <div class="kv">高度h(mm)<input type="number" id="p-height" step="0.5" value="${s.height}"></div>
      <div class="kv">最小桥宽(mm)<input type="number" id="p-bridge" step="0.5" value="${s.min_bridge}"></div>
      <div class="kv">羽化补偿
        <input type="range" id="p-fcomp" min="0" max="0.8" step="0.01" value="${s.feather_comp ?? 0.35}">
      </div>
      <div class="kv">手柄朝向
        <select id="p-dir">
          ${[0, 45, 90, 135, 180, 225, 270, 315].map(d =>
            `<option value="${d}" ${s.handle?.dir === d ? 'selected' : ''}>${d}°</option>`).join('')}
        </select></div>
      <div class="kv">手柄长(mm)<input type="number" id="p-hlen" step="1" value="${s.handle?.length ?? HANDLE_L}"></div>
      <div class="kv">手柄宽(mm)<input type="number" id="p-hwid" step="0.5" value="${s.handle?.width ?? HANDLE_W}"></div>
      <div class="muted" style="font-size:11px">
        板轮廓包围盒 ${bb ? bb.w.toFixed(1) + '×' + bb.h.toFixed(1) : '?'}mm<br>
        缩放 s=${(r.scale || 1).toFixed(4)} · 羽化外扩 ${(r.feather_grow_mm || 0).toFixed(2)}mm<br>
        误差带宽 ${(r.error_mm || 0).toFixed(2)}mm
        （外点数 ${s.outer?.length || 0}，孔洞 ${s.holes?.length || 0}）
      </div>`;
    this.$('p-name').onchange = e => { t.name = e.target.value; this.mark(); this.renderToolList(); };
    this.$('p-height').onchange = () => {
      s.height = Math.max(0.5, +this.$('p-height').value || 0.5);
      this.recomputeCurrent(); this.updateCalInfo(); this.mark();
    };
    this.$('p-bridge').onchange = () => {
      s.min_bridge = Math.max(0, +this.$('p-bridge').value || 0);
      this.recomputeCurrent(); this.mark();
    };
    this.$('p-fcomp').oninput = e => {
      s.feather_comp = +e.target.value;
      this.recomputeCurrent(); this.mark();
    };
    this.$('p-dir').onchange = e => {
      const dir = +e.target.value;
      const anchor = handleAnchorSuggest(s.outer, dir);
      s.handle = {...s.handle, dir, ax: anchor.x, ay: anchor.y};
      this.recomputeCurrent(); this.mark();
    };
    this.$('p-hlen').onchange = () => {
      s.handle.length = Math.max(5, +this.$('p-hlen').value);
      this.recomputeCurrent(); this.mark();
    };
    this.$('p-hwid').onchange = () => {
      s.handle.width = Math.max(2, +this.$('p-hwid').value);
      this.recomputeCurrent(); this.mark();
    };
    this.renderIssues();
  }

  renderIssues() {
    const t = this.tool();
    const box = this.$('issue-list');
    const sum = this.$('issue-summary');
    if (!t) return;
    const issues = t.result.issues || [];
    const errs = issues.filter(i => i.severity === 'error').length;
    sum.textContent = issues.length ? `· ${errs} 错 / ${issues.length - errs} 警` : '· 全部通过';
    if (!issues.length) { box.innerHTML = '<span style="color:var(--ok)">未发现问题，可导出切割。</span>'; return; }
    box.innerHTML = '';
    issues.forEach(w => {
      const d = document.createElement('div');
      d.className = 'warn-item ' + (w.severity === 'error' ? 'error' : '');
      const label = {
        height_out_of_range: '超校准范围', self_intersect: '轮廓自交',
        narrow_bridge: '窄桥', handle_over_target: '手柄压目标',
        sheet_overflow: '排版越界', uncalibrated: '未校准',
        calibration_rms: '校准残差', target_off_paper: '目标越界'}[w.kind] || w.kind;
      d.innerHTML = `<span class="tag ${w.severity}">${label}</span> ${esc(w.message)}`;
      d.onclick = () => this.locateIssue(w);
      box.appendChild(d);
    });
  }

  locateIssue(w) {
    const t = this.tool();
    if (!t) return;
    if (w.at && w.at.length) {
      // 视口居中到问题点（板坐标系 mm）
      this.focusMm = w.at[0];
      this.draw();
    } else if (w.bbox) {
      this.focusMm = {x: w.bbox.x + w.bbox.w / 2, y: w.bbox.y + w.bbox.h / 2};
      this.draw();
    }
  }

  // ------------------------------------------------------------ 重算
  currentFit() {
    const t = this.tool();
    if (!t) return null;
    // 优先用编辑器里当前拟合；否则用该工具绑定校准的拟合快照
    if (this.activeCalId && this.fit && this.fit.calibrated &&
        t.calibration_id === this.activeCalId) return this.fit;
    const c = this.calibrations.find(x => x.id === t.calibration_id);
    return c ? c.fit : null;
  }

  recomputeCurrent() {
    const t = this.tool();
    if (!t) return;
    const fit = this.currentFit();
    const region = this.regions.find(r => r.id === (t.spec.region_id || t.region_id));
    const pred = predict(fit, t.spec.height) ||
      {scale: 1, band: 0, scale_err: 0, band_err: 0, in_range: false};
    let calc;
    if (t.spec.outer && t.spec.outer.length) {
      calc = calcFromSpec(t.spec, pred);
    } else {
      calc = backCalc(region, pred, t.spec.feather_comp ?? 0.35);
      t.spec.outer = calc.board.outer;
      t.spec.holes = calc.board.holes;
    }
    t.result = {
      ...t.result, target: region, predict: pred,
      board: calc.board, projected: calc.projected,
      error_mm: calc.error_mm, error_max_mm: calc.error_max_mm,
      scale: calc.scale, bbox: calc.bbox,
      feather_grow_mm: calc.feather_grow_mm,
      issues: validate(t.spec, calc, region, fit,
                       this.paperMm.w, this.paperMm.h),
      fit_summary: fit && fit.calibrated
        ? {beta: fit.beta, L: fit.L, proj_rms: fit.proj_rms,
           h_min: fit.h_min, h_max: fit.h_max} : null};
    this.draw();
    this.renderIssues();
    this.renderToolList();
    this.updateCalInfo();
  }

  recalc() {
    const t = this.tool();
    if (!t) return;
    const fit = this.currentFit();
    const region = this.regions.find(r => r.id === (t.spec.region_id || t.region_id));
    const pred = predict(fit, t.spec.height);
    if (!pred) { this.toast('当前高度无有效校准'); return; }
    const calc = backCalc(region, pred, t.spec.feather_comp ?? 0.35);
    t.spec.outer = calc.board.outer;
    t.spec.holes = calc.board.holes;
    const anchor = handleAnchorSuggest(t.spec.outer, t.spec.handle?.dir ?? 0);
    t.spec.handle = {...(t.spec.handle || {}), ax: anchor.x, ay: anchor.y,
      dir: t.spec.handle?.dir ?? 0,
      length: t.spec.handle?.length ?? HANDLE_L,
      width: t.spec.handle?.width ?? HANDLE_W};
    this.recomputeCurrent();
    this.renderParams();
    this.mark();
    this.toast('已按当前高度/羽化重新反算（轮廓手改已重置）');
  }

  toggleHoleMode() {
    if (!this.tool()) return;
    this.addHoleMode = !this.addHoleMode;
    this.draftHole = [];
    this.$('mode-hole').classList.toggle('on', this.addHoleMode);
    this.$('mb-canvas').style.cursor = this.addHoleMode ? 'crosshair' : '';
    this.draw();
  }

  finishHoleDraft() {
    if (!this.addHoleMode || this.draftHole.length < 3) return;
    const t = this.tool();
    t.spec.holes = t.spec.holes || [];
    t.spec.holes.push(this.draftHole.map(p => ({...p})));
    this.draftHole = [];
    this.addHoleMode = false;
    this.$('mode-hole').classList.remove('on');
    this.$('mb-canvas').style.cursor = '';
    this.selLoop = `hole:${t.spec.holes.length - 1}`;
    this.recomputeCurrent(); this.renderParams(); this.mark();
  }

  deleteSelectedVertex() {
    const t = this.tool();
    if (!t || !this.selVertex) return;
    const lp = this.loopOf(t, this.selVertex.loop);
    if (!lp) return;
    if (lp.length <= 3) { this.toast('至少保留 3 个顶点'); return; }
    lp.splice(this.selVertex.i, 1);
    this.selVertex = null;
    this.recomputeCurrent(); this.mark();
  }

  deleteSelectedHole() {
    const t = this.tool();
    if (!t || !this.selLoop || !this.selLoop.startsWith('hole:')) return;
    const k = +this.selLoop.split(':')[1];
    t.spec.holes.splice(k, 1);
    this.selLoop = 'outer';
    this.selVertex = null;
    this.recomputeCurrent(); this.renderParams(); this.mark();
  }

  loopOf(t, which) {
    if (which === 'outer') return t.spec.outer;
    if (which.startsWith('hole:')) return t.spec.holes[+which.split(':')[1]];
    return null;
  }

  // ------------------------------------------------------------ Canvas
  layoutCanvas() {
    const stage = this.$('mb-stage');
    const cv = this.$('mb-canvas');
    const dpr = window.devicePixelRatio || 1;
    this.cssW = stage.clientWidth;
    this.cssH = stage.clientHeight;
    cv.width = this.cssW * dpr;
    cv.height = this.cssH * dpr;
    cv.style.width = this.cssW + 'px';
    cv.style.height = this.cssH + 'px';
    this.dpr = dpr;
    this.draw();
  }

  computeView() {
    const t = this.tool();
    const sw = +(t ? t.spec.sheet_w : this.$('sh-w').value);
    const sh = +(t ? t.spec.sheet_h : this.$('sh-h').value);
    const pad = 40;
    const k = Math.min((this.cssW - pad * 2) / sw,
                       (this.cssH - pad * 2) / sh);
    // 居中板材；毫米坐标 y 向上 → 屏幕 y 向下
    this.view = {
      k,
      ox: (this.cssW - sw * k) / 2,
      oy: (this.cssH + sh * k) / 2,
      sw, sh};
  }
  toScreen(p) { return {x: this.view.ox + p.x * this.view.k,
                         y: this.view.oy - p.y * this.view.k}; }
  toMm(sx, sy) { return {x: (sx - this.view.ox) / this.view.k,
                         y: (this.view.oy - sy) / this.view.k}; }

  draw() {
    if (!this.cssW) return;
    this.computeView();
    const cv = this.$('mb-canvas');
    const ctx = cv.getContext('2d');
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    const v = this.view;

    // 板材
    const p0 = this.toScreen({x: 0, y: 0});
    ctx.fillStyle = COL.sheet;
    ctx.fillRect(p0.x, p0.y - v.sh * v.k, v.sw * v.k, v.sh * v.k);
    ctx.strokeStyle = 'rgba(255,255,255,.5)';
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(p0.x, p0.y - v.sh * v.k, v.sw * v.k, v.sh * v.k);
    ctx.setLineDash([]);
    // 相纸范围（在板材内左下对齐显示参考）
    const pw = Math.min(this.paperMm.w, v.sw), ph = Math.min(this.paperMm.h, v.sh);
    const q0 = this.toScreen({x: 0, y: 0});
    ctx.fillStyle = COL.paper;
    ctx.fillRect(q0.x, q0.y - ph * v.k, pw * v.k, ph * v.k);
    ctx.strokeStyle = 'rgba(120,160,255,.6)';
    ctx.strokeRect(q0.x, q0.y - ph * v.k, pw * v.k, ph * v.k);
    ctx.fillStyle = 'rgba(150,170,220,.8)';
    ctx.font = '10px sans-serif';
    ctx.fillText(`相纸 ${pw.toFixed(0)}×${ph.toFixed(0)}mm`, q0.x + 4, q0.y - 4);
    ctx.fillText(`板材 ${v.sw}×${v.sh}mm（原点 ◧ 左下）`, p0.x + 4,
                 p0.y - v.sh * v.k + 12);

    // 网格（每 10mm）
    ctx.strokeStyle = 'rgba(255,255,255,.05)';
    ctx.beginPath();
    for (let x = 0; x <= v.sw; x += 10) {
      const a = this.toScreen({x, y: 0}), b = this.toScreen({x, y: v.sh});
      ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
    }
    for (let y = 0; y <= v.sh; y += 10) {
      const a = this.toScreen({x: 0, y}), b = this.toScreen({x: v.sw, y});
      ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();

    const t = this.tool();
    if (!t) return;
    const r = t.result;

    const poly = (loop, stroke, fill, width, dash) => {
      if (!loop || loop.length < 2) return;
      ctx.beginPath();
      loop.forEach((p, i) => {
        const q = this.toScreen(p);
        i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y);
      });
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (dash) ctx.setLineDash(dash);
      ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke();
      ctx.setLineDash([]);
    };

    // 目标区域（相纸期望投影）
    if (this.layers.target && r.target) {
      poly(r.target.outer, COL.target, COL.targetFill, 1.4, [5, 3]);
      for (const h of r.target.holes || []) poly(h, COL.target, null, 1, [3, 3]);
    }
    // 预计投影 + 误差带
    if (this.layers.project && r.projected) {
      const c = centroid(r.projected.outer);
      const e = r.error_mm || 0;
      for (const [sgn, col] of [[1, COL.err], [-1, COL.err]]) {
        const band = r.projected.outer.map(p => {
          const rr = Math.hypot(p.x - c.x, p.y - c.y) || 1;
          return {x: c.x + (p.x - c.x) * (rr + sgn * e) / rr,
                  y: c.y + (p.y - c.y) * (rr + sgn * e) / rr};
        });
        poly(band, col, null, 0.8, [2, 2]);
      }
      poly(r.projected.outer, COL.project, null, 1.2, [6, 3]);
    }
    // 切割线
    const issueKinds = new Set((r.issues || []).map(i => i.kind));
    if (this.layers.board) {
      const bad = issueKinds.has('self_intersect');
      poly(t.spec.outer, bad ? '#ff2020' : COL.board, 'rgba(255,92,92,.05)', 1.6);
      t.spec.holes.forEach((h, i) =>
        poly(h, this.selLoop === `hole:${i}` ? '#fff' : COL.board, null, 1.2, [3, 2]));
    }
    // 手柄
    if (this.layers.handle && t.spec.handle) {
      const hs = t.spec.handle;
      const hp = handlePolygon({x: hs.ax, y: hs.ay}, hs.dir, hs.length, hs.width);
      const bad = issueKinds.has('handle_over_target');
      poly(hp, bad ? '#ff2020' : COL.handle, 'rgba(77,159,255,.08)', 1.2);
      // 朝向箭头
      const a = hs.dir * Math.PI / 180;
      const tip = {x: hs.ax + Math.cos(a) * hs.length, y: hs.ay + Math.sin(a) * hs.length};
      const q = this.toScreen(tip);
      ctx.fillStyle = bad ? '#ff2020' : COL.handle;
      ctx.beginPath(); ctx.arc(q.x, q.y, 3, 0, 7); ctx.fill();
    }

    // 控制点
    const drawVerts = (loop, which, color) => {
      loop.forEach((p, i) => {
        const q = this.toScreen(p);
        const sel = this.selVertex && this.selVertex.loop === which &&
                    this.selVertex.i === i;
        ctx.beginPath();
        ctx.arc(q.x, q.y, sel ? 5 : 3, 0, 7);
        ctx.fillStyle = sel ? '#fff' : color;
        ctx.fill();
      });
    };
    drawVerts(t.spec.outer, 'outer', COL.board);
    t.spec.holes.forEach((h, i) =>
      drawVerts(h, `hole:${i}`, COL.board));
    if (t.spec.handle) {
      const a = this.toScreen({x: t.spec.handle.ax, y: t.spec.handle.ay});
      ctx.beginPath(); ctx.arc(a.x, a.y, 4, 0, 7);
      ctx.fillStyle = '#fff'; ctx.fill();
    }

    // 孔洞草稿
    if (this.draftHole.length) {
      ctx.strokeStyle = '#fff';
      ctx.beginPath();
      this.draftHole.forEach((p, i) => {
        const q = this.toScreen(p);
        i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y);
      });
      ctx.stroke();
    }
    // 比例尺
    this.drawScale(ctx);
  }

  drawScale(ctx) {
    const len = 50, x0 = 14, y0 = this.cssH - 16;
    const px = len * this.view.k;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + px, y0);
    ctx.moveTo(x0, y0 - 4); ctx.lineTo(x0, y0 + 4);
    ctx.moveTo(x0 + px, y0 - 4); ctx.lineTo(x0 + px, y0 + 4); ctx.stroke();
    ctx.fillStyle = '#cfd2da'; ctx.font = '10px sans-serif';
    ctx.fillText('50 mm', x0 + px / 2 - 14, y0 - 6);
  }

  // ------------------------------------------------------------ 指针交互
  eventMm(e) {
    const rect = this.$('mb-canvas').getBoundingClientRect();
    return this.toMm(e.clientX - rect.left, e.clientY - rect.top);
  }

  hitVertex(t, e, tolPx = 7) {
    const rect = this.$('mb-canvas').getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const groups = [['outer', t.spec.outer]];
    (t.spec.holes || []).forEach((h, i) => groups.push([`hole:${i}`, h]));
    for (const [which, lp] of groups) {
      for (let i = 0; i < lp.length; i++) {
        const q = this.toScreen(lp[i]);
        if (Math.hypot(q.x - sx, q.y - sy) <= tolPx)
          return {loop: which, i};
      }
    }
    return null;
  }

  onDown(e) {
    const m = this.eventMm(e);
    if (this.addHoleMode) {
      this.draftHole.push(m);
      this.draw();
      return;
    }
    const t = this.tool();
    if (!t) return;
    const hit = this.hitVertex(t, e);
    if (hit) {
      this.selVertex = hit;
      this.selLoop = hit.loop;
      this.drag = {kind: 'vertex'};
      this.draw();
      return;
    }
    // 手柄接入点
    if (t.spec.handle) {
      const a = {x: t.spec.handle.ax, y: t.spec.handle.ay};
      const qa = this.toScreen(a);
      const rect = this.$('mb-canvas').getBoundingClientRect();
      if (Math.hypot(qa.x - (e.clientX - rect.left),
                     qa.y - (e.clientY - rect.top)) <= 7) {
        this.drag = {kind: 'anchor'};
        this.selLoop = 'handle';
        return;
      }
    }
    // 点中环体则选中该环
    if (pointInPolygon(m, t.spec.outer)) {
      let chosen = 'outer';
      t.spec.holes.forEach((h, i) => {
        if (pointInPolygon(m, h)) chosen = `hole:${i}`;
      });
      this.selLoop = chosen;
      this.draw();
    }
  }

  onMove(e) {
    const m = this.eventMm(e);
    this.$('mb-cursor').textContent = `x=${m.x.toFixed(1)} y=${m.y.toFixed(1)} mm`;
    const t = this.tool();
    if (!this.drag || !t) {
      if (this.draftHole.length) this.draw();
      return;
    }
    if (this.drag.kind === 'vertex') {
      this.selVertex && (this.loopOf(t, this.selVertex.loop)[this.selVertex.i] = m);
      this.recomputeCurrent();
      this.mark();
    } else if (this.drag.kind === 'anchor') {
      // 吸附到最近的外环边中点
      const best = this.nearestOuterPoint(m);
      t.spec.handle.ax = best.x;
      t.spec.handle.ay = best.y;
      this.recomputeCurrent();
      this.mark();
    }
  }

  onUp() {
    if (this.drag && (this.drag.kind === 'vertex' || this.drag.kind === 'anchor'))
      this.renderParams();
    this.drag = null;
  }

  nearestOuterPoint(m) {
    const t = this.tool();
    const lp = t.spec.outer;
    let best = lp[0], bd = Infinity;
    for (let i = 0; i < lp.length; i++) {
      const a = lp[i], b = lp[(i + 1) % lp.length];
      const dx = b.x - a.x, dy = b.y - a.y;
      const den = dx * dx + dy * dy || 1;
      const u = Math.max(0, Math.min(1, ((m.x - a.x) * dx + (m.y - a.y) * dy) / den));
      const q = {x: a.x + u * dx, y: a.y + u * dy};
      const d = Math.hypot(q.x - m.x, q.y - m.y);
      if (d < bd) { bd = d; best = q; }
    }
    return best;
  }

  // ------------------------------------------------------------ 持久化
  mark() {
    this.dirty = true;
    this.updateExportLink();
  }

  async autosave() {
    if (!this.dirty) return;
    await this.save(false);
  }

  async save(manual) {
    if (!this.tool()) return;
    const t = this.tool();
    const r = await fetch(`/api/mask-tools/${t.id}`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: t.name, spec: t.spec})});
    if (r.ok) {
      this.dirty = false;
      const j = await r.json();
      // 以服务端检测结果为准（含基于区域栅格的校验），合并回显
      if (j.result) {
        t.result = {...t.result, ...j.result,
          target: t.result.target};  // 保留前端 target 供绘制
        this.renderIssues(); this.renderToolList();
      }
      if (manual) this.toast('已保存');
    }
  }

  updateExportLink() {
    const ids = this.tools.map(t => t.id);
    this.$('mb-export').href =
      `/projects/${this.pid}/maskboard.svg${ids.length ? '?tools=' + ids.join(',') : ''}`;
  }

  // ------------------------------------------------------------ 版本
  async saveVersion() {
    const t = this.tool();
    if (!t) return;
    await this.save(false);
    const label = this.$('tv-label').value ||
      `版本 ${new Date().toLocaleString()}`;
    const r = await fetch(`/api/mask-tools/${t.id}/versions`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({label})});
    if (r.ok) { this.toast('版本已保存'); this.$('tv-label').value = '';
      this.loadVersions(); }
  }

  async loadVersions() {
    const t = this.tool();
    const box = this.$('tv-list');
    if (!t) { box.innerHTML = '—'; return; }
    const j = await fetch(`/api/mask-tools/${t.id}/versions`).then(r => r.json());
    if (!j.versions.length) { box.innerHTML = '<span class="muted">暂无版本。</span>'; return; }
    box.innerHTML = '';
    j.versions.forEach(v => {
      const d = document.createElement('div');
      d.className = 'item';
      d.innerHTML =
        `<div class="row" style="margin:0">
           <label class="row" style="margin:0"><input type="checkbox" class="vc" value="${v.id}">
           <span class="name">${esc(v.label)}</span></label></div>
         <div class="muted" style="font-size:11px">${v.created_at}${v.note ? ' · ' + esc(v.note) : ''}</div>
         <div class="tools">
           <button class="vr">回滚</button>
           <button class="vd danger">删</button></div>`;
      d.querySelector('.vr').onclick = async () => {
        if (!confirm(`回滚到「${v.label}」？当前未存版本的改动会丢失。`)) return;
        await fetch(`/api/mask-tool-versions/${v.id}/restore`, {method: 'POST'});
        location.reload();
      };
      d.querySelector('.vd').onclick = async () => {
        if (!confirm('删除该版本？')) return;
        await fetch(`/api/mask-tool-versions/${v.id}`, {method: 'DELETE'});
        this.loadVersions();
      };
      box.appendChild(d);
    });
  }

  async compareVersionsDialog() {
    const ids = [...document.querySelectorAll('.vc:checked')].map(c => +c.value);
    const out = this.$('tv-diff');
    if (ids.length !== 2) { out.textContent = '请勾选恰好两个版本。'; return; }
    const j = await fetch(`/api/mask-tools/compare?a=${ids[0]}&b=${ids[1]}`)
      .then(r => r.json());
    out.innerHTML =
      `高度差 ${j.height_delta.toFixed(1)}mm · 缩放差 ${j.scale_delta.toFixed(5)}<br>` +
      (j.board_hausdorff_mm != null
        ? `板轮廓 Hausdorff 偏差 <strong>${j.board_hausdorff_mm.toFixed(2)}mm</strong>` +
          `（平均 ${j.board_mean_mm.toFixed(2)}mm）<br>` : '') +
      (j.proj_hausdorff_mm != null
        ? `预计投影偏差 <strong>${j.proj_hausdorff_mm.toFixed(2)}mm</strong>` +
          `（平均 ${j.proj_mean_mm.toFixed(2)}mm）<br>` : '') +
      (j.beta_delta != null ? `校准 β 差 ${j.beta_delta.toExponential(2)}<br>` : '') +
      `误差带 ${j.error_band_a_mm?.toFixed(2)} → ${j.error_band_b_mm?.toFixed(2)}mm`;
  }
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"]/g, c =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
}
