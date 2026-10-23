/**
 * Road viewport for the VANNET console (renderer v2).
 *
 * What is drawn comes from the backend snapshot:
 *   - road length, lane count and lane width: ``scene``;
 *   - RSU position and the road interval it covers: ``rsus``;
 *   - vehicle position, lane, class, speed, detector trust, activity: ``vehicles``;
 *   - offloading links: only events the backend reported (``events``);
 *   - federated-learning uploads, acceptances and exclusions: only the decisions
 *     of a completed round (``fl_viz``).
 *
 * Scale.  Positions along the road are true to scale.  A road is only about
 * 4 m x lanes wide and a car about 4.5 m long, so the lateral axis and the car
 * glyphs are enlarged (``ENLARGE``, stated in the footer) to keep them legible.
 * The scenery (trees, terrain stripes) is decorative and carries no data.
 *
 * Two views: a close-up window that follows the selected vehicle (or the
 * busiest stretch of road) and an overview of the whole road; a minimap of the
 * whole road is always shown underneath.
 */
class VANETRoadView {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) throw new Error('viewport container not found');
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'road-canvas';
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    if (!this.ctx) throw new Error('2D canvas is not available');

    this.scene = null;
    this.rsus = [];
    this.vehicles = new Map();      // vid -> {cur, prev, tCur, tPrev, ...}
    this.links = [];
    this.flAnim = null;
    this.flBanner = null;
    this.showGroundTruth = false;
    this.showCoverage = true;
    this.showLinks = true;
    this.selectedVid = null;
    this.hoverVid = null;
    this.onSelect = null;
    this.detectorTau = 0.4;

    this.mode = 'follow';           // 'follow' | 'overview'
    this.windowM = 520;             // visible road length in follow mode
    this.center = 0;
    this.targetCenter = 0;
    this.manualCenter = false;
    this._lastAuto = 0;
    this.ENLARGE = 2.4;             // glyph and lateral enlargement
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this._hit = [];
    this._bindEvents();
    this.resize();
    this._loop = this._loop.bind(this);
    this._raf = requestAnimationFrame(this._loop);
  }

  // ----------------------------------------------------------------- data
  setScene(scene, rsus) {
    if (scene) {
      const changed = !this.scene || this.scene.road_length_m !== scene.road_length_m
        || this.scene.mobility_mode !== scene.mobility_mode || this.scene.x_min_m !== scene.x_min_m;
      this.scene = scene;
      if (changed) this.fitToRoad();
    }
    if (Array.isArray(rsus)) this.rsus = rsus;
  }

  updateVehicles(list) {
    const now = performance.now();
    const seen = new Set();
    (Array.isArray(list) ? list : []).forEach((v) => {
      seen.add(v.vehicle_id);
      let e = this.vehicles.get(v.vehicle_id);
      const active = v.active !== false;
      if (!e) {
        e = { id: v.vehicle_id, prev: null, cur: null, tPrev: now, tCur: now };
        this.vehicles.set(v.vehicle_id, e);
      }
      // No interpolation across a wrap-around, a (de)activation or a long jump.
      const L = this.scene ? this.scene.road_length_m : 1e9;
      const jump = e.cur ? Math.abs(v.x - e.cur.x) : 0;
      e.prev = (e.cur && e.cur.active === active && jump < L * 0.4) ? e.cur : null;
      e.tPrev = e.tCur;
      e.cur = { x: v.x, y: v.y, lane: v.lane, speed: v.speed, active, trust: v.trust_score,
                role: v.role, vclass: v.vclass, queue: v.queue_length, cpu: v.cpu_capacity_ghz,
                rsus: v.connected_rsus || [], src: v.source_vehicle_id };
      e.tCur = now;
    });
    for (const id of Array.from(this.vehicles.keys())) if (!seen.has(id)) this.vehicles.delete(id);
  }

  /** Register an offloading event reported by the backend (never invented). */
  triggerOffloadingBeam(ev) {
    if (!ev) return;
    this.links.push({
      vid: ev.vid, target: ev.target, action: ev.action, ok: !!ev.success,
      alert: !!ev.detector_alert, born: performance.now(), src: ev.source_pos || null,
    });
    if (this.links.length > 120) this.links.splice(0, this.links.length - 120);
  }

  /** The decisions of a completed federated round (real backend data). */
  showFLRound(viz) {
    if (!viz || viz.round === this._lastFlRound) return;
    this._lastFlRound = viz.round;
    const clients = viz.clients || [];
    this.flAnim = { born: performance.now(), clients, round: viz.round };
    const excluded = clients.filter((c) => c.included === false).length;
    const accepted = clients.filter((c) => c.included === true).length;
    const uploaded = clients.filter((c) => c.submitted).length;
    this.flBanner = { born: performance.now(),
      text: `Round ${viz.round}: ${uploaded} uploads · ${accepted} accepted · ${excluded} excluded` };
  }

  fitToRoad() {
    if (!this.scene) return;
    const L = this.scene.road_length_m;
    this.windowM = Math.min(L, 520);
    this.center = (this.scene.x_min_m || 0) + L / 2;
    this.targetCenter = this.center;
    this.manualCenter = false;
  }

  setMode(mode) {
    this.mode = mode === 'overview' ? 'overview' : 'follow';
    this.manualCenter = false;
  }

  // --------------------------------------------------------------- layout
  resize() {
    const r = this.container.getBoundingClientRect();
    const w = Math.max(240, Math.floor(r.width));
    const h = Math.max(200, Math.floor(r.height));
    this.canvas.width = Math.floor(w * this.dpr);
    this.canvas.height = Math.floor(h * this.dpr);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.w = w; this.h = h;
    this.miniH = Math.min(70, Math.max(54, h * 0.2));
    this.mainH = h - this.miniH;
  }

  _roadX0() { return this.scene ? (this.scene.x_min_m || 0) : 0; }
  _roadL() { return this.scene ? this.scene.road_length_m : 1000; }
  _visibleM() { return this.mode === 'overview' ? this._roadL() : Math.min(this._roadL(), this.windowM); }
  _px(x) { return this.w / 2 + (x - this.center) * (this.w / this._visibleM()); }
  _xFromPx(px) { return this.center + (px - this.w / 2) / (this.w / this._visibleM()); }

  _geom() {
    const lanes = this.scene ? this.scene.num_lanes : 5;
    const laneW = this.scene ? this.scene.lane_width_m : 3.75;
    const pxPerM = this.w / this._visibleM();
    let laneH = Math.max(11, Math.min(32, laneW * pxPerM * this.ENLARGE));
    const maxRoad = this.mainH * 0.52;
    if (laneH * lanes > maxRoad) laneH = maxRoad / lanes;
    const roadH = lanes * laneH;
    const cy = this.mainH * 0.58;
    return { lanes, laneH, roadH, top: cy - roadH / 2, bottom: cy + roadH / 2, cy, pxPerM };
  }

  _color(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback || '#888';
  }

  // ------------------------------------------------------------- follow
  _autoCenter(now) {
    if (this.mode !== 'follow' || this.manualCenter) return;
    const sel = this.selectedVid !== null ? this.vehicles.get(this.selectedVid) : null;
    if (sel && sel.cur && sel.cur.active) {
      this.targetCenter = sel.cur.x;
    } else if (now - this._lastAuto > 1500) {
      this._lastAuto = now;
      const xs = [];
      this.vehicles.forEach((e) => { if (e.cur && e.cur.active) xs.push(e.cur.x); });
      if (xs.length) {
        xs.sort((a, b) => a - b);
        const W = this._visibleM();
        let best = 0, bestN = -1;
        for (let i = 0; i < xs.length; i++) {
          let j = i;
          while (j < xs.length && xs[j] - xs[i] <= W * 0.9) j++;
          if (j - i > bestN) { bestN = j - i; best = (xs[i] + xs[j - 1]) / 2; }
        }
        this.targetCenter = best;
      }
    }
    const L = this._roadL(), x0 = this._roadX0(), W = this._visibleM();
    this.targetCenter = Math.min(x0 + L - W / 2, Math.max(x0 + W / 2, this.targetCenter));
    if (Math.abs(this.targetCenter - this.center) > W * 1.2) this.center = this.targetCenter;
    this.center += (this.targetCenter - this.center) * 0.06;
  }

  // -------------------------------------------------------------- drawing
  _loop() {
    this._raf = requestAnimationFrame(this._loop);
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    const now = performance.now();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    const c = {
      terrain: this._color('--terrain', '#6b7a5e'), terrain2: this._color('--terrain-2', '#627156'),
      asphalt: this._color('--asphalt', '#3b4048'), marking: this._color('--marking', '#e8e8e2'),
      text: this._color('--text-muted', '#5d6877'), strong: this._color('--text', '#1c2430'),
      accent: this._color('--accent', '#1f5fbf'), ok: this._color('--ok', '#17803d'),
      warn: this._color('--warn', '#b26a00'), bad: this._color('--bad', '#c0392b'),
      panel: this._color('--viewport-bg', '#eef0f3'),
    };
    ctx.fillStyle = c.panel;
    ctx.fillRect(0, 0, this.w, this.h);
    if (!this.scene) {
      ctx.fillStyle = c.text;
      ctx.font = '13px Inter, system-ui, sans-serif';
      ctx.fillText('Waiting for the first backend snapshot', 24, 32);
      return;
    }
    this._autoCenter(now);
    this._hit = [];
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, this.w, this.mainH); ctx.clip();
    const g = this._geom();
    this._drawTerrain(ctx, g, c);
    this._drawRoad(ctx, g, c);
    if (this.showCoverage) this._drawCoverage(ctx, g, c);
    this._drawRSUs(ctx, g, c, now);
    this._drawVehicles(ctx, g, c, now);
    if (this.showLinks) this._drawLinks(ctx, g, c, now);
    this._drawFL(ctx, g, c, now);
    this._drawHUD(ctx, g, c, now);
    ctx.restore();
    this._drawMinimap(ctx, c, now);
  }

  _drawTerrain(ctx, g, c) {
    ctx.fillStyle = c.terrain;
    ctx.fillRect(0, 0, this.w, this.mainH);
    // Terrain stripes that move with the road give a sense of speed and scale.
    const W = this._visibleM(), step = W > 900 ? 100 : 25;
    ctx.fillStyle = c.terrain2;
    const x0 = Math.floor((this.center - W / 2) / step) * step;
    for (let x = x0; x < this.center + W / 2 + step; x += step * 2) {
      const px = this._px(x), pw = this._px(x + step) - px;
      ctx.fillRect(px, 0, pw, g.top - 6);
      ctx.fillRect(px, g.bottom + 6, pw, this.mainH);
    }
    // Decorative trees (deterministic, no data).
    if (W <= 700) {
      for (let x = Math.floor((this.center - W / 2) / 31) * 31; x < this.center + W / 2 + 31; x += 31) {
        const h1 = Math.abs(Math.sin(x * 12.9898) * 43758.5453) % 1;
        const h2 = Math.abs(Math.sin(x * 78.233 + 1.7) * 12543.17) % 1;
        const px = this._px(x + h1 * 20);
        const up = g.top - 12 - h2 * Math.max(8, g.top * 0.55);
        const dn = g.bottom + 14 + h1 * Math.max(8, (this.mainH - g.bottom) * 0.55);
        [up, dn].forEach((py, i) => {
          if (py < 4 || py > this.mainH - 4) return;
          ctx.fillStyle = i ? 'rgba(40,70,40,0.55)' : 'rgba(40,70,40,0.55)';
          ctx.beginPath(); ctx.arc(px, py, 4 + h2 * 4, 0, Math.PI * 2); ctx.fill();
        });
      }
    }
  }

  _drawRoad(ctx, g, c) {
    const x0 = this._roadX0(), L = this._roadL();
    const xl = Math.max(0, this._px(x0)), xr = Math.min(this.w, this._px(x0 + L));
    if (xr <= xl) return;
    // shoulders
    ctx.fillStyle = '#585d64';
    ctx.fillRect(xl, g.top - 6, xr - xl, g.roadH + 12);
    // asphalt with a light lengthwise gradient
    const grad = ctx.createLinearGradient(0, g.top, 0, g.bottom);
    grad.addColorStop(0, c.asphalt); grad.addColorStop(0.5, '#434850'); grad.addColorStop(1, c.asphalt);
    ctx.fillStyle = grad;
    ctx.fillRect(xl, g.top, xr - xl, g.roadH);
    // lane lines (dashed, scrolling with the road)
    ctx.strokeStyle = c.marking; ctx.lineWidth = 1.4;
    const dash = Math.max(8, 9 * g.pxPerM), gap = Math.max(10, 12 * g.pxPerM);
    ctx.setLineDash([dash, gap]);
    for (let i = 1; i < g.lanes; i++) {
      const y = g.top + i * g.laneH;
      ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(xr, y); ctx.stroke();
    }
    ctx.setLineDash([]);
    // solid edge lines
    ctx.lineWidth = 2;
    [g.top + 2, g.bottom - 2].forEach((y) => { ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(xr, y); ctx.stroke(); });
    // guardrails
    ctx.strokeStyle = '#9aa1ab'; ctx.lineWidth = 2.5;
    [g.top - 8, g.bottom + 8].forEach((y) => { ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(xr, y); ctx.stroke(); });
    // distance markers
    const W = this._visibleM();
    const step = W > 1500 ? 500 : (W > 500 ? 100 : 50);
    ctx.fillStyle = c.marking; ctx.font = '10px "JetBrains Mono", monospace'; ctx.textAlign = 'center';
    const first = Math.ceil((this.center - W / 2) / step) * step;
    for (let x = first; x <= this.center + W / 2; x += step) {
      if (x < x0 || x > x0 + L) continue;
      const px = this._px(x);
      ctx.fillRect(px - 0.5, g.bottom + 8, 1, 7);
      ctx.fillText(Math.round(x) + ' m', px, g.bottom + 27);
    }
    ctx.textAlign = 'left';
    // road ends (trace replay is a segment, not a loop)
    if (this.scene.mobility_mode !== 'highway') {
      ctx.fillStyle = 'rgba(0,0,0,0.25)';
      if (this._px(x0) > 0) ctx.fillRect(this._px(x0) - 4, g.top - 6, 4, g.roadH + 12);
      if (this._px(x0 + L) < this.w) ctx.fillRect(this._px(x0 + L), g.top - 6, 4, g.roadH + 12);
    }
  }

  _rsuHalfSpan(r) {
    const dz = (r.z_height || 0) - 1.5, dy = (r.y || 0);
    const rem = r.coverage_radius * r.coverage_radius - dy * dy - dz * dz;
    return rem > 0 ? Math.sqrt(rem) : 0;
  }

  _drawCoverage(ctx, g, c) {
    this.rsus.forEach((r, i) => {
      const half = this._rsuHalfSpan(r);
      if (half <= 0) return;
      const x0 = this._px(r.x - half), x1 = this._px(r.x + half);
      ctx.fillStyle = 'rgba(80,150,255,0.13)';
      ctx.fillRect(x0, g.top, x1 - x0, g.roadH);
      ctx.strokeStyle = 'rgba(120,180,255,0.75)'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
      [x0, x1].forEach((x) => { ctx.beginPath(); ctx.moveTo(x, g.top - 6); ctx.lineTo(x, g.bottom + 6); ctx.stroke(); });
      ctx.setLineDash([]);
    });
  }

  _drawRSUs(ctx, g, c, now) {
    ctx.font = '600 11px Inter, system-ui, sans-serif';
    this.rsus.forEach((r, i) => {
      const px = this._px(r.x);
      if (px < -40 || px > this.w + 40) return;
      const baseY = g.top - 14, topY = Math.max(16, g.top - 54);
      // pole and cabinet
      ctx.strokeStyle = '#2c3138'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(px, baseY); ctx.lineTo(px, topY + 6); ctx.stroke();
      ctx.fillStyle = '#4a515a'; ctx.fillRect(px - 8, baseY - 5, 16, 9);
      // antenna head, tinted by operational load
      const load = Math.min(1, r.utilization || 0);
      ctx.fillStyle = load > 0.75 ? c.bad : (load > 0.4 ? c.warn : c.accent);
      ctx.beginPath(); ctx.moveTo(px, topY - 4); ctx.lineTo(px - 8, topY + 8); ctx.lineTo(px + 8, topY + 8); ctx.closePath(); ctx.fill();
      // radio rings (decorative pulse; the covered interval itself is data-driven)
      for (let k = 0; k < 3; k++) {
        const ph = ((now / 1400) + k / 3) % 1;
        ctx.strokeStyle = `rgba(90,160,255,${0.55 * (1 - ph)})`; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(px, topY + 2, 8 + ph * 34, Math.PI * 0.18, Math.PI * 0.82); ctx.stroke();
      }
      // label with load and queue
      const label = `${r.rsu_id}  ${Math.round(load * 100)}% · q${r.queue_length ?? 0}` +
        (r.trust_score !== undefined ? ` · T ${Number(r.trust_score).toFixed(2)}` : '');
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(20,24,30,0.78)';
      ctx.fillRect(px - tw / 2 - 5, topY - 24, tw + 10, 16);
      ctx.fillStyle = '#f2f4f7'; ctx.textAlign = 'center';
      ctx.fillText(label, px, topY - 12);
      ctx.textAlign = 'left';
    });
  }

  _glyph(vclass) {
    switch (vclass) {
      case 'Truck': return { len: 9.0, color: '#8a6d3b' };
      case 'SUV': return { len: 4.9, color: '#5f7f5a' };
      case 'Autonomous_EV': return { len: 4.6, color: '#2f8f9d' };
      default: return { len: 4.5, color: '#4c6a92' };
    }
  }

  _vehicleScreen(e, g, now) {
    if (!e.cur || !e.cur.active) return null;
    const x = this._xInterp(e, now);
    const lane = Math.min(g.lanes, Math.max(1, e.cur.lane || 1));
    return { px: this._px(x), py: g.top + (lane - 0.5) * g.laneH, x };
  }

  _xInterp(e, now) {
    if (!e.prev) return e.cur.x;
    const span = Math.max(1, e.tCur - e.tPrev);
    const k = Math.min(1.4, Math.max(0, (now - e.tCur) / span));
    return e.cur.x + (e.cur.x - e.prev.x) * k;      // continue along the last displacement
  }

  _drawVehicles(ctx, g, c, now) {
    const detail = g.pxPerM * this.ENLARGE > 4;
    this.vehicles.forEach((e) => {
      const s = this._vehicleScreen(e, g, now);
      if (!s || s.px < -60 || s.px > this.w + 60) return;
      const gl = this._glyph(e.cur.vclass);
      const len = Math.max(8, gl.len * g.pxPerM * this.ENLARGE);
      const wid = Math.max(5, Math.min(g.laneH * 0.66, len * 0.42));
      const t = e.cur.trust;
      // trust ring (detector state), under the car
      if (t !== undefined && t !== null && t < 0.6) {
        ctx.fillStyle = t < this.detectorTau ? 'rgba(220,60,50,0.35)' : 'rgba(230,160,40,0.32)';
        ctx.beginPath(); ctx.ellipse(s.px, s.py, len * 0.78, wid * 0.95, 0, 0, Math.PI * 2); ctx.fill();
      }
      // body
      const x0 = s.px - len / 2, y0 = s.py - wid / 2, r = Math.min(4, wid / 3);
      ctx.fillStyle = gl.color;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x0, y0, len, wid, r) : ctx.rect(x0, y0, len, wid); ctx.fill();
      if (detail) {
        // windscreen, roof, rear window
        ctx.fillStyle = 'rgba(210,230,245,0.85)';
        ctx.fillRect(x0 + len * 0.62, y0 + wid * 0.14, len * 0.14, wid * 0.72);
        ctx.fillStyle = 'rgba(0,0,0,0.20)';
        ctx.fillRect(x0 + len * 0.30, y0 + wid * 0.12, len * 0.28, wid * 0.76);
        ctx.fillStyle = 'rgba(210,230,245,0.55)';
        ctx.fillRect(x0 + len * 0.16, y0 + wid * 0.16, len * 0.10, wid * 0.68);
        if (e.cur.vclass === 'Truck') {
          ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x0 + len * 0.42, y0); ctx.lineTo(x0 + len * 0.42, y0 + wid); ctx.stroke();
        }
        if (e.cur.vclass === 'Autonomous_EV') {
          ctx.fillStyle = '#d9f4ff'; ctx.beginPath(); ctx.arc(s.px, s.py, Math.max(1.5, wid * 0.13), 0, Math.PI * 2); ctx.fill();
        }
        // headlights
        ctx.fillStyle = '#ffe9a6';
        ctx.fillRect(x0 + len - 2, y0 + 1, 2, Math.max(2, wid * 0.22));
        ctx.fillRect(x0 + len - 2, y0 + wid - 1 - Math.max(2, wid * 0.22), 2, Math.max(2, wid * 0.22));
      }
      // ground-truth overlay (evaluation only; never an input of the detector)
      if (this.showGroundTruth && e.cur.role && e.cur.role !== 'honest') {
        ctx.strokeStyle = c.bad; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]);
        ctx.strokeRect(x0 - 3, y0 - 3, len + 6, wid + 6); ctx.setLineDash([]);
        ctx.fillStyle = c.bad; ctx.font = '600 9px Inter, sans-serif';
        ctx.fillText(e.cur.role === 'free_rider' ? 'FR' : (e.cur.role === 'poisoning' ? 'PZ' : 'UR'), x0, y0 - 5);
      }
      const sel = this.selectedVid === e.id, hov = this.hoverVid === e.id;
      if (sel || hov) {
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = sel ? 2 : 1.2;
        ctx.strokeRect(x0 - 3, y0 - 3, len + 6, wid + 6);
      }
      if (sel || (detail && this._visibleM() < 420)) {
        ctx.fillStyle = 'rgba(15,18,24,0.8)'; ctx.font = '10px "JetBrains Mono", monospace';
        const label = 'V' + e.id + (sel ? ' ' + Math.round(e.cur.speed * 3.6) + ' km/h' : '');
        const tw = ctx.measureText(label).width;
        ctx.fillRect(s.px - tw / 2 - 3, y0 - 17, tw + 6, 12);
        ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.fillText(label, s.px, y0 - 8); ctx.textAlign = 'left';
      }
      this._hit.push({ vid: e.id, x: s.px, y: s.py, hw: Math.max(len / 2, 9), hh: Math.max(wid / 2, 9) });
    });
  }

  _rsuById(id) { return this.rsus.find((r) => r.rsu_id === id); }

  _drawLinks(ctx, g, c, now) {
    this.links = this.links.filter((l) => now - l.born < 1300);
    const W = this._visibleM();
    let drawn = 0;
    this.links.forEach((l) => {
      const age = (now - l.born) / 1300;
      const ve = this.vehicles.get(l.vid);
      const src = ve ? this._vehicleScreen(ve, g, now) : null;
      if (!src) return;
      // In the close-up only links of vehicles in the window are drawn; in the
      // overview only detector alerts, so the view stays readable.
      if (this.mode === 'overview' && !l.alert) return;
      if (src.px < -20 || src.px > this.w + 20) return;
      if (++drawn > 60) return;
      let dst = null;
      if (l.action >= 1 && l.action <= 3) {
        const r = this._rsuById(l.target);
        if (r) dst = { px: this._px(r.x), py: Math.max(16, g.top - 50) + 4 };
      } else if (l.action === 4) {
        const pe = this.vehicles.get(parseInt(String(l.target).replace('V', ''), 10));
        dst = pe ? this._vehicleScreen(pe, g, now) : null;
      }
      if (!dst) return;
      const col = l.alert ? c.bad : (l.ok ? c.ok : '#8a929e');
      const v2v = l.action === 4;
      ctx.globalAlpha = Math.max(0, 0.95 * (1 - age));
      ctx.strokeStyle = v2v ? c.warn : col;
      ctx.lineWidth = l.alert ? 2.4 : 1.4;
      ctx.setLineDash(v2v ? [4, 3] : (l.ok ? [] : [2, 3]));
      const mx = (src.px + dst.px) / 2, my = Math.min(src.py, dst.py) - 14;
      ctx.beginPath(); ctx.moveTo(src.px, src.py); ctx.quadraticCurveTo(mx, my, dst.px, dst.py); ctx.stroke();
      ctx.setLineDash([]);
      // packet travelling to the target
      const k = Math.min(1, age * 1.6);
      const bx = (1 - k) * (1 - k) * src.px + 2 * (1 - k) * k * mx + k * k * dst.px;
      const by = (1 - k) * (1 - k) * src.py + 2 * (1 - k) * k * my + k * k * dst.py;
      ctx.fillStyle = v2v ? c.warn : col; ctx.beginPath(); ctx.arc(bx, by, l.alert ? 3.4 : 2.6, 0, Math.PI * 2); ctx.fill();
      if (l.alert && age < 0.7) {
        ctx.fillStyle = c.bad; ctx.font = '700 10px Inter, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('DETECTOR ALERT', mx, my - 4); ctx.textAlign = 'left';
      }
      ctx.globalAlpha = 1;
    });
  }

  _drawFL(ctx, g, c, now) {
    const a = this.flAnim;
    if (!a) return;
    const age = now - a.born;
    if (age > 4200) { this.flAnim = null; return; }
    const k = Math.min(1, age / 1800);
    // The aggregator is drawn as a badge; lines run from each vehicle in view.
    const bx = this.w - 112, by = 26;
    ctx.fillStyle = 'rgba(20,24,30,0.85)'; ctx.fillRect(bx - 4, by - 14, 112, 22);
    ctx.fillStyle = '#9fd0ff'; ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText('FL aggregator', bx + 4, by + 1);
    const fade = age > 3200 ? 1 - (age - 3200) / 1000 : 1;
    a.clients.forEach((cl) => {
      const e = this.vehicles.get(cl.vid);
      const s = e ? this._vehicleScreen(e, g, now) : null;
      if (!s || s.px < -10 || s.px > this.w + 10) return;
      const excluded = cl.included === false;
      const col = excluded ? c.bad : (cl.submitted ? '#6fb1ff' : '#8a929e');
      if (!cl.submitted && !excluded) return;
      ctx.globalAlpha = (excluded ? 0.95 : 0.55) * fade;
      ctx.strokeStyle = col; ctx.lineWidth = excluded ? 2 : 1;
      ctx.setLineDash(excluded ? [] : [3, 4]);
      ctx.beginPath(); ctx.moveTo(s.px, s.py); ctx.lineTo(bx + 52, by + 6); ctx.stroke(); ctx.setLineDash([]);
      const px = s.px + (bx + 52 - s.px) * k, py = s.py + (by + 6 - s.py) * k;
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px, py, excluded ? 3.6 : 2.4, 0, Math.PI * 2); ctx.fill();
      if (excluded && k >= 0.95) {
        ctx.strokeStyle = c.bad; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(s.px - 7, s.py - 7); ctx.lineTo(s.px + 7, s.py + 7);
        ctx.moveTo(s.px + 7, s.py - 7); ctx.lineTo(s.px - 7, s.py + 7); ctx.stroke();
        ctx.fillStyle = c.bad; ctx.font = '600 9px Inter, sans-serif';
        ctx.fillText('excluded' + (cl.reason ? ': ' + String(cl.reason).slice(0, 22) : ''), s.px + 9, s.py - 9);
      }
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = 1;
  }

  _drawHUD(ctx, g, c, now) {
    // round banner
    const b = this.flBanner;
    if (b && now - b.born < 5000) {
      ctx.font = '600 12px Inter, sans-serif';
      const tw = ctx.measureText(b.text).width;
      ctx.fillStyle = 'rgba(20,24,30,0.85)'; ctx.fillRect(10, 8, tw + 16, 22);
      ctx.fillStyle = '#fff'; ctx.fillText(b.text, 18, 24);
    }
    // legend / scale
    ctx.font = '10px Inter, sans-serif';
    const txt = (this.mode === 'overview' ? 'Overview of the whole road' : `Close-up ${Math.round(this._visibleM())} m`) +
      ` · position to scale · lateral and vehicle glyphs ×${this.ENLARGE} · scenery is decorative` +
      ' · wheel: zoom · drag: pan · click a vehicle';
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    const tw = ctx.measureText(txt).width;
    ctx.fillRect(8, this.mainH - 20, tw + 12, 16);
    ctx.fillStyle = '#e6e9ee'; ctx.fillText(txt, 14, this.mainH - 8);
  }

  _drawMinimap(ctx, c, now) {
    const y0 = this.mainH, h = this.miniH, pad = 12;
    ctx.fillStyle = this._color('--panel', '#fff'); ctx.fillRect(0, y0, this.w, h);
    ctx.strokeStyle = this._color('--border', '#ccc'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, y0 + 0.5); ctx.lineTo(this.w, y0 + 0.5); ctx.stroke();
    const x0 = this._roadX0(), L = this._roadL();
    const sx = (x) => pad + (x - x0) / L * (this.w - 2 * pad);
    const ry = y0 + h * 0.58, rh = 12;
    ctx.fillStyle = '#4a4f57'; ctx.fillRect(sx(x0), ry - rh / 2, sx(x0 + L) - sx(x0), rh);
    this.rsus.forEach((r) => {
      const half = this._rsuHalfSpan(r);
      ctx.fillStyle = 'rgba(80,150,255,0.35)';
      ctx.fillRect(sx(r.x - half), ry - rh / 2 - 3, sx(r.x + half) - sx(r.x - half), rh + 6);
      ctx.fillStyle = c.accent; ctx.beginPath();
      ctx.moveTo(sx(r.x), ry - rh / 2 - 6); ctx.lineTo(sx(r.x) - 4, ry - rh / 2 - 13); ctx.lineTo(sx(r.x) + 4, ry - rh / 2 - 13); ctx.closePath(); ctx.fill();
    });
    const excl = new Set();
    if (this.flAnim) this.flAnim.clients.forEach((cl) => { if (cl.included === false) excl.add(cl.vid); });
    this.vehicles.forEach((e) => {
      if (!e.cur || !e.cur.active) return;
      const t = e.cur.trust;
      ctx.fillStyle = excl.has(e.id) ? c.bad : (t !== undefined && t < this.detectorTau ? c.bad : (t < 0.6 ? c.warn : '#dfe6ef'));
      const x = sx(this._xInterp(e, now));
      ctx.beginPath(); ctx.arc(x, ry, excl.has(e.id) ? 4 : 2.8, 0, Math.PI * 2); ctx.fill();
      if (this.selectedVid === e.id) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke(); }
    });
    // window indicator
    if (this.mode === 'follow') {
      const W = this._visibleM();
      ctx.strokeStyle = '#ffd166'; ctx.lineWidth = 2;
      ctx.strokeRect(sx(this.center - W / 2), ry - rh / 2 - 8, sx(this.center + W / 2) - sx(this.center - W / 2), rh + 16);
    }
    ctx.fillStyle = c.text; ctx.font = '10px Inter, sans-serif';
    ctx.fillText('Whole road · click to move the close-up', pad, y0 + 12);
    this._mini = { sx, x0, L, pad, y0, h };
  }

  // -------------------------------------------------------------- events
  _bindEvents() {
    window.addEventListener('resize', () => this.resize());
    if (typeof ResizeObserver === 'function') {
      this._resizeObserver = new ResizeObserver(() => this.resize());
      this._resizeObserver.observe(this.container);
    }
    this.canvas.addEventListener('wheel', (e) => {
      if (!this.scene || this.mode === 'overview') return;
      e.preventDefault();
      this.windowM = Math.min(this._roadL(), Math.max(80, this.windowM * (e.deltaY < 0 ? 0.85 : 1.18)));
    }, { passive: false });
    let drag = null;
    this.canvas.addEventListener('mousedown', (e) => { drag = { x: e.clientX, c: this.center, moved: false }; });
    window.addEventListener('mouseup', (e) => { if (drag && !drag.moved) this._click(e); drag = null; });
    window.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      if (drag) {
        const dx = e.clientX - drag.x;
        if (Math.abs(dx) > 3) { drag.moved = true; this.manualCenter = true; }
        if (drag.moved && this.mode === 'follow') this.center = this.targetCenter = drag.c - dx / (this.w / this._visibleM());
        return;
      }
      const h = (this._hit || []).find((q) => Math.abs(q.x - mx) <= q.hw && Math.abs(q.y - my) <= q.hh);
      this.hoverVid = h ? h.vid : null;
      this.canvas.style.cursor = h ? 'pointer' : 'grab';
    });
    this.canvas.addEventListener('dblclick', () => { this.manualCenter = false; this.selectedVid = null; this.fitToRoad(); });
  }

  _click(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (mx < 0 || my < 0 || mx > rect.width || my > rect.height) return;
    if (my >= this.mainH && this._mini) {
      const m = this._mini;
      const x = m.x0 + (mx - m.pad) / (this.w - 2 * m.pad) * m.L;
      this.manualCenter = true; this.selectedVid = null; this.targetCenter = this.center = x;
      this.mode = 'follow';
      return;
    }
    const hit = (this._hit || []).find((q) => Math.abs(q.x - mx) <= q.hw && Math.abs(q.y - my) <= q.hh);
    this.selectedVid = hit ? hit.vid : null;
    if (hit) this.manualCenter = false;
    if (typeof this.onSelect === 'function') this.onSelect(this.selectedVid);
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._resizeObserver) this._resizeObserver.disconnect();
  }
}

window.VANETRoadView = VANETRoadView;
