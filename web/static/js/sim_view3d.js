/**
 * 3D road viewport for the VANNET console (Three.js, global build).
 *
 * Same data contract and public interface as the 2D renderer (sim_view.js):
 *   setScene, updateVehicles, triggerOffloadingBeam, showFLRound, setMode, resize,
 *   onSelect, showCoverage / showLinks / showGroundTruth, selectedVid, detectorTau.
 *
 * What is data and what is decoration.
 *   DATA        road length / lanes / lane width (scene), RSU position, height and
 *               coverage radius (rsus), vehicle position, lane, class, trust and
 *               role (vehicles), every offloading link (only events the backend
 *               reported), every federated upload / exclusion (only the decisions
 *               of a completed round).
 *   DECORATION  terrain, buildings, trees, street lights and the sky.  They are
 *               generated from a fixed seed, carry no information and are placed
 *               away from the road.
 *   SCALE       positions along the road are true to scale; vehicle bodies are
 *               drawn larger than life (stated in the on-screen note) so that
 *               they stay legible from the overview camera.
 *
 * Vehicle rings use the trust the detector currently assigns (green >= 0.8,
 * cyan 0.6-0.8, amber between the exclusion threshold and 0.6, red below the
 * threshold).  The ground-truth role is shown only when the overlay is enabled.
 */
class VANETRoadView3D {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) throw new Error('viewport container not found');
    if (typeof THREE === 'undefined') throw new Error('Three.js did not load');
    this.THREE = THREE;

    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    } catch (e) {
      throw new Error('WebGL is not available');
    }
    if (!this.renderer || !this.renderer.getContext()) throw new Error('WebGL is not available');
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.canvas = this.renderer.domElement;
    this.canvas.className = 'road-canvas road-canvas-3d';
    this.canvas.style.display = 'block';
    this.canvas.style.position = 'absolute';
    this.canvas.style.left = '0'; this.canvas.style.top = '0';   // never sizes its container
    this.container.appendChild(this.canvas);

    this.scene = null;                 // backend scene description
    this.rsus = [];
    this.vehicles = new Map();         // vid -> {prev, cur, tPrev, tCur, group, ...}
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
    this.mode = 'follow';
    this.center = 0;
    this.targetCenter = 0;
    this.manualCenter = false;
    this._lastAuto = 0;
    this.overviewScale = 2.6;
    this.SCALE_L = 1.8; this.SCALE_W = 1.35; this.SCALE_H = 1.5;
    this._lastFlRound = undefined;

    // camera state (spherical around the look-at point)
    this.cam = { yaw: -0.55, pitch: 0.42, dist: 190, fov: 48 };
    this.camGoal = { yaw: -0.55, pitch: 0.42, dist: 190 };
    this.userCam = false;

    this.three = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(this.cam.fov, 1.6, 0.3, 30000);
    // Free navigation: the look-at point can be moved in all three axes.
    this.focusY = 2; this.focusZ = 0; this.freeCam = false; this.keys = new Set(); this.pointerInside = false;
    this._buildSky();
    this._buildLights();
    this.world = new THREE.Group();             // rebuilt when the scene changes
    this.three.add(this.world);
    this.dyn = new THREE.Group();               // links, labels, alerts (dynamic)
    this.three.add(this.dyn);
    this.vehGroup = new THREE.Group();
    this.three.add(this.vehGroup);
    this.rsuObjs = new Map();

    this._sharedGeo();
    this._buildAggregator();
    this._buildOverlay();
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this._bindEvents();
    this.resize();
    this._loop = this._loop.bind(this);
    this._raf = requestAnimationFrame(this._loop);
  }

  // ---------------------------------------------------------------- setup
  _buildSky() {
    const c = document.createElement('canvas');
    c.width = 4; c.height = 256;
    const g = c.getContext('2d');
    const grad = g.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0.0, '#0e1b3d');
    grad.addColorStop(0.45, '#2f5088');
    grad.addColorStop(0.78, '#8fb0d8');
    grad.addColorStop(1.0, '#e9c9a1');
    g.fillStyle = grad; g.fillRect(0, 0, 4, 256);
    this.three.background = new THREE.CanvasTexture(c);
    this.three.fog = new THREE.Fog(0x7f9fca, 700, 3200);
  }

  _buildLights() {
    this.three.add(new THREE.HemisphereLight(0xa9c2ee, 0x2a3350, 0.95));
    const sun = new THREE.DirectionalLight(0xffe2b8, 1.25);
    sun.position.set(-140, 220, 160);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    const s = sun.shadow.camera;
    s.left = -170; s.right = 170; s.top = 120; s.bottom = -120; s.near = 10; s.far = 700;
    sun.shadow.bias = -0.0006;
    this.sun = sun;
    this.three.add(sun);
    this.three.add(sun.target);
  }

  _sharedGeo() {
    const T = THREE;
    this.geo = {
      box: new T.BoxGeometry(1, 1, 1),
      wheel: new T.CylinderGeometry(0.42, 0.42, 0.3, 14),
      sphere: new T.SphereGeometry(1, 12, 10),
      ring: new T.RingGeometry(0.78, 1, 40),
      cone: new T.ConeGeometry(1, 1, 10),
    };
    this.mat = {
      glass: new T.MeshStandardMaterial({ color: 0x1b2a40, metalness: 0.6, roughness: 0.18 }),
      tire: new T.MeshStandardMaterial({ color: 0x14161a, roughness: 0.9 }),
      head: new T.MeshBasicMaterial({ color: 0xfff1c4 }),
      tail: new T.MeshBasicMaterial({ color: 0xff3b3b }),
      lidar: new T.MeshStandardMaterial({ color: 0x20262e, metalness: 0.5, roughness: 0.4 }),
    };
  }

  _buildAggregator() {
    const T = THREE;
    const g = new T.Group();
    const core = new T.Mesh(new T.OctahedronGeometry(7, 0),
      new T.MeshStandardMaterial({ color: 0x4c8dff, emissive: 0x2a5fd0, emissiveIntensity: 0.9, metalness: 0.3, roughness: 0.3 }));
    g.add(core);
    const ring = new T.Mesh(new T.TorusGeometry(11, 0.5, 8, 40), new T.MeshBasicMaterial({ color: 0x7fb0ff, transparent: true, opacity: 0.8 }));
    ring.rotation.x = Math.PI / 2;
    g.add(ring);
    const label = this._label('Aggregation server', '#7fb0ff', 'rgba(10,18,40,0.8)');
    label.position.set(0, 16, 0);
    label.scale.multiplyScalar(1.3);
    g.add(label);
    g.visible = false;
    this.agg = { group: g, core, ring };
    this.three.add(g);
  }

  _buildOverlay() {
    const wrap = this.container;
    if (getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';
    wrap.style.overflow = 'hidden';
    const hud = document.createElement('div');
    hud.className = 'view3d-hud';
    hud.style.cssText = 'position:absolute;left:10px;top:8px;right:10px;pointer-events:none;display:flex;justify-content:space-between;gap:12px;align-items:flex-start;font:12px/1.35 Inter,system-ui,sans-serif;color:#e8eefc;text-shadow:0 1px 3px rgba(0,0,0,.7)';
    this.hudLeft = document.createElement('div');
    this.hudRight = document.createElement('div');
    this.hudRight.style.textAlign = 'right';
    hud.appendChild(this.hudLeft); hud.appendChild(this.hudRight);
    wrap.appendChild(hud);
    this.tip = document.createElement('div');
    this.tip.style.cssText = 'position:absolute;display:none;pointer-events:none;padding:4px 8px;border-radius:4px;background:rgba(10,16,32,.88);color:#e8eefc;font:12px Inter,system-ui,sans-serif;white-space:nowrap;z-index:3';
    wrap.appendChild(this.tip);
    this.mini = document.createElement('canvas');
    this.mini.style.cssText = 'position:absolute;left:10px;right:10px;bottom:8px;width:calc(100% - 20px);height:34px;border-radius:4px;background:rgba(10,16,32,.55);cursor:pointer';
    wrap.appendChild(this.mini);
    this.hudLeft.textContent = '3D view';
  }

  // ------------------------------------------------------------- helpers
  _label(text, color, bg) {
    const c = document.createElement('canvas');
    const g = c.getContext('2d');
    const font = '600 30px Inter, system-ui, sans-serif';
    g.font = font;
    const w = Math.ceil(g.measureText(text).width) + 28, h = 52;
    c.width = w; c.height = h;
    g.font = font;
    g.fillStyle = bg || 'rgba(10,16,32,0.78)';
    const r = 10;
    g.beginPath();
    g.moveTo(r, 0); g.lineTo(w - r, 0); g.quadraticCurveTo(w, 0, w, r); g.lineTo(w, h - r);
    g.quadraticCurveTo(w, h, w - r, h); g.lineTo(r, h); g.quadraticCurveTo(0, h, 0, h - r);
    g.lineTo(0, r); g.quadraticCurveTo(0, 0, r, 0); g.fill();
    g.fillStyle = color || '#fff';
    g.textBaseline = 'middle';
    g.fillText(text, 14, h / 2 + 1);
    const tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    sp.scale.set(w / 52 * 4.2, 4.2, 1);
    sp.renderOrder = 20;
    sp.userData.text = text;
    return sp;
  }

  _setLabel(sp, text, color, bg) {
    if (sp.userData.text === text && sp.userData.color === color) return sp;
    const fresh = this._label(text, color, bg);
    sp.material.map.dispose(); sp.material.dispose();
    sp.material = fresh.material;
    sp.scale.copy(fresh.scale);
    sp.userData.text = text; sp.userData.color = color;
    return sp;
  }

  _dispose(obj) {
    obj.traverse((o) => {
      if (o.geometry && !Object.values(this.geo).includes(o.geometry)) o.geometry.dispose();
      if (o.material) {
        const ms = Array.isArray(o.material) ? o.material : [o.material];
        ms.forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); });
      }
    });
  }

  _rand(seed) {                       // deterministic scenery
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  _laneZ(lane) {
    const lanes = this.scene ? this.scene.num_lanes : 5;
    const lw = this.scene ? this.scene.lane_width_m : 3.75;
    const l = Math.min(lanes, Math.max(1, lane || 1));
    return (l - 0.5 - lanes / 2) * lw;
  }
  _roadL() { return this.scene ? this.scene.road_length_m : 3200; }
  _roadX0() { return this.scene ? (this.scene.x_min_m || 0) : 0; }

  // ---------------------------------------------------------------- data
  setScene(scene, rsus) {
    let rebuild = false;
    if (scene) {
      const changed = !this.scene || this.scene.road_length_m !== scene.road_length_m
        || this.scene.mobility_mode !== scene.mobility_mode || this.scene.x_min_m !== scene.x_min_m
        || this.scene.num_lanes !== scene.num_lanes || this.scene.lane_width_m !== scene.lane_width_m;
      this.scene = scene;
      if (changed) { rebuild = true; this.fitToRoad(); }
    }
    if (Array.isArray(rsus)) {
      const sig = JSON.stringify(rsus.map((r) => [r.rsu_id, r.x, r.y, r.z_height, r.coverage_radius]));
      if (sig !== this._rsuSig) { this._rsuSig = sig; rebuild = true; }
      this.rsus = rsus;
    }
    if (rebuild && this.scene) this._buildWorld();
  }

  updateVehicles(list) {
    const now = performance.now();
    const seen = new Set();
    (Array.isArray(list) ? list : []).forEach((v) => {
      seen.add(v.vehicle_id);
      let e = this.vehicles.get(v.vehicle_id);
      const active = v.active !== false;
      if (!e) {
        e = { id: v.vehicle_id, prev: null, cur: null, tPrev: now - 300, tCur: now, group: null };
        this.vehicles.set(v.vehicle_id, e);
      }
      const L = this._roadL();
      const jump = e.cur ? Math.abs(v.x - e.cur.x) : 0;
      e.prev = (e.cur && e.cur.active === active && jump < L * 0.4) ? e.cur : null;
      e.tPrev = e.tCur;
      e.cur = { x: v.x, y: v.y, lane: v.lane, speed: v.speed, active, trust: v.trust_score,
                role: v.role, vclass: v.vclass, queue: v.queue_length, cpu: v.cpu_capacity_ghz,
                rsus: v.connected_rsus || [] };
      e.tCur = now;
    });
    for (const id of Array.from(this.vehicles.keys())) {
      if (!seen.has(id)) {
        const e = this.vehicles.get(id);
        if (e.group) { this.vehGroup.remove(e.group); this._dispose(e.group); }
        this.vehicles.delete(id);
      }
    }
  }

  triggerOffloadingBeam(ev) {
    if (!ev) return;
    this.links.push({
      vid: ev.vid, target: ev.target, action: ev.action, ok: !!ev.success,
      alert: !!ev.detector_alert, sab: !!(ev.v2v && ev.ground_truth_malicious && !ev.success), sabRole: ev.target_role || '', born: performance.now(), line: null, pulse: null, label: null, ring: null,
    });
    if (this.links.length > 80) this._removeLink(this.links.shift());
  }

  showFLRound(viz) {
    if (!viz || viz.round === this._lastFlRound) return;
    this._lastFlRound = viz.round;
    this._clearFL();
    const clients = viz.clients || [];
    this.flAnim = { born: performance.now(), clients, round: viz.round, items: [] };
    const excluded = clients.filter((c) => c.included === false).length;
    const accepted = clients.filter((c) => c.included === true).length;
    const uploaded = clients.filter((c) => c.submitted).length;
    this.flBanner = { born: performance.now(),
      text: `Round ${viz.round}: ${uploaded} uploads · ${accepted} accepted · ${excluded} excluded` };
  }

  resetSession() {
    this._lastFlRound = undefined;
    this._clearFL();
    this.links.forEach((l) => this._removeLink(l));
    this.links = [];
    this.flBanner = null;
  }

  fitToRoad() {
    if (!this.scene) return;
    this.center = this._roadX0() + this._roadL() / 2;
    this.targetCenter = this.center;
    this.manualCenter = false;
    this._applyModeCamera();
  }

  setMode(mode) {
    this.mode = mode === 'overview' ? 'overview' : 'follow';
    this.manualCenter = false;
    this.freeCam = false; this.focusY = 2; this.focusZ = 0;
    this.userCam = false;
    this._applyModeCamera();
  }

  _applyModeCamera() {
    if (this.mode === 'overview') {
      const L = this._roadL();
      this.camGoal = { yaw: -0.18, pitch: 0.8, dist: Math.min(2600, Math.max(260, L * 0.5)) };
      this.targetCenter = this._roadX0() + L / 2;
    } else {
      this.camGoal = { yaw: -0.55, pitch: 0.4, dist: 150 };
    }
  }

  // --------------------------------------------------------------- world
  _buildWorld() {
    while (this.world.children.length) { const o = this.world.children[0]; this.world.remove(o); this._dispose(o); }
    this.rsuObjs.clear();
    const T = THREE;
    const L = this._roadL(), x0 = this._roadX0();
    const lanes = this.scene.num_lanes || 5, lw = this.scene.lane_width_m || 3.75;
    const roadW = lanes * lw;
    const cx = x0 + L / 2;
    const rnd = this._rand(20240611);

    // ground with a faint grid
    const gc = document.createElement('canvas'); gc.width = gc.height = 256;
    const gg = gc.getContext('2d');
    gg.fillStyle = '#2b3c5c'; gg.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 1400; i++) { gg.fillStyle = `rgba(${40 + rnd() * 30 | 0},${60 + rnd() * 30 | 0},${88 + rnd() * 30 | 0},0.35)`; gg.fillRect(rnd() * 256, rnd() * 256, 2, 2); }
    gg.strokeStyle = 'rgba(140,175,230,0.18)'; gg.lineWidth = 2; gg.strokeRect(0, 0, 256, 256);
    const gt = new T.CanvasTexture(gc); gt.wrapS = gt.wrapT = T.RepeatWrapping; gt.anisotropy = 4;
    const gw = L + 3200, gd = 3600;
    gt.repeat.set(gw / 40, gd / 40);
    const ground = new T.Mesh(new T.PlaneGeometry(gw, gd), new T.MeshStandardMaterial({ map: gt, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2; ground.position.set(cx, -0.05, 0); ground.receiveShadow = true;
    this.world.add(ground);

    // asphalt
    const ac = document.createElement('canvas'); ac.width = 256; ac.height = 128;
    const ag = ac.getContext('2d');
    ag.fillStyle = '#3a3e46'; ag.fillRect(0, 0, 256, 128);
    for (let i = 0; i < 3500; i++) { const v = 45 + rnd() * 35 | 0; ag.fillStyle = `rgb(${v},${v + 2},${v + 8})`; ag.fillRect(rnd() * 256, rnd() * 128, 1.6, 1.6); }
    const at = new T.CanvasTexture(ac); at.wrapS = at.wrapT = T.RepeatWrapping; at.anisotropy = 8;
    at.repeat.set(L / 24, 1);
    const road = new T.Mesh(new T.PlaneGeometry(L, roadW), new T.MeshStandardMaterial({ map: at, roughness: 0.92, metalness: 0.02 }));
    road.rotation.x = -Math.PI / 2; road.position.set(cx, 0.02, 0); road.receiveShadow = true;
    this.world.add(road);
    // shoulders
    const shMat = new T.MeshStandardMaterial({ color: 0x4a4f5a, roughness: 1 });
    [-1, 1].forEach((s) => {
      const sh = new T.Mesh(new T.PlaneGeometry(L, 3.2), shMat);
      sh.rotation.x = -Math.PI / 2; sh.position.set(cx, 0.015, s * (roadW / 2 + 1.6)); sh.receiveShadow = true;
      this.world.add(sh);
    });
    // edge lines (solid) and lane dashes (instanced)
    const lineMat = new T.MeshBasicMaterial({ color: 0xf2f4f8 });
    [-1, 1].forEach((s) => {
      const el = new T.Mesh(new T.PlaneGeometry(L, 0.22), lineMat);
      el.rotation.x = -Math.PI / 2; el.position.set(cx, 0.04, s * (roadW / 2 - 0.25));
      this.world.add(el);
    });
    const yel = new T.Mesh(new T.PlaneGeometry(L, 0.18), new T.MeshBasicMaterial({ color: 0xf5c542 }));
    yel.rotation.x = -Math.PI / 2; yel.position.set(cx, 0.045, 0.0);
    // the centre line is the divider between the two lane halves on a highway
    if (lanes % 2 === 0) this.world.add(yel);
    const dashN = Math.floor(L / 12);
    const dash = new T.InstancedMesh(new T.PlaneGeometry(4.2, 0.18), lineMat, dashN * (lanes - 1));
    const m4 = new T.Matrix4(), rot = new T.Matrix4().makeRotationX(-Math.PI / 2);
    let k = 0;
    for (let l = 1; l < lanes; l++) {
      if (lanes % 2 === 0 && l === lanes / 2) continue;
      const z = (l - lanes / 2) * lw;
      for (let i = 0; i < dashN; i++) {
        m4.makeTranslation(x0 + i * 12 + 6, 0.045, z).multiply(rot);
        dash.setMatrixAt(k++, m4);
      }
    }
    dash.count = k; dash.instanceMatrix.needsUpdate = true;
    this.world.add(dash);
    // concrete barriers
    const bar = new T.Mesh(new T.BoxGeometry(L, 0.9, 0.55), new T.MeshStandardMaterial({ color: 0x9aa0a8, roughness: 0.9 }));
    [-1, 1].forEach((s) => {
      const b = bar.clone(); b.position.set(cx, 0.45, s * (roadW / 2 + 3.4)); b.castShadow = true; b.receiveShadow = true;
      this.world.add(b);
    });

    // street lights
    const spacing = 55, n = Math.floor(L / spacing);
    const pole = new T.InstancedMesh(new T.CylinderGeometry(0.16, 0.22, 11, 6), new T.MeshStandardMaterial({ color: 0x8b95a8, metalness: 0.6, roughness: 0.5 }), n * 2);
    const head = new T.InstancedMesh(new T.BoxGeometry(2.6, 0.3, 0.8), new T.MeshBasicMaterial({ color: 0xffe9b0 }), n * 2);
    const glow = new T.InstancedMesh(new T.CircleGeometry(7, 20), new T.MeshBasicMaterial({ color: 0xffd98a, transparent: true, opacity: 0.1, depthWrite: false }), n * 2);
    let q = 0;
    for (let i = 0; i < n; i++) {
      [-1, 1].forEach((s) => {
        const px = x0 + i * spacing + 20, pz = s * (roadW / 2 + 4.4);
        m4.makeTranslation(px, 5.5, pz); pole.setMatrixAt(q, m4);
        m4.makeTranslation(px, 11.1, pz - s * 1.1); head.setMatrixAt(q, m4);
        m4.makeTranslation(px, 0.06, pz - s * 4).multiply(rot); glow.setMatrixAt(q, m4);
        q++;
      });
    }
    pole.castShadow = true;
    this.world.add(pole); this.world.add(head); this.world.add(glow);

    // buildings (window textures lit at random), kept away from the road and the RSUs
    const wc = document.createElement('canvas'); wc.width = 128; wc.height = 128;
    const wg = wc.getContext('2d');
    wg.fillStyle = '#34486f'; wg.fillRect(0, 0, 128, 128);
    for (let gy = 0; gy < 8; gy++) for (let gx = 0; gx < 6; gx++) {
      const on = rnd() < 0.42;
      wg.fillStyle = on ? (rnd() < 0.5 ? '#ffe3a0' : '#bfe2ff') : '#1d2a45';
      wg.fillRect(6 + gx * 20, 6 + gy * 15, 12, 9);
    }
    const wt = new T.CanvasTexture(wc); wt.wrapS = wt.wrapT = T.RepeatWrapping; wt.anisotropy = 4;
    const tints = [0x6f86b8, 0x8396c4, 0x5d7099, 0x9aa9cc, 0x7b8fbd];
    const bMats = tints.map((c) => new T.MeshStandardMaterial({ map: wt, color: c, roughness: 0.75, metalness: 0.1,
      emissive: 0x1a2440, emissiveMap: wt, emissiveIntensity: 0.55 }));
    const rsuXs = (this.rsus || []).map((r) => r.x);
    const nb = Math.min(260, Math.floor(L / 14));
    for (let i = 0; i < nb; i++) {
      const w = 14 + rnd() * 26, d = 14 + rnd() * 26, h = 10 + Math.pow(rnd(), 2) * 85;
      const x = x0 + rnd() * L, side = rnd() < 0.5 ? -1 : 1;
      const z = side * (roadW / 2 + (side > 0 ? 150 : 60) + rnd() * 340);
      if (rsuXs.some((rx) => Math.abs(rx - x) < 40 && Math.abs(z) < 120)) continue;
      const geo = new T.BoxGeometry(w, h, d);
      const uv = geo.attributes.uv;
      for (let f = 0; f < 6; f++) {
        const su = (f < 2 ? d : w) / 22, sv = f === 2 || f === 3 ? 1 : h / 16;
        const su2 = f === 2 || f === 3 ? w / 22 : su, sv2 = f === 2 || f === 3 ? d / 22 : sv;
        for (let v = 0; v < 4; v++) { const idx = f * 4 + v; uv.setXY(idx, uv.getX(idx) * su2, uv.getY(idx) * sv2); }
      }
      const b = new T.Mesh(geo, bMats[i % tints.length]);
      b.position.set(x, h / 2, z); b.receiveShadow = false;
      this.world.add(b);
    }

    // trees
    const trunk = new T.InstancedMesh(new T.CylinderGeometry(0.3, 0.4, 3, 6), new T.MeshStandardMaterial({ color: 0x4b3a2a }), 160);
    const crown = new T.InstancedMesh(new T.ConeGeometry(2.6, 7, 8), new T.MeshStandardMaterial({ color: 0x2c5a45, roughness: 1 }), 160);
    for (let i = 0; i < 160; i++) {
      const x = x0 + rnd() * L, s = rnd() < 0.5 ? -1 : 1;
      const z = s * (roadW / 2 + 12 + rnd() * 40);
      const sc = 0.8 + rnd() * 0.7;
      m4.makeScale(sc, sc, sc).setPosition(x, 1.5 * sc, z); trunk.setMatrixAt(i, m4);
      m4.makeScale(sc, sc, sc).setPosition(x, 6 * sc, z); crown.setMatrixAt(i, m4);
    }
    crown.castShadow = true;
    this.world.add(trunk); this.world.add(crown);

    // gantry sign over the road every kilometre (decoration)
    for (let gx = x0 + 500; gx < x0 + L; gx += 1000) {
      const g = new T.Group();
      const mat = new T.MeshStandardMaterial({ color: 0x7d8799, metalness: 0.5, roughness: 0.5 });
      [-1, 1].forEach((s) => { const p = new T.Mesh(new T.BoxGeometry(0.6, 8, 0.6), mat); p.position.set(0, 4, s * (roadW / 2 + 1.5)); g.add(p); });
      const beam = new T.Mesh(new T.BoxGeometry(0.7, 0.7, roadW + 4), mat); beam.position.set(0, 8, 0); g.add(beam);
      const sign = new T.Mesh(new T.BoxGeometry(0.3, 2.4, 9), new T.MeshStandardMaterial({ color: 0x1d6b48, emissive: 0x0d3a26 }));
      sign.position.set(-0.2, 6.6, -roadW / 6); g.add(sign);
      g.position.x = gx; this.world.add(g);
    }

    // RSU towers (position, height and coverage come from the backend)
    (this.rsus || []).forEach((r) => this._buildRSU(r));
    this._buildMinimapBase();
  }

  _buildRSU(r) {
    const T = THREE;
    const grp = new T.Group();
    const lat = (typeof r.y === 'number') ? r.y : 14;
    const zSign = lat >= 0 ? 1 : -1;
    const roadHalf = ((this.scene.num_lanes || 5) * (this.scene.lane_width_m || 3.75)) / 2 + 8;
    const z = Math.abs(lat) < roadHalf ? zSign * roadHalf : lat;
    grp.position.set(r.x, 0, z);
    const body = new T.Mesh(new T.BoxGeometry(14, 11, 14), new T.MeshStandardMaterial({ color: 0xcfe3f2, roughness: 0.6, metalness: 0.2 }));
    body.position.y = 5.5; body.castShadow = true; body.receiveShadow = true; grp.add(body);
    for (let i = 0; i < 3; i++) {
      const band = new T.Mesh(new T.BoxGeometry(14.2, 0.7, 14.2), new T.MeshBasicMaterial({ color: 0x27d3c8 }));
      band.position.y = 2.2 + i * 3.3; grp.add(band);
    }
    const h = Math.max(14, Math.min(60, r.z_height || 20));
    const mast = new T.Mesh(new T.CylinderGeometry(0.35, 0.6, h, 8), new T.MeshStandardMaterial({ color: 0xaeb8c8, metalness: 0.7, roughness: 0.35 }));
    mast.position.y = 11 + h / 2 - 3; mast.castShadow = true; grp.add(mast);
    const dish = new T.Mesh(new T.SphereGeometry(2.2, 14, 10, 0, Math.PI * 2, 0, Math.PI / 2), new T.MeshStandardMaterial({ color: 0xf2f6fb, side: T.DoubleSide }));
    dish.rotation.x = Math.PI; dish.position.set(0, 11 + h - 4, 0); grp.add(dish);
    const beacon = new T.Mesh(new T.SphereGeometry(0.9, 10, 8), new T.MeshBasicMaterial({ color: 0xff4d4d }));
    beacon.position.y = 11 + h - 1.2; grp.add(beacon);
    const label = this._label(r.rsu_id, '#e8fbff');
    label.position.y = 11 + h + 6; grp.add(label);
    // coverage footprint on the ground (a disc of the reported radius around the RSU's road position)
    const cov = new T.Group();
    const disc = new T.Mesh(new T.CircleGeometry(r.coverage_radius || 300, 72), new T.MeshBasicMaterial({ color: 0x27d3c8, transparent: true, opacity: 0.07, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2; disc.position.set(0, 0.07, 0);
    const edge = new T.Mesh(new T.RingGeometry((r.coverage_radius || 300) - 1.2, (r.coverage_radius || 300), 96), new T.MeshBasicMaterial({ color: 0x27d3c8, transparent: true, opacity: 0.45, depthWrite: false, side: T.DoubleSide }));
    edge.rotation.x = -Math.PI / 2; edge.position.set(0, 0.08, 0);
    cov.add(disc); cov.add(edge);
    cov.position.set(0, 0, -z + (typeof r.y === 'number' ? 0 : 0));   // centred on the road line
    grp.add(cov);
    this.world.add(grp);
    this.rsuObjs.set(r.rsu_id, { group: grp, label, beacon, cov, top: new T.Vector3(r.x, 11 + h, z), data: r });
  }

  // ------------------------------------------------------------ vehicles
  _makeVehicle(vclass, colorHex) {
    const T = THREE;
    const dims = { Car: [4.6, 1.85, 1.45], SUV: [4.9, 2.0, 1.75], Autonomous_EV: [4.7, 1.9, 1.5], Truck: [11.5, 2.5, 3.5] }[vclass] || [4.6, 1.85, 1.45];
    const L = dims[0] * this.SCALE_L, W = dims[1] * this.SCALE_W, H = dims[2] * this.SCALE_H;
    const g = new T.Group();
    const bodyMat = new T.MeshStandardMaterial({ color: colorHex, metalness: 0.55, roughness: 0.38 });
    const wheelR = 0.42 * this.SCALE_H;
    const add = (mesh, sx, sy, sz, px, py, pz) => { mesh.scale.set(sx, sy, sz); mesh.position.set(px, py, pz); mesh.castShadow = true; g.add(mesh); return mesh; };
    if (vclass === 'Truck') {
      add(new T.Mesh(this.geo.box, new T.MeshStandardMaterial({ color: 0xd9dde4, roughness: 0.7 })), L * 0.66, H * 0.78, W, -L * 0.17, wheelR + H * 0.42, 0);
      add(new T.Mesh(this.geo.box, bodyMat), L * 0.26, H * 0.55, W, L * 0.36, wheelR + H * 0.3, 0);
      add(new T.Mesh(this.geo.box, this.mat.glass), L * 0.08, H * 0.26, W * 0.94, L * 0.44, wheelR + H * 0.58, 0);
    } else {
      add(new T.Mesh(this.geo.box, bodyMat), L, H * 0.42, W, 0, wheelR + H * 0.2, 0);
      add(new T.Mesh(this.geo.box, this.mat.glass), L * 0.52, H * 0.34, W * 0.92, -L * 0.04, wheelR + H * 0.58, 0);
      add(new T.Mesh(this.geo.box, bodyMat), L * 0.46, H * 0.06, W * 0.88, -L * 0.04, wheelR + H * 0.77, 0);
      if (vclass === 'Autonomous_EV') {
        add(new T.Mesh(new T.CylinderGeometry(0.3, 0.3, 0.3, 12), this.mat.lidar), 1, 1, 1, -L * 0.04, wheelR + H * 0.9, 0);
      }
    }
    const nAxle = vclass === 'Truck' ? 3 : 2;
    for (let a = 0; a < nAxle; a++) {
      const ax = vclass === 'Truck' ? [-L * 0.42, -L * 0.22, L * 0.3][a] : [-L * 0.3, L * 0.3][a];
      [-1, 1].forEach((s) => {
        const w = new T.Mesh(this.geo.wheel, this.mat.tire);
        w.rotation.x = Math.PI / 2; w.scale.set(wheelR / 0.42, W * 0.16 / 0.3, wheelR / 0.42);
        w.position.set(ax, wheelR, s * (W / 2 - 0.1)); g.add(w);
      });
    }
    [-1, 1].forEach((s) => {
      add(new T.Mesh(this.geo.box, this.mat.head), 0.12, 0.2, W * 0.22, L / 2, wheelR + H * 0.22, s * W * 0.3);
      add(new T.Mesh(this.geo.box, this.mat.tail), 0.12, 0.2, W * 0.22, -L / 2, wheelR + H * 0.24, s * W * 0.3);
    });
    const ring = new T.Mesh(this.geo.ring, new T.MeshBasicMaterial({ color: 0x2ee6a6, transparent: true, opacity: 0.95, side: T.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.09;
    ring.scale.set(L * 0.62, W * 0.95 + 1.2, 1);
    g.add(ring);
    g.userData = { ring, body: bodyMat, L, W, H, vclass };
    return g;
  }

  _vehColor(id) {
    const palette = [0xd8dde6, 0x2d6cdf, 0xc44536, 0x1f2933, 0x6aa84f, 0xf0b429, 0x8e44ad, 0x16a3b5, 0xe8e4d6, 0x7a869a];
    return palette[(id * 7 + 3) % palette.length];
  }

  _trustColor(t) {
    if (t === null || t === undefined) return 0x9aa4b8;
    if (t >= 0.8) return 0x2ee6a6;
    if (t >= 0.6) return 0x22d3ee;
    if (t >= this.detectorTau) return 0xffb020;
    return 0xff3b5c;
  }

  _vehicleX(e, now) {
    if (!e.cur) return 0;
    if (!e.prev) return e.cur.x;
    const interval = Math.min(1500, Math.max(80, e.tCur - e.tPrev));
    const a = Math.min(1, Math.max(0, (now - e.tCur) / interval));
    return e.prev.x + (e.cur.x - e.prev.x) * a;
  }

  _syncVehicles(now) {
    const showTruth = this.showGroundTruth;
    this.vehicles.forEach((e) => {
      if (!e.cur) return;
      if (!e.group) {
        e.group = this._makeVehicle(e.cur.vclass, this._vehColor(e.id));
        e.group.userData.vid = e.id;
        this.vehGroup.add(e.group);
      }
      const g = e.group;
      g.visible = !!e.cur.active;
      if (!g.visible) return;
      const vs = this.mode === 'overview' ? this.overviewScale : 1;
      g.scale.set(vs, vs, vs);
      const x = this._vehicleX(e, now);
      const prevZ = e.prev ? this._laneZ(e.prev.lane) : this._laneZ(e.cur.lane);
      const z = prevZ + (this._laneZ(e.cur.lane) - prevZ) * 0.5;
      g.position.set(x, 0, e.zSmooth === undefined ? (e.zSmooth = z) : (e.zSmooth += (z - e.zSmooth) * 0.12));
      g.userData.ring.material.color.setHex(this._trustColor(e.cur.trust));
      const sel = this.selectedVid === e.id, hov = this.hoverVid === e.id;
      g.userData.ring.material.opacity = sel ? 1 : 0.88;
      const k = sel ? 1.25 : (hov ? 1.12 : 1);
      g.userData.ring.scale.set(g.userData.L * 0.62 * k, g.userData.W * 0.95 * k + 1.2, 1);
      // ground-truth marker (only when the overlay is on)
      if (showTruth && e.cur.role && e.cur.role !== 'honest') {
        if (!e.marker) {
          const m = new THREE.Mesh(this.geo.cone, new THREE.MeshBasicMaterial({ color: 0xff3b5c }));
          m.rotation.x = Math.PI; m.scale.set(1.3, 2.6, 1.3); m.position.y = g.userData.H + 5.5;
          e.marker = m; g.add(m);
          e.markerLabel = this._label(e.cur.role.replace('_', ' '), '#ffc2cc', 'rgba(70,10,24,.85)');
          e.markerLabel.position.y = g.userData.H + 9; g.add(e.markerLabel);
        }
        e.marker.visible = e.markerLabel.visible = true;
      } else if (e.marker) { e.marker.visible = e.markerLabel.visible = false; }
      if (sel) {
        const txt = `Vehicle ${e.id} · trust ${e.cur.trust === undefined || e.cur.trust === null ? 'N/A' : Number(e.cur.trust).toFixed(2)}`;
        if (!e.selLabel) { e.selLabel = this._label(txt, '#ffffff'); e.selLabel.position.y = g.userData.H + 5; g.add(e.selLabel); }
        else this._setLabel(e.selLabel, txt, '#ffffff');
        e.selLabel.visible = true;
      } else if (e.selLabel) e.selLabel.visible = false;
    });
  }

  // --------------------------------------------------------- links, FL
  _vehPos(vid, lift) {
    const e = this.vehicles.get(vid);
    if (!e || !e.group || !e.group.visible) return null;
    return new THREE.Vector3(e.group.position.x, (lift === undefined ? e.group.userData.H + 1.5 : lift), e.group.position.z);
  }

  _resolveTarget(t) {
    if (!t) return null;
    const r = this.rsuObjs.get(t);
    if (r) return r.top.clone();
    const m = /^V?_?(\d+)$/.exec(String(t));
    if (m) return this._vehPos(parseInt(m[1], 10));
    return null;
  }

  /** In the close-up only links whose both ends are in the visible stretch are drawn. */
  _inView(p) { return this.mode === 'overview' || Math.abs(p.x - this.center) < 380; }

  _curvePoints(a, b, n, arc) {
    const mid = a.clone().add(b).multiplyScalar(0.5);
    mid.y += arc;
    const c = new THREE.QuadraticBezierCurve3(a, mid, b);
    return c.getPoints(n);
  }

  _removeLink(l) {
    [l.line, l.pulse, l.label, l.ring].forEach((o) => { if (o) { this.dyn.remove(o); this._dispose(o); } });
  }

  _updateLinks(now) {
    const live = [];
    for (const l of this.links) {
      const life = (l.alert || l.sab) ? 3800 : 1700;
      const age = now - l.born;
      if (age > life) { this._removeLink(l); continue; }
      live.push(l);
      if (!this.showLinks) { if (l.line) l.line.visible = false; if (l.pulse) l.pulse.visible = false; continue; }
      const a = this._vehPos(l.vid), b = this._resolveTarget(l.target);
      if (!a || !b || (!this.freeCam && (!this._inView(a) || !this._inView(b)))) { if (l.line) { l.line.visible = false; l.pulse.visible = false; if (l.ring) l.ring.visible = false; if (l.label) l.label.visible = false; } continue; }
      if (l.ring) l.ring.visible = true;
      if (l.label) l.label.visible = true;
      const isV2V = l.action === 4 || /^V?_?\d+$/.test(String(l.target));
      const color = l.alert ? 0xff2d55 : (l.sab ? 0xe040fb : (!l.ok ? 0x9aa4b8 : (isV2V ? 0xffb020 : 0x2ee6a6)));
      const fade = Math.max(0, 1 - Math.max(0, age - life * 0.55) / (life * 0.45));
      const pts = this._curvePoints(a, b, 28, Math.min(40, a.distanceTo(b) * 0.28 + 6));
      if (!l.line) {
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        l.line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 1, linewidth: 2, depthTest: false }));
        l.line.renderOrder = 10;
        l.pulse = new THREE.Mesh(this.geo.sphere, new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }));
        l.pulse.renderOrder = 11;
        this.dyn.add(l.line); this.dyn.add(l.pulse);
        if (l.alert || l.sab) {
          l.ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.1, 40), new THREE.MeshBasicMaterial({ color: l.sab && !l.alert ? 0xe040fb : 0xff2d55, transparent: true, side: THREE.DoubleSide, depthTest: false }));
          l.ring.rotation.x = -Math.PI / 2; l.ring.renderOrder = 12; this.dyn.add(l.ring);
          l.label = l.alert ? this._label('DETECTOR ALERT', '#ffd5dc', 'rgba(120,10,30,.9)') : this._label('V2V HELPER SABOTAGE' + (l.sabRole ? ' (' + l.sabRole.replace('_', ' ') + ')' : ''), '#f6d3ff', 'rgba(70,10,90,.9)');
          this.dyn.add(l.label);
        }
      }
      const pos = l.line.geometry.attributes.position;
      pts.forEach((p, i) => pos.setXYZ(i, p.x, p.y, p.z));
      pos.needsUpdate = true;
      l.line.geometry.computeBoundingSphere();
      l.line.visible = true;
      l.line.material.color.setHex(color);
      l.line.material.opacity = fade;
      const f = Math.min(1, age / 650);
      const pp = pts[Math.min(pts.length - 1, Math.floor(f * (pts.length - 1)))];
      l.pulse.position.copy(pp); l.pulse.scale.setScalar(l.alert ? 1.7 : 1.1);
      l.pulse.material.color.setHex(color); l.pulse.material.opacity = fade; l.pulse.visible = true;
      if (l.ring) {
        const k = 4 + (age / life) * 22;
        l.ring.position.set(a.x, 0.2, a.z); l.ring.scale.set(k, k, 1);
        l.ring.material.opacity = Math.max(0, 0.9 - age / life);
      }
      if (l.label) { l.label.position.set(a.x, a.y + 9 + Math.min(6, age / 400), a.z); l.label.material.opacity = fade; }
    }
    this.links = live;
  }

  _clearFL() {
    if (this.flAnim && this.flAnim.items) this.flAnim.items.forEach((it) => { [it.line, it.pulse, it.label, it.x].forEach((o) => { if (o) { this.dyn.remove(o); this._dispose(o); } }); });
    this.flAnim = null;
  }

  _updateFL(now) {
    const fl = this.flAnim;
    if (!fl) { this.agg.group.visible = false; return; }
    const age = now - fl.born, life = 4200;
    if (age > life) { this._clearFL(); this.agg.group.visible = false; return; }
    this.agg.group.visible = true;
    this.agg.group.position.set(this.center, 95, -140);
    this.agg.core.rotation.y += 0.04; this.agg.ring.rotation.z += 0.02;
    const fade = Math.max(0, Math.min(1, (life - age) / 900));
    const target = this.agg.group.position.clone(); target.y -= 3;
    if (!fl.items.length) {
      fl.clients.forEach((c) => {
        if (!c.submitted) return;
        const bad = c.included === false;
        const color = bad ? 0xff2d55 : 0x4c8dff;
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
          new THREE.LineBasicMaterial({ color, transparent: true, depthTest: false }));
        line.renderOrder = 9;
        const pulse = new THREE.Mesh(this.geo.sphere, new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }));
        pulse.renderOrder = 11;
        this.dyn.add(line); this.dyn.add(pulse);
        const it = { vid: c.vid, bad, reason: c.reason || '', line, pulse, color };
        if (bad) {
          it.x = this._label('excluded' + (it.reason ? ': ' + it.reason : ''), '#ffd5dc', 'rgba(120,10,30,.92)');
          this.dyn.add(it.x);
        }
        fl.items.push(it);
      });
    }
    fl.items.forEach((it) => {
      const a = this._vehPos(it.vid);
      if (!a || !this._inView(a)) { it.line.visible = it.pulse.visible = false; if (it.x) it.x.visible = false; return; }
      const pts = this._curvePoints(a, target, 26, 30);
      const pos = it.line.geometry.attributes.position;
      if (pos.count !== pts.length) it.line.geometry.setFromPoints(pts);
      else { pts.forEach((p, i) => pos.setXYZ(i, p.x, p.y, p.z)); pos.needsUpdate = true; }
      it.line.geometry.computeBoundingSphere();
      it.line.visible = it.pulse.visible = true;
      it.line.material.opacity = 0.85 * fade; it.pulse.material.opacity = fade;
      const f = (age % 1400) / 1400;
      it.pulse.position.copy(pts[Math.floor(f * (pts.length - 1))]); it.pulse.scale.setScalar(1.3);
      if (it.x) { it.x.visible = true; it.x.position.set(a.x, a.y + 8, a.z); it.x.material.opacity = fade; }
    });
  }

  // ---------------------------------------------------------- camera/loop
  _autoCenter(now) {
    if (this.mode !== 'follow' || this.manualCenter || this.freeCam) return;
    const sel = this.selectedVid !== null ? this.vehicles.get(this.selectedVid) : null;
    if (sel && sel.group && sel.cur && sel.cur.active) {
      this.targetCenter = sel.group.position.x;
    } else if (now - this._lastAuto > 1500) {
      this._lastAuto = now;
      const xs = [];
      this.vehicles.forEach((e) => { if (e.cur && e.cur.active) xs.push(e.cur.x); });
      if (xs.length) {
        xs.sort((a, b) => a - b);
        const W = 420;
        let best = xs[0], bestN = -1;
        for (let i = 0; i < xs.length; i++) {
          let j = i;
          while (j < xs.length && xs[j] - xs[i] <= W) j++;
          if (j - i > bestN) { bestN = j - i; best = (xs[i] + xs[j - 1]) / 2; }
        }
        this.targetCenter = best;
      }
    }
    const L = this._roadL(), x0 = this._roadX0();
    this.targetCenter = Math.min(x0 + L - 80, Math.max(x0 + 80, this.targetCenter));
    if (Math.abs(this.targetCenter - this.center) > 600) this.center = this.targetCenter;
    this.center += (this.targetCenter - this.center) * 0.05;
  }

  // Move the look-at point on the ground plane relative to the viewing direction.
  _pan(right, forward) {
    const yaw = this.camGoal.yaw;
    this.center += right * Math.cos(yaw) - forward * Math.sin(yaw);
    this.focusZ += -right * Math.sin(yaw) - forward * Math.cos(yaw);
    this.targetCenter = this.center;
    this.manualCenter = true; this.freeCam = true; this.userCam = true;
  }

  _flyKeys() {
    if (!this.keys.size || !this.pointerInside) return;
    const sp = Math.max(0.6, this.camGoal.dist * 0.02);
    const k = this.keys;
    let r = 0, f = 0;
    if (k.has('d') || k.has('arrowright')) r += sp;
    if (k.has('a') || k.has('arrowleft')) r -= sp;
    if (k.has('w') || k.has('arrowup')) f += sp;
    if (k.has('s') || k.has('arrowdown')) f -= sp;
    if (r || f) this._pan(r, f);
    if (k.has('e')) { this.focusY += sp; this.freeCam = true; this.userCam = true; }
    if (k.has('q')) { this.focusY -= sp; this.freeCam = true; this.userCam = true; }
  }

  resetCamera() {
    this.focusY = 2; this.focusZ = 0; this.freeCam = false; this.manualCenter = false;
    this.selectedVid = null; this.userCam = false; this._applyModeCamera();
  }

  _updateCamera() {
    const c = this.cam, g = this.camGoal;
    if (!this.userCam) { c.yaw += (g.yaw - c.yaw) * 0.06; c.pitch += (g.pitch - c.pitch) * 0.06; c.dist += (g.dist - c.dist) * 0.06; }
    else { c.yaw = g.yaw; c.pitch = g.pitch; c.dist = g.dist; }
    const cx = this.center, ty = this.focusY, tz = this.focusZ;
    this._flyKeys();
    const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
    this.camera.position.set(cx + Math.sin(c.yaw) * cp * c.dist, ty + sp * c.dist, tz + Math.cos(c.yaw) * cp * c.dist);
    this.camera.lookAt(cx, ty, tz);
    this.sun.position.set(cx - 140, 220, 160); this.sun.target.position.set(cx, 0, 0);
    this.sun.target.updateMatrixWorld();
  }

  resize() {
    const r = this.container.getBoundingClientRect();
    const w = Math.max(240, Math.floor(r.width)), h = Math.max(200, Math.floor(r.height));
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.canvas.style.width = w + 'px'; this.canvas.style.height = h + 'px';
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.mini.width = Math.floor((w - 20) * dpr); this.mini.height = Math.floor(34 * dpr);
  }

  _loop() {
    this._raf = requestAnimationFrame(this._loop);
    if (!this.container.offsetParent) return;           // hidden tab: do no work
    const t = performance.now();
    if (t - (this._lastDraw || 0) < 30) return;         // at most ~33 frames per second
    this._lastDraw = t;
    this.draw();
    this._adapt(performance.now() - t);
  }

  /** Lower the render cost when frames are slow (integrated GPU, software rendering). */
  _adapt(ms) {
    this._ft = (this._ft === undefined) ? ms : this._ft * 0.9 + ms * 0.1;
    this._nFrames = (this._nFrames || 0) + 1;
    if (this._nFrames < 25 || this._nFrames % 25 !== 0) return;
    if (this._ft > 45 && this.quality === undefined) {            // stage 1: no shadows
      this.quality = 1; this.sun.castShadow = false; this.renderer.shadowMap.enabled = false;
      this.three.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
    } else if (this._ft > 45 && this.quality === 1) {             // stage 2: lower resolution
      this.quality = 2; this.renderer.setPixelRatio(0.7); this.resize();
    } else if (this._ft > 80 && this.quality === 2) {
      this.slowNote = true; this._hudKey = null;
    }
  }

  draw() {
    const now = performance.now();
    if (this.w === undefined) this.resize();
    this._autoCenter(now);
    this._updateCamera();
    this._syncVehicles(now);
    this._updateLinks(now);
    this._updateFL(now);
    // RSU coverage and labels
    this.rsuObjs.forEach((o, id) => {
      const d = (this.rsus || []).find((q) => q.rsu_id === id) || o.data;
      o.cov.visible = !!this.showCoverage;
      const util = d.utilization !== undefined ? Math.round(d.utilization * 100) + '%' : 'N/A';
      const tr = d.trust_score !== undefined && d.trust_score !== null ? Number(d.trust_score).toFixed(2) : 'N/A';
      const txt = `${id} · load ${util} · trust ${tr}`;
      this._setLabel(o.label, txt, d.trust_score !== undefined && d.trust_score < this.detectorTau ? '#ffc2cc' : '#e8fbff');
      o.beacon.material.color.setHex(d.trust_score !== undefined && d.trust_score < this.detectorTau ? 0xff3b5c : 0x27d3c8);
    });
    this.renderer.render(this.three, this.camera);
    this._drawHud(now);
    this._drawMini();
  }

  _drawHud(now) {
    const modeTxt = this.mode === 'overview' ? 'Whole road' : 'Close-up';
    const key = modeTxt + '|' + (this.flBanner ? this.flBanner.text : '') + '|' + (this.flBanner && now - this.flBanner.born < 6000);
    if (key !== this._hudKey) {
      this._hudKey = key;
      this.hudLeft.textContent = (this.slowNote ? 'Rendering is slow on this device - switch to 2D for a lighter view. ' : '') + `3D · ${modeTxt} · vehicles drawn ${(this.SCALE_L * (this.mode === 'overview' ? this.overviewScale : 1)).toFixed(1)}× longer for legibility · drag = orbit · right-drag / shift-drag = pan · WASD/arrows move, Q/E down/up · wheel = zoom · R / double-click = reset`;
      this.hudRight.innerHTML = '';
      if (this.flBanner && now - this.flBanner.born < 6000) {
        const b = document.createElement('span');
        b.textContent = this.flBanner.text;
        b.style.cssText = 'background:rgba(30,60,140,.85);padding:3px 9px;border-radius:4px;font-weight:600';
        this.hudRight.appendChild(b);
      }
    }
  }

  _buildMinimapBase() { this._miniDirty = true; }

  _drawMini() {
    const c = this.mini, g = c.getContext('2d');
    if (!g || !this.scene) return;
    const w = c.width, h = c.height, pad = 8;
    g.clearRect(0, 0, w, h);
    const L = this._roadL(), x0 = this._roadX0();
    const X = (x) => pad + (x - x0) / L * (w - 2 * pad);
    g.fillStyle = 'rgba(70,80,100,.9)'; g.fillRect(pad, h * 0.42, w - 2 * pad, h * 0.16);
    this.rsuObjs.forEach((o) => {
      const r = o.data.coverage_radius || 300;
      g.fillStyle = 'rgba(39,211,200,.22)'; g.fillRect(X(o.data.x - r), h * 0.3, X(o.data.x + r) - X(o.data.x - r), h * 0.4);
      g.fillStyle = '#27d3c8'; g.fillRect(X(o.data.x) - 2, h * 0.2, 4, h * 0.6);
    });
    this.vehicles.forEach((e) => {
      if (!e.cur || !e.cur.active) return;
      g.fillStyle = '#' + this._trustColor(e.cur.trust).toString(16).padStart(6, '0');
      g.fillRect(X(e.group ? e.group.position.x : e.cur.x) - 1.5, h * 0.44 + ((e.id * 7) % 5) * h * 0.03, 3, 3);
    });
    const ww = 420;
    g.strokeStyle = '#fff'; g.lineWidth = 1.5;
    g.strokeRect(X(this.center - ww / 2), h * 0.12, X(this.center + ww / 2) - X(this.center - ww / 2), h * 0.76);
  }

  // -------------------------------------------------------------- events
  _bindEvents() {
    window.addEventListener('resize', () => this.resize());
    if (typeof ResizeObserver === 'function') {
      this._resizeObserver = new ResizeObserver(() => this.resize());
      this._resizeObserver.observe(this.container);
    }
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.userCam = true;
      const k = e.deltaY < 0 ? 0.88 : 1.14;
      this.camGoal.dist = Math.min(9000, Math.max(4, this.camGoal.dist * k));
    }, { passive: false });
    let drag = null;
    this.canvas.setAttribute('tabindex', '0');
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.canvas.addEventListener('mouseenter', () => { this.pointerInside = true; });
    this.canvas.addEventListener('mouseleave', () => { this.pointerInside = false; this.keys.clear(); });
    window.addEventListener('keydown', (e) => {
      if (!this.pointerInside || e.ctrlKey || e.metaKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if ('wasdqe'.includes(key) && key.length === 1 || key.startsWith('arrow')) { this.keys.add(key); e.preventDefault(); }
      else if (key === 'r') this.resetCamera();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    this.canvas.addEventListener('mousedown', (e) => { this.canvas.focus({ preventScroll: true }); drag = { x: e.clientX, y: e.clientY, yaw: this.camGoal.yaw, pitch: this.camGoal.pitch, moved: false, pan: e.button === 2 || e.button === 1 || e.shiftKey, lx: e.clientX, ly: e.clientY }; });
    window.addEventListener('mouseup', (e) => { if (drag && !drag.moved) this._click(e); drag = null; });
    window.addEventListener('mousemove', (e) => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
        if (drag.moved && drag.pan) {
          const k = Math.max(0.05, this.camGoal.dist * 0.0016);
          this._pan(-(e.clientX - drag.lx) * k, (e.clientY - drag.ly) * k);
        } else if (drag.moved) {
          this.userCam = true;
          this.camGoal.yaw = drag.yaw - dx * 0.006;
          this.camGoal.pitch = Math.min(1.52, Math.max(-1.2, drag.pitch + dy * 0.005));
        }
        drag.lx = e.clientX; drag.ly = e.clientY;
        this.tip.style.display = 'none';
        return;
      }
      this._hover(e);
    });
    this.canvas.addEventListener('mouseleave', () => { this.hoverVid = null; this.tip.style.display = 'none'; });
    this.canvas.addEventListener('dblclick', () => { this.resetCamera(); if (typeof this.onSelect === 'function') this.onSelect(null); });
    this.mini.addEventListener('click', (e) => {
      const r = this.mini.getBoundingClientRect();
      const f = (e.clientX - r.left - 8) / (r.width - 16);
      const L = this._roadL();
      this.center = this.targetCenter = this._roadX0() + Math.min(1, Math.max(0, f)) * L;
      this.manualCenter = true; this.selectedVid = null; this.mode = 'follow'; this.focusZ = 0; this.userCam = false; this._applyModeCamera();
    });
  }

  _pick(e) {
    const rect = this.canvas.getBoundingClientRect();
    this.mouse.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const groups = [];
    this.vehicles.forEach((v) => { if (v.group && v.group.visible) groups.push(v.group); });
    const hits = this.raycaster.intersectObjects(groups, true);
    for (const h of hits) {
      let o = h.object;
      while (o && o.userData.vid === undefined) o = o.parent;
      if (o && o.userData.vid !== undefined) return o.userData.vid;
    }
    return null;
  }

  _hover(e) {
    const rect = this.canvas.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) { this.tip.style.display = 'none'; return; }
    const vid = this._pick(e);
    this.hoverVid = vid;
    this.canvas.style.cursor = vid !== null ? 'pointer' : 'grab';
    if (vid === null) { this.tip.style.display = 'none'; return; }
    const v = this.vehicles.get(vid).cur;
    const t = v.trust === undefined || v.trust === null ? 'N/A' : Number(v.trust).toFixed(2);
    this.tip.textContent = `Vehicle ${vid} · ${String(v.vclass || '').replace('_', ' ')} · trust ${t} · ${Math.round((v.speed || 0) * 3.6)} km/h`;
    const wr = this.container.getBoundingClientRect();
    this.tip.style.left = Math.min(wr.width - 220, e.clientX - wr.left + 12) + 'px';
    this.tip.style.top = (e.clientY - wr.top + 12) + 'px';
    this.tip.style.display = 'block';
  }

  _click(e) {
    const rect = this.canvas.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;
    const vid = this._pick(e);
    this.selectedVid = vid;
    if (vid !== null && !this.freeCam) { this.manualCenter = false; this.mode = 'follow'; this.userCam = false; this._applyModeCamera(); }
    if (typeof this.onSelect === 'function') this.onSelect(this.selectedVid);
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._resizeObserver) this._resizeObserver.disconnect();
    this.renderer.dispose();
  }
}

window.VANETRoadView3D = VANETRoadView3D;
