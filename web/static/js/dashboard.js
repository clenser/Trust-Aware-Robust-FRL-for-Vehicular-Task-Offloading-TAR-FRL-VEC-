/**
 * Live console for VANNET.
 *
 * Rules this file follows:
 *
 * 1. The road viewport and the charts are enhancements.  A failure to
 *    initialise either one must never prevent the controls, the telemetry
 *    subscription or the federated-learning monitor from working.
 * 2. The authoritative simulation state always comes from the backend.  A
 *    missing field is never interpreted as "running".
 * 3. API failures are shown to the user and the control buttons are restored
 *    to the state the backend actually reports.
 * 4. Every rendered value comes from the backend snapshot.  When the backend
 *    reports `null` the UI shows "N/A" instead of a placeholder number.
 * 5. The page only subscribes to the central simulation; it never advances it.
 */

let roadView = null;
let latencyChart = null;
let trustChart = null;
let socket = null;
let lastVehicles = [];
let lastSessionSeen;
let completionChart = null;
let actionChart = null;
let lossChart = null;
let trustLinesChart = null;
let flCache = null;                       // last heavy federated block (sent only when it changes)
const SERIES_MAX = 360;
let SER = { steps: [], winC: [], cumC: [], winL: [], last: -1, trust: [], trustSteps: [] };
let ROUND_SER = [];
let roleMap = {};
const ACTION_LABELS = ['Local', 'RSU 1', 'RSU 2', 'RSU 3', 'V2V'];

function resetSeries() {
  SER = { steps: [], winC: [], cumC: [], winL: [], last: -1, trust: [], trustSteps: [] };
  ROUND_SER = [];
}

/** Back-fill the charts for a page that connects mid-session (one request). */
async function backfillSeries() {
  try {
    const res = await fetch('/api/series', { cache: 'no-store' });
    if (!res.ok) return;
    const d = await res.json();
    resetSeries();
    (d.series || []).forEach((pt) => pushPoint(pt, false));
    ROUND_SER = d.round_series || [];
    roleMap = d.roles || {};
    (d.trust_series || []).forEach((t) => { SER.trustSteps.push(t.step); SER.trust.push(t.trust); });
    redrawSeriesCharts();
  } catch (e) { /* the live stream fills the charts anyway */ }
}

async function refreshRoundSeries() {
  try {
    const res = await fetch('/api/series', { cache: 'no-store' });
    if (!res.ok) return;
    ROUND_SER = (await res.json()).round_series || [];
    redrawSeriesCharts();
  } catch (e) { /* ignore */ }
}   // session id of the previous snapshot (undefined before the first)

/** Authoritative simulation state, mirrored from the backend. */
let backendState = {
  state: 'STOPPED',
  isRunning: false,
  sessionId: null,
  flStage: 'idle',
  flRound: 0,
};

/** True while an error message must survive incoming telemetry frames. */
let bannerSticky = false;

const STAGE_LABELS = {
  idle: 'idle',
  waiting_experience: 'waiting for experience',
  training: 'local training',
  aggregating: 'aggregating',
  updating: 'distributing model',
  complete: 'complete',
  error: 'error',
};

const STRATEGY_LABELS = {
  trust_aware: 'trust-aware',
  fedavg: 'FedAvg',
  coordinate_median: 'coordinate median',
  trimmed_mean: 'trimmed mean',
  multi_krum: 'Multi-Krum',
  exclusion_no_weighting: 'exclusion only',
  trust_no_exclusion: 'weighting only',
};

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();

  try {
    if (typeof VANETRoadView === 'function') {
      roadView = new VANETRoadView('viewport');
      roadView.onSelect = (vid) => showInspector(vid);
    } else {
      showDegradation('The road viewport script did not load. Controls and the federated-learning monitor remain functional.');
    }
  } catch (err) {
    roadView = null;
    showDegradation('The road viewport could not start: ' + (err && err.message ? err.message : err) +
                    '. Controls and the federated-learning monitor remain functional.');
  }

  try {
    initCharts();
  } catch (err) {
    latencyChart = null;
    trustChart = null;
    showDegradation('Charts unavailable: ' + (err && err.message ? err.message : err) +
                    '. Live values are still shown in the panels.');
  }

  try {
    connectWebSocket();
  } catch (err) {
    showBanner('Telemetry connection failed: ' + (err && err.message ? err.message : err) +
               '. Retrying automatically.', 'error');
  }

  try {
    setupEventListeners();
  } catch (err) {
    showBanner('Control wiring failed: ' + (err && err.message ? err.message : err), 'error');
  }

  fetchInitialState().then(backfillSeries);
});

/* -------------------------------------------------------------------- */
/* Utilities                                                             */
/* -------------------------------------------------------------------- */

function fmt(value, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  return `${value}${suffix}`;
}

function fmtMs(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  return `${Math.round(value)} ms`;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function cssVar(name) {
  try {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  } catch (e) {
    return '';   // no style engine (headless test harness): callers have fallbacks
  }
}

function escapeHtml(text) {
  return String(text === null || text === undefined ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function showBanner(message, level = 'error', sticky = false) {
  const el = document.getElementById('error-banner');
  if (!el) {
    console.error('[console]', message);
    return;
  }
  el.textContent = message;
  el.className = 'banner' + (level === 'warn' ? ' warn' : '');
  el.style.display = 'block';
  if (sticky) bannerSticky = true;
}

/** Persistent capability notice (viewport or charts unavailable). */
function showDegradation(message) {
  const el = document.getElementById('degradation-banner');
  if (!el) {
    showBanner(message, 'warn', true);
    return;
  }
  el.textContent = message;
  el.style.display = 'block';
}

function clearBanner(force = false) {
  if (bannerSticky && !force) return;
  bannerSticky = false;
  const el = document.getElementById('error-banner');
  if (el) {
    el.style.display = 'none';
    el.textContent = '';
  }
}

/* -------------------------------------------------------------------- */
/* Theme and tabs                                                        */
/* -------------------------------------------------------------------- */

function initTheme() {
  try {
    const saved = window.localStorage.getItem('vannet-theme');
    if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-theme', saved);
  } catch (e) { /* storage may be unavailable; the OS preference applies */ }
  const btn = document.getElementById('btn-theme');
  if (btn) {
    btn.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme');
      const dark = cur ? cur === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
      const next = dark ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { window.localStorage.setItem('vannet-theme', next); } catch (e) { /* ignore */ }
      refreshChartColors();
    });
  }
}

function initTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => {
        const on = t === tab;
        t.classList.toggle('active', on);
        t.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      document.querySelectorAll('.view').forEach((v) => { v.hidden = (v.id !== tab.dataset.view); });
      if (tab.dataset.view === 'view-research' && typeof loadResearch === 'function') loadResearch();
      if (tab.dataset.view === 'view-sim' && roadView) roadView.resize();
    });
  });
}

/* -------------------------------------------------------------------- */
/* State application                                                     */
/* -------------------------------------------------------------------- */

async function fetchInitialState() {
  try {
    const res = await fetch('/api/state', { cache: 'no-store' });
    if (!res.ok) {
      showBanner(`Could not read the simulation state (HTTP ${res.status}).`, 'error');
      return;
    }
    applyState(await res.json());
  } catch (e) {
    showBanner('Could not reach the simulation backend: ' + (e && e.message ? e.message : e), 'error');
  }
}

function chartColors() {
  return {
    line: cssVar('--accent') || '#1f5fbf',
    grid: cssVar('--border') || '#dfe2e7',
    tick: cssVar('--text-muted') || '#5d6877',
    ok: cssVar('--ok') || '#17803d',
    grey: cssVar('--text-muted') || '#6b7380',
    neutral: cssVar('--vehicle') || '#2f4a6d',
    warn: cssVar('--warn') || '#b26a00',
    bad: cssVar('--bad') || '#c0392b',
  };
}

function initCharts() {
  const col = chartColors();
  const scaleBase = (title, extra) => Object.assign({
    grid: { color: col.grid }, ticks: { color: col.tick, font: { size: 10 } },
    title: { display: !!title, text: title, color: col.tick, font: { size: 10 } } }, extra || {});
  const common = { responsive: true, maintainAspectRatio: false, animation: false,
                   plugins: { legend: { display: false } } };

  completionChart = new Chart(document.getElementById('chart-completion').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Last 200 tasks', data: [], borderColor: col.line, borderWidth: 1.8, pointRadius: 0, tension: 0.2, spanGaps: true },
      { label: 'Cumulative', data: [], borderColor: col.grey, borderWidth: 1.2, borderDash: [4, 3], pointRadius: 0, tension: 0.2, spanGaps: true }] },
    options: Object.assign({}, common, {
      plugins: { legend: { display: true, labels: { color: col.tick, boxWidth: 14, font: { size: 10 } } } },
      scales: { x: { display: false }, y: scaleBase('%', { min: 0, max: 100 }) } }) });

  latencyChart = new Chart(document.getElementById('latency-chart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Latency of completed tasks (ms)', data: [], borderColor: col.ok, borderWidth: 1.6,
                                    pointRadius: 0, tension: 0.2, spanGaps: true }] },
    options: Object.assign({}, common, { scales: { x: { display: false }, y: scaleBase('ms', { beginAtZero: true }) } }) });

  actionChart = new Chart(document.getElementById('chart-actions').getContext('2d'), {
    type: 'bar',
    data: { labels: ACTION_LABELS, datasets: [{ data: [0, 0, 0, 0, 0],
      backgroundColor: [col.neutral, col.line, col.line, col.line, col.warn], borderWidth: 0 }] },
    options: Object.assign({}, common, { scales: { x: { grid: { display: false }, ticks: { color: col.tick, font: { size: 10 } } },
                                                  y: scaleBase('% of tasks', { min: 0, max: 100 }) } }) });

  lossChart = new Chart(document.getElementById('chart-loss').getContext('2d'), {
    data: { labels: [], datasets: [
      { type: 'line', label: 'Mean training loss', data: [], borderColor: col.line, borderWidth: 1.6, pointRadius: 2, yAxisID: 'y', spanGaps: true },
      { type: 'bar', label: 'Excluded clients', data: [], backgroundColor: col.bad, yAxisID: 'y2' }] },
    options: Object.assign({}, common, {
      plugins: { legend: { display: true, labels: { color: col.tick, boxWidth: 12, font: { size: 10 } } } },
      scales: { x: { ticks: { color: col.tick, font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                y: scaleBase('loss', { beginAtZero: true }),
                y2: { position: 'right', beginAtZero: true, grid: { display: false },
                      ticks: { color: col.tick, font: { size: 10 }, precision: 0 },
                      title: { display: true, text: 'excluded', color: col.tick, font: { size: 10 } } } } }) });

  trustLinesChart = new Chart(document.getElementById('chart-trust-lines').getContext('2d'), {
    type: 'line', data: { labels: [], datasets: [] },
    options: Object.assign({}, common, { scales: { x: { display: false }, y: scaleBase('joint trust', { min: 0, max: 1 }) } }) });

  const ctxTrust = document.getElementById('trust-chart').getContext('2d');
  trustChart = new Chart(ctxTrust, {
    type: 'bar',
    data: {
      labels: ['\u2265 0.8', '0.6\u20130.8', '0.4\u20130.6', '< 0.4'],
      datasets: [{ data: [0, 0, 0, 0],
                   backgroundColor: [col.ok, col.neutral, col.warn, col.bad], borderWidth: 0 }],
    },
    options: Object.assign({}, common, { scales: {
      x: { grid: { display: false }, ticks: { color: col.tick, font: { size: 10 } } },
      y: scaleBase('vehicles', { beginAtZero: true, ticks: { color: col.tick, font: { size: 10 }, precision: 0 } }) } }),
  });
}

function pushPoint(pt, redraw = true) {
  if (!pt || pt.step <= SER.last) return;
  SER.last = pt.step;
  SER.steps.push(pt.step); SER.winC.push(pt.win_completion); SER.cumC.push(pt.cum_completion); SER.winL.push(pt.win_latency);
  if (SER.steps.length > SERIES_MAX) { SER.steps.shift(); SER.winC.shift(); SER.cumC.shift(); SER.winL.shift(); }
  if (redraw) redrawSeriesCharts();
}

function trustColor(v) {
  const col = chartColors();
  return v < 0.4 ? col.bad : (v < 0.6 ? col.warn : col.line);
}

function redrawSeriesCharts() {
  const col = chartColors();
  if (completionChart) {
    completionChart.data.labels = SER.steps;
    completionChart.data.datasets[0].data = SER.winC;
    completionChart.data.datasets[1].data = SER.cumC;
    completionChart.update('none');
  }
  if (latencyChart) {
    latencyChart.data.labels = SER.steps;
    latencyChart.data.datasets[0].data = SER.winL;
    latencyChart.update('none');
  }
  if (lossChart) {
    lossChart.data.labels = ROUND_SER.map((r) => 'R' + r.round);
    lossChart.data.datasets[0].data = ROUND_SER.map((r) => r.loss);
    lossChart.data.datasets[1].data = ROUND_SER.map((r) => r.excluded);
    lossChart.update('none');
  }
  if (trustLinesChart && SER.trust.length) {
    const ids = Object.keys(SER.trust[SER.trust.length - 1] || {});
    const lastT = SER.trust[SER.trust.length - 1];
    trustLinesChart.data.labels = SER.trustSteps;
    trustLinesChart.data.datasets = ids.map((id) => ({
      label: 'V' + id, data: SER.trust.map((t) => t[id]), borderWidth: 1, pointRadius: 0, tension: 0.2,
      borderColor: trustColor(lastT[id]), spanGaps: true })).concat([{
      label: 'tau', data: SER.trustSteps.map(() => Number(document.getElementById('slider-tau').value)),
      borderColor: col.tick, borderDash: [5, 4], borderWidth: 1.2, pointRadius: 0 }]);
    trustLinesChart.update('none');
  }
}

function refreshChartColors() {
  const col = chartColors();
  if (latencyChart) {
    latencyChart.data.datasets[0].borderColor = col.line;
    latencyChart.options.scales.y.grid.color = col.grid;
    latencyChart.options.scales.y.ticks.color = col.tick;
    latencyChart.update('none');
  }
  if (completionChart) {
    completionChart.data.datasets[0].borderColor = col.line; completionChart.data.datasets[1].borderColor = col.grey;
    completionChart.update('none');
  }
  redrawSeriesCharts();
  if (trustChart) {
    trustChart.data.datasets[0].backgroundColor = [col.ok, col.neutral, col.warn, col.bad];
    trustChart.options.scales.y.grid.color = col.grid;
    trustChart.options.scales.y.ticks.color = col.tick;
    trustChart.options.scales.x.ticks.color = col.tick;
    trustChart.update('none');
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    setText('connection-status', 'connected');
  };

  socket.onmessage = (event) => {
    try {
      applyState(JSON.parse(event.data));
    } catch (e) {
      console.error('Malformed telemetry payload', e);
    }
  };

  socket.onerror = () => {
    showBanner('Telemetry stream error. Retrying automatically.', 'error');
  };

  socket.onclose = () => {
    setText('connection-status', 'reconnecting');
    // Only the subscriber reconnects; the backend keeps its single loop.
    setTimeout(connectWebSocket, 2000);
  };
}

/** Apply a backend state snapshot to the whole console. */
function applyState(data) {
  if (!data) return;

  // The heavy federated block is sent only when it changes; keep the last one.
  if (data.fl_rounds !== undefined) {
    flCache = { fl_rounds: data.fl_rounds, current_round: data.current_round, last_round: data.last_round,
                fl_readiness: data.fl_readiness, client_participation: data.client_participation,
                fl_version: data.fl_version };
  } else if (flCache && data.vehicles !== undefined && data.vehicles.length) {
    data = Object.assign({}, data, flCache);
  }

  const k = (data.kpis && typeof data.kpis === 'object') ? data.kpis : {};

  backendState.isRunning = (typeof k.is_running === 'boolean') ? k.is_running : false;
  backendState.state = k.sim_state || (backendState.isRunning ? 'RUNNING' : 'STOPPED');
  backendState.sessionId = k.session_id || null;

  if (k.last_error) {
    showBanner('Simulation error: ' + k.last_error, 'error');
  } else if (backendState.state !== 'ERROR') {
    clearBanner();
  }

  // A different session id (new experiment, reset) starts empty views: rows and
  // curve points of the previous session must not look like part of this one.
  const sid = backendState.sessionId;
  if (data.vehicles !== undefined && sid !== lastSessionSeen) {
    lastSessionSeen = sid;
    const feed = document.getElementById('event-feed');
    if (feed) {
      delete feed.dataset.hasRows;
      feed.innerHTML = '<div class="event-item muted">No offloading events in this session yet.</div>';
    }
    resetSeries();
    redrawSeriesCharts();
    if (roadView) { roadView._lastFlRound = undefined; roadView.flAnim = null; roadView.links = []; }
    if (sid) backfillSeries();
  }

  updateSimulationStatus(data);
  updateRoadView(data);
  updateCharts(data);
  updateFLPanels(data);
  updateClientParticipationDisplay(data);
  updateFLRoundsDisplay(data);
  updateEventFeed(data);
}

function updateRoadView(data) {
  if (!roadView) return;
  try {
    if (data.scene) roadView.setScene(data.scene, data.rsus);
    else if (data.rsus && data.rsus.length) roadView.setScene(null, data.rsus);
    if (data.vehicles) {
      lastVehicles = data.vehicles;
      roadView.updateVehicles(data.vehicles);
      if (roadView.selectedVid !== null) showInspector(roadView.selectedVid);
    }
    roadView.detectorTau = Number(document.getElementById('slider-tau').value);
    if (data.events) {
      data.events.forEach((ev) => {
        if (ev.action > 0) roadView.triggerOffloadingBeam(ev);
      });
    }
    if (data.fl_viz && data.fl_viz.round !== undefined) {
      const isNew = roadView._lastFlRound !== data.fl_viz.round;
      roadView.showFLRound(data.fl_viz);
      if (isNew) refreshRoundSeries();
    }
  } catch (err) {
    console.error('Viewport update failed', err);
  }
}

function showInspector(vid) {
  const el = document.getElementById('inspector');
  if (!el) return;
  const v = lastVehicles.find((x) => x.vehicle_id === vid);
  if (!v) {
    el.textContent = 'Click a vehicle to inspect it.';
    return;
  }
  const src = (v.source_vehicle_id !== null && v.source_vehicle_id !== undefined)
    ? ` · source trace id ${v.source_vehicle_id}` : '';
  el.textContent = `V${v.vehicle_id} · x ${v.x} m · lane ${v.lane} · ${v.speed} m/s · ` +
    `${v.cpu_capacity_ghz} GHz · queue ${v.queue_length} · joint trust ${v.trust_score} · ` +
    `RSUs in range: ${(v.connected_rsus && v.connected_rsus.length) ? v.connected_rsus.join(', ') : 'none'}${src}`;
}

function updateSimulationStatus(data) {
  const k = data.kpis || {};

  setText('kpi-latency', (k.avg_latency_ms === null || k.avg_latency_ms === undefined)
    ? 'N/A' : `${k.avg_latency_ms} ms`);
  const recent = (k.completion_rate_window !== undefined && k.completion_rate_window !== null)
    ? k.completion_rate_window : k.completion_rate;
  setText('kpi-completion', (recent === null || recent === undefined) ? 'N/A' : `${recent}%`);
  setText('kpi-completion-cum', (k.completion_rate === null || k.completion_rate === undefined)
    ? 'cumulative N/A' : `cumulative ${k.completion_rate}%`);
  if (k.mean_epsilon !== undefined && k.mean_epsilon !== null) {
    setText('fl-explore', k.exploring
      ? `Exploring: random actions are still taken (epsilon ${k.mean_epsilon}); exploration stops after round 5, so recent completion and latency improve as the policy settles.`
      : `Exploration is off (epsilon ${k.mean_epsilon}); the policy acts greedily and keeps learning from federated rounds.`);
    setText('explore-note', k.exploring ? 'Exploration phase: includes random actions' : 'Greedy policy (exploration off)');
  }
  setText('kpi-round', `#${k.fl_round || 0}`);
  setText('kpi-attackers', `${k.attackers ?? 0} / ${k.active_vehicles ?? 0}`);

  const simState = k.sim_state || (backendState.isRunning ? 'RUNNING' : 'STOPPED');
  const stateEl = document.getElementById('kpi-simstate');
  if (stateEl) {
    stateEl.textContent = simState.toUpperCase();
    stateEl.className = 'status ' + (simState === 'RUNNING' ? 'running'
      : simState === 'ERROR' ? 'error' : simState === 'PAUSED' ? 'paused' : 'muted');
  }

  const startBtn = document.getElementById('btn-start');
  const pauseBtn = document.getElementById('btn-pause');
  const stopBtn = document.getElementById('btn-stop');
  if (startBtn) {
    startBtn.disabled = !(simState === 'STOPPED' || simState === 'PAUSED');
    startBtn.textContent = (simState === 'PAUSED') ? 'Resume' : 'Start';
  }
  if (pauseBtn) pauseBtn.disabled = (simState !== 'RUNNING');
  if (stopBtn) stopBtn.disabled = !(simState === 'RUNNING' || simState === 'PAUSED' || simState === 'ERROR');

  setText('session-id', k.session_id || 'no session');
  setText('chip-session', 'session: ' + (k.session_id || 'none'));
  setText('session-started', k.session_started_at_iso
    ? new Date(k.session_started_at_iso).toLocaleTimeString() : '—');
  setText('session-steps', `${k.sim_steps ?? 0} steps`);

  const mobilitySel = document.getElementById('select-mobility');
  if (mobilitySel && k.mobility_mode && mobilitySel.value !== k.mobility_mode &&
      document.activeElement !== mobilitySel) {
    mobilitySel.value = k.mobility_mode;
  }
  const src = k.mobility_source;
  const srcEl = document.getElementById('mobility-source');
  if (srcEl && src) {
    srcEl.textContent = `Active: ${src.label}. ${src.note}`;
  }
  const chipMob = document.getElementById('chip-mobility');
  if (chipMob && src) {
    chipMob.textContent = 'mobility: ' + (src.authentic ? 'authentic NGSIM replay'
      : (src.mode === 'highway' ? 'simulated (IDM)' : 'synthetic trace'));
    chipMob.className = 'chip ' + (src.authentic ? 'authentic' : 'simulated');
  }
  const chipStrat = document.getElementById('chip-strategy');
  if (chipStrat && k.strategy) chipStrat.textContent = 'strategy: ' + (STRATEGY_LABELS[k.strategy] || k.strategy);

  // Keep the experiment form in step with the backend's real configuration
  // (never overwrite a field the user is editing).
  const syncField = (id, value) => {
    const el = document.getElementById(id);
    if (el && value !== undefined && value !== null && document.activeElement !== el) el.value = value;
  };
  syncField('select-speed', k.sim_speed);
  syncField('input-vehicles', k.configured_vehicles);
  syncField('input-attack-ratio', k.configured_attack_ratio);
  if (data.scene && data.scene.mobility_mode === 'highway' && data.scene.rsus) {
    data.scene.rsus.forEach((r, i) => syncField('input-rsu-' + (i + 1), Math.round(r.x)));
  }

  const stratSel = document.getElementById('select-strategy');
  if (stratSel && k.strategy && stratSel.value !== k.strategy && document.activeElement !== stratSel) {
    stratSel.value = k.strategy;
  }

  setText('m-completed', fmt(k.completed_tasks ?? 0));
  setText('m-failed', fmt(k.failed_tasks ?? 0));
  setText('m-dropped', fmt(k.dropped_tasks ?? 0));
  setText('m-latency', fmtMs(k.avg_latency_ms));
  setText('m-simtime', `${(k.simulation_time_sec ?? 0).toFixed(1)} s`);
}

function updateCharts(data) {
  if (data.point) pushPoint(data.point);
  const k = data.kpis || {};
  if (actionChart && k.action_mix) {
    const tot = k.action_mix.reduce((x, y) => x + y, 0) || 1;
    actionChart.data.datasets[0].data = k.action_mix.map((v) => Math.round(v / tot * 1000) / 10);
    actionChart.update('none');
  }
  // Trust trajectories: one sample every 4 steps, from the detector's own scores.
  if (data.trust_scores && data.point && data.point.step % 4 === 0 &&
      SER.trustSteps[SER.trustSteps.length - 1] !== data.point.step) {
    const t = {};
    (data.vehicles || []).forEach((v) => { t[String(v.vehicle_id)] = v.trust_score; });
    SER.trustSteps.push(data.point.step); SER.trust.push(t);
    if (SER.trustSteps.length > 120) { SER.trustSteps.shift(); SER.trust.shift(); }
    redrawSeriesCharts();
  }
  if (trustChart && data.vehicles) {
    let hi = 0, norm = 0, susp = 0, adv = 0;
    data.vehicles.forEach((v) => {
      if (v.active === false) return;
      const score = (v.trust_score !== undefined && v.trust_score !== null) ? v.trust_score : 0.5;
      if (score >= 0.8) hi++;
      else if (score >= 0.6) norm++;
      else if (score >= 0.4) susp++;
      else adv++;
    });
    trustChart.data.datasets[0].data = [hi, norm, susp, adv];
    trustChart.update('none');
  }
}

function updateFLPanels(data) {
  const k = data.kpis || {};
  const current = data.current_round || null;
  const readiness = data.fl_readiness || {};

  backendState.flStage = k.fl_stage || 'idle';
  backendState.flRound = k.fl_round || 0;

  const stage = k.fl_stage || 'idle';
  const strategy = k.strategy || 'unknown';
  const indicator = document.getElementById('fl-status-indicator');
  if (indicator) {
    indicator.textContent = `Round ${k.fl_round || 0} · ${STAGE_LABELS[stage] || stage} · ` +
      `${STRATEGY_LABELS[strategy] || strategy} · ${k.next_round_ready ? 'next round ready' : 'next round not ready'}`;
  }

  setText('fl-stage-detail', k.fl_stage_detail || 'Waiting for the simulation to start.');

  const roundSource = current
    || data.last_round
    || (data.fl_rounds && data.fl_rounds.length ? data.fl_rounds[data.fl_rounds.length - 1] : null);

  setText('fl-stage', STAGE_LABELS[stage] || stage);
  setText('fl-round-no', `#${k.fl_round || 0}`);
  const statusText = roundSource
    ? (roundSource.status === 'failed' && roundSource.error_message
        ? `failed (${roundSource.failure_stage || 'unknown stage'})`
        : (roundSource.status || 'unknown'))
    : 'no round yet';
  setText('fl-status', statusText);

  setText('fl-elapsed', roundSource && roundSource.duration_ms ? fmtMs(roundSource.duration_ms) : 'N/A');
  setText('fl-strategy', (roundSource && roundSource.aggregation_strategy)
    ? roundSource.aggregation_strategy : (k.strategy || 'N/A'));
  setText('fl-model-version', `v${k.global_model_version ?? 0}`);
  setText('fl-eligible', fmt(roundSource ? roundSource.eligible_clients : readiness.eligible_clients));
  setText('fl-trained', fmt(roundSource ? roundSource.trained_clients : 0));
  setText('fl-accepted', fmt(roundSource ? roundSource.accepted_clients : 0));
  setText('fl-excluded', fmt(roundSource ? roundSource.excluded_clients : 0));

  const statusEl = document.getElementById('fl-status');
  if (statusEl && roundSource) {
    statusEl.style.color = roundSource.status === 'success' ? cssVar('--ok')
      : roundSource.status === 'failed' ? cssVar('--bad')
      : roundSource.status === 'in_progress' ? cssVar('--accent') : cssVar('--warn');
  }

  setText('fl-readiness-reason', readiness.reason || k.next_round_reason || 'N/A');

  const exclusionList = document.getElementById('fl-exclusion-list');
  if (exclusionList) {
    const perClient = roundSource && roundSource.per_client_metrics ? roundSource.per_client_metrics : {};
    const excluded = Object.values(perClient).filter((c) => c.eligible && c.included_in_aggregation === false);
    if (excluded.length === 0) {
      const failed = roundSource && roundSource.status === 'failed';
      exclusionList.innerHTML = failed
        ? `<div class="empty">The round did not reach aggregation${roundSource.error_message ? ': ' + escapeHtml(roundSource.error_message) : ''}.</div>`
        : '<div class="empty">No clients excluded in the last round.</div>';
    } else {
      exclusionList.innerHTML = excluded.map((c) =>
        `<div class="fl-item excluded"><span class="fl-item-id">V${c.client_id}</span>` +
        `<span class="fl-item-role">${escapeHtml(c.role)}</span>` +
        `<span class="fl-item-reason">${escapeHtml(c.exclusion_reason || 'excluded by the aggregation strategy')}</span></div>`
      ).join('');
    }
  }

  const agg = roundSource && roundSource.aggregation_metrics ? roundSource.aggregation_metrics : {};
  const tpr = agg.attack_detection_rate;
  const fpr = agg.false_positive_rate;
  const fnr = agg.false_negative_rate;
  const tp = agg.true_positives, fp = agg.false_positives, fn = agg.false_negatives;
  const precision = (tp !== undefined && tp !== null && (tp + fp) > 0) ? (tp / (tp + fp)) : null;

  setText('m-tpr', tpr === null || tpr === undefined ? 'N/A' : tpr.toFixed(3));
  setText('m-fpr', fpr === null || fpr === undefined ? 'N/A' : fpr.toFixed(3));
  setText('m-fnr', fnr === null || fnr === undefined ? 'N/A' : fnr.toFixed(3));
  setText('m-precision', precision === null ? 'N/A' : precision.toFixed(3));

  const note = document.getElementById('m-eval-note');
  if (note) {
    note.textContent = (tpr === null || tpr === undefined)
      ? 'Detection metrics need a round with valid decisions and ground-truth labels; N/A until one exists.'
      : `Round ${roundSource.round_number}: TP ${tp ?? 0}, FP ${fp ?? 0}, FN ${fn ?? 0}. Computed against ground-truth roles for evaluation only.`;
  }
}

/** Render the per-client participation table. */
function updateClientParticipationDisplay(data) {
  const tbody = document.getElementById('client-table-body');
  if (!tbody) return;

  const participation = data.client_participation || {};
  const current = data.current_round || data.last_round || null;
  const perClient = current && current.per_client_metrics ? current.per_client_metrics : {};
  const ids = Object.keys(participation).sort((a, b) => Number(a) - Number(b));

  if (ids.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No participation data yet.</td></tr>';
    return;
  }

  tbody.innerHTML = ids.map((id) => {
    const p = participation[id];
    const detail = perClient[id] || null;

    const samples = detail && detail.replay_samples !== undefined ? detail.replay_samples : p.samples_contributed;
    const trained = detail ? detail.performed_training : (p.rounds_trained > 0);
    const loss = detail && detail.training_loss !== undefined && detail.training_loss !== null
      ? detail.training_loss : p.last_training_loss;
    const dur = detail && detail.training_duration_ms !== undefined && detail.training_duration_ms !== null
      ? detail.training_duration_ms : p.last_training_duration_ms;

    const state = p.inclusion_state || 'not_participated';
    let stateClass = 'neutral';
    if (state === 'included') stateClass = 'ok';
    else if (state === 'excluded') stateClass = 'bad';
    else if (state === 'not_in_aggregation') stateClass = 'warn';

    let note = '';
    if (detail && detail.note) note = ` (${detail.note})`;
    else if (p.exclusion_reason) note = ` (${p.exclusion_reason})`;

    return `<tr>
      <td>V${id}</td>
      <td>${escapeHtml(p.role)}</td>
      <td>${samples ?? 0}</td>
      <td>${trained ? 'yes' : 'no'}</td>
      <td>${loss === null || loss === undefined ? 'N/A' : loss.toFixed(4)}</td>
      <td>${dur === null || dur === undefined ? 'N/A' : Math.round(dur) + ' ms'}</td>
      <td class="state-${stateClass}">${escapeHtml(state + note)}</td>
    </tr>`;
  }).join('');
}

/** Render the federated round history. */
function updateFLRoundsDisplay(data) {
  const container = document.getElementById('fl-round-history');
  if (!container) return;

  const rounds = data.fl_rounds || [];
  if (rounds.length === 0) {
    container.innerHTML = '<div class="empty">No completed rounds yet.</div>';
    return;
  }

  container.innerHTML = rounds.slice().reverse().map((r) => {
    let cls = 'neutral';
    if (r.status === 'success') cls = 'ok';
    else if (r.status === 'failed') cls = 'bad';
    else if (r.status === 'insufficient_experience') cls = 'warn';

    const det = r.aggregation_metrics || {};
    const tpr = det.attack_detection_rate;
    const tprText = (tpr === null || tpr === undefined) ? 'N/A' : tpr.toFixed(2);
    const reason = r.error_message ? ` — ${escapeHtml(r.error_message)}` : '';
    return `<div class="fl-item ${cls}">
      <span class="fl-item-id">R${r.round_number}</span>
      <span class="fl-item-role">${escapeHtml(r.aggregation_strategy || 'unknown')}</span>
      <span class="fl-item-reason">${escapeHtml(r.status)} · ${Math.round(r.duration_ms || 0)} ms · trained ${r.trained_clients ?? 0} · accepted ${r.accepted_clients ?? 0} · excluded ${r.excluded_clients ?? 0} · TPR ${tprText}${reason}</span>
    </div>`;
  }).join('');
}

function updateEventFeed(data) {
  const feed = document.getElementById('event-feed');
  if (!feed) return;

  const events = data.events || [];
  if (events.length === 0) {
    // Keep the previous rows while running: an empty step is not "no history".
    if (!feed.dataset.hasRows) feed.innerHTML = '<div class="event-item muted">No offloading events in this session yet.</div>';
    return;
  }
  feed.dataset.hasRows = '1';

  const rows = events.slice(-12).reverse().map((ev) => {
    // Ground truth, detector decision and task outcome are separate concepts.
    const badges = [];
    if (ev.detector_alert) badges.push('<span class="badge badge-alert">detector alert</span>');
    if (ev.ground_truth_malicious) badges.push('<span class="badge badge-truth">role ' + escapeHtml(ev.role) + '</span>');
    if (ev.task_failed) badges.push('<span class="badge badge-fail">task failed</span>');
    return `<div class="event-item ${ev.success ? 'success' : 'failure'}">
      <span class="ev-head">V${ev.vid} ${ev.success ? 'completed' : 'failed'} → ${escapeHtml(ev.target)}</span>
      <span class="ev-body">${escapeHtml((ev.task_type || '').split('_')[0])} · ${ev.delay_ms} ms · reward ${ev.reward} · target trust ${ev.target_trust ?? 'N/A'}</span>
      <span class="ev-badges">${badges.join('')}</span>
    </div>`;
  });
  feed.innerHTML = rows.join('');
}

/* -------------------------------------------------------------------- */
/* Control                                                               */
/* -------------------------------------------------------------------- */

async function sendControl(action) {
  const btns = ['btn-start', 'btn-pause', 'btn-stop'].map((id) => document.getElementById(id));
  btns.forEach((b) => { if (b) b.disabled = true; });

  try {
    const res = await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });

    let payload = null;
    try { payload = await res.json(); } catch (e) { /* non-JSON body */ }

    if (!res.ok) {
      const detail = (payload && (payload.detail || payload.message))
        ? (payload.detail || payload.message) : `HTTP ${res.status}`;
      showBanner(`${action} failed: ${detail}`, 'error', true);
      await refreshState();
      return;
    }

    clearBanner(true);
    if (payload && payload.kpis) {
      applyState({ kpis: payload.kpis, vehicles: [], rsus: [], events: [] });
    } else {
      await refreshState();
    }
  } catch (e) {
    showBanner(`${action} failed: ${e && e.message ? e.message : e}`, 'error', true);
    await refreshState();
  }
}

async function refreshState() {
  try {
    const res = await fetch('/api/state', { cache: 'no-store' });
    if (!res.ok) return;
    applyState(await res.json());
  } catch (e) {
    /* the banner already reports the problem */
  }
}

/** POST a configuration change; returns {ok, payload}. */
async function postConfig(body) {
  const res = await fetch('/api/configure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let payload = {};
  try { payload = await res.json(); } catch (e) { /* ignore */ }
  return { ok: res.ok, payload, status: res.status };
}

function readExperimentForm() {
  const num = (id) => Number(document.getElementById(id).value);
  const body = {
    mobility_mode: document.getElementById('select-mobility').value,
    num_vehicles: num('input-vehicles'),
    attack_ratio: num('input-attack-ratio'),
    rebuild: true,
  };
  if (body.mobility_mode === 'highway') {
    body.rsu_x_positions = [num('input-rsu-1'), num('input-rsu-2'), num('input-rsu-3')];
  }
  return body;
}

/** Client-side range check that mirrors the backend; returns a message or ''. */
function validateExperimentForm(body) {
  if (!Number.isInteger(body.num_vehicles) || body.num_vehicles < 5 || body.num_vehicles > 60) {
    return 'Vehicles must be a whole number between 5 and 60.';
  }
  if (!(body.attack_ratio >= 0 && body.attack_ratio <= 0.5)) {
    return 'Attack ratio must be between 0 and 0.5; above 0.5 the detector has no honest majority.';
  }
  if (body.rsu_x_positions) {
    if (body.rsu_x_positions.some((x) => !(x >= 0 && x <= 3200))) {
      return 'RSU positions must lie within 0–3200 m.';
    }
    const sorted = body.rsu_x_positions.slice().sort((a, b) => a - b);
    if (sorted.some((x, i) => i > 0 && x - sorted[i - 1] < 50)) {
      return 'RSUs closer than 50 m would overlap; keep them at least 50 m apart.';
    }
  }
  return '';
}

/* -------------------------------------------------------------------- */
/* Event wiring                                                          */
/* -------------------------------------------------------------------- */

function setupEventListeners() {
  document.getElementById('btn-start').addEventListener('click', () => {
    sendControl(backendState.state === 'PAUSED' ? 'resume' : 'start');
  });
  document.getElementById('btn-pause').addEventListener('click', () => sendControl('pause'));
  document.getElementById('btn-stop').addEventListener('click', () => sendControl('stop'));

  const liveConfig = (body, label) => postConfig(body).then(({ ok, payload, status }) => {
    if (!ok) {
      showBanner(`${label} rejected: ${payload.detail || 'HTTP ' + status}`, 'error');
      return refreshState();
    }
    clearBanner();
    return null;
  }).catch((err) => showBanner(`${label} failed: ${err && err.message ? err.message : err}`, 'error'));

  document.getElementById('slider-alpha').addEventListener('input', (e) => {
    setText('val-alpha', Number(e.target.value).toFixed(2));
    liveConfig({ alpha: parseFloat(e.target.value) }, 'Weight change');
  });
  document.getElementById('slider-tau').addEventListener('input', (e) => {
    setText('val-tau', Number(e.target.value).toFixed(2));
    liveConfig({ exclusion_threshold: parseFloat(e.target.value) }, 'Threshold change');
  });
  document.getElementById('select-strategy').addEventListener('change', (e) => {
    liveConfig({ aggregation_strategy: e.target.value }, 'Strategy change');
  });
  document.getElementById('select-speed').addEventListener('change', (e) => {
    liveConfig({ sim_speed: parseFloat(e.target.value) }, 'Speed change');
  });
  document.getElementById('chk-auto-fl').addEventListener('change', (e) => {
    liveConfig({ auto_fl_enabled: e.target.checked }, 'Automatic rounds change');
  });

  // Mobility mode and the other experiment parameters need a new session.
  const mobilitySel = document.getElementById('select-mobility');
  const syncRsuInputs = () => {
    const on = mobilitySel.value === 'highway';
    ['input-rsu-1', 'input-rsu-2', 'input-rsu-3'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.disabled = !on;
    });
    setText('config-message', on ? '' :
      'RSU positions are fixed inside the observed road segment for trace replay.');
  };
  mobilitySel.addEventListener('change', syncRsuInputs);
  syncRsuInputs();

  document.getElementById('btn-apply-config').addEventListener('click', async () => {
    const body = readExperimentForm();
    const problem = validateExperimentForm(body);
    const msg = document.getElementById('config-message');
    if (problem) {
      if (msg) { msg.textContent = problem; msg.className = 'note error'; }
      return;
    }
    const applyBtn = document.getElementById('btn-apply-config');
    applyBtn.disabled = true;
    if (msg) { msg.textContent = 'Applying… a trace replay may need a few seconds to load.'; msg.className = 'note'; }
    try {
      const { ok, payload, status } = await postConfig(body);
      if (!ok) {
        if (msg) { msg.textContent = 'Rejected: ' + (payload.detail || 'HTTP ' + status); msg.className = 'note error'; }
        return;
      }
      if (msg) { msg.textContent = 'Applied. A new session was created; press Start to run it.'; msg.className = 'note'; }
      clearBanner(true);
      await refreshState();
    } catch (err) {
      if (msg) { msg.textContent = 'Request failed: ' + (err && err.message ? err.message : err); msg.className = 'note error'; }
    } finally {
      applyBtn.disabled = false;
    }
  });

  const setView = (mode) => {
    if (roadView) roadView.setMode(mode);
    document.getElementById('btn-view-follow').classList.toggle('active', mode === 'follow');
    document.getElementById('btn-view-overview').classList.toggle('active', mode === 'overview');
  };
  document.getElementById('btn-view-follow').addEventListener('click', () => setView('follow'));
  document.getElementById('btn-view-overview').addEventListener('click', () => setView('overview'));

  // Display toggles act on the viewport only.
  const bindToggle = (id, prop) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', (e) => { if (roadView) roadView[prop] = e.target.checked; });
  };
  bindToggle('chk-coverage', 'showCoverage');
  bindToggle('chk-links', 'showLinks');
  bindToggle('chk-groundtruth', 'showGroundTruth');
}
