/**
 * Minimal DOM/headless harness that actually EXECUTES web/static/js/dashboard.js
 * so that frontend failures can be reproduced without a real browser.
 *
 * Exports `createHarness({ sceneThrows, chartThrows })` which returns a fresh
 * sandbox plus helpers, so several scenarios can be exercised in one process.
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const HTML = fs.readFileSync(path.join(ROOT, 'web/static/index.html'), 'utf8');
const DASH = path.join(ROOT, 'web/static/js/dashboard.js');

// ---------------------------------------------------------------------------
// Element registry built from the real index.html, so a missing element in
// the markup is reproduced here exactly as a browser would.
// ---------------------------------------------------------------------------
const ALL_IDS = (() => {
  const ids = new Set();
  const re = /id="([^"]+)"/g;
  let m;
  while ((m = re.exec(HTML)) !== null) ids.add(m[1]);
  return ids;
})();

class FakeElement {
  constructor(id, tag) {
    this.id = id;
    this.tagName = (tag || 'div').toUpperCase();
    this.textContent = '';
    this.innerHTML = '';
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.style = {};
    this.classList = {
      _s: new Set(),
      add: (c) => this.classList._s.add(c),
      remove: (c) => this.classList._s.delete(c),
      toggle: (c, on) => (on ? this.classList._s.add(c) : this.classList._s.delete(c)),
      contains: (c) => this.classList._s.has(c),
    };
    this.children = [];
    this.options = [];
    this._listeners = {};
    this._ctx = {
      fillStyle: '', font: '', fillText() {}, strokeText() {},
      save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {},
      stroke() {}, fill() {}, closePath() {}, clearRect() {},
    };
  }
  addEventListener(type, fn) {
    (this._listeners[type] = this._listeners[type] || []).push(fn);
  }
  removeEventListener() {}
  appendChild(c) { this.children.push(c); return c; }
  insertBefore(c) { this.children.push(c); return c; }
  getContext() { return this._ctx; }
  getBoundingClientRect() { return { width: 800, height: 400, top: 0, left: 0 }; }
  resize() {}
  update() {}
  click() {
    (this._listeners['click'] || []).forEach((fn) =>
      fn({ target: this, preventDefault() {} }));
  }
  dispatch(type, evt) {
    (this._listeners[type] || []).forEach((fn) =>
      fn(evt || { target: this, preventDefault() {} }));
  }
  focus() {} blur() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
}

function ChartStub(ctx, config) {
  this.data = config.data;
  this.options = config.options;
  this.update = () => {};
  this.resize = () => {};
  this.destroy = () => {};
}

/** Stub for the road viewport (web/static/js/sim_view.js). */
function SceneStub(containerId) {
  this.container = containerId;
  this.selectedVid = null;
  this.showCoverage = true;
  this.showLinks = true;
  this.showGroundTruth = false;
  this.onSelect = null;
  this.setScene = () => {};
  this.updateVehicles = () => {};
  this.triggerOffloadingBeam = () => {};
  this.resize = () => {};
}

function createHarness(options = {}) {
  const { sceneThrows = false, chartThrows = false, controlResponse = null } = options;

  const registry = new Map();
  for (const id of ALL_IDS) {
    const tag = id.startsWith('btn-') ? 'button'
      : id.startsWith('select-') ? 'select'
      : id.startsWith('chk-') ? 'input'
      : id.startsWith('kpi-') || id.startsWith('m-') || id.startsWith('fl-') ? 'span'
      : id.endsWith('-chart') ? 'canvas'
      : 'div';
    registry.set(id, new FakeElement(id, tag));
  }

  const fetchCalls = [];
  const consoleErrors = [];
  let controlResult = controlResponse;
  let stateResult = { kpis: {} };

  const documentStub = {
    getElementById: (id) => registry.get(id) || null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: (tag) => new FakeElement('', tag),
    addEventListener: (type, fn) => {
      if (type === 'DOMContentLoaded') documentStub._domReady = fn;
    },
    _domReady: null,
    activeElement: { tagName: 'BODY' },
  };

  class WebSocketStub {
    constructor(url) {
      this.url = url;
      this.readyState = 1;
      WebSocketStub.instances.push(this);
    }
    send() {} close() {}
    emit(data) { if (this.onmessage) this.onmessage({ data: JSON.stringify(data) }); }
  }
  WebSocketStub.instances = [];

  const sandbox = {
    console: {
      log: () => {},
      error: (...a) => consoleErrors.push(a.map(String).join(' ')),
      warn: () => {}, info: () => {}, debug: () => {},
    },
    document: documentStub,
    window: { addEventListener: () => {}, location: { protocol: 'http:', host: 'localhost:8000' } },
    fetch: async (url, opts) => {
      fetchCalls.push({ url, opts });
      if (url === '/api/state') {
        return { ok: true, status: 200, json: async () => stateResult };
      }
      if (url === '/api/control') {
        const body = JSON.parse(opts.body);
        const res = controlResult
          || { status: 'ok', is_running: true, state: { is_running: true, sim_state: 'running' } };
        return { ok: true, status: 200, json: async () => res };
      }
      return { ok: true, status: 200, json: async () => ({}) };
    },
    WebSocket: WebSocketStub,
    Chart: chartThrows
      ? function () { throw new Error('ReferenceError: Chart is not defined'); }
      : ChartStub,
    VANETRoadView: sceneThrows
      ? function () { throw new Error('ReferenceError: THREE is not defined'); }
      : SceneStub,
    setTimeout: (fn) => 0,
    clearTimeout: () => {},
    setInterval: () => 0,
    clearInterval: () => {},
    Math, JSON, Date, Number, String, Object, Array, Boolean, Error, Promise,
    isNaN, parseFloat, parseInt,
  };
  sandbox.globalThis = sandbox;

  return {
    sandbox, registry, documentStub, fetchCalls, consoleErrors, WebSocketStub,
    ids: ALL_IDS,
    load(file = DASH) {
      const vm = require('vm');
      const code = fs.readFileSync(file, 'utf8');
      vm.createContext(sandbox);
      vm.runInContext(code, sandbox, { filename: file });
    },
    fireReady() {
      let err = null;
      try { if (documentStub._domReady) documentStub._domReady(); }
      catch (e) { err = e; }
      return err;
    },
    setControlResponse(r) { controlResult = r; },
    setState(r) { stateResult = r; },
  };
}

module.exports = { createHarness, ALL_IDS, DASH, ROOT };
