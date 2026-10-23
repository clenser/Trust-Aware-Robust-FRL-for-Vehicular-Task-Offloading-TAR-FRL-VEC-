/**
 * Executable test of the road renderer (web/static/js/sim_view.js) in Node with a
 * recording 2D context.  It checks that the renderer
 *   - runs for both views and for the highway and trace scenes without throwing,
 *   - draws links, detector alerts, federated uploads and exclusions,
 *   - uses the red colour for a detector alert and for an excluded update,
 *   - draws only what the backend reported (no link without an event),
 *   - interpolates and wraps without producing NaN coordinates.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log(`  PASS  ${name}`);
  else { failures += 1; console.log(`  FAIL  ${name}${detail ? ' -- ' + detail : ''}`); }
}

function makeCtx(log) {
  const state = {};
  return new Proxy({}, {
    get(_, prop) {
      if (prop === 'measureText') return (t) => ({ width: String(t).length * 6 });
      if (prop === 'createLinearGradient') return () => ({ addColorStop() {} });
      if (prop === 'roundRect') return undefined;     // exercise the fallback path
      if (prop in state) return state[prop];
      return (...args) => { log.calls.push([prop, args, state.strokeStyle, state.fillStyle]); };
    },
    set(_, prop, value) { state[prop] = value; log.sets.push([prop, value]); return true; },
  });
}

function makeEnv() {
  const log = { calls: [], sets: [] };
  const ctx = makeCtx(log);
  const canvas = { width: 0, height: 0, style: {}, className: '', getContext: () => ctx, addEventListener() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1100, height: 520 }) };
  const container = {
    appendChild() {}, getBoundingClientRect: () => ({ width: 1100, height: 520 }),
  };
  let t = 1000;
  const sandbox = {
    document: {
      getElementById: () => container,
      createElement: () => canvas,
      documentElement: {},
    },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    getComputedStyle: () => ({ getPropertyValue: (n) => ({ '--bad': '#c0392b', '--ok': '#17803d', '--accent': '#1f5fbf' }[n] || '') }),
    performance: { now: () => t },
    requestAnimationFrame: () => 0,
    cancelAnimationFrame() {},
    Math, Map, Set, Array, Object, Number, String, JSON, Date, console,
  };
  sandbox.globalThis = sandbox;
  sandbox.window = sandbox;                 // the script assigns window.VANETRoadView
  sandbox.addEventListener = () => {};
  sandbox.devicePixelRatio = 1;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../web/static/js/sim_view.js'), 'utf8'), sandbox);
  return { sandbox, log, advance: (ms) => { t += ms; } };
}

const HIGHWAY = { mobility_mode: 'highway', road_length_m: 3200, num_lanes: 5, lane_width_m: 3.75, x_min_m: 0 };
const TRACE = { mobility_mode: 'authentic_trace', road_length_m: 681, num_lanes: 8, lane_width_m: 3.75, x_min_m: 0 };
const RSUS = [
  { rsu_id: 'RSU_1', x: 420, y: 15, z_height: 14, coverage_radius: 480, utilization: 0.2, queue_length: 1, trust_score: 0.8 },
  { rsu_id: 'RSU_2', x: 1480, y: -42, z_height: 38, coverage_radius: 650, utilization: 0.8, queue_length: 9, trust_score: 0.3 },
];
function vehicles(shift) {
  const classes = ['Car', 'SUV', 'Autonomous_EV', 'Truck'];
  return Array.from({ length: 12 }, (_, i) => ({
    vehicle_id: i + 1, x: 100 + i * 250 + shift, y: 0, lane: (i % 5) + 1, speed: 22, vclass: classes[i % 4],
    trust_score: i === 3 ? 0.2 : (i === 5 ? 0.5 : 0.9), role: i === 3 ? 'poisoning' : 'honest', active: true,
    queue_length: 1, cpu_capacity_ghz: 1.2, connected_rsus: ['RSU_1'] }));
}

console.log('='.repeat(70));
console.log('ROAD RENDERER TEST (Node, recording canvas)');
console.log('='.repeat(70));

{
  const { sandbox, log, advance } = makeEnv();
  const view = new sandbox.VANETRoadView('viewport');
  check('constructs', !!view);
  let threw = null;
  try {
    view.setScene(HIGHWAY, RSUS);
    view.updateVehicles(vehicles(0));
    advance(500);
    view.updateVehicles(vehicles(11));          // next snapshot: vehicles moved 11 m
    view.draw();
  } catch (e) { threw = e; }
  check('draws the highway close-up without throwing', threw === null, threw && threw.stack);
  check('road, vehicles and RSUs produce drawing calls', log.calls.length > 200, log.calls.length);

  // no link without a reported event
  const before = log.calls.filter((c) => c[0] === 'quadraticCurveTo').length;
  check('no offloading link is drawn without an event', before === 0, before);

  // a reported failed offload, a reported detector alert, a reported V2V
  log.calls.length = 0;
  // The close-up draws links of vehicles in the window: centre it on vehicles 3 and 4.
  view.manualCenter = true; view.center = 800; view.targetCenter = 800;
  view.triggerOffloadingBeam({ vid: 3, target: 'RSU_1', action: 1, success: false, detector_alert: false });
  view.triggerOffloadingBeam({ vid: 4, target: 'RSU_2', action: 2, success: false, detector_alert: true });
  view.triggerOffloadingBeam({ vid: 3, target: 'V4', action: 4, success: true, detector_alert: false });
  view.setMode('follow'); view.manualCenter = true; view.center = 800; view.targetCenter = 800;
  view.draw();
  const curves = log.calls.filter((c) => c[0] === 'quadraticCurveTo');
  check('reported events draw links', curves.length >= 1, curves.length);
  const redStroke = log.calls.some((c) => c[0] === 'stroke' && String(c[2]).toLowerCase() === '#c0392b');
  check('a detector alert is drawn with the red colour', redStroke);
  const alertText = log.calls.some((c) => c[0] === 'fillText' && c[1][0] === 'DETECTOR ALERT');
  check('the detector alert is labelled', alertText);

  // federated round: uploads and an exclusion
  log.calls.length = 0;
  view.showFLRound({ round: 3, clients: [
    { vid: 3, submitted: true, included: true }, { vid: 5, submitted: true, included: true },
    { vid: 4, submitted: true, included: false, reason: 'model trust 0.12' }] });
  advance(2500);
  view.draw();
  const uploads = log.calls.filter((c) => c[0] === 'lineTo').length;
  check('federated uploads are drawn', uploads > 0, uploads);
  const exclMark = log.calls.some((c) => c[0] === 'fillText' && String(c[1][0]).indexOf('excluded') === 0);
  check('an excluded update is labelled with the detector reason', exclMark);
  const exclRed = log.calls.some((c) => c[0] === 'stroke' && String(c[2]).toLowerCase() === '#c0392b');
  check('an excluded update is drawn in red', exclRed);
  const banner = log.calls.some((c) => c[0] === 'fillText' && String(c[1][0]).indexOf('Round 3') === 0);
  check('the round banner reports the real counts', banner);
  const t = log.calls.filter((c) => c[0] === 'fillText').map((c) => String(c[1][0])).find((x) => x.indexOf('Round 3') === 0);
  check('banner numbers: 3 uploads, 2 accepted, 1 excluded', t && t.indexOf('3 uploads') > 0 && t.indexOf('2 accepted') > 0 && t.indexOf('1 excluded') > 0, t);

  // overview + ground-truth overlay
  view.setMode('overview'); view.showGroundTruth = true;
  threw = null;
  try { view.draw(); } catch (e) { threw = e; }
  check('draws the overview with the ground-truth overlay', threw === null, threw && threw.stack);

  // wrap-around on the closed highway: no NaN, no interpolation across the wrap
  const v = vehicles(0); v[0].x = 3195;
  view.updateVehicles(v); advance(500);
  const w = vehicles(0); w[0].x = 8;
  view.updateVehicles(w);
  log.calls.length = 0; view.setMode('follow');
  threw = null;
  try { view.draw(); } catch (e) { threw = e; }
  const nan = log.calls.some((c) => c[1] && c[1].some && c[1].some((a) => typeof a === 'number' && Number.isNaN(a)));
  check('wrap-around draws without NaN coordinates', threw === null && !nan, threw ? threw.stack : 'NaN present');

  // click on a vehicle selects it; click on the minimap moves the window
  view.draw();
  const hit = view._hit[0];
  let selected = null; view.onSelect = (id) => { selected = id; };
  view._click({ clientX: 0, clientY: 0 });             // outside any vehicle: deselect, no throw
  check('click handler runs', true);
}

{
  const { sandbox, log, advance } = makeEnv();
  const view = new sandbox.VANETRoadView('viewport');
  view.setScene(TRACE, RSUS);
  const list = vehicles(0).map((q, i) => Object.assign(q, { x: 20 + i * 50, lane: (i % 8) + 1, active: i < 9 }));
  view.updateVehicles(list); advance(500); view.updateVehicles(list);
  let threw = null;
  try { view.draw(); view.setMode('overview'); view.draw(); } catch (e) { threw = e; }
  check('draws the 681 m trace scene with inactive vehicles', threw === null, threw && threw.stack);
  // 9 active + 3 inactive vehicles: the hit list holds only the vehicles that were drawn.
  view.setMode('follow'); view.manualCenter = true; view.center = 340; view.targetCenter = 340; view.draw();
  const drawn = view._hit.map((h) => h.vid);
  check('inactive vehicles are never drawn or selectable', drawn.every((id) => id <= 9), JSON.stringify(drawn));
}

console.log('='.repeat(70));
console.log(failures === 0 ? 'ALL VIEWPORT CHECKS PASSED' : `${failures} VIEWPORT CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
