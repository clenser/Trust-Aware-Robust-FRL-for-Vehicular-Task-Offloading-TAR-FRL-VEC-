/**
 * Executable front-end behaviour test.
 *
 * Loads the real dashboard.js in a headless DOM harness, fires the
 * DOMContentLoaded handler, clicks START and asserts that the backend is
 * actually asked to start the simulation, that the button state follows the
 * authoritative backend state, and that API failures are surfaced.
 */
const path = require('path');
const { createHarness } = require('./frontend_harness.js');

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log(`  PASS  ${name}`);
  else { failures += 1; console.log(`  FAIL  ${name}${detail ? ' -- ' + detail : ''}`); }
}

function kpis(overrides = {}) {
  return Object.assign({
    is_running: false, sim_state: 'STOPPED', fl_round: 0, fl_stage: 'idle',
    strategy: 'trust_aware', avg_latency_ms: null, completion_rate: null,
    total_tasks: 0, active_vehicles: 5, attackers: 1, next_round_ready: false,
    session_id: null, sim_steps: 0,
  }, overrides);
}

console.log('='.repeat(70));
console.log('FRONT-END BEHAVIOUR TEST (headless DOM harness)');
console.log('='.repeat(70));

// =========================================================================
// 1. Happy path
// =========================================================================
console.log('');
console.log('SCENARIO: healthy environment');
{
  const h = createHarness();
  h.load();
  const readyThrew = h.fireReady();
  check('DOMContentLoaded handler executes', !readyThrew, readyThrew && readyThrew.message);
  check('no console errors during init', h.consoleErrors.length === 0, h.consoleErrors.join(' | '));
  check('websocket subscription opened', h.WebSocketStub.instances.length === 1);

  const startBtn = h.registry.get('btn-start');
  check('START has a click handler', !!(startBtn._listeners['click'] || []).length);

  // The initial state request happens during DOMContentLoaded.
  const stateCall = h.fetchCalls.find((c) => c.url === '/api/state');
  check('initial state fetched on load', !!stateCall);

  h.fetchCalls.length = 0;
  startBtn.click();
  const controlCall = h.fetchCalls.find((c) => c.url === '/api/control');
  check('START reaches /api/control', !!controlCall);
  if (controlCall) {
    check('START sends action=start', JSON.parse(controlCall.opts.body).action === 'start');
    check('START uses POST', controlCall.opts.method === 'POST');
  }
}

// =========================================================================
// 2. Button state machine follows the authoritative backend state
// =========================================================================
console.log('');
console.log('SCENARIO: control button state machine');
{
  const h = createHarness();
  h.load();
  h.fireReady();
  const ws = h.WebSocketStub.instances[0];
  const startBtn = h.registry.get('btn-start');
  const pauseBtn = h.registry.get('btn-pause');
  const stopBtn = h.registry.get('btn-stop');

  ws.emit({ kpis: kpis({ sim_state: 'STOPPED' }) });
  check('STOPPED -> START enabled', startBtn.disabled === false);
  check('STOPPED -> PAUSE disabled', pauseBtn.disabled === true);
  check('STOPPED -> STOP disabled', stopBtn.disabled === true);
  check('STOPPED -> START label', startBtn.textContent.toUpperCase().includes('START'));

  ws.emit({ kpis: kpis({ sim_state: 'RUNNING', is_running: true }) });
  check('RUNNING -> START disabled', startBtn.disabled === true);
  check('RUNNING -> PAUSE enabled', pauseBtn.disabled === false);
  check('RUNNING -> STOP enabled', stopBtn.disabled === false);

  ws.emit({ kpis: kpis({ sim_state: 'PAUSED', is_running: false }) });
  check('PAUSED -> START enabled (as RESUME)', startBtn.disabled === false);
  check('PAUSED -> START label is RESUME', startBtn.textContent.toUpperCase().includes('RESUME'));
  check('PAUSED -> PAUSE disabled', pauseBtn.disabled === true);
  check('PAUSED -> STOP enabled', stopBtn.disabled === false);

  ws.emit({ kpis: kpis({ sim_state: 'ERROR', is_running: false, last_error: 'boom' }) });
  check('ERROR -> STOP enabled', stopBtn.disabled === false);
  const banner = h.registry.get('error-banner');
  check('backend error is shown in the banner', banner.style.display === 'block' &&
        banner.textContent.includes('boom'), banner.textContent);
}

// =========================================================================
// 3. A missing is_running must never be read as "running"
// =========================================================================
console.log('');
console.log('SCENARIO: missing state fields');
{
  const h = createHarness();
  h.load();
  h.fireReady();
  const ws = h.WebSocketStub.instances[0];
  const startBtn = h.registry.get('btn-start');
  const pauseBtn = h.registry.get('btn-pause');

  ws.emit({ kpis: { fl_round: 3, fl_stage: 'idle' } });   // no is_running at all
  check('missing is_running -> PAUSE disabled', pauseBtn.disabled === true);
  check('missing is_running -> START enabled', startBtn.disabled === false);
  check('missing is_running -> sim state STOPPED',
        h.registry.get('kpi-simstate').textContent.toUpperCase() === 'STOPPED',
        h.registry.get('kpi-simstate').textContent);
}

// =========================================================================
// 4. A failed control request is surfaced and restores the button state
// =========================================================================
console.log('');
console.log('SCENARIO: failed control request');
{
  const h = createHarness();
  h.load();
  h.fireReady();
  const ws = h.WebSocketStub.instances[0];
  const startBtn = h.registry.get('btn-start');

  // The backend rejects the transition.
  h.setControlResponse({ status: 'error', detail: 'Cannot PAUSE from state STOPPED' });
  const origFetch = h.sandbox.fetch;
  h.sandbox.fetch = async (url, opts) => {
    if (url === '/api/control') {
      h.fetchCalls.push({ url, opts });
      return { ok: false, status: 409, json: async () => ({ detail: 'Cannot PAUSE from state STOPPED' }) };
    }
    return origFetch(url, opts);
  };

  startBtn.click();

  // The banner must show the API error and stay visible.
  setTimeout(() => {
    const banner = h.registry.get('error-banner');
    check('failed request shows the API error', banner.style.display === 'block' &&
          banner.textContent.includes('Cannot PAUSE'), banner.textContent);
    // The button state must have been restored from the backend.
    check('button state restored after failure', startBtn.disabled === false,
          'START disabled=' + startBtn.disabled);

    // A later telemetry frame must NOT wipe the error away.
    ws.emit({ kpis: kpis({ sim_state: 'STOPPED' }) });
    check('error survives a later telemetry frame',
          h.registry.get('error-banner').style.display === 'block',
          h.registry.get('error-banner').textContent);

    h.sandbox.fetch = origFetch;

    console.log('');
    console.log('='.repeat(70));
    if (failures === 0) console.log('ALL FRONT-END CHECKS PASSED');
    else console.log(`${failures} FRONT-END CHECK(S) FAILED`);
    console.log('='.repeat(70));
    process.exit(failures === 0 ? 0 : 1);
  }, 60);
}
