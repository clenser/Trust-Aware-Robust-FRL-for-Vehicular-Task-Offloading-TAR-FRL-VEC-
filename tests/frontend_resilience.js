/**
 * Regression test: a failure in the optional 3D renderer or chart library
 * must NOT prevent the simulation console, its controls or the telemetry
 * subscription from working.
 *
 * This reproduces the "dashboard renders but clicking START does nothing"
 * failure mode.
 */
const path = require('path');
const { createHarness } = require('./frontend_harness.js');

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log(`  PASS  ${name}`);
  else { failures += 1; console.log(`  FAIL  ${name}${detail ? ' -- ' + detail : ''}`); }
}

function runScenario(name, opts) {
  console.log('');
  console.log(`SCENARIO: ${name}`);
  const h = createHarness(opts);
  h.load();

  const readyThrew = h.fireReady();

  const startBtn = h.registry.get('btn-start');
  const hasHandler = startBtn && startBtn._listeners['click'] && startBtn._listeners['click'].length > 0;

  h.fetchCalls.length = 0;
  if (hasHandler) {
    try { startBtn.click(); } catch (e) { /* handled inside */ }
  }
  const controlCall = h.fetchCalls.find((c) => c.url === '/api/control');

  check(`${name}: START handler registered`, !!hasHandler,
        'no click listener on #btn-start');
  check(`${name}: START reaches /api/control`, !!controlCall,
        'no control request issued');
  check(`${name}: WebSocket subscribed`, h.WebSocketStub.instances.length === 1,
        `instances=${h.WebSocketStub.instances.length}`);
  check(`${name}: init did not abort`, readyThrew === null,
        readyThrew ? readyThrew.message : '');
}

console.log('='.repeat(70));
console.log('FRONT-END RESILIENCE TEST');
console.log('='.repeat(70));

runScenario('healthy environment', {});
runScenario('road viewport unavailable', { sceneThrows: true });
runScenario('chart library unavailable', { chartThrows: true });
runScenario('both viewport and charts unavailable', { sceneThrows: true, chartThrows: true });

console.log('');
console.log('='.repeat(70));
if (failures === 0) {
  console.log('ALL RESILIENCE CHECKS PASSED');
} else {
  console.log(`${failures} RESILIENCE CHECK(S) FAILED`);
}
console.log('='.repeat(70));
process.exit(failures === 0 ? 0 : 1);
