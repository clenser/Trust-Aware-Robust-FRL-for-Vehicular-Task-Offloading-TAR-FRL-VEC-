# Live integration and browser verification

These scripts drive the **running** application over HTTP/WebSocket (and, for
the browser test, in a real Chromium instance).  They are not part of the
pytest suite because they need a live server.

## Usage

```powershell
# 1. start the application (in one shell)
python run_simulation.py

# 2. in a second shell, run the verifications
python tests/live/verify_live_workflow.py    # the complete end-to-end flow
python tests/live/verify_live_lifecycle.py   # the state machine only
python tests/live/verify_live_autofl.py      # automatic federated learning
python tests/live/verify_live_authentic_trace.py  # the authentic NGSIM mode
python tests/browser_verification.py        # REAL BROWSER (Playwright)
```

Every script prints a PASS/FAIL line per assertion and exits non-zero if any
check failed.

## What they cover

| Script | Coverage |
|---|---|
| `verify_live_workflow.py` | The full flow: idle state → START → steps advance → telemetry → automatic FL round → status pill and round history → PAUSE → RESUME → STOP. |
| `verify_live_lifecycle.py` | The STOPPED/RUNNING/PAUSED/ERROR state machine, invalid transitions, session identity, counters, and that a repeated START does not spawn a second loop. |
| `verify_live_autofl.py` | The automatic FL lifecycle, live stage publication, real training loss, model versioning, and honest failure reporting. |
| `verify_live_authentic_trace.py` | The authentic NGSIM trace mode: the reported source, the active vehicle count, and the evolving source-vehicle bindings. |
| `browser_verification.py` | A **real Chromium** (Playwright) run: the WebGL canvas is actually painted, Chart.js instances exist, START/PAUSE/RESUME/STOP drive the backend, the automatic FL round is visible in the console, the round history / client table / trust chart are populated, missing metrics show `N/A`, a failing control request raises the error banner and restores the buttons, a second tab does not accelerate the simulation, a reload restores the running state, and the console still works when the 3D/chart CDN is blocked. |

## Browser-test prerequisites

```powershell
python -m pip install playwright
python -m playwright install chromium
```

If the browser cannot be installed, the headless DOM harnesses
(`tests/frontend_behaviour.js`, `tests/frontend_resilience.js`) still exercise
the front-end *logic*, but they cannot verify the actual WebGL rendering —
that part of the acceptance checklist remains a manual step.
