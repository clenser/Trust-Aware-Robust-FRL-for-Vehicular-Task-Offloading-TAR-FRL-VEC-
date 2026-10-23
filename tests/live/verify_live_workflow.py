"""
End-to-end live verification of the complete console workflow.

Mirrors the manual browser flow:
  1. load the dashboard in its initial idle state
  2. click START
  3. the backend state changes and the step counter increases
  4. vehicles and telemetry update
  5. experience accumulates
  6. an FL round starts automatically
  7. the FL status pill and round history update without manual triggering
  8. PAUSE stops simulation time
  9. RESUME continues the same session
 10. STOP terminates the background work cleanly
"""
import asyncio
import json
import re
import time
import urllib.error
import urllib.request

import websockets

BASE = "http://127.0.0.1:8000"
WS = "ws://127.0.0.1:8000/ws/telemetry"

ok = True


def check(name, cond, detail=""):
    global ok
    if cond:
        print(f"  PASS  {name}")
    else:
        ok = False
        print(f"  FAIL  {name}  {detail}")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return json.loads(r.read().decode())


def get_text(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return r.read().decode()


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


async def main():
    print("=" * 76)
    print("FULL LIVE WORKFLOW VERIFICATION (simulating the browser flow)")
    print("=" * 76)

    # ---- 1. initial idle state -------------------------------------
    print("\n1) Load the dashboard in its initial idle state")
    html = get_text("/")
    check("index.html is served", "btn-start" in html and "btn-pause" in html)
    check("no manual FL trigger button in the markup", "btn-trigger-fl" not in html)
    check("asset versions are present", "dashboard.js?v=" in html)

    js = get_text("/static/js/dashboard.js")
    check("dashboard.js is served", "applyState" in js)
    check("the served JS isolates the 3D init", js.count("try {") >= 2)

    st = get("/api/state")
    k = st["kpis"]
    print("   sim_state=%s is_running=%s session=%s"
          % (k["sim_state"], k["is_running"], k["session_id"]))
    check("initial state is STOPPED", k["sim_state"] == "STOPPED")
    check("no metrics are shown before any observation",
          k["avg_latency_ms"] is None and k["completion_rate"] is None)

    # ---- 2. click START ---------------------------------------------
    print("\n2) Click START")
    code, body = post("/api/control", {"action": "start"})
    check("START accepted", code == 200, str(code))
    print("   status=%s state=%s session=%s"
          % (body.get("status"), body.get("state"), body.get("session", {}).get("id")))
    check("backend state changed to RUNNING", body.get("state") == "RUNNING")

    # ---- 3. step counter increases ----------------------------------
    print("\n3) The step counter increases")
    before = get("/api/state")["kpis"]["sim_steps"]
    time.sleep(2.5)
    after = get("/api/state")["kpis"]["sim_steps"]
    print("   sim_steps %s -> %s" % (before, after))
    check("simulation time advanced", after > before)

    # ---- 4. vehicles and telemetry update ---------------------------
    print("\n4) Vehicles and telemetry update over the WebSocket")
    stages = []
    frames = 0
    positions = []
    async with websockets.connect(WS) as ws:
        await ws.recv()
        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=4))
            except asyncio.TimeoutError:
                break
            frames += 1
            kk = m["kpis"]
            if kk.get("fl_stage"):
                stages.append(kk["fl_stage"])
            if m.get("vehicles"):
                positions.append(tuple(round(v["x"], 1) for v in m["vehicles"][:3]))
            if kk.get("fl_round", 0) >= 1 and "complete" in stages:
                break

    check("telemetry frames were delivered", frames > 3, str(frames))
    check("vehicle positions changed over time", len(set(positions)) > 1)
    check("vehicles are present in the payload", len(positions) > 0)

    # ---- 5/6. experience + automatic FL ------------------------------
    print("\n5-6) Experience accumulates and an FL round starts automatically")
    k = get("/api/state")["kpis"]
    print("   tasks=%s eligible_clients=%s fl_round=%s stage=%s"
          % (k["total_tasks"], k["eligible_clients"], k["fl_round"], k["fl_stage"]))
    check("task transitions accumulated", k["total_tasks"] > 0)
    check("a FL round completed automatically", k["fl_round"] >= 1, str(k["fl_round"]))
    check("successful rounds counted", k["fl_rounds_completed"] >= 1)
    check("model version incremented", k["global_model_version"] >= 1)
    check("all FL stages were streamed", set(stages) >= {"training", "aggregating", "updating", "complete"},
          str(sorted(set(stages))))

    # ---- 7. status pill and round history ---------------------------
    print("\n7) Status pill and round history update without manual triggering")
    rounds = get("/api/fl_rounds")["rounds"]
    check("round history contains the round", len(rounds) >= 1)
    last = rounds[-1]
    print("   R%s %s trained=%s accepted=%s excluded=%s %.0fms"
          % (last["round_number"], last["status"], last["trained_clients"],
             last["accepted_clients"], last["excluded_clients"], last["duration_ms"]))
    check("round is a genuine success", last["status"] == "success")
    check("clients really performed gradient steps",
          any(c.get("grad_steps", 0) > 0 and c.get("training_loss") is not None
              for c in last["per_client_metrics"].values()))
    check("round is tagged with the session", bool(last.get("session_id")))
    state = get("/api/state")
    check("the dashboard payload exposes the same round",
          state["last_round"]["round_number"] == last["round_number"])

    # ---- 8. PAUSE ----------------------------------------------------
    print("\n8) PAUSE stops simulation time")
    code, body = post("/api/control", {"action": "pause"})
    check("PAUSE accepted", code == 200 and body["state"] == "PAUSED")
    paused_at = get("/api/state")["kpis"]["sim_steps"]
    time.sleep(2.0)
    still = get("/api/state")["kpis"]["sim_steps"]
    print("   sim_steps %s -> %s over 2s while paused" % (paused_at, still))
    check("simulation time stopped", still == paused_at)

    # ---- 9. RESUME ----------------------------------------------------
    print("\n9) RESUME continues the same session")
    sess = get("/api/state")["kpis"]["session_id"]
    code, body = post("/api/control", json_payload := {"action": "resume"})
    check("RESUME accepted", code == 200 and body["state"] == "RUNNING")
    check("same session continued", body["session"]["id"] == sess)
    time.sleep(2.0)
    resumed = get("/api/state")["kpis"]
    print("   sim_steps=%s tasks=%s" % (resumed["sim_steps"], resumed["total_tasks"]))
    check("simulation continued", resumed["sim_steps"] > still)
    check("task counters continued", resumed["total_tasks"] > 0)

    # ---- 10. STOP -----------------------------------------------------
    print("\n10) STOP terminates the background work cleanly")
    code, body = post("/api/control", {"action": "stop"})
    check("STOP accepted", code == 200 and body["state"] == "STOPPED")
    time.sleep(1.5)
    k = get("/api/state")["kpis"]
    print("   sim_state=%s sim_steps=%s tasks=%s" % (k["sim_state"], k["sim_steps"], k["total_tasks"]))
    check("state is STOPPED", k["sim_state"] == "STOPPED")
    check("no further steps are executed", k["sim_steps"] == 0)
    check("live metrics reset", k["total_tasks"] == 0)
    check("no stale metrics are shown", k["avg_latency_ms"] is None)

    print("\n" + "=" * 76)
    print("LIVE WORKFLOW: " + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    print("=" * 76)
    return 0 if ok else 1


raise SystemExit(asyncio.run(main()))
