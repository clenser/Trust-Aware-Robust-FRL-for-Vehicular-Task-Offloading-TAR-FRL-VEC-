"""Live verification of automatic federated learning in the running console."""
import asyncio
import json
import time
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


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


async def main():
    print("=" * 74)
    print("LIVE AUTOMATIC FEDERATED LEARNING VERIFICATION")
    print("=" * 74)

    post("/api/control", {"action": "stop"})
    post("/api/configure", {"fl_warmup_steps": 3, "fl_interval_steps": 6,
                            "fl_min_buffer_size": 4, "fl_min_clients": 2})

    st = get("/api/state")
    check("auto FL enabled by default in the backend config",
          st["kpis"]["auto_fl_enabled"] is True)
    check("FL round counter starts at 0", st["kpis"]["fl_round"] == 0)

    stages_seen = []
    rounds_seen = []

    async with websockets.connect(WS) as ws:
        await ws.recv()
        post("/api/control", {"action": "start"})
        print("\n  started - observing the FL lifecycle over the websocket...")

        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            except asyncio.TimeoutError:
                break
            k = m["kpis"]
            stage = k.get("fl_stage")
            if stage and (not stages_seen or stages_seen[-1] != stage):
                stages_seen.append(stage)
                print("   stage -> %-22s round=%s steps=%s"
                      % (stage, k.get("fl_round"), k.get("sim_steps")))
            if k.get("fl_round", 0) >= 1 and "complete" in stages_seen:
                break

    print("\n  stages observed: %s" % stages_seen)
    check("FL stage changes were published live", len(stages_seen) >= 2, str(stages_seen))
    check("a training stage was published", "training" in stages_seen, str(stages_seen))
    check("an aggregation stage was published",
          "aggregating" in stages_seen, str(stages_seen))
    check("a distribution stage was published", "updating" in stages_seen, str(stages_seen))
    check("a completion stage was published", "complete" in stages_seen, str(stages_seen))

    st = get("/api/state")
    k = st["kpis"]
    print("\n  final: fl_round=%s completed=%s failed=%s model_v=%s"
          % (k["fl_round"], k["fl_rounds_completed"], k["fl_rounds_failed"],
             k["global_model_version"]))
    check("a real FL round completed", k["fl_round"] >= 1, str(k["fl_round"]))
    check("successful round counted", k["fl_rounds_completed"] >= 1, str(k["fl_rounds_completed"]))
    check("no rounds failed", k["fl_rounds_failed"] == 0, str(k["fl_rounds_failed"]))
    check("global model version incremented", k["global_model_version"] >= 1,
          str(k["global_model_version"]))

    lr = st.get("last_round")
    check("round detail is available", lr is not None)
    if lr:
        print("  round #%s status=%s trained=%s accepted=%s excluded=%s duration=%.0fms"
              % (lr["round_number"], lr["status"], lr["trained_clients"],
                 lr["accepted_clients"], lr["excluded_clients"], lr["duration_ms"]))
        check("round status is success", lr["status"] == "success", lr.get("error_message") or "")
        check("clients really trained", lr["trained_clients"] >= 2, str(lr["trained_clients"]))
        check("per-client real training loss recorded",
              any(c.get("training_loss") is not None and c.get("grad_steps", 0) > 0
                  for c in lr["per_client_metrics"].values()))
        check("round tagged with the session", bool(lr.get("session_id")))

    hist = get("/api/fl_rounds")["rounds"]
    check("round history contains the round", len(hist) >= 1, str(len(hist)))

    post("/api/control", {"action": "stop"})
    print("\n" + "=" * 74)
    print("LIVE AUTOMATIC FL: " + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    print("=" * 74)
    return 0 if ok else 1


raise SystemExit(asyncio.run(main()))
