"""Live verification of the simulation lifecycle state machine."""
import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return json.loads(r.read().decode())


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            body = json.loads(body)
        except Exception:
            pass
        return e.code, body


ok = True


def check(name, cond, detail=""):
    global ok
    if cond:
        print(f"  PASS  {name}")
    else:
        ok = False
        print(f"  FAIL  {name}  {detail}")


print("=" * 74)
print("LIVE SIMULATION LIFECYCLE VERIFICATION")
print("=" * 74)

# --- initial state -----------------------------------------------------
st = get("/api/state")
k = st["kpis"]
print("\n1) Initial state")
print("   sim_state=%s is_running=%s session=%s steps=%s"
      % (k["sim_state"], k["is_running"], k["session_id"], k["sim_steps"]))
check("initial state is STOPPED", k["sim_state"] == "STOPPED")
check("no session yet", k["session_id"] is None)
check("no observations -> latency N/A", k["avg_latency_ms"] is None)
check("no observations -> completion N/A", k["completion_rate"] is None)

# --- invalid transitions ----------------------------------------------
print("\n2) Invalid transitions are rejected")
code, body = post("/api/control", {"action": "pause"})
check("PAUSE from STOPPED rejected", code == 409, f"got {code} {body}")
code, body = post("/api/control", {"action": "resume"})
check("RESUME from STOPPED rejected", code == 409, f"got {code} {body}")
code, body = post("/api/control", {"action": "bogus"})
check("unknown action rejected", code == 400, f"got {code} {body}")
code, body = post("/api/control", {})
check("missing action rejected", code == 400, f"got {code} {body}")

# --- START -------------------------------------------------------------
print("\n3) START")
code, body = post("/api/control", {"action": "start"})
check("START accepted", code == 200, f"got {code}")
print("   status=%s state=%s session=%s" % (body.get("status"), body.get("state"), body.get("session_id")))
check("state is RUNNING", body.get("state") == "RUNNING")
check("session id assigned", bool(body.get("session_id")))

sess = body["session_id"]

# --- simulation advances ----------------------------------------------
print("\n4) The simulation advances (measured over 3 s)")
before = get("/api/state")["kpis"]["sim_steps"]
time.sleep(3.0)
after = get("/api/state")["kpis"]["sim_steps"]
print("   sim_steps %s -> %s" % (before, after))
check("step counter advances", after > before, f"{before} -> {after}")

k = get("/api/state")["kpis"]
check("tasks accumulate", k["total_tasks"] > 0, str(k["total_tasks"]))
check("latency now available", k["avg_latency_ms"] is not None)
check("completion now available", k["completion_rate"] is not None)

# --- duplicate START ---------------------------------------------------
print("\n5) Repeated START does not create a duplicate loop")
before = get("/api/state")["kpis"]["sim_steps"]
code, body = post("/api/control", {"action": "start"})
check("second START accepted idempotently", code == 200 and body.get("status") == "already_running",
      f"got {code} {body.get('status')}")
check("session unchanged", body.get("session_id") == sess)
time.sleep(2.0)
after = get("/api/state")["kpis"]["sim_steps"]
print("   sim_steps %s -> %s over 2s" % (before, after))
check("step rate not doubled", (after - before) <= 12, f"advanced {after-before} steps in 2s")

# --- PAUSE -------------------------------------------------------------
print("\n6) PAUSE")
code, body = post("/api/control", {"action": "pause"})
check("PAUSE accepted", code == 200 and body.get("state") == "PAUSED", f"got {code} {body}")
paused_at = get("/api/state")["kpis"]["sim_steps"]
time.sleep(2.0)
still = get("/api/state")["kpis"]["sim_steps"]
print("   sim_steps %s -> %s over 2s while paused" % (paused_at, still))
check("simulation time stops while paused", still == paused_at, f"{paused_at} -> {still}")
k = get("/api/state")["kpis"]
check("session preserved while paused", k["session_id"] == sess)
check("counters preserved while paused", k["total_tasks"] > 0)

# --- RESUME ------------------------------------------------------------
print("\n7) RESUME continues the same session")
code, body = post("/api/control", {"action": "resume"})
check("RESUME accepted", code == 200 and body.get("state") == "RUNNING", f"got {code} {body}")
check("same session continued", body.get("session_id") == sess)
tasks_before = get("/api/state")["kpis"]["total_tasks"]
time.sleep(2.0)
k = get("/api/state")["kpis"]
print("   sim_steps=%s tasks %s -> %s" % (k["sim_steps"], tasks_before, k["total_tasks"]))
check("simulation advances again", k["sim_steps"] > still)
check("task counters continue (not reset)", k["total_tasks"] > tasks_before)

# --- STOP --------------------------------------------------------------
print("\n8) STOP")
code, body = post("/api/control", {"action": "stop"})
check("STOP accepted", code == 200 and body.get("state") == "STOPPED", f"got {code} {body}")
time.sleep(1.5)
k = get("/api/state")["kpis"]
print("   after stop: sim_state=%s steps=%s tasks=%s" % (k["sim_state"], k["sim_steps"], k["total_tasks"]))
check("state is STOPPED", k["sim_state"] == "STOPPED")
check("step counter reset", k["sim_steps"] == 0)
check("live metrics reset", k["total_tasks"] == 0)
check("no observations after stop -> N/A", k["avg_latency_ms"] is None)

# --- new session -------------------------------------------------------
print("\n9) START after STOP creates a NEW session")
code, body = post("/api/control", {"action": "start"})
check("START accepted", code == 200, f"got {code}")
check("new session id", body.get("session_id") != sess, f"{body.get('session_id')} vs {sess}")
time.sleep(1.5)
k = get("/api/state")["kpis"]
print("   session=%s started=%s steps=%s" % (k["session_id"], k["session_started_at_iso"], k["sim_steps"]))
check("run-start timestamp recorded", bool(k["session_started_at_iso"]))
check("simulation advancing in the new session", k["sim_steps"] > 0)

print("\n" + "=" * 74)
print("LIVE LIFECYCLE: " + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
print("=" * 74)
raise SystemExit(0 if ok else 1)
