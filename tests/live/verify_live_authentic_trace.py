"""Live check of the authentic NGSIM trace mode in the running console."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

ok = True


def check(name, cond, detail=""):
    global ok
    if cond:
        print(f"  PASS  {name}")
    else:
        ok = False
        print(f"  FAIL  {name}  {detail}")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return json.loads(r.read().decode())


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


print("=" * 74)
print("LIVE AUTHENTIC NGSIM TRACE MODE")
print("=" * 74)

post("/api/control", {"action": "stop"})
r = post("/api/configure", {"mobility_mode": "authentic_trace", "rebuild": True})
print("configure ->", r.get("config", {}).get("mobility_mode"))

k = get("/api/state")["kpis"]
src = k["mobility_source"]
print("mobility mode   :", k["mobility_mode"])
print("source label    :", src["label"])
print("authentic flag  :", src["authentic"])
print("road length (m) :", round(k.get("simulation_time_sec", 0), 1), "s of sim time")
check("the console reports the authentic trace", src["authentic"] is True, str(src))
check("the active dataset is the processed authentic trace",
      "authentic" in str(src.get("dataset", "")), str(src.get("dataset")))

post("/api/control", {"action": "start"})
time.sleep(4.0)

st = get("/api/state")
k = st["kpis"]
vehicles = st["vehicles"]
active = [v for v in vehicles if v.get("active")]
with_source = [v for v in active if v.get("source_vehicle_id") is not None]

print("\nafter 4 s of replay:")
print("  vehicles in the payload :", len(vehicles))
print("  actively following a source trajectory :", len(active))
print("  with a source vehicle id               :", len(with_source))
print("  tasks executed          :", k["total_tasks"])
print("  simulated time (s)       :", round(k["simulation_time_sec"], 1))

check("some vehicles follow a recorded source trajectory", len(with_source) > 0)
check("absent vehicles are marked inactive", len(active) < len(vehicles),
      f"active={len(active)} of {len(vehicles)}")
check("the simulation advanced with the authentic trace", k["total_tasks"] > 0,
      str(k["total_tasks"]))

# Sample the fleet again and confirm the bindings change over time (arrivals).
ids_a = {v["vehicle_id"]: v.get("source_vehicle_id") for v in vehicles}
time.sleep(4.0)
vehicles2 = get("/api/state")["vehicles"]
ids_b = {v["vehicle_id"]: v.get("source_vehicle_id") for v in vehicles2}
changed = sum(1 for kk in ids_b if ids_a.get(kk) != ids_b[kk])
print("  slots whose source binding changed in 4 s :", changed)
check("source bindings evolve as vehicles arrive and depart", changed > 0, str(changed))

post("/api/control", {"action": "stop"})
print("\n" + "=" * 74)
print("LIVE AUTHENTIC TRACE: " + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
print("=" * 74)
raise SystemExit(0 if ok else 1)
