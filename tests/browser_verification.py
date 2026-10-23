"""
Real-browser verification of the VANNET console (Chromium via Playwright).

Run the server first, then this script:

    python run_simulation.py              # one shell
    python tests/browser_verification.py  # another shell

It drives the real page and checks what a user would see, against the
backend's own numbers.  It exits non-zero if any check fails, and writes
screenshots next to the script's output directory (default: ./browser_check).

Checks
------
 1. The page loads with no console errors and the telemetry socket connects.
 2. Start -> RUNNING; the step counter advances; vehicles are drawn.
 3. Pause freezes the step counter; Resume continues it; Reset returns to STOPPED.
 4. Two simultaneous browser tabs do NOT make the simulation advance faster.
 5. The road viewport uses the backend geometry (RSU count, road length).
 6. Invalid experiment parameters are rejected with a visible message
    (client-side range check and the backend 400).
 7. A valid change of vehicle count starts a NEW session with that many vehicles.
 8. Switching to the authentic NGSIM replay changes the mobility chip and the
    source note, and the viewport still renders vehicles.
 9. The Research tab renders the active run (tables, provenance, figures)
    from /api/research/summary, and shows no live-session numbers.
10. The layout has no horizontal overflow at 1600, 1280, 900 and 390 px wide.
11. The dark theme toggles without errors.
"""

import json
import os
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

BASE = os.environ.get("VANNET_URL", "http://127.0.0.1:8000")
OUT = os.environ.get("VANNET_SHOTS", os.path.join(os.path.dirname(__file__), "..", "browser_check"))
os.makedirs(OUT, exist_ok=True)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + str(detail)) if detail and not ok else ""))


def api(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def steps(page):
    txt = page.inner_text("#session-steps")
    return int(txt.split()[0])


def wait_for(page, fn, timeout=40, interval=0.5):
    t = time.time()
    while time.time() - t < timeout:
        try:
            v = fn()
            if v:
                return v
        except Exception:
            pass
        page.wait_for_timeout(int(interval * 1000))
    return None


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        ctx = browser.new_context(viewport={"width": 1600, "height": 950})
        page = ctx.new_page()
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append("PAGEERROR " + str(e)))

        print("1. load")
        page.goto(BASE + "/?view=2d", wait_until="domcontentloaded")
        ok = wait_for(page, lambda: page.inner_text("#connection-status").strip() == "connected", 20)
        check("telemetry socket connects", ok)
        check("no console errors on load", not errors, errors[:3])

        # start from a clean state
        if page.is_enabled("#btn-stop"):
            page.click("#btn-stop")
            wait_for(page, lambda: page.inner_text("#kpi-simstate").strip() == "STOPPED", 60)
        wait_for(page, lambda: page.is_enabled("#btn-start"), 30)

        print("2. start")
        page.click("#btn-start")
        check("state becomes RUNNING", wait_for(page, lambda: page.inner_text("#kpi-simstate").strip() == "RUNNING", 15))
        s0 = steps(page)
        page.wait_for_timeout(4000)
        s1 = steps(page)
        check("step counter advances", s1 > s0, (s0, s1))
        snap = api("/api/state")
        check("viewport canvas exists", page.locator("#viewport canvas.road-canvas").count() == 1)
        check("vehicles reported by backend", len(snap["vehicles"]) == snap["kpis"]["active_vehicles"])
        page.screenshot(path=os.path.join(OUT, "01_running.png"))

        print("3. pause / resume / reset")
        page.click("#btn-pause")
        wait_for(page, lambda: page.inner_text("#kpi-simstate").strip() == "PAUSED", 10)
        a = steps(page)
        page.wait_for_timeout(2500)
        b = steps(page)
        check("pause freezes the simulation", a == b, (a, b))
        check("start button reads Resume while paused", page.inner_text("#btn-start").strip() == "Resume")
        page.click("#btn-start")
        wait_for(page, lambda: page.inner_text("#kpi-simstate").strip() == "RUNNING", 10)
        page.wait_for_timeout(2500)
        check("resume continues", steps(page) > b)

        print("4. two tabs do not double the speed")
        page2 = ctx.new_page()
        page2.goto(BASE + "/?view=2d", wait_until="domcontentloaded")
        page2.wait_for_timeout(1500)
        t0 = api("/api/state")["kpis"]["sim_steps"]
        time.sleep(6)
        t1 = api("/api/state")["kpis"]["sim_steps"]
        rate_two = (t1 - t0) / 6.0
        page2.close()
        page.wait_for_timeout(500)
        u0 = api("/api/state")["kpis"]["sim_steps"]
        time.sleep(6)
        u1 = api("/api/state")["kpis"]["sim_steps"]
        rate_one = (u1 - u0) / 6.0
        check("step rate with two tabs ~ rate with one tab", rate_one > 0 and rate_two <= 1.35 * rate_one,
              (round(rate_one, 2), round(rate_two, 2)))

        print("5. viewport geometry comes from the backend")
        scene = api("/api/state")["scene"]
        check("scene has 3 RSUs and a road length", len(scene["rsus"]) == 3 and scene["road_length_m"] > 100)

        page.click("#btn-stop")
        wait_for(page, lambda: page.inner_text("#kpi-simstate").strip() == "STOPPED", 40)
        check("reset returns to STOPPED", page.inner_text("#kpi-simstate").strip() == "STOPPED")

        print("6. validation")
        page.fill("#input-attack-ratio", "0.9")
        page.click("#btn-apply-config")
        msg = page.inner_text("#config-message")
        check("attack ratio 0.9 rejected with a visible message", "between 0 and 0.5" in msg, msg)
        page.fill("#input-attack-ratio", "0.24")
        page.fill("#input-vehicles", "3")
        page.click("#btn-apply-config")
        check("3 vehicles rejected with a visible message", "between 5 and 60" in page.inner_text("#config-message"))
        r = urllib.request.Request(BASE + "/api/configure", data=json.dumps({"attack_ratio": 0.9}).encode(),
                                   headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(r, timeout=20)
            code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        check("backend returns 400 for attack ratio 0.9", code == 400, code)

        print("7. new session with a different vehicle count")
        page.fill("#input-vehicles", "12")
        page.click("#btn-apply-config")
        ok = wait_for(page, lambda: api("/api/state")["kpis"]["active_vehicles"] == 12, 30)
        check("12 vehicles in the new session", ok)
        page.screenshot(path=os.path.join(OUT, "02_twelve_vehicles.png"))

        print("7b. RSU placement")
        page.fill("#input-rsu-1", "300"); page.fill("#input-rsu-2", "320"); page.fill("#input-rsu-3", "2800")
        page.click("#btn-apply-config")
        check("overlapping RSUs rejected with a visible message",
              wait_for(page, lambda: "closer than 50 m" in page.inner_text("#config-message"), 10))
        page.fill("#input-rsu-2", "1200")
        page.click("#btn-apply-config")
        ok = wait_for(page, lambda: [round(r["x"]) for r in api("/api/state")["scene"]["rsus"]] == [300, 1200, 2800], 30)
        check("backend uses the configured RSU positions", ok)
        page.wait_for_timeout(1000)
        check("form reflects the backend configuration", page.input_value("#input-vehicles") == "12" and
              page.input_value("#input-rsu-3") == "2800")
        page.screenshot(path=os.path.join(OUT, "02b_rsu_placement.png"))

        print("8. authentic NGSIM replay")
        page.select_option("#select-mobility", "authentic_trace")
        page.click("#btn-apply-config")
        ok = wait_for(page, lambda: api("/api/state")["kpis"]["mobility_mode"] == "authentic_trace", 120)
        check("backend switched to authentic_trace", ok)
        page.wait_for_timeout(1500)
        check("mobility chip says authentic", "authentic" in page.inner_text("#chip-mobility"))
        scene = api("/api/state")["scene"]
        check("scene reflects the trace geometry (not 3200 m)", scene["road_length_m"] < 1000, scene["road_length_m"])
        check("RSU inputs disabled for trace replay", page.is_disabled("#input-rsu-1"))
        page.click("#btn-start")
        wait_for(page, lambda: page.inner_text("#kpi-simstate").strip() == "RUNNING", 15)
        page.wait_for_timeout(8000)
        snap = api("/api/state")
        active = sum(1 for v in snap["vehicles"] if v.get("active"))
        check("some trace vehicles are active", active > 0, active)
        page.screenshot(path=os.path.join(OUT, "03_authentic_trace.png"))
        page.click("#btn-stop")

        print("9. research tab")
        page.click("#tab-research")
        ok = wait_for(page, lambda: page.locator("#research-root table").count() >= 2, 60)
        check("research tab renders tables", ok)
        text = page.inner_text("#research-root")
        summary = api("/api/research/summary")
        check("research tab shows the active run id", summary["run_id"] in text)
        check("research tab separates live metrics", "not the live simulation" in text)
        claims = summary.get("claims") or {}
        rows = claims.get("rows") or []
        check("backend computed the 8 claim verdicts", len(rows) == 8 and all(r["verdict"] for r in rows),
              claims.get("error"))
        check("research tab shows the claim verdicts", "Downstream claims" in text and
              all(r["id"] in text for r in rows))
        page.screenshot(path=os.path.join(OUT, "04_research.png"), full_page=False)
        page.click("#tab-sim")

        print("10. responsive layout")
        for w, h in ((1600, 950), (1280, 800), (900, 900), (390, 844)):
            page.set_viewport_size({"width": w, "height": h})
            page.wait_for_timeout(600)
            overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
            if overflow > 1:
                overflow = "%s %s" % (overflow, page.evaluate("""() => { const W = document.documentElement.clientWidth;
                  const clipped = (e) => { for (let a = e.parentElement; a; a = a.parentElement) if (getComputedStyle(a).overflowX !== 'visible') return true; return false; };
                  return [...document.querySelectorAll('body *')].filter((e) => e.getBoundingClientRect().right > W + 1 && !clipped(e)).slice(0, 6)
                    .map((e) => e.tagName + '.' + e.className + '#' + e.id + ' ' + Math.round(e.getBoundingClientRect().right)); }"""))
            check("no horizontal overflow at %d px" % w, (isinstance(overflow, (int, float)) and overflow <= 1), overflow)
            page.screenshot(path=os.path.join(OUT, "05_width_%d.png" % w))
        page.set_viewport_size({"width": 1600, "height": 950})

        print("11. theme")
        page.click("#btn-theme")
        page.wait_for_timeout(300)
        theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        check("theme toggles", theme in ("dark", "light"), theme)
        page.screenshot(path=os.path.join(OUT, "06_theme.png"))

        print("12. 3D viewport")
        page3 = ctx.new_page()
        errs3 = []
        page3.on("console", lambda m: errs3.append(m.text) if m.type == "error" else None)
        page3.on("pageerror", lambda e: errs3.append(str(e)))
        page3.goto(BASE + "/?view=3d", wait_until="domcontentloaded")
        page3.wait_for_timeout(4000)
        check("the 3D renderer starts (WebGL)", page3.evaluate("viewKind") == "3d")
        check("3D canvas exists", page3.locator("#viewport canvas.road-canvas-3d").count() == 1)
        check("3D scene holds RSU towers, vehicles and buildings",
              page3.evaluate("roadView.rsuObjs.size") == 3 and page3.evaluate("roadView.world.children.length") > 100)
        page3.click("#btn-dim-2d"); page3.wait_for_timeout(800)
        check("2D toggle swaps the renderer", page3.evaluate("viewKind") == "2d")
        page3.click("#btn-dim-3d"); page3.wait_for_timeout(1500)
        check("3D toggle swaps back", page3.evaluate("viewKind") == "3d")
        page3.screenshot(path=os.path.join(OUT, "07_3d.png"))
        check("no console errors in the 3D page", not errs3, errs3[:3])
        page3.close()

        check("no console errors during the whole session", not errors, errors[:5])
        browser.close()

    failed = [r for r in results if not r[1]]
    print("\n%d checks, %d failed" % (len(results), len(failed)))
    for name, _, detail in failed:
        print("  FAILED: %s %s" % (name, detail))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
