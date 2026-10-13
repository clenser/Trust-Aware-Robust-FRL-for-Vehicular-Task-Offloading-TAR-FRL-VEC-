"""
Three.js 3D WebGL vehicular edge-network simulation renderer.
Generates an interactive, production-grade 3D visualization featuring:
- Multi-lane 3D highway with lane markings, road shoulders, and terrain
- 3D composite vehicle meshes (chassis, cabin, wheels, headlights)
- 3D roadside communication towers (RSUs) with visible radio coverage domes (R_comm = 300m)
- Dynamic 3D curved offloading laser beams with animated traveling packet pulses
- Local onboard CPU execution visual auras
- OrbitControls (rotate, zoom, pan) with camera view presets (Isometric, Chase, Side, Top-Down)
- Client-side 60 FPS timeline scrubber and play/pause animation loop
- Interactive raycaster for clicking and inspecting vehicles and RSUs
- Graceful WebGL error handling and 100% offline local asset bundling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from environment.config import EnvConfig
from dashboard.simulator_engine import StepSnapshot

STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_bundled_js_assets() -> tuple[str, str]:
    """Load locally bundled Three.js and OrbitControls libraries."""
    three_path = STATIC_DIR / "three.min.js"
    orbit_path = STATIC_DIR / "OrbitControls.js"
    
    three_src = three_path.read_text(encoding="utf-8") if three_path.exists() else ""
    orbit_src = orbit_path.read_text(encoding="utf-8") if orbit_path.exists() else ""
    return three_src, orbit_src


def build_3d_simulation_html(
    snapshots: list[StepSnapshot],
    summary: dict,
    cfg: EnvConfig
) -> str:
    """Build standalone, self-contained Three.js 3D simulation HTML markup."""
    three_js, orbit_js = load_bundled_js_assets()

    # Serialize snapshots into client-side JSON trace
    steps_data = []
    for s in snapshots:
        steps_data.append({
            "step_idx": s.step_idx,
            "task_id": s.task_id,
            "arrival_s": s.arrival_s,
            "requester_vid": s.requester_vid,
            "data_kb": s.data_bits / 8e3,
            "cycles_m": s.cycles / 1e6,
            "deadline_s": s.deadline_s,
            "priority": s.priority,
            "action": s.action,
            "dest_kind": s.dest_kind,
            "dest_id": s.dest_id,
            "dest_name": s.dest_name,
            "dest_trust": s.dest_trust,
            "tx_latency_s": s.tx_latency_s,
            "exe_latency_s": s.exe_latency_s,
            "total_latency_s": s.total_latency_s,
            "energy_j": s.energy_j,
            "success": s.success,
            "deadline_missed": s.deadline_missed,
            "failed": s.failed,
            "updated_trust": s.updated_trust,
            "vehicles": s.vehicles,
            "edges": s.edges,
            "candidates": s.candidates,
        })

    trace_json = json.dumps({
        "road_length_m": getattr(cfg, "road_length_m", 1000.0),
        "comm_range_m": getattr(cfg, "comm_range_m", 300.0),
        "num_vehicles": getattr(cfg, "num_vehicles", len(snapshots[0].vehicles if snapshots else 6)),
        "num_edges": getattr(cfg, "num_edge_nodes", len(snapshots[0].edges if snapshots else 2)),
        "summary": summary,
        "steps": steps_data,
    })

    # CDN fallback scripts if local assets were empty
    cdn_fallback_three = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>' if not three_js else f"<script>{three_js}</script>"
    cdn_fallback_orbit = '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>' if not orbit_js else f"<script>{orbit_js}</script>"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VANET FRL 3D Interactive Simulation</title>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }}
    body {{
      background-color: #0b0f19;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      z-index: 1;
    }}

    /* Top HUD Header */
    .hud-top {{
      position: absolute;
      top: 12px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
      pointer-events: none;
    }}
    .hud-badge {{
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(51, 65, 85, 0.7);
      border-radius: 8px;
      padding: 8px 14px;
      font-size: 0.82rem;
      font-weight: 600;
      color: #94a3b8;
      pointer-events: auto;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .hud-badge strong {{
      color: #38bdf8;
    }}
    .hud-badge .status-pill {{
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: bold;
      text-transform: uppercase;
    }}
    .pill-success {{ background: #065f46; color: #34d399; border: 1px solid #10b981; }}
    .pill-danger {{ background: #7f1d1d; color: #f87171; border: 1px solid #ef4444; }}
    .pill-warning {{ background: #78350f; color: #fbbf24; border: 1px solid #f59e0b; }}

    /* Camera View Presets */
    .cam-presets {{
      display: flex;
      gap: 6px;
      pointer-events: auto;
    }}
    .cam-btn {{
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(51, 65, 85, 0.8);
      color: #cbd5e1;
      padding: 6px 11px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      backdrop-filter: blur(4px);
    }}
    .cam-btn:hover {{
      background: #1e293b;
      border-color: #38bdf8;
      color: #38bdf8;
    }}

    /* Bottom Playback & Scrubber Controls */
    .hud-bottom {{
      position: absolute;
      bottom: 14px;
      left: 14px;
      right: 14px;
      background: rgba(15, 23, 42, 0.92);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(51, 65, 85, 0.75);
      border-radius: 10px;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      gap: 16px;
      z-index: 10;
      box-shadow: 0 4px 16px rgba(0,0,0,0.45);
    }}
    .btn-ctrl {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #f1f5f9;
      width: 34px;
      height: 34px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .btn-ctrl:hover {{
      background: #0284c7;
      border-color: #38bdf8;
    }}
    .scrubber-wrap {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .scrubber-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      color: #94a3b8;
      font-weight: 500;
    }}
    .slider {{
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      cursor: pointer;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #38bdf8;
      border: 2px solid #0f172a;
      cursor: pointer;
      box-shadow: 0 0 6px #0284c7;
    }}
    .speed-select {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #cbd5e1;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      outline: none;
      cursor: pointer;
    }}

    /* Floating Inspector Card */
    .inspector-panel {{
      position: absolute;
      top: 60px;
      right: 14px;
      width: 290px;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(51, 65, 85, 0.8);
      border-radius: 8px;
      padding: 14px;
      z-index: 10;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      font-size: 0.8rem;
    }}
    .inspector-title {{
      font-size: 0.82rem;
      font-weight: 700;
      color: #38bdf8;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-bottom: 1px solid #334155;
      padding-bottom: 6px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }}
    .prop-row {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      color: #cbd5e1;
    }}
    .prop-val {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-weight: 600;
      color: #f8fafc;
    }}

    /* Legend Box */
    .legend-box {{
      position: absolute;
      bottom: 74px;
      left: 14px;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(51, 65, 85, 0.6);
      border-radius: 6px;
      padding: 6px 12px;
      display: flex;
      gap: 12px;
      font-size: 0.72rem;
      color: #94a3b8;
      z-index: 10;
      backdrop-filter: blur(6px);
    }}
    .leg-item {{ display: flex; align-items: center; gap: 5px; }}
    .leg-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

    #webgl-error {{
      display: none;
      position: absolute;
      top: 40%;
      left: 20%;
      right: 20%;
      background: #7f1d1d;
      color: #fef2f2;
      padding: 20px;
      border-radius: 8px;
      text-align: center;
      z-index: 99;
      border: 1px solid #ef4444;
    }}
  </style>
  {cdn_fallback_three}
  {cdn_fallback_orbit}
</head>
<body>
  <div id="canvas-container"></div>
  <div id="webgl-error">
    <h3>WebGL Rendering Unavailable</h3>
    <p>Hardware acceleration or WebGL is required for the 3D vehicular simulation.</p>
  </div>

  <!-- Top Header HUD -->
  <div class="hud-top">
    <div class="hud-badge">
      <span>Step: <strong id="hud-step">1 / 30</strong></span>
      <span>Task: <strong id="hud-task">#0</strong></span>
      <span>Req: <strong id="hud-req">V0</strong> ➔ <strong id="hud-dest">RSU 0</strong></span>
      <span id="hud-outcome" class="status-pill pill-success">SUCCESS</span>
    </div>
    <div class="cam-presets">
      <button class="cam-btn" onclick="setCameraPreset('iso')">🎥 Isometric</button>
      <button class="cam-btn" onclick="setCameraPreset('chase')">🚗 Chase Cam</button>
      <button class="cam-btn" onclick="setCameraPreset('side')">🔭 Side View</button>
      <button class="cam-btn" onclick="setCameraPreset('top')">🗺️ Top-Down</button>
      <button class="cam-btn" onclick="resetCamera()">🔄 Reset</button>
    </div>
  </div>

  <!-- Entity Inspector Panel -->
  <div class="inspector-panel" id="inspector-panel">
    <div class="inspector-title">
      <span id="insp-header">Step Telemetry</span>
      <span style="font-size:0.7rem; color:#64748b;">(Click Object)</span>
    </div>
    <div class="prop-row"><span>Requester:</span><span class="prop-val" id="insp-req">Vehicle 0</span></div>
    <div class="prop-row"><span>Destination:</span><span class="prop-val" id="insp-dest">RSU 0 (Edge)</span></div>
    <div class="prop-row"><span>Data Size:</span><span class="prop-val" id="insp-size">42.0 KB</span></div>
    <div class="prop-row"><span>Compute Demand:</span><span class="prop-val" id="insp-cycles">12.5 Mcyc</span></div>
    <div class="prop-row"><span>Transmission:</span><span class="prop-val" id="insp-tx">0.045 s</span></div>
    <div class="prop-row"><span>Execution:</span><span class="prop-val" id="insp-exe">0.120 s</span></div>
    <div class="prop-row"><span>Total Latency:</span><span class="prop-val" id="insp-tot">0.165 s</span></div>
    <div class="prop-row"><span>Deadline:</span><span class="prop-val" id="insp-dl">0.850 s</span></div>
    <div class="prop-row"><span>Energy Cost:</span><span class="prop-val" id="insp-energy">1.45 J</span></div>
    <div class="prop-row"><span>Dest Trust Score:</span><span class="prop-val" id="insp-trust">0.850</span></div>
  </div>

  <!-- Legend -->
  <div class="legend-box">
    <div class="leg-item"><div class="leg-dot" style="background:#2563eb;"></div> Vehicle</div>
    <div class="leg-item"><div class="leg-dot" style="background:#f59e0b;"></div> Requester</div>
    <div class="leg-item"><div class="leg-dot" style="background:#0284c7;"></div> RSU Tower</div>
    <div class="leg-item"><div class="leg-dot" style="background:#06b6d4;"></div> R_comm (300m)</div>
    <div class="leg-item"><div class="leg-dot" style="background:#10b981;"></div> Success Offload</div>
    <div class="leg-item"><div class="leg-dot" style="background:#ef4444;"></div> Deadline Miss</div>
    <div class="leg-item"><div class="leg-dot" style="background:#8b5cf6;"></div> Local CPU</div>
  </div>

  <!-- Bottom Playback Timeline HUD -->
  <div class="hud-bottom">
    <button class="btn-ctrl" onclick="stepFirst()" title="First Step">⏮️</button>
    <button class="btn-ctrl" onclick="stepPrev()" title="Previous Step">◀️</button>
    <button class="btn-ctrl" id="btn-play" onclick="togglePlay()" title="Play/Pause">▶️</button>
    <button class="btn-ctrl" onclick="stepNext()" title="Next Step">▶️</button>
    <button class="btn-ctrl" onclick="stepLast()" title="Last Step">⏭️</button>

    <div class="scrubber-wrap">
      <div class="scrubber-label">
        <span>Timeline Scrubber</span>
        <span id="scrub-step-txt">Step 1 / 30</span>
      </div>
      <input type="range" class="slider" id="timeline-slider" min="0" max="29" value="0" oninput="onScrub(this.value)">
    </div>

    <select class="speed-select" id="speed-select" onchange="onSpeedChange(this.value)">
      <option value="1500">0.5x Speed</option>
      <option value="800" selected>1.0x Speed</option>
      <option value="400">2.0x Speed</option>
      <option value="200">4.0x Speed</option>
    </select>
  </div>

  <script>
    // Verify WebGL availability
    if (!window.WebGLRenderingContext) {{
      document.getElementById('webgl-error').style.display = 'block';
    }}

    // Simulation Data Trace injected from Python
    const simTrace = {trace_json};
    const totalSteps = simTrace.steps.length;
    let currentStepIdx = 0;
    let isPlaying = false;
    let playTimer = null;
    let playIntervalMs = 800;

    document.getElementById('timeline-slider').max = Math.max(0, totalSteps - 1);

    // Three.js Scene Setup
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f19);
    scene.fog = new THREE.FogExp2(0x0b0f19, 0.0018);

    const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 1, 3500);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2 - 0.02; // Prevent camera clipping through ground
    controls.minDistance = 20;
    controls.maxDistance = 1200;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x94a3b8, 0.65);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 0.85);
    sunLight.position.set(200, 350, 150);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 10;
    sunLight.shadow.camera.far = 1200;
    const shadowD = 400;
    sunLight.shadow.camera.left = -shadowD;
    sunLight.shadow.camera.right = shadowD;
    sunLight.shadow.camera.top = shadowD;
    sunLight.shadow.camera.bottom = -shadowD;
    scene.add(sunLight);

    const blueRimLight = new THREE.DirectionalLight(0x0284c7, 0.35);
    blueRimLight.position.set(-200, 100, -200);
    scene.add(blueRimLight);

    // -------------------------------------------------------------
    // 3D Environment: Road, Terrain, Markers
    // -------------------------------------------------------------
    const roadLengthM = simTrace.road_length_m || 1000.0;
    const roadScale = 0.8; // 1 meter = 0.8 Three.js units
    const roadLength3D = roadLengthM * roadScale;
    const halfRoad3D = roadLength3D / 2;

    // Coordinate mapping: meter x in [0, roadLengthM] -> 3D X in [-halfRoad3D, +halfRoad3D]
    function toWorldX(xMeters) {{
      const wrapped = (xMeters % roadLengthM + roadLengthM) % roadLengthM;
      return (wrapped - roadLengthM / 2) * roadScale;
    }}

    // Ground Terrain
    const groundGeo = new THREE.PlaneGeometry(roadLength3D + 400, 600, 20, 20);
    groundGeo.rotateX(-Math.PI / 2);
    const groundMat = new THREE.MeshStandardMaterial({{
      color: 0x090d16,
      roughness: 0.95,
      metalness: 0.1
    }});
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.position.y = -0.1;
    ground.receiveShadow = true;
    scene.add(ground);

    // Subtle Ground Grid
    const gridHelper = new THREE.GridHelper(roadLength3D + 200, 40, 0x1e293b, 0x0f172a);
    gridHelper.position.y = 0;
    scene.add(gridHelper);

    // Highway Asphalt Bed (3 Lanes: width = 24 units, Z in [-12, +12])
    const roadGeo = new THREE.PlaneGeometry(roadLength3D, 24);
    roadGeo.rotateX(-Math.PI / 2);
    const roadMat = new THREE.MeshStandardMaterial({{
      color: 0x182030,
      roughness: 0.8,
      metalness: 0.2
    }});
    const road = new THREE.Mesh(roadGeo, roadMat);
    road.position.y = 0.1;
    road.receiveShadow = true;
    scene.add(road);

    // Yellow Shoulders
    const shoulderMat = new THREE.MeshBasicMaterial({{ color: 0xeab308 }});
    const s1Geo = new THREE.PlaneGeometry(roadLength3D, 0.4);
    s1Geo.rotateX(-Math.PI / 2);
    const shoulderTop = new THREE.Mesh(s1Geo, shoulderMat);
    shoulderTop.position.set(0, 0.15, -11.8);
    scene.add(shoulderTop);

    const shoulderBottom = new THREE.Mesh(s1Geo, shoulderMat);
    shoulderBottom.position.set(0, 0.15, 11.8);
    scene.add(shoulderBottom);

    // White Dashed Lane Dividers
    // Lane 0: Z = -7.0, Lane 1: Z = 0.0, Lane 2: Z = +7.0
    // Dividers at Z = -3.8 and Z = +3.8
    const dashMat = new THREE.MeshBasicMaterial({{ color: 0xe2e8f0 }});
    const dashGeo = new THREE.PlaneGeometry(3.5, 0.3);
    dashGeo.rotateX(-Math.PI / 2);

    for (let x = -halfRoad3D + 5; x < halfRoad3D - 5; x += 9) {{
      const d1 = new THREE.Mesh(dashGeo, dashMat);
      d1.position.set(x, 0.16, -3.8);
      scene.add(d1);

      const d2 = new THREE.Mesh(dashGeo, dashMat);
      d2.position.set(x, 0.16, 3.8);
      scene.add(d2);
    }}

    // Roadside Distance Posts
    for (let m = 0; m <= roadLengthM; m += 200) {{
      const wx = toWorldX(m);
      const postGeo = new THREE.CylinderGeometry(0.2, 0.2, 3.0, 8);
      const postMat = new THREE.MeshStandardMaterial({{ color: 0x475569 }});
      const post = new THREE.Mesh(postGeo, postMat);
      post.position.set(wx, 1.5, 13.5);
      scene.add(post);
    }}

    // Lane Z coordinates mapping: lane in 0..2
    const laneZ = [-7.0, 0.0, 7.0];

    // -------------------------------------------------------------
    // 3D Vehicle Models Construction
    // -------------------------------------------------------------
    const vehicleMeshes = new Map();

    function build3DVehicle(vid) {{
      const group = new THREE.Group();

      // Lower Chassis Body
      const bodyGeo = new THREE.BoxGeometry(6.2, 1.3, 3.0);
      const bodyMat = new THREE.MeshStandardMaterial({{
        color: 0x2563eb,
        roughness: 0.3,
        metalness: 0.5
      }});
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      body.position.y = 1.0;
      body.castShadow = true;
      body.receiveShadow = true;
      group.add(body);

      // Cabin / Windshield
      const cabinGeo = new THREE.BoxGeometry(3.4, 1.1, 2.6);
      const cabinMat = new THREE.MeshStandardMaterial({{
        color: 0x0f172a,
        roughness: 0.1,
        metalness: 0.9,
        transparent: true,
        opacity: 0.85
      }});
      const cabin = new THREE.Mesh(cabinGeo, cabinMat);
      cabin.position.set(-0.3, 2.0, 0);
      cabin.castShadow = true;
      group.add(cabin);

      // Four Rubber Wheels
      const wheelGeo = new THREE.CylinderGeometry(0.65, 0.65, 0.55, 14);
      wheelGeo.rotateX(Math.PI / 2);
      const wheelMat = new THREE.MeshStandardMaterial({{ color: 0x111827, roughness: 0.9 }});
      const wheelPositions = [
        [1.9, 0.65, 1.5],
        [1.9, 0.65, -1.5],
        [-1.9, 0.65, 1.5],
        [-1.9, 0.65, -1.5]
      ];
      wheelPositions.forEach(p => {{
        const w = new THREE.Mesh(wheelGeo, wheelMat);
        w.position.set(...p);
        w.castShadow = true;
        group.add(w);
      }});

      // Headlights (+X forward)
      const hlGeo = new THREE.BoxGeometry(0.15, 0.25, 0.45);
      const hlMat = new THREE.MeshBasicMaterial({{ color: 0xfef08a }});
      const hl1 = new THREE.Mesh(hlGeo, hlMat);
      hl1.position.set(3.15, 1.1, 0.95);
      const hl2 = new THREE.Mesh(hlGeo, hlMat);
      hl2.position.set(3.15, 1.1, -0.95);
      group.add(hl1);
      group.add(hl2);

      // Taillights (-X rear)
      const tlMat = new THREE.MeshBasicMaterial({{ color: 0xef4444 }});
      const tl1 = new THREE.Mesh(hlGeo, tlMat);
      tl1.position.set(-3.15, 1.1, 0.95);
      const tl2 = new THREE.Mesh(hlGeo, tlMat);
      tl2.position.set(-3.15, 1.1, -0.95);
      group.add(tl1);
      group.add(tl2);

      // Pulsing Requester Halo Ring
      const haloGeo = new THREE.RingGeometry(2.4, 2.9, 32);
      haloGeo.rotateX(-Math.PI / 2);
      const haloMat = new THREE.MeshBasicMaterial({{
        color: 0xf59e0b,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.0
      }});
      const halo = new THREE.Mesh(haloGeo, haloMat);
      halo.position.y = 3.2;
      group.add(halo);

      // Local CPU Energy Aura Rings
      const auraGeo = new THREE.TorusGeometry(3.2, 0.15, 8, 24);
      const auraMat = new THREE.MeshBasicMaterial({{
        color: 0x8b5cf6,
        transparent: true,
        opacity: 0.0
      }});
      const aura = new THREE.Mesh(auraGeo, auraMat);
      aura.position.y = 1.6;
      aura.rotation.x = Math.PI / 2;
      group.add(aura);

      group.userData = {{
        type: 'vehicle',
        vid: vid,
        bodyMat: bodyMat,
        halo: halo,
        haloMat: haloMat,
        aura: aura,
        auraMat: auraMat
      }};
      scene.add(group);
      return group;
    }}

    // -------------------------------------------------------------
    // 3D Roadside Unit (RSU) Tower Construction
    // -------------------------------------------------------------
    const rsuMeshes = new Map();
    const commRadius3D = (simTrace.comm_range_m || 300.0) * roadScale;

    function build3DRsuTower(eid, edgeData) {{
      const group = new THREE.Group();
      const wx = toWorldX(edgeData.x_m);
      const rsuZ = -18.0; // Positioned along north roadside

      // Concrete Foundation
      const baseGeo = new THREE.CylinderGeometry(3.6, 4.4, 1.6, 8);
      const baseMat = new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.9 }});
      const base = new THREE.Mesh(baseGeo, baseMat);
      base.position.y = 0.8;
      base.receiveShadow = true;
      group.add(base);

      // Lattice Steel Truss Mast
      const mastGeo = new THREE.CylinderGeometry(0.9, 2.0, 26.0, 6, 4, true);
      const mastMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, wireframe: true }});
      const mast = new THREE.Mesh(mastGeo, mastMat);
      mast.position.y = 14.0;
      group.add(mast);

      // Central Antenna Mast
      const poleGeo = new THREE.CylinderGeometry(0.4, 0.4, 28.0, 8);
      const poleMat = new THREE.MeshStandardMaterial({{ color: 0x0284c7, metalness: 0.8 }});
      const pole = new THREE.Mesh(poleGeo, poleMat);
      pole.position.y = 15.0;
      group.add(pole);

      // Directional Microwave Dishes
      const dishGeo = new THREE.CylinderGeometry(1.7, 1.7, 0.8, 16);
      dishGeo.rotateZ(Math.PI / 2);
      const dishMat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8, metalness: 0.5 }});
      const d1 = new THREE.Mesh(dishGeo, dishMat);
      d1.position.set(1.5, 22.0, 0);
      const d2 = new THREE.Mesh(dishGeo, dishMat);
      d2.position.set(-1.5, 24.5, 0);
      group.add(d1);
      group.add(d2);

      // Top Beacon Light
      const beaconGeo = new THREE.SphereGeometry(0.8, 16, 16);
      const beaconMat = new THREE.MeshBasicMaterial({{ color: 0x06b6d4 }});
      const beacon = new THREE.Mesh(beaconGeo, beaconMat);
      beacon.position.y = 29.5;
      group.add(beacon);

      // Translucent Wireless Coverage Dome (Hemisphere: R = 300m)
      const domeGeo = new THREE.SphereGeometry(commRadius3D, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2);
      const domeMat = new THREE.MeshStandardMaterial({{
        color: 0x06b6d4,
        transparent: true,
        opacity: 0.07,
        wireframe: true,
        side: THREE.DoubleSide
      }});
      const dome = new THREE.Mesh(domeGeo, domeMat);
      dome.position.y = 0;
      group.add(dome);

      // Ground Coverage Boundary Ring
      const ringGeo = new THREE.RingGeometry(commRadius3D - 1.5, commRadius3D, 64);
      ringGeo.rotateX(-Math.PI / 2);
      const ringMat = new THREE.MeshBasicMaterial({{
        color: 0x06b6d4,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide
      }});
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.y = 0.22;
      group.add(ring);

      group.position.set(wx, 0, rsuZ);
      group.userData = {{
        type: 'rsu',
        eid: eid,
        worldX: wx,
        worldZ: rsuZ,
        antennaY: 28.0,
        beacon: beacon
      }};
      scene.add(group);
      return group;
    }}

    // -------------------------------------------------------------
    // Dynamic 3D Offload Laser Beam & Packet Pulses
    // -------------------------------------------------------------
    let offloadCurve = null;
    let offloadTubeMesh = null;
    const packetPulses = [];
    const numPackets = 4;

    const packetGeo = new THREE.SphereGeometry(0.8, 12, 12);
    const packetMat = new THREE.MeshBasicMaterial({{ color: 0x10b981 }});
    for (let i = 0; i < numPackets; i++) {{
      const p = new THREE.Mesh(packetGeo, packetMat);
      p.visible = false;
      scene.add(p);
      packetPulses.push(p);
    }}

    function clearOffloadVisuals() {{
      if (offloadTubeMesh) {{
        scene.remove(offloadTubeMesh);
        offloadTubeMesh.geometry.dispose();
        offloadTubeMesh.material.dispose();
        offloadTubeMesh = null;
      }}
      offloadCurve = null;
      packetPulses.forEach(p => p.visible = false);
    }}

    function createOffloadBeam(startPos, endPos, isSuccess) {{
      clearOffloadVisuals();

      // Elevated midpoint curve
      const midX = (startPos.x + endPos.x) / 2;
      const midY = Math.max(startPos.y, endPos.y) + 14.0;
      const midZ = (startPos.z + endPos.z) / 2;
      const midPos = new THREE.Vector3(midX, midY, midZ);

      offloadCurve = new THREE.QuadraticBezierCurve3(startPos, midPos, endPos);

      const colorHex = isSuccess ? 0x10b981 : 0xef4444;
      const tubeGeo = new THREE.TubeGeometry(offloadCurve, 32, 0.45, 8, false);
      const tubeMat = new THREE.MeshBasicMaterial({{
        color: colorHex,
        transparent: true,
        opacity: 0.85
      }});
      offloadTubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
      scene.add(offloadTubeMesh);

      packetMat.color.setHex(colorHex);
      packetPulses.forEach(p => p.visible = true);
    }}

    // -------------------------------------------------------------
    // Initialize Simulation Entities from Step 0
    // -------------------------------------------------------------
    const step0 = simTrace.steps[0];
    step0.vehicles.forEach(v => {{
      const mesh = build3DVehicle(v.vid);
      vehicleMeshes.set(v.vid, mesh);
    }});

    step0.edges.forEach(e => {{
      const mesh = build3DRsuTower(e.eid, e);
      rsuMeshes.set(e.eid, mesh);
    }});

    // -------------------------------------------------------------
    // Step State Synchronization
    // -------------------------------------------------------------
    function updateStepVisuals(stepIdx) {{
      if (stepIdx < 0 || stepIdx >= totalSteps) return;
      currentStepIdx = stepIdx;
      const snap = simTrace.steps[stepIdx];

      // Update Top HUD
      document.getElementById('hud-step').innerText = `${{stepIdx + 1}} / ${{totalSteps}}`;
      document.getElementById('hud-task').innerText = `#${{snap.task_id}}`;
      document.getElementById('hud-req').innerText = `V${{snap.requester_vid}}`;
      document.getElementById('hud-dest').innerText = snap.dest_name;

      const pill = document.getElementById('hud-outcome');
      if (snap.success) {{
        pill.className = 'status-pill pill-success';
        pill.innerText = `SUCCESS (${{snap.total_latency_s.toFixed(2)}}s)`;
      }} else if (snap.deadline_missed) {{
        pill.className = 'status-pill pill-danger';
        pill.innerText = `DEADLINE MISS (${{snap.total_latency_s.toFixed(2)}}s)`;
      }} else {{
        pill.className = 'status-pill pill-warning';
        pill.innerText = `FAILED (${{snap.total_latency_s.toFixed(2)}}s)`;
      }}

      // Update Slider & Label
      document.getElementById('timeline-slider').value = stepIdx;
      document.getElementById('scrub-step-txt').innerText = `Step ${{stepIdx + 1}} / ${{totalSteps}}`;

      // Update Inspector Card
      document.getElementById('insp-header').innerText = `Step ${{stepIdx + 1}} Telemetry`;
      document.getElementById('insp-req').innerText = `Vehicle ${{snap.requester_vid}}`;
      document.getElementById('insp-dest').innerText = snap.dest_name;
      document.getElementById('insp-size').innerText = `${{snap.data_kb.toFixed(1)}} KB`;
      document.getElementById('insp-cycles').innerText = `${{snap.cycles_m.toFixed(1)}} Mcyc`;
      document.getElementById('insp-tx').innerText = `${{snap.tx_latency_s.toFixed(3)}} s`;
      document.getElementById('insp-exe').innerText = `${{snap.exe_latency_s.toFixed(3)}} s`;
      document.getElementById('insp-tot').innerText = `${{snap.total_latency_s.toFixed(3)}} s`;
      document.getElementById('insp-dl').innerText = `${{snap.deadline_s.toFixed(2)}} s`;
      document.getElementById('insp-energy').innerText = `${{snap.energy_j.toFixed(2)}} J`;
      document.getElementById('insp-trust').innerText = snap.dest_trust.toFixed(3);

      // Update Vehicle Positions & Styles
      snap.vehicles.forEach(vData => {{
        const vMesh = vehicleMeshes.get(vData.vid);
        if (!vMesh) return;

        const wx = toWorldX(vData.x_m);
        const wz = laneZ[vData.lane];
        vMesh.position.set(wx, 0, wz);

        const isReq = (vData.vid === snap.requester_vid);
        const isPeerDest = (snap.dest_kind === 'vehicle' && snap.dest_id === vData.vid);

        // Body color
        if (isReq) {{
          vMesh.userData.bodyMat.color.setHex(0xf59e0b); // Gold
          vMesh.userData.haloMat.opacity = 0.85;
        }} else if (isPeerDest) {{
          vMesh.userData.bodyMat.color.setHex(0x10b981); // Emerald target
          vMesh.userData.haloMat.opacity = 0.0;
        }} else {{
          vMesh.userData.bodyMat.color.setHex(0x2563eb); // Blue standard
          vMesh.userData.haloMat.opacity = 0.0;
        }}

        // Local execution aura
        if (isReq && snap.dest_kind === 'local') {{
          vMesh.userData.auraMat.opacity = 0.85;
        }} else {{
          vMesh.userData.auraMat.opacity = 0.0;
        }}
      }});

      // Update Offload Beams
      clearOffloadVisuals();
      const reqMesh = vehicleMeshes.get(snap.requester_vid);

      if (snap.dest_kind === 'edge' && reqMesh) {{
        const rsuMesh = rsuMeshes.get(snap.dest_id);
        if (rsuMesh) {{
          const startPt = new THREE.Vector3(reqMesh.position.x, 2.4, reqMesh.position.z);
          const endPt = new THREE.Vector3(rsuMesh.userData.worldX, rsuMesh.userData.antennaY, rsuMesh.userData.worldZ);
          createOffloadBeam(startPt, endPt, snap.success);
        }}
      }} else if (snap.dest_kind === 'vehicle' && reqMesh) {{
        const targetMesh = vehicleMeshes.get(snap.dest_id);
        if (targetMesh) {{
          const startPt = new THREE.Vector3(reqMesh.position.x, 2.0, reqMesh.position.z);
          const endPt = new THREE.Vector3(targetMesh.position.x, 2.0, targetMesh.position.z);
          createOffloadBeam(startPt, endPt, snap.success);
        }}
      }}
    }}

    // -------------------------------------------------------------
    // Camera Presets & Navigation
    // -------------------------------------------------------------
    function resetCamera() {{
      camera.position.set(0, 140, 220);
      controls.target.set(0, 6, 0);
      controls.update();
    }}

    function setCameraPreset(mode) {{
      const snap = simTrace.steps[currentStepIdx];
      const reqMesh = vehicleMeshes.get(snap.requester_vid);
      const rx = reqMesh ? reqMesh.position.x : 0;
      const rz = reqMesh ? reqMesh.position.z : 0;

      if (mode === 'iso') {{
        camera.position.set(rx - 70, 90, rz + 110);
        controls.target.set(rx, 4, rz);
      }} else if (mode === 'chase') {{
        camera.position.set(rx - 38, 12, rz);
        controls.target.set(rx + 40, 2, rz);
      }} else if (mode === 'side') {{
        camera.position.set(rx, 18, rz + 85);
        controls.target.set(rx, 6, rz);
      }} else if (mode === 'top') {{
        camera.position.set(rx, 260, 0);
        controls.target.set(rx, 0, 0);
      }}
      controls.update();
    }}

    // Initial camera position
    resetCamera();

    // -------------------------------------------------------------
    // Playback Controls
    // -------------------------------------------------------------
    function stepFirst() {{ updateStepVisuals(0); }}
    function stepLast() {{ updateStepVisuals(totalSteps - 1); }}
    function stepPrev() {{ updateStepVisuals(Math.max(0, currentStepIdx - 1)); }}
    function stepNext() {{ updateStepVisuals(Math.min(totalSteps - 1, currentStepIdx + 1)); }}
    function onScrub(val) {{ updateStepVisuals(parseInt(val, 10)); }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      const btn = document.getElementById('btn-play');
      if (isPlaying) {{
        btn.innerText = '⏸️';
        btn.title = 'Pause';
        playTimer = setInterval(() => {{
          if (currentStepIdx < totalSteps - 1) {{
            stepNext();
          }} else {{
            togglePlay();
          }}
        }}, playIntervalMs);
      }} else {{
        btn.innerText = '▶️';
        btn.title = 'Play';
        if (playTimer) clearInterval(playTimer);
      }}
    }}

    function onSpeedChange(val) {{
      playIntervalMs = parseInt(val, 10);
      if (isPlaying) {{
        clearInterval(playTimer);
        playTimer = setInterval(() => {{
          if (currentStepIdx < totalSteps - 1) stepNext();
          else togglePlay();
        }}, playIntervalMs);
      }}
    }}

    // -------------------------------------------------------------
    // Raycasting for Click-Selection
    // -------------------------------------------------------------
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    window.addEventListener('click', (event) => {{
      // Ignore clicks on HUD controls
      if (event.target.closest('.hud-top') || event.target.closest('.hud-bottom') || event.target.closest('.inspector-panel')) {{
        return;
      }}
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(scene.children, true);

      for (let hit of intersects) {{
        let obj = hit.object;
        while (obj && !obj.userData.type && obj.parent) {{
          obj = obj.parent;
        }}
        if (obj && obj.userData.type === 'vehicle') {{
          const vid = obj.userData.vid;
          const snap = simTrace.steps[currentStepIdx];
          const vData = snap.vehicles.find(v => v.vid === vid);
          if (vData) {{
            document.getElementById('insp-header').innerText = `Vehicle ${{vid}} Details`;
            document.getElementById('insp-req').innerText = `V${{vid}} (Lane ${{vData.lane}})`;
            document.getElementById('insp-dest').innerText = `Speed: ${{vData.speed_mps.toFixed(1)}} m/s`;
            document.getElementById('insp-size').innerText = `CPU: ${{vData.cpu_ghz.toFixed(2)}} GHz`;
            document.getElementById('insp-cycles').innerText = `Queue: ${{vData.queue_mcyc.toFixed(1)}} Mcyc`;
            document.getElementById('insp-trust').innerText = `${{vData.trust.toFixed(3)}}`;
          }}
          break;
        }} else if (obj && obj.userData.type === 'rsu') {{
          const eid = obj.userData.eid;
          const snap = simTrace.steps[currentStepIdx];
          const eData = snap.edges.find(e => e.eid === eid);
          if (eData) {{
            document.getElementById('insp-header').innerText = `RSU ${{eid}} (Edge Unit)`;
            document.getElementById('insp-req').innerText = `Position: ${{eData.x_m.toFixed(0)}}m`;
            document.getElementById('insp-dest').innerText = `Range: ${{eData.comm_range_m}}m`;
            document.getElementById('insp-size').innerText = `CPU: ${{eData.cpu_ghz.toFixed(1)}} GHz`;
            document.getElementById('insp-cycles').innerText = `Queue: ${{eData.queue_mcyc.toFixed(1)}} Mcyc`;
            document.getElementById('insp-trust').innerText = `${{eData.trust.toFixed(3)}}`;
          }}
          break;
        }}
      }}
    }});

    // -------------------------------------------------------------
    // Animation Render Loop
    // -------------------------------------------------------------
    let clock = new THREE.Clock();

    function animate() {{
      requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Pulse beacon lights on RSUs
      rsuMeshes.forEach(r => {{
        if (r.userData.beacon) {{
          const s = 0.8 + 0.3 * Math.sin(elapsed * 4.0);
          r.userData.beacon.scale.set(s, s, s);
        }}
      }});

      // Animate packet pulses along offload beam
      if (offloadCurve && packetPulses[0].visible) {{
        packetPulses.forEach((p, idx) => {{
          const t = (elapsed * 0.9 + idx * (1.0 / numPackets)) % 1.0;
          const pt = offloadCurve.getPointAt(t);
          p.position.copy(pt);
        }});
      }}

      // Rotate local CPU energy aura
      vehicleMeshes.forEach(v => {{
        if (v.userData.aura && v.userData.auraMat.opacity > 0) {{
          v.userData.aura.rotation.z = elapsed * 3.0;
        }}
        if (v.userData.halo && v.userData.haloMat.opacity > 0) {{
          const hs = 1.0 + 0.12 * Math.sin(elapsed * 5.0);
          v.userData.halo.scale.set(hs, hs, hs);
        }}
      }});

      controls.update();
      renderer.render(scene, camera);
    }}

    // Window Resize Handling
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    // Initial trigger to render step 0
    updateStepVisuals(0);
    animate();
  </script>
</body>
</html>"""

    return html_template
