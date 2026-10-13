"""
Three.js 3D WebGL realistic daylight vehicular edge-network simulation renderer.
Generates an interactive, publication-grade realistic 3D daytime road scene featuring:
- Natural daylight illumination, sky dome, atmospheric fog, and soft PBR PCF shadows
- Realistic multi-lane highway (3 lanes at 3.6m width, road shoulders, W-beam steel guardrails, overhead gantry)
- Roadside environment: grass verges, distance marker posts, roadside vegetation
- Highly detailed 3D composite vehicle models (Executive Sedan, Premium SUV, Commercial Cargo Van)
  with realistic automotive paint, transparent windshields, alloy wheels with rotation physics,
  projector LED headlights, taillight lightbars, dynamic suspension and steering pitch/roll
- Authentic roadside communication infrastructure (RSU): concrete pad, NEMA equipment cabinet,
  galvanized steel mast (7.5m), directional C-V2X sector antennas, microwave dish, solar panel,
  and dynamic 3D coverage footprint
- Visually clear, staged task offloading animation (transmission vs execution phases),
  distinguishing V2I (infrastructure cyan), V2V (direct emerald), local CPU halo (purple),
  inter-RSU handovers (amber link), and outcome indicators
- OrbitControls with daylight glassmorphic HUD, camera presets (Highway 3/4, Chase Cam, Roadside, Top-Down),
  frame-rate-independent follow tracking, timeline scrubber, and raycaster entity inspector
- 100% offline local asset bundling with Three.js r128 and OrbitControls
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
    """Build standalone, self-contained Three.js realistic daylight 3D simulation HTML markup."""
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
            "handovers": getattr(s, "handovers", []),
        })

    road_len = getattr(cfg, "road_length_m", 1200.0)
    comm_range = getattr(cfg, "comm_range_m", 250.0)
    n_veh = getattr(cfg, "num_vehicles", len(snapshots[0].vehicles if snapshots else 6))
    n_edge = getattr(cfg, "num_edge_nodes", len(snapshots[0].edges if snapshots else 2))

    trace_json = json.dumps({
        "road_length_m": road_len,
        "comm_range_m": comm_range,
        "num_vehicles": n_veh,
        "num_edges": n_edge,
        "summary": summary,
        "steps": steps_data,
    })

    cdn_fallback_three = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>' if not three_js else f"<script>{three_js}</script>"
    cdn_fallback_orbit = '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>' if not orbit_js else f"<script>{orbit_js}</script>"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TAR-FRL-VEC Realistic Daylight 3D Vehicular Simulation</title>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }}
    body {{
      background-color: #dbeafe;
      color: #0f172a;
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

    /* Top HUD Header (Daylight Glassmorphism) */
    .hud-top {{
      position: absolute;
      top: 14px;
      left: 16px;
      right: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
      pointer-events: none;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .hud-badge {{
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 8px;
      padding: 8px 16px;
      font-size: 0.82rem;
      font-weight: 700;
      color: #0f172a;
      pointer-events: auto;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
      display: flex;
      gap: 14px;
      align-items: center;
    }}
    .hud-badge strong {{
      color: #0369a1;
    }}
    .hud-badge .status-pill {{
      padding: 3px 10px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .pill-success {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
    .pill-danger {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
    .pill-warning {{ background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }}
    .pill-stage {{ background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }}

    /* Camera View Presets & Toggles */
    .cam-presets {{
      display: flex;
      gap: 6px;
      pointer-events: auto;
      flex-wrap: wrap;
    }}
    .hud-btn {{
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(203, 213, 225, 0.9);
      color: #0f172a;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s ease;
      backdrop-filter: blur(6px);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }}
    .hud-btn:hover {{
      background: #0284c7;
      border-color: #0284c7;
      color: #ffffff;
    }}
    .hud-btn.active {{
      background: #0f172a;
      border-color: #0f172a;
      color: #ffffff;
    }}

    /* Bottom Playback & Scrubber Controls */
    .hud-bottom {{
      position: absolute;
      bottom: 14px;
      left: 16px;
      right: 16px;
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 10px;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      gap: 16px;
      z-index: 10;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10);
    }}
    .btn-ctrl {{
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      color: #0f172a;
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
      border-color: #0284c7;
      color: #ffffff;
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
      color: #000000;
      font-weight: 700;
    }}
    .slider {{
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: #e2e8f0;
      outline: none;
      cursor: pointer;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #0284c7;
      border: 2px solid #ffffff;
      cursor: pointer;
      box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }}
    .speed-select {{
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      color: #0f172a;
      padding: 5px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      outline: none;
      cursor: pointer;
    }}

    /* Floating Inspector Panel */
    .inspector-panel {{
      position: absolute;
      top: 66px;
      right: 16px;
      width: 295px;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 8px;
      padding: 14px;
      z-index: 10;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
      font-size: 0.8rem;
      color: #000000;
    }}
    .inspector-title {{
      font-size: 0.82rem;
      font-weight: 800;
      color: #000000;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 6px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }}
    .prop-row {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      color: #000000;
      font-weight: 600;
    }}
    .prop-row span {{
      color: #000000;
    }}
    .prop-val {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-weight: 700;
      color: #000000;
    }}

    /* Legend Box */
    .legend-box {{
      position: absolute;
      bottom: 74px;
      left: 16px;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 6px;
      padding: 6px 14px;
      display: flex;
      gap: 14px;
      font-size: 0.75rem;
      color: #000000;
      font-weight: 700;
      z-index: 10;
      backdrop-filter: blur(8px);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      flex-wrap: wrap;
    }}
    .legend-box * {{
      color: #000000;
    }}
    .leg-item {{ display: flex; align-items: center; gap: 5px; }}
    .leg-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

    #webgl-error {{
      display: none;
      position: absolute;
      top: 40%;
      left: 20%;
      right: 20%;
      background: #fef2f2;
      color: #991b1b;
      padding: 20px;
      border-radius: 8px;
      text-align: center;
      z-index: 99;
      border: 1px solid #ef4444;
      box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }}
  </style>
  {cdn_fallback_three}
  {cdn_fallback_orbit}
</head>
<body>
  <div id="canvas-container"></div>
  <div id="webgl-error">
    <h3>WebGL Acceleration Unavailable</h3>
    <p>Hardware WebGL support is required to render the 3D vehicular simulation.</p>
  </div>

  <!-- Top Header HUD -->
  <div class="hud-top">
    <div class="hud-badge">
      <span>Step: <strong id="hud-step">1 / 30</strong></span>
      <span>Task: <strong id="hud-task">#0</strong></span>
      <span>Req: <strong id="hud-req">V0</strong> ➔ <strong id="hud-dest">RSU 0</strong></span>
      <span id="hud-stage" class="status-pill pill-stage">TX STAGE</span>
      <span id="hud-outcome" class="status-pill pill-success">SUCCESS</span>
    </div>
    <div class="cam-presets">
      <button class="hud-btn active" id="btn-cov" onclick="toggleCoverage()">📡 Coverage Footprint</button>
      <button class="hud-btn active" id="btn-net" onclick="toggleNetworkOverlay()">📶 Network Overlay</button>
      <button class="hud-btn active" id="btn-follow" onclick="toggleCameraFollow()">🎯 Follow Vehicle</button>
      <button class="hud-btn active" id="btn-cam-iso" onclick="setCameraPreset('iso')">🎥 Highway 3/4</button>
      <button class="hud-btn" id="btn-cam-chase" onclick="setCameraPreset('chase')">🚗 Chase Cam</button>
      <button class="hud-btn" id="btn-cam-side" onclick="setCameraPreset('side')">🔭 Roadside</button>
      <button class="hud-btn" id="btn-cam-top" onclick="setCameraPreset('top')">🗺️ Top-Down</button>
      <button class="hud-btn" onclick="resetCamera()">🔄 Reset Cam</button>
    </div>
  </div>

  <!-- Entity Inspector Panel -->
  <div class="inspector-panel" id="inspector-panel">
    <div class="inspector-title">
      <span id="insp-header">Step Telemetry</span>
      <span style="font-size:0.7rem; color:#64748b;">(Click Entity)</span>
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
    <div class="prop-row"><span>Dest Trust:</span><span class="prop-val" id="insp-trust">0.850</span></div>
    <div class="prop-row"><span>Network Event:</span><span class="prop-val" id="insp-event">Stable Link</span></div>
  </div>

  <!-- Dynamic Legend -->
  <div class="legend-box">
    <div class="leg-item"><div class="leg-dot" style="background:#2563eb;"></div> Vehicle</div>
    <div class="leg-item"><div class="leg-dot" style="background:#f59e0b;"></div> Requester</div>
    <div class="leg-item"><div class="leg-dot" style="background:#475569;"></div> Roadside RSU</div>
    <div class="leg-item"><div class="leg-dot" style="background:#0284c7;"></div> V2I Channel ({comm_range:.0f}m)</div>
    <div class="leg-item"><div class="leg-dot" style="background:#059669;"></div> V2V Channel</div>
    <div class="leg-item"><div class="leg-dot" style="background:#8b5cf6;"></div> Local Execution</div>
    <div class="leg-item"><div class="leg-dot" style="background:#10b981;"></div> Success</div>
    <div class="leg-item"><div class="leg-dot" style="background:#ef4444;"></div> Deadline Miss / Drop</div>
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
        <span>Continuous 60 FPS Deterministic Timeline</span>
        <span id="scrub-step-txt">Step 1.0 / 30</span>
      </div>
      <input type="range" class="slider" id="timeline-slider" min="0" max="29" step="0.01" value="0" oninput="onScrubInput(this.value)" onmousedown="onScrubStart()" onmouseup="onScrubEnd()" ontouchstart="onScrubStart()" ontouchend="onScrubEnd()">
    </div>

    <select class="speed-select" id="speed-select" onchange="onSpeedChange(this.value)">
      <option value="0.5">0.5x Speed (Slow)</option>
      <option value="1.0" selected>1.0x Speed (Normal)</option>
      <option value="2.0">2.0x Speed (Fast)</option>
      <option value="3.5">3.5x Speed (Ultra)</option>
    </select>
  </div>

  <script>
    if (!window.WebGLRenderingContext) {{
      document.getElementById('webgl-error').style.display = 'block';
    }}

    const simTrace = {trace_json};
    const totalSteps = simTrace.steps.length;
    let currentStepIdx = 0;
    let currentProgress = 0.0;
    let targetProgress = 0.0;
    let isPlaying = false;
    let isTweening = false;
    let tweenStartTime = 0;
    let tweenStartProgress = 0;
    const tweenDuration = 0.35;
    let playbackSpeed = 1.0;
    let lastDiscreteStep = -1;
    let isScrubbing = false;
    let cameraFollow = true;
    let activeCameraMode = 'iso';

    let showCoverageFootprint = true;
    let showNetworkOverlay = true;

    document.getElementById('timeline-slider').max = Math.max(0, totalSteps - 1);

    // -------------------------------------------------------------
    // Three.js Scene Setup with Photorealistic Daylight & Color Management
    // -------------------------------------------------------------
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();

    // Natural daytime sky dome and atmospheric depth fog
    scene.background = new THREE.Color(0xa7cbf0);
    scene.fog = new THREE.FogExp2(0xa7cbf0, 0.00075);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 4000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: 'high-performance' }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    controls.minDistance = 12;
    controls.maxDistance = 1400;

    // -------------------------------------------------------------
    // Sunlight & Hemispheric Sky Illumination
    // -------------------------------------------------------------
    // Sky blue ambient + warm grass ground bounce
    const hemiLight = new THREE.HemisphereLight(0xd0e8ff, 0x4a5d35, 0.70);
    scene.add(hemiLight);

    // Warm Sun Directional Light casting soft PCF shadows
    const sunLight = new THREE.DirectionalLight(0xfffaea, 1.30);
    sunLight.position.set(180, 260, 120);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 10;
    sunLight.shadow.camera.far = 1200;
    const sArea = 400;
    sunLight.shadow.camera.left = -sArea;
    sunLight.shadow.camera.right = sArea;
    sunLight.shadow.camera.top = sArea;
    sunLight.shadow.camera.bottom = -sArea;
    sunLight.shadow.bias = -0.0002;
    scene.add(sunLight);

    // Soft sky fill light
    const fillLight = new THREE.DirectionalLight(0xb6d7f9, 0.45);
    fillLight.position.set(-180, 90, -120);
    scene.add(fillLight);

    // -------------------------------------------------------------
    // Physical Road Geometry & Environment (1:1 Metric Scale)
    // -------------------------------------------------------------
    const roadLengthM = simTrace.road_length_m || 1200.0;
    const halfRoadM = roadLengthM / 2;

    function toWorldX(xMeters) {{
      const wrapped = (xMeters % roadLengthM + roadLengthM) % roadLengthM;
      return (wrapped - halfRoadM);
    }}

    // 1. Natural Grass Terrain
    const terrainGeo = new THREE.PlaneGeometry(roadLengthM + 500, 280, 32, 32);
    terrainGeo.rotateX(-Math.PI / 2);
    const terrainMat = new THREE.MeshStandardMaterial({{
      color: 0x415b32,
      roughness: 0.95,
      metalness: 0.05
    }});
    const terrain = new THREE.Mesh(terrainGeo, terrainMat);
    terrain.position.y = -0.05;
    terrain.receiveShadow = true;
    scene.add(terrain);

    // 2. Asphalt Highway Road Bed
    // 3 Lanes (each 3.6m wide) = 10.8m driving corridor
    // Total paved asphalt width with shoulders = 16.0m (Z in [-8.0, +8.0])
    const roadAsphaltGeo = new THREE.PlaneGeometry(roadLengthM, 16.0);
    roadAsphaltGeo.rotateX(-Math.PI / 2);
    const roadAsphaltMat = new THREE.MeshStandardMaterial({{
      color: 0x24282f,
      roughness: 0.85,
      metalness: 0.06
    }});
    const roadBed = new THREE.Mesh(roadAsphaltGeo, roadAsphaltMat);
    roadBed.position.y = 0.05;
    roadBed.receiveShadow = true;
    scene.add(roadBed);

    // 3. Pavement Markings: Solid Edge Lines and Dashed Lane Dividers
    const lineWhiteMat = new THREE.MeshBasicMaterial({{ color: 0xf8fafc }});
    const edgeLineGeo = new THREE.PlaneGeometry(roadLengthM, 0.22);
    edgeLineGeo.rotateX(-Math.PI / 2);

    const edgeNorth = new THREE.Mesh(edgeLineGeo, lineWhiteMat);
    edgeNorth.position.set(0, 0.065, -5.4);
    scene.add(edgeNorth);

    const edgeSouth = new THREE.Mesh(edgeLineGeo, lineWhiteMat);
    edgeSouth.position.set(0, 0.065, 5.4);
    scene.add(edgeSouth);

    // Yellow Boundary Shoulder Lines (at Z = -7.5m and Z = +7.5m)
    const lineYellowMat = new THREE.MeshBasicMaterial({{ color: 0xfacc15 }});
    const yellowLineGeo = new THREE.PlaneGeometry(roadLengthM, 0.24);
    yellowLineGeo.rotateX(-Math.PI / 2);

    const yellowNorth = new THREE.Mesh(yellowLineGeo, lineYellowMat);
    yellowNorth.position.set(0, 0.065, -7.5);
    scene.add(yellowNorth);

    const yellowSouth = new THREE.Mesh(yellowLineGeo, lineYellowMat);
    yellowSouth.position.set(0, 0.065, 7.5);
    scene.add(yellowSouth);

    // Dashed Lane Dividers (3.0m stripe, 6.0m gap at Z = -1.8m and Z = +1.8m)
    const dashStripeGeo = new THREE.PlaneGeometry(3.0, 0.16);
    dashStripeGeo.rotateX(-Math.PI / 2);

    for (let x = -halfRoadM + 4; x < halfRoadM - 4; x += 9.0) {{
      const d1 = new THREE.Mesh(dashStripeGeo, lineWhiteMat);
      d1.position.set(x, 0.068, -1.8);
      scene.add(d1);

      const d2 = new THREE.Mesh(dashStripeGeo, lineWhiteMat);
      d2.position.set(x, 0.068, 1.8);
      scene.add(d2);
    }}

    // 4. Steel W-Beam Guardrails (along both sides at Z = -8.1m and Z = +8.1m)
    const railMat = new THREE.MeshStandardMaterial({{
      color: 0x94a3b8,
      metalness: 0.85,
      roughness: 0.35
    }});
    const railBeamGeo = new THREE.BoxGeometry(roadLengthM, 0.35, 0.08);

    const railN = new THREE.Mesh(railBeamGeo, railMat);
    railN.position.set(0, 0.75, -8.1);
    railN.castShadow = true;
    scene.add(railN);

    const railS = new THREE.Mesh(railBeamGeo, railMat);
    railS.position.set(0, 0.75, 8.1);
    railS.castShadow = true;
    scene.add(railS);

    // Guardrail Support Posts (every 4.0m)
    const postGeo = new THREE.BoxGeometry(0.12, 0.85, 0.12);
    for (let x = -halfRoadM + 2; x < halfRoadM; x += 4.0) {{
      const pN = new THREE.Mesh(postGeo, railMat);
      pN.position.set(x, 0.42, -8.1);
      pN.castShadow = true;
      scene.add(pN);

      const pS = new THREE.Mesh(postGeo, railMat);
      pS.position.set(x, 0.42, 8.1);
      pS.castShadow = true;
      scene.add(pS);
    }}

    // 5. Overhead Highway Signage Gantry (at X = -150m)
    function buildHighwayGantry(wx) {{
      const gantryGroup = new THREE.Group();
      const trussMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, metalness: 0.7, roughness: 0.4 }});
      
      // Vertical Uprights
      const poleGeo = new THREE.CylinderGeometry(0.25, 0.25, 7.2, 12);
      const poleL = new THREE.Mesh(poleGeo, trussMat);
      poleL.position.set(wx, 3.6, -9.0);
      poleL.castShadow = true;
      gantryGroup.add(poleL);

      const poleR = new THREE.Mesh(poleGeo, trussMat);
      poleR.position.set(wx, 3.6, 9.0);
      poleR.castShadow = true;
      gantryGroup.add(poleR);

      // Overhead Crossbar Truss
      const beamGeo = new THREE.BoxGeometry(0.6, 0.8, 18.4);
      const beam = new THREE.Mesh(beamGeo, trussMat);
      beam.position.set(wx, 7.0, 0);
      beam.castShadow = true;
      gantryGroup.add(beam);

      // Green Overhead Highway Sign Board
      const signGeo = new THREE.BoxGeometry(0.12, 1.8, 8.5);
      const signMat = new THREE.MeshStandardMaterial({{ color: 0x15803d, roughness: 0.3 }});
      const signBoard = new THREE.Mesh(signGeo, signMat);
      signBoard.position.set(wx - 0.35, 7.0, -1.0);
      signBoard.castShadow = true;
      gantryGroup.add(signBoard);

      // Sign border
      const borderGeo = new THREE.BoxGeometry(0.14, 1.9, 8.6);
      const borderMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
      const signBorder = new THREE.Mesh(borderGeo, borderMat);
      signBorder.position.set(wx - 0.33, 7.0, -1.0);
      gantryGroup.add(signBorder);

      scene.add(gantryGroup);
    }}
    buildHighwayGantry(-150.0);

    // 6. Roadside Distance Marker Posts (every 100m)
    for (let m = 0; m <= roadLengthM; m += 100) {{
      const wx = toWorldX(m);
      const postGroup = new THREE.Group();
      const postMat = new THREE.MeshStandardMaterial({{ color: 0xffffff }});
      const mPostGeo = new THREE.BoxGeometry(0.18, 1.1, 0.18);
      const mPost = new THREE.Mesh(mPostGeo, postMat);
      mPost.position.y = 0.55;
      mPost.castShadow = true;
      postGroup.add(mPost);

      // Blue reflector cap
      const capGeo = new THREE.BoxGeometry(0.20, 0.22, 0.20);
      const capMat = new THREE.MeshStandardMaterial({{ color: 0x0284c7 }});
      const cap = new THREE.Mesh(capGeo, capMat);
      cap.position.y = 0.95;
      postGroup.add(cap);

      postGroup.position.set(wx, 0, -8.8);
      scene.add(postGroup);
    }}

    // 7. Roadside Trees and Shrubs (planted outside road clearance zone |Z| > 12m)
    function createTree(x, z, scaleFactor) {{
      const treeGroup = new THREE.Group();
      const trunkGeo = new THREE.CylinderGeometry(0.35 * scaleFactor, 0.5 * scaleFactor, 3.8 * scaleFactor, 8);
      const trunkMat = new THREE.MeshStandardMaterial({{ color: 0x4a3728, roughness: 0.9 }});
      const trunk = new THREE.Mesh(trunkGeo, trunkMat);
      trunk.position.y = 1.9 * scaleFactor;
      trunk.castShadow = true;
      treeGroup.add(trunk);

      const leavesMat = new THREE.MeshStandardMaterial({{ color: 0x2e5a27, roughness: 0.85 }});
      const c1Geo = new THREE.ConeGeometry(2.4 * scaleFactor, 3.8 * scaleFactor, 7);
      const c1 = new THREE.Mesh(c1Geo, leavesMat);
      c1.position.y = 4.2 * scaleFactor;
      c1.castShadow = true;
      treeGroup.add(c1);

      const c2Geo = new THREE.ConeGeometry(1.8 * scaleFactor, 3.2 * scaleFactor, 7);
      const c2 = new THREE.Mesh(c2Geo, leavesMat);
      c2.position.y = 6.0 * scaleFactor;
      c2.castShadow = true;
      treeGroup.add(c2);

      treeGroup.position.set(x, 0, z);
      scene.add(treeGroup);
    }}

    for (let x = -halfRoadM + 15; x < halfRoadM - 15; x += 22.0) {{
      const offsetZ1 = -13.0 - ((Math.abs(x * 13) % 17));
      createTree(x, offsetZ1, 0.85 + ((Math.abs(x * 7) % 5) * 0.08));

      const offsetZ2 = 13.0 + ((Math.abs(x * 19) % 17));
      createTree(x + 10, offsetZ2, 0.90 + ((Math.abs(x * 11) % 5) * 0.08));
    }}

    // Driving Lane Centers (3 Lanes: 3.6m width each)
    const laneZ = [-3.6, 0.0, 3.6];

    // -------------------------------------------------------------
    // Reusable Materials & Color Management for 3D Vehicles
    // -------------------------------------------------------------
    const vehicleMeshes = new Map();

    const paintPalette = [
      0x1c2d44, // Deep Sapphire Navy
      0xc0c6cb, // Silver Metallic
      0xeef2f5, // Glacier White
      0x373e48, // Graphite Metallic
      0x7a1820, // Crimson Red
      0x14171a, // Obsidian Black
    ];

    const glassMat = new THREE.MeshStandardMaterial({{
      color: 0x0f172a,
      roughness: 0.05,
      metalness: 0.95,
      transparent: true,
      opacity: 0.82
    }});
    const trimMat = new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.88, metalness: 0.12 }});
    const chromeMat = new THREE.MeshStandardMaterial({{ color: 0xe2e8f0, metalness: 0.92, roughness: 0.15 }});
    const rubberMat = new THREE.MeshStandardMaterial({{ color: 0x171717, roughness: 0.94, metalness: 0.06 }});
    const rimMat = new THREE.MeshStandardMaterial({{ color: 0xcfd4dc, metalness: 0.88, roughness: 0.20 }});
    const contactShadowMat = new THREE.MeshBasicMaterial({{
      color: 0x0a0f1d,
      transparent: true,
      opacity: 0.35
    }});

    // Function to build high-detail 5-spoke alloy wheels
    function buildAlloyWheel(radius, width) {{
      const wheelGroup = new THREE.Group();
      
      // Tire outer tread
      const tireGeo = new THREE.CylinderGeometry(radius, radius, width, 20);
      tireGeo.rotateX(Math.PI / 2);
      const tireMesh = new THREE.Mesh(tireGeo, rubberMat);
      tireMesh.castShadow = true;
      wheelGroup.add(tireMesh);

      // Center Hub
      const hubGeo = new THREE.CylinderGeometry(radius * 0.35, radius * 0.35, width + 0.02, 14);
      hubGeo.rotateX(Math.PI / 2);
      const hubMesh = new THREE.Mesh(hubGeo, rimMat);
      wheelGroup.add(hubMesh);

      // 5-Spoke Alloy Star Pattern
      const spokeGeo = new THREE.BoxGeometry(radius * 0.55, 0.06, width + 0.015);
      for (let s = 0; s < 5; s++) {{
        const spoke = new THREE.Mesh(spokeGeo, rimMat);
        const angle = (s * Math.PI * 2) / 5;
        spoke.position.set(Math.cos(angle) * radius * 0.38, Math.sin(angle) * radius * 0.38, 0);
        spoke.rotation.z = angle;
        wheelGroup.add(spoke);
      }}

      return wheelGroup;
    }}

    function buildRealistic3DVehicle(vid) {{
      const group = new THREE.Group();
      const vehType = vid % 3; // 0: Sedan, 1: SUV, 2: Commercial Van
      const bodyColor = paintPalette[vid % paintPalette.length];

      const paintMat = new THREE.MeshStandardMaterial({{
        color: bodyColor,
        roughness: 0.24,
        metalness: 0.82
      }});

      let wheelRadius = 0.34;
      let bodyLength = 4.65;
      let bodyWidth = 1.86;

      if (vehType === 0) {{
        // ------------------ 1. EXECUTIVE SEDAN (4.65m L, 1.86m W, 1.44m H) ------------------
        wheelRadius = 0.34;
        bodyLength = 4.65;
        bodyWidth = 1.86;

        // Lower Chassis Body
        const lowerBodyGeo = new THREE.BoxGeometry(4.65, 0.62, 1.86);
        const lowerBody = new THREE.Mesh(lowerBodyGeo, paintMat);
        lowerBody.position.y = 0.60;
        lowerBody.castShadow = true;
        lowerBody.receiveShadow = true;
        group.add(lowerBody);

        // Aerodynamic Hood Bonnet
        const hoodGeo = new THREE.BoxGeometry(1.45, 0.24, 1.76);
        const hood = new THREE.Mesh(hoodGeo, paintMat);
        hood.position.set(1.42, 0.93, 0);
        hood.castShadow = true;
        group.add(hood);

        // Front Grille
        const grilleGeo = new THREE.BoxGeometry(0.12, 0.32, 1.25);
        const grille = new THREE.Mesh(grilleGeo, trimMat);
        grille.position.set(2.33, 0.62, 0);
        group.add(grille);

        // Rear Trunk Deck
        const trunkGeo = new THREE.BoxGeometry(1.05, 0.26, 1.76);
        const trunk = new THREE.Mesh(trunkGeo, paintMat);
        trunk.position.set(-1.62, 0.93, 0);
        trunk.castShadow = true;
        group.add(trunk);

        // Cabin Greenhouse (Windshield, Side Windows, Rear Glass)
        const cabinGeo = new THREE.BoxGeometry(2.35, 0.64, 1.62);
        const cabin = new THREE.Mesh(cabinGeo, glassMat);
        cabin.position.set(-0.15, 1.25, 0);
        cabin.castShadow = true;
        group.add(cabin);

        // Roof Panel
        const roofGeo = new THREE.BoxGeometry(1.85, 0.08, 1.56);
        const roof = new THREE.Mesh(roofGeo, paintMat);
        roof.position.set(-0.15, 1.58, 0);
        roof.castShadow = true;
        group.add(roof);

        // Aerodynamic Side Mirrors
        const mirrorGeo = new THREE.BoxGeometry(0.14, 0.12, 0.24);
        const mL = new THREE.Mesh(mirrorGeo, paintMat);
        mL.position.set(0.72, 1.05, 0.98);
        const mR = new THREE.Mesh(mirrorGeo, paintMat);
        mR.position.set(0.72, 1.05, -0.98);
        group.add(mL);
        group.add(mR);

        // Dual Chrome Exhaust Tips
        const exhGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.22, 8);
        exhGeo.rotateZ(Math.PI / 2);
        const exh1 = new THREE.Mesh(exhGeo, chromeMat);
        exh1.position.set(-2.36, 0.35, 0.55);
        const exh2 = new THREE.Mesh(exhGeo, chromeMat);
        exh2.position.set(-2.36, 0.35, -0.55);
        group.add(exh1);
        group.add(exh2);

      }} else if (vehType === 1) {{
        // ------------------ 2. PREMIUM SUV / CROSSOVER (4.88m L, 1.96m W, 1.74m H) ------------------
        wheelRadius = 0.38;
        bodyLength = 4.88;
        bodyWidth = 1.96;

        // Lower Chassis
        const lowerBodyGeo = new THREE.BoxGeometry(4.88, 0.82, 1.96);
        const lowerBody = new THREE.Mesh(lowerBodyGeo, paintMat);
        lowerBody.position.y = 0.78;
        lowerBody.castShadow = true;
        group.add(lowerBody);

        // Protective Cladding Underbody Sills
        const trimGeo = new THREE.BoxGeometry(4.90, 0.24, 1.99);
        const trimMesh = new THREE.Mesh(trimGeo, trimMat);
        trimMesh.position.y = 0.44;
        group.add(trimMesh);

        // Front Silver Skid Plate
        const skidGeo = new THREE.BoxGeometry(0.35, 0.22, 1.2);
        const skid = new THREE.Mesh(skidGeo, chromeMat);
        skid.position.set(2.35, 0.42, 0);
        group.add(skid);

        // Taller Cabin Glass
        const cabinGeo = new THREE.BoxGeometry(2.95, 0.82, 1.76);
        const cabin = new THREE.Mesh(cabinGeo, glassMat);
        cabin.position.set(-0.35, 1.45, 0);
        cabin.castShadow = true;
        group.add(cabin);

        // Roof Panel
        const roofGeo = new THREE.BoxGeometry(2.65, 0.10, 1.72);
        const roof = new THREE.Mesh(roofGeo, paintMat);
        roof.position.set(-0.35, 1.88, 0);
        group.add(roof);

        // Chrome Roof Rails
        const railGeo = new THREE.BoxGeometry(2.35, 0.06, 0.06);
        const rL = new THREE.Mesh(railGeo, chromeMat);
        rL.position.set(-0.35, 1.95, 0.76);
        const rR = new THREE.Mesh(railGeo, chromeMat);
        rR.position.set(-0.35, 1.95, -0.76);
        group.add(rL);
        group.add(rR);

      }} else {{
        // ------------------ 3. COMMERCIAL CARGO VAN (5.30m L, 2.05m W, 2.25m H) ------------------
        wheelRadius = 0.35;
        bodyLength = 5.30;
        bodyWidth = 2.05;

        // Main Cargo Enclosure
        const vanBodyGeo = new THREE.BoxGeometry(5.30, 1.55, 2.05);
        const vanBody = new THREE.Mesh(vanBodyGeo, paintMat);
        vanBody.position.set(-0.20, 1.20, 0);
        vanBody.castShadow = true;
        group.add(vanBody);

        // High Cab Windshield
        const cabGeo = new THREE.BoxGeometry(1.35, 0.78, 1.90);
        const cab = new THREE.Mesh(cabGeo, glassMat);
        cab.position.set(1.45, 1.40, 0);
        group.add(cab);

        // Commercial Heavy-Duty Bumpers
        const bumperGeo = new THREE.BoxGeometry(0.42, 0.52, 2.02);
        const bumper = new THREE.Mesh(bumperGeo, trimMat);
        bumper.position.set(2.50, 0.58, 0);
        group.add(bumper);

        // Rear Split Cargo Doors seam
        const seamGeo = new THREE.BoxGeometry(0.04, 1.45, 0.05);
        const seam = new THREE.Mesh(seamGeo, trimMat);
        seam.position.set(-2.86, 1.20, 0);
        group.add(seam);
      }}

      // Contact Ground Shadow Plane
      const shadowGeo = new THREE.PlaneGeometry(bodyLength + 0.3, bodyWidth + 0.3);
      shadowGeo.rotateX(-Math.PI / 2);
      const contactShadow = new THREE.Mesh(shadowGeo, contactShadowMat);
      contactShadow.position.y = 0.02;
      group.add(contactShadow);

      // 4 Realistic 5-Spoke Alloy Wheels
      const wheelWidth = 0.24;
      const wheelZ = (vehType === 1) ? 1.00 : 0.94;
      const wheelFrontX = (vehType === 2) ? 1.65 : 1.48;
      const wheelRearX = (vehType === 2) ? -1.75 : -1.48;

      const wheelCoords = [
        [wheelFrontX, wheelRadius, wheelZ],
        [wheelFrontX, wheelRadius, -wheelZ],
        [wheelRearX, wheelRadius, wheelZ],
        [wheelRearX, wheelRadius, -wheelZ]
      ];

      const wheelMeshes = [];
      wheelCoords.forEach(pos => {{
        const wMesh = buildAlloyWheel(wheelRadius, wheelWidth);
        wMesh.position.set(...pos);
        group.add(wMesh);
        wheelMeshes.push(wMesh);
      }});

      // Projector LED Headlights (Cool White)
      const hlMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
      const hlGeo = new THREE.BoxGeometry(0.12, 0.18, 0.38);
      const hlL = new THREE.Mesh(hlGeo, hlMat);
      hlL.position.set(bodyLength / 2 - 0.05, 0.76, 0.72);
      const hlR = new THREE.Mesh(hlGeo, hlMat);
      hlR.position.set(bodyLength / 2 - 0.05, 0.76, -0.72);
      group.add(hlL);
      group.add(hlR);

      // Red LED Taillight Bars
      const tlMat = new THREE.MeshBasicMaterial({{ color: 0xdc2626 }});
      const tlGeo = new THREE.BoxGeometry(0.12, 0.18, 0.40);
      const tlL = new THREE.Mesh(tlGeo, tlMat);
      tlL.position.set(-bodyLength / 2 + 0.05, 0.76, 0.72);
      const tlR = new THREE.Mesh(tlGeo, tlMat);
      tlR.position.set(-bodyLength / 2 + 0.05, 0.76, -0.72);
      group.add(tlL);
      group.add(tlR);

      // Amber Turn Indicators
      const indMat = new THREE.MeshBasicMaterial({{ color: 0xf59e0b }});
      const indGeo = new THREE.BoxGeometry(0.12, 0.10, 0.12);
      const indLeftGroup = new THREE.Group();
      const indRightGroup = new THREE.Group();

      const indFL = new THREE.Mesh(indGeo, indMat);
      indFL.position.set(bodyLength / 2 - 0.05, 0.74, 0.92);
      const indRL = new THREE.Mesh(indGeo, indMat);
      indRL.position.set(-bodyLength / 2 + 0.05, 0.74, 0.92);
      indLeftGroup.add(indFL);
      indLeftGroup.add(indRL);
      indLeftGroup.visible = false;
      group.add(indLeftGroup);

      const indFR = new THREE.Mesh(indGeo, indMat);
      indFR.position.set(bodyLength / 2 - 0.05, 0.74, -0.92);
      const indRR = new THREE.Mesh(indGeo, indMat);
      indRR.position.set(-bodyLength / 2 + 0.05, 0.74, -0.92);
      indRightGroup.add(indFR);
      indRightGroup.add(indRR);
      indRightGroup.visible = false;
      group.add(indRightGroup);

      // Rooftop C-V2X Transponder Antenna Beacon
      const beaconGeo = new THREE.CylinderGeometry(0.14, 0.18, 0.16, 12);
      const beaconMat = new THREE.MeshBasicMaterial({{ color: 0xf59e0b }});
      const beacon = new THREE.Mesh(beaconGeo, beaconMat);
      beacon.position.y = (vehType === 2) ? 2.32 : (vehType === 1 ? 2.02 : 1.70);
      beacon.visible = false;
      group.add(beacon);

      // Local CPU Computation Holographic Halo
      const localAuraGeo = new THREE.RingGeometry(1.6, 1.9, 32);
      localAuraGeo.rotateX(-Math.PI / 2);
      const localAuraMat = new THREE.MeshBasicMaterial({{
        color: 0x8b5cf6,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.0
      }});
      const localAura = new THREE.Mesh(localAuraGeo, localAuraMat);
      localAura.position.y = (vehType === 2) ? 2.45 : 1.95;
      group.add(localAura);

      // Destination Computing Pulse Halo
      const exeHaloGeo = new THREE.RingGeometry(1.4, 1.7, 32);
      exeHaloGeo.rotateX(-Math.PI / 2);
      const exeHaloMat = new THREE.MeshBasicMaterial({{
        color: 0x10b981,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.0
      }});
      const exeHalo = new THREE.Mesh(exeHaloGeo, exeHaloMat);
      exeHalo.position.y = (vehType === 2) ? 2.42 : 1.90;
      group.add(exeHalo);

      group.userData = {{
        type: 'vehicle',
        vid: vid,
        paintMat: paintMat,
        beacon: beacon,
        localAura: localAura,
        localAuraMat: localAuraMat,
        exeHalo: exeHalo,
        exeHaloMat: exeHaloMat,
        baseColor: bodyColor,
        wheels: wheelMeshes,
        wheelRadius: wheelRadius,
        indLeft: indLeftGroup,
        indRight: indRightGroup,
        bodyLength: bodyLength
      }};

      scene.add(group);
      return group;
    }}

    // -------------------------------------------------------------
    // Realistic Roadside Unit (RSU) Physical Infrastructure
    // -------------------------------------------------------------
    const rsuMeshes = new Map();
    const commRadiusM = simTrace.comm_range_m || {comm_range:.1f};

    function buildRealistic3DRsu(eid, edgeData) {{
      const group = new THREE.Group();
      const wx = toWorldX(edgeData.x_m);
      const rsuZ = -10.8; // Setback beyond guardrail into road verge

      // 1. Reinforced Concrete Foundation Pad
      const padGeo = new THREE.BoxGeometry(2.2, 0.28, 2.2);
      const padMat = new THREE.MeshStandardMaterial({{ color: 0x94a3b8, roughness: 0.95 }});
      const pad = new THREE.Mesh(padGeo, padMat);
      pad.position.y = 0.14;
      pad.receiveShadow = true;
      group.add(pad);

      // 2. Weatherproof NEMA 4X Equipment Cabinet
      const cabGeo = new THREE.BoxGeometry(0.90, 1.45, 0.70);
      const cabMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, roughness: 0.5, metalness: 0.35 }});
      const cabinet = new THREE.Mesh(cabGeo, cabMat);
      cabinet.position.set(0.45, 1.00, 0);
      cabinet.castShadow = true;
      group.add(cabinet);

      // 3. Galvanized Steel Tubular Mast (7.5m Height)
      const poleGeo = new THREE.CylinderGeometry(0.12, 0.16, 7.5, 16);
      const poleMat = new THREE.MeshStandardMaterial({{ color: 0x718096, metalness: 0.85, roughness: 0.3 }});
      const pole = new THREE.Mesh(poleGeo, poleMat);
      pole.position.set(-0.45, 3.85, 0);
      pole.castShadow = true;
      group.add(pole);

      // Climbing rungs on mast
      const rungGeo = new THREE.BoxGeometry(0.04, 0.04, 0.35);
      for (let ry = 1.5; ry < 6.5; ry += 0.6) {{
        const rung = new THREE.Mesh(rungGeo, poleMat);
        rung.position.set(-0.55, ry, 0);
        group.add(rung);
      }}

      // 4. Crossarm Mounting Bracket at 6.8m
      const armGeo = new THREE.BoxGeometry(0.08, 0.08, 1.6);
      const arm = new THREE.Mesh(armGeo, poleMat);
      arm.position.set(-0.45, 6.8, 0);
      group.add(arm);

      // 5. Dual Directional C-V2X Sector Antennas (Tilted toward highway)
      const panelGeo = new THREE.BoxGeometry(0.24, 1.10, 0.18);
      const panelMat = new THREE.MeshStandardMaterial({{ color: 0xf8fafc, roughness: 0.3 }});
      
      const panel1 = new THREE.Mesh(panelGeo, panelMat);
      panel1.position.set(-0.45, 6.8, 0.55);
      panel1.rotation.y = 0.20;
      panel1.castShadow = true;
      group.add(panel1);

      const panel2 = new THREE.Mesh(panelGeo, panelMat);
      panel2.position.set(-0.45, 6.8, -0.55);
      panel2.rotation.y = -0.20;
      panel2.castShadow = true;
      group.add(panel2);

      // 6. Circular Microwave Backhaul Dish
      const dishGeo = new THREE.CylinderGeometry(0.38, 0.38, 0.18, 16);
      dishGeo.rotateZ(Math.PI / 2);
      const dish = new THREE.Mesh(dishGeo, panelMat);
      dish.position.set(-0.45, 7.3, 0);
      dish.castShadow = true;
      group.add(dish);

      // 7. Auxiliary Photovoltaic Solar Panel (Tilted at 35 degrees)
      const solarGeo = new THREE.BoxGeometry(1.2, 0.06, 0.8);
      const solarMat = new THREE.MeshStandardMaterial({{ color: 0x0f2744, metalness: 0.9, roughness: 0.1 }});
      const solarPanel = new THREE.Mesh(solarGeo, solarMat);
      solarPanel.position.set(-0.45, 5.2, 0.6);
      solarPanel.rotation.x = -0.6;
      group.add(solarPanel);

      // 8. Placard ("RSU-0", "RSU-1")
      const signGeo = new THREE.BoxGeometry(0.05, 0.40, 0.85);
      const signMat = new THREE.MeshStandardMaterial({{ color: 0x0369a1 }});
      const sign = new THREE.Mesh(signGeo, signMat);
      sign.position.set(-0.45, 4.6, 0);
      group.add(sign);

      // 9. 3D Radio Coverage Footprint (R_comm radius)
      const covGroup = new THREE.Group();

      // Outer Perimeter Boundary Ring
      const discGeo = new THREE.RingGeometry(commRadiusM - 1.2, commRadiusM, 64);
      discGeo.rotateX(-Math.PI / 2);
      const discMat = new THREE.MeshBasicMaterial({{
        color: 0x0284c7,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide
      }});
      const covRing = new THREE.Mesh(discGeo, discMat);
      covRing.position.y = 0.12;
      covGroup.add(covRing);

      // Translucent 3D Coverage Dome
      const domeGeo = new THREE.SphereGeometry(commRadiusM, 24, 10, 0, Math.PI * 2, 0, Math.PI / 2);
      const domeMat = new THREE.MeshStandardMaterial({{
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.04,
        wireframe: true,
        side: THREE.DoubleSide
      }});
      const covDome = new THREE.Mesh(domeGeo, domeMat);
      covDome.position.y = 0;
      covGroup.add(covDome);

      group.add(covGroup);

      // RSU Active Execution Computing Pulse
      const rsuHaloGeo = new THREE.RingGeometry(1.8, 2.2, 32);
      rsuHaloGeo.rotateX(-Math.PI / 2);
      const rsuHaloMat = new THREE.MeshBasicMaterial({{
        color: 0x10b981,
        transparent: true,
        opacity: 0.0,
        side: THREE.DoubleSide
      }});
      const rsuHalo = new THREE.Mesh(rsuHaloGeo, rsuHaloMat);
      rsuHalo.position.y = 6.9;
      group.add(rsuHalo);

      group.position.set(wx, 0, rsuZ);
      group.userData = {{
        type: 'rsu',
        eid: eid,
        worldX: wx,
        worldZ: rsuZ,
        antennaY: 6.8,
        covGroup: covGroup,
        rsuHalo: rsuHalo,
        rsuHaloMat: rsuHaloMat
      }};

      scene.add(group);
      return group;
    }}

    // -------------------------------------------------------------
    // Task Offloading & Inter-RSU Handover Visuals
    // -------------------------------------------------------------
    let offloadLineMesh = null;
    let handoverLineMesh = null;
    let packetPulses = [];
    const numPackets = 4;

    const packetGeo = new THREE.SphereGeometry(0.38, 12, 12);
    const packetMat = new THREE.MeshBasicMaterial({{ color: 0x0284c7 }});
    for (let i = 0; i < numPackets; i++) {{
      const p = new THREE.Mesh(packetGeo, packetMat);
      p.visible = false;
      scene.add(p);
      packetPulses.push(p);
    }}

    let offloadCurve = null;

    function clearOffloadVisuals() {{
      if (offloadLineMesh) {{
        scene.remove(offloadLineMesh);
        offloadLineMesh.geometry.dispose();
        offloadLineMesh.material.dispose();
        offloadLineMesh = null;
      }}
      if (handoverLineMesh) {{
        scene.remove(handoverLineMesh);
        handoverLineMesh.geometry.dispose();
        handoverLineMesh.material.dispose();
        handoverLineMesh = null;
      }}
      offloadCurve = null;
      packetPulses.forEach(p => p.visible = false);
    }}

    function createOffloadVector(startPos, endPos, isV2I, isSuccess) {{
      clearOffloadVisuals();
      if (!showNetworkOverlay) return;

      const midX = (startPos.x + endPos.x) / 2;
      const midY = Math.max(startPos.y, endPos.y) + (isV2I ? 4.2 : 2.5);
      const midZ = (startPos.z + endPos.z) / 2;
      const midPos = new THREE.Vector3(midX, midY, midZ);

      offloadCurve = new THREE.QuadraticBezierCurve3(startPos, midPos, endPos);

      // V2I is Infrastructure Cyan (#0284c7); V2V is Direct Emerald (#059669)
      const colorHex = isSuccess ? (isV2I ? 0x0284c7 : 0x059669) : 0xef4444;
      const tubeGeo = new THREE.TubeGeometry(offloadCurve, 24, 0.12, 8, false);
      const tubeMat = new THREE.MeshBasicMaterial({{
        color: colorHex,
        transparent: true,
        opacity: 0.85
      }});
      offloadLineMesh = new THREE.Mesh(tubeGeo, tubeMat);
      scene.add(offloadLineMesh);

      packetMat.color.setHex(colorHex);
      packetPulses.forEach(p => p.visible = true);
    }}

    function createHandoverLink(rsu1Pos, rsu2Pos) {{
      if (!showNetworkOverlay) return;
      const midPos = new THREE.Vector3(
        (rsu1Pos.x + rsu2Pos.x) / 2,
        Math.max(rsu1Pos.y, rsu2Pos.y) + 3.0,
        (rsu1Pos.z + rsu2Pos.z) / 2
      );
      const hoCurve = new THREE.QuadraticBezierCurve3(rsu1Pos, midPos, rsu2Pos);
      const hoGeo = new THREE.TubeGeometry(hoCurve, 20, 0.10, 6, false);
      const hoMat = new THREE.MeshBasicMaterial({{
        color: 0xf59e0b,
        transparent: true,
        opacity: 0.85
      }});
      handoverLineMesh = new THREE.Mesh(hoGeo, hoMat);
      scene.add(handoverLineMesh);
    }}

    // -------------------------------------------------------------
    // Instantiate Entities from Trace
    // -------------------------------------------------------------
    const step0 = simTrace.steps[0];
    step0.vehicles.forEach(v => {{
      const mesh = buildRealistic3DVehicle(v.vid);
      vehicleMeshes.set(v.vid, mesh);
    }});

    step0.edges.forEach(e => {{
      const mesh = buildRealistic3DRsu(e.eid, e);
      rsuMeshes.set(e.eid, mesh);
    }});

    // -------------------------------------------------------------
    // Discrete HUD & Inspector Synchronization
    // -------------------------------------------------------------
    function updateDiscreteHUD(stepIdx) {{
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

      // Handover event string
      let eventLabel = "Continuous Connection";
      if (snap.handovers && snap.handovers.length > 0) {{
        const ho = snap.handovers[0];
        const fromR = (ho.from_rsu !== undefined && ho.from_rsu !== null) ? ho.from_rsu : ho.prev_rsu;
        const toR = (ho.to_rsu !== undefined && ho.to_rsu !== null) ? ho.to_rsu : ho.new_rsu;
        eventLabel = `Handover (RSU ${{fromR}} ➔ RSU ${{toR}})`;
      }}

      // Update Inspector Panel
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
      document.getElementById('insp-event').innerText = eventLabel;

      updateOffloadVisuals(stepIdx);
    }}

    function updateOffloadVisuals(stepIdx) {{
      clearOffloadVisuals();
      if (!showNetworkOverlay) return;
      const snap = simTrace.steps[stepIdx];
      const reqMesh = vehicleMeshes.get(snap.requester_vid);
      if (!reqMesh) return;

      if (snap.dest_kind === 'edge') {{
        const rsuMesh = rsuMeshes.get(snap.dest_id);
        if (rsuMesh) {{
          const startPt = new THREE.Vector3(reqMesh.position.x, 1.6, reqMesh.position.z);
          const endPt = new THREE.Vector3(rsuMesh.userData.worldX, rsuMesh.userData.antennaY, rsuMesh.userData.worldZ);
          createOffloadVector(startPt, endPt, true, snap.success);
        }}
      }} else if (snap.dest_kind === 'vehicle') {{
        const targetMesh = vehicleMeshes.get(snap.dest_id);
        if (targetMesh) {{
          const startPt = new THREE.Vector3(reqMesh.position.x, 1.5, reqMesh.position.z);
          const endPt = new THREE.Vector3(targetMesh.position.x, 1.5, targetMesh.position.z);
          createOffloadVector(startPt, endPt, false, snap.success);
        }}
      }}

      // Render handover link if handover occurred
      if (snap.handovers && snap.handovers.length > 0) {{
        const ho = snap.handovers[0];
        const fromR = (ho.from_rsu !== undefined && ho.from_rsu !== null) ? ho.from_rsu : ho.prev_rsu;
        const toR = (ho.to_rsu !== undefined && ho.to_rsu !== null) ? ho.to_rsu : ho.new_rsu;
        const r1 = rsuMeshes.get(fromR);
        const r2 = rsuMeshes.get(toR);
        if (r1 && r2) {{
          const p1 = new THREE.Vector3(r1.userData.worldX, r1.userData.antennaY, r1.userData.worldZ);
          const p2 = new THREE.Vector3(r2.userData.worldX, r2.userData.antennaY, r2.userData.worldZ);
          createHandoverLink(p1, p2);
        }}
      }}
    }}

    // -------------------------------------------------------------
    // Continuous 60 FPS Deterministic Interpolation Engine
    // -------------------------------------------------------------
    function updateInterpolatedScene(progress, delta, elapsed) {{
      const s0 = Math.max(0, Math.min(totalSteps - 1, Math.floor(progress)));
      const s1 = Math.min(totalSteps - 1, s0 + 1);
      const alpha = progress - s0;

      if (s0 !== lastDiscreteStep) {{
        lastDiscreteStep = s0;
        updateDiscreteHUD(s0);
      }}

      const snap0 = simTrace.steps[s0];
      const snap1 = simTrace.steps[s1];

      // Update transmission vs execution stage pill
      const totLat = Math.max(0.001, snap0.total_latency_s);
      const txRatio = Math.min(0.9, snap0.tx_latency_s / totLat);
      const stagePill = document.getElementById('hud-stage');
      if (snap0.dest_kind === 'local') {{
        stagePill.className = 'status-pill pill-stage';
        stagePill.innerText = 'LOCAL CPU COMPUTE';
      }} else if (alpha < txRatio) {{
        stagePill.className = 'status-pill pill-stage';
        stagePill.innerText = `WIRELESS TX (${{(alpha / txRatio * 100).toFixed(0)}}%)`;
      }} else {{
        stagePill.className = 'status-pill pill-stage';
        stagePill.innerText = `NODE EXECUTION (${{((alpha - txRatio) / (1.0 - txRatio) * 100).toFixed(0)}}%)`;
      }}

      const map0 = new Map();
      snap0.vehicles.forEach(v => map0.set(v.vid, v));
      const map1 = new Map();
      snap1.vehicles.forEach(v => map1.set(v.vid, v));

      const activeVids = new Set([...map0.keys(), ...map1.keys()]);

      // Hide inactive vehicles
      vehicleMeshes.forEach((mesh, vid) => {{
        if (!activeVids.has(vid)) mesh.visible = false;
      }});

      activeVids.forEach(vid => {{
        let vMesh = vehicleMeshes.get(vid);
        if (!vMesh) {{
          vMesh = buildRealistic3DVehicle(vid);
          vehicleMeshes.set(vid, vMesh);
        }}

        const v0 = map0.get(vid);
        const v1 = map1.get(vid);

        if (v0 && v1) {{
          vMesh.visible = true;

          // 1. Forward highway X motion with continuous torus wrap handling
          let x0 = v0.x_m;
          let x1 = v1.x_m;
          if (x1 < x0 - roadLengthM * 0.5) x1 += roadLengthM;
          else if (x1 > x0 + roadLengthM * 0.5) x1 -= roadLengthM;

          let curX = x0 + (x1 - x0) * alpha;
          if (curX > roadLengthM) curX -= roadLengthM;
          if (curX < 0) curX += roadLengthM;
          const wx = toWorldX(curX);

          // 2. Lateral lane Z position with smooth cubic Hermite S-curve
          const l0 = (typeof v0.lane === 'number' && v0.lane >= 0 && v0.lane < laneZ.length) ? v0.lane : 1;
          const l1 = (typeof v1.lane === 'number' && v1.lane >= 0 && v1.lane < laneZ.length) ? v1.lane : 1;
          const z0 = laneZ[l0];
          const z1 = laneZ[l1];
          const laneT = alpha * alpha * (3.0 - 2.0 * alpha);
          const wz = z0 + (z1 - z0) * laneT;

          // 3. Realistic dynamic steering yaw during lane changing
          const laneDelta = z1 - z0;
          const steerYaw = laneDelta * Math.sin(Math.PI * alpha) * 0.055;
          vMesh.rotation.y = steerYaw;

          // 4. Subtle chassis body roll into turn
          const roll = laneDelta * Math.sin(Math.PI * alpha) * -0.012;
          vMesh.rotation.x = roll;

          // 5. Subtle acceleration squat and braking nose-dive
          const accel = (typeof v0.accel_mps2 === 'number') ? v0.accel_mps2 : 0.0;
          const pitch = Math.max(-0.04, Math.min(0.04, accel * 0.006));
          vMesh.rotation.z = -pitch;

          // 6. Asphalt micro-suspension vibration
          const rumble = Math.sin(elapsed * 24.0 + vid * 2.1) * 0.005;
          vMesh.position.set(wx, rumble, wz);

          // 7. Deterministic, drift-free wheel rotation based on physical displacement
          const wheelRad = vMesh.userData.wheelRadius || 0.35;
          const wheelAngle = (curX / wheelRad);
          if (vMesh.userData.wheels) {{
            vMesh.userData.wheels.forEach(w => {{
              w.rotation.z = -wheelAngle;
            }});
          }}

          // 8. State highlights (Requester beacon vs destination halo)
          const isReq = (vid === snap0.requester_vid);
          const isPeerDest = (snap0.dest_kind === 'vehicle' && snap0.dest_id === vid);

          if (isReq) {{
            vMesh.userData.beacon.visible = true;
            vMesh.userData.paintMat.color.setHex(0xb45309);
          }} else if (isPeerDest) {{
            vMesh.userData.beacon.visible = false;
            vMesh.userData.paintMat.color.setHex(0x15803d);
          }} else {{
            vMesh.userData.beacon.visible = false;
            vMesh.userData.paintMat.color.setHex(vMesh.userData.baseColor);
          }}

          // Local CPU computation holographic aura
          if (isReq && snap0.dest_kind === 'local' && showNetworkOverlay) {{
            vMesh.userData.localAuraMat.opacity = 0.85;
          }} else {{
            vMesh.userData.localAuraMat.opacity = 0.0;
          }}

          // Peer vehicle execution halo during execution phase
          if (isPeerDest && alpha >= txRatio && showNetworkOverlay) {{
            vMesh.userData.exeHaloMat.opacity = 0.75;
          }} else {{
            vMesh.userData.exeHaloMat.opacity = 0.0;
          }}

          // Turn indicator blinking during lane changing
          if (Math.abs(laneDelta) > 0.5) {{
            const blink = Math.sin(elapsed * 12.0) > 0;
            if (laneDelta > 0 && vMesh.userData.indLeft) {{
              vMesh.userData.indLeft.visible = blink;
              if (vMesh.userData.indRight) vMesh.userData.indRight.visible = false;
            }} else if (laneDelta < 0 && vMesh.userData.indRight) {{
              vMesh.userData.indRight.visible = blink;
              if (vMesh.userData.indLeft) vMesh.userData.indLeft.visible = false;
            }}
          }} else {{
            if (vMesh.userData.indLeft) vMesh.userData.indLeft.visible = false;
            if (vMesh.userData.indRight) vMesh.userData.indRight.visible = false;
          }}

        }} else if (v0 && !v1) {{
          // Vehicle departing highway in SUMO: glide forward smoothly
          vMesh.visible = (alpha < 0.85);
          const speed = v0.speed_mps || 25.0;
          const curX = v0.x_m + speed * alpha * 0.5;
          const wx = toWorldX(curX);
          const wz = laneZ[v0.lane || 1];
          vMesh.position.set(wx, 0, wz);
        }} else if (!v0 && v1) {{
          // Vehicle entering highway in SUMO: glide in from entry
          vMesh.visible = (alpha > 0.15);
          const speed = v1.speed_mps || 25.0;
          const curX = v1.x_m - speed * (1.0 - alpha) * 0.5;
          const wx = toWorldX(curX);
          const wz = laneZ[v1.lane || 1];
          vMesh.position.set(wx, 0, wz);
        }}
      }});

      // Update RSU computation halo during execution phase
      rsuMeshes.forEach((rsuMesh, eid) => {{
        if (snap0.dest_kind === 'edge' && snap0.dest_id === eid && alpha >= txRatio && showNetworkOverlay) {{
          rsuMesh.userData.rsuHaloMat.opacity = 0.85;
          rsuMesh.userData.rsuHalo.rotation.z = elapsed * 2.5;
        }} else {{
          rsuMesh.userData.rsuHaloMat.opacity = 0.0;
        }}
      }});

      // Dynamic offload arc tracking
      if (offloadCurve && offloadLineMesh && showNetworkOverlay) {{
        const reqMesh = vehicleMeshes.get(snap0.requester_vid);
        let endPt = null;
        if (snap0.dest_kind === 'edge') {{
          const rsuMesh = rsuMeshes.get(snap0.dest_id);
          if (rsuMesh) endPt = new THREE.Vector3(rsuMesh.userData.worldX, rsuMesh.userData.antennaY, rsuMesh.userData.worldZ);
        }} else if (snap0.dest_kind === 'vehicle') {{
          const targetMesh = vehicleMeshes.get(snap0.dest_id);
          if (targetMesh) endPt = new THREE.Vector3(targetMesh.position.x, 1.5, targetMesh.position.z);
        }}

        if (reqMesh && endPt) {{
          const startPt = new THREE.Vector3(reqMesh.position.x, 1.6, reqMesh.position.z);
          offloadCurve.v0.copy(startPt);
          offloadCurve.v1.set((startPt.x + endPt.x)/2, Math.max(startPt.y, endPt.y) + (snap0.dest_kind === 'edge' ? 4.2 : 2.5), (startPt.z + endPt.z)/2);
          offloadCurve.v2.copy(endPt);

          offloadLineMesh.geometry.dispose();
          offloadLineMesh.geometry = new THREE.TubeGeometry(offloadCurve, 20, 0.12, 6, false);
        }}
      }}

      // Smooth camera follow tracking with frame-rate independent exponential lerp
      if (cameraFollow) {{
        const reqMesh = vehicleMeshes.get(snap0.requester_vid);
        if (reqMesh) {{
          const targetPt = new THREE.Vector3(reqMesh.position.x, 1.2, reqMesh.position.z);
          let camOffset;
          if (activeCameraMode === 'chase') camOffset = new THREE.Vector3(-22, 5.5, 0);
          else if (activeCameraMode === 'side') camOffset = new THREE.Vector3(0, 8.5, 28);
          else if (activeCameraMode === 'top') camOffset = new THREE.Vector3(0, 75, -reqMesh.position.z);
          else camOffset = new THREE.Vector3(-34, 18, 30); // Highway 3/4 default

          const desiredCam = targetPt.clone().add(camOffset);

          // Wrap check for smooth camera tracking across highway corridor
          if (Math.abs(controls.target.x - targetPt.x) > 200) {{
            controls.target.copy(targetPt);
            camera.position.copy(desiredCam);
          }} else {{
            const lerpFactor = 1.0 - Math.exp(-5.0 * delta);
            controls.target.lerp(targetPt, lerpFactor);
            camera.position.lerp(desiredCam, lerpFactor);
          }}
        }}
      }}
    }}

    // -------------------------------------------------------------
    // Camera Presets & Navigation
    // -------------------------------------------------------------
    function updateCamButtonStyles(activeMode) {{
      ['iso', 'chase', 'side', 'top'].forEach(m => {{
        const btn = document.getElementById('btn-cam-' + m);
        if (btn) btn.className = (m === activeMode) ? 'hud-btn active' : 'hud-btn';
      }});
    }}

    function resetCamera() {{
      activeCameraMode = 'iso';
      updateCamButtonStyles('iso');
      const snap = (simTrace && simTrace.steps) ? simTrace.steps[currentStepIdx] : null;
      let rx = 0;
      let rz = 0;
      if (snap) {{
        const reqMesh = vehicleMeshes.get(snap.requester_vid);
        if (reqMesh) {{
          rx = reqMesh.position.x;
          rz = reqMesh.position.z;
        }} else if (snap.vehicles && snap.vehicles.length > 0) {{
          rx = toWorldX(snap.vehicles[0].x_m);
          rz = laneZ[snap.vehicles[0].lane || 0] || 0;
        }}
      }}
      camera.position.set(rx - 34, 18, rz + 30);
      controls.target.set(rx, 1.2, rz);
      controls.update();
    }}

    function setCameraPreset(mode) {{
      activeCameraMode = mode;
      updateCamButtonStyles(mode);
      const snap = (simTrace && simTrace.steps) ? simTrace.steps[currentStepIdx] : null;
      let rx = 0;
      let rz = 0;
      if (snap) {{
        const reqMesh = vehicleMeshes.get(snap.requester_vid);
        if (reqMesh) {{
          rx = reqMesh.position.x;
          rz = reqMesh.position.z;
        }} else if (snap.vehicles && snap.vehicles.length > 0) {{
          rx = toWorldX(snap.vehicles[0].x_m);
          rz = laneZ[snap.vehicles[0].lane || 0] || 0;
        }}
      }}

      if (mode === 'iso') {{
        camera.position.set(rx - 34, 18, rz + 30);
        controls.target.set(rx, 1.2, rz);
      }} else if (mode === 'chase') {{
        camera.position.set(rx - 22, 5.5, rz);
        controls.target.set(rx + 25, 1.5, rz);
      }} else if (mode === 'side') {{
        camera.position.set(rx, 8.5, rz + 28);
        controls.target.set(rx, 1.8, rz);
      }} else if (mode === 'top') {{
        camera.position.set(rx, 75, 0);
        controls.target.set(rx, 0, 0);
      }}
      controls.update();
    }}

    function toggleCoverage() {{
      showCoverageFootprint = !showCoverageFootprint;
      rsuMeshes.forEach(r => {{
        if (r.userData.covGroup) r.userData.covGroup.visible = showCoverageFootprint;
      }});
      const btn = document.getElementById('btn-cov');
      btn.className = showCoverageFootprint ? 'hud-btn active' : 'hud-btn';
    }}

    function toggleNetworkOverlay() {{
      showNetworkOverlay = !showNetworkOverlay;
      const btn = document.getElementById('btn-net');
      btn.className = showNetworkOverlay ? 'hud-btn active' : 'hud-btn';
      updateOffloadVisuals(currentStepIdx);
    }}

    function toggleCameraFollow() {{
      cameraFollow = !cameraFollow;
      const btn = document.getElementById('btn-follow');
      if (btn) btn.className = cameraFollow ? 'hud-btn active' : 'hud-btn';
    }}

    // Set initial button active states
    document.getElementById('btn-cov').className = 'hud-btn active';
    document.getElementById('btn-net').className = 'hud-btn active';
    document.getElementById('btn-follow').className = 'hud-btn active';

    // -------------------------------------------------------------
    // Timeline Playback Controls (Smooth Tweening & 60 FPS)
    // -------------------------------------------------------------
    function smoothTweenTo(targetStep) {{
      if (isPlaying) togglePlay();
      tweenStartProgress = currentProgress;
      targetProgress = Math.max(0, Math.min(totalSteps - 1, targetStep));
      tweenStartTime = clock.getElapsedTime();
      isTweening = true;
    }}

    function stepFirst() {{ smoothTweenTo(0.0); }}
    function stepLast() {{ smoothTweenTo(totalSteps - 1.0); }}
    function stepPrev() {{ smoothTweenTo(Math.max(0.0, Math.floor(currentProgress - 0.05))); }}
    function stepNext() {{ smoothTweenTo(Math.min(totalSteps - 1.0, Math.floor(currentProgress + 1.0))); }}

    function onScrubStart() {{
      isScrubbing = true;
      if (isPlaying) togglePlay();
    }}

    function onScrubEnd() {{
      isScrubbing = false;
    }}

    function onScrubInput(val) {{
      currentProgress = parseFloat(val);
      isTweening = false;
    }}

    function onScrub(val) {{
      onScrubInput(val);
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      isTweening = false;
      const btn = document.getElementById('btn-play');
      if (isPlaying) {{
        btn.innerText = '⏸️';
        btn.title = 'Pause';
        if (currentProgress >= totalSteps - 1.0) {{
          currentProgress = 0.0;
        }}
      }} else {{
        btn.innerText = '▶️';
        btn.title = 'Play';
      }}
    }}

    function onSpeedChange(val) {{
      playbackSpeed = parseFloat(val);
    }}

    // -------------------------------------------------------------
    // Raycaster for Interactive Entity Selection
    // -------------------------------------------------------------
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    window.addEventListener('click', (event) => {{
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
            const nameLabel = vData.sumo_vid ? `${{vData.sumo_vid}} (V${{vid}})` : `Vehicle ${{vid}}`;
            const rsuLabel = (vData.connected_rsu !== null && vData.connected_rsu !== undefined) ? `RSU ${{vData.connected_rsu}}` : 'Out of range';
            document.getElementById('insp-header').innerText = `${{nameLabel}} Telemetry`;
            document.getElementById('insp-req').innerText = `Lane: ${{vData.lane}} | RSU: ${{rsuLabel}}`;
            document.getElementById('insp-dest').innerText = `Speed: ${{vData.speed_mps.toFixed(1)}} m/s`;
            document.getElementById('insp-size').innerText = `CPU: ${{vData.cpu_ghz.toFixed(2)}} GHz`;
            document.getElementById('insp-cycles').innerText = `Queue: ${{vData.queue_mcyc.toFixed(1)}} Mcyc`;
            document.getElementById('insp-trust').innerText = `${{vData.trust.toFixed(3)}}`;
            document.getElementById('insp-event').innerText = (vData.is_requester ? 'Active Requester' : 'Candidate Peer');
          }}
          break;
        }} else if (obj && obj.userData.type === 'rsu') {{
          const eid = obj.userData.eid;
          const snap = simTrace.steps[currentStepIdx];
          const eData = snap.edges.find(e => e.eid === eid);
          if (eData) {{
            document.getElementById('insp-header').innerText = `RSU ${{eid}} (Edge Infrastructure)`;
            document.getElementById('insp-req').innerText = `Position: ${{eData.x_m.toFixed(0)}}m`;
            document.getElementById('insp-dest').innerText = `Coverage: ${{eData.comm_range_m}}m`;
            document.getElementById('insp-size').innerText = `CPU Clock: ${{eData.cpu_ghz.toFixed(1)}} GHz`;
            document.getElementById('insp-cycles').innerText = `Queue: ${{eData.queue_mcyc.toFixed(1)}} Mcyc`;
            document.getElementById('insp-trust').innerText = `${{eData.trust.toFixed(3)}}`;
            document.getElementById('insp-event').innerText = 'Stationary Edge Server';
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
      const delta = Math.min(clock.getDelta(), 0.1);
      const elapsed = clock.getElapsedTime();

      // Continuous Playback Progression
      if (isPlaying) {{
        currentProgress += delta * playbackSpeed;
        if (currentProgress >= totalSteps - 1.0) {{
          currentProgress = totalSteps - 1.0;
          togglePlay();
        }}
      }} else if (isTweening) {{
        const t = Math.min(1.0, (elapsed - tweenStartTime) / tweenDuration);
        const ease = 1.0 - Math.pow(1.0 - t, 3.0);
        currentProgress = tweenStartProgress + (targetProgress - tweenStartProgress) * ease;
        if (t >= 1.0) {{
          currentProgress = targetProgress;
          isTweening = false;
        }}
      }}

      // Smooth Scrubber Sync
      if (!isScrubbing) {{
        document.getElementById('timeline-slider').value = currentProgress;
      }}
      document.getElementById('scrub-step-txt').innerText = `Step ${{(Math.min(totalSteps, currentProgress + 1)).toFixed(1)}} / ${{totalSteps}}`;

      // 60 FPS continuous interpolation
      updateInterpolatedScene(currentProgress, delta, elapsed);

      // Animate packet pulses along offload vector during transmission stage
      if (offloadCurve && packetPulses[0].visible) {{
        packetPulses.forEach((p, idx) => {{
          const t = (elapsed * 1.6 + idx * (1.0 / numPackets)) % 1.0;
          const pt = offloadCurve.getPointAt(t);
          p.position.copy(pt);
        }});
      }}

      // Rotate local CPU execution aura
      vehicleMeshes.forEach(v => {{
        if (v.userData.localAura && v.userData.localAuraMat.opacity > 0) {{
          v.userData.localAura.rotation.z = elapsed * 2.5;
        }}
        if (v.userData.exeHalo && v.userData.exeHaloMat.opacity > 0) {{
          v.userData.exeHalo.rotation.z = -elapsed * 2.5;
        }}
      }});

      controls.update();
      renderer.render(scene, camera);
    }}

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    // Initial render
    updateDiscreteHUD(0);
    updateInterpolatedScene(0, 0.016, 0);
    resetCamera();
    animate();
  </script>
</body>
</html>"""

    return html_template
