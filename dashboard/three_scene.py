"""
Three.js 3D WebGL realistic daylight vehicular edge-network simulation renderer.
Generates an interactive, publication-grade realistic 3D daytime road scene featuring:
- Natural daylight illumination, sky dome, atmospheric fog, and soft PBR shadows
- Realistic multi-lane highway (3 lanes at 3.6m width, road shoulders, W-beam guardrails)
- Roadside environment: grass verges, trees, shrubs, and distance marker posts
- Highly detailed 3D composite vehicle models (Sedan, SUV, Commercial Van) with realistic automotive paints,
  transparent windshields, side mirrors, wheels with rims, and directional headlights
- Authentic roadside communication infrastructure (RSU): concrete pad, equipment cabinet,
  galvanized steel pole (7.5m), directional sector antennas, microwave dish, and optional coverage footprint
- Professional, restrained task offloading overlay with toggleable packet vectors
- OrbitControls with daylight glassmorphic HUD, camera presets, timeline scrubber, and raycaster inspector
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
        })

    trace_json = json.dumps({
        "road_length_m": getattr(cfg, "road_length_m", 1000.0),
        "comm_range_m": getattr(cfg, "comm_range_m", 300.0),
        "num_vehicles": getattr(cfg, "num_vehicles", len(snapshots[0].vehicles if snapshots else 6)),
        "num_edges": getattr(cfg, "num_edge_nodes", len(snapshots[0].edges if snapshots else 2)),
        "summary": summary,
        "steps": steps_data,
    })

    cdn_fallback_three = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>' if not three_js else f"<script>{three_js}</script>"
    cdn_fallback_orbit = '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>' if not orbit_js else f"<script>{orbit_js}</script>"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VANET FRL Realistic Daylight 3D Simulation</title>
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
    }}
    .hud-badge {{
      background: rgba(255, 255, 255, 0.90);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(203, 213, 225, 0.85);
      border-radius: 8px;
      padding: 8px 16px;
      font-size: 0.82rem;
      font-weight: 600;
      color: #334155;
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
      padding: 3px 9px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .pill-success {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
    .pill-danger {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
    .pill-warning {{ background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }}

    /* Camera View Presets & Toggles */
    .cam-presets {{
      display: flex;
      gap: 6px;
      pointer-events: auto;
    }}
    .hud-btn {{
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(203, 213, 225, 0.9);
      color: #334155;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
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
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(203, 213, 225, 0.85);
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
      color: #1e293b;
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
      color: #64748b;
      font-weight: 600;
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
      color: #334155;
      padding: 5px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      outline: none;
      cursor: pointer;
    }}

    /* Floating Inspector Panel */
    .inspector-panel {{
      position: absolute;
      top: 64px;
      right: 16px;
      width: 290px;
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 8px;
      padding: 14px;
      z-index: 10;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
      font-size: 0.8rem;
    }}
    .inspector-title {{
      font-size: 0.82rem;
      font-weight: 700;
      color: #0369a1;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 6px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }}
    .prop-row {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      color: #475569;
    }}
    .prop-val {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-weight: 600;
      color: #0f172a;
    }}

    /* Legend Box */
    .legend-box {{
      position: absolute;
      bottom: 74px;
      left: 16px;
      background: rgba(255, 255, 255, 0.90);
      border: 1px solid rgba(203, 213, 225, 0.85);
      border-radius: 6px;
      padding: 6px 14px;
      display: flex;
      gap: 14px;
      font-size: 0.72rem;
      color: #475569;
      font-weight: 600;
      z-index: 10;
      backdrop-filter: blur(8px);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
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
      <span id="hud-outcome" class="status-pill pill-success">SUCCESS</span>
    </div>
    <div class="cam-presets">
      <button class="hud-btn" id="btn-cov" onclick="toggleCoverage()">📡 Coverage Footprint</button>
      <button class="hud-btn" id="btn-net" onclick="toggleNetworkOverlay()">📶 Network Overlay</button>
      <button class="hud-btn" onclick="setCameraPreset('iso')">🎥 Highway 3/4</button>
      <button class="hud-btn" onclick="setCameraPreset('chase')">🚗 Chase Cam</button>
      <button class="hud-btn" onclick="setCameraPreset('side')">🔭 Roadside</button>
      <button class="hud-btn" onclick="setCameraPreset('top')">🗺️ Top-Down</button>
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
    <div class="prop-row"><span>Dest Trust Score:</span><span class="prop-val" id="insp-trust">0.850</span></div>
  </div>

  <!-- Legend -->
  <div class="legend-box">
    <div class="leg-item"><div class="leg-dot" style="background:#2563eb;"></div> Vehicle</div>
    <div class="leg-item"><div class="leg-dot" style="background:#f59e0b;"></div> Requester</div>
    <div class="leg-item"><div class="leg-dot" style="background:#475569;"></div> Roadside RSU</div>
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
    if (!window.WebGLRenderingContext) {{
      document.getElementById('webgl-error').style.display = 'block';
    }}

    const simTrace = {trace_json};
    const totalSteps = simTrace.steps.length;
    let currentStepIdx = 0;
    let isPlaying = false;
    let playTimer = null;
    let playIntervalMs = 800;

    let showCoverageFootprint = true;
    let showNetworkOverlay = true;

    document.getElementById('timeline-slider').max = Math.max(0, totalSteps - 1);

    // -------------------------------------------------------------
    // Three.js Scene Setup with Realistic Daylight & Color Management
    // -------------------------------------------------------------
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();

    // Natural daytime sky and atmospheric fog
    scene.background = new THREE.Color(0xa7cbf0);
    scene.fog = new THREE.FogExp2(0xa7cbf0, 0.0010);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 3500);
    const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: 'high-performance' }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    // Official Three.js Photorealistic Lighting & Color Management Configuration
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    controls.minDistance = 15;
    controls.maxDistance = 1200;

    // -------------------------------------------------------------
    // Natural Sunlight & Hemispheric Sky Illumination
    // -------------------------------------------------------------
    // Sky blue ambient + warm grass ground bounce reflection
    const hemiLight = new THREE.HemisphereLight(0xbfe0ff, 0x475535, 0.65);
    scene.add(hemiLight);

    // Warm Sun Directional Light
    const sunLight = new THREE.DirectionalLight(0xfffaea, 1.25);
    sunLight.position.set(160, 240, 110);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 10;
    sunLight.shadow.camera.far = 1000;
    const sArea = 350;
    sunLight.shadow.camera.left = -sArea;
    sunLight.shadow.camera.right = sArea;
    sunLight.shadow.camera.top = sArea;
    sunLight.shadow.camera.bottom = -sArea;
    sunLight.shadow.bias = -0.0003;
    scene.add(sunLight);

    // Soft sky fill light
    const fillLight = new THREE.DirectionalLight(0xcde3fa, 0.35);
    fillLight.position.set(-160, 80, -110);
    scene.add(fillLight);

    // -------------------------------------------------------------
    // Physical Road Geometry & Roadside Environment (1:1 Metric Scale)
    // -------------------------------------------------------------
    // 1 Three.js unit = 1 meter!
    const roadLengthM = simTrace.road_length_m || 1000.0;
    const halfRoadM = roadLengthM / 2;

    function toWorldX(xMeters) {{
      const wrapped = (xMeters % roadLengthM + roadLengthM) % roadLengthM;
      return (wrapped - halfRoadM);
    }}

    // 1. Natural Grass Terrain
    const terrainGeo = new THREE.PlaneGeometry(roadLengthM + 400, 240, 32, 32);
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
    // With paved shoulders = 15.6m total paved asphalt width (Z in [-7.8, +7.8])
    const roadAsphaltGeo = new THREE.PlaneGeometry(roadLengthM, 15.6);
    roadAsphaltGeo.rotateX(-Math.PI / 2);
    const roadAsphaltMat = new THREE.MeshStandardMaterial({{
      color: 0x22272e,
      roughness: 0.85,
      metalness: 0.08
    }});
    const roadBed = new THREE.Mesh(roadAsphaltGeo, roadAsphaltMat);
    roadBed.position.y = 0.05;
    roadBed.receiveShadow = true;
    scene.add(roadBed);

    // 3. Pavement Markings: Solid Edge Lines and Dashed Lane Dividers
    // Solid Edge Lines (white 0.2m wide at Z = -5.4m and Z = +5.4m)
    const lineWhiteMat = new THREE.MeshBasicMaterial({{ color: 0xf1f5f9 }});
    const edgeLineGeo = new THREE.PlaneGeometry(roadLengthM, 0.22);
    edgeLineGeo.rotateX(-Math.PI / 2);

    const edgeNorth = new THREE.Mesh(edgeLineGeo, lineWhiteMat);
    edgeNorth.position.set(0, 0.065, -5.4);
    scene.add(edgeNorth);

    const edgeSouth = new THREE.Mesh(edgeLineGeo, lineWhiteMat);
    edgeSouth.position.set(0, 0.065, 5.4);
    scene.add(edgeSouth);

    // Yellow Boundary Shoulder Lines (at Z = -7.3m and Z = +7.3m)
    const lineYellowMat = new THREE.MeshBasicMaterial({{ color: 0xebb305 }});
    const yellowLineGeo = new THREE.PlaneGeometry(roadLengthM, 0.25);
    yellowLineGeo.rotateX(-Math.PI / 2);

    const yellowNorth = new THREE.Mesh(yellowLineGeo, lineYellowMat);
    yellowNorth.position.set(0, 0.065, -7.3);
    scene.add(yellowNorth);

    const yellowSouth = new THREE.Mesh(yellowLineGeo, lineYellowMat);
    yellowSouth.position.set(0, 0.065, 7.3);
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

    // 4. Steel W-Beam Guardrails (along both sides at Z = -7.9m and Z = +7.9m)
    const railMat = new THREE.MeshStandardMaterial({{
      color: 0x94a3b8,
      metalness: 0.85,
      roughness: 0.35
    }});
    const railBeamGeo = new THREE.BoxGeometry(roadLengthM, 0.35, 0.08);

    const railN = new THREE.Mesh(railBeamGeo, railMat);
    railN.position.set(0, 0.75, -7.9);
    railN.castShadow = true;
    scene.add(railN);

    const railS = new THREE.Mesh(railBeamGeo, railMat);
    railS.position.set(0, 0.75, 7.9);
    railS.castShadow = true;
    scene.add(railS);

    // Guardrail Support Posts (every 4.0m)
    const postGeo = new THREE.BoxGeometry(0.12, 0.85, 0.12);
    for (let x = -halfRoadM + 2; x < halfRoadM; x += 4.0) {{
      const pN = new THREE.Mesh(postGeo, railMat);
      pN.position.set(x, 0.42, -7.9);
      pN.castShadow = true;
      scene.add(pN);

      const pS = new THREE.Mesh(postGeo, railMat);
      pS.position.set(x, 0.42, 7.9);
      pS.castShadow = true;
      scene.add(pS);
    }}

    // 5. Roadside Vegetation (Procedural Trees & Shrubs along the highway)
    function createTree(x, z, scaleFactor) {{
      const treeGroup = new THREE.Group();
      // Trunk
      const trunkGeo = new THREE.CylinderGeometry(0.35 * scaleFactor, 0.5 * scaleFactor, 3.8 * scaleFactor, 8);
      const trunkMat = new THREE.MeshStandardMaterial({{ color: 0x4a3728, roughness: 0.9 }});
      const trunk = new THREE.Mesh(trunkGeo, trunkMat);
      trunk.position.y = 1.9 * scaleFactor;
      trunk.castShadow = true;
      treeGroup.add(trunk);

      // Layered Canopy
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

    // Plant trees deterministically outside the road clearance corridor (Z < -11m and Z > +11m)
    for (let x = -halfRoadM + 15; x < halfRoadM - 15; x += 22.0) {{
      const offsetZ1 = -13.0 - ((Math.abs(x * 13) % 17));
      createTree(x, offsetZ1, 0.85 + ((Math.abs(x * 7) % 5) * 0.08));

      const offsetZ2 = 13.0 + ((Math.abs(x * 19) % 17));
      createTree(x + 10, offsetZ2, 0.90 + ((Math.abs(x * 11) % 5) * 0.08));
    }}

    // 6. Kilometer & Distance Marker Posts along Highway
    for (let m = 0; m <= roadLengthM; m += 200) {{
      const wx = toWorldX(m);
      const mPostGeo = new THREE.BoxGeometry(0.25, 1.2, 0.25);
      const mPostMat = new THREE.MeshStandardMaterial({{ color: 0x0284c7 }});
      const mPost = new THREE.Mesh(mPostGeo, mPostMat);
      mPost.position.set(wx, 0.6, -8.6);
      mPost.castShadow = true;
      scene.add(mPost);
    }}

    // Driving Lane Centers (3 Lanes: 3.6m width)
    const laneZ = [-3.6, 0.0, 3.6];

    // -------------------------------------------------------------
    // Realistic 3D Procedural Vehicles (1:1 Metric Scale)
    // -------------------------------------------------------------
    const vehicleMeshes = new Map();

    // Realistic Automotive Paint Palette
    const paintPalette = [
      0x1c2d44, // Deep Sapphire Navy
      0xc0c6cb, // Silver Metallic
      0xeef2f5, // Glacier White
      0x373e48, // Graphite Metallic
      0x7a1820, // Crimson Red
      0x14171a, // Obsidian Black
    ];

    function buildRealistic3DVehicle(vid) {{
      const group = new THREE.Group();
      const vehType = vid % 3; // 0: Sedan, 1: SUV, 2: Delivery Van
      const bodyColor = paintPalette[vid % paintPalette.length];

      const paintMat = new THREE.MeshStandardMaterial({{
        color: bodyColor,
        roughness: 0.28,
        metalness: 0.82
      }});

      const glassMat = new THREE.MeshStandardMaterial({{
        color: 0x111827,
        roughness: 0.05,
        metalness: 0.95,
        transparent: true,
        opacity: 0.85
      }});

      const trimMat = new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.9 }});
      const chromeMat = new THREE.MeshStandardMaterial({{ color: 0xe2e8f0, metalness: 0.95, roughness: 0.1 }});
      const rubberMat = new THREE.MeshStandardMaterial({{ color: 0x171717, roughness: 0.95 }});
      const rimMat = new THREE.MeshStandardMaterial({{ color: 0xcfd4dc, metalness: 0.9, roughness: 0.2 }});

      if (vehType === 0) {{
        // ------------------ SEDAN (4.6m L, 1.85m W, 1.45m H) ------------------
        // Lower Chassis Body
        const lowerBodyGeo = new THREE.BoxGeometry(4.6, 0.65, 1.85);
        const lowerBody = new THREE.Mesh(lowerBodyGeo, paintMat);
        lowerBody.position.y = 0.62;
        lowerBody.castShadow = true;
        lowerBody.receiveShadow = true;
        group.add(lowerBody);

        // Aerodynamic Hood Bonnet
        const hoodGeo = new THREE.BoxGeometry(1.4, 0.25, 1.75);
        const hood = new THREE.Mesh(hoodGeo, paintMat);
        hood.position.set(1.4, 0.95, 0);
        hood.castShadow = true;
        group.add(hood);

        // Rear Trunk Deck
        const trunkGeo = new THREE.BoxGeometry(1.0, 0.28, 1.75);
        const trunk = new THREE.Mesh(trunkGeo, paintMat);
        trunk.position.set(-1.6, 0.95, 0);
        trunk.castShadow = true;
        group.add(trunk);

        // Cabin Greenhouse (Windshields & Roof)
        const cabinGeo = new THREE.BoxGeometry(2.3, 0.65, 1.6);
        const cabin = new THREE.Mesh(cabinGeo, glassMat);
        cabin.position.set(-0.15, 1.25, 0);
        cabin.castShadow = true;
        group.add(cabin);

        // Roof Panel
        const roofGeo = new THREE.BoxGeometry(1.8, 0.08, 1.55);
        const roof = new THREE.Mesh(roofGeo, paintMat);
        roof.position.set(-0.15, 1.58, 0);
        roof.castShadow = true;
        group.add(roof);

        // Side Mirrors
        const mirrorGeo = new THREE.BoxGeometry(0.12, 0.12, 0.22);
        const mLeft = new THREE.Mesh(mirrorGeo, paintMat);
        mLeft.position.set(0.7, 1.05, 0.95);
        const mRight = new THREE.Mesh(mirrorGeo, paintMat);
        mRight.position.set(0.7, 1.05, -0.95);
        group.add(mLeft);
        group.add(mRight);

      }} else if (vehType === 1) {{
        // ------------------ SUV / CROSSOVER (4.85m L, 1.95m W, 1.72m H) ------------------
        const lowerBodyGeo = new THREE.BoxGeometry(4.85, 0.85, 1.95);
        const lowerBody = new THREE.Mesh(lowerBodyGeo, paintMat);
        lowerBody.position.y = 0.78;
        lowerBody.castShadow = true;
        group.add(lowerBody);

        // Matte Black Underbody Cladding
        const trimGeo = new THREE.BoxGeometry(4.88, 0.25, 1.98);
        const trimMesh = new THREE.Mesh(trimGeo, trimMat);
        trimMesh.position.y = 0.45;
        group.add(trimMesh);

        // Taller Cabin & Cargo Tailgate
        const cabinGeo = new THREE.BoxGeometry(2.9, 0.82, 1.75);
        const cabin = new THREE.Mesh(cabinGeo, glassMat);
        cabin.position.set(-0.35, 1.45, 0);
        cabin.castShadow = true;
        group.add(cabin);

        const roofGeo = new THREE.BoxGeometry(2.6, 0.10, 1.7);
        const roof = new THREE.Mesh(roofGeo, paintMat);
        roof.position.set(-0.35, 1.88, 0);
        group.add(roof);

        // Roof Luggage Rails
        const railGeo = new THREE.BoxGeometry(2.2, 0.06, 0.06);
        const rL = new THREE.Mesh(railGeo, chromeMat);
        rL.position.set(-0.35, 1.95, 0.75);
        const rR = new THREE.Mesh(railGeo, chromeMat);
        rR.position.set(-0.35, 1.95, -0.75);
        group.add(rL);
        group.add(rR);

      }} else {{
        // ------------------ COMMERCIAL CARGO VAN (5.2m L, 2.0m W, 2.15m H) ------------------
        const vanBodyGeo = new THREE.BoxGeometry(5.2, 1.45, 2.0);
        const vanBody = new THREE.Mesh(vanBodyGeo, paintMat);
        vanBody.position.set(-0.2, 1.15, 0);
        vanBody.castShadow = true;
        group.add(vanBody);

        // Front Cab Windshield
        const cabGeo = new THREE.BoxGeometry(1.3, 0.75, 1.85);
        const cab = new THREE.Mesh(cabGeo, glassMat);
        cab.position.set(1.4, 1.35, 0);
        group.add(cab);

        // Bumper Grille
        const bumperGeo = new THREE.BoxGeometry(0.4, 0.5, 1.95);
        const bumper = new THREE.Mesh(bumperGeo, trimMat);
        bumper.position.set(2.4, 0.55, 0);
        group.add(bumper);
      }}

      // 4 Realistic Wheels (Tire + Alloy Spoke Rims)
      const wheelRadius = (vehType === 1) ? 0.38 : 0.34;
      const wheelWidth = 0.24;
      const tireGeo = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 18);
      tireGeo.rotateX(Math.PI / 2);

      const rimGeo = new THREE.CylinderGeometry(wheelRadius * 0.65, wheelRadius * 0.65, wheelWidth + 0.02, 12);
      rimGeo.rotateX(Math.PI / 2);

      const wheelZ = (vehType === 1) ? 0.98 : 0.92;
      const wheelFrontX = (vehType === 2) ? 1.6 : 1.45;
      const wheelRearX = (vehType === 2) ? -1.7 : -1.45;

      const wheelCoords = [
        [wheelFrontX, wheelRadius, wheelZ],
        [wheelFrontX, wheelRadius, -wheelZ],
        [wheelRearX, wheelRadius, wheelZ],
        [wheelRearX, wheelRadius, -wheelZ]
      ];

      wheelCoords.forEach(pos => {{
        const wMesh = new THREE.Mesh(tireGeo, rubberMat);
        const rMesh = new THREE.Mesh(rimGeo, rimMat);
        wMesh.add(rMesh);
        wMesh.position.set(...pos);
        wMesh.castShadow = true;
        group.add(wMesh);
      }});

      // Realistic Front Headlights (Projector LED White)
      const hlMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
      const hlGeo = new THREE.BoxGeometry(0.12, 0.18, 0.38);
      const hlL = new THREE.Mesh(hlGeo, hlMat);
      hlL.position.set(2.28, 0.75, 0.72);
      const hlR = new THREE.Mesh(hlGeo, hlMat);
      hlR.position.set(2.28, 0.75, -0.72);
      group.add(hlL);
      group.add(hlR);

      // Realistic Rear Taillights (Red LED)
      const tlMat = new THREE.MeshBasicMaterial({{ color: 0xdc2626 }});
      const tlGeo = new THREE.BoxGeometry(0.12, 0.18, 0.38);
      const tlL = new THREE.Mesh(tlGeo, tlMat);
      tlL.position.set(-2.28, 0.75, 0.72);
      const tlR = new THREE.Mesh(tlGeo, tlMat);
      tlR.position.set(-2.28, 0.75, -0.72);
      group.add(tlL);
      group.add(tlR);

      // License Plates (Front & Rear)
      const plateMat = new THREE.MeshBasicMaterial({{ color: 0xfef08a }});
      const plateGeo = new THREE.BoxGeometry(0.06, 0.14, 0.42);
      const plateF = new THREE.Mesh(plateGeo, plateMat);
      plateF.position.set(2.32, 0.45, 0);
      const plateR = new THREE.Mesh(plateGeo, plateMat);
      plateR.position.set(-2.32, 0.52, 0);
      group.add(plateF);
      group.add(plateR);

      // Subtle Connected Vehicle Requester Beacon
      const beaconGeo = new THREE.CylinderGeometry(0.15, 0.15, 0.12, 12);
      const beaconMat = new THREE.MeshBasicMaterial({{ color: 0xf59e0b }});
      const beacon = new THREE.Mesh(beaconGeo, beaconMat);
      beacon.position.y = (vehType === 2) ? 2.22 : (vehType === 1 ? 1.95 : 1.65);
      beacon.visible = false;
      group.add(beacon);

      // Local CPU Execution Subtle Ring
      const localAuraGeo = new THREE.RingGeometry(1.6, 1.85, 24);
      localAuraGeo.rotateX(-Math.PI / 2);
      const localAuraMat = new THREE.MeshBasicMaterial({{
        color: 0x8b5cf6,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.0
      }});
      const localAura = new THREE.Mesh(localAuraGeo, localAuraMat);
      localAura.position.y = (vehType === 2) ? 2.3 : 1.8;
      group.add(localAura);

      group.userData = {{
        type: 'vehicle',
        vid: vid,
        paintMat: paintMat,
        beacon: beacon,
        localAura: localAura,
        localAuraMat: localAuraMat,
        baseColor: bodyColor
      }};
      scene.add(group);
      return group;
    }}

    // -------------------------------------------------------------
    // Realistic Roadside Unit (RSU) Infrastructure
    // -------------------------------------------------------------
    const rsuMeshes = new Map();
    const commRadiusM = simTrace.comm_range_m || 300.0;

    function buildRealistic3DRsu(eid, edgeData) {{
      const group = new THREE.Group();
      const wx = toWorldX(edgeData.x_m);
      const rsuZ = -10.5; // Roadside grass verge clearance

      // 1. Concrete Foundation Pad
      const padGeo = new THREE.BoxGeometry(1.8, 0.25, 1.8);
      const padMat = new THREE.MeshStandardMaterial({{ color: 0x94a3b8, roughness: 0.95 }});
      const pad = new THREE.Mesh(padGeo, padMat);
      pad.position.y = 0.125;
      pad.receiveShadow = true;
      group.add(pad);

      // 2. Weatherproof Equipment Cabinet (NEMA Enclosure)
      const cabGeo = new THREE.BoxGeometry(0.85, 1.35, 0.65);
      const cabMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, roughness: 0.5, metalness: 0.3 }});
      const cabinet = new THREE.Mesh(cabGeo, cabMat);
      cabinet.position.set(0.35, 0.92, 0);
      cabinet.castShadow = true;
      group.add(cabinet);

      // 3. Galvanized Steel Support Mast (Plausible 7.5m Height)
      const poleGeo = new THREE.CylinderGeometry(0.11, 0.14, 7.5, 16);
      const poleMat = new THREE.MeshStandardMaterial({{ color: 0x718096, metalness: 0.85, roughness: 0.3 }});
      const pole = new THREE.Mesh(poleGeo, poleMat);
      pole.position.set(-0.45, 3.85, 0);
      pole.castShadow = true;
      group.add(pole);

      // 4. Crossarm Mounting Bracket at 6.8m
      const armGeo = new THREE.BoxGeometry(0.08, 0.08, 1.4);
      const arm = new THREE.Mesh(armGeo, poleMat);
      arm.position.set(-0.45, 6.8, 0);
      group.add(arm);

      // 5. C-V2X Sector Panel Antenna (Tilted toward highway)
      const panelGeo = new THREE.BoxGeometry(0.25, 0.95, 0.18);
      const panelMat = new THREE.MeshStandardMaterial({{ color: 0xf8fafc, roughness: 0.3 }});
      const panelAnt = new THREE.Mesh(panelGeo, panelMat);
      panelAnt.position.set(-0.45, 6.8, 0.5);
      panelAnt.rotation.y = 0.25;
      panelAnt.castShadow = true;
      group.add(panelAnt);

      // 6. Circular Microwave Backhaul Dish
      const dishGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.18, 16);
      dishGeo.rotateZ(Math.PI / 2);
      const dish = new THREE.Mesh(dishGeo, panelMat);
      dish.position.set(-0.45, 7.2, -0.45);
      dish.castShadow = true;
      group.add(dish);

      // 7. Identifiable Placard
      const signGeo = new THREE.BoxGeometry(0.05, 0.35, 0.65);
      const signMat = new THREE.MeshStandardMaterial({{ color: 0x0369a1 }});
      const sign = new THREE.Mesh(signGeo, signMat);
      sign.position.set(-0.45, 4.8, 0);
      group.add(sign);

      // 8. 3D Radio Coverage Footprint (R_comm = 300m scaled)
      const covGroup = new THREE.Group();

      const discGeo = new THREE.RingGeometry(commRadiusM - 1.2, commRadiusM, 64);
      discGeo.rotateX(-Math.PI / 2);
      const discMat = new THREE.MeshBasicMaterial({{
        color: 0x0284c7,
        transparent: true,
        opacity: 0.30,
        side: THREE.DoubleSide
      }});
      const covRing = new THREE.Mesh(discGeo, discMat);
      covRing.position.y = 0.15;
      covGroup.add(covRing);

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

      group.position.set(wx, 0, rsuZ);
      group.userData = {{
        type: 'rsu',
        eid: eid,
        worldX: wx,
        worldZ: rsuZ,
        antennaY: 6.8,
        covGroup: covGroup
      }};
      scene.add(group);
      return group;
    }}

    // -------------------------------------------------------------
    // Professional Task Offloading Visualization Overlay
    // -------------------------------------------------------------
    let offloadLineMesh = null;
    let packetPulses = [];
    const numPackets = 3;

    const packetGeo = new THREE.SphereGeometry(0.45, 12, 12);
    const packetMat = new THREE.MeshBasicMaterial({{ color: 0x10b981 }});
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
      offloadCurve = null;
      packetPulses.forEach(p => p.visible = false);
    }}

    function createProfessionalOffloadVector(startPos, endPos, isSuccess) {{
      clearOffloadVisuals();
      if (!showNetworkOverlay) return;

      const midX = (startPos.x + endPos.x) / 2;
      const midY = Math.max(startPos.y, endPos.y) + 4.5;
      const midZ = (startPos.z + endPos.z) / 2;
      const midPos = new THREE.Vector3(midX, midY, midZ);

      offloadCurve = new THREE.QuadraticBezierCurve3(startPos, midPos, endPos);

      const colorHex = isSuccess ? 0x10b981 : 0xef4444;
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

    // -------------------------------------------------------------
    // Instantiate Entities from Step 0
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

      // Update Vehicle Positions and Realistic State Highlights
      const activeVids = new Set(snap.vehicles.map(v => v.vid));
      vehicleMeshes.forEach((mesh, vid) => {{
        mesh.visible = activeVids.has(vid);
      }});

      snap.vehicles.forEach(vData => {{
        let vMesh = vehicleMeshes.get(vData.vid);
        if (!vMesh) {{
          vMesh = buildRealistic3DVehicle(vData.vid);
          vehicleMeshes.set(vData.vid, vMesh);
        }}
        vMesh.visible = true;

        const wx = toWorldX(vData.x_m);
        const lIdx = (typeof vData.lane === 'number' && vData.lane >= 0 && vData.lane < laneZ.length) ? vData.lane : 1;
        const wz = laneZ[lIdx];
        vMesh.position.set(wx, 0, wz);

        const isReq = (vData.vid === snap.requester_vid);
        const isPeerDest = (snap.dest_kind === 'vehicle' && snap.dest_id === vData.vid);

        // Highlight requester and target realistically
        if (isReq) {{
          vMesh.userData.beacon.visible = true;
          vMesh.userData.paintMat.color.setHex(0xb45309); // Highlighted amber automotive
        }} else if (isPeerDest) {{
          vMesh.userData.beacon.visible = false;
          vMesh.userData.paintMat.color.setHex(0x15803d); // Subdued emerald target
        }} else {{
          vMesh.userData.beacon.visible = false;
          vMesh.userData.paintMat.color.setHex(vMesh.userData.baseColor);
        }}

        // Local onboard CPU execution aura
        if (isReq && snap.dest_kind === 'local' && showNetworkOverlay) {{
          vMesh.userData.localAuraMat.opacity = 0.85;
        }} else {{
          vMesh.userData.localAuraMat.opacity = 0.0;
        }}
      }});

      // Update Network Offloading Vectors
      clearOffloadVisuals();
      const reqMesh = vehicleMeshes.get(snap.requester_vid);

      if (snap.dest_kind === 'edge' && reqMesh) {{
        const rsuMesh = rsuMeshes.get(snap.dest_id);
        if (rsuMesh) {{
          const startPt = new THREE.Vector3(reqMesh.position.x, 1.6, reqMesh.position.z);
          const endPt = new THREE.Vector3(rsuMesh.userData.worldX, rsuMesh.userData.antennaY, rsuMesh.userData.worldZ);
          createProfessionalOffloadVector(startPt, endPt, snap.success);
        }}
      }} else if (snap.dest_kind === 'vehicle' && reqMesh) {{
        const targetMesh = vehicleMeshes.get(snap.dest_id);
        if (targetMesh) {{
          const startPt = new THREE.Vector3(reqMesh.position.x, 1.5, reqMesh.position.z);
          const endPt = new THREE.Vector3(targetMesh.position.x, 1.5, targetMesh.position.z);
          createProfessionalOffloadVector(startPt, endPt, snap.success);
        }}
      }}
    }}

    // -------------------------------------------------------------
    // Camera Presets & Navigation
    // -------------------------------------------------------------
    function resetCamera() {{
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
      // Highway 3/4 elevated view framed directly on the active requester vehicle
      camera.position.set(rx - 32, 16, rz + 28);
      controls.target.set(rx, 1.2, rz);
      controls.update();
    }}

    function setCameraPreset(mode) {{
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
        camera.position.set(rx - 32, 16, rz + 28);
        controls.target.set(rx, 1.2, rz);
      }} else if (mode === 'chase') {{
        camera.position.set(rx - 20, 5.5, rz);
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
      updateStepVisuals(currentStepIdx);
    }}

    // Set initial button active states
    document.getElementById('btn-cov').className = 'hud-btn active';
    document.getElementById('btn-net').className = 'hud-btn active';

    // -------------------------------------------------------------
    // Timeline Playback Controls
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

      // Animate packet pulses along offload vector
      if (offloadCurve && packetPulses[0].visible) {{
        packetPulses.forEach((p, idx) => {{
          const t = (elapsed * 1.1 + idx * (1.0 / numPackets)) % 1.0;
          const pt = offloadCurve.getPointAt(t);
          p.position.copy(pt);
        }});
      }}

      // Rotate local CPU execution aura
      vehicleMeshes.forEach(v => {{
        if (v.userData.localAura && v.userData.localAuraMat.opacity > 0) {{
          v.userData.localAura.rotation.z = elapsed * 2.5;
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
    updateStepVisuals(0);
    resetCamera();
    animate();
  </script>
</body>
</html>"""

    return html_template
