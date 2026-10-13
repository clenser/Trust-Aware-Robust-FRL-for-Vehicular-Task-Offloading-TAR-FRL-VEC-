"""
Vehicular simulation engine and visual rendering module.
Provides:
- Policy resolution (heuristics + trained DQN / FedDQN checkpoints)
- Multi-lane 2D spatial coordinate mapping
- Full step-by-step episode tracing with channel & computing telemetry
- Scalable SVG road scene rendering with RSU radio coverage zones and offload beams
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import baselines
from environment.vanet_env import VehicularEdgeEnv, EnvConfig, StepResult, Candidate
from environment.sumo_env import SumoVehicularEdgeEnv
from environment.sumo_adapter import is_sumo_available, get_sumo_status
from environment.network import link_available, link_rate_hz
from rl.agent import DQNAgent

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_available_policy_names() -> list[str]:
    """Return all available policy options including trained checkpoints if present."""
    policies = [
        "resource_aware",
        "nearest_edge",
        "random_valid",
        "local_only",
    ]
    dqn_path = REPO_ROOT / "checkpoints" / "dqn_default" / "checkpoint_final.pt"
    if dqn_path.exists():
        policies.append("dqn_trained_default")
    fed_path = REPO_ROOT / "checkpoints" / "feddqn_default" / "global_final.pt"
    if fed_path.exists():
        policies.append("feddqn_global_final")
    return policies


def get_policy_label(name: str) -> str:
    """Format user-facing label for policies."""
    labels = {
        "resource_aware": "Resource-Aware Heuristic (Queue & Channel)",
        "nearest_edge": "Nearest Edge Node (RSU First)",
        "random_valid": "Random Valid Candidate",
        "local_only": "Local Vehicle CPU Only",
        "dqn_trained_default": "DQN Agent (Trained Final Checkpoint)",
        "feddqn_global_final": "FedDQN Agent (Global Aggregated Model)",
    }
    return labels.get(name, name)


def load_simulation_policy(
    policy_name: str,
    seed: int = 42,
    top_k: int = 5
) -> Tuple[Callable[[list[float], list[bool]], int], str]:
    """
    Resolve policy callable and description.
    Falls back gracefully if checkpoint fails or dimension mismatch.
    """
    if policy_name in baselines.POLICIES:
        pol = baselines.make_policy(policy_name, seed=seed)
        return pol, f"Baseline heuristic: {policy_name}"

    if policy_name == "dqn_trained_default":
        dqn_path = REPO_ROOT / "checkpoints" / "dqn_default" / "checkpoint_final.pt"
        if dqn_path.exists():
            try:
                agent = DQNAgent.load(dqn_path)
                agent.epsilon = 0.0  # Pure greedy evaluation
                return (lambda obs, mask, ctx=None: agent.act(obs, mask)), "Trained Single-Agent DQN (checkpoint_final.pt)"
            except Exception as e:
                # Fallback to resource aware
                return baselines.make_policy("resource_aware", seed=seed), f"DQN Load Error ({e}), fallback: resource_aware"

    if policy_name == "feddqn_global_final":
        fed_path = REPO_ROOT / "checkpoints" / "feddqn_default" / "global_final.pt"
        if fed_path.exists():
            try:
                ckpt = torch.load(fed_path, map_location="cpu", weights_only=True)
                # Obs dim = 7 + 3*top_k = 22 for top_k=5
                agent = DQNAgent(obs_dim=22, action_dim=6, seed=seed)
                agent.q_net.load_state_dict(ckpt)
                agent.epsilon = 0.0
                return (lambda obs, mask, ctx=None: agent.act(obs, mask)), "Trained FedDQN Global Model (global_final.pt)"
            except Exception as e:
                return baselines.make_policy("resource_aware", seed=seed), f"FedDQN Load Error ({e}), fallback: resource_aware"

    return baselines.make_policy("resource_aware", seed=seed), "Default: resource_aware"


@dataclass
class StepSnapshot:
    """Detailed telemetry and spatial state for a single simulation step."""
    step_idx: int
    task_id: int
    arrival_s: float
    requester_vid: int
    data_bits: float
    cycles: float
    deadline_s: float
    priority: float
    
    # Node snapshots
    vehicles: list[dict]
    edges: list[dict]
    candidates: list[dict]
    
    # Decision taken
    action: int
    dest_kind: str
    dest_id: int
    dest_name: str
    dest_trust: float
    
    # Timing & outcome
    tx_latency_s: float
    exe_latency_s: float
    total_latency_s: float
    energy_j: float
    success: bool
    deadline_missed: bool
    failed: bool
    
    # Updated trust
    updated_trust: float


def run_traced_episode(
    cfg: EnvConfig,
    policy_name: str,
    max_tasks: int = 30,
    mobility_mode: str = "synthetic",
) -> Tuple[list[StepSnapshot], dict]:
    """
    Run an episode on SumoVehicularEdgeEnv (or fallback VehicularEdgeEnv) and capture granular per-step snapshots.
    """
    use_sumo = (mobility_mode == "sumo") and is_sumo_available()
    if use_sumo:
        env = SumoVehicularEdgeEnv(cfg, use_sumo=True)
        mobility_engine = "SUMO Microscopic Traffic (TraCI)"
    else:
        env = VehicularEdgeEnv(cfg)
        mobility_engine = "Synthetic Fallback (Torus 1D)"

    obs, mask = env.reset()
    policy, pol_desc = load_simulation_policy(policy_name, seed=cfg.seed, top_k=cfg.top_k)

    snapshots: list[StepSnapshot] = []

    for step_i in range(max_tasks):
        if env.pending is None:
            break

        t = env.pending
        req = next((v for v in env.vehicles if v.vid == t.requester_id), env.vehicles[0])
        cands = env.candidates()

        # Capture pre-decision candidates
        cand_dicts = []
        for c in cands:
            cand_node = env._node_of(c) if hasattr(env, "_node_of") else (env.edges[c.node_id] if c.kind == "edge" else next((v for v in env.vehicles if v.vid == c.node_id), req))
            ch_rate = link_rate_hz(
                c.dist_m, req.bandwidth_hz, req.tx_power_w,
                cfg.pl_k0, cfg.pl_gamma, cfg.noise_w
            )
            cand_dicts.append({
                "kind": c.kind,
                "node_id": c.node_id,
                "name": f"Edge {c.node_id}" if c.kind == "edge" else f"Veh {c.node_id}",
                "dist_m": c.dist_m,
                "link_rate_mbps": ch_rate / 1e6,
                "trust": env.trust.get(c.kind, c.node_id),
                "queue_mcyc": cand_node.queue_cycles / 1e6,
                "freq_ghz": (cand_node.cpu_freq_hz if c.kind == "edge" else cand_node.effective_freq_hz) / 1e9,
            })

        # Pre-step snapshot of all vehicles
        veh_snapshots = []
        for v in env.vehicles:
            if hasattr(env, "_sumo_states") and v.vid in env._sumo_states:
                s_state = env._sumo_states[v.vid]
                lane = s_state.lane_idx
                heading_deg = s_state.heading_deg
                sumo_vid = s_state.vid
                veh_type = s_state.veh_type
                accel_mps2 = s_state.accel_mps2
            else:
                lane = v.vid % 3
                heading_deg = 90.0
                sumo_vid = f"v_{v.vid}"
                veh_type = "passenger_sedan" if v.vid % 3 == 0 else ("passenger_suv" if v.vid % 3 == 1 else "commercial_van")
                accel_mps2 = 0.0

            connected_rsu = env.vehicle_associations.get(v.vid) if hasattr(env, "vehicle_associations") else None

            veh_snapshots.append({
                "vid": v.vid,
                "sumo_vid": sumo_vid,
                "veh_type": veh_type,
                "x_m": v.x_m,
                "speed_mps": v.speed_mps,
                "accel_mps2": accel_mps2,
                "heading_deg": heading_deg,
                "lane": lane,
                "connected_rsu": connected_rsu,
                "cpu_ghz": v.cpu_freq_hz / 1e9,
                "effective_cpu_ghz": v.effective_freq_hz / 1e9,
                "utilisation": v.utilisation,
                "queue_mcyc": v.queue_cycles / 1e6,
                "trust": env.trust.get("vehicle", v.vid),
                "available": v.available,
                "is_requester": (v.vid == t.requester_id),
            })

        # Pre-step snapshot of all edge nodes
        edge_snapshots = []
        for e in env.edges:
            edge_snapshots.append({
                "eid": e.eid,
                "x_m": e.x_m,
                "cpu_ghz": e.cpu_freq_hz / 1e9,
                "queue_mcyc": e.queue_cycles / 1e6,
                "trust": env.trust.get("edge", e.eid),
                "comm_range_m": cfg.comm_range_m,
                "available": e.available,
            })

        # Select action
        obs, mask = env.current_observation()
        ctx = {"candidates": cands, "env": env}
        try:
            action = int(policy(obs, mask, ctx))
        except TypeError:
            try:
                action = int(policy(obs, mask))
            except Exception:
                action = 0
        except Exception:
            # Fallback to local
            action = 0

        # Validate action
        if action < 0 or action >= len(mask) or not mask[action]:
            action = 0

        # Execute step
        res = env.step(action)

        dest_name = (
            f"Local (V{req.vid})" if res.dest_kind == "local"
            else f"RSU {res.dest_id}" if res.dest_kind == "edge"
            else f"Vehicle {res.dest_id}"
        )
        dest_key = "vehicle" if res.dest_kind == "local" else res.dest_kind
        updated_trust = env.trust.get(dest_key, res.dest_id)

        # Transmission vs execution latency decomposition
        if res.dest_kind == "local":
            tx_s = 0.0
            exe_s = res.latency_s
        else:
            cand_match = [c for c in cands if c.kind == res.dest_kind and c.node_id == res.dest_id]
            if cand_match:
                dist = cand_match[0].dist_m
                rate = link_rate_hz(dist, req.bandwidth_hz, req.tx_power_w, cfg.pl_k0, cfg.pl_gamma, cfg.noise_w)
                tx_s = t.data_bits / max(1.0, rate) + cfg.handshake_s
                exe_s = max(0.0, res.latency_s - tx_s)
            else:
                tx_s = 0.0
                exe_s = res.latency_s

        snapshot = StepSnapshot(
            step_idx=step_i,
            task_id=t.task_id,
            arrival_s=t.arrival_s,
            requester_vid=t.requester_id,
            data_bits=t.data_bits,
            cycles=t.cycles,
            deadline_s=t.deadline_s,
            priority=t.priority,
            vehicles=veh_snapshots,
            edges=edge_snapshots,
            candidates=cand_dicts,
            action=action,
            dest_kind=res.dest_kind,
            dest_id=res.dest_id,
            dest_name=dest_name,
            dest_trust=res.dest_trust,
            tx_latency_s=tx_s,
            exe_latency_s=exe_s,
            total_latency_s=res.latency_s,
            energy_j=res.energy_j,
            success=res.success,
            deadline_missed=res.deadline_missed,
            failed=res.failed,
            updated_trust=updated_trust,
        )
        snapshots.append(snapshot)

    # Calculate overall episode summary
    n = len(snapshots)
    successes = sum(s.success for s in snapshots)
    deadline_misses = sum(s.deadline_missed for s in snapshots)
    failures = sum(s.failed for s in snapshots)
    mean_lat = float(np.mean([s.total_latency_s for s in snapshots])) if snapshots else 0.0
    mean_ene = float(np.mean([s.energy_j for s in snapshots])) if snapshots else 0.0
    offload_frac = float(np.mean([s.action > 0 for s in snapshots])) if snapshots else 0.0

    handovers_list = getattr(env, "handovers", [])
    total_handovers = sum(1 for h in handovers_list if h.event_type == "handover")
    total_associations = sum(1 for h in handovers_list if h.event_type == "associate")
    total_disconnections = sum(1 for h in handovers_list if h.event_type == "disconnect")

    summary = {
        "total_tasks": n,
        "successes": successes,
        "deadline_misses": deadline_misses,
        "failures": failures,
        "success_rate": successes / max(1, n),
        "mean_latency_s": mean_lat,
        "mean_energy_j": mean_ene,
        "offload_fraction": offload_frac,
        "policy_name": policy_name,
        "policy_description": pol_desc,
        "mobility_engine": mobility_engine,
        "total_handovers": total_handovers,
        "total_associations": total_associations,
        "total_disconnections": total_disconnections,
        "handover_events": [
            {
                "step": h.step_idx,
                "vid": h.vid,
                "sumo_vid": h.sumo_vid,
                "x_m": h.x_m,
                "from_rsu": h.from_rsu,
                "to_rsu": h.to_rsu,
                "type": h.event_type
            }
            for h in handovers_list
        ],
    }

    if hasattr(env, "close"):
        env.close()

    return snapshots, summary


def generate_road_scene_svg(
    snapshot: StepSnapshot,
    road_length_m: float = 1000.0,
    width_px: int = 960,
    height_px: int = 240,
) -> str:
    """
    Generate responsive SVG markup representing the 2D multi-lane highway,
    roadside RSUs with radio coverage discs, moving vehicles, and active offload beams.
    """
    pad_x = 40
    usable_w = width_px - 2 * pad_x

    # Coordinate mapping: road x [0, road_length] -> SVG x [pad_x, pad_x + usable_w]
    def to_svg_x(x_m: float) -> float:
        norm = (x_m % road_length_m) / max(1.0, road_length_m)
        return pad_x + norm * usable_w

    # Geometry layout
    # Y-coordinates:
    # 0 - 55: Roadside RSU zone
    # 60: Road Top Edge
    # 60 - 170: 3 Lanes (each 36px wide):
    #   Lane 0: center y = 78
    #   Lane 1: center y = 114
    #   Lane 2: center y = 150
    # 170: Road Bottom Edge
    # 175 - 230: Distance markers & legend
    lane_y = [78, 114, 150]
    rsu_y = 32

    svg_parts = []
    svg_parts.append(
        f'<svg viewBox="0 0 {width_px} {height_px}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background-color: #0b0f19; border-radius: 10px; width: 100%; height: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">'
    )

    # SVG Defs for gradients & glowing filters
    svg_parts.append("""
    <defs>
      <!-- Glow filter for lasers and highlights -->
      <filter id="glow-beam" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <!-- RSU Coverage Radial Gradient -->
      <radialGradient id="rsu-cov-grad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#06b6d4" stop-opacity="0.18" />
        <stop offset="70%" stop-color="#06b6d4" stop-opacity="0.08" />
        <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.0" />
      </radialGradient>
      <!-- Requester halo gradient -->
      <radialGradient id="req-halo-grad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.6" />
        <stop offset="100%" stop-color="#f59e0b" stop-opacity="0.0" />
      </radialGradient>
    </defs>
    """)

    # 1. Background Grid & Distance Markers
    svg_parts.append(f'<line x1="{pad_x}" y1="175" x2="{pad_x + usable_w}" y2="175" stroke="#334155" stroke-width="1"/>')
    num_ticks = 6
    for i in range(num_ticks):
        x_val = i * (road_length_m / (num_ticks - 1))
        sx = to_svg_x(x_val)
        svg_parts.append(f'<line x1="{sx}" y1="175" x2="{sx}" y2="182" stroke="#64748b" stroke-width="1.5"/>')
        svg_parts.append(
            f'<text x="{sx}" y="196" fill="#94a3b8" font-size="11" font-family="monospace" text-anchor="middle">{int(x_val)}m</text>'
        )

    # 2. Highway Asphalt & Lane Markings
    # Road bed
    svg_parts.append(
        f'<rect x="{pad_x}" y="60" width="{usable_w}" height="110" fill="#182234" rx="4" stroke="#334155" stroke-width="1.5"/>'
    )
    # Yellow shoulder lines (top and bottom)
    svg_parts.append(f'<line x1="{pad_x}" y1="62" x2="{pad_x + usable_w}" y2="62" stroke="#eab308" stroke-width="2"/>')
    svg_parts.append(f'<line x1="{pad_x}" y1="168" x2="{pad_x + usable_w}" y2="168" stroke="#eab308" stroke-width="2"/>')

    # Dashed white lane dividers
    svg_parts.append(
        f'<line x1="{pad_x}" y1="96" x2="{pad_x + usable_w}" y2="96" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="16,14"/>'
    )
    svg_parts.append(
        f'<line x1="{pad_x}" y1="132" x2="{pad_x + usable_w}" y2="132" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="16,14"/>'
    )

    # 3. Roadside RSUs & Radio Coverage Discs
    for edge in snapshot.edges:
        ex = to_svg_x(edge["x_m"])
        cov_radius_px = (edge["comm_range_m"] / road_length_m) * usable_w

        # Coverage circle / disc centered at RSU
        svg_parts.append(
            f'<circle cx="{ex}" cy="{rsu_y}" r="{cov_radius_px:.1f}" '
            f'fill="url(#rsu-cov-grad)" stroke="#06b6d4" stroke-width="1" stroke-dasharray="6,4" opacity="0.85">'
            f'<title>RSU {edge["eid"]} Coverage Zone: {edge["comm_range_m"]}m radius</title>'
            f'</circle>'
        )
        # Handle wrap-around coverage visual if near edges
        if ex - cov_radius_px < pad_x:
            wrap_cx = ex + usable_w
            svg_parts.append(
                f'<circle cx="{wrap_cx:.1f}" cy="{rsu_y}" r="{cov_radius_px:.1f}" '
                f'fill="url(#rsu-cov-grad)" stroke="#06b6d4" stroke-width="1" stroke-dasharray="6,4" opacity="0.4"/>'
            )
        elif ex + cov_radius_px > pad_x + usable_w:
            wrap_cx = ex - usable_w
            svg_parts.append(
                f'<circle cx="{wrap_cx:.1f}" cy="{rsu_y}" r="{cov_radius_px:.1f}" '
                f'fill="url(#rsu-cov-grad)" stroke="#06b6d4" stroke-width="1" stroke-dasharray="6,4" opacity="0.4"/>'
            )

        # RSU Tower Icon
        # Mast
        svg_parts.append(f'<line x1="{ex}" y1="{rsu_y - 14}" x2="{ex}" y2="{rsu_y + 14}" stroke="#38bdf8" stroke-width="2.5"/>')
        svg_parts.append(f'<path d="M {ex-8} {rsu_y+14} L {ex+8} {rsu_y+14} L {ex} {rsu_y-14} Z" fill="none" stroke="#38bdf8" stroke-width="1.2"/>')
        # Transmitter dish / emitter
        svg_parts.append(f'<circle cx="{ex}" cy="{rsu_y - 14}" r="4" fill="#0284c7" stroke="#38bdf8" stroke-width="1.5"/>')
        # Signal pulse waves
        svg_parts.append(
            f'<path d="M {ex-7} {rsu_y-20} A 8 8 0 0 1 {ex+7} {rsu_y-20}" fill="none" stroke="#06b6d4" stroke-width="1.5">'
            f'<animate attributeName="opacity" values="0.2;1;0.2" dur="1.5s" repeatCount="indefinite"/>'
            f'</path>'
        )

        # RSU Label
        svg_parts.append(
            f'<rect x="{ex-28}" y="{rsu_y - 30}" width="56" height="13" fill="#0f172a" rx="3" stroke="#0284c7" stroke-width="0.8"/>'
            f'<text x="{ex}" y="{rsu_y - 20}" fill="#38bdf8" font-size="9" font-family="sans-serif" font-weight="bold" text-anchor="middle">'
            f'RSU {edge["eid"]} ({edge["cpu_ghz"]:.1f}G)'
            f'</text>'
        )

    # 4. Offload Beams / Links
    # Find requester vehicle coordinate
    req_v = next((v for v in snapshot.vehicles if v["vid"] == snapshot.requester_vid), None)
    if req_v is not None:
        rx = to_svg_x(req_v["x_m"])
        ry = lane_y[req_v["lane"]]

        # Determine beam color based on outcome
        if snapshot.success:
            beam_color = "#10b981"  # Emerald Green
            beam_label = "SUCCESS"
        elif snapshot.deadline_missed:
            beam_color = "#ef4444"  # Ruby Red
            beam_label = "DEADLINE MISS"
        else:
            beam_color = "#f59e0b"  # Amber
            beam_label = "FAILURE"

        if snapshot.dest_kind == "edge":
            target_edge = next((e for e in snapshot.edges if e["eid"] == snapshot.dest_id), None)
            if target_edge:
                tx = to_svg_x(target_edge["x_m"])
                ty = rsu_y

                # Beam line
                svg_parts.append(
                    f'<line x1="{rx}" y1="{ry}" x2="{tx}" y2="{ty}" '
                    f'stroke="{beam_color}" stroke-width="3" stroke-dasharray="8,5" filter="url(#glow-beam)">'
                    f'<animate attributeName="stroke-dashoffset" from="30" to="0" dur="0.8s" repeatCount="indefinite"/>'
                    f'</line>'
                )
                # Animated packet pulse circle
                svg_parts.append(
                    f'<circle r="5" fill="{beam_color}" filter="url(#glow-beam)">'
                    f'<animateMotion path="M {rx} {ry} L {tx} {ty}" dur="0.8s" repeatCount="indefinite" />'
                    f'</circle>'
                )

        elif snapshot.dest_kind == "vehicle":
            target_veh = next((v for v in snapshot.vehicles if v["vid"] == snapshot.dest_id), None)
            if target_veh:
                tx = to_svg_x(target_veh["x_m"])
                ty = lane_y[target_veh["lane"]]

                svg_parts.append(
                    f'<line x1="{rx}" y1="{ry}" x2="{tx}" y2="{ty}" '
                    f'stroke="{beam_color}" stroke-width="3" stroke-dasharray="6,4" filter="url(#glow-beam)">'
                    f'<animate attributeName="stroke-dashoffset" from="24" to="0" dur="0.8s" repeatCount="indefinite"/>'
                    f'</line>'
                )
                svg_parts.append(
                    f'<circle r="4" fill="{beam_color}" filter="url(#glow-beam)">'
                    f'<animateMotion path="M {rx} {ry} L {tx} {ty}" dur="0.8s" repeatCount="indefinite" />'
                    f'</circle>'
                )

        elif snapshot.dest_kind == "local":
            # Local processing pulsing ring
            svg_parts.append(
                f'<circle cx="{rx}" cy="{ry}" r="22" fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-dasharray="4,3">'
                f'<animate attributeName="r" values="18;25;18" dur="1.2s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.9;0.3;0.9" dur="1.2s" repeatCount="indefinite"/>'
                f'</circle>'
            )

    # 5. Vehicles
    for v in snapshot.vehicles:
        vx = to_svg_x(v["x_m"])
        vy = lane_y[v["lane"]]
        is_req = v["is_requester"]
        is_dest = (snapshot.dest_kind == "vehicle" and snapshot.dest_id == v["vid"])

        # Vehicle styling
        if is_req:
            body_fill = "#f59e0b"  # Bright Gold / Requester
            body_stroke = "#fbbf24"
            # Halo around requester
            svg_parts.append(
                f'<circle cx="{vx}" cy="{vy}" r="20" fill="url(#req-halo-grad)">'
                f'<animate attributeName="r" values="16;22;16" dur="1.0s" repeatCount="indefinite"/>'
                f'</circle>'
            )
        elif is_dest:
            body_fill = "#10b981"  # Emerald Target
            body_stroke = "#34d399"
        else:
            body_fill = "#2563eb"  # Electric Blue Standard
            body_stroke = "#60a5fa"

        # Vehicle Body (Rounded box representing car)
        car_w = 34
        car_h = 18
        svg_parts.append(
            f'<rect x="{vx - car_w/2}" y="{vy - car_h/2}" width="{car_w}" height="{car_h}" '
            f'fill="{body_fill}" rx="4" stroke="{body_stroke}" stroke-width="1.5">'
            f'<title>Vehicle {v["vid"]} | Speed: {v["speed_mps"]:.1f} m/s | Trust: {v["trust"]:.2f} | CPU: {v["cpu_ghz"]:.1f} GHz</title>'
            f'</rect>'
        )

        # Windshield / Cabin
        svg_parts.append(
            f'<rect x="{vx - car_w/4}" y="{vy - car_h/3}" width="{car_w/2}" height="{car_h*2/3}" '
            f'fill="#0f172a" rx="2" opacity="0.6"/>'
        )

        # Headlights (rightward facing)
        svg_parts.append(
            f'<circle cx="{vx + car_w/2 - 2}" cy="{vy - 5}" r="1.5" fill="#fef08a"/>'
            f'<circle cx="{vx + car_w/2 - 2}" cy="{vy + 5}" r="1.5" fill="#fef08a"/>'
        )

        # Vehicle ID Label inside car
        label_color = "#000000" if is_req else "#ffffff"
        svg_parts.append(
            f'<text x="{vx}" y="{vy + 3.5}" fill="{label_color}" font-size="9" font-family="sans-serif" font-weight="bold" text-anchor="middle">'
            f'V{v["vid"]}'
            f'</text>'
        )

        # Speed and Trust Pill below vehicle
        svg_parts.append(
            f'<text x="{vx}" y="{vy + 16}" fill="#94a3b8" font-size="8" font-family="monospace" text-anchor="middle">'
            f'{v["speed_mps"]:.0f}m/s · T:{v["trust"]:.2f}'
            f'</text>'
        )

    # 6. Top Legend / Status Bar inside SVG
    status_text = f"Step {snapshot.step_idx + 1} | Task #{snapshot.task_id} from V{snapshot.requester_vid} ➔ {snapshot.dest_name}"
    outcome_badge_color = "#10b981" if snapshot.success else ("#ef4444" if snapshot.deadline_missed else "#f59e0b")
    outcome_str = "SUCCESS" if snapshot.success else ("DEADLINE EXCEEDED" if snapshot.deadline_missed else "FAILED")

    svg_parts.append(
        f'<rect x="{pad_x}" y="208" width="{usable_w}" height="24" fill="#0f172a" rx="5" stroke="#1e293b" stroke-width="1"/>'
        f'<text x="{pad_x + 12}" y="224" fill="#cbd5e1" font-size="11" font-family="sans-serif">'
        f'{status_text}'
        f'</text>'
        f'<rect x="{pad_x + usable_w - 140}" y="212" width="130" height="16" fill="{outcome_badge_color}" rx="3"/>'
        f'<text x="{pad_x + usable_w - 75}" y="224" fill="#ffffff" font-size="9.5" font-family="sans-serif" font-weight="bold" text-anchor="middle">'
        f'{outcome_str} ({snapshot.total_latency_s:.2f}s)'
        f'</text>'
    )

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)
