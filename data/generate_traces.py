"""
Calibrated Synthetic Trace Generator & Loader for VANET Offloading Simulation.

Note on Provenance:
These traces are synthetically generated models calibrated against published
empirical distributions and standards (e.g., NGSIM highway statistics, Manhattan
grid urban models, and 3GPP/ETSI V2X specifications). They are synthetic simulation
benchmarks and not raw, direct sensor measurements.
"""

import numpy as np
import pandas as pd
import os

def generate_ngsim_highway_trace(filename="data/synthetic_traces/synthetic_ngsim_highway.csv", num_vehicles=60, duration_seconds=300, dt=0.5):
    """
    Generates a synthetic highway trajectory trace calibrated to statistical
    dynamics reported in the FHWA NGSIM US-101 highway dataset (headway distributions,
    speed variance, and lane-change frequencies).
    Saves to data/synthetic_traces/ to protect authentic raw NGSIM source data.
    """
    np.random.seed(42)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    records = []
    lane_width = 3.75  # standard highway lane width (m)
    road_length = 2400.0  # 2.4 km full highway segment
    num_lanes = 5
    
    # Initialize vehicles
    vehicles = []
    for vid in range(num_vehicles):
        lane = np.random.randint(1, num_lanes + 1)
        # distribute along highway
        x = (vid / num_vehicles) * road_length + np.random.uniform(-10, 10)
        y = (lane - 0.5) * lane_width
        speed = np.random.uniform(16.0, 30.0)  # ~58 - 108 km/h
        vclass = np.random.choice(["Car", "SUV", "Truck", "Autonomous_EV"], p=[0.55, 0.25, 0.10, 0.10])
        # Computational capabilities (GHz CPU, Memory MB)
        if vclass == "Autonomous_EV":
            cpu_capacity = np.random.uniform(4.0, 8.0)
        elif vclass == "Car":
            cpu_capacity = np.random.uniform(2.0, 4.0)
        elif vclass == "SUV":
            cpu_capacity = np.random.uniform(2.5, 4.5)
        else: # Truck
            cpu_capacity = np.random.uniform(3.0, 5.0)
            
        vehicles.append({
            "vid": vid + 1,
            "x": x % road_length,
            "y": y,
            "lane": lane,
            "speed": speed,
            "vclass": vclass,
            "cpu_capacity": cpu_capacity,
            "lane_change_cooldown": 0
        })
        
    num_steps = int(duration_seconds / dt)
    for step in range(num_steps):
        t = round(step * dt, 2)
        for v in vehicles:
            # IDM (Intelligent Driver Model) style acceleration/speed fluctuation
            speed_noise = np.random.normal(0, 0.25)
            target_speed = 22.0 + (v["lane"] - 1) * 2.0  # fast lanes on left
            accel = 0.5 * (target_speed - v["speed"]) + speed_noise
            v["speed"] = np.clip(v["speed"] + accel * dt, 10.0, 35.0)
            
            # Forward motion
            v["x"] = (v["x"] + v["speed"] * dt) % road_length
            
            # Lane changing logic
            v["lane_change_cooldown"] -= dt
            if v["lane_change_cooldown"] <= 0 and np.random.rand() < 0.03:
                new_lane = np.clip(v["lane"] + np.random.choice([-1, 1]), 1, num_lanes)
                if new_lane != v["lane"]:
                    v["lane"] = new_lane
                    v["y"] = (new_lane - 0.5) * lane_width
                    v["lane_change_cooldown"] = 8.0  # seconds between lane changes
                    
            records.append({
                "time_sec": t,
                "vehicle_id": v["vid"],
                "pos_x": round(v["x"], 2),
                "pos_y": round(v["y"], 2),
                "speed_mps": round(v["speed"], 2),
                "lane_id": v["lane"],
                "vehicle_class": v["vclass"],
                "cpu_capacity_ghz": round(v["cpu_capacity"], 2)
            })
            
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Generated NGSIM Highway trace: {filename} ({len(df)} records, {num_vehicles} vehicles, {duration_seconds}s)")
    return df

def generate_manhattan_trace(filename="data/synthetic_traces/synthetic_manhattan_grid.csv", num_vehicles=50, duration_seconds=300, dt=0.5):
    """
    Simulates Manhattan Urban Grid (4x4 blocks, 200m block spacing).
    Saves to data/synthetic_traces/.
    """
    np.random.seed(101)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    grid_size = 4
    block_len = 200.0  # meters
    records = []
    
    vehicles = []
    for vid in range(num_vehicles):
        axis = np.random.choice(["H", "V"])
        fixed_coord = np.random.choice(np.arange(grid_size)) * block_len
        prog = np.random.uniform(0, (grid_size - 1) * block_len)
        direction = np.random.choice([1, -1])
        speed = np.random.uniform(8.0, 16.0) # 30 - 60 km/h in city
        vclass = np.random.choice(["Car", "Bus", "Taxi", "Delivery"], p=[0.5, 0.1, 0.25, 0.15])
        vehicles.append({
            "vid": vid + 1,
            "axis": axis,
            "x": prog if axis == "H" else fixed_coord,
            "y": fixed_coord if axis == "H" else prog,
            "dir": direction,
            "speed": speed,
            "vclass": vclass,
            "cpu_capacity": np.random.uniform(2.5, 6.0)
        })
        
    num_steps = int(duration_seconds / dt)
    max_coord = (grid_size - 1) * block_len
    for step in range(num_steps):
        t = round(step * dt, 2)
        for v in vehicles:
            if v["axis"] == "H":
                v["x"] += v["dir"] * v["speed"] * dt
                if v["x"] < 0 or v["x"] > max_coord:
                    v["dir"] *= -1
                    v["x"] = np.clip(v["x"], 0, max_coord)
            else:
                v["y"] += v["dir"] * v["speed"] * dt
                if v["y"] < 0 or v["y"] > max_coord:
                    v["dir"] *= -1
                    v["y"] = np.clip(v["y"], 0, max_coord)
                    
            records.append({
                "time_sec": t,
                "vehicle_id": v["vid"],
                "pos_x": round(v["x"], 2),
                "pos_y": round(v["y"], 2),
                "speed_mps": round(v["speed"], 2),
                "vehicle_class": v["vclass"],
                "cpu_capacity_ghz": round(v["cpu_capacity"], 2)
            })
            
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Generated Manhattan Urban trace: {filename} ({len(df)} records)")
    return df

def generate_3gpp_task_profiles(filename="data/synthetic_traces/synthetic_v2x_task_profiles.csv"):
    """
    3GPP Release 16/17 V2X Service task profiles:
    - Cooperative Perception (Object detection from camera/LiDAR)
    - Sensor Fusion & Localization
    - Video Analytics / Blind Spot HD Stream
    - Platooning Coordination / Trajectory Planning
    """
    tasks = [
        {
            "task_type": "Cooperative_Perception",
            "data_size_mb_mean": 0.45,
            "data_size_mb_std": 0.08,
            "cpu_cycles_giga_mean": 0.22,
            "cpu_cycles_giga_std": 0.04,
            "deadline_ms": 160.0,
            "priority": 3,
            "energy_weight": 0.8
        },
        {
            "task_type": "Sensor_Fusion",
            "data_size_mb_mean": 0.18,
            "data_size_mb_std": 0.03,
            "cpu_cycles_giga_mean": 0.12,
            "cpu_cycles_giga_std": 0.02,
            "deadline_ms": 90.0,
            "priority": 4,
            "energy_weight": 0.6
        },
        {
            "task_type": "Video_Analytics",
            "data_size_mb_mean": 0.95,
            "data_size_mb_std": 0.15,
            "cpu_cycles_giga_mean": 0.40,
            "cpu_cycles_giga_std": 0.06,
            "deadline_ms": 280.0,
            "priority": 1,
            "energy_weight": 1.2
        },
        {
            "task_type": "Trajectory_Planning",
            "data_size_mb_mean": 0.08,
            "data_size_mb_std": 0.02,
            "cpu_cycles_giga_mean": 0.06,
            "cpu_cycles_giga_std": 0.01,
            "deadline_ms": 65.0,
            "priority": 5,
            "energy_weight": 0.4
        }
    ]
    df = pd.DataFrame(tasks)
    df.to_csv(filename, index=False)
    print(f"Generated 3GPP V2X task profile catalog: {filename}")
    return df

if __name__ == "__main__":
    generate_ngsim_highway_trace()
    generate_manhattan_trace()
    generate_3gpp_task_profiles()
