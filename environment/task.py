"""
Task Model & Realistic Non-IID Workload Generator.
Models 3GPP V2X autonomous driving computation tasks:
- Cooperative Perception
- Sensor Fusion
- Video Analytics
- Trajectory Planning
"""

import numpy as np
import scipy.stats as stats

class Task:
    def __init__(self, task_id, source_id, task_type, data_size_mb, cpu_cycles_giga, deadline_ms, priority=1, created_time=0.0):
        self.task_id = task_id
        self.source_id = source_id
        self.task_type = task_type
        self.data_size_mb = float(data_size_mb)
        self.cpu_cycles_giga = float(cpu_cycles_giga)
        self.deadline_ms = float(deadline_ms)
        self.deadline_sec = self.deadline_ms / 1000.0
        self.priority = int(priority)
        self.created_time = float(created_time)
        
        # Outcome tracking
        self.completed = False
        self.failed = False
        self.execution_node = None
        self.total_delay_sec = 0.0
        self.energy_joules = 0.0
        self.offloaded = False

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "task_type": self.task_type,
            "data_size_mb": round(self.data_size_mb, 2),
            "cpu_cycles_giga": round(self.cpu_cycles_giga, 2),
            "deadline_ms": self.deadline_ms,
            "completed": self.completed,
            "failed": self.failed,
            "total_delay_ms": round(self.total_delay_sec * 1000.0, 2),
            "energy_j": round(self.energy_joules, 4),
            "execution_node": self.execution_node
        }

import os
import pandas as pd
import numpy as np
import scipy.stats as stats

class TaskGenerator:
    def __init__(self, non_iid=True, random_seed=42, profile_csv="data/synthetic_traces/synthetic_v2x_task_profiles.csv", catalog_path=None,
                 dirichlet_alpha=0.4, fast_sampling=False):
        self.rng = np.random.RandomState(random_seed)
        # ``fast_sampling`` replaces scipy's truncated-normal draw (about 4.5 ms
        # per task) by rejection sampling from a plain normal (a few microseconds).
        # The distribution is the same, the random stream is not, so it is OFF for
        # every benchmark (workloads stay bit-identical and paired across runs)
        # and used only by the interactive console.
        self.fast_sampling = bool(fast_sampling)
        self._mix_cache = {}
        self.non_iid = non_iid
        # Concentration of the per-vehicle task-type distribution: smaller means
        # more heterogeneous clients (0.4 is the historical value).
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.task_counter = 0
        self.profile_csv = catalog_path if catalog_path is not None else profile_csv
        self.profiles = {}
        self._load_profiles()
        self.types = list(self.profiles.keys())

    def _load_profiles(self):
        """Loads authoritative 3GPP task profiles from CSV catalog."""
        csv_path = self.profile_csv
        if not os.path.exists(csv_path):
            alt_path = csv_path.replace("data/synthetic_traces/", "data/traces/")
            if os.path.exists(alt_path):
                csv_path = alt_path
            else:
                csv_path = None

        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.profiles[str(row["task_type"])] = {
                    "d_mean": float(row["data_size_mb_mean"]),
                    "d_std": float(row["data_size_mb_std"]),
                    "c_mean": float(row["cpu_cycles_giga_mean"]),
                    "c_std": float(row["cpu_cycles_giga_std"]),
                    "deadline": float(row["deadline_ms"]),
                    "prio": int(row["priority"]),
                    "energy_weight": float(row.get("energy_weight", 1.0))
                }
        else:
            # Authoritative calibrated 3GPP TR 36.885 profile specifications
            self.profiles = {
                "Cooperative_Perception": {"d_mean": 0.45, "d_std": 0.08, "c_mean": 0.22, "c_std": 0.04, "deadline": 160.0, "prio": 3, "energy_weight": 0.8},
                "Sensor_Fusion": {"d_mean": 0.18, "d_std": 0.03, "c_mean": 0.12, "c_std": 0.02, "deadline": 90.0, "prio": 4, "energy_weight": 0.6},
                "Video_Analytics": {"d_mean": 0.95, "d_std": 0.15, "c_mean": 0.40, "c_std": 0.06, "deadline": 280.0, "prio": 1, "energy_weight": 1.2},
                "Trajectory_Planning": {"d_mean": 0.08, "d_std": 0.02, "c_mean": 0.06, "c_std": 0.01, "deadline": 65.0, "prio": 5, "energy_weight": 0.4}
            }

    def get_client_task_distribution(self, vehicle_id):
        """
        Creates Non-IID Dirichlet distribution over task types per vehicle.
        Ensures vehicle 1 generates mostly Video Analytics, vehicle 2 mostly Sensor Fusion, etc.
        """
        if not self.non_iid:
            return [0.25, 0.25, 0.25, 0.25]
        # Deterministic pseudo-random Dirichlet per vehicle.  It is a pure function
        # of the vehicle id (and the fixed concentration), so it is computed once.
        cached = self._mix_cache.get(vehicle_id)
        if cached is None:
            v_rng = np.random.RandomState(int(vehicle_id) * 31 + 7)
            alpha = np.ones(len(self.types)) * self.dirichlet_alpha
            cached = v_rng.dirichlet(alpha)
            self._mix_cache[vehicle_id] = cached
        return cached

    def _sample_bounded_normal(self, mean, std, min_val, max_val=None):
        """
        Samples from a truncated normal distribution that preserves the intended mean/std
        within the specified bounds. Uses scipy.stats.truncnorm for proper sampling.
        """
        if max_val is None:
            max_val = mean + 4 * std  # Practical upper bound

        if self.fast_sampling:
            for _ in range(64):
                x = float(self.rng.normal(mean, std))
                if min_val <= x <= max_val:
                    return x
            return float(min(max(mean, min_val), max_val))
        
        # Convert to standard normal bounds
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        
        # Sample from truncated normal
        return float(stats.truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.rng))

    def generate_task(self, vehicle_id, current_time=0.0):
        self.task_counter += 1
        probs = self.get_client_task_distribution(vehicle_id)
        chosen_type = self.rng.choice(self.types, p=probs)
        p = self.profiles[chosen_type]
        
        # Use truncated normal to preserve distribution parameters within physical bounds
        # Data size: min 0.01 MB (10 KB), max 4x mean to prevent extreme outliers
        d_size = self._sample_bounded_normal(p["d_mean"], p["d_std"], 0.01, p["d_mean"] * 4)
        # CPU cycles: min 0.01 Gcycles, max 4x mean
        c_cycles = self._sample_bounded_normal(p["c_mean"], p["c_std"], 0.01, p["c_mean"] * 4)
        deadline = p["deadline"]
        
        return Task(
            task_id=f"T_{vehicle_id}_{self.task_counter}",
            source_id=vehicle_id,
            task_type=chosen_type,
            data_size_mb=d_size,
            cpu_cycles_giga=c_cycles,
            deadline_ms=deadline,
            priority=p["prio"],
            created_time=current_time
        )
