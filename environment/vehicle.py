"""
Vehicular node model in a VANET.

Represents mobile computing vehicles capable of:
1. Generating local tasks.
2. Executing tasks locally.
3. Cooperatively assisting peers (V2V helper node).
4. Participating in Federated Reinforcement Learning.
5. Acting under adversarial roles (honest, poisoning, free-rider, unreliable).

Queue model
-----------
A vehicle is modelled as a **single deterministic server** with a persistent
queue: a task that arrives while the on-board processor is busy waits until
the processor becomes free.  The service time is deterministic (``N / f``),
so this is *not* an M/M/1 queue and no stochastic service-time assumption is
made.  The same simplification applies to the V2V helper role.

As with the RSU, a task that the model reports as completed is one whose
service time elapsed inside the simulation; it is not evidence that a
physical on-board computer executed it.
"""

import numpy as np


class Vehicle:
    def __init__(self, vehicle_id, x, y, speed=20.0, lane=1, vclass="Car", cpu_capacity_ghz=3.0, role="honest"):
        self.vehicle_id = int(vehicle_id)
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)
        self.lane = int(lane)
        self.vclass = vclass
        self.cpu_capacity_ghz = float(cpu_capacity_ghz)

        # Operational parameters
        self.max_queue = 15
        self.queue = []
        self.battery_level = float(np.random.uniform(0.7, 1.0))
        self.v2v_range_m = 180.0

        # When the on-board processor becomes free.
        self.processor_free_time = 0.0

        # --- Counters -------------------------------------------------
        self.total_processed_tasks = 0
        self.total_arrived = 0
        self.dropped_tasks = 0
        self.total_wait_time = 0.0
        self.total_service_time = 0.0

        # Security role: "honest", "poisoning", "free_rider", "unreliable"
        self.role = role
        self.poison_type = "sign_flip"  # or "gaussian_noise", "scaling"
        self.poison_strength = 0.5

        # Performance counters
        self.tasks_generated = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_energy_consumed = 0.0
        self.total_latency_ms = 0.0

        # Historical helper behavior tracking
        self.helper_tasks_accepted = 0
        self.helper_tasks_completed = 0
        self.helper_tasks_dropped = 0

        # Trace-replay bookkeeping.  ``active`` is False when the vehicle is
        # not currently following a recorded source trajectory (it has not
        # arrived in the observation window yet, or it has left it).
        self.active = True
        self.source_vehicle_id = None

    def get_distance_to(self, other_x, other_y):
        return np.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

    # ------------------------------------------------------------------
    # Local execution
    # ------------------------------------------------------------------
    def execute_local(self, task, channel, current_time=0.0):
        """Executes a task on the on-board processor.

        Returns (accepted, sojourn_time_sec, energy_j).  The sojourn time
        includes the queueing wait; the caller compares it with the deadline.
        """
        if len(self.queue) >= self.max_queue:
            task.completed = False
            task.failed = True
            task.execution_node = f"V{self.vehicle_id}"
            self.tasks_failed += 1
            self.dropped_tasks += 1
            self.total_arrived += 1
            return False, 0.0, 0.0

        self.total_arrived += 1
        service_time, energy_j = channel.compute_computation_delay(
            task.cpu_cycles_giga, self.cpu_capacity_ghz)
        start_time = max(current_time, self.processor_free_time)
        wait_time = max(0.0, start_time - current_time)
        finish_time = start_time + service_time
        self.processor_free_time = finish_time

        self.queue.append({
            'task': task,
            'arrival_time': current_time,
            'start_time': start_time,
            'finish_time': finish_time,
            'service_time': service_time,
            'wait_time': wait_time,
        })
        self.total_wait_time += wait_time
        self.total_service_time += service_time

        sojourn = wait_time + service_time
        task.total_delay_sec = sojourn
        task.energy_joules = energy_j
        task.execution_node = f"V{self.vehicle_id}"
        self.total_energy_consumed += energy_j
        self.total_latency_ms += sojourn * 1000.0
        return True, sojourn, energy_j

    # ------------------------------------------------------------------
    # V2V helper
    # ------------------------------------------------------------------
    def execute_as_v2v_helper(self, task, channel, current_time=0.0):
        """Serves a task offloaded by a peer vehicle.

        Returns (accepted, sojourn_time_sec, energy_j).
        """
        self.helper_tasks_accepted += 1

        if self.role == "unreliable" and np.random.rand() < 0.75:
            # Drop the task maliciously or due to an artificial failure.
            self.helper_tasks_dropped += 1
            self.dropped_tasks += 1
            self.total_arrived += 1
            task.completed = False
            task.failed = True
            task.execution_node = f"V{self.vehicle_id}"
            return False, 0.0, 0.0

        if len(self.queue) >= self.max_queue:
            self.helper_tasks_dropped += 1
            self.dropped_tasks += 1
            self.total_arrived += 1
            task.completed = False
            task.failed = True
            task.execution_node = f"V{self.vehicle_id}"
            return False, 0.0, 0.0

        self.total_arrived += 1
        service_time, comp_energy = channel.compute_computation_delay(
            task.cpu_cycles_giga, self.cpu_capacity_ghz)

        if self.role == "unreliable":
            # Inject malicious processing delay.
            service_time *= np.random.uniform(2.5, 5.0)

        start_time = max(current_time, self.processor_free_time)
        wait_time = max(0.0, start_time - current_time)
        finish_time = start_time + service_time
        self.processor_free_time = finish_time

        self.queue.append({
            'task': task,
            'arrival_time': current_time,
            'start_time': start_time,
            'finish_time': finish_time,
            'service_time': service_time,
            'wait_time': wait_time,
        })
        self.total_wait_time += wait_time
        self.total_service_time += service_time
        self.helper_tasks_completed += 1
        task.execution_node = f"V{self.vehicle_id}"
        return True, wait_time + service_time, comp_energy

    def step(self, current_time):
        """Retires tasks whose service time has elapsed."""
        still_running = []
        for entry in self.queue:
            if entry['finish_time'] > current_time:
                still_running.append(entry)
            else:
                self.total_processed_tasks += 1
        self.queue = still_running

    def get_queue_length(self):
        return len(self.queue)

    @property
    def queue_length(self):
        """Number of tasks waiting or in service (backwards-compatible alias)."""
        return len(self.queue)

    def get_avg_wait_time(self):
        admitted = self.total_arrived - self.dropped_tasks
        if admitted <= 0:
            return 0.0
        return self.total_wait_time / admitted

    def to_dict(self):
        axis = getattr(self, "axis", "H")
        dir_val = getattr(self, "dir", 1)
        if axis == "H":
            heading = 1.5708 if dir_val > 0 else -1.5708
        else:
            heading = 0.0 if dir_val > 0 else 3.14159
        return {
            "vehicle_id": self.vehicle_id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "speed": round(self.speed, 1),
            "lane": self.lane,
            "axis": axis,
            "dir": dir_val,
            "heading": heading,
            "vclass": self.vclass,
            "cpu_capacity_ghz": round(self.cpu_capacity_ghz, 2),
            "queue_length": self.get_queue_length(),
            "battery_level": round(self.battery_level, 2),
            "trust_score": round(getattr(self, "trust_score", 0.75), 2),
            "role": self.role,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "v2v_completed": self.helper_tasks_completed,
            "v2v_dropped": self.helper_tasks_dropped,
            "queue_model": "deterministic single server",
            "active": bool(getattr(self, "active", True)),
            "source_vehicle_id": getattr(self, "source_vehicle_id", None)
        }
