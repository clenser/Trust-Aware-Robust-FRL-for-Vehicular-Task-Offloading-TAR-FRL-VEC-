"""
Road-Side Unit (RSU) edge server model.

Scheduling model
----------------
This is a **deterministic multi-core server approximation**, not an M/M/c
queue.  The distinction matters and is stated explicitly:

* The RSU has ``num_cores`` identical cores.  A task is assigned to the core
  that becomes free earliest; it waits until that core is free and is then
  served for a *deterministic* service time ``N / f_core``.
* The service time therefore has **no exponential (memoryless) distribution**,
  and the waiting time is a deterministic function of the arrival pattern.
  An M/M/c queue would additionally require a Poisson arrival process and
  exponential service times, and would produce a different (stochastic) queue
  length distribution.  Those properties do **not** hold here.
* The RSU is a *simulated* edge resource.  A task that this model reports as
  completed is a task whose service time elapsed inside the simulation; it is
  **not** evidence that a physical edge processor executed it.

What the model does provide, and what the statistics report:

* a real waiting time (a task that arrives while every core is busy waits),
* a real service time, and therefore a real sojourn time,
* real capacity limits: the queue is bounded and overflow is counted as a
  drop, and
* correct counters for processed, dropped, deadline-violating and queued
  tasks, and for accumulated waiting and service time.
"""

import numpy as np


class RSU:
    def __init__(self, rsu_id, x, y=12.0, z_height=12.0, cpu_capacity_ghz=24.0, coverage_radius_m=350.0,
                 is_malicious=False, num_cores=8, is_rooftop=False, building_name=""):
        self.rsu_id = rsu_id
        self.x = float(x)
        self.y = float(y)
        self.z_height = float(z_height)
        self.cpu_capacity_ghz = float(cpu_capacity_ghz)
        self.coverage_radius_m = float(coverage_radius_m)
        self.is_malicious = is_malicious
        # Fault model of an UNRELIABLE edge node (problem statement: "edge nodes
        # ... may be unreliable"): with probability ``fault_drop_prob`` an
        # admitted task is silently dropped, and every service time is
        # multiplied by ``fault_delay_factor``.  Both default to a healthy node.
        self.fault_drop_prob = 0.0
        self.fault_delay_factor = 1.0
        self.num_cores = int(num_cores)  # Number of parallel processing cores
        self.is_rooftop = bool(is_rooftop)
        self.building_name = str(building_name)

        # Queue state: tasks that have arrived but have not finished service.
        self.queue = []
        self.max_queue = 50

        # --- Counters -------------------------------------------------
        self.total_processed_tasks = 0     # tasks whose service time elapsed
        self.dropped_tasks = 0             # rejected: queue overflow or malicious
        self.deadline_violations = 0       # processed but late
        self.total_arrived = 0
        self.current_utilization = 0.0

        # Server state: when each core becomes free.
        self.core_free_times = [0.0] * self.num_cores

        # Accumulated times (seconds).
        self.total_wait_time = 0.0
        self.total_service_time = 0.0
        self.last_wait_time = 0.0
        self.last_service_time = 0.0

    def get_distance_to(self, x, y, v_height=1.5):
        """Calculates 3D Euclidean distance (m) between vehicle and RSU/5G base station."""
        return float(np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2 + (self.z_height - v_height) ** 2))

    def is_in_range(self, x, y, v_height=1.5):
        return self.get_distance_to(x, y, v_height) <= self.coverage_radius_m

    def _get_available_core(self, current_time):
        """Returns (core_index, start_time) for the next available core."""
        earliest_free = min(self.core_free_times)
        core_idx = self.core_free_times.index(earliest_free)
        start_time = max(current_time, earliest_free)
        return core_idx, start_time

    def _compute_service_time(self, task, channel):
        """Deterministic service time of a task on a single core."""
        cpu_per_core = self.cpu_capacity_ghz / self.num_cores
        delay_sec, energy_j = channel.compute_computation_delay(task.cpu_cycles_giga, cpu_per_core)
        return delay_sec, energy_j

    def process_task(self, task, channel, current_time=0.0):
        """
        Schedules an offloaded task on the RSU.

        Returns:
            (accepted: bool, sojourn_time_sec: float, energy_j: float)

        ``sojourn_time_sec`` is the waiting time plus the service time.  The
        caller compares it with the task deadline to decide success.
        """
        if self.is_malicious and np.random.rand() < 0.8:
            # Unreliable/malicious edge node: drops or corrupts the task.
            task.completed = False
            task.failed = True
            task.execution_node = self.rsu_id
            self.dropped_tasks += 1
            self.total_arrived += 1
            return False, 0.0, 0.0

        if self.fault_drop_prob > 0.0 and np.random.rand() < self.fault_drop_prob:
            task.completed = False
            task.failed = True
            task.execution_node = self.rsu_id
            self.dropped_tasks += 1
            self.total_arrived += 1
            return False, 0.0, 0.0

        if len(self.queue) >= self.max_queue:
            # Queue overflow: the task is dropped, not queued.
            task.completed = False
            task.failed = True
            task.execution_node = self.rsu_id
            self.dropped_tasks += 1
            self.total_arrived += 1
            return False, 0.0, 0.0

        self.total_arrived += 1

        service_time, comp_energy = self._compute_service_time(task, channel)
        service_time *= self.fault_delay_factor

        core_idx, start_time = self._get_available_core(current_time)
        wait_time = max(0.0, start_time - current_time)
        finish_time = start_time + service_time

        self.core_free_times[core_idx] = finish_time

        self.queue.append({
            'task': task,
            'arrival_time': current_time,
            'start_time': start_time,
            'finish_time': finish_time,
            'core_idx': core_idx,
            'service_time': service_time,
            'wait_time': wait_time,
        })

        self.total_wait_time += wait_time
        self.total_service_time += service_time
        self.last_wait_time = wait_time
        self.last_service_time = service_time

        busy_cores = sum(1 for ft in self.core_free_times if ft > current_time)
        self.current_utilization = busy_cores / self.num_cores

        sojourn_time = wait_time + service_time
        task.execution_node = self.rsu_id
        return True, sojourn_time, comp_energy

    def step(self, current_time):
        """
        Advances the queue simulation by retiring completed tasks.

        Must be called at each simulation time step so that the counters and
        the utilisation stay consistent with the work that has actually been
        done.
        """
        still_running = []
        for entry in self.queue:
            if entry['finish_time'] > current_time:
                still_running.append(entry)
            else:
                # The task's service time has elapsed inside the simulation.
                self.total_processed_tasks += 1
        self.queue = still_running

        busy_cores = sum(1 for ft in self.core_free_times if ft > current_time)
        self.current_utilization = busy_cores / self.num_cores

    def note_deadline_violation(self):
        """Records that a processed task missed its deadline."""
        self.deadline_violations += 1

    def get_queue_length(self):
        """Tasks that have arrived but have not finished service."""
        return len(self.queue)

    @property
    def queue_length(self):
        """Number of queued tasks (backwards-compatible alias)."""
        return len(self.queue)

    def get_avg_wait_time(self):
        """Average waiting time over the tasks that have been admitted."""
        admitted = self.total_arrived - self.dropped_tasks
        if admitted <= 0:
            return 0.0
        return self.total_wait_time / admitted

    def get_avg_service_time(self):
        """Average service time over the tasks that have been admitted."""
        admitted = self.total_arrived - self.dropped_tasks
        if admitted <= 0:
            return 0.0
        return self.total_service_time / admitted

    def get_statistics(self):
        """Machine-readable queue statistics."""
        admitted = self.total_arrived - self.dropped_tasks
        return {
            "model": "deterministic multi-core server (NOT M/M/c)",
            "num_cores": self.num_cores,
            "cpu_per_core_ghz": self.cpu_capacity_ghz / max(1, self.num_cores),
            "arrived": self.total_arrived,
            "processed": self.total_processed_tasks,
            "dropped": self.dropped_tasks,
            "deadline_violations": self.deadline_violations,
            "in_queue": self.get_queue_length(),
            "max_queue": self.max_queue,
            "utilization": round(self.current_utilization, 4),
            "total_wait_time_s": round(self.total_wait_time, 6),
            "total_service_time_s": round(self.total_service_time, 6),
            "avg_wait_time_s": round(self.get_avg_wait_time(), 6),
            "avg_service_time_s": round(self.get_avg_service_time(), 6),
        }

    def to_dict(self):
        return {
            "rsu_id": self.rsu_id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "z_height": round(self.z_height, 1),
            "is_rooftop": self.is_rooftop,
            "building_name": self.building_name,
            "cpu_capacity_ghz": self.cpu_capacity_ghz,
            "num_cores": self.num_cores,
            "queue_length": self.get_queue_length(),
            "utilization": round(self.current_utilization, 2),
            "is_malicious": self.is_malicious,
            "coverage_radius": self.coverage_radius_m,
            "total_processed": self.total_processed_tasks,
            "dropped_tasks": self.dropped_tasks,
            "deadline_violations": self.deadline_violations,
            "avg_wait_time_s": round(self.get_avg_wait_time(), 4),
            "avg_service_time_s": round(self.get_avg_service_time(), 4),
            "queue_model": "deterministic multi-core server",
        }
