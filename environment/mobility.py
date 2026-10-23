"""
VANET Highway Mobility Engine with IDM Collision-Free Dynamics.
Simulates realistic 2400m highway traffic with Intelligent Driver Model (IDM).
Guarantees:
- Zero vehicle overlapping (car-following headway buffer >= 18m).
- Smooth continuous cruising speeds (70 - 100 km/h).
- Periodic boundary looping along the 2400m highway.
"""

import numpy as np

class HighwayMobility:
    def __init__(self, highway_length=2400.0, num_lanes=5, dt=0.5):
        self.highway_length = float(highway_length)
        self.num_lanes = int(num_lanes)
        self.dt = float(dt)
        self.lane_width = 3.75

    def initialize_spaced_positions(self, vehicles_dict):
        """Uniformly distributes vehicles along the 2400m highway across lanes with safe spacing."""
        vehicles = list(vehicles_dict.values())
        num_v = len(vehicles)
        spacing = self.highway_length / max(1, num_v)

        for i, v in enumerate(vehicles):
            v.lane = (i % self.num_lanes) + 1  # 1 to 5
            v.y = float((v.lane - 3.0) * self.lane_width)
            v.x = float((i * spacing + np.random.uniform(5.0, 20.0)) % self.highway_length)
            v.desired_speed = float(np.random.uniform(20.0, 27.0)) # ~72-97 km/h
            v.speed = v.desired_speed
            v.axis = "H"
            v.dir = 1
            v.heading = 1.5708

    def step(self, vehicles_dict):
        """Advances vehicle positions along the highway using IDM car-following dynamics."""
        vehicles = list(vehicles_dict.values())
        if not vehicles:
            return

        for v in vehicles:
            # 1. Locate lead vehicle in the same lane
            min_dist = 9999.0
            lead_speed = v.desired_speed

            for other in vehicles:
                if other.vehicle_id == v.vehicle_id or other.lane != v.lane:
                    continue
                dist = other.x - v.x
                if dist < 0:
                    dist += self.highway_length
                if 0.0 < dist < min_dist:
                    min_dist = dist
                    lead_speed = other.speed

            # 2. Intelligent Driver Model (IDM) acceleration
            s0 = 16.0     # Minimum bumper-to-bumper distance (meters)
            T = 1.4       # Safe time headway (seconds)
            a_max = 2.4   # Maximum acceleration (m/s^2)
            b_comf = 3.2  # Comfortable deceleration (m/s^2)

            delta_v = v.speed - lead_speed
            s_star = s0 + max(0.0, v.speed * T + (v.speed * delta_v) / (2.0 * np.sqrt(a_max * b_comf)))
            
            free_road_ratio = (v.speed / max(0.1, v.desired_speed)) ** 4
            interaction_ratio = (s_star / max(1.0, min_dist)) ** 2
            accel = a_max * (1.0 - free_road_ratio - interaction_ratio)

            v.speed = float(np.clip(v.speed + accel * self.dt, 14.0, 32.0))

            # 3. Update position along highway
            v.x = float((v.x + v.speed * self.dt) % self.highway_length)
            v.y = float((v.lane - 3.0) * self.lane_width)
            v.axis = "H"
            v.dir = 1
            v.heading = 1.5708

import os
import pandas as pd
import numpy as np

class TraceDrivenMobility:
    """
    Genuine trace-replay mobility engine that parses vehicle trajectories from CSV.
    Supports both:
    1. Calibrated synthetic NGSIM format: time_sec, vehicle_id, pos_x, pos_y, speed_mps, lane_id
    2. Authentic FHWA NGSIM format: Global_Time, Vehicle_ID, Local_X, Local_Y, v_Vel, Lane_ID (with feet-to-meters conversion)
    
    Preserves stable mapping from source vehicle IDs to simulated vehicles to prevent
    trajectory swapping when vehicles enter/leave the observation window.
    """
    # Parsed traces are cached per (path, mtime) so that repeatedly building an
    # environment (for example when the mobility mode is changed in the
    # console) does not re-parse a multi-megabyte CSV every time.
    _TRACE_CACHE = {}

    #: What happens when the replay passes the end of the recorded trace.
    #:
    #: ``"stop"``    - the replay ends; every vehicle becomes absent and the
    #:                 simulation continues with an empty road.  This is the
    #:                 default, because silently wrapping around would replay
    #:                 the recording out of order and invent trajectories.
    #: ``"empty"``   - identical to ``"stop"`` but spelled out for clarity.
    #: ``"loop"``    - the recording is replayed again from the start.  This
    #:                 is *opt-in* and is only meaningful for a trace that
    #:                 represents a repeating traffic pattern.
    TRACE_END_POLICIES = ("stop", "empty", "loop")

    def __init__(self, trace_file="data/synthetic_traces/synthetic_ngsim_highway.csv", dt=0.5, road_length=2400.0,
                 trace_end_policy="stop"):
        self.trace_file = trace_file
        self.dt = float(dt)
        self.road_length = float(road_length)
        self.highway_length = self.road_length
        self.current_time_sec = 0.0
        self.trace_data = {}
        self.time_keys = []
        policy = str(trace_end_policy).lower()
        if policy not in self.TRACE_END_POLICIES:
            raise ValueError(
                "trace_end_policy must be one of %s, got %r"
                % (self.TRACE_END_POLICIES, trace_end_policy))
        self.trace_end_policy = policy
        self.trace_exhausted = False
        self._load_trace()

    def _load_trace(self):
        if not os.path.exists(self.trace_file):
            alt_path = self.trace_file.replace("data/synthetic_traces/", "data/traces/")
            if os.path.exists(alt_path):
                self.trace_file = alt_path
            else:
                raise FileNotFoundError(
                    f"Trajectory trace file not found: {self.trace_file}. "
                    "Ensure authentic trace is downloaded to data/source/ or synthetic calibration traces exist in data/synthetic_traces/."
                )

        # --- Parse (with a per-file cache) ------------------------------
        try:
            mtime = os.path.getmtime(self.trace_file)
        except OSError:
            mtime = 0.0
        cache_key = (os.path.abspath(self.trace_file), mtime, round(self.dt, 6))

        cached = self._TRACE_CACHE.get(cache_key)
        if cached is None:
            cached = self._parse_trace()
            self._TRACE_CACHE[cache_key] = cached

        self.df = cached["df"]
        self.trace_data = cached["trace_data"]
        self.time_keys = cached["time_keys"]
        self.max_time = cached["max_time"]
        self.trace_x_min = cached["x_min"]
        self.trace_x_max = cached["x_max"]
        self.trace_road_length = cached["road_length"]
        self.trace_lane_count = cached["lane_count"]
        self.trace_vehicle_count = cached["vehicle_count"]
        self.trace_duration_sec = cached["duration"]

    def _parse_trace(self):
        """Read the CSV, convert the schema and index it by timestamp."""
        header = pd.read_csv(self.trace_file, nrows=0).columns
        wanted = ["time_sec", "vehicle_id", "pos_x", "pos_y", "speed_mps", "lane_id",
                  "Global_Time", "Vehicle_ID", "Local_X", "Local_Y", "v_Vel", "Lane_ID"]
        # Only the columns the replay needs are loaded (the authentic export has
        # 4 million rows; an unused text column would cost hundreds of MB).
        df = pd.read_csv(self.trace_file, usecols=[c for c in header if c in wanted])
        if df.empty:
            raise ValueError(f"Trace file is empty: {self.trace_file}")

        # Schema detection and conversion
        if "Local_X" in df.columns and "Local_Y" in df.columns:
            # Authentic FHWA NGSIM schema:
            # - Local_Y: Longitudinal position along highway (feet -> meters)
            # - Local_X: Lateral position across lanes (feet -> meters)
            # - v_Vel: Longitudinal speed (feet/sec -> meters/sec)
            # - Global_Time: Epoch timestamp in milliseconds
            required_fhwa = ["Global_Time", "Vehicle_ID", "Local_X", "Local_Y", "v_Vel", "Lane_ID"]
            missing = [c for c in required_fhwa if c not in df.columns]
            if missing:
                raise ValueError(f"Authentic FHWA NGSIM trace missing required columns: {missing}")

            t_col = "Global_Time"
            v_col = "Vehicle_ID"
            t_min = df[t_col].min()
            df["time_sec"] = ((df[t_col] - t_min) / 1000.0).round(1)
            df["pos_x"] = df["Local_Y"] * 0.3048  # Longitudinal along highway
            df["pos_y"] = df["Local_X"] * 0.3048  # Lateral across highway lanes
            df["speed_mps"] = df["v_Vel"] * 0.3048
            df["lane_id"] = df["Lane_ID"].astype(int)
            df["vehicle_id"] = df[v_col].astype(int)
        elif "pos_x" in df.columns and "pos_y" in df.columns:
            # Standard metric trace schema
            required_metric = ["time_sec", "vehicle_id", "pos_x", "pos_y", "speed_mps", "lane_id"]
            missing = [c for c in required_metric if c not in df.columns]
            if missing:
                raise ValueError(f"Metric trace missing required columns: {missing}")
        else:
            raise ValueError(
                f"Unknown trajectory schema in {self.trace_file}. "
                "Expected either FHWA NGSIM columns [Global_Time, Vehicle_ID, Local_X, Local_Y, v_Vel, Lane_ID] "
                "or Metric columns [time_sec, vehicle_id, pos_x, pos_y, speed_mps, lane_id]."
            )

        # Geometry is measured on the FULL trace ...
        pos_x = df["pos_x"]
        x_min = float(pos_x.min())
        x_max = float(pos_x.max())
        lane_count = int(df["lane_id"].nunique())
        vehicle_count = int(df["vehicle_id"].nunique())
        max_time = float(df["time_sec"].max())

        # ... but the replay only ever samples the trace at multiples of the
        # simulation step.  When that step is a multiple of the 0.1 s recording
        # interval, the frames in between are never read, so they are dropped.
        # This changes no replayed position (the nearest-frame lookup lands
        # exactly on a kept frame) and cuts memory by the decimation factor.
        step10 = self.dt * 10.0
        if abs(step10 - round(step10)) < 1e-9 and round(step10) > 1:
            keep = np.isclose((df["time_sec"] / self.dt) % 1.0, 0.0, atol=1e-6) |                    np.isclose((df["time_sec"] / self.dt) % 1.0, 1.0, atol=1e-6)
            df = df[keep].reset_index(drop=True)

        trace_data = {}
        time_keys = []
        self._index_trace(df, trace_data, time_keys)

        time_keys = sorted(list(set(time_keys)))

        return {
            "df": df,
            "trace_data": trace_data,
            "time_keys": time_keys,
            "max_time": max_time,
            "x_min": x_min,
            "x_max": x_max,
            "road_length": max(1.0, x_max - x_min),
            "lane_count": lane_count,
            "vehicle_count": vehicle_count,
            "duration": float(max_time),
        }

    @staticmethod
    @staticmethod
    def _index_trace(df, trace_data, time_keys):
        """Index the trace by timestamp using plain Python.

        Much faster than a pandas ``groupby`` + ``to_dict`` over tens of
        thousands of small frames, and much lighter on memory: a frame entry
        is a 4-tuple rather than a dictionary, which matters when the trace
        holds several million records.

        The per-frame payload is the tuple
        ``(pos_x, pos_y, speed_mps, lane_id)``.
        """
        df = df.sort_values(["time_sec", "vehicle_id"], kind="mergesort")
        times = df["time_sec"].to_numpy(dtype=float)
        vids = df["vehicle_id"].to_numpy()
        pos_x = df["pos_x"].to_numpy(dtype=float)
        pos_y = df["pos_y"].to_numpy(dtype=float)
        speed = df["speed_mps"].to_numpy(dtype=float)
        lane = df["lane_id"].to_numpy()

        n = len(times)
        if n == 0:
            return

        # Locate the frame boundaries (a new frame starts where the rounded
        # timestamp changes).
        boundaries = np.flatnonzero(np.diff(times) != 0.0) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n]))

        for start, end in zip(starts, ends):
            t_key = round(float(times[start]), 1)
            frame = {}
            for i in range(start, end):
                frame[int(vids[i])] = (float(pos_x[i]), float(pos_y[i]),
                                       float(speed[i]), int(lane[i]))
            trace_data[t_key] = frame
            time_keys.append(t_key)

        
    @property
    def geometry(self) -> dict:
        """Observed geometry of the replayed trace."""
        return {
            "road_length_m": round(self.trace_road_length, 1),
            "lane_count": self.trace_lane_count,
            "vehicle_count": self.trace_vehicle_count,
            "duration_sec": round(self.trace_duration_sec, 1),
            "x_min_m": round(self.trace_x_min, 2),
            "x_max_m": round(self.trace_x_max, 2),
            "trace_end_policy": self.trace_end_policy,
        }
        
    def rsu_positions_for_segment(self, num_rsus: int) -> list:
        """Evenly spaced RSU positions inside the observed road segment.
        
        The authentic US-101 study segment is roughly 640-680 m long, so the
        default 2400 m multi-kilometre deployment would leave most RSUs
        outside the observed data.  Placing the RSUs inside the observed
        extent keeps the deployment consistent with the replayed trajectories.
        """
        if num_rsus <= 0:
            return []
        lo, hi = self.trace_x_min, self.trace_x_max
        span = max(1.0, hi - lo)
        # Place the RSUs at 20%, 50% and 80% of the observed segment.
        fractions = [0.2, 0.5, 0.8] if num_rsus == 3 else [
            (i + 1) / (num_rsus + 1) for i in range(num_rsus)
        ]
        return [round(lo + span * f, 1) for f in fractions]

    def initialize_spaced_positions(self, vehicles_dict):
        """Bind the simulated fleet to the source vehicles present in the trace.

        The trace is a *dynamic* fleet: vehicles enter and leave the
        observation window, and the first recorded frame usually contains only
        one or two vehicles.  The replay therefore maintains a pool of slots
        that are bound to source vehicles for as long as those source vehicles
        stay in the window, and are released (and re-bound to a new arrival)
        when they leave.
        """
        self.current_time_sec = 0.0
        self._slot_to_source = {}       # sim_vid -> source_vid (active bindings)
        self._source_to_slot = {}       # source_vid -> sim_vid
        self._unbound_slots = set(vehicles_dict.keys())
        self._seen_sources = set()
        self._active_sim_vids = set()
        self.update_vehicles(vehicles_dict, t_sec=0.0)

    def _match_frame(self, t_sec):
        """Return the trace frame closest to ``t_sec``.

        The end-of-trace behaviour is governed by ``trace_end_policy``.  With
        the default ``"stop"`` policy the replay does **not** wrap around:
        once the recorded data is exhausted the frame is empty and every
        vehicle is reported as absent.
        """
        duration = max(0.1, self.max_time)
        if t_sec > duration:
            if self.trace_end_policy == "loop":
                t_key = round(t_sec % duration, 1)
            else:
                self.trace_exhausted = True
                return None, {}
        else:
            self.trace_exhausted = False
            t_key = round(t_sec, 1)
        idx = int(np.argmin([abs(k - t_key) for k in self.time_keys]))
        return self.time_keys[idx], self.trace_data.get(self.time_keys[idx], {})

    def _rebind(self, frame_records, vehicles_dict):
        """Reconcile the slot bindings with the vehicles present in the frame.

        * A source vehicle that is still present keeps its slot (the identity
          of the source vehicle is preserved for the whole of its stay).
        * A source vehicle that has left releases its slot.
        * A newly arrived source vehicle takes a free slot.
        """
        present = set(frame_records.keys())

        # 1. Release the slots whose source vehicle has departed.
        for src_vid, sim_vid in list(self._source_to_slot.items()):
            if src_vid not in present:
                del self._source_to_slot[src_vid]
                self._slot_to_source.pop(sim_vid, None)
                self._unbound_slots.add(sim_vid)
                self._active_sim_vids.discard(sim_vid)

        # 2. Bind the newly arrived source vehicles to the free slots.
        arrivals = sorted(present - self._seen_sources)
        for src_vid in arrivals:
            if not self._unbound_slots:
                break
            sim_vid = min(self._unbound_slots)
            self._unbound_slots.discard(sim_vid)
            self._slot_to_source[sim_vid] = src_vid
            self._source_to_slot[src_vid] = sim_vid
            self._active_sim_vids.add(sim_vid)

        self._seen_sources |= present

    def update_vehicles(self, vehicles_dict, t_sec=None):
        if t_sec is None:
            t_sec = self.current_time_sec
        matched_time, frame_records = self._match_frame(t_sec)

        # Reconcile arrivals / departures with the current frame.  When the
        # recording is exhausted the frame is empty, which releases every
        # binding: no vehicle is on the road any more.
        self._rebind(frame_records, vehicles_dict)

        for sim_vid, v in vehicles_dict.items():
            src_vid = self._slot_to_source.get(sim_vid)
            if src_vid is not None and src_vid in frame_records:
                rec = frame_records[src_vid]
                # The frame payload is the compact tuple
                # (pos_x, pos_y, speed_mps, lane_id).
                v.x = float(rec[0]) % self.road_length
                v.y = float(rec[1])
                v.speed = float(rec[2])
                v.lane = int(rec[3])
                v.axis = "H"
                v.dir = 1
                v.heading = 1.5708
                v.active = True
                v.source_vehicle_id = src_vid
            else:
                # The slot has no source vehicle at this instant (the vehicle
                # has not arrived yet, or it has left the observation window).
                # It must NOT be extrapolated along the road, because that
                # would invent a trajectory that the source data does not
                # contain.  The slot is parked and marked inactive instead.
                v.active = False
                v.source_vehicle_id = None
                v.speed = 0.0
                v.x = float(self.trace_x_min)
                v.y = float(-999.0)   # parked off the carriageway
                v.lane = 1
                v.axis = "H"
                v.dir = 1
                v.heading = 1.5708

    @property
    def active_source_bindings(self) -> dict:
        """The current source-vehicle -> simulated-vehicle bindings."""
        return dict(self._source_to_slot)

    @property
    def active_slot_bindings(self) -> dict:
        """The current simulated-vehicle -> source-vehicle bindings."""
        return dict(self._slot_to_source)

    @property
    def active_vehicle_count(self) -> int:
        """Number of simulated vehicles currently following a source trajectory."""
        return len(self._active_sim_vids)

    def step(self, vehicles_dict):
        self.current_time_sec += self.dt
        self.update_vehicles(vehicles_dict, t_sec=self.current_time_sec)

# Backwards compatibility aliases
MobilityManager = HighwayMobility
UrbanCityMobility = HighwayMobility
