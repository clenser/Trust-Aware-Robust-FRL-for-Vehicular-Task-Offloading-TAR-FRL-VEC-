"""
VANET Task Offloading Gym-compatible Environment.
Integrates:
- Real-world NGSIM / Manhattan mobility or synthetic highway.
- 3GPP V2X wireless communication channel.
- Multiple RSU edge servers and mobile vehicle helpers.
- 3GPP autonomous driving workloads with Non-IID distributions.
- Behavioral and combined trust evaluation.
"""

import numpy as np
import copy
from environment.network import V2XChannel
from environment.task import TaskGenerator
from environment.rsu import RSU
from environment.edge_calibration import EdgeCalibration
from environment.vehicle import Vehicle
from environment.mobility import HighwayMobility, TraceDrivenMobility
from trust.trust_engine import TrustEngine
from rl.reward import RewardCalculator

#: 14 original features (task, kinematics, RSU distances, local queue, trust)
#: plus the normalised load of each of the three RSUs.
STATE_DIM = 17
#: Version 2.1 replaces the three road-length-normalised RSU distances by
#: distances relative to each RSU's coverage radius and adds three in-coverage
#: flags, so the coverage boundary is an explicit feature.
STATE_DIM_V21 = 20


class VANETOffloadingEnv:
    def __init__(self, num_vehicles=None, attack_ratio=None, mobility_mode=None, trace_file=None, trust_engine=None, alpha=None, exclusion_tau=None, config=None, random_seed=None, rsu_x_positions=None):
        if config is None:
            try:
                from configs.config_loader import load_config
                config = load_config()
            except Exception:
                config = None

        cfg_sys = config.get("system", {}) if config else {}
        cfg_mob = config.get("mobility", {}) if config else {}
        cfg_adv = config.get("adversarial", {}) if config else {}
        cfg_edge = config.get("edge_nodes", {}) if config else {}
        cfg_net = config.get("network", {}) if config else {}
        cfg_task = config.get("task_profiles", {}) if config else {}
        cfg_trust = config.get("trust_management", {}) if config else {}
        cfg_model = config.get("environment_model", {}) if config else {}
        cfg_reward = config.get("reward", {}) if config else {}

        self.config = config
        # Model-version switches.  ``honor_rsu_choice`` and ``uplink_contention``
        # are the version-2 behaviour; setting them to False reproduces the
        # version-1 simulator (silent RSU re-association, uncontended uplink)
        # so that before/after comparisons stay possible.
        self.honor_rsu_choice = bool(cfg_model.get("honor_rsu_choice", True))
        self.uplink_contention = bool(cfg_model.get("uplink_contention", True))
        self.cell_bandwidth_mhz = float(config.get("network", {}).get("cell_bandwidth_mhz", 100.0)) if config else 100.0
        self.unreliable_ratio = float(cfg_adv.get("unreliable_helper_ratio", 0.0))
        # Ablation switch: hide every trust feature from the policy (the three
        # RSU trust values and the best-peer trust), so the value of
        # behavioural trust for the offloading decision can be measured.
        self.mask_trust_features = bool(cfg_model.get("mask_trust_features", False))
        self.faulty_rsus_cfg = list(cfg_adv.get("faulty_rsus", []) or [])
        # Data-plane behaviour of the model-level attackers.  A poisoning
        # vehicle may also sabotage the V2V tasks it is asked to serve, and a
        # free rider is selfish: it refuses most helper work.  Both default to
        # off, so the published benchmark (where the attackers behave honestly
        # as helpers) is reproduced exactly.
        dp = cfg_adv.get("data_plane") or {}
        self.data_plane_enabled = bool(dp.get("enabled", False))
        self.dp_malicious_drop = float(dp.get("malicious_drop_prob", 0.7))
        self.dp_malicious_delay = float(dp.get("malicious_delay_factor", 3.0))
        self.dp_selfish_drop = float(dp.get("selfish_drop_prob", 0.8))
        # V2V peer choice: "nearest" (original) or "trust_aware" (scores every
        # in-range, present peer by observed trust, proximity and spare queue).
        # Optional compute-rich helper vehicles (e.g. trucks carrying an on-board
        # server).  An ASSUMPTION, off by default: with the default 0.8-2 GHz
        # automotive SoCs a peer is never faster than the vehicle itself, so V2V
        # offloading cannot pay off whatever the policy.
        vc = (config or {}).get("mobility", {}).get("vehicle_compute") or {}
        self.rich_fraction = float(vc.get("rich_fraction", 0.0))
        self.rich_cpu_range = tuple(vc.get("rich_cpu_ghz", (4.0, 8.0)))
        self.v2v_peer_selection = str((cfg_net.get("v2v_peer_selection") or "nearest"))
        if self.v2v_peer_selection not in ("nearest", "trust_aware"):
            raise ValueError("network.v2v_peer_selection must be 'nearest' or 'trust_aware'")
        self.num_vehicles = int(num_vehicles if num_vehicles is not None else cfg_mob.get("num_vehicles", 20))
        self.attack_ratio = float(attack_ratio if attack_ratio is not None else cfg_adv.get("attack_ratio", 0.2))
        self.mobility_mode = str(mobility_mode if mobility_mode is not None else cfg_mob.get("scenario", "highway"))
        self.trace_file = str(trace_file if trace_file is not None else cfg_mob.get("trace_dataset", "data/synthetic_traces/synthetic_ngsim_highway.csv"))
        
        # Use provided seed or config seed
        self.random_seed = random_seed if random_seed is not None else cfg_sys.get("random_seed", 42)
        
        self.dt = float(cfg_sys.get("time_step_dt", 0.5))
        self.road_length = float(cfg_mob.get("road_length_m", 3200.0))
        self.num_lanes = int(cfg_mob.get("num_lanes", 5))
        self.lane_width = float(cfg_mob.get("lane_width_m", 3.75))
        
        self.channel = V2XChannel(
            carrier_freq_ghz=float(cfg_net.get("carrier_frequency_ghz", 5.9)),
            bandwidth_mhz=float(cfg_net.get("channel_bandwidth_mhz", 20.0)),
            tx_power_dbm=float(cfg_net.get("tx_power_dbm", 23.0)),
            noise_power_dbm=float(cfg_net.get("noise_power_dbm", -95.0))
        )
        catalog_csv = cfg_task.get("catalog_csv", "data/synthetic_traces/synthetic_v2x_task_profiles.csv")
        self.task_generator = TaskGenerator(
            non_iid=bool(cfg_task.get("non_iid", True)), random_seed=self.random_seed,
            catalog_path=catalog_csv, dirichlet_alpha=float(cfg_task.get("dirichlet_alpha", 0.4)),
            fast_sampling=bool(cfg_task.get("fast_sampling", False)))
        
        # Select genuine trace replayer vs IDM continuous mobility
        if self.mobility_mode in ("trace", "authentic_trace"):
            trace_path = self.trace_file
            if self.mobility_mode == "authentic_trace":
                trace_path = cfg_mob.get(
                    "authentic_trace_dataset",
                    "data/traces/ngsim_us101_authentic.csv",
                )
            self.trace_file = trace_path
            # The road length is provisional: it is corrected below from the
            # geometry actually observed in the trace.
            self.mobility = TraceDrivenMobility(
                trace_file=trace_path, dt=self.dt, road_length=self.road_length
            )
            # Match the deployment to the observed road segment so that no RSU
            # sits outside the replayed trajectories.
            geom = self.mobility.geometry
            self.road_length = max(1.0, geom["x_max_m"] + 1.0)
            self.num_lanes = max(1, geom["lane_count"])
            self.mobility.road_length = self.road_length
            self.mobility.highway_length = self.road_length
            self.trace_geometry = geom
        else:
            self.mobility = HighwayMobility(highway_length=self.road_length, num_lanes=self.num_lanes, dt=self.dt)
            self.trace_geometry = None
            
        alpha_val = alpha if alpha is not None else cfg_trust.get("alpha", 0.5)
        tau_val = exclusion_tau if exclusion_tau is not None else cfg_trust.get("exclusion_threshold_tau", 0.40)
        
        if trust_engine is not None:
            self.trust_engine = trust_engine
            if alpha is not None:
                self.trust_engine.alpha = float(alpha)
            if exclusion_tau is not None:
                self.trust_engine.exclusion_threshold = float(exclusion_tau)
        else:
            self.trust_engine = TrustEngine(alpha=alpha_val, exclusion_threshold=tau_val, config=config)
            
        self.reward_calculator = RewardCalculator(**{
            k: float(v) for k, v in cfg_reward.items()
            if k in ("alpha_lat", "beta_energy", "gamma_fail", "delta_success",
                     "lambda_trust", "e_norm")})

        self.state_features = str(cfg_model.get("state_features", "v2.0"))
        if self.state_features not in ("v2.0", "v2.1"):
            raise ValueError("environment_model.state_features must be 'v2.0' or 'v2.1'")
        self.state_dim = STATE_DIM_V21 if self.state_features == "v2.1" else STATE_DIM
        self.action_dim = 5  # 0: Local, 1: RSU1, 2: RSU2, 3: RSU3, 4: Best V2V Peer
        # Ordering of vehicle decisions inside one time slot is shuffled with a
        # dedicated seeded generator, so that no vehicle identifier is
        # systematically served first by the edge queues.
        self._order_rng = np.random.RandomState(int(self.random_seed) + 17)
        self._uplinks_this_slot = {}
        
        # RSUs from config (supporting non-uniform positions, 3D rooftop elevation, and multi-core edge servers)
        #
        # For trace-replay modes the deployment is adapted to the *observed*
        # road segment: the authentic US-101 study segment is ~675 m long, so
        # the default multi-kilometre deployment would leave most RSUs outside
        # the replayed trajectories.
        segment_rsu_positions = None
        if self.mobility_mode in ("trace", "authentic_trace") and hasattr(self.mobility, "rsu_positions_for_segment"):
            num_rsus_cfg = int(cfg_edge.get("num_rsus", 3))
            segment_rsu_positions = self.mobility.rsu_positions_for_segment(num_rsus_cfg)
            
        rsu_nodes_cfg = cfg_edge.get("rsu_nodes")
        if rsu_nodes_cfg:
            self.rsus = [
                RSU(
                    rsu_id=node.get("id", f"RSU_{i+1}"),
                    x=float(segment_rsu_positions[i]) if (segment_rsu_positions and i < len(segment_rsu_positions))
                       else float(node.get("x_m", 420.0)),
                    y=float(node.get("y_m", 15.0)),
                    z_height=float(node.get("z_height_m", 14.0)),
                    cpu_capacity_ghz=float(node.get("cpu_capacity_ghz", 32.0)),
                    coverage_radius_m=float(node.get("coverage_radius_m", 480.0)),
                    num_cores=int(node.get("num_cores", 8)),
                    is_rooftop=bool(node.get("is_rooftop", False)),
                    building_name=str(node.get("building_name", ""))
                )
                for i, node in enumerate(rsu_nodes_cfg)
            ]
        else:
            rsu_positions = (segment_rsu_positions if segment_rsu_positions
                             else cfg_edge.get("positions_m", [420.0, 1480.0, 2460.0]))
            rsu_cpu = float(cfg_edge.get("rsu_cpu_capacity_ghz", 32.0))
            rsu_radius = float(cfg_edge.get("coverage_radius_m", 480.0))
            self.rsus = [
                RSU(
                    f"RSU_{i+1}",
                    x=float(pos),
                    y=15.0 if i == 0 else (-42.0 if i == 1 else 38.0),
                    z_height=14.0 if i == 0 else (38.0 if i == 1 else 28.0),
                    cpu_capacity_ghz=32.0 if i == 0 else (48.0 if i == 1 else 40.0),
                    coverage_radius_m=480.0 if i == 0 else (650.0 if i == 1 else 550.0),
                    num_cores=8 if i == 0 else (16 if i == 1 else 12),
                    is_rooftop=(i > 0),
                    building_name="Roadside Pole" if i == 0 else ("Apex Tech Rooftop" if i == 1 else "Logistics Rooftop")
                )
                for i, pos in enumerate(rsu_positions)
            ]
        
        # Optional empirical edge calibration (Alibaba edge-platform trace).
        self.current_time_sec = 0.0
        self.edge_cal = EdgeCalibration.from_config(config, seed=self.random_seed)
        if self.edge_cal is not None:
            scales = self.edge_cal.capacity_scale(len(self.rsus))
            self._site_factors = self.edge_cal.site_factors(len(self.rsus))
            for rsu, sc in zip(self.rsus, scales):
                rsu.cpu_capacity_ghz *= sc
            self._apply_background_load()
        else:
            self._site_factors = [1.0] * len(self.rsus)

        # Optional user-specified RSU placement along the road axis.  It is only
        # valid for the procedural highway: a trace replay places the RSUs
        # inside the observed road segment, and arbitrary positions there would
        # leave the deployment inconsistent with the recorded trajectories.
        if rsu_x_positions is not None:
            if self.mobility_mode != "highway":
                raise ValueError(
                    "RSU positions can only be configured for the procedural "
                    "highway; trace replay places RSUs inside the observed segment")
            xs = [float(x) for x in rsu_x_positions]
            if len(xs) != len(self.rsus):
                raise ValueError("expected %d RSU positions, got %d" % (len(self.rsus), len(xs)))
            if any(x < 0.0 or x > self.road_length for x in xs):
                raise ValueError("RSU positions must lie within [0, %.0f] m" % self.road_length)
            for rsu, x in zip(self.rsus, xs):
                rsu.x = x

        # A fault is either FIXED (an integer RSU index) or ROTATING (``"random"``):
        # a rotating fault moves to a different, randomly chosen RSU every
        # ``rotate_every_steps`` simulation steps.  The schedule is a seeded function
        # of the environment seed (identical for every method evaluated on it), and
        # it differs between training and evaluation environments, so a policy
        # cannot memorise WHICH node is faulty - it has to observe behaviour.
        self._rotating = None
        self._fault_rng = np.random.RandomState(int(self.random_seed) + 23)
        for spec in self.faulty_rsus_cfg:
            if str(spec.get("index", 0)) == "random":
                self._rotating = {"drop_prob": float(spec.get("drop_prob", 0.0)),
                                  "delay_factor": float(spec.get("delay_factor", 1.0)),
                                  "every": max(1, int(spec.get("rotate_every_steps", 40))),
                                  "current": None}
                continue
            idx = int(spec.get("index", 0))
            if not 0 <= idx < len(self.rsus):
                raise ValueError("faulty_rsus index %d is outside 0..%d" % (idx, len(self.rsus) - 1))
            self.rsus[idx].fault_drop_prob = float(spec.get("drop_prob", 0.0))
            self.rsus[idx].fault_delay_factor = float(spec.get("delay_factor", 1.0))
        if self._rotating is not None:
            self._rotate_fault()

        self.vehicles = {}
        self.current_time_sec = 0.0
        self.step_count = 0
        self.reset()

    def reset(self):
        self.current_time_sec = 0.0
        self.step_count = 0
        self.vehicles.clear()
        
        num_attackers = int(self.num_vehicles * self.attack_ratio)
        num_unreliable = int(round(self.num_vehicles * self.unreliable_ratio))
        roles = (["poisoning"] * (num_attackers // 2) + 
                 ["free_rider"] * (num_attackers - num_attackers // 2) + 
                 ["unreliable"] * num_unreliable +
                 ["honest"] * max(0, self.num_vehicles - num_attackers - num_unreliable))
        np.random.shuffle(roles)
        
        for i in range(self.num_vehicles):
            vid = i + 1
            role = roles[i]
            
            # Neutral, role-agnostic initial trust prior for all nodes.
            # Zero ground-truth leakage: trust evolves strictly through observable evidence.
            init_t = float(self.config.get("trust_management", {}).get("initial_trust_prior", 0.70)) if self.config else 0.70

            v = Vehicle(
                vehicle_id=vid,
                x=0.0,
                y=0.0,
                speed=22.0,
                lane=(i % 5) + 1,
                vclass=["Car", "SUV", "Autonomous_EV", "Truck"][i % 4],
                # On-board compute is calibrated to the automotive SoC class
                # (roughly 0.8-2.0 GHz of *effective single-thread*
                # throughput), not to a desktop-class core.  With a desktop-class
                # CPU every task in the catalogue is trivially executable
                # locally, so offloading is never necessary and no offloading
                # policy - however sophisticated - can show an advantage.
                cpu_capacity_ghz=np.random.uniform(0.8, 2.0),
                role=role
            )
            v.v2v_range_m = float(self.config.get("network", {}).get("v2v_comm_range_m", 180.0)) if self.config else 180.0
            v.trust_score = init_t
            if self.data_plane_enabled:
                if role == "poisoning":
                    v.helper_drop_prob = self.dp_malicious_drop
                    v.helper_delay_factor = self.dp_malicious_delay
                elif role == "free_rider":
                    v.helper_drop_prob = self.dp_selfish_drop
            self.vehicles[vid] = v

            node_key = f"V{vid}"
            # A freshly created node has *no observed history*.  The record
            # therefore starts with the documented, role-agnostic prior
            # parameters and zero real observations; the last SINR and
            # utilization are unknown until a first interaction happens.
            tracker = self.trust_engine.behavior_tracker
            self.trust_engine.behavior_tracker.node_records[node_key] = {
                "trust_score": init_t,
                "prior_successes": tracker.prior_successes,
                "prior_total": tracker.prior_total,
                "observed_successes": 0,
                "observed_total": 0,
                "recent_observations": [],
                "last_sinr": None,
                "last_utilization": None,
            }
            self.trust_engine.final_trust_scores[node_key] = init_t
            
        if self.rich_fraction > 0.0:
            rs = np.random.RandomState(int(self.random_seed) + 41)
            n_rich = int(round(self.rich_fraction * self.num_vehicles))
            for vid in rs.choice(sorted(self.vehicles), size=n_rich, replace=False):
                self.vehicles[int(vid)].cpu_capacity_ghz = float(rs.uniform(*self.rich_cpu_range))
                self.vehicles[int(vid)].vclass = "Truck"

        # Distribute along 2400m highway with safe headway spacing
        self.mobility.initialize_spaced_positions(self.vehicles)
            
        for rsu in self.rsus:
            self.trust_engine.compute_joint_trust(rsu.rsu_id)
            
        return self._get_global_state_dict()


    def update_positions(self):
        """Advances vehicle mobility positions along highway using IDM car-following."""
        self.current_time_sec += self.dt
        self.step_count += 1
        self._uplinks_this_slot = {}
        if self._rotating is not None and self.step_count % self._rotating["every"] == 0:
            self._rotate_fault()
        self.mobility.step(self.vehicles)
        self._apply_background_load()

        # Advance the edge and vehicle queue simulations so that the waiting
        # times, service times and completion counters reflect the work that
        # has actually elapsed.
        for rsu in self.rsus:
            rsu.step(self.current_time_sec)
        for v in self.vehicles.values():
            v.step(self.current_time_sec)



    def _apply_background_load(self):
        """Set each RSU's co-tenant CPU share from the empirical diurnal profile."""
        if self.edge_cal is None:
            return
        for rsu, f in zip(self.rsus, self._site_factors):
            rsu.background_share = self.edge_cal.background_share(self.current_time_sec, f)

    def _v2v_candidates(self, v):
        """In-range, currently present peers of ``v`` with their distance."""
        out = []
        for p_id, p in self.vehicles.items():
            if p_id == v.vehicle_id or not getattr(p, "active", True):
                continue
            d = v.get_distance_to(p.x, p.y)
            if d <= v.v2v_range_m:
                out.append((p, d))
        return out

    def select_v2v_peer(self, v):
        """Peer used by action 4 and described by the peer state features."""
        cands = self._v2v_candidates(v)
        if not cands:
            return None, 999.0
        if self.v2v_peer_selection == "nearest":
            return min(cands, key=lambda c: c[1])
        tracker = self.trust_engine.behavior_tracker

        def score(c):
            p, d = c
            spare = 1.0 - p.queue_length / float(p.max_queue)
            return (tracker.get_trust(f"V{p.vehicle_id}")
                    + 0.35 * (1.0 - d / v.v2v_range_m) + 0.15 * spare
                    + 0.1 * (p.cpu_capacity_ghz / 2.0))
        return max(cands, key=score)

    def _rotate_fault(self):
        """Move the rotating fault to a different RSU (never the same one twice in a row)."""
        r = self._rotating
        choices = [i for i in range(len(self.rsus)) if i != r["current"]]
        r["current"] = int(self._fault_rng.choice(choices))
        for i, rsu in enumerate(self.rsus):
            on = (i == r["current"])
            rsu.fault_drop_prob = r["drop_prob"] if on else 0.0
            rsu.fault_delay_factor = r["delay_factor"] if on else 1.0

    def faulty_rsu_ids(self):
        """Identifiers of the RSUs that are currently faulty (evaluation only)."""
        return [r.rsu_id for r in self.rsus if r.fault_drop_prob > 0.0 or r.fault_delay_factor > 1.0]

    def scene_description(self):
        """Geometry the visual representation must use (the backend's own numbers)."""
        return {
            "mobility_mode": self.mobility_mode,
            "road_length_m": float(self.road_length),
            "num_lanes": int(self.num_lanes),
            "lane_width_m": float(self.lane_width),
            "periodic": self.mobility_mode == "highway",
            "x_min_m": float(getattr(self.mobility, "trace_x_min", 0.0)) if self.trace_geometry else 0.0,
            "rsus": [{"id": r.rsu_id, "x": r.x, "y": r.y,
                      "coverage_radius_m": r.coverage_radius_m,
                      "cpu_capacity_ghz": r.cpu_capacity_ghz, "num_cores": r.num_cores}
                     for r in self.rsus],
        }

    def connected_rsus(self, v):
        """Identifiers of the RSUs whose coverage currently contains vehicle ``v``."""
        if not getattr(v, "active", True):
            return []
        return [r.rsu_id for r in self.rsus
                if r.get_distance_to(v.x, v.y) <= r.coverage_radius_m]

    def vehicle_order(self):
        """Seeded random service order of the vehicles for the current slot."""
        ids = sorted(self.vehicles.keys())
        self._order_rng.shuffle(ids)
        return ids

    def _rsu_load(self, rsu):
        """Normalised edge load: queued tasks relative to twice the core count."""
        return min(1.0, rsu.get_queue_length() / float(2 * max(1, rsu.num_cores)))

    def _get_vehicle_state(self, v, current_task):
        """Constructs 14-dimensional state vector for vehicle v."""
        d_rsu = [r.get_distance_to(v.x, v.y) for r in self.rsus]
        t_rsu = [self.trust_engine.behavior_tracker.get_trust(r.rsu_id) for r in self.rsus]
        
        # Find best candidate peer vehicle
        best_peer_dist = 999.0
        best_peer_trust = 0.5
        peer, dist = self.select_v2v_peer(v)
        if peer is not None:
            best_peer_dist = dist
            best_peer_trust = self.trust_engine.behavior_tracker.get_trust(f"V{peer.vehicle_id}")
                
        half_len = max(100.0, self.road_length / 2.0)
        if self.state_features == "v2.1":
            rel = [d / max(1.0, r.coverage_radius_m) for d, r in zip(d_rsu, self.rsus)]
            dist_feats = [min(x, 2.0) / 2.0 for x in rel]          # 0.5 = coverage boundary
            cover_feats = [1.0 if x <= 1.0 else 0.0 for x in rel]  # in coverage of RSU i
        else:
            dist_feats = [d / half_len for d in d_rsu]
            cover_feats = []
        state = [
            current_task.data_size_mb / 10.0,
            current_task.cpu_cycles_giga / 5.0,
            current_task.deadline_ms / 300.0,
            v.speed / 35.0,
            dist_feats[0],
            dist_feats[1],
            dist_feats[2],
            v.queue_length / float(v.max_queue),
            v.cpu_capacity_ghz / 8.0,
            t_rsu[0],
            t_rsu[1],
            t_rsu[2],
            best_peer_dist / 200.0,
            best_peer_trust
        ] + [self._rsu_load(r) for r in self.rsus[:3]] + cover_feats
        if self.mask_trust_features:
            for i in (9, 10, 11, 13):      # RSU trust x3, best-peer trust
                state[i] = 0.0
        return np.array(state, dtype=np.float32)

    def _get_global_state_dict(self):
        states = {}
        for vid, v in self.vehicles.items():
            dummy_task = self.task_generator.generate_task(vid, self.current_time_sec)
            states[vid] = (self._get_vehicle_state(v, dummy_task), dummy_task)
        return states

    def step_vehicle(self, vid, action, task):
        """
        Executes one offloading decision for vehicle vid on task.
        Actions:
        0: Local
        1: RSU 1
        2: RSU 2
        3: RSU 3
        4: V2V Peer
        """
        v = self.vehicles[vid]

        # A vehicle that is not currently present in the trace observation
        # window does not exist on the road: it must not generate traffic,
        # consume resources or contribute a training transition.
        if not getattr(v, "active", True):
            info = {
                "vid": vid,
                "target": "not_present",
                "action": action,
                "success": False,
                "present": False,
                "delay_ms": 0.0,
                "deadline_ms": task.deadline_ms,
                "energy_j": 0.0,
                "reward": 0.0,
                "target_trust": 0.0,
            }
            return None, 0.0, True, info, None

        target_name = "Local"
        target_trust = 1.0
        fail_reason = None
        success = False
        delay_sec = 0.0
        energy_j = 0.0
        
        if action == 0:
            # 1. Local execution.
            # The trust term in the reward exists to discriminate between
            # *candidate offloading targets* (which third party to trust).
            # Local execution involves no third party, so it contributes a
            # NEUTRAL trust score of 0.0.
            #
            # Awarding the maximum trust score (1.0) to self-execution is a
            # modelling error: it adds a constant bonus of
            # lambda_trust * 1.0 to every local decision, which dominates the
            # latency and energy terms and makes local execution optimal for
            # every task.  The learned policy then degenerates to always
            # choosing action 0, the trust mechanism has no effect on the
            # decision, and the proposed method becomes numerically identical
            # to the Local-Only baseline.  With a neutral score the trust
            # term only ranks genuine offloading candidates.
            #
            # DEADLINE SYMMETRY: the local path is subject to exactly the
            # same success criterion as an offloaded task - the task counts
            # as completed only if it was accepted *and* finished within its
            # deadline.  ``Vehicle.execute_local`` returns the sojourn time
            # (queueing wait + service time) and documents that the caller
            # compares it with the deadline; without that comparison a
            # purely local baseline could never miss a deadline, which made
            # the task-completion comparison between the local and the
            # offloading paths asymmetric.
            accepted, delay_sec, energy_j = v.execute_local(
                task, self.channel, self.current_time_sec)
            success = bool(accepted) and (delay_sec <= task.deadline_sec)
            if not accepted:
                # The task was dropped by the local queue: report the
                # deadline overrun in the same way the offloading branches do.
                delay_sec = task.deadline_sec * 1.5
            target_name = f"V{vid}_Local"
            target_trust = 0.0
            if success:
                v.tasks_completed += 1
            else:
                v.tasks_failed += 1

        elif action in [1, 2, 3]:
            # 2. Edge offloading to the RSU the policy selected.
            req_rsu = self.rsus[action - 1]

            if self.honor_rsu_choice:
                # Version-2 semantics: the action names a specific RSU.  If the
                # vehicle is outside that RSU's coverage no link exists, no
                # transmission takes place, and the task fails.  The decision
                # is the learner's responsibility; the simulator does not
                # silently re-associate the vehicle with a better cell.
                d_req = req_rsu.get_distance_to(v.x, v.y)
                candidates = [(req_rsu, d_req)] if d_req <= req_rsu.coverage_radius_m else []
            else:
                # Version-1 semantics (kept for before/after comparison).
                candidates = []
                for r in self.rsus:
                    dist_3d = r.get_distance_to(v.x, v.y)
                    if dist_3d <= r.coverage_radius_m:
                        candidates.append((r, dist_3d))

            if not candidates:
                rsu = req_rsu
                target_name = rsu.rsu_id
                target_trust = self.trust_engine.behavior_tracker.get_trust(rsu.rsu_id)
                success = False
                delay_sec = task.deadline_sec * 1.5
                task.completed = False
                task.failed = True
                if self.honor_rsu_choice:
                    # No link was established, so nothing was transmitted and
                    # the RSU is not at fault: no energy, no trust penalty.
                    energy_j = 0.0
                    fail_reason = "out_of_coverage"
                else:
                    energy_j = 0.01
                    self.trust_engine.behavior_tracker.penalize_unresponsive(rsu.rsu_id)
            else:
                if self.honor_rsu_choice:
                    rsu, dist = candidates[0]
                else:
                    def rsu_utility(cand_item):
                        c_rsu, c_dist = cand_item
                        u_close = max(0.0, 1.0 - (c_dist / max(1.0, c_rsu.coverage_radius_m)))
                        u_res = c_rsu.cpu_capacity_ghz / 48.0
                        u_load = (c_rsu.get_queue_length() / 20.0) + (0.5 * c_rsu.current_utilization)
                        u_trust = self.trust_engine.behavior_tracker.get_trust(c_rsu.rsu_id)
                        u_affinity = 0.15 if (c_rsu.rsu_id == req_rsu.rsu_id and c_rsu.get_queue_length() < 5) else 0.0
                        return (0.35 * u_close) + (0.25 * u_res) - (0.25 * u_load) + (0.15 * u_trust) + u_affinity
                    rsu, dist = max(candidates, key=rsu_utility)
                target_name = rsu.rsu_id
                target_trust = self.trust_engine.behavior_tracker.get_trust(rsu.rsu_id)

                # ONE channel sample is drawn for this interaction and reused
                # for the trust update below.
                channel_obs = self.channel.sample_channel(
                    dist, max_range=rsu.coverage_radius_m)
                tx_delay, tx_energy = self.channel.transmission_delay_from_observation(
                    channel_obs, task.data_size_mb)

                # Uplink contention: the cell's bandwidth is shared (OFDMA,
                # equal allocation) among the vehicles that transmit to this
                # RSU in the current slot, this vehicle included.  A vehicle
                # keeps its full UE channel until more than
                # cell_bandwidth / UE_bandwidth vehicles are active.  Airtime,
                # and therefore transmit energy, grows when it is squeezed.
                if self.uplink_contention and tx_delay != float('inf'):
                    n_up = self._uplinks_this_slot.get(rsu.rsu_id, 0) + 1
                    self._uplinks_this_slot[rsu.rsu_id] = n_up
                    tx_delay, tx_energy = self.channel.transmission_delay_shared(
                        channel_obs, task.data_size_mb, self.cell_bandwidth_mhz, n_up)

                # Displacement during the transmission.  The procedural
                # highway is a closed loop; a trace replay is not.
                end_x = v.x + v.speed * (tx_delay if tx_delay != float('inf') else 0.0)
                if isinstance(self.mobility, HighwayMobility):
                    end_x = end_x % self.road_length
                end_dist = rsu.get_distance_to(end_x, v.y)
                handover_delay = 0.0

                if tx_delay != float('inf') and end_dist > rsu.coverage_radius_m:
                    # The vehicle left the cell mid-transmission: continue via an
                    # adjacent RSU (Xn handover, 25 ms signalling) if one covers
                    # the end position, otherwise the task is lost.
                    handover_candidates = [r for r in self.rsus if r.get_distance_to(end_x, v.y) <= r.coverage_radius_m]
                    if handover_candidates and self.edge_cal is not None:
                        # Measured inter-site backhaul: delay and loss of the hand-over.
                        handover_delay = self.edge_cal.handover_delay_sec()
                        if self.edge_cal.handover_lost():
                            tx_delay = float('inf')
                    elif handover_candidates:
                        handover_delay = 0.025
                    else:
                        tx_delay = float('inf')

                if tx_delay != float('inf'):
                    proc_ok, comp_delay, comp_energy = rsu.process_task(task, self.channel, self.current_time_sec)
                else:
                    proc_ok = False
                    comp_delay = 0.0
                    comp_energy = 0.0

                if proc_ok:
                    delay_sec = tx_delay + comp_delay + handover_delay
                    energy_j = tx_energy + comp_energy
                    success = (delay_sec <= task.deadline_sec)
                else:
                    success = False
                    delay_sec = task.deadline_sec * 1.5
                    energy_j = tx_energy if tx_delay != float('inf') else 0.0

                sinr_db = channel_obs.sinr_db
                self.trust_engine.behavior_tracker.record_interaction(
                    rsu.rsu_id, success, delay_sec, task.deadline_sec, sinr_db, rsu.current_utilization
                )

        elif action == 4:
            # 3. V2V Peer Offloading
            target_name = "V2V_Peer"
            # Locate nearest vehicle peer in front or behind
            best_peer, best_dist = self.select_v2v_peer(v)

            if best_peer is None:
                # No peer in range -> Task drops
                target_name = "None"
                success = False
                delay_sec = task.deadline_sec * 1.5
                energy_j = 0.0
                task.completed = False
                task.failed = True
            else:
                peer_node_id = f"V{best_peer.vehicle_id}"
                target_name = peer_node_id
                target_trust = self.trust_engine.behavior_tracker.get_trust(peer_node_id)
                
                # ONE channel sample for this V2V interaction, reused for the
                # trust update below.
                channel_obs = self.channel.sample_channel(
                    best_dist, max_range=v.v2v_range_m)
                tx_delay, tx_energy = self.channel.transmission_delay_from_observation(
                    channel_obs, task.data_size_mb)
                
                # Check relative mobility displacement during transmission
                rel_speed = abs(v.speed - best_peer.speed)
                end_v2v_dist = best_dist + rel_speed * tx_delay
                if end_v2v_dist > v.v2v_range_m:
                    # Vehicle pair separated during transmission
                    proc_ok = False
                    comp_delay = 0.0
                    comp_energy = 0.0
                else:
                    proc_ok, comp_delay, comp_energy = best_peer.execute_as_v2v_helper(
                        task, self.channel, self.current_time_sec)
                
                if proc_ok:
                    delay_sec = tx_delay + comp_delay
                    energy_j = tx_energy + comp_energy
                    success = (delay_sec <= task.deadline_sec)
                else:
                    success = False
                    delay_sec = task.deadline_sec * 2.0
                    energy_j = tx_energy
                    
                # The trust update uses the SAME channel sample that produced
                # the transmission outcome above.
                sinr_db = channel_obs.sinr_db
                util = best_peer.queue_length / float(best_peer.max_queue)
                self.trust_engine.behavior_tracker.record_interaction(
                    peer_node_id, success, delay_sec, task.deadline_sec, sinr_db, util
                )

        reward = self.reward_calculator.compute_reward(
            success=success,
            delay_sec=delay_sec,
            deadline_sec=task.deadline_sec,
            energy_j=energy_j,
            target_trust_score=target_trust
        )
        
        task.completed = success
        task.failed = not success
        task.total_delay_sec = delay_sec
        task.energy_joules = energy_j
        
        # Generate next task and next state
        next_task = self.task_generator.generate_task(vid, self.current_time_sec)
        next_state = self._get_vehicle_state(v, next_task)
        done = False
        
        info = {
            "vid": vid,
            "target": target_name,
            "action": action,
            "success": success,
            "failure_reason": None if success else (fail_reason or "deadline_or_rejected"),
            "present": True,
            "delay_ms": round(delay_sec * 1000.0, 2),
            "deadline_ms": task.deadline_ms,
            "energy_j": round(energy_j, 4),
            "reward": round(reward, 3),
            "target_trust": round(target_trust, 2)
        }
        return next_state, reward, done, info, next_task
