"""Record the real federated updates of training runs so that exclusion rules
can be designed and compared OFFLINE on identical data.

Each trace file holds, for every federated round, the flattened update
(local - global) of every uploading client, its role label and the round index.
Updates are stored in float16 (the detectors use cosines, norms and distances,
which are insensitive to the precision loss).

    python -m experiments.collect_update_traces OUT_DIR [--ratios 0 0.24 0.4] [--seeds 3]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from experiments import experiment_runner as er
from federated.server import FederatedServer


def collect(out_dir, ratios, seeds, strategy="fedavg", poison_type=None, strength=None):
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config()
    orig = FederatedServer.run_aggregation_round
    for ratio in ratios:
        for seed in seeds:
            rec = {"round": [], "cid": [], "role": [], "delta": []}

            def spy(self, models, samples, ground_truth_roles=None, staleness=None, _rec=rec):
                r = self.current_round + 1
                for cid, w in models.items():
                    d = self.trust_engine.model_evaluator.extract_flat_update(w, self.global_weights)
                    _rec["round"].append(r)
                    _rec["cid"].append(int(cid))
                    _rec["role"].append((ground_truth_roles or {}).get(cid, "honest"))
                    _rec["delta"].append(d.astype(np.float16))
                return orig(self, models, samples, ground_truth_roles=ground_truth_roles, staleness=staleness)

            FederatedServer.run_aggregation_round = spy
            try:
                kw = dict(attack_ratio=ratio, seed=seed, config=cfg, aggregation_strategy=strategy)
                if poison_type:
                    kw["poison_type"] = poison_type
                if strength is not None:
                    kw["poison_strength"] = strength
                er.run_simulation_trial(**kw)
            finally:
                FederatedServer.run_aggregation_round = orig
            name = "trace_%s_r%.2f_s%d.npz" % (poison_type or "sign_flip", ratio, seed)
            np.savez_compressed(os.path.join(out_dir, name), round=np.array(rec["round"]),
                                cid=np.array(rec["cid"]), role=np.array(rec["role"]),
                                delta=np.stack(rec["delta"]))
            print("saved", name, len(rec["round"]), "updates", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.24, 0.4])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--strategy", default="fedavg")
    ap.add_argument("--poison", default=None)
    ap.add_argument("--strength", type=float, default=None)
    a = ap.parse_args()
    collect(a.out_dir, a.ratios, [1000 + i for i in range(a.seeds)], a.strategy, a.poison, a.strength)
