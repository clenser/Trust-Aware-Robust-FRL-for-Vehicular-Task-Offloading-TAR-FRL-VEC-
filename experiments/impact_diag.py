"""Feasibility diagnostic for impact-validated exclusion.

For several attack families, run the federated protocol with FedAvg (the defence under
test is NOT active, so the data are the same for every candidate rule), and at every
round record for each uploading client the change in value error on the server's
trusted root data caused by its model:

    delta_i = (L(w_i) - L(w_global)) / L(w_global)

plus the geometry statistics the do-no-harm rule uses.  The output is the ROC AUC of
each statistic for separating poisoned from honest clients, per attack.

    python -m experiments.impact_diag [--seeds 2] [--out FILE]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["VANNET_ROOT"] = "1"

from scipy.stats import rankdata

from configs.config_loader import load_config
from experiments import experiment_runner as er
from federated.server import FederatedServer
from trust.dnh_detector import loo_features

ATTACKS = (("sign_flip", None), ("stealth_sign_flip", None), ("partial_sign_flip", 0.5), ("scaling", None))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def collect(attack, strength, seed, cfg):
    rows = []
    orig = FederatedServer.run_aggregation_round

    def spy(self, models, samples, ground_truth_roles=None, staleness=None):
        g = self.get_global_weights()
        self.root_trainer(g)                                  # refresh the trusted root data
        base = self.root_trainer.loss(g)
        deltas = {c: self.trust_engine.model_evaluator.extract_flat_update(w, g) for c, w in models.items()}
        ids = list(deltas)
        mat = np.array([deltas[c] for c in ids])
        ratio, cos, _, live = loo_features(mat)
        for i, c in enumerate(ids):
            role = (ground_truth_roles or {}).get(c, "honest")
            if role not in ("honest", "poisoning") or not live[i] or base is None:
                continue
            li = self.root_trainer.loss(models[c])
            rows.append({"role": role, "impact": (li - base) / max(base, 1e-9), "ratio": float(ratio[i]),
                         "cos": float(cos[i]), "round": self.current_round + 1})
        return orig(self, models, samples, ground_truth_roles=ground_truth_roles, staleness=staleness)

    FederatedServer.run_aggregation_round = spy
    try:
        kw = dict(attack_ratio=0.24, seed=seed, config=cfg, aggregation_strategy="fedavg", poison_type=attack)
        if strength is not None:
            kw["poison_strength"] = strength
        er.run_simulation_trial(**kw)
    finally:
        FederatedServer.run_aggregation_round = orig
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(er.PROJECT_ROOT, "results", "impact_diag.json"))
    a = ap.parse_args()
    cfg = load_config()
    out = {}
    for attack, strength in ATTACKS:
        rows = []
        for s in range(a.seeds):
            rows += collect(attack, strength, 3000 + s, cfg)
        h = [r for r in rows if r["role"] == "honest"]
        p = [r for r in rows if r["role"] == "poisoning"]
        res = {"n_honest": len(h), "n_poison": len(p)}
        for stat, sign in (("impact", 1), ("ratio", 1), ("cos", -1)):
            res["auc_" + stat] = auc([sign * r[stat] for r in p], [sign * r[stat] for r in h])
        res["impact_honest_median"] = float(np.median([r["impact"] for r in h]))
        res["impact_poison_median"] = float(np.median([r["impact"] for r in p]))
        res["impact_honest_p95"] = float(np.percentile([r["impact"] for r in h], 95))
        res["impact_poison_p5"] = float(np.percentile([r["impact"] for r in p], 5))
        out[attack] = res
        print("%-18s n=%d/%d  AUC impact %.3f | ratio %.3f | cos %.3f   impact med honest %.3f poison %.3f" % (
            attack, len(h), len(p), res["auc_impact"], res["auc_ratio"], res["auc_cos"],
            res["impact_honest_median"], res["impact_poison_median"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
