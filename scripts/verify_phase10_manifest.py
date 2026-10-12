"""Phase 10 v4 manifest: build SHA-256 artifact manifest or verify it.

Build mode (default): hash every artifact under the v4 results/figures/tables
dirs, recompute matrix counts and effective core parameters FROM the CSVs
(never from the plan), and write reports/PHASE10_FINAL_V4_RUN_MANIFEST.json.
Verify mode (--check): re-hash every manifest entry and fail on any mismatch,
missing file, or row-count drift.

Expected base accounting: 880 checkpoints + 3 CSVs + 1 summary JSON +
6 figures + 4 tables = 894 artifacts (excluding the manifest itself).
If the actual count differs, the manifest records the real list and the
audit report must explain why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts(results_dir: Path, figs_dir: Path, tables_dir: Path) -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    for base in (results_dir, figs_dir, tables_dir):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".csv", ".json", ".pt", ".png", ".tex"}:
                rel = p.relative_to(REPO_ROOT).as_posix()
                artifacts[rel] = {
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
    return artifacts


def summarize_matrices(results_dir: Path) -> dict:
    core = pd.read_csv(results_dir / "phase10_experiment_runs.csv")
    abl = pd.read_csv(results_dir / "phase10_ablation_runs.csv")
    sens = pd.read_csv(results_dir / "phase10_sensitivity_runs.csv")
    # Effective core parameters recomputed FROM OUTPUT, not from the plan.
    core_alpha = sorted(core["alpha"].unique().tolist())
    core_tau = sorted(core["tau_exc"].unique().tolist())
    core_lam = sorted(core["runtime_trust_lambda"].unique().tolist())
    return {
        "core_rows": len(core),
        "ablation_rows": len(abl),
        "sensitivity_rows": len(sens),
        "total_rows": len(core) + len(abl) + len(sens),
        "core_run_ids": sorted(core["run_id"].tolist()),
        "ablation_run_ids": sorted(abl["run_id"].tolist()),
        "sensitivity_run_ids": sorted(sens["run_id"].tolist()),
        "effective_core_alpha": core_alpha,
        "effective_core_tau_exc": core_tau,
        "effective_core_lambda": core_lam,
    }


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=REPO_ROOT, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_manifest(results_dir: Path, figs_dir: Path, tables_dir: Path,
                   config_path: str, commands: list[str], tests: dict) -> dict:
    import yaml
    with open(REPO_ROOT / config_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    matrices = summarize_matrices(results_dir)
    artifacts = collect_artifacts(results_dir, figs_dir, tables_dir)
    return {
        "phase": "Phase 10 Final v4 Corrected Evaluation",
        "config_file": config_path,
        "declared_core": {
            "alpha": 0.50,
            "tau_exc": 0.35,
            "trust_lambda": 0.70,
            "m2_dir_threshold": 0.45,
            "m3_stale_threshold": 0.50,
            "m4_orig_threshold": 0.00,
            "tau_anomaly": 0.50,
            "rounds": 8,
            "num_clients": 10,
        },
        "matrix_counts": {
            "core_rows": matrices.pop("core_rows"),
            "ablation_rows": matrices.pop("ablation_rows"),
            "sensitivity_rows": matrices.pop("sensitivity_rows"),
            "total_rows": matrices.pop("total_rows"),
        },
        "matrix_row_ids": {
            "core": matrices.pop("core_run_ids"),
            "ablation": matrices.pop("ablation_run_ids"),
            "sensitivity": matrices.pop("sensitivity_run_ids"),
        },
        "effective_parameters_from_output": {
            "core_alpha_values": matrices.pop("effective_core_alpha"),
            "core_tau_exc_values": matrices.pop("effective_core_tau_exc"),
            "core_lambda_values": matrices.pop("effective_core_lambda"),
        },
        "config_snapshot": {
            "federated": raw_cfg.get("federated", {}),
            "detector": raw_cfg.get("detector", {}),
            "scenario_trust_lambda": (raw_cfg.get("scenario", {}) or {}).get("trust_lambda"),
        },
        "code_revision": git_revision(),
        "tests": tests,
        "commands": commands,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }


def check_manifest(manifest_path: Path, repo_root: Path | None = None,
                   results_subdir: str = "results/phase10_final_v4") -> list[str]:
    """Re-hash every entry; return list of failures (empty = all verified)."""
    root = repo_root or REPO_ROOT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for rel, info in manifest["artifacts"].items():
        p = root / rel
        if not p.is_file():
            failures.append(f"missing: {rel}")
            continue
        actual = sha256_file(p)
        if actual != info["sha256"]:
            failures.append(f"hash mismatch: {rel}")
    # Row-count drift check against live CSVs
    matrices = summarize_matrices(root / results_subdir)
    for key in ("core_rows", "ablation_rows", "sensitivity_rows", "total_rows"):
        if matrices[key] != manifest["matrix_counts"][key]:
            failures.append(f"row drift: {key} manifest={manifest['matrix_counts'][key]} live={matrices[key]}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/phase10_final_v4")
    parser.add_argument("--figs-dir", default="reports/figures/phase10_final_v4")
    parser.add_argument("--tables-dir", default="reports/tables/phase10_final_v4")
    parser.add_argument("--config", default="configs/feddqn_phase10.yaml")
    parser.add_argument("--out", default="reports/PHASE10_FINAL_V4_RUN_MANIFEST.json")
    parser.add_argument("--check", action="store_true", help="verify an existing manifest instead of building")
    parser.add_argument("--tests-passed", type=int, default=-1)
    parser.add_argument("--tests-total", type=int, default=-1)
    parser.add_argument("--pytest-exit", type=int, default=-1)
    parser.add_argument("--commands", default="", help="; separated command provenance lines")
    args = parser.parse_args(argv)

    if args.check:
        failures = check_manifest(REPO_ROOT / args.out)
        if failures:
            print(f"MANIFEST CHECK FAILED ({len(failures)} problems):")
            for f in failures:
                print(f"  - {f}")
            return 1
        manifest = json.loads((REPO_ROOT / args.out).read_text(encoding="utf-8"))
        print(f"MANIFEST CHECK PASSED: {len(manifest['artifacts'])} artifacts re-hashed, "
              f"row counts stable.")
        return 0

    manifest = build_manifest(
        REPO_ROOT / args.results_dir, REPO_ROOT / args.figs_dir, REPO_ROOT / args.tables_dir,
        args.config,
        [c for c in args.commands.split(";") if c.strip()],
        {"passed": args.tests_passed, "total": args.tests_total, "pytest_exit": args.pytest_exit},
    )
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE manifest: {args.out} ({manifest['artifact_count']} artifacts, "
          f"{manifest['matrix_counts']['total_rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
