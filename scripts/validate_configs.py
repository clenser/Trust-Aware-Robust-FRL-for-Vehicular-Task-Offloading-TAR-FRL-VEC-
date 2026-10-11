"""Phase 0 helper: validate config skeleton files load and carry required keys."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REQUIRED_TOP_KEYS = {"project", "config_name", "seed"}
REQUIRED_SCENARIO_KEYS = {
    "num_vehicles",
    "num_edge_nodes",
    "mobility",
    "task_load",
    "malicious_client_ratio",
    "connectivity_range_m",
    "num_steps",
}


def validate_file(path: Path) -> list[str]:
    """Return a list of validation errors (empty means valid)."""
    errors: list[str] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # reported, never silent
        return [f"{path.name}: YAML parse error: {exc}"]
    if not isinstance(data, dict):
        return [f"{path.name}: top level must be a mapping"]
    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing and path.name != "experiments.yaml":
        errors.append(f"{path.name}: missing top-level keys: {sorted(missing)}")
    if path.name in {"smoke.yaml", "default.yaml"}:
        scenario = data.get("scenario")
        if not isinstance(scenario, dict):
            errors.append(f"{path.name}: 'scenario' must be a mapping")
        else:
            missing_s = REQUIRED_SCENARIO_KEYS - set(scenario.keys())
            if missing_s:
                errors.append(f"{path.name}: scenario missing keys: {sorted(missing_s)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate config YAML skeleton files.")
    parser.add_argument("--configs", default="configs", help="Directory holding YAML configs.")
    args = parser.parse_args(argv)
    config_dir = Path(args.configs)
    if not config_dir.is_dir():
        print(f"Config directory not found: {config_dir}", file=sys.stderr)
        return 2
    files = sorted(config_dir.glob("*.yaml"))
    if not files:
        print(f"No YAML configs in {config_dir}", file=sys.stderr)
        return 2
    all_errors: list[str] = []
    for path in files:
        errs = validate_file(path)
        if errs:
            all_errors.extend(errs)
            for err in errs:
                print(f"INVALID: {err}")
        else:
            print(f"OK: {path.name}")
    if all_errors:
        return 1
    print(f"All {len(files)} config(s) valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
