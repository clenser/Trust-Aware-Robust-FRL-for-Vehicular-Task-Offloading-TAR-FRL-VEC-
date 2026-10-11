"""Phase 0 skeleton tests: repo structure, configs, and honesty guards.

No simulation/RL/federated/trust logic is tested here — none exists yet.
These tests lock the Phase 0 skeleton so later phases can build on it.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_required_top_level_files_exist():
    for name in ["AGENTS.md", "PHASE_PROMPTS.md", "README.md", "requirements.txt", "pyproject.toml"]:
        assert (ROOT / name).is_file(), f"missing required file: {name}"


def test_config_skeleton_loads():
    for name in ["smoke.yaml", "default.yaml", "experiments.yaml"]:
        path = ROOT / "configs" / name
        assert path.is_file(), f"missing config: {name}"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)


def test_smoke_config_has_seed_and_scenario():
    data = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    assert isinstance(data["seed"], int)
    scenario = data["scenario"]
    assert scenario["num_vehicles"] > 0
    assert scenario["malicious_client_ratio"] == 0.0


def test_no_premature_algorithm_code():
    """Phase 0 must not contain simulation/RL/FL/trust implementations."""
    premature_markers = ["class DQN", "FedAvg", "T_final", "replay_buffer", "SUMO"]
    hits = []
    self_file = Path(__file__).name
    for path in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        if path.name == self_file:
            continue  # do not match the marker list inside this guard test itself
        text = path.read_text(encoding="utf-8")
        for marker in premature_markers:
            if marker in text:
                hits.append(f"{path.name}: {marker}")
    assert not hits, f"premature implementation markers found: {hits}"


def test_phase_placeholders_are_not_claimed_results():
    """Docs must not claim measured outcomes in Phase 0."""
    banned = ["accuracy 99%", "convergence achieved", "outperforms all baselines"]
    for name in ["README.md", "docs/PLAN.md", "docs/ASSUMPTIONS.md"]:
        text = (ROOT / name).read_text(encoding="utf-8").lower()
        for phrase in banned:
            assert phrase not in text, f"{name} contains unsupported claim: {phrase}"
