"""
Configuration Loader for VANET Simulation and Research Experiments.
Provides unified, authoritative access to configs/default_config.yaml.
"""

import os
import yaml
import copy

_DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "default_config.yaml")
)

def load_config(config_path=None):
    """
    Loads and returns configuration dictionary from default_config.yaml.
    """
    path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        # Fallback search if called from subdirectories
        alt_path = os.path.abspath(os.path.join(os.getcwd(), "configs", "default_config.yaml"))
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_effective_config_snapshot(overrides=None):
    """
    Returns deep-copied configuration snapshot with optional key-value overrides.
    """
    cfg = copy.deepcopy(load_config())
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg
