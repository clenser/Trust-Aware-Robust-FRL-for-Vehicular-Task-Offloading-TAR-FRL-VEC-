"""Microscopic traffic simulation adapter using Eclipse SUMO and TraCI.

Provides:
- Automatic detection of SUMO binaries and SUMO_HOME across standard and custom paths.
- Clean fallback handling when SUMO or TraCI is unavailable.
- Programmatic scenario execution with configurable density, seed, and step length.
- TraCI vehicle telemetry retrieval (positions, speeds, lanes, headings, accelerations).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Known default candidate locations for portable and system SUMO on Windows
CANDIDATE_SUMO_HOMES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "sumo-1.28.0" / "sumo-win64-1.28.0",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Eclipse" / "Sumo",
    Path("C:/Program Files/Eclipse/Sumo"),
    Path("C:/Program Files (x86)/Eclipse/Sumo"),
    Path("C:/sumo"),
]


class SumoNotInstalledError(RuntimeError):
    """Raised when SUMO or TraCI is not installed or available on the host system."""


def get_sumo_home() -> Optional[Path]:
    """Resolve SUMO_HOME directory from environment, PATH, or standard installation paths."""
    env_home = os.environ.get("SUMO_HOME")
    if env_home and Path(env_home).is_dir():
        return Path(env_home).resolve()

    for cand in CANDIDATE_SUMO_HOMES:
        if cand.is_dir() and (cand / "bin" / "sumo.exe").is_file():
            return cand.resolve()

    # Check if 'sumo' is on system PATH
    which_sumo = shutil.which("sumo")
    if which_sumo:
        sumo_bin = Path(which_sumo).resolve()
        if sumo_bin.parent.name.lower() == "bin":
            return sumo_bin.parent.parent

    return None


def find_sumo_binary(gui: bool = False) -> Optional[str]:
    """Return path to sumo or sumo-gui executable."""
    bin_name = "sumo-gui.exe" if gui and sys.platform == "win32" else "sumo-gui" if gui else "sumo.exe" if sys.platform == "win32" else "sumo"

    home = get_sumo_home()
    if home:
        exe_path = home / "bin" / bin_name
        if exe_path.is_file():
            return str(exe_path)

    which_path = shutil.which(bin_name)
    if which_path:
        return which_path

    # Fallback to non-gui if gui requested but not found
    if gui:
        return find_sumo_binary(gui=False)

    return None


def is_sumo_available() -> bool:
    """Check whether both SUMO binary and python traci library are available."""
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return find_sumo_binary() is not None


def get_sumo_status() -> dict[str, Any]:
    """Return system readiness dictionary for SUMO and TraCI."""
    home = get_sumo_home()
    exe = find_sumo_binary(gui=False)
    gui_exe = find_sumo_binary(gui=True)
    
    traci_installed = False
    traci_version = None
    try:
        import traci
        traci_installed = True
        traci_version = getattr(traci, "__version__", "installed")
    except ImportError:
        pass

    available = (exe is not None and traci_installed)

    instructions = (
        "SUMO is ready for microscopic traffic simulation."
        if available else
        "To enable SUMO: run 'pip install traci sumolib' and install Eclipse SUMO from "
        "https://sumo.dlr.de/docs/Downloads.php or set SUMO_HOME to your installation folder."
    )

    return {
        "available": available,
        "sumo_home": str(home) if home else None,
        "sumo_binary": exe,
        "sumo_gui_binary": gui_exe,
        "traci_installed": traci_installed,
        "traci_version": traci_version,
        "instructions": instructions,
    }


@dataclass
class SumoVehicleState:
    """Rich vehicle state extracted from SUMO via TraCI at a discrete timestep."""
    vid: str
    int_id: int
    x_m: float
    y_m: float
    lane_idx: int
    speed_mps: float
    accel_mps2: float
    heading_deg: float
    length_m: float
    width_m: float
    veh_type: str
    vclass: str
    road_id: str


@dataclass
class SumoScenarioConfig:
    """Configuration parameters for microscopic highway simulation."""
    scenario_dir: Path = field(default_factory=lambda: REPO_ROOT / "environment" / "sumo")
    road_length_m: float = 1200.0
    num_lanes: int = 3
    lane_width_m: float = 3.6
    speed_limit_mps: float = 33.33
    density: str = "medium"  # "low" (600 veh/h), "medium" (1550 veh/h), "high" (2500 veh/h)
    step_length_s: float = 0.1
    seed: int = 42
    gui: bool = False


class SumoTrafficSimulator:
    """High-level controller for running microscopic highway traffic via TraCI."""

    def __init__(self, config: Optional[SumoScenarioConfig] = None) -> None:
        self.cfg = config or SumoScenarioConfig()
        self.home = get_sumo_home()
        if self.home:
            os.environ["SUMO_HOME"] = str(self.home)
            bin_dir = str(self.home / "bin")
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

        self._traci: Any = None
        self._is_running: bool = False
        self._current_time_s: float = 0.0
        self._vid_map: dict[str, int] = {}
        self._next_int_id: int = 0
        self._label: str = f"tar_frl_sumo_{os.getpid()}_{id(self)}"

    def _ensure_scenario_files(self) -> Path:
        """Verify scenario files exist or generate them."""
        scen_dir = self.cfg.scenario_dir
        scen_dir.mkdir(parents=True, exist_ok=True)
        sumocfg = scen_dir / "highway.sumocfg"

        # Generate custom routes if density differs
        rou_file = scen_dir / f"highway_{self.cfg.density}.rou.xml"
        if not rou_file.is_file():
            self._write_route_file(rou_file, self.cfg.density)

        # Write or update sumocfg pointing to density route
        cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="{rou_file.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="{self.cfg.step_length_s}"/>
    </time>
    <processing>
        <collision.action value="none"/>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>"""
        with open(sumocfg, "w", encoding="utf-8") as f:
            f.write(cfg_content)

        return sumocfg

    def _write_route_file(self, path: Path, density: str) -> None:
        """Write density-calibrated route XML file."""
        flow_rates = {
            "low": (400, 250, 100),       # ~750 veh/hr
            "medium": (750, 550, 250),    # ~1550 veh/hr
            "high": (1200, 900, 400),     # ~2500 veh/hr
        }
        sedan_rate, suv_rate, van_rate = flow_rates.get(density, flow_rates["medium"])

        content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <vType id="passenger_sedan" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="4.60" width="1.85" minGap="2.5" maxSpeed="33.33" lcStrategic="1.0" lcCooperative="1.0" lcSpeedGain="1.0"/>
    <vType id="passenger_suv" vClass="passenger" accel="2.4" decel="4.2" sigma="0.5" length="4.85" width="1.95" minGap="2.5" maxSpeed="30.00" lcStrategic="1.0" lcCooperative="1.0" lcSpeedGain="1.0"/>
    <vType id="commercial_van" vClass="delivery" accel="2.0" decel="4.0" sigma="0.5" length="5.20" width="2.00" minGap="3.0" maxSpeed="28.00" lcStrategic="1.0" lcCooperative="0.8" lcSpeedGain="0.8"/>

    <route id="r_0" edges="highway"/>

    <flow id="flow_sedan" type="passenger_sedan" route="r_0" begin="0" end="3600" vehsPerHour="{sedan_rate}" departLane="random" departSpeed="max"/>
    <flow id="flow_suv" type="passenger_suv" route="r_0" begin="0" end="3600" vehsPerHour="{suv_rate}" departLane="random" departSpeed="max"/>
    <flow id="flow_van" type="commercial_van" route="r_0" begin="0" end="3600" vehsPerHour="{van_rate}" departLane="0" departSpeed="25.0"/>
</routes>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def start(self, gui: Optional[bool] = None, seed: Optional[int] = None) -> None:
        """Start SUMO process and connect via TraCI."""
        if not is_sumo_available():
            raise SumoNotInstalledError(
                "Cannot start SUMO simulation: SUMO binary or traci python package not found. "
                "See get_sumo_status() for installation instructions."
            )

        import traci

        use_gui = self.cfg.gui if gui is None else gui
        sim_seed = self.cfg.seed if seed is None else seed
        sumo_bin = find_sumo_binary(gui=use_gui)
        if not sumo_bin:
            raise SumoNotInstalledError(f"No executable found for gui={use_gui}")

        cfg_file = self._ensure_scenario_files()

        cmd = [
            sumo_bin,
            "-c", str(cfg_file),
            "--seed", str(sim_seed),
            "--step-length", str(self.cfg.step_length_s),
            "--no-warnings",
            "--no-step-log",
        ]
        if use_gui:
            cmd.extend(["--start", "--quit-on-end"])

        # Start TraCI session with unique connection label
        traci.start(cmd, label=self._label)
        self._traci = traci.getConnection(self._label)
        self._is_running = True
        self._current_time_s = 0.0
        self._vid_map.clear()
        self._next_int_id = 0

    def step(self, num_steps: int = 1) -> float:
        """Advance the microscopic simulation by num_steps."""
        if not self._is_running or not self._traci:
            raise RuntimeError("SUMO simulation is not running. Call start() first.")

        for _ in range(num_steps):
            self._traci.simulationStep()

        self._current_time_s = float(self._traci.simulation.getTime())
        return self._current_time_s

    def advance_time(self, target_time_s: float) -> float:
        """Advance simulation until target_time_s is reached or exceeded."""
        if not self._is_running or not self._traci:
            raise RuntimeError("SUMO simulation is not running. Call start() first.")

        while self._current_time_s < target_time_s:
            self._traci.simulationStep()
            self._current_time_s = float(self._traci.simulation.getTime())

        return self._current_time_s

    def get_active_vehicles(self) -> list[SumoVehicleState]:
        """Fetch full telemetry for all vehicles currently driving on the road network."""
        if not self._is_running or not self._traci:
            return []

        active: list[SumoVehicleState] = []
        veh_ids = self._traci.vehicle.getIDList()

        for vid in veh_ids:
            if vid not in self._vid_map:
                self._vid_map[vid] = self._next_int_id
                self._next_int_id += 1

            int_id = self._vid_map[vid]
            x, y = self._traci.vehicle.getPosition(vid)
            lane_idx = self._traci.vehicle.getLaneIndex(vid)
            spd = float(self._traci.vehicle.getSpeed(vid))
            accel = float(self._traci.vehicle.getAcceleration(vid))
            heading = float(self._traci.vehicle.getAngle(vid))
            vtype = str(self._traci.vehicle.getTypeID(vid))
            length = float(self._traci.vehicle.getLength(vid))
            width = float(self._traci.vehicle.getWidth(vid))
            vclass = str(self._traci.vehicle.getVehicleClass(vid))
            road_id = str(self._traci.vehicle.getRoadID(vid))

            active.append(SumoVehicleState(
                vid=vid,
                int_id=int_id,
                x_m=float(x),
                y_m=float(y),
                lane_idx=int(lane_idx),
                speed_mps=spd,
                accel_mps2=accel,
                heading_deg=heading,
                length_m=length,
                width_m=width,
                veh_type=vtype,
                vclass=vclass,
                road_id=road_id,
            ))

        return active

    def get_departed_and_arrived(self) -> tuple[list[str], list[str]]:
        """Return lists of vehicle IDs that departed (entered) and arrived (exited) this step."""
        if not self._is_running or not self._traci:
            return [], []
        departed = list(self._traci.simulation.getDepartedIDList())
        arrived = list(self._traci.simulation.getArrivedIDList())
        return departed, arrived

    @property
    def current_time_s(self) -> float:
        return self._current_time_s

    @property
    def is_running(self) -> bool:
        return self._is_running

    def close(self) -> None:
        """Safely close TraCI connection and shut down SUMO."""
        if self._is_running and self._traci:
            try:
                self._traci.close()
            except Exception:
                pass
            finally:
                self._traci = None
                self._is_running = False

    def __enter__(self) -> "SumoTrafficSimulator":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
