import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "capsule_orientation.py"


if "keyboard" not in sys.modules:
    sys.modules["keyboard"] = types.ModuleType("keyboard")

if "watchdog" not in sys.modules:
    sys.modules["watchdog"] = types.ModuleType("watchdog")
    sys.modules["watchdog.observers"] = types.ModuleType("watchdog.observers")
    sys.modules["watchdog.observers"].Observer = object
    sys.modules["watchdog.events"] = types.ModuleType("watchdog.events")
    sys.modules["watchdog.events"].FileSystemEventHandler = object


spec = importlib.util.spec_from_file_location("capsule_orientation", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_lockin_rejects_stationary_opposite_branch():
    lockin = module.LockInAmplifier(target_freq=1.0, alpha=1.0, hp_alpha=0.0)
    lockin.last_output_vector = np.array([70.0, 0.0, 0.0])
    lockin.last_sign_resolution_time = 1.0

    resolved = lockin.resolve_polarity(
        np.array([-70.0, 0.0, 0.0]),
        current_time=1.01,
        predicted_field=np.array([-1.0, 0.0, 0.0]),
        gyro_dps=np.zeros(3),
    )

    assert np.allclose(resolved, [70.0, 0.0, 0.0])


def test_lockin_allows_gyro_supported_half_turn():
    lockin = module.LockInAmplifier(target_freq=1.0, alpha=1.0, hp_alpha=0.0)
    lockin.last_output_vector = np.array([70.0, 0.0, 0.0])
    lockin.last_sign_resolution_time = 1.0

    resolved = lockin.resolve_polarity(
        np.array([-70.0, 0.0, 0.0]),
        current_time=1.2,
        predicted_field=np.array([-1.0, 0.0, 0.0]),
        gyro_dps=np.array([0.0, 0.0, 720.0]),
    )

    assert np.allclose(resolved, [-70.0, 0.0, 0.0])