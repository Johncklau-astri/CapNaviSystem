import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'aprilTagLocalization.py'

if 'pupil_apriltags' not in sys.modules:
    fake_detector = types.SimpleNamespace(Detector=object)
    sys.modules['pupil_apriltags'] = fake_detector

if 'cv2' not in sys.modules:
    sys.modules['cv2'] = types.ModuleType('cv2')

if 'serial' not in sys.modules:
    sys.modules['serial'] = types.ModuleType('serial')
    sys.modules['serial'].Serial = object
    sys.modules['serial'].SerialException = Exception

if 'matplotlib' not in sys.modules:
    class DummyAxes:
        def plot(self, *args, **kwargs):
            return []
        def set_xlabel(self, *args, **kwargs):
            pass
        def set_ylabel(self, *args, **kwargs):
            pass
        def set_ylim(self, *args, **kwargs):
            pass
        def set_xlim(self, *args, **kwargs):
            pass
        def legend(self, *args, **kwargs):
            pass
        def grid(self, *args, **kwargs):
            pass
        def relim(self, *args, **kwargs):
            pass
        def autoscale_view(self, *args, **kwargs):
            pass

    class DummyFigure:
        def canvas(self):
            return None

    matplotlib_module = types.ModuleType('matplotlib')
    pyplot_module = types.ModuleType('matplotlib.pyplot')
    pyplot_module.ion = lambda: None
    pyplot_module.subplots = lambda *args, **kwargs: (DummyFigure(), DummyAxes())
    matplotlib_module.pyplot = pyplot_module
    sys.modules['matplotlib'] = matplotlib_module
    sys.modules['matplotlib.pyplot'] = pyplot_module

spec = importlib.util.spec_from_file_location('aprilTagLocalization', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_default_telemetry_source_is_capsule_orientation(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda prompt: pytest.fail('Unexpected input prompt'))
    module.select_telemetry_source()

    magnetometer, _ = module.decode_bt_telemetry({'magX': 1.0, 'magY': 2.0, 'magZ': 3.0})

    assert module.telemetry_source == module.TELEMETRY_SOURCE_CAPSULE_ORIENTATION
    np.testing.assert_array_equal(magnetometer, [1.0, 2.0, 3.0])


def test_bt_angle_est_telemetry_remains_selectable():
    try:
        module.select_telemetry_source(module.TELEMETRY_SOURCE_BT_ANGLE_EST)
        magnetometer, _ = module.decode_bt_telemetry({'magX': 1.0, 'magY': 2.0, 'magZ': 3.0})
        np.testing.assert_array_equal(magnetometer, [-2.0, -1.0, -3.0])
    finally:
        module.select_telemetry_source()


def test_compute_bt_only_quaternion_returns_expected_shape():
    motor_data = {'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0}
    bt_quat = [1.0, 0.0, 0.0, 0.0]

    q = module.compute_bt_only_quaternion(motor_data, bt_quat)

    assert q is not None
    assert len(q) == 4


def test_position_latency_plotter_records_motor_and_estimate():
    plotter = module.PositionLatencyPlotter(maxlen=10, refresh_s=0.01)
    plotter.push({'x': 1.0, 'y': 2.0, 'z': 3.0}, [4.0, 5.0, 6.0])

    assert len(plotter.t_buf) == 1
    assert plotter.motor_x_buf[-1] == 1.0
    assert plotter.motor_y_buf[-1] == 2.0
    assert plotter.motor_z_buf[-1] == 3.0
    assert plotter.est_x_buf[-1] == 4.0
    assert plotter.est_y_buf[-1] == 5.0
    assert plotter.est_z_buf[-1] == 6.0

    plotter.stop()
