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


def test_dc_estimator_rejects_single_spike_and_tracks_sustained_change():
    estimator = module.MagneticFieldDCEstimator(time_constant_s=0.1, median_window=3)
    baseline = np.array([10.0, -5.0, 2.0])

    for timestamp in (0.0, 0.01, 0.02):
        estimate = estimator.update(baseline, timestamp)

    spike_estimate = estimator.update(np.array([1000.0, 1000.0, 1000.0]), 0.03)
    assert np.allclose(spike_estimate, baseline)

    changed_field = np.array([20.0, 5.0, -8.0])
    estimator.update(changed_field, 0.13)
    estimate = estimator.update(changed_field, 0.23)

    change = changed_field - baseline
    assert np.all((estimate - baseline) * change > 0)
    assert np.all(np.abs(estimate - baseline) < np.abs(change))


def test_directory_reconciliation_processes_file_missed_by_watcher(tmp_path):
    class Processor:
        last_file_num = -1

        @staticmethod
        def extract_file_number(filename):
            return int(filename.split("_")[1])

    class RecordingHandler(module.DataFileHandler):
        def __init__(self):
            self.processor = Processor()
            self.paths = []
            self.known_revisions = {}

        def handle_file_with_retry(self, filepath):
            self.paths.append(filepath)

    missed_file = tmp_path / "imgInfo_42_0x20240719.txt"
    missed_file.write_text("data")
    handler = RecordingHandler()

    handler.scan_directory(tmp_path)

    assert handler.paths == [str(missed_file)]


def test_directory_reconciliation_detects_overwritten_restarted_file(tmp_path):
    class Processor:
        last_file_num = 100

        @staticmethod
        def extract_file_number(filename):
            return int(filename.split("_")[1])

    class RecordingHandler(module.DataFileHandler):
        def __init__(self):
            self.processor = Processor()
            self.paths = []
            self.known_revisions = {}

        def handle_file_with_retry(self, filepath):
            self.paths.append(filepath)

    restarted_file = tmp_path / "imgInfo_00001_0x20240719.txt"
    restarted_file.write_text("old data")
    handler = RecordingHandler()
    handler.prime_directory(tmp_path)

    restarted_file.write_text("newer data")
    handler.scan_directory(tmp_path)

    assert handler.paths == [str(restarted_file)]


def test_successfully_processed_file_is_deleted(tmp_path, monkeypatch):
    telemetry_file = tmp_path / "imgInfo_00001_0x20240719.txt"
    telemetry_file.write_text(bytes(244).hex())
    processor = module.DataProcessor()
    monkeypatch.setattr(module, "DELETE_PROCESSED_FILES", True)

    processor.process_file(telemetry_file)

    assert not telemetry_file.exists()


def test_compute_magnetometer_weight_rejects_small_ac_to_dc_ratio():
    weight = module.compute_magnetometer_weight(
        np.array([15000.0, 0.0, 0.0]),
        np.array([200.0, 0.0, 0.0]),
    )

    assert weight == 0.0


def test_madgwick_zero_mag_weight_matches_imu_only_update():
    filter_with_zero_weight = module.MadgwickFilter(alpha=1.0, beta=0.3)
    filter_with_zero_weight.init()

    filter_without_mag = module.MadgwickFilter(alpha=1.0, beta=0.3)
    filter_without_mag.init()

    gyro = np.array([0.0, 0.0, np.radians(15.0)])
    accel = np.array([0.0, 0.0, -1.0])
    disturbed_mag = np.array([0.0, 1.0, 0.0])

    q_zero_weight, _, _ = filter_with_zero_weight.update(
        gyro,
        accel,
        disturbed_mag,
        current_time=0.01,
        mag_weight=0.0,
    )
    q_imu_only, _, _ = filter_without_mag.update(
        gyro,
        accel,
        None,
        current_time=0.01,
    )

    assert np.allclose(q_zero_weight, q_imu_only)