import csv
import queue
import threading
import time

import pytest
import numpy as np
from scipy.spatial.transform import Rotation

import capsule_orientation as module
from capsule_orientation import (
    LockInAmplifier,
    MadgwickFilter,
    MotorReferenceHistory,
    MotorYawPersistence,
    UncorrectedYawEstimator,
    compute_magnetometer_weight,
    motor_yaw_correction_diagnostic,
    quaternion_to_euler_from_array,
    quaternion_to_rotation_matrix,
    wrapped_angle_difference,
)


def diagnose(time, yaw, persistence=None, *, reference=None, computer_time=None, dt=0.01):
    if reference is None:
        reference = (time + 1000.0, 150.0, -1.0)
    if computer_time is None:
        computer_time = time + 1000.0
    return motor_yaw_correction_diagnostic(
        time, dt, -1000.0, computer_time, reference, time + 1000.0, yaw, persistence
    )


def test_persistent_wrapped_error_proposes_bounded_nudge():
    _, error, correction, status, persistence = diagnose(10.0, 170.0)
    assert error == pytest.approx(40.0)
    assert (correction, status) == (0.0, "waiting")

    for i in range(1, 21):
        _, error, correction, status, persistence = diagnose(
            10.0 + i * 0.05, 170.0, persistence, dt=0.05
        )
    assert error == pytest.approx(40.0)
    assert correction == pytest.approx(0.5)
    assert status == "proposed"


def test_rejects_stale_or_unaligned_reference():
    persistence = MotorYawPersistence(1.0, 10.99)
    stale = diagnose(11.0, 170.0, persistence, computer_time=1012.0)
    assert stale[3:] == ("unaligned_or_stale", None)
    unaligned = diagnose(11.0, 170.0, persistence, reference=(1010.0, 150.0, -1.0))
    assert unaligned[3:] == ("unaligned_or_stale", None)


@pytest.mark.parametrize("yaw", [60.0, -60.0, 179.0, -180.0])
def test_emergency_snap_proposes_full_wrapped_error(yaw):
    reference = (1010.0, 0.0, -1.0)
    _, error, correction, status, persistence = diagnose(
        10.0, yaw, reference=reference
    )
    assert error == pytest.approx(wrapped_angle_difference(0.0, yaw))
    assert correction == pytest.approx(error)
    assert (status, persistence) == ("snap_proposed", None)


def test_below_snap_threshold_retains_rate_limited_correction():
    _, _, correction, status, _ = diagnose(
        10.0, 59.9, MotorYawPersistence(1.0, 9.99),
        reference=(1010.0, 0.0, -1.0),
    )
    assert status == "proposed"
    assert correction == pytest.approx(-0.1)


def test_snap_requires_valid_sample_and_motor_reference():
    persistence = MotorYawPersistence(1.0, 9.99)
    reference = (1010.0, 0.0, -1.0)
    for result in (
        diagnose(10.0, 90.0, persistence, reference=reference, dt=0.5),
        diagnose(10.0, 90.0, persistence, reference=reference, computer_time=1011.0),
        diagnose(10.0, 90.0, persistence, reference=(1009.0, 0.0, -1.0)),
    ):
        assert result[2] == 0.0
        assert result[3] in ("invalid_sample", "unaligned_or_stale")


def test_short_gap_preserves_only_observed_waiting_time():
    persistence = MotorYawPersistence(0.5, 10.0)
    _, _, correction, status, persistence = diagnose(10.5, 170.0, persistence, dt=0.5)
    assert (correction, status) == (0.0, "invalid_sample")
    assert persistence.observed_seconds == pytest.approx(0.5)

    for i in range(1, 11):
        _, _, correction, status, persistence = diagnose(
            10.5 + i * 0.05, 170.0, persistence, dt=0.05
        )
        if i < 10:
            assert (correction, status) == (0.0, "waiting")
    assert status == "proposed"
    assert correction == pytest.approx(0.5)


def test_gap_limit_and_invalid_conditions_reset_persistence():
    persistence = MotorYawPersistence(0.5, 10.0)
    at_limit = diagnose(11.0, 170.0, persistence, dt=1.0)
    assert at_limit[3:] == ("invalid_sample", persistence)
    resumed = diagnose(11.01, 170.0, at_limit[4], dt=0.01)
    assert resumed[3] == "waiting"
    assert resumed[4].observed_seconds == pytest.approx(0.51)
    assert diagnose(11.01, 170.0, persistence, dt=1.01)[3:] == (
        "invalid_sample", None
    )
    first_gap = diagnose(10.6, 170.0, persistence, dt=0.6)
    assert diagnose(11.2, 170.0, first_gap[4], dt=0.6)[3:] == (
        "invalid_sample", None
    )
    assert diagnose(10.01, -145.0, persistence)[3:] == ("below_threshold", None)
    assert diagnose(10.01, 170.0, persistence, dt=float("nan"))[3:] == (
        "invalid_sample", None
    )


def test_armed_correction_resumes_only_with_valid_reference_and_error():
    persistence = MotorYawPersistence(1.0, 10.0)
    gap = diagnose(10.8, 170.0, persistence, dt=0.8)
    assert gap[2:4] == (0.0, "invalid_sample")

    resumed = diagnose(10.81, 170.0, gap[4])
    assert resumed[3] == "proposed"
    assert resumed[2] == pytest.approx(0.1)
    assert diagnose(10.81, 170.0, gap[4], computer_time=1012.0)[3:] == (
        "unaligned_or_stale", None
    )
    assert diagnose(10.81, -145.0, gap[4])[3:] == ("below_threshold", None)


def test_negative_error_proposes_negative_nudge():
    _, error, correction, status, _ = diagnose(
        11.0, -100.0, MotorYawPersistence(1.0, 10.99)
    )
    assert error == pytest.approx(-50.0)
    assert correction == pytest.approx(-0.1)
    assert status == "proposed"


def test_historical_sample_accepted_if_latest_reference_is_fresh():
    result = motor_yaw_correction_diagnostic(
        10.0, 0.01, -1000.0, 1012.0, (1010.02, 150.0, -1.0),
        1011.99, 170.0, MotorYawPersistence(1.0, 9.99),
    )
    assert result[0] == pytest.approx(0.02)
    assert result[2] == pytest.approx(0.1)
    assert result[3] == "proposed"


def test_history_matches_nearest_and_handles_partial_rows_and_new_run(tmp_path):
    path = tmp_path / "motor.csv"
    header = "timestamp_computer,rotary_output_angle_deg,expected_yaw_direction\n"
    path.write_text(header + "1000.0,20,-1\n1000.1,30", encoding="utf-8")
    history = MotorReferenceHistory(path)
    history.update()
    assert history.nearest(1000.08) == (1000.0, 20.0, -1.0)

    with path.open("a", encoding="utf-8") as output:
        output.write(",-1\n1000.2,40,-1\n")
    history.update()
    assert history.nearest(1000.08) == (1000.1, 30.0, -1.0)
    assert history.nearest(1000.08, 1000.0) == (1000.0, 20.0, -1.0)

    path.write_text(header + "2000.0,0,1\n", encoding="utf-8")
    history.update()
    assert history.nearest(1000.0) == (2000.0, 0.0, 1.0)
    assert len(history.references) == 1


def test_internal_correction_persists_and_preserves_tilt_across_yaw_wrap():
    filter = MadgwickFilter(alpha=0.4, beta=0.0)
    xyzw = Rotation.from_euler("xyz", [25.0, -20.0, 179.0], degrees=True).as_quat()
    filter.q = np.array([xyzw[3], *xyzw[:3]])
    filter.filtered_q = filter.q.copy()

    for _ in range(100):
        filter.apply_yaw_correction(0.05)

    for q in (filter.q, filter.filtered_q):
        roll, pitch, yaw = quaternion_to_euler_from_array(q)
        assert roll == pytest.approx(25.0)
        assert pitch == pytest.approx(-20.0)
        assert wrapped_angle_difference(yaw, 179.0) == pytest.approx(5.0)
        assert np.linalg.norm(q) == pytest.approx(1.0)

    filter.update(np.zeros(3), np.array([0.0, 0.0, -1.0]), None, current_time=1.0)
    assert wrapped_angle_difference(
        quaternion_to_euler_from_array(filter.q)[2], 179.0
    ) == pytest.approx(5.0)


def test_snap_sets_both_quaternion_yaws_and_preserves_tilt():
    filter = MadgwickFilter(alpha=0.4, beta=0.0)
    poses = [(25.0, -20.0, 179.0), (-10.0, 15.0, 160.0)]
    for name, pose in zip(("q", "filtered_q"), poses):
        xyzw = Rotation.from_euler("xyz", pose, degrees=True).as_quat()
        setattr(filter, name, np.array([xyzw[3], *xyzw[:3]]))

    filter.set_yaw(-150.0)

    for q, (expected_roll, expected_pitch, _) in zip(
        (filter.q, filter.filtered_q), poses
    ):
        roll, pitch, yaw = quaternion_to_euler_from_array(q)
        assert roll == pytest.approx(expected_roll)
        assert pitch == pytest.approx(expected_pitch)
        assert wrapped_angle_difference(yaw, -150.0) == pytest.approx(0.0)
        assert np.linalg.norm(q) == pytest.approx(1.0)


def test_persistent_motor_error_changes_filter_but_stale_reference_does_not():
    filter = MadgwickFilter(alpha=1.0, beta=0.0)
    xyzw = Rotation.from_euler("z", -100.0, degrees=True).as_quat()
    filter.q = np.array([xyzw[3], *xyzw[:3]])
    filter.filtered_q = filter.q.copy()

    _, _, delta, status, persistence = diagnose(10.0, -100.0)
    assert status == "waiting"
    assert delta == 0.0
    for i in range(1, 21):
        _, _, delta, status, persistence = diagnose(
            10.0 + i * 0.05, -100.0, persistence, dt=0.05
        )
    assert status == "proposed"
    filter.apply_yaw_correction(delta)
    yaw = quaternion_to_euler_from_array(filter.q)[2]
    assert yaw == pytest.approx(-100.5)

    _, _, delta, status, _ = diagnose(
        11.01, yaw, persistence, computer_time=1013.0
    )
    assert status == "unaligned_or_stale"
    assert delta == 0.0
    assert quaternion_to_euler_from_array(filter.q)[2] == pytest.approx(yaw)


def test_uncorrected_shadow_matches_unmodified_filter_then_diverges():
    filter = MadgwickFilter(alpha=1.0, beta=0.3)
    lockin = LockInAmplifier(target_freq=10.0, alpha=0.06, hp_alpha=0.06)
    shadow = UncorrectedYawEstimator(filter, lockin, None, 0.4)
    previous_q = None
    lowpass_q = None
    accel = np.array([0.0, 0.0, -1.0])
    gyro = np.array([0.0, 0.0, 2.0])

    for i in range(120):
        t = i * 0.01
        mag = np.array([20 * np.sin(2 * np.pi * 10 * t), 10.0, 3.0])
        predicted = (
            quaternion_to_rotation_matrix(previous_q).T @ np.array([1.0, 0.0, 0.0])
            if previous_q is not None else None
        )
        amplitudes, _, _, _ = lockin.update(
            mag, t, predicted_field=predicted, gyro_dps=gyro
        )
        if i == 20:
            assert lockin.calibrate_polarity(np.array([1.0, 1.0, 0.0]))
            amplitudes = lockin.last_relative_vector * lockin.polarity
            previous_q = filter.init().copy()
            lowpass_q = previous_q.copy()
        q, _, _ = filter.update(
            np.radians(gyro), accel, amplitudes, current_time=t,
            mag_weight=compute_magnetometer_weight(mag, amplitudes),
        )
        if i == 80:
            q = filter.apply_yaw_correction(20.0)
        if lowpass_q is not None and np.dot(q, lowpass_q) < 0:
            q = -q
        previous_q = q.copy()
        if lowpass_q is None:
            lowpass_q = q.copy()
        lowpass_q = filter.normalize_quaternion(0.4 * q + 0.6 * lowpass_q)
        actual = quaternion_to_euler_from_array(lowpass_q)[2]
        uncorrected = shadow.update(accel, gyro, mag, t, calibrated=(i == 20))
        if i < 80:
            assert wrapped_angle_difference(actual, uncorrected) == pytest.approx(0.0)

    assert abs(wrapped_angle_difference(actual, uncorrected)) > 10.0


def test_uncorrected_shadow_reports_failed_calibration():
    filter = MadgwickFilter(alpha=1.0, beta=0.3)
    lockin = LockInAmplifier(target_freq=10.0, alpha=0.06, hp_alpha=0.06)
    shadow = UncorrectedYawEstimator(filter, lockin, None, 0.4)
    yaw = shadow.update(np.array([0.0, 0.0, -1.0]), np.zeros(3), np.zeros(3), 1.0, True)
    assert np.isnan(yaw)
    assert not shadow.valid


def test_processing_logs_uncorrected_yaw_without_changing_live_output(
    tmp_path, monkeypatch
):
    stop = threading.Event()

    class StopWhenEmptyQueue(queue.Queue):
        def get(self, *args, **kwargs):
            if self.empty():
                stop.set()
                raise queue.Empty
            return super().get(*args, **kwargs)

    samples = StopWhenEmptyQueue()
    for i in range(20):
        t = i * 0.01
        samples.put((
            t, np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 2.0]),
            np.array([20 * np.sin(2 * np.pi * 10 * t), 10.0, 3.0]), 100.0,
        ))
    log_path = tmp_path / "orientation.csv"
    monkeypatch.setattr(module, "data_queue", samples)
    monkeypatch.setattr(module, "shutdown_event", stop)
    monkeypatch.setattr(module, "q_prev", None)
    monkeypatch.setattr(module, "calibration_requested", False)
    monkeypatch.setattr(module, "ORIENTATION_DEBUG_LOG_FILENAME", str(log_path))
    monkeypatch.setattr(module, "MOTOR_MOTION_DEBUG_LOG_FILENAME", str(tmp_path / "missing.csv"))
    monkeypatch.setattr(module, "send_to_tcp_clients", lambda *args, **kwargs: None)

    module.processing_thread(module.DataProcessor())

    with log_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 20
    for row in rows:
        assert None not in row
        assert float(row["yaw_without_motor_correction_deg"]) == pytest.approx(
            float(row["yaw_before_motor_guard_deg"])
        )


@pytest.mark.parametrize("enabled", [True, False])
def test_processing_applies_or_reports_emergency_snap(tmp_path, monkeypatch, enabled):
    stop = threading.Event()

    class StopWhenEmptyQueue(queue.Queue):
        def get(self, *args, **kwargs):
            if self.empty():
                stop.set()
                raise queue.Empty
            return super().get(*args, **kwargs)

    samples = StopWhenEmptyQueue()
    for t in (10.0, 10.01):
        samples.put((
            t, np.array([0.0, 0.0, -1.0]), np.zeros(3),
            np.array([20.0, 10.0, 3.0]), 100.0,
        ))
    now = time.time()
    reference = (now, 0.0, -1.0)
    motor_path = tmp_path / "motor.csv"
    motor_path.write_text(
        "timestamp_computer,rotary_output_angle_deg,expected_yaw_direction\n"
        f"{now},0,-1\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "orientation.csv"
    monkeypatch.setattr(module, "data_queue", samples)
    monkeypatch.setattr(module, "shutdown_event", stop)
    monkeypatch.setattr(module, "q_prev", None)
    monkeypatch.setattr(module, "calibration_requested", False)
    monkeypatch.setattr(module, "CAPSULE_CLOCK_OFFSET_SECONDS", 10.01 - now)
    monkeypatch.setattr(module, "MOTOR_INTERNAL_CORRECTION_ENABLED", enabled)
    monkeypatch.setattr(module, "ORIENTATION_DEBUG_LOG_FILENAME", str(log_path))
    monkeypatch.setattr(module, "MOTOR_MOTION_DEBUG_LOG_FILENAME", str(motor_path))
    monkeypatch.setattr(module, "read_latest_motor_motion_reference", lambda: reference)
    monkeypatch.setattr(module, "send_to_tcp_clients", lambda *args, **kwargs: None)
    processor = module.DataProcessor()
    xyzw = Rotation.from_euler("z", 90.0, degrees=True).as_quat()
    processor.filter.q = np.array([xyzw[3], *xyzw[:3]])
    processor.filter.filtered_q = processor.filter.q.copy()

    module.processing_thread(processor)

    with log_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    row = rows[-1]
    assert row["motor_diagnostic_status"] == (
        "snap_applied" if enabled else "snap_proposed"
    )
    assert float(row["motor_diagnostic_correction_deg"]) == pytest.approx(-90.0, abs=1.0)
    q_filter = np.array([float(row[f"q_filter_{axis}"]) for axis in "wxyz"])
    yaw = quaternion_to_euler_from_array(q_filter)[2]
    assert yaw == pytest.approx(0.0 if enabled else 90.0, abs=1.0)
