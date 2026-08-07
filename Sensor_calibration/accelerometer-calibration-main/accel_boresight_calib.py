"""Estimate the accelerometer-to-capsule boresight from a six-position tumble log.

The log must contain five comma-separated X,Y,Z samples for each pose, in this
exact order: +X, -X, +Y, -Y, +Z, -Z. The samples must be in the same frame and
already use the same bias/scale calibration as ``DataProcessor``.
"""

from pathlib import Path

import numpy as np


SAMPLES_PER_POSITION = 5
POSE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
EXPECTED_GRAVITY_VECTORS = np.array(
    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
)

def load_tumble_log(file_path):
    """Load exactly six consecutive five-sample tumble positions."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                values = [float(value.strip()) for value in line.split(",")]
            except ValueError as error:
                raise ValueError(f"Invalid numeric data on line {line_number}: {line}") from error
            if len(values) != 3:
                raise ValueError(f"Expected three values on line {line_number}: {line}")
            samples.append(values)

    samples = np.asarray(samples, dtype=float)
    expected_count = len(POSE_NAMES) * SAMPLES_PER_POSITION
    if samples.shape != (expected_count, 3):
        raise ValueError(
            f"Expected {expected_count} samples ({len(POSE_NAMES)} poses x "
            f"{SAMPLES_PER_POSITION} samples), but found {len(samples)}."
        )
    return samples


def estimate_boresight_rotation(calibrated_samples):
    """Fit the proper rotation mapping measured gravity directions to the six axes."""
    expected_count = len(POSE_NAMES) * SAMPLES_PER_POSITION
    if calibrated_samples.shape != (expected_count, 3):
        raise ValueError(f"Expected calibrated samples with shape ({expected_count}, 3).")

    pose_means = calibrated_samples.reshape(len(POSE_NAMES), SAMPLES_PER_POSITION, 3).mean(axis=1)
    pose_norms = np.linalg.norm(pose_means, axis=1)
    if np.any(pose_norms == 0):
        raise ValueError("A tumble position has zero gravity magnitude.")
    measured_directions = pose_means / pose_norms[:, np.newaxis]

    covariance = EXPECTED_GRAVITY_VECTORS.T @ measured_directions
    left_vectors, _, right_vectors_t = np.linalg.svd(covariance)
    boresight_rotation = left_vectors @ right_vectors_t
    if np.linalg.det(boresight_rotation) < 0:
        left_vectors[:, -1] *= -1
        boresight_rotation = left_vectors @ right_vectors_t

    corrected_directions = (boresight_rotation @ measured_directions.T).T
    angular_errors_deg = np.degrees(np.arccos(np.clip(
        np.sum(corrected_directions * EXPECTED_GRAVITY_VECTORS, axis=1), -1.0, 1.0
    )))
    return boresight_rotation, pose_means, angular_errors_deg


def rotation_to_euler_zyx(rotation):
    """Return roll, pitch, yaw degrees for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    pitch_rad = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if np.isclose(abs(rotation[2, 0]), 1.0):
        roll_rad = 0.0
        yaw_rad = np.arctan2(-rotation[0, 1], rotation[1, 1])
    else:
        roll_rad = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw_rad = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.degrees([roll_rad, pitch_rad, yaw_rad])


def print_report(boresight_rotation, pose_means, angular_errors_deg):
    """Print a paste-ready matrix and fit diagnostics."""
    print("=" * 68)
    print("ACCELEROMETER SIX-POSITION BORESIGHT CALIBRATION")
    print("=" * 68)
    for pose_name, pose_mean, error_deg in zip(POSE_NAMES, pose_means, angular_errors_deg):
        print(
            f"{pose_name:>2} mean: [{pose_mean[0]:+.6f}, {pose_mean[1]:+.6f}, "
            f"{pose_mean[2]:+.6f}] g   residual: {error_deg:.3f} deg"
        )
    print(f"Maximum residual: {np.max(angular_errors_deg):.3f} deg")
    print("\nPaste this over ACCEL_BORESIGHT_R in capsule_orientation.py:")
    print("ACCEL_BORESIGHT_R = np.array([")
    for row in boresight_rotation:
        print(f"    [{row[0]: .9f}, {row[1]: .9f}, {row[2]: .9f}],")
    print("])")
    roll_deg, pitch_deg, yaw_deg = rotation_to_euler_zyx(boresight_rotation)
    print("\nBoresight offset angles (degrees, ZYX order):")
    print(f"  Roll:  {roll_deg:+.3f}")
    print(f"  Pitch: {pitch_deg:+.3f}")
    print(f"  Yaw:   {yaw_deg:+.3f}")
    print("=" * 68)


def calibrate_tumble_log(file_path):
    """Load a six-position log, estimate boresight, and print its report."""
    calibrated_samples = load_tumble_log(file_path)
    result = estimate_boresight_rotation(calibrated_samples)
    print_report(*result)
    return result


if __name__ == "__main__":
    log_file = Path(__file__).with_name("accelerometer_boresight_log.txt")
    try:
        calibrate_tumble_log(log_file)
    except (FileNotFoundError, ValueError) as error:
        print(f"Calibration failed: {error}")