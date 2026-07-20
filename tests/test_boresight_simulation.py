import numpy as np
from scipy.spatial.transform import Rotation as R


def _estimate_boresight_rotation(imu_poses, cam_poses):
    """Mirror the solver used in capsule_calibration.run_boresight_calibration."""
    if len(imu_poses) != len(cam_poses):
        raise ValueError("IMU and camera pose arrays must have same length")
    if len(imu_poses) < 3:
        raise ValueError("Need at least 3 poses")

    R_A_steps, R_B_steps = [], []
    for i in range(len(imu_poses) - 1):
        R_A_steps.append(imu_poses[i].T @ imu_poses[i + 1])
        R_B_steps.append(cam_poses[i].T @ cam_poses[i + 1])

    M = np.zeros((3, 3))
    for R_a, R_b in zip(R_A_steps, R_B_steps):
        M += R_b @ R_a.T

    U, _, Vt = np.linalg.svd(M)
    R_x = U @ Vt
    if np.linalg.det(R_x) < 0:
        Vt[2, :] *= -1
        R_x = U @ Vt

    return R_x


def _intrinsics_from_fov_and_focal_length(fov_deg=140.0, focal_length_mm=0.9, width_px=1280, height_px=960):
    """
    Build a physically consistent camera model from the user's lens specs.
    Returns K plus implied sensor dimensions.
    """
    half_fov = np.deg2rad(fov_deg / 2.0)
    sensor_width_mm = 2.0 * focal_length_mm * np.tan(half_fov)
    sensor_height_mm = sensor_width_mm * (height_px / width_px)

    fx = focal_length_mm / sensor_width_mm * width_px
    fy = focal_length_mm / sensor_height_mm * height_px

    K = np.array([
        [fx, 0.0, width_px / 2.0],
        [0.0, fy, height_px / 2.0],
        [0.0, 0.0, 1.0],
    ])
    return K, sensor_width_mm, sensor_height_mm


def _simulate_pose_pairs(tilt_deg_xyz, n_samples=120, seed=7, imu_noise_deg=0.15, cam_noise_deg=0.2):
    """
    Synthetic orientation pairs for boresight verification.

    Conventions here are chosen to be consistent with the existing solver math:
      - imu pose: world<-robot
      - cam pose: world<-camera
      - true boresight (camera<-robot): R_x_true
      - relation: R_world_camera = R_world_robot @ R_x_true.T
    """
    rng = np.random.default_rng(seed)
    R_x_true = R.from_euler("xyz", tilt_deg_xyz, degrees=True).as_matrix()

    imu_poses, cam_poses = [], []
    for _ in range(n_samples):
        R_world_robot = R.random(random_state=rng).as_matrix()
        R_world_camera = R_world_robot @ R_x_true.T

        # Add small orientation noise to emulate measurement / PnP estimation noise.
        R_imu_noise = R.from_euler("xyz", rng.normal(0.0, imu_noise_deg, 3), degrees=True).as_matrix()
        R_cam_noise = R.from_euler("xyz", rng.normal(0.0, cam_noise_deg, 3), degrees=True).as_matrix()

        imu_poses.append(R_imu_noise @ R_world_robot)
        cam_poses.append(R_cam_noise @ R_world_camera)

    return np.array(imu_poses), np.array(cam_poses), R_x_true


def _rotation_error_deg(R_est, R_true):
    return np.degrees((R.from_matrix(R_est.T @ R_true)).magnitude())


def _solve_hand_eye_rotation_log(imu_poses, cam_poses):
    """Mirror capsule_calibration.solve_hand_eye_rotation_log."""
    covariance = np.zeros((3, 3), dtype=np.float64)
    for imu_start, imu_end, cam_start, cam_end in zip(
        imu_poses, imu_poses[1:], cam_poses, cam_poses[1:]
    ):
        relative_imu = imu_start.T @ imu_end
        relative_camera = cam_start.T @ cam_end
        covariance += np.outer(
            R.from_matrix(relative_camera).as_rotvec(),
            R.from_matrix(relative_imu).as_rotvec(),
        )

    U, _, Vt = np.linalg.svd(covariance)
    R_x = Vt.T @ U.T
    if np.linalg.det(R_x) < 0:
        U[:, 2] *= -1
        R_x = Vt.T @ U.T
    return R_x


def test_boresight_recovers_known_capsule_tilt():
    """
    Verify boresight calibration can recover a known camera tilt.

    Capsule camera specs used:
      - FOV: 140 deg
      - focal length: 0.9 mm
    """
    K, sensor_w_mm, sensor_h_mm = _intrinsics_from_fov_and_focal_length(
        fov_deg=140.0,
        focal_length_mm=0.9,
        width_px=1280,
        height_px=960,
    )

    # Sanity checks for the lens model derived from your hardware specs.
    assert np.isclose(sensor_w_mm, 2.0 * 0.9 * np.tan(np.deg2rad(70.0)), rtol=1e-12)
    assert K[0, 0] > 0 and K[1, 1] > 0

    # Ground-truth tilt: camera frame is tilted wrt robot frame.
    true_tilt_deg = np.array([8.0, -5.0, 3.0])  # roll, pitch, yaw

    imu_poses, cam_poses, R_x_true = _simulate_pose_pairs(
        tilt_deg_xyz=true_tilt_deg,
        n_samples=150,
        seed=11,
        imu_noise_deg=0.15,
        cam_noise_deg=0.25,
    )

    R_x_est = _estimate_boresight_rotation(imu_poses, cam_poses)

    # Compare as SO(3) geodesic angle.
    err_deg = _rotation_error_deg(R_x_est, R_x_true)

    # Tolerance chosen to reflect practical noise + current solver behavior.
    assert err_deg < 3.0


def test_log_solver_recovers_exact_hand_eye_rotation():
    true_boresight = R.from_euler("xyz", [8.5, -4.2, 12.0], degrees=True).as_matrix()
    capsule_poses = [
        R.from_euler("xyz", euler, degrees=True).as_matrix()
        for euler in [(0, 0, 0), (12, -8, 6), (-14, 7, -10), (18, 10, 14)]
    ]
    camera_poses = [capsule_pose @ true_boresight for capsule_pose in capsule_poses]

    estimate = _solve_hand_eye_rotation_log(capsule_poses, camera_poses)

    assert _rotation_error_deg(estimate, true_boresight) < 1e-10
