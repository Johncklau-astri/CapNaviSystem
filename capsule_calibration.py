import os
import glob
import time
import socket
import json
import struct
import threading
import csv
import shutil
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# --- SYSTEM CONFIGURATION ---
DATA_DIRECTORY = r"R:\Release_Internal_V1A\output"
INTRINSIC_FILE = "capsule_intrinsics.npz"
STAGE_KINEMATICS_FILE = "stage_kinematics.npz" 
STAGE_ARUCO_ID = 0
# Use the black marker side length, not the full paper patch with white margins.
STAGE_ARUCO_MARKER_SIZE_MM = 40.0
STAGE_MEASUREMENT_SAMPLES = 20
TAG_POSE_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainingData")
TAG_POSE_LOG_FILE = os.path.join(TAG_POSE_OUTPUT_DIRECTORY, "camera_pose_relative_to_tag.csv")
TAG_POSE_IMAGE_DIRECTORY = TAG_POSE_OUTPUT_DIRECTORY
TCP_HOST = 'localhost'
TCP_PORT = 12347

# --- CHARUCO BOARD CONFIGURATION ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CHARUCO_BOARD = cv2.aruco.CharucoBoard((7, 9), 0.026, 0.01976, ARUCO_DICT)
detector = cv2.aruco.CharucoDetector(CHARUCO_BOARD)

# Mode 5 uses this compact target rather than the larger intrinsic-calibration board.
# Five 10 mm squares make the printed board exactly 50 mm x 50 mm.
MODE5_CHARUCO_SQUARES_X = 5
MODE5_CHARUCO_SQUARES_Y = 5
MODE5_CHARUCO_SQUARE_LENGTH_M = 0.010
MODE5_CHARUCO_MARKER_LENGTH_M = 0.0075
MODE5_CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    (MODE5_CHARUCO_SQUARES_X, MODE5_CHARUCO_SQUARES_Y),
    MODE5_CHARUCO_SQUARE_LENGTH_M,
    MODE5_CHARUCO_MARKER_LENGTH_M,
    ARUCO_DICT,
)
# OpenCV defines ChArUco board coordinates from an outer corner. Mode 5 shifts
# that frame so the physical center of this 50 mm x 50 mm board is [0, 0, 0].
MODE5_BOARD_CENTER_IN_OPENCV_FRAME_M = np.array(
    [
        MODE5_CHARUCO_SQUARES_X * MODE5_CHARUCO_SQUARE_LENGTH_M / 2.0,
        MODE5_CHARUCO_SQUARES_Y * MODE5_CHARUCO_SQUARE_LENGTH_M / 2.0,
        0.0,
    ],
    dtype=np.float64,
).reshape(3, 1)
# Mode 5's physical tag frame uses the centered origin with Y and Z reversed
# from OpenCV's ChArUco board frame, equivalent to a 180 degree local X rotation.
MODE5_TAG_FRAME_FROM_OPENCV_FRAME = np.diag([1.0, -1.0, -1.0])
MODE5_CHARUCO_OUTPUT_FILE = "mode5_charuco_50mm.png"
MODE5_CHARUCO_PRINT_DPI = 600
MODE7_VERIFICATION_OUTPUT_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tag_pose_synthetic_output"
)
MODE7_BOARD_RENDER_SIZE_PX = 2000
MODE7_OVERLAY_ALPHA = 0.55
MODE5_MIN_CORNERS = 8
MODE5_FULL_CORNERS = 12
MODE5_FALLBACK_MIN_BOARD_COVERAGE = 0.0
MODE5_MAX_REPROJECTION_RMSE_PX = 1.0
MODE5_MIN_INLIER_SAMPLES = 10
MODE5_POSITION_OUTLIER_FLOOR_MM = 0.05
MODE5_ROTATION_OUTLIER_FLOOR_DEG = 0.05

# --- GLOBAL VARIABLES ---
imu_lock = threading.Lock()
latest_quaternion = None
imu_thread_running = False

# -----------------------------------------------------------------------------
# IMU LISTENER
# -----------------------------------------------------------------------------
def imu_listener_thread(sock):
    global latest_quaternion, imu_thread_running
    sock.settimeout(1.0)
    
    while imu_thread_running:
        try:
            raw_len = sock.recv(4)
            if not raw_len or len(raw_len) < 4:
                continue
            length = struct.unpack('>I', raw_len)[0]
            
            data_bytes = bytearray()
            while len(data_bytes) < length:
                packet = sock.recv(length - len(data_bytes))
                if not packet:
                    break
                data_bytes.extend(packet)
            
            if len(data_bytes) == length:
                data = json.loads(data_bytes.decode('utf-8'))
                q = np.array([data['qw'], data['qx'], data['qy'], data['qz']])
                
                with imu_lock:
                    latest_quaternion = q
                    
        except socket.timeout:
            continue
        except Exception as e:
            if imu_thread_running:
                print(f"\n[IMU Background Thread Error] {e}")
            break

def get_live_quaternion():
    global latest_quaternion
    with imu_lock:
        val = latest_quaternion
        latest_quaternion = None
        return val

# -----------------------------------------------------------------------------
# HAND-EYE SOLVER
# -----------------------------------------------------------------------------
def solve_hand_eye_rotation_log(imu_poses, cam_poses):
    if len(imu_poses) != len(cam_poses):
        raise ValueError("IMU and camera pose counts must match.")
    if len(imu_poses) < 3:
        raise ValueError("Need at least 3 poses to solve hand-eye rotation.")

    H = np.zeros((3, 3), dtype=np.float64)
    for i in range(len(imu_poses) - 1):
        R_a = imu_poses[i].T @ imu_poses[i + 1]
        R_b = cam_poses[i].T @ cam_poses[i + 1]

        a_vec = R.from_matrix(R_a).as_rotvec()
        b_vec = R.from_matrix(R_b).as_rotvec()
        H += np.outer(b_vec, a_vec)

    U, _, Vt = np.linalg.svd(H)
    R_x = Vt.T @ U.T
    if np.linalg.det(R_x) < 0:
        U[:, 2] *= -1
        R_x = Vt.T @ U.T
    return R_x

def verify_hand_eye_constraint(imu_poses, cam_poses, R_x_est):
    print("\n--- VERIFYING HAND-EYE CONSTRAINT ---")
    errors = []
    for i in range(len(imu_poses) - 1):
        A_i = imu_poses[i].T @ imu_poses[i + 1]
        B_i = cam_poses[i].T @ cam_poses[i + 1]
        
        left = A_i @ R_x_est
        right = R_x_est @ B_i
        
        error = np.linalg.norm(left - right, 'fro')
        errors.append(error)
        print(f"Constraint error for pose pair {i+1}: {error:.8f}")
    
    print(f"Mean constraint error: {np.mean(errors):.8f}")
    print(f"Max constraint error: {np.max(errors):.8f}")
    return errors

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def get_latest_image(directory):
    files = [
        path for path in glob.glob(os.path.join(directory, "*.jpg"))
        if not path.endswith(".tmp.jpg")
    ]
    if not files:
        return None
    return max(files, key=lambda path: (os.stat(path).st_mtime_ns, path))

def get_image_revision(path):
    try:
        stat = os.stat(path)
        return path, stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


def load_capsule_frame(path):
    frame = cv2.imread(path)
    if frame is None:
        return None
    return cv2.rotate(frame, cv2.ROTATE_180)


def generate_mode5_charuco_board():
    """Generate the 50 mm square ChArUco target at a fixed print resolution."""
    edge_px = round(
        MODE5_CHARUCO_SQUARES_X
        * MODE5_CHARUCO_SQUARE_LENGTH_M
        / 0.0254
        * MODE5_CHARUCO_PRINT_DPI
    )
    board_image = MODE5_CHARUCO_BOARD.generateImage((edge_px, edge_px), marginSize=0)
    if not cv2.imwrite(MODE5_CHARUCO_OUTPUT_FILE, board_image):
        print(f"Could not save board image to '{MODE5_CHARUCO_OUTPUT_FILE}'.")
        return
    print(
        f"Saved 50.00 mm x 50.00 mm ChArUco board to '{MODE5_CHARUCO_OUTPUT_FILE}' "
        f"({edge_px} px at {MODE5_CHARUCO_PRINT_DPI} DPI)."
    )
    print(
        "In the print dialog, set image resolution to 600 DPI or set the outer edge to 50.00 mm; "
        "disable fit-to-page and verify the outer edge with calipers."
    )


def _create_mode5_charuco_detector():
    detector_parameters = cv2.aruco.DetectorParameters()
    detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_parameters.cornerRefinementWinSize = 5
    detector_parameters.cornerRefinementMaxIterations = 50
    detector_parameters.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.CharucoDetector(
        MODE5_CHARUCO_BOARD,
        cv2.aruco.CharucoParameters(),
        detector_parameters,
    )


def _mode5_has_sufficient_board_coverage(obj_pts):
    board_points = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    if len(board_points) == 0:
        return False

    board_width = MODE5_CHARUCO_SQUARES_X * MODE5_CHARUCO_SQUARE_LENGTH_M
    board_height = MODE5_CHARUCO_SQUARES_Y * MODE5_CHARUCO_SQUARE_LENGTH_M
    x_coverage = (np.max(board_points[:, 0]) - np.min(board_points[:, 0])) / board_width
    y_coverage = (np.max(board_points[:, 1]) - np.min(board_points[:, 1])) / board_height
    return min(x_coverage, y_coverage) >= MODE5_FALLBACK_MIN_BOARD_COVERAGE


def _solve_mode5_charuco_pose(charuco_corners, charuco_ids, camera_matrix, dist_coeffs):
    if charuco_corners is None or len(charuco_corners) < MODE5_MIN_CORNERS:
        corner_count = 0 if charuco_corners is None else len(charuco_corners)
        return None, f"need {MODE5_MIN_CORNERS} ChArUco corners; found {corner_count}"

    obj_pts, img_pts = MODE5_CHARUCO_BOARD.matchImagePoints(charuco_corners, charuco_ids)
    if len(obj_pts) < MODE5_MIN_CORNERS:
        return None, f"only {len(obj_pts)} matched ChArUco corners"
    if len(obj_pts) < MODE5_FULL_CORNERS and not _mode5_has_sufficient_board_coverage(obj_pts):
        return None, (
            f"8-11 corners cover less than {MODE5_FALLBACK_MIN_BOARD_COVERAGE:.0%} "
            "of the board"
        )

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, "PnP solve failed"

    rvec, tvec = cv2.solvePnPRefineLM(
        obj_pts, img_pts, camera_matrix, dist_coeffs, rvec, tvec
    )
    projected_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    residuals = img_pts.reshape(-1, 2) - projected_points.reshape(-1, 2)
    reprojection_rmse_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    if reprojection_rmse_px > MODE5_MAX_REPROJECTION_RMSE_PX:
        return None, (
            f"RMSE {reprojection_rmse_px:.2f} px exceeds "
            f"{MODE5_MAX_REPROJECTION_RMSE_PX:.2f} px"
        )

    R_board_to_cam, _ = cv2.Rodrigues(rvec)
    tvec_center_origin = tvec + R_board_to_cam @ MODE5_BOARD_CENTER_IN_OPENCV_FRAME_M
    R_tag_to_cam = R_board_to_cam @ MODE5_TAG_FRAME_FROM_OPENCV_FRAME
    rvec_tag, _ = cv2.Rodrigues(R_tag_to_cam)
    return (rvec_tag, tvec_center_origin, reprojection_rmse_px, len(obj_pts)), None


def _robust_average_mode5_poses(positions_mm, rotations):
    positions_mm = np.asarray(positions_mm, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    position_median = np.median(positions_mm, axis=0)
    initial_rotation = _average_rotation_matrices(rotations)

    position_errors_mm = np.linalg.norm(positions_mm - position_median, axis=1)
    rotation_errors_deg = np.degrees(
        R.from_matrix(np.transpose(initial_rotation) @ rotations).magnitude()
    )
    position_mad_mm = float(np.median(np.abs(position_errors_mm - np.median(position_errors_mm))))
    rotation_mad_deg = float(np.median(np.abs(rotation_errors_deg - np.median(rotation_errors_deg))))
    position_limit_mm = max(
        MODE5_POSITION_OUTLIER_FLOOR_MM,
        float(np.median(position_errors_mm)) + 3.0 * 1.4826 * position_mad_mm,
    )
    rotation_limit_deg = max(
        MODE5_ROTATION_OUTLIER_FLOOR_DEG,
        float(np.median(rotation_errors_deg)) + 3.0 * 1.4826 * rotation_mad_deg,
    )
    inliers = (position_errors_mm <= position_limit_mm) & (rotation_errors_deg <= rotation_limit_deg)

    if int(np.count_nonzero(inliers)) < MODE5_MIN_INLIER_SAMPLES:
        return None

    return (
        np.mean(positions_mm[inliers], axis=0),
        _average_rotation_matrices(rotations[inliers]),
        inliers,
        position_errors_mm,
        rotation_errors_deg,
        position_limit_mm,
        rotation_limit_deg,
    )


def _mode7_tag_to_camera_pose(row):
    """Invert a Mode 5 camera-in-tag pose into an OpenCV tag-to-camera pose."""
    camera_position_mm = np.array(
        [
            float(row['camera_x_mm']),
            float(row['camera_y_mm']),
            float(row['camera_z_mm']),
        ],
        dtype=np.float64,
    ).reshape(3, 1)
    camera_euler_deg = np.array(
        [
            float(row['camera_roll_deg']),
            float(row['camera_pitch_deg']),
            float(row['camera_yaw_deg']),
        ],
        dtype=np.float64,
    )
    R_cam_to_tag = R.from_euler('xyz', camera_euler_deg, degrees=True).as_matrix()
    R_tag_to_cam = R_cam_to_tag.T
    t_tag_to_cam_m = -R_tag_to_cam @ camera_position_mm / 1000.0
    rvec_tag_to_cam, _ = cv2.Rodrigues(R_tag_to_cam)
    return R_tag_to_cam, rvec_tag_to_cam, t_tag_to_cam_m


def _render_mode7_charuco_board(image_shape, R_tag_to_cam, t_tag_to_cam_m, camera_matrix, dist_coeffs):
    """Project the full Mode 5 board using the logged pose and camera model."""
    board_image = MODE5_CHARUCO_BOARD.generateImage(
        (MODE7_BOARD_RENDER_SIZE_PX, MODE7_BOARD_RENDER_SIZE_PX), marginSize=0
    )
    board_image = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
    R_board_to_cam = R_tag_to_cam @ MODE5_TAG_FRAME_FROM_OPENCV_FRAME
    t_board_to_cam_m = t_tag_to_cam_m - R_board_to_cam @ MODE5_BOARD_CENTER_IN_OPENCV_FRAME_M
    rvec_board_to_cam, _ = cv2.Rodrigues(R_board_to_cam)
    board_width_m = MODE5_CHARUCO_SQUARES_X * MODE5_CHARUCO_SQUARE_LENGTH_M
    board_height_m = MODE5_CHARUCO_SQUARES_Y * MODE5_CHARUCO_SQUARE_LENGTH_M
    board_corners_m = np.array(
        [
            [0.0, 0.0, 0.0],
            [board_width_m, 0.0, 0.0],
            [board_width_m, board_height_m, 0.0],
            [0.0, board_height_m, 0.0],
        ],
        dtype=np.float32,
    )
    undistorted_corners, _ = cv2.projectPoints(
        board_corners_m, rvec_board_to_cam, t_board_to_cam_m, camera_matrix, None
    )
    undistorted_corners = undistorted_corners.reshape(4, 2).astype(np.float32)
    source_size = MODE7_BOARD_RENDER_SIZE_PX - 1
    source_corners = np.array(
        [[0, 0], [source_size, 0], [source_size, source_size], [0, source_size]], dtype=np.float32
    )
    height, width = image_shape[:2]
    homography = cv2.getPerspectiveTransform(source_corners, undistorted_corners)
    inverse_homography = np.linalg.inv(homography)
    pixel_grid = np.stack(
        np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)), axis=-1
    ).reshape(-1, 1, 2)
    undistorted_pixels = cv2.undistortPoints(
        pixel_grid, camera_matrix, dist_coeffs, P=camera_matrix
    )
    board_pixels = cv2.perspectiveTransform(undistorted_pixels, inverse_homography).reshape(height, width, 2)
    map_x = board_pixels[:, :, 0].astype(np.float32)
    map_y = board_pixels[:, :, 1].astype(np.float32)
    synthetic = np.full((height, width, 3), 255, dtype=np.uint8)
    warped_board = cv2.remap(
        board_image, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    mask = np.where(
        (map_x >= 0) & (map_x <= source_size) & (map_y >= 0) & (map_y <= source_size), 255, 0
    ).astype(np.uint8)
    synthetic[mask > 0] = warped_board[mask > 0]
    return synthetic, mask


def _mode7_comparison_image(real_image, synthetic, board_mask):
    overlay = real_image.copy()
    synthetic_gray = cv2.cvtColor(synthetic, cv2.COLOR_BGR2GRAY)
    synthetic_inverted = cv2.bitwise_not(synthetic_gray)
    synthetic_tinted = cv2.cvtColor(synthetic_inverted, cv2.COLOR_GRAY2BGR)
    synthetic_tinted[:, :, 2] = 0
    blended = cv2.addWeighted(real_image, 0.80, synthetic_tinted, 0.20, 0.0)
    overlay[board_mask > 0] = blended[board_mask > 0]
    board_outline, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, board_outline, -1, (0, 255, 255), 2, cv2.LINE_AA)
    label_height = 34
    panels = []
    for image, label in (
        (real_image, 'Captured tag image (rotated)'),
        (synthetic_tinted, 'Inverted cyan synthetic ChArUco from logged pose'),
        (overlay, '40% inverted cyan synthetic overlay'),
    ):
        panel = cv2.copyMakeBorder(image, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    return cv2.hconcat(panels), overlay

# -----------------------------------------------------------------------------
# PHASE 1: INTRINSIC CALIBRATION
# -----------------------------------------------------------------------------
def run_intrinsic_calibration():
    print("\n--- PHASE 1: CAMERA INTRINSIC CALIBRATION ---")
    print("Instructions:")
    print("1. Bring the 'Capsule Camera Feed' window into focus.")
    print("2. Move/tilt the capsule until you see clean green corner marks.")
    print("3. Press [SPACEBAR] on your keyboard to save a snapshot pose.")
    print("4. Press [ENTER] when you are finished collecting frames to solve.\n")
    
    all_corners = []
    all_ids = []
    imsize = None
    last_valid_frame = None
    frame_size_wh = None
    
    cv2.namedWindow("Capsule Camera Feed", cv2.WINDOW_NORMAL)
    last_processed_img = None
    last_processed_revision = None
    
    while True:
        img_path = get_latest_image(DATA_DIRECTORY)
        image_revision = get_image_revision(img_path) if img_path else None
        if image_revision is not None and image_revision != last_processed_revision:
            frame = load_capsule_frame(img_path)
            if frame is not None:
                last_valid_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                imsize = gray.shape[::-1]
                frame_size_wh = imsize
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
                
                vis_frame = frame.copy()
                is_detected = charuco_corners is not None and len(charuco_corners) >= 6
                
                if is_detected:
                    cv2.aruco.drawDetectedCornersCharuco(vis_frame, charuco_corners, charuco_ids, (0, 255, 0))
                    cv2.putText(vis_frame, f"Corners: {len(charuco_corners)} (Ready)", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_frame, "BOARD UNLOCKED / BLURRY", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(vis_frame, f"Saved Frames: {len(all_corners)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Capsule Camera Feed", vis_frame)
                last_processed_img = img_path
                last_processed_revision = image_revision

        key = cv2.waitKey(10) & 0xFF
        
        if key == ord(' '):
            if 'is_detected' in locals() and is_detected:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print(f"--> Frame {len(all_corners)} locked! Recognized {len(charuco_corners)} intersections.")
            else:
                print("--> Capture Denied: Target board features missing or unresolvable right now.")
        elif key == 13 or key == 10:
            break
            
    cv2.destroyWindow("Capsule Camera Feed")
    
    if len(all_corners) < 10:
        print("Aborted: Need at least 10 valid images to generate an accurate intrinsic profile.")
        return

    print("\nProcessing lens matrix parameters (Calibrating)...")
    all_object_points, all_image_points = [], []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = CHARUCO_BOARD.matchImagePoints(corners, ids)
        if len(obj_pts) >= 4:
            all_object_points.append(obj_pts)
            all_image_points.append(img_pts)

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        all_object_points, all_image_points, imsize, None, None
    )
    
    if ret:
        np.savez(INTRINSIC_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print(f"\n✅ Intrinsic dataset compiled successfully and cached to: '{INTRINSIC_FILE}'")
        print("Camera Matrix:\n", camera_matrix)

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])

        if frame_size_wh is None:
            frame_size_wh = imsize
        width_px, height_px = frame_size_wh
        center_offset_x_px = cx - (width_px / 2.0)
        center_offset_y_px = cy - (height_px / 2.0)
        pixel_pitch_um = 1.4

        dist = dist_coeffs.ravel().astype(float)
        k1 = float(dist[0]) if dist.size > 0 else 0.0
        k2 = float(dist[1]) if dist.size > 1 else 0.0
        p1 = float(dist[2]) if dist.size > 2 else 0.0
        p2 = float(dist[3]) if dist.size > 3 else 0.0
        k3 = float(dist[4]) if dist.size > 4 else 0.0

        pitch_deg = np.degrees(-p2 / (2.0 * fx))
        yaw_deg = np.degrees(p1 / (2.0 * fy))

        print("\n=== Calibration Diagnostic Summary ===")
        print(f"fx: {fx:.6f} px ({fx * pixel_pitch_um:.6f} µm)")
        print(f"fy: {fy:.6f} px ({fy * pixel_pitch_um:.6f} µm)")
        print(f"cx: {cx:.6f} px ({cx * pixel_pitch_um:.6f} µm)")
        print(f"cy: {cy:.6f} px ({cy * pixel_pitch_um:.6f} µm)")
        print(
            "Center offset: "
            f"({center_offset_x_px:.6f}, {center_offset_y_px:.6f}) px "
            f"({center_offset_x_px * pixel_pitch_um:.6f}, {center_offset_y_px * pixel_pitch_um:.6f}) µm"
        )
        print(f"k1: {k1:.8f}")
        print(f"k2: {k2:.8f}")
        print(f"k3: {k3:.8f}")
        print(f"p1: {p1:.8f}")
        print(f"p2: {p2:.8f}")
        print(f"RMS reprojection error: {float(ret):.8f}")
        print(f"Estimated lens tilt pitch: {pitch_deg:.6f} deg")
        print(f"Estimated lens tilt yaw:   {yaw_deg:.6f} deg")

        if abs(pitch_deg) > 5.0 or abs(yaw_deg) > 5.0:
            print(
                "WARNING: Estimated lens tilt magnitude exceeds 5.0 deg. "
                "Consider re-calibrating with cv2.CALIB_ZERO_TANGENT_DIST."
            )

        if last_valid_frame is not None:
            undistorted = cv2.undistort(last_valid_frame, camera_matrix, dist_coeffs)
            side_by_side = cv2.hconcat([last_valid_frame, undistorted])
            cv2.namedWindow("Raw vs Undistorted Feed", cv2.WINDOW_NORMAL)
            cv2.imshow("Raw vs Undistorted Feed", side_by_side)
            print("Displaying visual verification in window: 'Raw vs Undistorted Feed'")
            print("Press any key while the image window is focused to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow("Raw vs Undistorted Feed")
        else:
            print("Visual verification skipped: no valid frame was captured.")
    else:
        print("Calibration engine processing exception error.")

# -----------------------------------------------------------------------------
# PHASE 2: BORESIGHT CALIBRATION
# -----------------------------------------------------------------------------
def run_boresight_calibration():
    global imu_thread_running
    
    if not os.path.exists(INTRINSIC_FILE):
        print(f"Error: Missing profile config dependency '{INTRINSIC_FILE}'. Run intrinsic pass first.")
        return
        
    data = np.load(INTRINSIC_FILE)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((TCP_HOST, TCP_PORT))
        print(f"Connected to IMU telemetry server on {TCP_HOST}:{TCP_PORT}")
    except Exception:
        print(f"Network error: Unable to connect to port {TCP_PORT}. Verify capsule_orientation.py is active.")
        return

    imu_thread_running = True
    listener = threading.Thread(target=imu_listener_thread, args=(sock,))
    listener.daemon = True
    listener.start()

    imu_poses, cam_poses = [], []
    logged_quaternions = []  
    
    print("\n--- PHASE 2: BORESIGHT OFFSET CALIBRATION ---")
    print("Instructions:")
    print("1. Click inside the 'Capsule Camera Feed' window to focus it.")
    print("2. Press [SPACEBAR] in the calibration window to save each pose.")
    print("3. Press [ENTER] when all poses are captured to compute the result.\n")
    print("NOTE: In simulation mode, press ENTER in the simulator window first,")
    print("      then press SPACEBAR in the calibration window to capture.\n")
    
    cv2.namedWindow("Capsule Camera Feed", cv2.WINDOW_NORMAL)
    last_processed_img = None
    last_processed_revision = None
    
    vis_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(vis_frame, "Waiting for stream sync...", (30, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
    current_corners, current_ids = None, None
    is_detected = False
    
    try:
        while True:
            img_path = get_latest_image(DATA_DIRECTORY)
            image_revision = get_image_revision(img_path) if img_path else None
            
            if image_revision is not None and image_revision != last_processed_revision:
                time.sleep(0.05)
                frame = load_capsule_frame(img_path)
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    current_corners, current_ids, _, _ = detector.detectBoard(gray)
                    
                    vis_frame = frame.copy()
                    is_detected = current_corners is not None and len(current_corners) >= 4
                    
                    if is_detected:
                        cv2.aruco.drawDetectedCornersCharuco(vis_frame, current_corners, current_ids, (0, 255, 0))
                        cv2.putText(vis_frame, "Ready for Alignment Match", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(vis_frame, "Vision Track Lost", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                    last_processed_img = img_path
                    last_processed_revision = image_revision

            display_copy = vis_frame.copy()
            cv2.putText(display_copy, f"Logged Poses: {len(imu_poses)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if last_processed_img:
                cv2.putText(display_copy, f"File: {os.path.basename(last_processed_img)}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Capsule Camera Feed", display_copy)

            key = cv2.waitKey(15) & 0xFF
            
            if key == ord(' '):
                q_live = get_live_quaternion()
                
                if not is_detected or q_live is None:
                    reasons = []
                    if not is_detected:
                        reasons.append("Board Tracking Lost (Needs >= 4 corners)")
                    if q_live is None:
                        reasons.append("IMU Socket returned None (Timeout/No data)")
                    print(f"--> Sync Frame Dropped. Reason(s): {', '.join(reasons)}")
                    continue

                obj_pts, img_pts = CHARUCO_BOARD.matchImagePoints(current_corners, current_ids)
                
                if len(obj_pts) >= 4:
                    retval, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, camera_matrix, dist_coeffs, 
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if retval:
                        R_board_to_cam, _ = cv2.Rodrigues(rvec)
                        R_cam_to_board = R_board_to_cam.T
                        
                        R_imu = R.from_quat([q_live[1], q_live[2], q_live[3], q_live[0]]).as_matrix()
                        
                        imu_poses.append(R_imu)
                        cam_poses.append(R_cam_to_board)
                        logged_quaternions.append(q_live)  
                        
                        print(f"--> Pose pairing {len(imu_poses)} saved! "
                              f"q: [{q_live[0]:.4f}, {q_live[1]:.4f}, {q_live[2]:.4f}, {q_live[3]:.4f}]")
                    else:
                        print("--> PnP Geometry Engine failed to converge.")
                else:
                    print("--> Insufficient paired feature points matched.")
                    
            elif key == 13 or key == 10:
                break
                
    finally:
        print("\nStopping IMU data listener thread...")
        imu_thread_running = False
        listener.join(timeout=1.0)
        sock.close()
        cv2.destroyAllWindows()
    
    if len(imu_poses) < 3:
        print("Aborted: Need at least 3 distinct poses to solve rotation matrix.")
        return
    
    R_x = solve_hand_eye_rotation_log(imu_poses, cam_poses)
    verify_hand_eye_constraint(imu_poses, cam_poses, R_x)
        
    euler = R.from_matrix(R_x).as_euler('xyz', degrees=True)
    
    print("\n" + "="*70)
    print("LOGGED DATA POSE HISTORY SUMMARY")
    print("="*70)
    print(f"{'Pose #':<8}{'qw':<12}{'qx':<12}{'qy':<12}{'qz':<12}")
    for idx, q in enumerate(logged_quaternions):
        print(f"{idx+1:<8}{q[0]:.6f}    {q[1]:.6f}    {q[2]:.6f}    {q[3]:.6f}")
        
    print("\n" + "="*70 + "\nFINAL BORESIGHT CALIBRATION RESULT\n" + "="*70)
    print("Rotation Matrix (R_cam_to_pcb):\n", np.round(R_x, 6))
    print(f"\nEuler Angles Offset:\nRoll(X): {euler[0]:.4f}° | Pitch(Y): {euler[1]:.4f}° | Yaw(Z): {euler[2]:.4f}°")
    
    if os.path.exists("ground_truth_offset.npz"):
        gt_data = np.load("ground_truth_offset.npz")
        R_gt = gt_data['R_cam_to_pcb_GT']
        euler_gt = R.from_matrix(R_gt).as_euler('xyz', degrees=True)
        print("\n" + "="*70)
        print("GROUND TRUTH COMPARISON (Simulation Mode)")
        print("="*70)
        print(f"Ground Truth Euler:\nRoll(X): {euler_gt[0]:.4f}° | Pitch(Y): {euler_gt[1]:.4f}° | Yaw(Z): {euler_gt[2]:.4f}°")
        error_euler = np.abs(euler - euler_gt)
        print(f"\nErrors:\nRoll: {error_euler[0]:.4f}° | Pitch: {error_euler[1]:.4f}° | Yaw: {error_euler[2]:.4f}°")
    print("="*70)

# -----------------------------------------------------------------------------
# ADDITIONAL PASS: ECCENTRICITY CHECK MODE
# -----------------------------------------------------------------------------
def run_eccentricity_check():
    print("\n--- ECCENTRICITY CHECK (PIVOT CALIBRATION) ---")
    print("Checking if the lens center of projection lies exactly on the stage rotation axis.")
    print("Instructions:")
    print("1. Set your target board flat and steady.")
    print("2. Rotate the precision stage to different angles (e.g. -20 to +20 deg).")
    print("3. Press [SPACEBAR] to log the static 3D lens location at each position.")
    print("4. Press [ENTER] to process the pivot arc and estimate the mechanical offset.\n")

    if not os.path.exists(INTRINSIC_FILE):
        print(f"Error: Missing profile config dependency '{INTRINSIC_FILE}'. Run intrinsic pass first.")
        return
        
    data = np.load(INTRINSIC_FILE)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    lens_positions_in_tag_frame = []
    
    cv2.namedWindow("Capsule Camera Feed", cv2.WINDOW_NORMAL)
    last_processed_img = None
    last_processed_revision = None
    
    vis_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_corners, current_ids = None, None
    is_detected = False

    while True:
        img_path = get_latest_image(DATA_DIRECTORY)
        image_revision = get_image_revision(img_path) if img_path else None
        
        if image_revision is not None and image_revision != last_processed_revision:
            time.sleep(0.05)
            frame = load_capsule_frame(img_path)
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_corners, current_ids, _, _ = detector.detectBoard(gray)
                
                vis_frame = frame.copy()
                is_detected = current_corners is not None and len(current_corners) >= 4
                
                if is_detected:
                    cv2.aruco.drawDetectedCornersCharuco(vis_frame, current_corners, current_ids, (0, 255, 0))
                    cv2.putText(vis_frame, "Ready for Position Check", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_frame, "Vision Track Lost", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                last_processed_img = img_path
                last_processed_revision = image_revision

        display_copy = vis_frame.copy()
        cv2.putText(display_copy, f"Logged Poses: {len(lens_positions_in_tag_frame)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Capsule Camera Feed", display_copy)

        key = cv2.waitKey(15) & 0xFF
        
        if key == ord(' '):
            if not is_detected:
                print("--> Frame Dropped: Tag not tracked.")
                continue

            obj_pts, img_pts = CHARUCO_BOARD.matchImagePoints(current_corners, current_ids)
            
            if len(obj_pts) >= 4:
                retval, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, camera_matrix, dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if retval:
                    R_board_to_cam, _ = cv2.Rodrigues(rvec)
                    R_cam_to_board = R_board_to_cam.T
                    
                    # Project lens center back into the tag frame: P = -R^T * t
                    p_lens = -R_cam_to_board @ tvec
                    
                    # Store as simple 3D vector (in millimeters for intuitive debugging)
                    lens_positions_in_tag_frame.append(p_lens.flatten() * 1000.0)
                    
                    print(f"--> Position {len(lens_positions_in_tag_frame)} stored! "
                          f"Lens in tag: X={p_lens[0][0]*1000.0:.2f}mm, "
                          f"Y={p_lens[1][0]*1000.0:.2f}mm, "
                          f"Z={p_lens[2][0]*1000.0:.2f}mm")
                else:
                    print("--> PnP Geometry Engine failed to converge.")
            else:
                print("--> Insufficient paired features.")
                
        elif key == 13 or key == 10:
            break

    cv2.destroyAllWindows()

    if len(lens_positions_in_tag_frame) < 3:
        print("Check Aborted: Need at least 3 logged positions to perform pivot analysis.")
        return

    pts = np.array(lens_positions_in_tag_frame)
    mean_pos = np.mean(pts, axis=0)
    
    # Calculate distance of each point from the centroid
    deviations = np.linalg.norm(pts - mean_pos, axis=1)
    max_deviation = np.max(deviations)
    std_deviation = np.std(deviations)

    print("\n" + "="*70)
    print("ECCENTRICITY CHECK / PIVOT ANALYSIS RESULTS (All values in mm)")
    print("="*70)
    print(f"Centroid Position (Mean): X={mean_pos[0]:.2f} | Y={mean_pos[1]:.2f} | Z={mean_pos[2]:.2f}")
    print("-"*70)
    print(f"{'Point #':<10}{'X':<12}{'Y':<12}{'Z':<12}{'Dev from Mean':<15}")
    for idx, p in enumerate(pts):
        print(f"{idx+1:<10}{p[0]:.2f}{p[1]:.2f}{p[2]:.2f}{deviations[idx]:.4f}")
    
    print("-"*70)
    print(f"Maximum translation deviation (Translational Sweep): {max_deviation:.3f} mm")
    print(f"Standard deviation of offset:                    {std_deviation:.4f} mm")
    print("="*70)

    if max_deviation < 0.25:
        print("STATUS: EXCELLENT alignment! Lens center of projection matches the pivot axis closely.")
    elif max_deviation < 0.6:
        print("STATUS: GOOD. The concentric mechanical offset is acceptable for typical SLAM.")
    else:
        print("WARNING: HIGH ECCENTRICITY! The lens center is rotating along a visible arc.")
        print("Check if the physical capsule is clamped correctly inside your V-groove or adapter.")
    print("="*70 + "\n")

# -----------------------------------------------------------------------------
# PHASE 4: STAGE KINEMATIC CALIBRATION
# -----------------------------------------------------------------------------
def _prompt_vector3_with_default(prompt, default_vec):
    default_str = f"{default_vec[0]}, {default_vec[1]}, {default_vec[2]}"
    user_in = input(f"{prompt} [{default_str}]: ").strip()
    if not user_in:
        return np.array(default_vec, dtype=np.float64)
    try:
        parts = [float(v.strip()) for v in user_in.split(',')]
        if len(parts) != 3:
            raise ValueError
        return np.array(parts, dtype=np.float64)
    except ValueError:
        print("Invalid input. Using default values.")
        return np.array(default_vec, dtype=np.float64)


def _prompt_nonzero_float(prompt):
    user_in = input(prompt).strip()
    try:
        val = float(user_in)
    except ValueError:
        print("Error: value must be numeric.")
        return None
    if abs(val) < 1e-9:
        print("Error: displacement must be non-zero.")
        return None
    return val


def _average_rotation_matrices(rotation_matrices):
    """Average rotation matrices using quaternion eigen averaging."""
    if len(rotation_matrices) == 0:
        raise ValueError("Cannot average an empty rotation set.")

    quats_xyzw = R.from_matrix(np.asarray(rotation_matrices)).as_quat()
    accum = np.zeros((4, 4), dtype=np.float64)
    for q in quats_xyzw:
        q = q / np.linalg.norm(q)
        accum += np.outer(q, q)
    accum /= float(len(quats_xyzw))

    eig_vals, eig_vecs = np.linalg.eigh(accum)
    q_mean = eig_vecs[:, np.argmax(eig_vals)]
    if q_mean[3] < 0:
        q_mean = -q_mean

    return R.from_quat(q_mean).as_matrix()


def _capture_tag_pose_mm(camera_matrix, dist_coeffs, window_name="Capsule Camera Feed"):
    print("Press [SPACEBAR] on the camera window to capture this pose.")
    print("Press [Q] to abort this stage calibration mode.")
    print(f"Each measurement now captures {STAGE_MEASUREMENT_SAMPLES} photos and averages them.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_processed_revision = None
    marker_corners, marker_ids = None, None
    vis_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    found_target = False
    current_rvec = None
    current_tvec = None
    marker_size_m = STAGE_ARUCO_MARKER_SIZE_MM / 1000.0
    obj_pts = np.array([
        [-marker_size_m / 2.0, marker_size_m / 2.0, 0.0],
        [ marker_size_m / 2.0, marker_size_m / 2.0, 0.0],
        [ marker_size_m / 2.0,-marker_size_m / 2.0, 0.0],
        [-marker_size_m / 2.0,-marker_size_m / 2.0, 0.0],
    ], dtype=np.float32)
    axis_length_m = marker_size_m * 0.75
    solvepnp_flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
    stage_detector = cv2.aruco.ArucoDetector(ARUCO_DICT)
    Rz_180 = R.from_euler('z', 180.0, degrees=True).as_matrix()

    def _solve_pose_from_detection(corners, ids):
        if ids is None or len(ids) == 0:
            return None
        ids_flat = ids.flatten()
        if STAGE_ARUCO_ID not in ids_flat:
            return None

        target_idx = int(np.where(ids_flat == STAGE_ARUCO_ID)[0][0])
        img_pts = corners[target_idx].reshape(4, 2).astype(np.float32)
        retval, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            flags=solvepnp_flag
        )
        if not retval:
            return None
        return rvec, tvec

    while True:
        img_path = get_latest_image(DATA_DIRECTORY)
        image_revision = get_image_revision(img_path) if img_path else None

        if image_revision is not None and image_revision != last_processed_revision:
            time.sleep(0.05)
            frame = load_capsule_frame(img_path)
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                marker_corners, marker_ids, _ = stage_detector.detectMarkers(gray)

                vis_frame = frame.copy()
                found_target = False
                current_rvec = None
                current_tvec = None
                if marker_ids is not None and len(marker_ids) > 0:
                    ids_flat = marker_ids.flatten()
                    found_target = np.any(ids_flat == STAGE_ARUCO_ID)
                    cv2.aruco.drawDetectedMarkers(vis_frame, marker_corners, marker_ids)
                    solved_pose = _solve_pose_from_detection(marker_corners, marker_ids)
                    if solved_pose is not None:
                        current_rvec, current_tvec = solved_pose

                if found_target and current_rvec is not None:
                    R_tag_to_cam, _ = cv2.Rodrigues(current_rvec)
                    R_axis_display = R_tag_to_cam @ Rz_180
                    rvec_axis_display, _ = cv2.Rodrigues(R_axis_display)
                    cv2.drawFrameAxes(
                        vis_frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec_axis_display,
                        current_tvec,
                        axis_length_m,
                        2,
                    )
                    cv2.putText(vis_frame, "Ready: SPACE to capture", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_frame, f"Marker ID {STAGE_ARUCO_ID} not detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                last_processed_revision = image_revision

        cv2.imshow(window_name, vis_frame)
        key = cv2.waitKey(15) & 0xFF

        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None

        if key == ord(' '):
            if marker_ids is None or len(marker_ids) == 0 or not found_target:
                print("--> Capture denied: no target marker pose available.")
                continue

            measurement_positions_mm = []
            measurement_rotations = []
            sample_count = 0

            if current_rvec is not None and current_tvec is not None:
                R_tag_to_cam_now, _ = cv2.Rodrigues(current_rvec)
                measurement_positions_mm.append(current_tvec.flatten() * 1000.0)
                measurement_rotations.append(R_tag_to_cam_now @ Rz_180)
                sample_count += 1
                print(f"   Sample {sample_count}/{STAGE_MEASUREMENT_SAMPLES} captured")

            retry_limit = STAGE_MEASUREMENT_SAMPLES * 60
            retries = 0
            while sample_count < STAGE_MEASUREMENT_SAMPLES and retries < retry_limit:
                retries += 1
                next_path = get_latest_image(DATA_DIRECTORY)
                next_revision = get_image_revision(next_path) if next_path else None
                if next_revision is None or next_revision == last_processed_revision:
                    cv2.waitKey(10)
                    continue

                time.sleep(0.02)
                next_frame = load_capsule_frame(next_path)
                if next_frame is None:
                    last_processed_revision = next_revision
                    continue

                next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                next_corners, next_ids, _ = stage_detector.detectMarkers(next_gray)
                solved_pose = _solve_pose_from_detection(next_corners, next_ids)

                vis_frame = next_frame.copy()
                marker_corners = next_corners
                marker_ids = next_ids
                found_target = solved_pose is not None
                current_rvec, current_tvec = (solved_pose if solved_pose is not None else (None, None))

                if next_ids is not None and len(next_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(vis_frame, next_corners, next_ids)

                if solved_pose is not None:
                    sample_rvec, sample_tvec = solved_pose
                    R_tag_to_cam_sample, _ = cv2.Rodrigues(sample_rvec)
                    R_axis_display = R_tag_to_cam_sample @ Rz_180
                    rvec_axis_display, _ = cv2.Rodrigues(R_axis_display)
                    cv2.drawFrameAxes(
                        vis_frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec_axis_display,
                        sample_tvec,
                        axis_length_m,
                        2,
                    )

                    measurement_positions_mm.append(sample_tvec.flatten() * 1000.0)
                    measurement_rotations.append(R_axis_display)
                    sample_count += 1
                    print(f"   Sample {sample_count}/{STAGE_MEASUREMENT_SAMPLES} captured")

                cv2.putText(
                    vis_frame,
                    f"Averaging {sample_count}/{STAGE_MEASUREMENT_SAMPLES}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(window_name, vis_frame)
                last_processed_revision = next_revision
                cv2.waitKey(1)

            if sample_count < STAGE_MEASUREMENT_SAMPLES:
                print("--> Capture denied: insufficient valid samples to average. Please hold steady and retry.")
                continue

            P_tag_in_cam_mm = np.mean(np.asarray(measurement_positions_mm), axis=0)
            R_tag_to_cam = _average_rotation_matrices(measurement_rotations)
            print(
                f"--> Captured averaged tag pose ({STAGE_MEASUREMENT_SAMPLES} frames): "
                f"X={P_tag_in_cam_mm[0]:.3f} mm, "
                f"Y={P_tag_in_cam_mm[1]:.3f} mm, Z={P_tag_in_cam_mm[2]:.3f} mm"
            )
            cv2.destroyWindow(window_name)
            return P_tag_in_cam_mm, R_tag_to_cam


def run_stage_kinematic_calibration():
    print("\n--- PHASE 4: STAGE KINEMATIC CALIBRATION ---")

    if not os.path.exists(INTRINSIC_FILE):
        print(f"Error: Missing profile config dependency '{INTRINSIC_FILE}'. Run intrinsic pass first.")
        return

    data = np.load(INTRINSIC_FILE)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    print(
        f"Using stage marker ID {STAGE_ARUCO_ID} "
        f"with side length {STAGE_ARUCO_MARKER_SIZE_MM:.2f} mm."
    )
    print("\nStep 0: Enter known offset from temporary stage marker center to model landmark origin.")
    P_tag_to_model = _prompt_vector3_with_default(
        "P_tag_to_model [off_x, off_y, off_z] in mm (comma-separated)",
        [0.0, -45.0, -1.0]
    )
    euler_tag_to_model_deg = _prompt_vector3_with_default(
        "Tag->Model Euler angles [roll, pitch, yaw] in deg (comma-separated, blank for identity)",
        [0.0, 0.0, 0.0]
    )
    R_tag_to_model = R.from_euler('xyz', euler_tag_to_model_deg, degrees=True).as_matrix()

    print("\nStep 1: Baseline pose at stage origin [0, 0, 0].")
    print("Set stage to [0, 0, 0] with CharUco attached to the model, then capture.")
    baseline_capture = _capture_tag_pose_mm(camera_matrix, dist_coeffs)
    if baseline_capture is None:
        print("Stage kinematic calibration aborted by user.")
        return

    P0, R_base = baseline_capture
    P_model_base_in_cam = P0 + (R_base @ P_tag_to_model)
    R_model_base_in_cam = R_base @ R_tag_to_model

    print("\nStep 2: X-axis motion vector.")
    Dx = _prompt_nonzero_float("Enter stage displacement Dx (mm): ")
    if Dx is None:
        return
    print(f"Move stage by +X = {Dx} mm, then capture.")
    x_capture = _capture_tag_pose_mm(camera_matrix, dist_coeffs)
    if x_capture is None:
        print("Stage kinematic calibration aborted by user.")
        return
    Px, _ = x_capture
    u_X = (Px - P0) / Dx
    print("Return stage to baseline [0, 0, 0] before next step.")

    print("\nStep 3: Y-axis motion vector.")
    Dy = _prompt_nonzero_float("Enter stage displacement Dy (mm): ")
    if Dy is None:
        return
    print(f"Move stage by +Y = {Dy} mm, then capture.")
    y_capture = _capture_tag_pose_mm(camera_matrix, dist_coeffs)
    if y_capture is None:
        print("Stage kinematic calibration aborted by user.")
        return
    Py, _ = y_capture
    u_Y = (Py - P0) / Dy
    print("Return stage to baseline [0, 0, 0] before next step.")

    print("\nStep 4: Z-axis motion vector.")
    Dz = _prompt_nonzero_float("Enter stage displacement Dz (mm): ")
    if Dz is None:
        return
    print(f"Move stage by +Z = {Dz} mm, then capture.")
    z_capture = _capture_tag_pose_mm(camera_matrix, dist_coeffs)
    if z_capture is None:
        print("Stage kinematic calibration aborted by user.")
        return
    Pz, _ = z_capture
    u_Z = (Pz - P0) / Dz

    R_stage_to_cam = np.column_stack((u_X, u_Y, u_Z))

    R_cam_in_model = R_model_base_in_cam.T
    P_cam_in_model = -R_cam_in_model @ P_model_base_in_cam

    euler_cam_in_model_xyz = R.from_matrix(R_cam_in_model).as_euler('xyz', degrees=True)
    roll_deg = float(euler_cam_in_model_xyz[0])
    pitch_deg = float(euler_cam_in_model_xyz[1])
    yaw_deg = float(euler_cam_in_model_xyz[2])

    print("\n" + "=" * 70)
    print("STAGE KINEMATIC CALIBRATION DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"u_X: {u_X} | ||u_X|| = {np.linalg.norm(u_X):.6f}")
    print(f"u_Y: {u_Y} | ||u_Y|| = {np.linalg.norm(u_Y):.6f}")
    print(f"u_Z: {u_Z} | ||u_Z|| = {np.linalg.norm(u_Z):.6f}")
    print("-" * 70)
    print(f"u_X · u_Y = {float(np.dot(u_X, u_Y)):.6f}")
    print(f"u_Y · u_Z = {float(np.dot(u_Y, u_Z)):.6f}")
    print(f"u_X · u_Z = {float(np.dot(u_X, u_Z)):.6f}")
    print("-" * 70)
    print(f"Baseline tag position P0 (mm): [{P0[0]:.3f}, {P0[1]:.3f}, {P0[2]:.3f}]")
    print("Camera pose relative to Model Landmark Frame at baseline:")
    print(
        f"Position (mm): X={P_cam_in_model[0]:.3f}, "
        f"Y={P_cam_in_model[1]:.3f}, Z={P_cam_in_model[2]:.3f}"
    )
    print(
        f"Orientation (deg): Pitch={pitch_deg:.4f}, "
        f"Yaw={yaw_deg:.4f}, Roll={roll_deg:.4f}"
    )
    print("=" * 70)

    np.savez(
        STAGE_KINEMATICS_FILE,
        R_stage_to_cam=R_stage_to_cam,
        P_tag_base=P0,
        R_tag_base=R_base,
        P_tag_to_model=P_tag_to_model,
        R_tag_to_model=R_tag_to_model,
        stage_marker_id=STAGE_ARUCO_ID,
        stage_marker_size_mm=STAGE_ARUCO_MARKER_SIZE_MM,
    )
    print(f"Saved stage kinematic calibration to '{STAGE_KINEMATICS_FILE}'.")


def convert_stage_to_camera_relative_to_model(delta_x, delta_y, delta_z):
    if not os.path.exists(STAGE_KINEMATICS_FILE):
        raise FileNotFoundError(
            f"Missing '{STAGE_KINEMATICS_FILE}'. Run stage kinematic calibration first."
        )

    data = np.load(STAGE_KINEMATICS_FILE)
    R_stage_to_cam = data['R_stage_to_cam']
    P_tag_base = data['P_tag_base']
    R_tag_base = data['R_tag_base']
    P_tag_to_model = data['P_tag_to_model']
    R_tag_to_model = data['R_tag_to_model']

    delta_stage = np.array([delta_x, delta_y, delta_z], dtype=np.float64)
    P_tag_in_cam = P_tag_base + (R_stage_to_cam @ delta_stage)

    R_model_in_cam = R_tag_base @ R_tag_to_model
    P_model_in_cam = P_tag_in_cam + (R_tag_base @ P_tag_to_model)

    R_cam_in_model = R_model_in_cam.T
    P_cam_in_model = -R_cam_in_model @ P_model_in_cam

    euler_xyz = R.from_matrix(R_cam_in_model).as_euler('xyz', degrees=True)
    result = {
        'camera_position_mm': P_cam_in_model,
        'camera_orientation_deg_xyz': euler_xyz,
        'pitch_yaw_roll_deg': np.array([euler_xyz[1], euler_xyz[2], euler_xyz[0]], dtype=np.float64),
        'R_cam_in_model': R_cam_in_model,
    }
    return result


def run_tag_pose_logger():
    """Log camera poses relative to the configured stage tag in an Excel-compatible CSV."""
    print("\n--- TAG POSE LOGGER ---")
    if not os.path.exists(INTRINSIC_FILE):
        print(f"Error: Missing profile config dependency '{INTRINSIC_FILE}'. Run intrinsic pass first.")
        return

    data = np.load(INTRINSIC_FILE)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    axis_length_m = MODE5_CHARUCO_SQUARE_LENGTH_M * 2.0
    mode5_detector = _create_mode5_charuco_detector()
    log_exists = os.path.exists(TAG_POSE_LOG_FILE)
    log_count = 0
    os.makedirs(TAG_POSE_IMAGE_DIRECTORY, exist_ok=True)
    existing_capture_numbers = []
    for image_path in glob.glob(os.path.join(TAG_POSE_IMAGE_DIRECTORY, "Photo*_tag.jpg")):
        image_name = os.path.basename(image_path)
        number_text = image_name[len("Photo"):-len("_tag.jpg")]
        if number_text.isdigit():
            existing_capture_numbers.append(int(number_text))
    next_capture_number = max(existing_capture_numbers, default=0) + 1
    last_processed_revision = None
    current_pose = None
    live_camera_position_mm = None
    live_camera_orientation_deg = None
    current_image_path = None
    vis_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print(
        "Detecting the 50 mm x 50 mm ChArUco board "
        f"({MODE5_MIN_CORNERS}+ corners; max RMSE {MODE5_MAX_REPROJECTION_RMSE_PX:.2f} px)."
    )
    print(
        "Displayed XYZ axes and logged positions use the physical board center as [0, 0, 0]. "
        "Press [SPACEBAR] to log; [Q] to finish."
    )
    cv2.namedWindow("Tag Pose Logger", cv2.WINDOW_NORMAL)

    try:
        while True:
            img_path = get_latest_image(DATA_DIRECTORY)
            image_revision = get_image_revision(img_path) if img_path else None
            if image_revision is not None and image_revision != last_processed_revision:
                time.sleep(0.05)
                frame = load_capsule_frame(img_path)
                if frame is not None:
                    current_image_path = img_path
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    charuco_corners, charuco_ids, marker_corners, marker_ids = mode5_detector.detectBoard(gray)
                    vis_frame = frame.copy()
                    current_pose = None
                    live_camera_position_mm = None
                    live_camera_orientation_deg = None

                    if marker_ids is not None:
                        cv2.aruco.drawDetectedMarkers(vis_frame, marker_corners, marker_ids)
                    if charuco_corners is not None:
                        cv2.aruco.drawDetectedCornersCharuco(vis_frame, charuco_corners, charuco_ids)
                    solved_pose, pose_rejection_reason = _solve_mode5_charuco_pose(
                        charuco_corners, charuco_ids, camera_matrix, dist_coeffs
                    )
                    marker_count = 0 if marker_ids is None else len(marker_ids)
                    charuco_corner_count = 0 if charuco_corners is None else len(charuco_corners)
                    cv2.putText(
                        vis_frame,
                        f"Markers: {marker_count}/12 | ChArUco corners: "
                        f"{charuco_corner_count}/{MODE5_MIN_CORNERS} min",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )
                    if solved_pose is not None:
                        rvec, tvec, reprojection_rmse_px, corner_count = solved_pose
                        current_pose = (rvec, tvec, reprojection_rmse_px, corner_count)
                        R_tag_to_cam, _ = cv2.Rodrigues(rvec)
                        R_cam_to_tag = R_tag_to_cam.T
                        live_camera_position_mm = (-R_cam_to_tag @ tvec).flatten() * 1000.0
                        live_camera_orientation_deg = R.from_matrix(R_cam_to_tag).as_euler(
                            'xyz', degrees=True
                        )
                        cv2.drawFrameAxes(
                            vis_frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m, 2
                        )
                        cv2.putText(vis_frame, "ChArUco tracked: SPACE to log", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(vis_frame, "ChArUco pose rejected", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_frame, pose_rejection_reason, (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    last_processed_revision = image_revision

            display_frame = vis_frame.copy()
            if live_camera_position_mm is not None:
                cv2.putText(
                    display_frame,
                    "Camera in board-center (mm): "
                    f"X={live_camera_position_mm[0]:.2f} "
                    f"Y={live_camera_position_mm[1]:.2f} "
                    f"Z={live_camera_position_mm[2]:.2f}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2,
                )
                cv2.putText(
                    display_frame,
                    "Camera in board-center (deg): "
                    f"Roll={live_camera_orientation_deg[0]:.2f} "
                    f"Pitch={live_camera_orientation_deg[1]:.2f} "
                    f"Yaw={live_camera_orientation_deg[2]:.2f}",
                    (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2,
                )
            cv2.putText(display_frame, f"Logged poses: {log_count}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Tag Pose Logger", display_frame)
            key = cv2.waitKey(15) & 0xFF

            if key == ord('q'):
                break
            if key != ord(' '):
                continue
            if current_pose is None:
                print("--> Capture denied: target marker pose unavailable.")
                continue

            sample_positions_mm = []
            sample_rotations = []
            sample_count = 0

            while sample_count < STAGE_MEASUREMENT_SAMPLES:
                if sample_count == 0:
                    sample_path = current_image_path
                    rvec, tvec, reprojection_rmse_px, corner_count = current_pose
                else:
                    next_path = get_latest_image(DATA_DIRECTORY)
                    next_revision = get_image_revision(next_path) if next_path else None
                    if next_revision is None or next_revision == last_processed_revision:
                        cv2.waitKey(10)
                        continue

                    time.sleep(0.02)
                    sample_frame = load_capsule_frame(next_path)
                    last_processed_revision = next_revision
                    if sample_frame is None:
                        continue

                    sample_gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
                    sample_charuco_corners, sample_charuco_ids, sample_marker_corners, sample_marker_ids = (
                        mode5_detector.detectBoard(sample_gray)
                    )
                    solved_pose, _ = _solve_mode5_charuco_pose(
                        sample_charuco_corners, sample_charuco_ids, camera_matrix, dist_coeffs
                    )
                    if solved_pose is None:
                        continue
                    rvec, tvec, reprojection_rmse_px, corner_count = solved_pose

                    sample_path = next_path
                    current_image_path = next_path
                    vis_frame = sample_frame.copy()
                    if sample_marker_ids is not None:
                        cv2.aruco.drawDetectedMarkers(
                            vis_frame, sample_marker_corners, sample_marker_ids
                        )
                    if sample_charuco_corners is not None:
                        cv2.aruco.drawDetectedCornersCharuco(
                            vis_frame, sample_charuco_corners, sample_charuco_ids
                        )

                R_tag_to_cam, _ = cv2.Rodrigues(rvec)
                R_cam_to_tag = R_tag_to_cam.T
                sample_positions_mm.append((-R_cam_to_tag @ tvec).flatten() * 1000.0)
                sample_rotations.append(R_cam_to_tag)
                sample_count += 1

                cv2.putText(
                    vis_frame,
                    f"Averaging {sample_count}/{STAGE_MEASUREMENT_SAMPLES}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
                cv2.imshow("Tag Pose Logger", vis_frame)
                cv2.waitKey(1)
                current_image_path = sample_path
                print(
                    f"   Sample {sample_count}/{STAGE_MEASUREMENT_SAMPLES} captured "
                    f"({corner_count} corners, RMSE={reprojection_rmse_px:.3f} px)"
                )

            robust_pose = _robust_average_mode5_poses(sample_positions_mm, sample_rotations)
            if robust_pose is None:
                print(
                    f"--> Capture denied: fewer than {MODE5_MIN_INLIER_SAMPLES} samples survived "
                    "outlier rejection. Hold the camera and board steady, then retry."
                )
                continue
            (
                P_cam_in_tag_mm,
                R_cam_to_tag,
                inliers,
                position_errors_mm,
                rotation_errors_deg,
                position_limit_mm,
                rotation_limit_deg,
            ) = robust_pose
            print(
                f"   Outlier rejection kept {np.count_nonzero(inliers)}/{len(inliers)} samples "
                f"(position limit {position_limit_mm:.3f} mm; "
                f"rotation limit {rotation_limit_deg:.3f} deg)."
            )
            euler_cam_in_tag_deg = R.from_matrix(R_cam_to_tag).as_euler('xyz', degrees=True)
            quaternion_cam_in_tag_xyzw = R.from_matrix(R_cam_to_tag).as_quat()
            write_header = not log_exists
            tag_snapshot_name = f"Photo{next_capture_number}_tag.jpg"
            tag_snapshot_path = os.path.join(TAG_POSE_IMAGE_DIRECTORY, tag_snapshot_name)
            no_tag_snapshot_name = f"Photo{next_capture_number}_NoTag.jpg"
            no_tag_snapshot_path = os.path.join(TAG_POSE_IMAGE_DIRECTORY, no_tag_snapshot_name)
            if (
                current_image_path is None
                or os.path.exists(tag_snapshot_path)
                or os.path.exists(no_tag_snapshot_path)
            ):
                print("--> Capture denied: source image unavailable or capture filename already exists.")
                continue

            tag_image_revision = get_image_revision(current_image_path)

            try:
                shutil.move(current_image_path, tag_snapshot_path)
            except OSError as error:
                print(f"--> Capture denied: source image was not moved: {error}")
                continue

            print("Remove the tag, then press [SPACEBAR] to capture the 3D model image.")
            while True:
                no_tag_image_path = get_latest_image(DATA_DIRECTORY)
                no_tag_revision = get_image_revision(no_tag_image_path) if no_tag_image_path else None
                if no_tag_revision is not None and no_tag_revision != tag_image_revision:
                    no_tag_frame = load_capsule_frame(no_tag_image_path)
                    if no_tag_frame is not None:
                        vis_frame = no_tag_frame

                display_frame = vis_frame.copy()
                cv2.putText(
                    display_frame,
                    "Remove tag, then SPACE for model image",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
                cv2.imshow("Tag Pose Logger", display_frame)
                if cv2.waitKey(15) & 0xFF != ord(' '):
                    continue

                if no_tag_revision is None or no_tag_revision == tag_image_revision:
                    print("--> No-tag capture denied: waiting for a new source image.")
                    continue

                try:
                    shutil.move(no_tag_image_path, no_tag_snapshot_path)
                except OSError as error:
                    print(f"--> No-tag capture denied: source image was not moved: {error}")
                    continue
                break

            with open(TAG_POSE_LOG_FILE, 'a', newline='', encoding='utf-8') as log_file:
                writer = csv.writer(log_file)
                if write_header:
                    writer.writerow([
                        'image_file', 'tag_id',
                        'camera_x_mm', 'camera_y_mm', 'camera_z_mm',
                        'camera_orientation_reference_frame',
                        'euler_rotation_convention',
                        'camera_roll_deg', 'camera_pitch_deg', 'camera_yaw_deg',
                        'camera_qx', 'camera_qy', 'camera_qz', 'camera_qw',
                    ])
                writer.writerow([
                    os.path.basename(tag_snapshot_path), 'charuco_50mm_5x5',
                    *P_cam_in_tag_mm,
                    'camera axes relative to 50 mm ChArUco board center',
                    "extrinsic XYZ; scipy Rotation.as_euler('xyz')",
                    *euler_cam_in_tag_deg, *quaternion_cam_in_tag_xyzw,
                ])
            log_exists = True
            log_count += 1
            next_capture_number += 1
            last_processed_revision = no_tag_revision
            print(
                f"--> Logged pose {log_count}: camera in tag frame: "
                f"X={P_cam_in_tag_mm[0]:.3f}, Y={P_cam_in_tag_mm[1]:.3f}, "
                f"Z={P_cam_in_tag_mm[2]:.3f} mm"
            )
    finally:
        cv2.destroyWindow("Tag Pose Logger")

    print(f"Saved {log_count} pose(s) to '{TAG_POSE_LOG_FILE}'.")


def run_mode7_pose_verification():
    """Render each Mode 5 pose and overlay the synthetic board on its source image."""
    print("\n--- MODE 7: TAG POSE SYNTHETIC OVERLAY ---")
    if not os.path.exists(INTRINSIC_FILE):
        print(f"Error: Missing calibration dependency '{INTRINSIC_FILE}'. Run intrinsic pass first.")
        return
    if not os.path.exists(TAG_POSE_LOG_FILE):
        print(f"Error: No Mode 5 pose log found at '{TAG_POSE_LOG_FILE}'.")
        return

    calibration = np.load(INTRINSIC_FILE)
    camera_matrix = calibration['camera_matrix']
    dist_coeffs = calibration['dist_coeffs']
    os.makedirs(MODE7_VERIFICATION_OUTPUT_DIRECTORY, exist_ok=True)
    rendered_count = 0
    stopped_early = False
    print("Press [ENTER] to view the next comparison, or [Q] to stop.")

    with open(TAG_POSE_LOG_FILE, newline='', encoding='utf-8') as pose_file:
        for row_index, row in enumerate(csv.DictReader(pose_file), start=1):
            source_path = os.path.join(TAG_POSE_IMAGE_DIRECTORY, row['image_file'])
            source_image = cv2.imread(source_path)
            if source_image is None:
                print(f"Skipping row {row_index}: could not read '{source_path}'.")
                continue
            real_image = cv2.rotate(source_image, cv2.ROTATE_180)
            R_tag_to_cam, _, t_tag_to_cam_m = _mode7_tag_to_camera_pose(row)
            synthetic, board_mask = _render_mode7_charuco_board(
                real_image.shape, R_tag_to_cam, t_tag_to_cam_m, camera_matrix, dist_coeffs
            )
            comparison, overlay = _mode7_comparison_image(real_image, synthetic, board_mask)
            stem = os.path.splitext(os.path.basename(row['image_file']))[0]
            synthetic_path = os.path.join(
                MODE7_VERIFICATION_OUTPUT_DIRECTORY, f'{row_index:02d}_{stem}_synthetic.png'
            )
            overlay_path = os.path.join(
                MODE7_VERIFICATION_OUTPUT_DIRECTORY, f'{row_index:02d}_{stem}_overlay.png'
            )
            comparison_path = os.path.join(
                MODE7_VERIFICATION_OUTPUT_DIRECTORY, f'{row_index:02d}_{stem}_comparison.png'
            )
            cv2.imwrite(synthetic_path, synthetic)
            cv2.imwrite(overlay_path, overlay)
            cv2.imwrite(comparison_path, comparison)
            cv2.imshow('Mode 7 Pose Verification', overlay)
            rendered_count += 1
            print(f"Showing overlay for '{row['image_file']}' ({rendered_count}); press ENTER for the next image.")
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), ord('Q')):
                stopped_early = True
                break

    cv2.destroyWindow('Mode 7 Pose Verification')
    if stopped_early:
        print("Mode 7 stopped before all comparisons were shown.")
    print(
        f"Saved {rendered_count} synthetic, overlay, and comparison image set(s) to "
        f"'{MODE7_VERIFICATION_OUTPUT_DIRECTORY}'."
    )

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mode = input(
        "Select Execution Mode "
        "('1: intrinsics', '2: boresight', '3: eccentricity', '4: stage', "
        "'5: tag logger', '6: generate mode 5 board', or '7: pose overlay'): "
    ).strip().lower()
    if mode == '1' or mode == 'intrinsics':
        run_intrinsic_calibration()
    elif mode == '2' or mode == 'boresight':
        run_boresight_calibration()
    elif mode == '3' or mode == 'eccentricity':
        run_eccentricity_check()
    elif mode == '4' or mode == 'stage':
        run_stage_kinematic_calibration()
    elif mode == '5' or mode in ('tag logger', 'tag_logger', 'logger'):
        run_tag_pose_logger()
    elif mode == '6' or mode in ('generate board', 'generate_board', 'board'):
        generate_mode5_charuco_board()
    elif mode == '7' or mode in ('pose overlay', 'pose_overlay', 'overlay', 'verify'):
        run_mode7_pose_verification()
    else:
        print(
            "Invalid choice. Select 'intrinsics', 'boresight', 'eccentricity', 'stage', "
            "'tag logger', 'generate board', or 'pose overlay'."
        )