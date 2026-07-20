import os
import glob
import time
import socket
import json
import struct
import threading
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# --- SYSTEM CONFIGURATION ---
DATA_DIRECTORY = r"R:\Release_Internal_V1A\output"
INTRINSIC_FILE = "capsule_intrinsics.npz"
TCP_HOST = 'localhost'
TCP_PORT = 12347

# --- CHARUCO BOARD CONFIGURATION ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CHARUCO_BOARD = cv2.aruco.CharucoBoard((7, 9), 0.026, 0.01976, ARUCO_DICT)
detector = cv2.aruco.CharucoDetector(CHARUCO_BOARD)

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
    
    cv2.namedWindow("Capsule Camera Feed", cv2.WINDOW_NORMAL)
    last_processed_img = None
    last_processed_revision = None
    
    while True:
        img_path = get_latest_image(DATA_DIRECTORY)
        image_revision = get_image_revision(img_path) if img_path else None
        if image_revision is not None and image_revision != last_processed_revision:
            frame = cv2.imread(img_path)
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                imsize = gray.shape[::-1]
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
                frame = cv2.imread(img_path)
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
            frame = cv2.imread(img_path)
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
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mode = input("Select Execution Mode ('1: intrinsics', '2: boresight', or '3: eccentricity'): ").strip().lower()
    if mode == '1' or mode == 'intrinsics':
        run_intrinsic_calibration()
    elif mode == '2' or mode == 'boresight':
        run_boresight_calibration()
    elif mode == '3' or mode == 'eccentricity':
        run_eccentricity_check()
    else:
        print("Invalid choice. Select 'intrinsics', 'boresight', or 'eccentricity'.")