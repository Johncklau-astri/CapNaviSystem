import cv2
import numpy as np
try:
    from pupil_apriltags import Detector
except ImportError:
    print("Error: 'pupil-apriltags' not installed. Run 'pip install pupil-apriltags'.")
    exit()
import os
import serial
import time
import csv
from pathlib import Path
import socket
import json
import struct
import threading
import queue

# --- Configuration ---
TAG_SIZE = 0.06
CUBE_SIZE = 0.08
CHECKERBOARD = (6, 8)
CHECKERBOARD_SQUARE_SIZE = 0.025
CALIBRATION_FILE = "VidAprilTagCalibration.npz"
AXIS_LENGTH = TAG_SIZE
DISPLAY_WIDTH = 1600
DISPLAY_HEIGHT = 1000
NUM_CALIBRATION_FRAMES = 30
Z_SMOOTHING_WINDOW = 5
TAG_IDS = [0, 1, 2, 3, 4, 5]
LOG_FILE = "magnetic_field_log.csv"
cube1MagOffset_X = 0.0
cube1MagOffset_Y = 0.0
cube1MagOffset_Z = 0.112
POSITION_SERVER_HOST = 'localhost'
POSITION_SERVER_PORT = 12346
MOTOR_SERVER_PORT = 12345

# Global variables
magX = 0.0
magY = 0.0
magZ = 0.0
estX = 0.0
estY = 0.0
estZ = 0.0
auto_record_flag = False
rel_Rsensor = np.eye(3)
latest_position_data = None
position_data_lock = threading.Lock()
position_data_queue = queue.Queue()
stop_event = threading.Event()

# Serial connection
try:
    ser = serial.Serial('COM7', 115200, timeout=1)
    ser.flush()
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

def send_position_request(mag_data, quat_data):
    global estX, estY, estZ
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(2.0)
        client_socket.connect((POSITION_SERVER_HOST, POSITION_SERVER_PORT))
        data = {
            'magX': float(mag_data[0]),
            'magY': float(mag_data[1]),
            'magZ': float(mag_data[2]),
            'qw': float(quat_data[0]),
            'qx': float(quat_data[1]),
            'qy': float(quat_data[2]),
            'qz': float(quat_data[3])
        }
        data_str = json.dumps(data)
        data_bytes = data_str.encode('utf-8')
        length = len(data_bytes)
        client_socket.sendall(struct.pack('>I', length))
        client_socket.sendall(data_bytes)
        
        # Receive response
        length_bytes = b''
        while len(length_bytes) < 4:
            chunk = client_socket.recv(4 - len(length_bytes))
            if not chunk:
                raise ConnectionError("Position server disconnected")
            length_bytes += chunk
        length = struct.unpack('>I', length_bytes)[0]
        
        response_bytes = b''
        while len(response_bytes) < length:
            chunk = client_socket.recv(length - len(response_bytes))
            if not chunk:
                raise ConnectionError("Position server disconnected")
            response_bytes += chunk
        
        response = json.loads(response_bytes.decode('utf-8'))
        if 'error' in response:
            print(f"Server error: {response['error']}")
            estX, estY, estZ = 0.0, 0.0, 0.0
        else:
            estX = response['estX']
            estY = response['estY']
            estZ = response['estZ']
    except (ConnectionError, socket.error, json.JSONDecodeError, KeyError) as e:
        print(f"TCP position estimation error: {e}")
        estX, estY, estZ = 0.0, 0.0, 0.0
    except Exception as e:
        print(f"Unexpected TCP error: {e}")
        estX, estY, estZ = 0.0, 0.0, 0.0
    finally:
        if 'client_socket' in locals():
            client_socket.close()
    return estX, estY, estZ

# --- TCP Server for Motor Data ---
def run_tcp_server(host='localhost', port=MOTOR_SERVER_PORT, stop_event=None):
    global latest_position_data, auto_record_flag
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)
        while not stop_event.is_set():
            try:
                client_socket, addr = server_socket.accept()
                client_socket.settimeout(2.0)
                while not stop_event.is_set():
                    length_bytes = b''
                    while len(length_bytes) < 4:
                        chunk = client_socket.recv(4 - len(length_bytes))
                        if not chunk:
                            raise ConnectionError("Client disconnected")
                        length_bytes += chunk
                    length = struct.unpack('>I', length_bytes)[0]
                    data_bytes = b''
                    while len(data_bytes) < length:
                        chunk = client_socket.recv(length - len(data_bytes))
                        if not chunk:
                            raise ConnectionError("Client disconnected")
                        data_bytes += chunk
                    data_str = data_bytes.decode('utf-8')
                    data = json.loads(data_str)
                    if 'record_data' in data and data['record_data']:
                        auto_record_flag = True
                    with position_data_lock:
                        latest_position_data = data
                        position_data_queue.put(data)
                client_socket.close()
            except socket.timeout:
                continue
            except (ConnectionError, socket.error, json.JSONDecodeError) as e:
                print(f"Motor client error: {e}")
                if 'client_socket' in locals():
                    client_socket.close()
                continue
        server_socket.close()
    except Exception as e:
        print(f"Motor TCP server error: {e}")
    finally:
        if 'server_socket' in locals():
            server_socket.close()

# --- Helper Functions ---
def resize_frame(frame, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        data = np.load(CALIBRATION_FILE)
        return data['camera_matrix'], data['dist_coeffs']
    return None, None

def save_calibration(camera_matrix, dist_coeffs):
    np.savez(CALIBRATION_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

def calibrate_camera():
    camera_matrix, dist_coeffs = load_calibration()
    if camera_matrix is not None and dist_coeffs is not None:
        user_input = input("Use existing calibration data? (y/n): ").strip().lower()
        if user_input == 'y':
            return camera_matrix, dist_coeffs
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Could not open camera with index 1.")
        return None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * CHECKERBOARD_SQUARE_SIZE
    objpoints = []
    imgpoints = []
    print(f"Show a 6x8 checkerboard (25mm squares). Press 'c' to capture, 'q' to finish (needs {NUM_CALIBRATION_FRAMES} frames).")
    count = 0
    cv2.namedWindow("Checkerboard Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Checkerboard Calibration", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    while count < NUM_CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
        frame_display = resize_frame(frame)
        cv2.imshow("Checkerboard Calibration", frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ret:
            objpoints.append(objp)
            imgpoints.append(corners2)
            count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(objpoints) < 15:
        print("Insufficient frames for calibration.")
        return None, None
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)
        if mean_error > 0.5:
            print("High reprojection error.")
        save_calibration(camera_matrix, dist_coeffs)
        return camera_matrix, dist_coeffs
    else:
        print("Calibration failed.")
        return None, None

def rotation_matrix_to_euler(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return np.degrees([roll, pitch, yaw])

def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return [qw, qx, qy, qz]

def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length, origin_point):
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.reshape(-1, 2)
    projected_origin = img_points[0].astype(int)
    origin_int = origin_point.astype(int)
    offset = np.abs(projected_origin - origin_int)
    if np.any(offset > 5):
        print(f"Axes origin {projected_origin} does not align with origin point {origin_int}")
    cv2.line(frame, tuple(projected_origin), tuple(img_points[1].astype(int)), (0, 0, 255), 2)  # X: red
    cv2.line(frame, tuple(projected_origin), tuple(img_points[2].astype(int)), (0, 255, 0), 2)  # Y: green
    cv2.line(frame, tuple(projected_origin), tuple(img_points[3].astype(int)), (255, 0, 0), 2)  # Z: blue
    cv2.putText(frame, "X", tuple(img_points[1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Y", tuple(img_points[2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Z", tuple(img_points[3].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def refine_pose(corners, camera_matrix, dist_coeffs):
    obj_points = np.float32([[-TAG_SIZE/2, -TAG_SIZE/2, 0], [TAG_SIZE/2, -TAG_SIZE/2, 0], [TAG_SIZE/2, TAG_SIZE/2, 0], [-TAG_SIZE/2, TAG_SIZE/2, 0]])
    corners = corners.reshape(-1, 2)
    ret, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if ret:
        return rvec, tvec
    return None, None

def magnetic_data_thread(stop_event):
    global magX, magY, magZ
    mag_lock = threading.Lock()
    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('M:'):
                    mag_data = line[2:].split(',')
                    if len(mag_data) == 3:
                        try:
                            sensor_mag_x, sensor_mag_y, sensor_mag_z = map(float, mag_data)
                            with mag_lock:
                                magX = sensor_mag_x
                                magY = -sensor_mag_y
                                magZ = -sensor_mag_z
                        except ValueError:
                            print(f"Error parsing magnetic field data: {line}")
            time.sleep(0.01)
        except Exception as e:
            print(f"Magnetic data thread error: {e}")
            time.sleep(0.1)
    ser.close()

def compute_cube_center(tags, cube_size, tag_ids, reference_tag_id=None):
    cube_tags = [tag for tag in tags if tag['tag_id'] in tag_ids]
    if not cube_tags:
        return None, None
    cube_center_positions = []
    reference_tag = None
    s = cube_size
    for tag in cube_tags:
        tag_id = tag['tag_id']
        R = tag['R']
        tvec = tag['tvec']
        offset_tag = np.array([0.0, 0.0, -s/2], dtype=np.float32)
        cube_center = tvec + R @ offset_tag
        cube_center_positions.append(cube_center)
        if reference_tag_id is not None and tag_id == reference_tag_id:
            reference_tag = tag
    if not cube_center_positions:
        return None, None
    cube_center = np.mean(cube_center_positions, axis=0)
    R_ref = reference_tag['R'] if reference_tag else (cube_tags[0]['R'] if cube_tags else np.eye(3))
    return cube_center, R_ref

def transform_to_ref_cube_frame(point_camera, ref_cube_center, ref_tag_R):
    point_relative = point_camera - ref_cube_center
    point_ref = ref_tag_R.T @ point_relative
    return point_ref

def detect_apriltags(camera_matrix, dist_coeffs):
    global auto_record_flag, rel_Rsensor, estX, estY, estZ
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Could not open camera with index 1.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    detector = Detector(families="tag36h11", nthreads=1, quad_decimate=2.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.5)
    cube1_center_z_buffer = []
    cube2_center_z_buffer = []
    measurement_order = 0
    motor_data = None
    last_motor_update = 0
    MOTOR_UPDATE_INTERVAL = 0.1
    server_thread = threading.Thread(target=run_tcp_server, args=('localhost', MOTOR_SERVER_PORT, stop_event))
    server_thread.daemon = True
    server_thread.start()
    mag_thread = threading.Thread(target=magnetic_data_thread, args=(stop_event,))
    mag_thread.daemon = True
    mag_thread.start()
    
    cv2.namedWindow("AprilTag Detection Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Detection Camera", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    with open(LOG_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Order', 'Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz', 'Cube2_PitchDeg', 'Cube2_RollDeg', 'Cube2_YawDeg',
                         'Rel_Xm', 'Rel_Ym', 'Rel_Zm', 'Mag_XuT', 'Mag_YuT', 'Mag_ZuT', 'Est_Xm', 'Est_Ym', 'Est_Zm'])
        while not stop_event.is_set():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray, estimate_tag_pose=True,
                                        camera_params=(camera_matrix[0, 0], camera_matrix[1, 1],
                                                       camera_matrix[0, 2], camera_matrix[1, 2]),
                                        tag_size=TAG_SIZE)
            
            mag_text = f"Mag: X={magX:.2f} uT, Y={magY:.2f} uT, Z={magZ:.2f} uT"
            cv2.putText(frame, mag_text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
            
            if time.time() - last_motor_update > MOTOR_UPDATE_INTERVAL:
                with position_data_lock:
                    if latest_position_data:
                        motor_data = latest_position_data.copy()
                last_motor_update = time.time()
            
            if motor_data:
                motor_text = [
                    f"Motor Pos: X={motor_data['x']:.2f} cm, Y={motor_data['y']:.2f} cm, Z={motor_data['z']:.2f} cm",
                    f"Motor Quat: [{motor_data['qw']:.3f}, {motor_data['qx']:.3f}, {motor_data['qy']:.3f}, {motor_data['qz']:.3f}]"
                ]
                y_offset = 220
                for line in motor_text:
                    cv2.putText(frame, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    y_offset += 30
            
            tags = []
            tag1_pose = None
            tag2_pose = None
            for det in detections:
                if det.tag_id in TAG_IDS:
                    corners = det.corners
                    for i in range(4):
                        cv2.line(frame, tuple(corners[i].astype(int)), tuple(corners[(i+1)%4].astype(int)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {det.tag_id}", tuple(corners[0].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    center = np.mean(corners, axis=0)
                    cv2.circle(frame, tuple(center.astype(int)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Center ID: {det.tag_id}", (center[0].astype(int) + 10, center[1].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    rvec, tvec = refine_pose(corners, camera_matrix, dist_coeffs)
                    if rvec is None or tvec is None:
                        print(f"Pose refinement failed for tag {det.tag_id}")
                        continue
                    R, _ = cv2.Rodrigues(rvec)
                    tag_data = {'tag_id': det.tag_id, 'tvec': tvec.flatten(), 'R': R, 'center': center}
                    tags.append(tag_data)
                    if det.tag_id == 4:
                        tag1_pose = tag_data
                    if det.tag_id == 1:
                        tag2_pose = tag_data
            
            cube1_center_camera, _ = compute_cube_center(tags, CUBE_SIZE, tag_ids=[3, 4, 5], reference_tag_id=4)
            if cube1_center_camera is not None:
                cube1_center_z_buffer.append(cube1_center_camera[2])
                if len(cube1_center_z_buffer) > Z_SMOOTHING_WINDOW:
                    cube1_center_z_buffer.pop(0)
                smoothed_z1 = np.mean(cube1_center_z_buffer)
                cube1_center_camera_smoothed = cube1_center_camera.copy()
                cube1_center_camera_smoothed[2] = smoothed_z1
                cube1_center_point = np.array([[cube1_center_camera_smoothed]], dtype=np.float32)
                img_points, _ = cv2.projectPoints(cube1_center_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                img_cube1_center = img_points[0][0].astype(int)
                cv2.circle(frame, tuple(img_cube1_center), 8, (255, 255, 0), -1)
                cv2.putText(frame, f"Cube 1 Center: x={cube1_center_camera_smoothed[0]:.3f} m, y={cube1_center_camera_smoothed[1]:.3f} m, z={cube1_center_camera_smoothed[2]:.3f} m",
                            (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, "Origin Cube", (img_cube1_center[0] + 10, img_cube1_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if tag1_pose is not None:
                    R_tag4 = tag1_pose['R']
                    theta_cube1 = np.radians(270)
                    R_modifier_cube1 = np.array([[1, 0, 0], [0, np.cos(theta_cube1), -np.sin(theta_cube1)], [0, np.sin(theta_cube1), np.cos(theta_cube1)]])
                    R_modified_cube1 = R_tag4 @ R_modifier_cube1
                    cube1_rvec, _ = cv2.Rodrigues(R_modified_cube1)
                    draw_axes(frame, camera_matrix, dist_coeffs, cube1_rvec, cube1_center_camera_smoothed, AXIS_LENGTH, img_cube1_center)
            
            cube2_center_camera, _ = compute_cube_center(tags, CUBE_SIZE, tag_ids=[0, 1, 2], reference_tag_id=1)
            if cube2_center_camera is not None:
                cube2_center_z_buffer.append(cube2_center_camera[2])
                if len(cube2_center_z_buffer) > Z_SMOOTHING_WINDOW:
                    cube2_center_z_buffer.pop(0)
                smoothed_z2 = np.mean(cube2_center_z_buffer)
                cube2_center_camera_smoothed = cube2_center_camera.copy()
                cube2_center_camera_smoothed[2] = smoothed_z2
                cube2_center_point = np.array([[cube2_center_camera_smoothed]], dtype=np.float32)
                img_points, _ = cv2.projectPoints(cube2_center_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                img_cube2_center = img_points[0][0].astype(int)
                cv2.circle(frame, tuple(img_cube2_center), 8, (0, 255, 255), -1)
                if tag2_pose is not None:
                    R_tag1 = tag2_pose['R']
                    theta_x = np.radians(270)
                    theta_y = np.radians(0)
                    R_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
                    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
                    R_modifier_cube2 = R_y @ R_x
                    theta_z = np.radians(0)
                    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
                    R_modifier_cube2 = R_modifier_cube2 @ R_z
                    R_modified_cube2 = R_tag1 @ R_modifier_cube2
                    cube2_rvec, _ = cv2.Rodrigues(R_modified_cube2)
                    draw_axes(frame, camera_matrix, dist_coeffs, cube2_rvec, cube2_center_camera_smoothed, AXIS_LENGTH, img_cube2_center)
                    if cube1_center_camera is not None and tag1_pose is not None:
                        cube2_center_ref = transform_to_ref_cube_frame(cube2_center_camera_smoothed, cube1_center_camera_smoothed, R_modified_cube1)
                        cube2_center_ref[0] += cube1MagOffset_X
                        cube2_center_ref[1] += cube1MagOffset_Y
                        cube2_center_ref[2] += cube1MagOffset_Z
                        cv2.putText(frame, f"Cube 2 Center (Ref): x={cube2_center_ref[0]:.3f} m, y={cube2_center_ref[1]:.3f} m, z={cube2_center_ref[2]:.3f} m",
                                    (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if tag2_pose is not None:
                            rel_Rsensor = R_modified_cube1.T @ R_modified_cube2
                            cube2_quat = rotation_matrix_to_quaternion(rel_Rsensor)
                            cube2_euler = rotation_matrix_to_euler(rel_Rsensor)
                            cv2.putText(frame, f"Cube 2 Orient: Pitch={cube2_euler[1]:.1f}°, Roll={cube2_euler[0]:.1f}°, Yaw={cube2_euler[2]:.1f}°",
                                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        distance = np.linalg.norm(img_cube1_center - img_cube2_center)
                        if distance < 20:
                            print(f"Cube centers too close in image: {distance:.2f} pixels")

            cube2_quat = rotation_matrix_to_quaternion(rel_Rsensor)
            estX, estY, estZ = send_position_request([magX, magY, magZ], cube2_quat)
            
            cv2.putText(frame, f"Est Pos: x={estX:.3f} m, y={estY:.3f} m, z={estZ:.3f} m",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t') or auto_record_flag:
                mag_x, mag_y, mag_z = magX, magY, magZ
                if mag_x is not None and mag_y is not None and mag_z is not None and tag2_pose is not None:
                    measurement_order += 1
                    cube2_quat = rotation_matrix_to_quaternion(rel_Rsensor)
                    errorX = estX - cube2_center_ref[0]
                    errorY = estY - cube2_center_ref[1] 
                    errorZ = estZ - cube2_center_ref[2]
                    total_error = np.sqrt(errorX**2 + errorY**2 + errorZ**2)
                    writer.writerow([
                        measurement_order,
                        f'{cube2_quat[0]:.4f}', f'{cube2_quat[1]:.4f}', f'{cube2_quat[2]:.4f}', f'{cube2_quat[3]:.3f}',
                        f'{cube2_euler[2]:.2f}', f'{cube2_euler[0]:.2f}', f'{cube2_euler[2]:.2f}',
                        f'{cube2_center_ref[0]:.6f}', f'{cube2_center_ref[1]:.6f}', f'{cube2_center_ref[2]:.6f}',
                        f'{mag_x:.3f}', f'{mag_y:.3f}', f'{mag_z:.3f}',
                        f'{estX:.3f}', f'{estY:.3f}', f'{estZ:.3f}',
                        f'{errorX:.3f}', f'{errorY:.3f}', f'{errorZ:.3f}', f'{total_error:.3f}'
                    ])
                    csvfile.flush()
                    cv2.putText(frame, "MAG FIELD & POS LOGGED", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    time.sleep(0.2)
                    auto_record_flag = False
            
            frame_display = resize_frame(frame)
            cv2.imshow("AprilTag Detection Camera", frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    
    cap.release()
    server_thread.join()
    mag_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        camera_matrix, dist_coeffs = calibrate_camera()
        if camera_matrix is not None and dist_coeffs is not None:
            detect_apriltags(camera_matrix, dist_coeffs)
    except Exception as e:
        print(f"Main program error: {e}")
    finally:
        stop_event.set()