
## FAST CYCLE

import cv2
import numpy as np
try:
    from pupil_apriltags import Detector
except ImportError:
    print("Error: 'pupil-apriltags' not installed. Run 'pip install pupil-apriltags'.")
    exit()
import os
import time
import csv
from pathlib import Path
import socket
import json
import struct
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import select
import serial

# --- Configuration ---
## side camera id is 0
## top camera id is 3
camera_id = 0
top_camera_id = 3

TAG_SIZE = 0.06
CUBE_SIZE = 0.08
CHECKERBOARD = (6, 8)
CHECKERBOARD_SQUARE_SIZE = 0.025

# CALIBRATION_FILE = "VidAprilTagCalibration.npz"
# TOP_CALIBRATION_FILE = "VidAprilTagCalibration.npz"

AXIS_LENGTH = TAG_SIZE
DISPLAY_WIDTH = 1920    
DISPLAY_HEIGHT = 1080  # Total height: 600 for text, 600 for camera
TEXT_FRAME_WIDTH = 1720
TEXT_FRAME_HEIGHT = 600
CAMERA_FRAME_HEIGHT = 500
NUM_CALIBRATION_FRAMES = 50
Z_SMOOTHING_WINDOW = 5
TAG_IDS = [0, 1, 2, 3, 4, 5 ,7,8,9]
LOG_FILE = "magnetic_field_log.csv"

cube2MagOffset_X = 0.004
cube2MagOffset_Y = 0.0
cube2MagOffset_Z = -0.105 

top_R_1000_cube1_camera = None
R_1000_cube1_camera = None


##### Position weights
SIDE_POS_WEIGHTS = {
    'x': 0.05,  # Side camera is bad at X
    'y': 0.9,  # Side camera is okay at Y  
    'z': 0.95   # Side camera is good at Z
}

TOP_POS_WEIGHTS = {
    'x': 0.95,  # Top camera is good at X
    'y': 0.1,  # Top camera is good at Y
    'z': 0.05   # Top camera is bad at Z
}

# Orientation weights
SIDE_ORI_WEIGHTS = {
    'roll': 0.9,   # Side camera good at roll (X rotation)
    'pitch': 0.5,  # Side camera good at pitch (Y rotation)
    'yaw': 0.1     # Side camera bad at yaw (Z rotation)
}

TOP_ORI_WEIGHTS = {
    'roll': 0.1,   # Top camera bad at roll
    'pitch': 0.5,  # Top camera bad at pitch
    'yaw': 0.9     # Top camera good at yaw
}

POSITION_SERVER_HOST = 'localhost'
POSITION_SERVER_PORT = 12346
MOTOR_SERVER_PORT = 12345
BT_ANGLE_EST_PORT = 12347  # Port for receiving data from bt_angleEst

# Global variables
magX = 0.0
magY = 0.0
magZ = 0.0
estX = 0.0
estY = 0.0
estZ = 0.0
bt_quat_global = [1.0, 0.0, 0.0, 0.0]  # Quaternion from bt_angleEst
R_senToCube = np.eye(3)

# ADD THESE LINES FOR SERIAL DATA
serial_magX = 0.0
serial_magY = 0.0
serial_magZ = 0.0
serial_mag_lock = threading.Lock()
serial_port = None


# Thread-safe queues
bt_data_queue = queue.Queue(maxsize=100)  # Buffer up to 100 samples
position_request_queue = queue.Queue(maxsize=10)  # Position estimation requests
position_response_queue = queue.Queue(maxsize=10)  # Position estimation responses

R_worldCube_camera = None
top_R_worldCube_camera = None

auto_record_flag = False
rel_Rsensor = np.eye(3)
latest_position_data = None
latest_bt_data = None
position_data_lock = threading.Lock()
bt_data_lock = threading.Lock()
position_data_queue = queue.Queue()
stop_event = threading.Event()

# Position estimation thread control
position_thread_running = True
position_socket = None
position_socket_lock = threading.Lock()

# Add serial port initialization
def init_serial_connection(port='COM19', baudrate=115200):
    global serial_port
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.flush()
        time.sleep(2)
        print(f"Serial port {port} opened successfully at {baudrate} baud")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        print("Continuing without serial data...")
        return None
# Initialize serial connection
serial_port = init_serial_connection('COM19', 115200)  # Change COM port as needed

def run_serial_data_receiver(stop_event):
    """Receive magnetic field data from Arduino via serial"""
    global serial_magX, serial_magY, serial_magZ, serial_port
    
    if serial_port is None:
        print("Serial port not available, skipping serial data thread")
        return
    
    buffer = ""
    
    while not stop_event.is_set():
        try:
            if serial_port.in_waiting > 0:
                # Read available data
                data = serial_port.read(serial_port.in_waiting).decode('utf-8', errors='ignore')
                buffer += data
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line.startswith('M:'):
                        mag_data = line[2:].split(',')
                        if len(mag_data) == 3:
                            try:
                                sensor_mag_x, sensor_mag_y, sensor_mag_z = map(float, mag_data)
                                with serial_mag_lock:
                                    # Apply coordinate transformations if needed
                                    serial_magX = sensor_mag_x
                                    serial_magY = -sensor_mag_y  # Invert Y to match coordinate system
                                    serial_magZ = -sensor_mag_z  # Invert Z to match coordinate system
                                    # print(f"Raw Serial Mag: X={sensor_mag_x:.2f}, Y={sensor_mag_y:.2f}, Z={sensor_mag_z:.2f}")
                                    
                                    # Optional: print at lower rate to avoid console spam
                                    if not hasattr(run_serial_data_receiver, "last_print") or \
                                       time.time() - run_serial_data_receiver.last_print > 1.0:
                                        # print(f"Serial Mag: X={serial_magX:.2f}, Y={serial_magY:.2f}, Z={serial_magZ:.2f}")
                                        run_serial_data_receiver.last_print = time.time()
                            except ValueError as e:
                                print(f"Error parsing magnetic field data: {line} - {e}")
                    elif line.startswith('M:Error'):
                        print(f"Arduino error: {line}")
            time.sleep(0.005)  # 5ms polling interval
            
        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
            # Attempt to reconnect
            try:
                if serial_port:
                    serial_port.close()
                time.sleep(1)
                serial_port = init_serial_connection('COM20', 115200)
            except Exception as reconnect_error:
                print(f"Failed to reconnect: {reconnect_error}")
                time.sleep(5)
        except Exception as e:
            print(f"Serial data thread error: {e}")
            time.sleep(0.1)
    
    if serial_port:
        serial_port.close()
    print("Serial data thread stopped")



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

def run_bt_data_receiver(host='localhost', port=BT_ANGLE_EST_PORT, stop_event=None):
    """Receive magnetometer and quaternion data from bt_angleEst.py"""
    global latest_bt_data
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(2.0)
        client_socket.connect((host, port))
        print(f"Connected to bt_angleEst TCP server at {host}:{port}")
        
        while not stop_event.is_set():
            try:
                # Receive length prefix
                length_bytes = b''
                while len(length_bytes) < 4 and not stop_event.is_set():
                    try:
                        chunk = client_socket.recv(4 - len(length_bytes))
                        if not chunk:
                            raise ConnectionError("bt_angleEst disconnected")
                        length_bytes += chunk
                    except socket.timeout:
                        continue
                
                if len(length_bytes) < 4:
                    continue
                    
                length = struct.unpack('>I', length_bytes)[0]
                
                # Receive data
                data_bytes = b''
                while len(data_bytes) < length and not stop_event.is_set():
                    try:
                        chunk = client_socket.recv(length - len(data_bytes))
                        if not chunk:
                            raise ConnectionError("bt_angleEst disconnected")
                        data_bytes += chunk
                    except socket.timeout:
                        continue
                
                if len(data_bytes) < length:
                    continue
                
                # Parse JSON data
                data_str = data_bytes.decode('utf-8')
                data = json.loads(data_str)
                
                # Put data in queue (non-blocking, drop oldest if full)
                try:
                    bt_data_queue.put_nowait(data)
                except queue.Full:
                    # Remove oldest item to make space
                    try:
                        bt_data_queue.get_nowait()
                    except queue.Empty:
                        pass
                    bt_data_queue.put_nowait(data)
                    
            except (ConnectionError, socket.error, json.JSONDecodeError) as e:
                print(f"bt_angleEst data receiver error: {e}")
                # Attempt to reconnect
                try:
                    client_socket.close()
                except:
                    pass
                time.sleep(2)
                try:
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client_socket.settimeout(2.0)
                    client_socket.connect((host, port))
                    print("Reconnected to bt_angleEst TCP server")
                except:
                    print("Failed to reconnect to bt_angleEst")
                    time.sleep(5)
            except Exception as e:
                print(f"Unexpected error in bt data receiver: {e}")
                time.sleep(1)
                
    except Exception as e:
        print(f"bt_angleEst receiver setup error: {e}")
    finally:
        try:
            client_socket.close()
        except:
            pass

def get_latest_bt_data():
    """Get the most recent BT data, clearing the queue"""
    latest_data = None
    while not bt_data_queue.empty():
        try:
            latest_data = bt_data_queue.get_nowait()
        except queue.Empty:
            break
    return latest_data

def position_estimation_worker():
    """Dedicated thread for handling position estimation requests and responses"""
    global position_socket, position_thread_running, estX, estY, estZ
    
    def connect_to_position_server():
        """Helper function to connect to position server"""
        try:
            socket_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket_obj.settimeout(5.0)
            socket_obj.connect((POSITION_SERVER_HOST, POSITION_SERVER_PORT))
            socket_obj.settimeout(0.1)  # Short timeout for non-blocking ops
            print(f"Connected to position server at {POSITION_SERVER_HOST}:{POSITION_SERVER_PORT}")
            return socket_obj
        except Exception as e:
            # print(f"Failed to connect to position server: {e}")
            return None
    
    def send_position_request(socket_obj, magX, magY, magZ, quat_w, quat_x, quat_y, quat_z):
        """Send request to position server"""
        try:
            data = {
                'magX': magX,
                'magY': magY, 
                'magZ': magZ,
                'qw': quat_w,
                'qx': quat_x,
                'qy': quat_y,
                'qz': quat_z
            }
            
            data_str = json.dumps(data)
            data_bytes = data_str.encode('utf-8')
            length = len(data_bytes)
            
            # Send length prefix
            length_bytes = struct.pack('>I', length)
            socket_obj.sendall(length_bytes)
            
            # Send data
            socket_obj.sendall(data_bytes)
            
            return True
        except Exception as e:
            # print(f"Error sending position request: {e}")
            return False
    
    def receive_position_response(socket_obj):
        """Receive response from position server (non-blocking)"""
        try:
            # Check if data is available
            ready = select.select([socket_obj], [], [], 0.01)  # 10ms timeout
            if not ready[0]:
                return None  # No data available
            
            # Read response length
            length_bytes = b''
            while len(length_bytes) < 4:
                chunk = socket_obj.recv(4 - len(length_bytes))
                if not chunk:
                    raise ConnectionError("Connection closed")
                length_bytes += chunk
            
            length = struct.unpack('>I', length_bytes)[0]
            
            # Read response data
            data_bytes = b''
            while len(data_bytes) < length:
                chunk = socket_obj.recv(length - len(data_bytes))
                if not chunk:
                    raise ConnectionError("Connection closed")
                data_bytes += chunk
            
            # Parse response
            response_str = data_bytes.decode('utf-8')
            response = json.loads(response_str)
            return response
        
        except socket.timeout:
            return None  # No data available, normal condition
        except BlockingIOError:
            return None  # No data available, normal condition
        except Exception as e:
            print(f"Error receiving position response: {e}")
            return None
    
    # Main worker loop
    pending_request = None
    response_buffer = b''
    expected_response_length = 0
    
    while position_thread_running and not stop_event.is_set():
        try:
            # Get new request from queue (non-blocking)
            try:
                if pending_request is None:
                    pending_request = position_request_queue.get_nowait()
                    # print("Got new position estimation request from queue")
            except queue.Empty:
                pass  # No new requests
            
            # Ensure we have a connection
            if position_socket is None:
                with position_socket_lock:
                    position_socket = connect_to_position_server()
                if position_socket is None:
                    time.sleep(0.1)  # Wait before retrying connection
                    continue
            
            # Process pending request
            if pending_request is not None:
                magX, magY, magZ, quat_w, quat_x, quat_y, quat_z = pending_request
                
                # Send request
                success = send_position_request(position_socket, magX, magY, magZ, 
                                              quat_w, quat_x, quat_y, quat_z)
                if success:
                    # print(f"Sent position request - Mag: ({magX:.2f}, {magY:.2f}, {magZ:.2f})")
                    pending_request = None  # Request sent, wait for response
                else:
                    print("Failed to send position request, will retry")
                    with position_socket_lock:
                        if position_socket:
                            try:
                                position_socket.close()
                            except:
                                pass
                            position_socket = None
                    time.sleep(0.1)
            
            # Check for responses (non-blocking)
            if position_socket:
                response = receive_position_response(position_socket)
                if response:
                    # print(f"Received position response: {response}")
                    
                    if 'error' in response:
                        print(f"Server error: {response['error']}")
                        estX, estY, estZ = 0.0, 0.0, 0.0
                    else:
                        estX = response['estX']
                        estY = response['estY']
                        estZ = response['estZ']
                        # print(f"Updated position: ({estX:.3f}, {estY:.3f}, {estZ:.3f})")
                    
                    # Put response in queue for main thread (if needed)
                    try:
                        position_response_queue.put_nowait((estX, estY, estZ))
                    except queue.Full:
                        pass  # Response queue full, drop the response
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
        except Exception as e:
            print(f"Error in position estimation worker: {e}")
            with position_socket_lock:
                if position_socket:
                    try:
                        position_socket.close()
                    except:
                        pass
                    position_socket = None
            time.sleep(0.1)
    
    # Cleanup
    with position_socket_lock:
        if position_socket:
            try:
                position_socket.close()
            except:
                pass
            position_socket = None
    print("Position estimation worker stopped")

def request_position_estimation(magX, magY, magZ, quat_w, quat_x, quat_y, quat_z):
    """Non-blocking request for position estimation"""
    try:
        position_request_queue.put_nowait((magX, magY, magZ, quat_w, quat_x, quat_y, quat_z))
        return True
    except queue.Full:
        # print("Position request queue full, dropping request")
        return False

def get_latest_position_estimate():
    """Get the latest position estimate (non-blocking)"""
    try:
        # Get the most recent response from queue
        latest_response = None
        while not position_response_queue.empty():
            latest_response = position_response_queue.get_nowait()
        return latest_response
    except queue.Empty:
        return None

def update_position_estimation(q,magX, magY, magZ):
    """Update position estimation in a non-blocking way"""    
    success = False
    if q is not None:
        # Send non-blocking request for position estimation
        success = request_position_estimation(magX, magY, magZ, 
                                            q[0], q[1], 
                                            q[2], q[3])
    if success:
        return True
   

def resize_frame(frame, max_height=CAMERA_FRAME_HEIGHT):
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    new_h = max_height
    new_w = int(new_h * aspect_ratio)
    if new_h <= 0 or new_w <= 0:
        print("Invalid resize dimensions.")
        return frame
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if new_w < DISPLAY_WIDTH:
        border_left = (DISPLAY_WIDTH - new_w) // 2
        border_right = DISPLAY_WIDTH - new_w - border_left
        padded_frame = cv2.copyMakeBorder(resized_frame, 0, 0, border_left, border_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded_frame
    return resized_frame

# def load_calibration():
#     if os.path.exists(CALIBRATION_FILE):
#         data = np.load(CALIBRATION_FILE)
#         return data['camera_matrix'], data['dist_coeffs']
#     return None, None

# def save_calibration(camera_matrix, dist_coeffs):
#     np.savez(CALIBRATION_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
def calibrate_camera():
    """
    Calibrate both side and top cameras.
    
    Returns:
        side_camera_matrix, side_dist_coeffs, top_camera_matrix, top_dist_coeffs
    """
    # Define calibration files for each camera
    SIDE_CALIBRATION_FILE = "side_camera_calibration.npz"
    TOP_CALIBRATION_FILE = "top_camera_calibration.npz"
    
    # Helper function to load calibration if exists
    def load_camera_calibration(calib_file):
        if os.path.exists(calib_file):
            data = np.load(calib_file)
            return data['camera_matrix'], data['dist_coeffs']
        return None, None
    
    # Helper function to save calibration
    def save_camera_calibration(calib_file, camera_matrix, dist_coeffs):
        np.savez(calib_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    
    # Helper function to perform actual calibration for a specific camera
    def calibrate_specific_camera(cam_id, calib_file, camera_name):
        print(f"\n=== CALIBRATING {camera_name.upper()} CAMERA (ID: {cam_id}) ===")
        
        # Check for existing calibration
        existing_matrix, existing_dist = load_camera_calibration(calib_file)
        if existing_matrix is not None and existing_dist is not None:
            print(f"\nExisting calibration found for {camera_name} camera:")
            print(f"Camera Matrix:\n{existing_matrix}")
            print(f"Distortion Coefficients: {existing_dist.flatten()}")
            
            user_input = input(f"Use existing calibration data for {camera_name} camera? (y/n): ").strip().lower()
            if user_input == 'y':
                return existing_matrix, existing_dist
        
        # If no existing calibration or user wants to recalibrate
        print(f"\nStarting new calibration for {camera_name} camera...")
        print(f"Show a 6x8 checkerboard (25mm squares) to the {camera_name} camera.")
        print(f"Press 'c' to capture, 'q' to finish (need at least 15 frames).")
        
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"Could not open {camera_name} camera with index {cam_id}.")
            return None, None
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * CHECKERBOARD_SQUARE_SIZE
        objpoints = []
        imgpoints = []
        
        count = 0
        cv2.namedWindow(f"{camera_name} Camera Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{camera_name} Camera Calibration", DISPLAY_WIDTH, DISPLAY_HEIGHT)
        
        while count < NUM_CALIBRATION_FRAMES:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame from {camera_name} camera.")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
                
            frame_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imshow(f"{camera_name} Camera Calibration", frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and ret:
                objpoints.append(objp)
                imgpoints.append(corners2)
                count += 1
                print(f"Captured {count}/{NUM_CALIBRATION_FRAMES}")
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyWindow(f"{camera_name} Camera Calibration")
        
        if len(objpoints) < 15:
            print(f"Insufficient frames ({len(objpoints)}) for {camera_name} camera calibration.")
            return None, None
            
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                                 camera_matrix, dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            mean_error /= len(objpoints)
            
            # Display calibration results
            print(f"\n=== {camera_name.upper()} CAMERA CALIBRATION RESULTS ===")
            print(f"Camera Matrix:\n{camera_matrix}")
            print(f"Distortion Coefficients: {dist_coeffs.flatten()}")
            print(f"Reprojection Error: {mean_error:.4f} pixels")
            print(f"Focal Length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
            print(f"Principal Point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")
            
            if mean_error > 0.5:
                print(f"WARNING: High reprojection error - consider recalibrating {camera_name} camera")
            
            # Save calibration
            save_camera_calibration(calib_file, camera_matrix, dist_coeffs)
            print(f"Calibration saved to {calib_file}")
            
            return camera_matrix, dist_coeffs
        else:
            print(f"Calibration failed for {camera_name} camera.")
            return None, None
    
    # Calibrate side camera (ID: 0)
    side_matrix, side_dist = calibrate_specific_camera(camera_id, SIDE_CALIBRATION_FILE, "side")
    
    # Calibrate top camera (ID: 3)
    top_matrix, top_dist = calibrate_specific_camera(top_camera_id, TOP_CALIBRATION_FILE, "top")
    
    return side_matrix, side_dist, top_matrix, top_dist


# def calibrate_camera():
#     camera_matrix, dist_coeffs = load_calibration()
#     if camera_matrix is not None and dist_coeffs is not None:
#         user_input = input("Use existing calibration data? (y/n): ").strip().lower()
#         if user_input == 'y':
#             # Display existing calibration parameters
#             print("\n=== EXISTING CALIBRATION PARAMETERS ===")
#             print(f"Camera Matrix:\n{camera_matrix}")
#             print(f"Distortion Coefficients: {dist_coeffs.flatten()}")
#             return camera_matrix, dist_coeffs
#     cap = cv2.VideoCapture(camera_id)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
#     cap.set(cv2.CAP_PROP_FPS, 30)

#     if not cap.isOpened():
#         print(f"Could not open camera with index {camera_id}.")
#         return None, None
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
#     objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * CHECKERBOARD_SQUARE_SIZE
#     objpoints = []
#     imgpoints = []
#     print(f"Show a 6x8 checkerboard (25mm squares). Press 'c' to capture, 'q' to finish (needs {NUM_CALIBRATION_FRAMES} frames).")
#     count = 0
#     cv2.namedWindow("Checkerboard Calibration", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Checkerboard Calibration", DISPLAY_WIDTH, DISPLAY_HEIGHT)
#     while count < NUM_CALIBRATION_FRAMES:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame.")
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
#         if ret:
#             corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#             cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
#         frame_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
#         cv2.imshow("Checkerboard Calibration", frame_display)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('c') and ret:
#             objpoints.append(objp)
#             imgpoints.append(corners2)
#             count += 1
#             print("Captured")
#         elif key == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     if len(objpoints) < 15:
#         print("Insufficient frames for calibration.")
#         return None, None
#     ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



#     if ret:
#         mean_error = 0
#         for i in range(len(objpoints)):
#             imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
#             error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#             mean_error += error
#         mean_error /= len(objpoints)
        
#         # Display calibration results
#         print("\n=== CALIBRATION RESULTS ===")
#         print(f"Camera Matrix:\n{camera_matrix}")
#         print(f"Distortion Coefficients: {dist_coeffs.flatten()}")
#         print(f"Reprojection Error: {mean_error:.4f} pixels")
#         print(f"Focal Length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
#         print(f"Principal Point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")
        
#         if mean_error > 0.5:
#             print("WARNING: High reprojection error - consider recalibrating")
        
#         save_calibration(camera_matrix, dist_coeffs)
#         return camera_matrix, dist_coeffs
#     else:
#         print("Calibration failed.")
#         return None, None

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
    """Convert rotation matrix to quaternion [qw, qx, qy, qz]"""
    if R is None:
        return [1.0, 0.0, 0.0, 0.0]
    
    # Ensure R is a proper 3x3 matrix
    if R.shape != (3, 3):
        print(f"Warning: Invalid rotation matrix shape {R.shape}, returning identity quaternion")
        return [1.0, 0.0, 0.0, 0.0]
    
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


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [qw, qx, qy, qz] to rotation matrix"""
    qw, qx, qy, qz = q
    # Normalize quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Calculate rotation matrix
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length, origin_point, draw_labels=True):
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.reshape(-1, 2)
    projected_origin = img_points[0].astype(int)  # Fixed: astype
    origin_int = origin_point.astype(int)  # Fixed: astype
    offset = np.abs(projected_origin - origin_int)
    if np.any(offset > 5):
        print(f"Axes origin {projected_origin} does not align with origin point {origin_int}")
    cv2.line(frame, tuple(projected_origin), tuple(img_points[1].astype(int)), (0, 0, 255), 2)  # Fixed: astype
    cv2.line(frame, tuple(projected_origin), tuple(img_points[2].astype(int)), (0, 255, 0), 2)  # Fixed: astype
    cv2.line(frame, tuple(projected_origin), tuple(img_points[3].astype(int)), (255, 0, 0), 2)  # Fixed: astype
    if draw_labels:
        cv2.putText(frame, "X", tuple(img_points[1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Fixed: astype
        cv2.putText(frame, "Y", tuple(img_points[2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Fixed: astype
        cv2.putText(frame, "Z", tuple(img_points[3].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)  # Fixed: astype
        
def refine_pose(corners, camera_matrix, dist_coeffs):
    obj_points = np.float32([[-TAG_SIZE/2, -TAG_SIZE/2, 0], [TAG_SIZE/2, -TAG_SIZE/2, 0], [TAG_SIZE/2, TAG_SIZE/2, 0], [-TAG_SIZE/2, TAG_SIZE/2, 0]])
    corners = corners.reshape(-1, 2)
    ret, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if ret:
        return rvec, tvec
    return None, None


def compute_cube_center(tags, cube_size, tag_ids, reference_tag_id=None):
    cube_tags = [tag for tag in tags if tag['tag_id'] in tag_ids]
    if len(cube_tags) == 0:
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

def draw_text_with_background(frame, text, org, font, font_scale, color, thickness, bg_color=(0, 0, 0), alpha=0.5):
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top_left = (org[0] - 5, org[1] - text_size[1] - 5)
    bottom_right = (org[0] + text_size[0] + 5, org[1] + baseline + 5)
    overlay = frame.copy()
    # cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, org, font, font_scale, color, thickness)

def compute_cube_orientation(tags, tag_ids):
    """
    Compute cube orientation by averaging the orientation of multiple tags
    
    Args:
        tags: List of tag data dictionaries
        tag_ids: List of tag IDs to use for this cube
    
    Returns:
        avg_R: Average rotation matrix (3x3) representing cube orientation
    """
    cube_tags = [tag for tag in tags if tag['tag_id'] in tag_ids]
    if not cube_tags:
        return np.eye(3)  # Return identity if no tags detected
    
    cube_rotations = []
    
    for tag in cube_tags:
        tag_id = tag['tag_id']
        R_tag = tag['R']  # Tag orientation in camera coordinates
        
        # Apply tag-specific adjustment
        R_adjust = get_tag_rotation_adjustment(tag_id)
        
        # Apply adjustment: R_cube = R_tag * R_adjust
        R_cube = R_tag @ R_adjust
        
        cube_rotations.append(R_cube)
    
    # Average the rotation matrices using quaternion averaging
    return average_rotation_matrices(cube_rotations)

def get_tag_rotation_adjustment(tag_id):
    """
    Define rotation adjustments for each tag using manual rotation matrix construction
    Modify these values based on how each tag is mounted on the cube
    """
    if tag_id == 0:  # Cube 2 - Tag 0
        # Example: Adjust as needed for tag 0 mounting
        theta_x = np.radians(0)
        theta_y = np.radians(0) 
        theta_z = np.radians(0)
        
    elif tag_id == 1:  # Cube 2 - Tag 1 (front face)
        # Your existing adjustment for tag 1
        theta_x = np.radians(270)
        theta_y = np.radians(0)
        theta_z = np.radians(0)
        
    elif tag_id == 2:  # Cube 2 - Tag 2
        # Example: Adjust as needed for tag 2 mounting
        theta_x = np.radians(-90)
        theta_y = np.radians(-90)
        theta_z = np.radians(0)
        
    elif tag_id == 3:  # Cube 1 - Tag 3
        # Example: Adjust as needed for tag 3 mounting
        theta_x = np.radians(0)
        theta_y = np.radians(0)
        theta_z = np.radians(0)
        
    elif tag_id == 4:  # Cube 1 - Tag 4 (front face)
        # Your existing adjustment for tag 4
        theta_x = np.radians(270)
        theta_y = np.radians(0)
        theta_z = np.radians(0)
        
    elif tag_id == 5:  # Cube 1 - Tag 5
        # Example: Adjust as needed for tag 5 mounting
        theta_x = np.radians(-90)
        theta_y = np.radians(-90)
        theta_z = np.radians(0)

    elif tag_id == 7:
        # Example: Adjust as needed for tag 7 mounting
        theta_x = np.radians(0)
        theta_y = np.radians(-90)
        theta_z = np.radians(90)

    elif tag_id == 8:
        # Example: Adjust as needed for tag 8 mounting
        theta_x = np.radians(90)
        theta_y = np.radians(0)
        theta_z = np.radians(90)

    elif tag_id == 9:
        # Example: Adjust as needed for tag 9 mounting
        theta_x = np.radians(0)
        theta_y = np.radians(0)
        theta_z = np.radians(90)

    else:
        # Default: no adjustment
        theta_x = np.radians(0)
        theta_y = np.radians(0)
        theta_z = np.radians(0)
    
    # Build rotation matrices (same style as your original code)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations: R_adjust = R_y * R_x, then * R_z (same as your original)
    R_adjust = R_y @ R_x
    R_adjust = R_adjust @ R_z
    
    return R_adjust

def average_rotation_matrices(rotation_matrices):
    """
    Average multiple rotation matrices using quaternion averaging
    """
    if not rotation_matrices:
        return np.eye(3)
    
    # Convert rotation matrices to quaternions
    quaternions = []
    for R in rotation_matrices:
        q = rotation_matrix_to_quaternion(R)
        quaternions.append(q)
    
    # Average quaternions
    avg_quat = average_quaternions(quaternions)
    
    # Convert back to rotation matrix
    return quaternion_to_rotation_matrix(avg_quat)

def average_quaternions(quaternions):
    """
    Average multiple quaternions using the eigen decomposition method
    """
    if not quaternions:
        return [1.0, 0.0, 0.0, 0.0]
    
    # Build 4x4 accumulator matrix
    Q = np.zeros((4, 4))
    for q in quaternions:
        q_array = np.array(q).reshape(4, 1)
        Q += q_array @ q_array.T
    
    # Find eigenvector corresponding to largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    
    # Get real parts only (ignore complex components)
    eigenvalues_real = np.real(eigenvalues)
    eigenvectors_real = np.real(eigenvectors)
    
    avg_quat = eigenvectors_real[:, np.argmax(eigenvalues_real)]
    
    # Ensure positive real part
    if avg_quat[0] < 0:
        avg_quat = -avg_quat
    
    # Normalize
    norm = np.linalg.norm(avg_quat)
    if norm > 0:
        avg_quat = avg_quat / norm
    
    return avg_quat.tolist()
def simple_position_fusion(side_pos, top_pos):
    """
    Fuse position with simple weights
    side_pos, top_pos: [x, y, z] arrays or None
    """
    # If only one camera has data, use that
    if side_pos is None and top_pos is None:
        return None
    if side_pos is None:
        return top_pos
    if top_pos is None:
        return side_pos
    
    # Weighted average per axis
    fused_x = (side_pos[0] * SIDE_POS_WEIGHTS['x'] + top_pos[0] * TOP_POS_WEIGHTS['x']) / \
              (SIDE_POS_WEIGHTS['x'] + TOP_POS_WEIGHTS['x'])
    
    fused_y = (side_pos[1] * SIDE_POS_WEIGHTS['y'] + top_pos[1] * TOP_POS_WEIGHTS['y']) / \
              (SIDE_POS_WEIGHTS['y'] + TOP_POS_WEIGHTS['y'])
    
    fused_z = (side_pos[2] * SIDE_POS_WEIGHTS['z'] + top_pos[2] * TOP_POS_WEIGHTS['z']) / \
              (SIDE_POS_WEIGHTS['z'] + TOP_POS_WEIGHTS['z'])
    
    return np.array([fused_x, fused_y, fused_z])

def simple_orientation_fusion(side_q, top_q, side_R, top_R):
    """
    Fuse orientation with simple weights
    Converts to Euler angles for component-wise fusion
    """
    # If only one camera has data, use that
    if (side_q is None or side_R is None) and (top_q is None or top_R is None):
        return None
    if side_q is None or side_R is None:
        return top_q
    if top_q is None or top_R is None:
        return side_q
    
    # Convert quaternions to Euler angles
    side_euler = rotation_matrix_to_euler(side_R)
    top_euler = rotation_matrix_to_euler(top_R)
    
    # Weighted average per angle
    fused_roll = (side_euler[0] * SIDE_ORI_WEIGHTS['roll'] + top_euler[0] * TOP_ORI_WEIGHTS['roll']) / \
                 (SIDE_ORI_WEIGHTS['roll'] + TOP_ORI_WEIGHTS['roll'])
    
    fused_pitch = (side_euler[1] * SIDE_ORI_WEIGHTS['pitch'] + top_euler[1] * TOP_ORI_WEIGHTS['pitch']) / \
                  (SIDE_ORI_WEIGHTS['pitch'] + TOP_ORI_WEIGHTS['pitch'])
    
    # For yaw, handle angle wrapping (0/360 boundary)
    side_yaw = side_euler[2]
    top_yaw = top_euler[2]
    
    # Adjust if angles cross 0/360 boundary
    if abs(side_yaw - top_yaw) > 180:
        if side_yaw > top_yaw:
            top_yaw += 360
        else:
            side_yaw += 360
    
    fused_yaw = (side_yaw * SIDE_ORI_WEIGHTS['yaw'] + top_yaw * TOP_ORI_WEIGHTS['yaw']) / \
                (SIDE_ORI_WEIGHTS['yaw'] + TOP_ORI_WEIGHTS['yaw'])
    
    # Normalize back to -180 to 180
    fused_yaw = ((fused_yaw + 180) % 360) - 180
    
    # Convert back to quaternion
    return euler_to_quaternion_simple(fused_roll, fused_pitch, fused_yaw)

def euler_to_quaternion_simple(roll, pitch, yaw):
    """Simple Euler to quaternion conversion"""
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    # Normalize
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    return [qw/norm, qx/norm, qy/norm, qz/norm]

from scipy.linalg import svd

def calibrate_sensor_from_measurements(x_meas, y_meas, z_meas):
    """
    Compute sensor-to-cube calibration matrix from three measurements.
    
    Args:
        x_meas: Sensor reading when cube's +X points to field
        y_meas: Sensor reading when cube's +Y points to field
        z_meas: Sensor reading when cube's +Z points to field
    
    Returns:
        R_sensor_to_cube: 3x3 rotation matrix
    """
    # Normalize each measurement to unit vectors
    u_x = x_meas / np.linalg.norm(x_meas)
    u_y = y_meas / np.linalg.norm(y_meas)
    u_z = z_meas / np.linalg.norm(z_meas)
    
    # Build matrix and orthogonalize
    R_raw = np.column_stack([u_x, u_y, u_z])
    U, _, Vt = svd(R_raw)
    R_orth = U @ Vt
    
    # Return sensor→cube (transpose of cube→sensor)
    return R_orth.T

# Example usage
x_meas = np.array([12.3, 45.6, -7.8])
y_meas = np.array([-8.9, 52.1, 3.4])
z_meas = np.array([5.6, -12.3, 48.7])

R = calibrate_sensor_from_measurements(x_meas, y_meas, z_meas)
print("Calibration Matrix:")
print(R)

# Save it
np.save('sensor_cube_calibration.npy', R)



def sensor_to_mounting_frame(sensor_reading):
    """
    Transform a single sensor reading from sensor frame to mounting frame
    
    Args:
        sensor_reading: [x, y, z] in sensor's coordinate system
    
    Returns:
        mounting_frame: [x, y, z] in mounting frame coordinate system
    """

    # Inverted matrix (flip Y and Z rows)
    R_mc_inverted = np.array([
    [0.99729097, -0.05141686, 0.05260256],
    [0.04623249, 0.99436218, 0.09542749],
    [-0.05721258, -0.09273703, 0.99404555]
    ])
    sensor_reading = np.array(sensor_reading)
    # R_mc @ sensor_reading  (matrix multiplication)
    mounting_frame = R_mc_inverted @ sensor_reading
    return mounting_frame


def detect_apriltags(camera_matrix, dist_coeffs ,top_camera_matrix, top_dist_coeffs):
    global auto_record_flag, rel_Rsensor, estX, estY, estZ, magX, magY, magZ
    global bt_quat_global, position_thread_running, R_worldCube_camera, top_R_worldCube_camera, top_R_1000_cube1_camera,R_1000_cube1_camera,R_senToCube
    global serial_magX, serial_magY, serial_magZ, serial_mag_lock

    ## Sen to cube Calibration
    x_meas = np.array([12.3, 45.6, -7.8])
    y_meas = np.array([-8.9, 52.1, 3.4])
    z_meas = np.array([5.6, -12.3, 48.7])
    R_senToCube = calibrate_sensor_from_measurements(x_meas, y_meas, z_meas)

    

    cap = cv2.VideoCapture(camera_id)
    top_cap = cv2.VideoCapture(top_camera_id)
    if not cap.isOpened():
        print(f"Could not open camera with index {camera_id}.")
        return
    if not top_cap.isOpened():
        print(f"Could not open top camera with index {top_camera_id}.")
        return


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Camera set to: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} @ 30fps")

    top_cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    top_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    top_cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Top camera set to: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} @ 30fps")

    detector = Detector(families="tag36h11", nthreads=8, quad_decimate=2.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.5)

    cube1_center_z_buffer = []
    cube2_center_z_buffer = []
    worldCube_center_z_buffer = []

    top_cube1_center_z_buffer = []
    top_cube2_center_z_buffer = []
    top_worldCube_center_z_buffer = []

    MAG_OFFSET_FROM_CUBE = np.array([cube2MagOffset_X, cube2MagOffset_Y, cube2MagOffset_Z])

    measurement_order = 0
    motor_data = None
    last_motor_update = 0
    MOTOR_UPDATE_INTERVAL = 0.1
    
    # Position estimation update rate control
    last_position_update = 0
    POSITION_UPDATE_INTERVAL = 0.05  # Update 20 times per second

    # Start position estimation worker thread
    position_thread = threading.Thread(target=position_estimation_worker, daemon=True)
    position_thread_running = True
    position_thread.start()
    
    # Start TCP servers
    server_thread = threading.Thread(target=run_tcp_server, args=('localhost', MOTOR_SERVER_PORT, stop_event))
    server_thread.daemon = True
    server_thread.start()
    
    # Start bt_angleEst data receiver
    bt_receiver_thread = threading.Thread(target=run_bt_data_receiver, args=('localhost', BT_ANGLE_EST_PORT, stop_event))
    bt_receiver_thread.daemon = True
    bt_receiver_thread.start()
    
    # ADD THIS: Start serial data receiver thread
    if serial_port is not None:
        serial_thread = threading.Thread(target=run_serial_data_receiver, args=(stop_event,))
        serial_thread.daemon = True
        serial_thread.start()
        print("Serial data receiver thread started")
    else:
        print("Serial data receiver not started (no serial connection)")


    capture_time = None

    map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, 
    camera_matrix, 
    (DISPLAY_WIDTH, DISPLAY_HEIGHT),  # Use your camera resolution
    cv2.CV_16SC2
    )

    map3, map4 = cv2.initUndistortRectifyMap(
    top_camera_matrix, top_dist_coeffs, None,
    top_camera_matrix,
    (DISPLAY_WIDTH, DISPLAY_HEIGHT),
    cv2.CV_16SC2
    )
    
    cv2.namedWindow("AprilTag Localization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Localization", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    text_frame = np.zeros((TEXT_FRAME_HEIGHT, TEXT_FRAME_WIDTH, 3), dtype=np.uint8)
    
    try:
        with open(LOG_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Order', 'Rel_X_m', 'Rel_Y_m', 'Rel_Z_m', 'Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz', 'Cube2_Pitch_deg', 'Cube2_Roll_deg', 'Cube2_Yaw_deg',
                             'Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT', 'Est_Xm', 'Est_Ym', 'Est_Zm', 'Error_Xm', 'Error_Ym', 'Error_Zm'])
            while not stop_event.is_set():
                start_time = time.time()
                ret, frame = cap.read()
                top_ret, top_frame = top_cap.read()

                capture_time = time.time()

                if not ret and not top_ret:
                    print("Failed to capture frame from all camera.")
                    break

                # frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
                frame_undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                top_frame_undistorted = cv2.remap(top_frame, map3, map4, cv2.INTER_LINEAR)
                undistort_time = time.time()
              
                text_frame[:] = 0  # Reset to black
                
                gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
                top_gray = cv2.cvtColor(top_frame_undistorted, cv2.COLOR_BGR2GRAY)
                convert_time = time.time()
   

                detections = detector.detect(gray, estimate_tag_pose=True,
                                            camera_params=(camera_matrix[0, 0], camera_matrix[1, 1],
                                                           camera_matrix[0, 2], camera_matrix[1, 2]),
                                            tag_size=TAG_SIZE)
           
                top_detections = detector.detect(top_gray, estimate_tag_pose=True,
                                                 camera_params=(top_camera_matrix[0, 0], top_camera_matrix[1, 1],
                                                                top_camera_matrix[0, 2], top_camera_matrix[1, 2]),
                                                 tag_size=TAG_SIZE)
                detection_time = time.time()

                # side cam data
                tags = []
                worldCube_center_camera = None
                R_cube1_camera = None
                R_cube2_camera = None
                
                R_cube2_global = None
                R_cube1_global = None
                R_cube1_cube2 = None
                q_cube1_cube2 = None

                worldCube_center_camera = None
                cube1_center_camera = None
                cube2_center_camera = None

                magnet_position_camera = None
                t_cube1_cube2_tag = None


                # top cam data
                top_tags = []
                top_worldCube_center_camera = None

                top_R_cube1_camera = None
                top_R_cube2_camera = None
                
                top_R_cube2_global = None
                top_R_cube1_global = None
                top_R_cube1_cube2 = None
                top_q_cube1_cube2 = None

                top_worldCube_center_camera = None
                top_cube1_center_camera = None
                top_cube2_center_camera = None

                top_magnet_position_camera = None
                top_t_cube1_cube2_tag = None

                # Fused data
                fused_t_cube1_cube2_tag = None
                fused_q_cube1_cube2 = None

                b_global = None
                # Non-camera specific data
                magX = 0.0
                magY = 0.0
                magZ = 0.0
                q_btcap_global = None
                R_btcap_cube2 = np.eye(3)
                est_pos_ref = None
                q_cube2_global = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion as default
                q_btcap_cube2 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion as default
                q_cube1_global = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion as default
                q_cube1_cube2 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion as default                
               
                for det in detections:
                    if det.tag_id in TAG_IDS:
                        corners = det.corners
                        for i in range(4):
                            cv2.line(frame_undistorted, tuple(corners[i].astype(int)), tuple(corners[(i+1)%4].astype(int)), (0, 255, 0), 2)
                        rvec, tvec = refine_pose(corners, camera_matrix, dist_coeffs)
                        if rvec is None or tvec is None:
                            print(f"Pose refinement failed for tag {det.tag_id}")
                            continue
                        R, _ = cv2.Rodrigues(rvec)
                        tag_data = {'tag_id': det.tag_id, 'tvec': tvec.flatten(), 'R': R, 'center': np.mean(corners, axis=0)}
                        tags.append(tag_data)


                for det in top_detections:
                    if det.tag_id in TAG_IDS:
                        corners = det.corners
                        for i in range(4):
                            cv2.line(top_frame_undistorted, tuple(corners[i].astype(int)), tuple(corners[(i+1)%4].astype(int)), (0, 255, 0), 2)
                        rvec, tvec = refine_pose(corners, top_camera_matrix, top_dist_coeffs)
                        if rvec is None or tvec is None:
                            print(f"Pose refinement failed for top tag {det.tag_id}")
                            continue
                        R, _ = cv2.Rodrigues(rvec)
                        tag_data = {'tag_id': det.tag_id, 'tvec': tvec.flatten(), 'R': R, 'center': np.mean(corners, axis=0)}
                        top_tags.append(tag_data)

                processing_time = time.time()
 
                # Get latest BT data
                latest_bt_data = get_latest_bt_data()
                if latest_bt_data:
                    with bt_data_lock:
                        magX = -latest_bt_data.get('magY', 0.0)
                        magY = -latest_bt_data.get('magX', 0.0)
                        magZ = -latest_bt_data.get('magZ', 0.0)
                        q_btcap_global = [
                            latest_bt_data.get('qw', 1.0),
                            latest_bt_data.get('qx', 0.0),
                            latest_bt_data.get('qy', 0.0),
                            latest_bt_data.get('qz', 0.0)
                        ]


                ################################################################################################################################
                ###################                     side Cam cube Extraction                 #########################################
                ################################################################################################################################
                if len(tags)>0: 
                        
                    worldCube_center_camera, _ = compute_cube_center(tags, CUBE_SIZE, tag_ids=[7, 8, 9], reference_tag_id=8)
                    if worldCube_center_camera is not None:
                        # print(f"World Cube Center Camera: {worldCube_center_camera}")
                        worldCube_center_z_buffer.append(worldCube_center_camera[2])
                        if len(worldCube_center_z_buffer) > Z_SMOOTHING_WINDOW:
                            worldCube_center_z_buffer.pop(0)
                        smoothed_z_world = np.mean(worldCube_center_z_buffer)
                        worldCube_center_camera_smoothed = worldCube_center_camera.copy()
                        worldCube_center_camera_smoothed[2] = smoothed_z_world
                        worldCube_center_point = np.array([[worldCube_center_camera_smoothed]], dtype=np.float32)
                        img_points, _ = cv2.projectPoints(worldCube_center_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                        img_worldCube_center = img_points[0][0].astype(int)
                        cv2.circle(frame_undistorted, tuple(img_worldCube_center), 8, (255, 0, 255), -1)
                        draw_text_with_background(frame_undistorted, "W.C.", (img_worldCube_center[0]+10, img_worldCube_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        R_worldCube_camera = compute_cube_orientation(tags, [7, 8, 9])
                        worldCube_rvec, _ = cv2.Rodrigues(R_worldCube_camera)
                        draw_axes(frame_undistorted, camera_matrix, dist_coeffs, worldCube_rvec, worldCube_center_camera_smoothed, AXIS_LENGTH, img_worldCube_center)
                    
                    ## CUBE 1

                    cube1_center_camera, _ = compute_cube_center(tags, CUBE_SIZE, tag_ids=[1,2], reference_tag_id=1)
                    cube1_center_text = "Cube 1 Center: Not detected"
                    if cube1_center_camera is not None:
                        # print(f"Cube 1 Center Camera: {cube1_center_camera}")
                        cube1_center_z_buffer.append(cube1_center_camera[2])
                        if len(cube1_center_z_buffer) > Z_SMOOTHING_WINDOW:
                            cube1_center_z_buffer.pop(0)
                        smoothed_z1 = np.mean(cube1_center_z_buffer)
                        cube1_center_camera_smoothed = cube1_center_camera.copy()
                        cube1_center_camera_smoothed[2] = smoothed_z1
                        cube1_center_point = np.array([[cube1_center_camera_smoothed]], dtype=np.float32)
                        img_points, _ = cv2.projectPoints(cube1_center_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                        img_cube1_center = img_points[0][0].astype(int)
                        cv2.circle(frame_undistorted, tuple(img_cube1_center), 8, (255, 255, 0), -1)

                        R_cube1_camera = compute_cube_orientation(tags, [1,2])
                        cube1_rvec_camera, _ = cv2.Rodrigues(R_cube1_camera)
                        draw_axes(frame_undistorted, camera_matrix, dist_coeffs, cube1_rvec_camera, cube1_center_camera_smoothed, AXIS_LENGTH, img_cube1_center)
                        draw_text_with_background(frame_undistorted, "CAP", (img_cube1_center[0]+10, img_cube1_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cube1_center_text = f"Cube 1 Center: x={cube1_center_camera_smoothed[0]:.3f} m, y={cube1_center_camera_smoothed[1]:.3f} m, z={cube1_center_camera_smoothed[2]:.3f} m"
                    
                    ## CUBE 2

                    cube2_center_camera, _ = compute_cube_center(tags, CUBE_SIZE, tag_ids=[3,4,5], reference_tag_id=5)

                    cube2_center_text = "Cube 2 Center: Not detected"
                    cube2_orient_text = "Cube 2 Orient: Not detected"
                    if cube2_center_camera is not None:
                        # print(f"Cube 2 Center Camera: {cube2_center_camera}")
                        cube2_center_z_buffer.append(cube2_center_camera[2])
                        if len(cube2_center_z_buffer) > Z_SMOOTHING_WINDOW:
                            cube2_center_z_buffer.pop(0)
                        smoothed_z2 = np.mean(cube2_center_z_buffer)
                        cube2_center_camera_smoothed = cube2_center_camera.copy()
                        cube2_center_camera_smoothed[2] = smoothed_z2
                        cube2_center_point = np.array([[cube2_center_camera_smoothed]], dtype=np.float32)
                        img_points, _ = cv2.projectPoints(cube2_center_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                        img_cube2_center = img_points[0][0].astype(int)
                        cv2.circle(frame_undistorted, tuple(img_cube2_center), 8, (0, 255, 255), -1)
                        R_cube2_camera = compute_cube_orientation(tags, [3,4,5])
                        cube2_rvec_camera, _ = cv2.Rodrigues(R_cube2_camera)
                        draw_text_with_background(frame_undistorted, "MAG", (img_cube2_center[0]+10, img_cube2_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        draw_axes(frame_undistorted, camera_matrix, dist_coeffs, cube2_rvec_camera, cube2_center_camera_smoothed, AXIS_LENGTH, img_cube2_center)

                    ## Safety check: Ensure cube centers are not too close in image
                    if(cube1_center_camera is not None and cube2_center_camera is not None):
                        distance = np.linalg.norm(img_cube1_center - img_cube2_center)
                        if distance < 20:
                            print(f"Cube centers too close in image: {distance:.2f} pixels")
                        

                ################################################################################################################################
                ###################                     top Cam cube Extraction                 #########################################
                ################################################################################################################################
                ## Extract cube positions and orientations from top cam
                if len(top_tags) > 0:

                    top_worldCube_center_camera, _ = compute_cube_center(top_tags, CUBE_SIZE, tag_ids=[7, 8, 9], reference_tag_id=8)
                    if top_worldCube_center_camera is not None:
                        # print(f"World Cube Center Camera: {top_worldCube_center_camera}")
                        top_worldCube_center_z_buffer.append(top_worldCube_center_camera[2])
                        if len(top_worldCube_center_z_buffer) > Z_SMOOTHING_WINDOW:
                            top_worldCube_center_z_buffer.pop(0)
                        smoothed_z_world = np.mean(top_worldCube_center_z_buffer)
                        top_worldCube_center_camera_smoothed = top_worldCube_center_camera.copy()
                        top_worldCube_center_camera_smoothed[2] = smoothed_z_world
                        top_worldCube_center_point = np.array([[top_worldCube_center_camera_smoothed]], dtype=np.float32)
                        img_points, _ = cv2.projectPoints(top_worldCube_center_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                        img_top_worldCube_center = img_points[0][0].astype(int)
                        cv2.circle(top_frame_undistorted, tuple(img_top_worldCube_center), 8, (255, 0, 255), -1)
                        draw_text_with_background(top_frame_undistorted, "W.C.", (img_top_worldCube_center[0]+10, img_top_worldCube_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        top_R_worldCube_camera = compute_cube_orientation(top_tags, [7, 8, 9])
                        top_worldCube_rvec, _ = cv2.Rodrigues(top_R_worldCube_camera)
                        draw_axes(top_frame_undistorted, camera_matrix, dist_coeffs, top_worldCube_rvec, top_worldCube_center_camera_smoothed, AXIS_LENGTH, img_top_worldCube_center)
                    
                    ## CUBE 1

                    top_cube1_center_camera, _ = compute_cube_center(top_tags, CUBE_SIZE, tag_ids=[0,1,2], reference_tag_id=1)
                    if top_cube1_center_camera is not None:
                        # print(f"Cube 1 Center Camera: {cube1_center_camera}")
                        top_cube1_center_z_buffer.append(top_cube1_center_camera[2])
                        if len(top_cube1_center_z_buffer) > Z_SMOOTHING_WINDOW:
                            top_cube1_center_z_buffer.pop(0)
                        smoothed_z1 = np.mean(top_cube1_center_z_buffer)
                        top_cube1_center_camera_smoothed = top_cube1_center_camera.copy()
                        top_cube1_center_camera_smoothed[2] = smoothed_z1
                        top_cube1_center_point = np.array([[top_cube1_center_camera_smoothed]], dtype=np.float32)
                        img_points, _ = cv2.projectPoints(top_cube1_center_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                        img_top_cube1_center = img_points[0][0].astype(int)
                        cv2.circle(top_frame_undistorted, tuple(img_top_cube1_center), 8, (255, 255, 0), -1)

                        top_R_cube1_camera = compute_cube_orientation(top_tags, [0,1,2])
                        top_cube1_rvec_camera, _ = cv2.Rodrigues(top_R_cube1_camera)
                        draw_axes(top_frame_undistorted, camera_matrix, dist_coeffs, top_cube1_rvec_camera, top_cube1_center_camera_smoothed, AXIS_LENGTH, img_top_cube1_center)
                        draw_text_with_background(top_frame_undistorted, "CAP", (img_top_cube1_center[0]+10, img_top_cube1_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        top_cube1_center_text = f"Cube 1 Center: x={top_cube1_center_camera_smoothed[0]:.3f} m, y={top_cube1_center_camera_smoothed[1]:.3f} m, z={top_cube1_center_camera_smoothed[2]:.3f} m"

                    ## CUBE 2

                    top_cube2_center_camera, _ = compute_cube_center(top_tags, CUBE_SIZE, tag_ids=[3,4,5], reference_tag_id=5)

                    top_cube2_center_text = "Cube 2 Center: Not detected"
                    top_cube2_orient_text = "Cube 2 Orient: Not detected"
                    if top_cube2_center_camera is not None:
                        # print(f"Cube 2 Center Camera: {cube2_center_camera}")
                        top_cube2_center_z_buffer.append(top_cube2_center_camera[2])
                        if len(top_cube2_center_z_buffer) > Z_SMOOTHING_WINDOW:
                            top_cube2_center_z_buffer.pop(0)
                        smoothed_z2 = np.mean(top_cube2_center_z_buffer)
                        top_cube2_center_camera_smoothed = top_cube2_center_camera.copy()
                        top_cube2_center_camera_smoothed[2] = smoothed_z2
                        top_cube2_center_point = np.array([[top_cube2_center_camera_smoothed]], dtype=np.float32)
                        img_points, _ = cv2.projectPoints(top_cube2_center_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                        img_top_cube2_center = img_points[0][0].astype(int)
                        cv2.circle(top_frame_undistorted, tuple(img_top_cube2_center), 8, (0, 255, 255), -1)
                        top_R_cube2_camera = compute_cube_orientation(top_tags, [3,4,5])
                        top_cube2_rvec_camera, _ = cv2.Rodrigues(top_R_cube2_camera)
                        draw_text_with_background(top_frame_undistorted, "MAG", (img_top_cube2_center[0]+10, img_top_cube2_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        draw_axes(top_frame_undistorted, camera_matrix, dist_coeffs, top_cube2_rvec_camera, top_cube2_center_camera_smoothed, AXIS_LENGTH, img_top_cube2_center)

                    ## Safety check: Ensure cube centers are not too close in image
                    if(top_cube1_center_camera is not None and top_cube2_center_camera is not None):
                        distance = np.linalg.norm(img_top_cube1_center - img_top_cube2_center)
                        if distance < 20:
                            print(f"Cube centers too close in image: {distance:.2f} pixels")

                




                ################################################################################################################################
                ###################                     side Cam cube calculations                     #########################################
                ################################################################################################################################
                # Calculate magnet position in camera frame by applying offset
                if cube2_center_camera is not None and R_cube2_camera is not None:
                    magnet_position_camera = cube2_center_camera_smoothed + R_cube2_camera @ MAG_OFFSET_FROM_CUBE
                    # visualize magnet position
                    magnet_point = np.array([[magnet_position_camera]], dtype=np.float32)
                    img_magnet, _ = cv2.projectPoints(magnet_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                    img_magnet_center = img_magnet[0][0].astype(int)
                    cv2.circle(frame_undistorted, tuple(img_magnet_center), 6, (255, 0, 255), -1)  # Purple dot for magnet

                # Calculate relative position and orientation of cube 1 relative to cube 2:
                if cube1_center_camera is not None and cube2_center_camera is not None and R_cube2_camera is not None:
                    # Get sensor (Cube 1) position relative to magnet
                    t_cube1_cube2_tag = transform_to_ref_cube_frame(
                        cube1_center_camera_smoothed,
                        magnet_position_camera,
                        R_cube2_camera
                    )
                    R_cube1_cube2 = R_cube2_camera.T @ R_cube1_camera
                    cube2_euler = rotation_matrix_to_euler(R_cube1_cube2)
                    q_cube1_cube2 = rotation_matrix_to_quaternion(R_cube1_cube2) if R_cube1_cube2 is not None else [1.0, 0.0, 0.0, 0.0]


                #Calcuate global orientation of cube 2 using world cube as reference
                if R_cube2_camera is not None and R_worldCube_camera is not None:
                    R_cube2_global = R_worldCube_camera.T @ R_cube2_camera
                    q_cube2_global = rotation_matrix_to_quaternion(R_cube2_global) if R_cube2_global is not None else [1.0, 0.0, 0.0, 0.0]

                ## Cap Global orientation
                if R_cube1_camera is not None and R_worldCube_camera is not None:
                    R_cube1_global = R_worldCube_camera.T @ R_cube1_camera    
                    q_cube1_global = rotation_matrix_to_quaternion(R_cube1_global) if R_cube1_global is not None else [1.0, 0.0, 0.0, 0.0]




                ################################################################################################################################
                ###################                     top Cam cube calculations                     #########################################
                ################################################################################################################################
                # Calculate magnet position in camera frame by applying offset
                if top_cube2_center_camera is not None and top_R_cube2_camera is not None:
                    top_magnet_position_camera = top_cube2_center_camera + top_R_cube2_camera @ MAG_OFFSET_FROM_CUBE
                    # visualize magnet position
                    top_magnet_point = np.array([[top_magnet_position_camera]], dtype=np.float32)
                    top_img_magnet, _ = cv2.projectPoints(top_magnet_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                    top_img_magnet_center = top_img_magnet[0][0].astype(int)
                    cv2.circle(top_frame_undistorted, tuple(top_img_magnet_center), 6, (255, 0, 255), -1)  # Purple dot for magnet

                # Calculate relative position and orientation of cube 1 relative to cube 2:
                if top_cube1_center_camera is not None and top_cube2_center_camera is not None and top_R_cube2_camera is not None:
                    # Get sensor (Cube 1) position relative to magnet
                    top_t_cube1_cube2_tag = transform_to_ref_cube_frame(
                        top_cube1_center_camera_smoothed,
                        top_magnet_position_camera,
                        top_R_cube2_camera
                    )
                    top_R_cube1_cube2 = top_R_cube2_camera.T @ top_R_cube1_camera
                    top_cube2_euler = rotation_matrix_to_euler(top_R_cube1_cube2)
                    top_q_cube1_cube2 = rotation_matrix_to_quaternion(top_R_cube1_cube2) if top_R_cube1_cube2 is not None else [1.0, 0.0, 0.0, 0.0]


                #Calcuate global orientation of cube 2 using world cube as reference
                if top_R_cube2_camera is not None and top_R_worldCube_camera is not None:
                    top_R_cube2_global = top_R_worldCube_camera.T @ top_R_cube2_camera
                    top_q_cube2_global = rotation_matrix_to_quaternion(top_R_cube2_global) if top_R_cube2_global is not None else [1.0, 0.0, 0.0, 0.0]

                ## Cap Global orientation
                if top_R_cube1_camera is not None and top_R_worldCube_camera is not None:
                    top_R_cube1_global = top_R_worldCube_camera.T @ top_R_cube1_camera    
                    top_q_cube1_global = rotation_matrix_to_quaternion(top_R_cube1_global) if top_R_cube1_global is not None else [1.0, 0.0, 0.0, 0.0]



                # if(top_t_cube1_cube2_tag is not None and t_cube1_cube2_tag is not None):
                #     print(f"top cam x:{top_t_cube1_cube2_tag[0]:.3f}\ty:{top_t_cube1_cube2_tag[1]:.3f}\tz:{top_t_cube1_cube2_tag[2]:.3f}\t\tSide cam x:{t_cube1_cube2_tag[0]:.3f}\ty:{t_cube1_cube2_tag[1]:.3f}\tz:{t_cube1_cube2_tag[2]:.3f}")
                # if(top_q_cube1_cube2 is not None and q_cube1_cube2 is not None):
                #     print(f"top cam qx:{top_q_cube1_cube2[0]:.3f}\tqy:{top_q_cube1_cube2[1]:.3f}\tqz:{top_q_cube1_cube2[2]:.3f}\tqw:{top_q_cube1_cube2[3]:.3f}\t\tSide cam qx:{q_cube1_cube2[0]:.3f}\tqy:{q_cube1_cube2[1]:.3f}\tqz:{q_cube1_cube2[2]:.3f}\tqw:{q_cube1_cube2[3]:.3f}")


                ################################################################################################################################
                ###### Data fusing  to calculate relative position and orientation of cube 1 relative to cube 2           ######################
                ################################################################################################################################

                if(top_t_cube1_cube2_tag is not None and t_cube1_cube2_tag is not None and
                   top_q_cube1_cube2 is not None and q_cube1_cube2 is not None):
                    fused_t_cube1_cube2_tag = simple_position_fusion(t_cube1_cube2_tag, top_t_cube1_cube2_tag)
                    fused_q_cube1_cube2 = simple_orientation_fusion(q_cube1_cube2, top_q_cube1_cube2, 
                                                                    R_cube1_cube2, top_R_cube1_cube2)
                    


                # # Print for debugging
                # if fused_t_cube1_cube2_tag is not None and fused_q_cube1_cube2 is not None:
                #     print(f"Fused pos: x={fused_t_cube1_cube2_tag[0]:.3f}, y={fused_t_cube1_cube2_tag[1]:.3f}, z={fused_t_cube1_cube2_tag[2]:.3f}")
                #     print(f"Fused ori: qx={fused_q_cube1_cube2[0]:.3f}, qy={fused_q_cube1_cube2[1]:.3f}, qz={fused_q_cube1_cube2[2]:.3f}, qw={fused_q_cube1_cube2[3]:.3f}")









                ################################################################################################################################
                #################################################### handling BT DATA      #####################################################
                ################################################################################################################################


                    
                if q_btcap_global is not None and R_cube2_global is not None:
                    R_btcap_global = quaternion_to_rotation_matrix(q_btcap_global)  # Convert bt quat to rotation matrix
                    R_btcap_cube2 =  R_cube2_global.T @ R_btcap_global
                    q_btcap_cube2 = rotation_matrix_to_quaternion(R_btcap_cube2) if R_btcap_cube2 is not None else [1.0, 0.0, 0.0, 0.0]

    

                if (R_cube1_cube2 is not None) and (magX != 0.0 or magY != 0.0 or magZ != 0.0):
                    b_global = R_cube1_cube2 @ np.array([magX, magY, magZ])




                # Update position estimation at controlled rate (NON-BLOCKING)
                current_time = time.time()
                if current_time - last_position_update > POSITION_UPDATE_INTERVAL:
                    update_position_estimation(fused_q_cube1_cube2, magX, magY, magZ)  # This is now non-blocking
                    last_position_update = current_time

                # Draw estimated position and axes
                if cube2_center_camera is not None and R_cube2_camera is not None and (estX != 0 or estY != 0 or estZ != 0) and q_btcap_global is not None and R_worldCube_camera is not None:
                    
                    # Step 1: Get magnet position in camera frames
                    magnet_position_camera = cube2_center_camera_smoothed + R_cube2_camera @ MAG_OFFSET_FROM_CUBE
                    
                    # Step 2: estX, estY, estZ are in magnet's coordinate frame
                    # Transform this vector to camera frame
                    est_vector_camera = R_cube2_camera @ np.array([estX, estY, estZ])
                    
                    # Step 3: Estimated position in camera frame = magnet position + transformed vector
                    est_pos_camera = magnet_position_camera + est_vector_camera
                    
                    # Project to image coordinates
                    est_pos_point = np.array([[est_pos_camera]], dtype=np.float32)
                    img_points, _ = cv2.projectPoints(est_pos_point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                    img_est_pos = img_points[0][0].astype(int)
                    
                    # Draw if within bounds
                    if (0 <= img_est_pos[0] < frame_undistorted.shape[1] and 
                        0 <= img_est_pos[1] < frame_undistorted.shape[0]):
                        cv2.circle(frame_undistorted, tuple(img_est_pos), 8, (255, 0, 0), -1)
                        
                        R_bt = quaternion_to_rotation_matrix(q_btcap_global)
                        R_btcap_cam = R_worldCube_camera @ R_bt
                        draw_axes(frame_undistorted, camera_matrix, dist_coeffs, R_btcap_cam, est_pos_camera, AXIS_LENGTH, img_est_pos)
                    else:
                        print(f"Estimated position projected outside image: {img_est_pos}")


                calculation_time = time.time()

                mag_magnitude = np.sqrt(magX**2 + magY**2 + magZ**2)
                tag_euler = rotation_matrix_to_euler(quaternion_to_rotation_matrix(fused_q_cube1_cube2)) if fused_q_cube1_cube2 is not None else [0.0, 0.0, 0.0]
                bt_euler = rotation_matrix_to_euler(R_btcap_cube2) if R_btcap_cube2 is not None else [0.0, 0.0, 0.0]

                # Draw text on text_frame with semi-transparent backgrounds
                mag_text = f"Mag: X={magX:.2f} uT Y={magY:.2f} uT Z={magZ:.2f} uT M={mag_magnitude:.2f} uT"
                q_Mag_global_text =     f"q_Mag_global: [{q_cube2_global[0]:.3f}, {q_cube2_global[1]:.3f}, {q_cube2_global[2]:.3f}, {q_cube2_global[3]:.3f}]" if q_cube2_global is not None else "Cube2 Quat Global: N/A"
                q_cube1_global_text =     f"q_Cap_global: [{q_cube1_global[0]:.3f}, {q_cube1_global[1]:.3f}, {q_cube1_global[2]:.3f}, {q_cube1_global[3]:.3f}]" if q_cube1_global is not None else "Cube2 Quat Cap: N/A"
                q_TAG_Cap_Mag_text =    f"q_Cap_Mag: [{q_cube1_cube2[0]:.3f}, {q_cube1_cube2[1]:.3f}, {q_cube1_cube2[2]:.3f}, {q_cube1_cube2[3]:.3f}]" if q_cube1_cube2 is not None else "Rel Relsensor: N/A"
                q_cap_Mag_text =        f"q_btcap_cube2: [{q_btcap_cube2[0]:.3f}, {q_btcap_cube2[1]:.3f}, {q_btcap_cube2[2]:.3f}, {q_btcap_cube2[3]:.3f}]" if q_btcap_cube2 is not None else "Cube2 Quat Cap: N/A"
                q_q_btcap_global_text =   f"q_btcap_global: [{q_btcap_global[0]:.3f}, {q_btcap_global[1]:.3f}, {q_btcap_global[2]:.3f}, {q_btcap_global[3]:.3f}]" if q_btcap_global is not None else "BT Quat Global: N/A"
                est_pos_text =          f"Est Pos:        x={estX:.3f} m, y={estY:.3f} m, z={estZ:.3f} m"
                pos_error_text =        f"Pos Err: x={estX-fused_t_cube1_cube2_tag[0]:.3f} m, y={estY-fused_t_cube1_cube2_tag[1]:.3f} m, z={estZ-fused_t_cube1_cube2_tag[2]:.3f} m" if fused_t_cube1_cube2_tag is not None else "Pos Err: N/A"
                bglobal_text =     f"bglobal: x={b_global[0]:.3f} uT, y={b_global[1]:.3f} uT, z={b_global[2]:.3f} uT" if b_global is not None else "Global mag vector: N/A"
                diff_euler = [tag_euler[i] - bt_euler[i] for i in range(3)]
                diff_euler_text = f"Euler Diff: Pitch={diff_euler[0]:.1f}deg, Roll={diff_euler[1]:.1f}deg, Yaw={diff_euler[2]:.1f}deg"
                side_cam_text = f"SIDE: x={t_cube1_cube2_tag[0]:.3f}, y={t_cube1_cube2_tag[1]:.3f}, z={t_cube1_cube2_tag[2]:.3f} q:x={q_cube1_cube2[0]:.3f},{q_cube1_cube2[1]:.3f},{q_cube1_cube2[2]:.3f},{q_cube1_cube2[3]:.3f}" if t_cube1_cube2_tag is not None and q_cube1_cube2 is not None else "SIDE: N/A"
                top_cam_text = f"TOP:  x={top_t_cube1_cube2_tag[0]:.3f}, y={top_t_cube1_cube2_tag[1]:.3f}, z={top_t_cube1_cube2_tag[2]:.3f} q:x={top_q_cube1_cube2[0]:.3f},{top_q_cube1_cube2[1]:.3f},{top_q_cube1_cube2[2]:.3f},{top_q_cube1_cube2[3]:.3f}" if top_t_cube1_cube2_tag is not None and top_q_cube1_cube2 is not None else "TOP: N/A"
                fused_cam_text = f"FUSED:x={fused_t_cube1_cube2_tag[0]:.3f}, y={fused_t_cube1_cube2_tag[1]:.3f}, z={fused_t_cube1_cube2_tag[2]:.3f} q:x={fused_q_cube1_cube2[0]:.3f},{fused_q_cube1_cube2[1]:.3f},{fused_q_cube1_cube2[2]:.3f},{fused_q_cube1_cube2[3]:.3f}" if fused_t_cube1_cube2_tag is not None and fused_q_cube1_cube2 is not None else "FUSED: N/A"
                serial_mag_text = f"Serial Mag: X={serial_magX:.2f} uT, Y={serial_magY:.2f} uT, Z={serial_magZ:.2f} uT" if serial_magX is not None else "Serial Mag: N/A"
                intervalOffset = 1
                FONT_SCALE = 1
                draw_text_with_background(text_frame, mag_text, (50, int(50*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 3)
                draw_text_with_background(text_frame, side_cam_text, (50, int(90*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), 3)
                draw_text_with_background(text_frame, top_cam_text, (50, int(130*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), 3)
                draw_text_with_background(text_frame, fused_cam_text, (50, int(170*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), 3)
                draw_text_with_background(text_frame, q_cap_Mag_text, (50, int(210*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 3)
                draw_text_with_background(text_frame, q_q_btcap_global_text, (50, int(250*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 3)
                draw_text_with_background(text_frame, est_pos_text, (50, int(290*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 3)    
                draw_text_with_background(text_frame, bglobal_text, (50, int(330*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), 3)
                draw_text_with_background(text_frame, pos_error_text, (50, int(370*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, ( 100, 255, 0), 3)
                draw_text_with_background(text_frame, serial_mag_text, (50, int(410*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), 3)
                # draw_text_with_background(text_frame, bt_quat_global_text, (50, int(290*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), 2)
                # draw_text_with_background(text_frame, rel_Relsensor_text, (50, int(330*intervalOffset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 2)


                
                if time.time() - last_motor_update > MOTOR_UPDATE_INTERVAL:
                    with position_data_lock:
                        if latest_position_data:
                            motor_data = latest_position_data.copy()
                    last_motor_update = time.time()

                drawtext_time = time.time()

                print(
                    f"Timing Capture: {(capture_time-start_time)*1000:.1f}ms, "
                    f"Undistort: {(undistort_time-capture_time)*1000:.1f}ms, "
                    f"Convert: {(convert_time-undistort_time)*1000:.1f}ms, "
                    f"Detection: {(detection_time-convert_time)*1000:.1f}ms, "
                    f"Processing: {(processing_time-detection_time)*1000:.1f}ms, "
                    f"Calculation: {(calculation_time-processing_time)*1000:.1f}ms, "
                    f"drawText: {(drawtext_time-calculation_time)*1000:.1f}ms, "
                    f"Total: {(time.time()-start_time)*1000:.1f}ms"
                )
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('t') or auto_record_flag:
                    if magX is not None and magY is not None and magZ is not None and q_cube1_cube2 is not None:
                        measurement_order += 1
                        # q_cube1_cube2 = rotation_matrix_to_quaternion(rel_Rsensor)
                        errorX = estX - t_cube1_cube2_tag[0]
                        errorY = estY - t_cube1_cube2_tag[1]
                        errorZ = estZ - t_cube1_cube2_tag[2]
                        writer.writerow([
                            measurement_order,
                            f'{t_cube1_cube2_tag[0]:.6f}', f'{t_cube1_cube2_tag[1]:.6f}', f'{t_cube1_cube2_tag[2]:.6f}',
                            f'{q_cube1_cube2[0]:.5f}', f'{q_cube1_cube2[1]:.5f}', f'{q_cube1_cube2[2]:.5f}', f'{q_cube1_cube2[3]:.5f}',
                            f'{cube2_euler[0]:.5f}', f'{cube2_euler[1]:.5f}', f'{cube2_euler[2]:.5f}',
                            f'{magX:.3f}', f'{magY:.3f}', f'{magZ:.3f}',
                            f'{estX:.3f}', f'{estY:.3f}', f'{estZ:.3f}',
                            f'{errorX:.3f}', f'{errorY:.3f}', f'{errorZ:.3f}'
                        ])
                        csvfile.flush()
                        draw_text_with_background(text_frame, "MAG FIELD & POS LOGGED", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        time.sleep(0.2)
                        auto_record_flag = False

                elif key == ord ('c'):
                    if (R_cube1_camera is not None and top_R_cube1_camera):
                        top_R_1000_cube1_camera = top_R_cube1_camera.copy()
                        R_1000_cube1_camera = R_cube1_camera.copy()
                        print("Manually set worldCube reference to Cube 1")

                    
                    
                # text_frame_resized = cv2.resize(text_frame, (DISPLAY_WIDTH, TEXT_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                # camera_frame_resized = resize_frame(frame_undistorted, max_height=CAMERA_FRAME_HEIGHT)
                # top_camera_frame_resized = resize_frame(top_frame_undistorted, max_height=CAMERA_FRAME_HEIGHT)
                # combined_frame = np.vstack((text_frame_resized, camera_frame_resized, top_camera_frame_resized))


                # Calculate half width based on text frame width
                half_width = DISPLAY_WIDTH // 2

                # Resize both camera frames to half width
                camera_frame_resized = cv2.resize(frame_undistorted, (half_width, CAMERA_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                top_camera_frame_resized = cv2.resize(top_frame_undistorted, (half_width, CAMERA_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

                # Stack horizontally - now combined width = half_width + half_width = DISPLAY_WIDTH
                cameras_combined = np.hstack((camera_frame_resized, top_camera_frame_resized))

                # Text frame already has DISPLAY_WIDTH
                text_frame_resized = cv2.resize(text_frame, (DISPLAY_WIDTH, TEXT_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

                # Stack vertically
                combined_frame = np.vstack((text_frame_resized, cameras_combined))
                

                cv2.imshow("AprilTag Localization", combined_frame)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
    finally:
        # Cleanup
        position_thread_running = False
        cap.release()
        if 'server_thread' in locals():
            server_thread.join(timeout=1.0)
        if 'bt_receiver_thread' in locals():
            bt_receiver_thread.join(timeout=1.0)
        if 'position_thread' in locals():
            position_thread.join(timeout=1.0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        
        camera_matrix, dist_coeffs, top_camera_matrix, top_dist_coeffs = calibrate_camera()
        if camera_matrix is not None and dist_coeffs is not None and top_camera_matrix is not None and top_dist_coeffs is not None:
            detect_apriltags(camera_matrix, dist_coeffs, top_camera_matrix, top_dist_coeffs)
    except Exception as e:
        print(f"Main program error: {e}")
    finally:
        stop_event.set()