"""
Capsule Orientation Calculator from Sequential Image Info Files
Reads from imgInfo_XXXXX_0x20240719.txt files and calculates orientation using Madgwick filter
"""

import atexit
import os
import sys
import time
import numpy as np
from collections import deque
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from concurrent.futures import ThreadPoolExecutor
import keyboard
import socket
import json
import struct
import re
from pathlib import Path
import signal as system_signal  # Rename to avoid conflict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

shutdown_event = threading.Event()  # Add this line
observers = []  # Move this to global scope

# Constants
DATA_DIRECTORY = r"R:\Release_Internal_V1A\output"  # Directory to monitor for txt files
TIMESTAMP_FILE = "last_processed.txt"  # File to store last processed file number
LOG_FILENAME = "calibration_log.txt"

# TCP Server for sending data to aprilTagLocalization
TCP_SERVER_HOST = 'localhost'
TCP_SERVER_PORT = 12347
tcp_clients = []
tcp_server_running = False
tcp_server_socket = None

# Data structures for plotting and processing
PLOT_BUFFER_LEN = 1000
plot_time = deque(maxlen=PLOT_BUFFER_LEN)
plot_mag_x = deque(maxlen=PLOT_BUFFER_LEN)
plot_mag_y = deque(maxlen=PLOT_BUFFER_LEN)
plot_mag_z = deque(maxlen=PLOT_BUFFER_LEN)
plot_amp_x = deque(maxlen=PLOT_BUFFER_LEN)
plot_amp_y = deque(maxlen=PLOT_BUFFER_LEN)
plot_amp_z = deque(maxlen=PLOT_BUFFER_LEN)
plot_lp_x = deque(maxlen=PLOT_BUFFER_LEN)
plot_lp_y = deque(maxlen=PLOT_BUFFER_LEN)
plot_lp_z = deque(maxlen=PLOT_BUFFER_LEN)
plot_hp_x = deque(maxlen=PLOT_BUFFER_LEN)
plot_hp_y = deque(maxlen=PLOT_BUFFER_LEN)
plot_hp_z = deque(maxlen=PLOT_BUFFER_LEN)

# Data queue for processing
data_queue = queue.Queue()
forced_invert_signs = False
logging_enabled = False
log_file = None
logging_source = "magnetometer"
one_shot_logging_trigger = threading.Event()
logging_lock = threading.Lock()
q_prev = None  # Global reference tracking for gyro-predicted field direction


def get_log_filename():
    """Return the calibration file associated with the selected sensor."""
    return LOG_FILENAME


def log_sensor_sample(sample, one_shot=False):
    """Write one raw X,Y,Z sensor sample in the calibration tool format."""
    global log_file, logging_enabled
    values = np.asarray(sample, dtype=float)
    if values.shape != (3,):
        raise ValueError(f"Expected a three-axis sample, received shape {values.shape}")

    with logging_lock:
        if one_shot:
            with open(get_log_filename(), "a") as output_file:
                output_file.write(f"{values[0]:.6f},{values[1]:.6f},{values[2]:.6f}\n")
            print(f"Logged one {logging_source} sample to {get_log_filename()}")
        elif logging_enabled and log_file:
            log_file.write(f"{values[0]:.6f},{values[1]:.6f},{values[2]:.6f}\n")
            log_file.flush()


def start_logging():
    """Start continuous raw data logging for the selected sensor."""
    global log_file, logging_enabled
    with logging_lock:
        log_file = open(get_log_filename(), "w")
        logging_enabled = True
    print(f"Started continuous {logging_source} logging to {get_log_filename()}")


def stop_logging():
    """Stop continuous data logging."""
    global log_file, logging_enabled
    with logging_lock:
        if log_file:
            log_file.close()
            log_file = None
        logging_enabled = False
    print("Stopped continuous logging")


# def toggle_logging_source():
#     """Switch the sensor whose samples are captured by the logging controls."""
#     global logging_source
#     if logging_enabled:
#         print("Stop continuous logging before switching the logging source")
#         return
#     logging_source = "imu" if logging_source == "magnetometer" else "magnetometer"
#     print(f"Logging source set to {logging_source} ({get_log_filename()})")

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qw*qz,     2*qx*qz + 2*qw*qy],
        [    2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qw*qx],
        [    2*qx*qz - 2*qw*qy,     2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ])


def quaternion_ned_to_seu(q):
    """Express a body-to-world quaternion in the SEU world frame instead of NED."""
    qw, qx, qy, qz = q
    return np.array([-qy, qz, qw, -qx])


class MadgwickFilter:
    def __init__(self, alpha, beta):
        """
        Initialize Madgwick filter parameters.
        
        Args:
            beta (float): Algorithm gain beta
            dt (float): Sample period [s]
        """
        self.alpha = alpha  # Low-pass filter coefficient for quaternion smoothing
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.filtered_q = np.array([1.0, 0.0, 0.0, 0.0])  # For low-pass filtering of quaternion
        self.last_time = None  # Track last update time
        
        # Reference vectors in Earth frame
        self.gravity_ref = np.array([0.0, 0.0, 1.0])  # Gravity points down
        self.magnetic_ref = np.array([1.0, 0.0, 0.0])  # Magnetic field points North

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quaternion_conjugate(self, q):
        """Compute quaternion conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def normalize_quaternion(self, q):
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm > 0:
            return q / norm
        return q

    def normalize_vector(self, v):
        """Normalize vector to unit length."""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    def init(self):
        """
        Initialize or reset filter state.
        """
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        return self.q

    def update(self, gyro, accel, mag, current_time=None):
        """
        Update Madgwick filter with gyro, accelerometer and magnetometer data (9DOF).
        
        Args:
            gyro (np.ndarray): Angular velocity [rad/s] (3x1: [wx, wy, wz])
            accel (np.ndarray): Accelerometer data [g] (3x1: [ax, ay, az])
            mag (np.ndarray): Magnetometer data [uT] (3x1: [mx, my, mz])
        
        Returns:
            tuple: Updated quaternion, error vector, corrected gyro
        """

        # Calculate actual dt
        if current_time is None:
            current_time = time.time()
            
        if self.last_time is None:
            dt = 0.01  # Initial guess
        else:
            dt = current_time - self.last_time
            
        self.last_time = current_time

        # Normalize accelerometer and magnetometer
        accel_normalized = self.normalize_vector(accel)
        mag_normalized = self.normalize_vector(mag)
        
        # Extract quaternion elements
        q0, q1, q2, q3 = self.q
        
        # Reference direction of Earth's magnetic field
        h = self.quaternion_multiply(
            self.quaternion_multiply(self.q, np.array([0.0, *mag_normalized])),
            self.quaternion_conjugate(self.q)
        )[1:]
        
        b = np.array([0.0, np.linalg.norm([self.magnetic_ref[0], self.magnetic_ref[1]]), 
                        0.0, self.magnetic_ref[2]])        
        
        # Gradient descent algorithm corrective step
        # Objective function and Jacobian for accelerometer and magnetometer
        f = np.array([
            2.0 * (q1*q3 - q0*q2) - accel_normalized[0],
            2.0 * (q0*q1 + q2*q3) - accel_normalized[1],
            2.0 * (0.5 - q1*q1 - q2*q2) - accel_normalized[2],
            2.0 * b[1] * (0.5 - q2*q2 - q3*q3) + 2.0 * b[3] * (q1*q3 - q0*q2) - mag_normalized[0],
            2.0 * b[1] * (q1*q2 - q0*q3) + 2.0 * b[3] * (q0*q1 + q2*q3) - mag_normalized[1],
            2.0 * b[1] * (q0*q2 + q1*q3) + 2.0 * b[3] * (0.5 - q1*q1 - q2*q2) - mag_normalized[2]
        ])
        
        J = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0.0, -4.0*q1, -4.0*q2, 0.0],
            [-2.0*b[3]*q2, 2.0*b[3]*q3, -4.0*b[1]*q2 - 2.0*b[3]*q0, -4.0*b[1]*q3 + 2.0*b[3]*q1],
            [-2.0*b[1]*q3 + 2.0*b[3]*q1, 2.0*b[1]*q2 + 2.0*b[3]*q0, 2.0*b[1]*q1 + 2.0*b[3]*q3, -2.0*b[1]*q0 + 2.0*b[3]*q2],
            [2.0*b[1]*q2, 2.0*b[1]*q3 - 4.0*b[3]*q1, 2.0*b[1]*q0 - 4.0*b[3]*q2, 2.0*b[1]*q1]
        ])
        
        # Compute gradient
        gradient = J.T @ f
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / np.linalg.norm(gradient)
        
        # Compute quaternion derivative from gyro and gradient descent
        q_dot_gyro = 0.5 * self.quaternion_multiply(self.q, np.array([0.0, *gyro]))
        q_dot_gradient = self.beta * gradient
        
        # For debugging - estimate corrected gyro
        # This is an approximation for visualization purposes
        corrected_gyro = gyro - 2.0 * self.beta * gradient[1:]
        
        # Integrate rate of change of quaternion
        q_dot = q_dot_gyro - q_dot_gradient
        self.q = self.q + q_dot * dt
        self.q = self.normalize_quaternion(self.q)

        # adding low pass filter for q
        self.filtered_q = (1 - self.alpha) * self.filtered_q + self.alpha * self.q
        
        # Return error vector for debugging (approximated from gradient)
        error = gradient[1:] * self.beta
        
        return self.filtered_q, error, corrected_gyro
    


class LockInAmplifier:
    def __init__(self, target_freq, alpha, hp_alpha):
        self.target_freq = target_freq
        self.alpha = alpha
        self.hp_alpha = hp_alpha  # High-pass filter coefficient
        self.x_in = [0.0, 0.0, 0.0]
        self.x_quad = [0.0, 0.0, 0.0]
        self.hpf_dc_offsets = [0.0, 0.0, 0.0]  # DC offset estimates for X, Y, Z
        self.hpf_mag_uT = [0.0, 0.0, 0.0]  # High-pass filtered signals
        self.last_time = time.time()
        self.phase_offsets = [0.0, 0.0, 0.0]  # Store phase offsets for X, Y, Z
        self.is_calibrated = False

    def high_pass_filter(self, mag_uT):
            """Apply high-pass filter to remove DC offset from input signals."""
            _hpf_filtered = [0.0, 0.0, 0.0]
            
            for i in range(3):
                # 1. Update the running DC offset estimate
                self.hpf_dc_offsets[i] = (1 - self.hp_alpha) * self.hpf_dc_offsets[i] + self.hp_alpha * mag_uT[i]
                
                # 2. Subtract it from the raw value to isolate the AC wave
                _hpf_filtered[i] = mag_uT[i] - self.hpf_dc_offsets[i]
                
            return _hpf_filtered

    def calibrate(self, mag_uT, current_time):
        """Calibrate phase angles to be negative in the current orientation."""
        # Apply high-pass filter to remove DC offset
        filtered_mag = self.high_pass_filter(mag_uT)
        

        ref_in = np.sin(2 * np.pi * self.target_freq * current_time)
        ref_quad = np.cos(2 * np.pi * self.target_freq * current_time)

        for i in range(3):
            prod_in = filtered_mag[i] * ref_in
            prod_quad = filtered_mag[i] * ref_quad

            self.x_in[i] = (1 - self.alpha) * self.x_in[i] + self.alpha * prod_in
            self.x_quad[i] = (1 - self.alpha) * self.x_quad[i] + self.alpha * prod_quad

            current_phase = np.arctan2(self.x_quad[i], self.x_in[i])

            # Adjust phase to be negative (shift to range [-180, 0])
            if current_phase >= 0:
                self.phase_offsets[i] = 1

        self.is_calibrated = True
        print("Calibration complete. Phase offsets (degrees):", [np.degrees(off) for off in self.phase_offsets])
        print("DC offsets (uT):", self.dc_offsets)

    def update(self, mag_uT, current_time):
    # """Paper-based lock-in amplifier that uses phase differences only."""
        global gyro_predicted_field,forced_invert_signs

        self.hpf_mag_uT = self.high_pass_filter(mag_uT)
        _filtered_mag = self.hpf_mag_uT
        
        # Use PLL or fixed frequency reference
        _ref_in = np.sin(2 * np.pi * self.target_freq * current_time)
        _ref_quad = np.cos(2 * np.pi * self.target_freq * current_time)
        
        # Get amplitudes and raw phases for each axis
        amplitudes = []
        raw_phases = []
        
        for i in range(3):
            _prod_in = _filtered_mag[i] * _ref_in
            _prod_quad = _filtered_mag[i] * _ref_quad
            ##
            self.x_in[i] = (1 - self.alpha) * self.x_in[i] + self.alpha * _prod_in
            self.x_quad[i] = (1 - self.alpha) * self.x_quad[i] + self.alpha * _prod_quad
            
            _amplitude = np.sqrt(self.x_in[i]**2 + self.x_quad[i]**2)
            _phase = np.arctan2(self.x_quad[i], self.x_in[i])
            
            amplitudes.append(_amplitude)
            raw_phases.append(_phase)
        
        # PAPER METHOD: Determine signs using phase differences (Equation 1)
        # Reference: Choose axis with strongest signal as reference
        ref_axis = np.argmax(amplitudes)
        signs = [1.0, 1.0, 1.0]
        
        for i in range(3):
            if i == ref_axis:
                signs[i] = 1.0  # Reference axis sign arbitrary, will be resolved by gyro
            else:
                # Calculate phase difference
                phase_diff = abs(raw_phases[i] - raw_phases[ref_axis])
                # If phase difference > 90°, they're opposite
                if phase_diff > np.pi/2:
                    signs[i] = -1.0
                else:
                    signs[i] = 1.0
        
        # Create signed amplitude vector (still ambiguous: ±vector)
        _signed_amplitudes = [amplitudes[i] * signs[i] for i in range(3)]
        # print(f"Raw phases (degrees): {[np.degrees(p) for p in raw_phases]}, Signs: {signs}, Amplitudes: {amplitudes}")
        
        # PAPER METHOD: Use gyro to resolve final sign (Equation 2)
        # You need to pass the gyro-predicted field direction
        if gyro_predicted_field is not None:
            # print(f"Gyro predicted field: {gyro_predicted_field}, Signed amplitudes: {signed_amplitudes}")
            # Dot product to determine if vector points roughly same direction as predicted
            dot = (_signed_amplitudes[0] * gyro_predicted_field[0] +
                _signed_amplitudes[1] * gyro_predicted_field[1] +
                _signed_amplitudes[2] * gyro_predicted_field[2])
            
            if dot < 0:
                # Flip all signs if pointing opposite to prediction
                _signed_amplitudes = [-s for s in _signed_amplitudes]
                
        # if forced_invert_signs:
        #     _signed_amplitudes = [-s for s in _signed_amplitudes] 
        #     forced_invert_signs = False  # Reset after forcing sign inversion for testing
        
        return _signed_amplitudes, [np.degrees(p) for p in raw_phases], _filtered_mag, current_time

    def reset_dc_offsets(self):
        """Reset DC offset estimates (useful when conditions change dramatically)."""
        self.dc_offsets = [0.0, 0.0, 0.0]
        self.filtered_signals = [0.0, 0.0, 0.0]
        print("DC offsets reset")


class DataFileHandler(FileSystemEventHandler):
    """Handler for new file creation events."""
    def __init__(self, processor):
        self.processor = processor
        self.processed_files = set()
        self.load_last_processed()
        
    def load_last_processed(self):
        """Load the last processed file number from timestamp file."""
        try:
            if os.path.exists(TIMESTAMP_FILE):
                with open(TIMESTAMP_FILE, 'r') as f:
                    last_num = int(f.read().strip())
                    self.processed_files.add(last_num)
                    print(f"Resumed from file number: {last_num}")
        except Exception as e:
            print(f"Error loading last processed file: {e}")
    
    def save_last_processed(self, file_num):
        """Save the last processed file number."""
        try:
            with open(TIMESTAMP_FILE, 'w') as f:
                f.write(str(file_num))
        except Exception as e:
            print(f"Error saving last processed file: {e}")
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            self.handle_file_with_retry(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            self.handle_file_with_retry(event.src_path)

    def handle_file_with_retry(self, filepath):
        """Handles race conditions when reading files that are being written."""
        max_retries = 10
        retry_delay = 0.02
        for attempt in range(max_retries):
            try:
                with open(filepath, 'r') as f:
                    pass
                self.processor.process_file(filepath)
                return
            except PermissionError:
                time.sleep(retry_delay)
        # Final safety net check if loop terminates locked
        threading.Thread(target=self.delayed_fallback_parse, args=(filepath,), daemon=True).start()

    def delayed_fallback_parse(self, filepath):
        time.sleep(0.15)
        try:
            self.processor.process_file(filepath)
        except:
            pass

class DataProcessor:
    """Process data from image info files."""
    
    # Magnetometer calibration parameters
    HARD_IRON_BIAS = [-24.425272, -9.840492, 16.169081]
    

    
    def __init__(self):
        self.filter = MadgwickFilter( alpha=1.0, beta=0.3)
        self.q = self.filter.init()
        self.lockin = LockInAmplifier(target_freq=10.0, alpha=0.06 , hp_alpha=0.06)
        self.lowpass_mag = None
        self.ac_lowpass_mag = None
        self.mod_q_lowpass = None
        self.last_file_num = -1
        self.last_packet_timestamp = -1.0
        self.data_buffer = []
        self.packet_timestamps = deque(maxlen=100)
        self.input_rate_hz = 0.0
        self.files_completed = 0
        self.file_extraction_rate_hz = 0.0
        self.file_rate_window_start_time = None
        self.file_rate_window_file_count = 0
        
    def extract_file_number(self, filename):
        """Extract the sequence number from filename (imgInfo_XXXXX_...)."""
        match = re.search(r'imgInfo_(\d+)_', filename)
        return int(match.group(1)) if match else -1

    def update_sensor_input_rate(self, timestamp):
        """Estimate the sensor packet rate from the timestamps embedded in the logged file data."""
        self.packet_timestamps.append(timestamp)
        if len(self.packet_timestamps) >= 2:
            elapsed = self.packet_timestamps[-1] - self.packet_timestamps[0]
            if elapsed > 0:
                self.input_rate_hz = (len(self.packet_timestamps) - 1) / elapsed
        return self.input_rate_hz

    def update_file_extraction_rate(self):
        """Estimate how quickly files finish data extraction."""
        self.files_completed += 1
        if self.file_rate_window_start_time is None:
            self.file_rate_window_start_time = time.time()
            self.file_rate_window_file_count = 1
        else:
            self.file_rate_window_file_count += 1
            elapsed = time.time() - self.file_rate_window_start_time
            if elapsed > 0:
                self.file_extraction_rate_hz = self.file_rate_window_file_count / elapsed
            if self.file_rate_window_file_count >= 10:
                self.file_rate_window_start_time = time.time()
                self.file_rate_window_file_count = 0
        return self.file_extraction_rate_hz
    
    def calibrate_magnetometer(self, mag_raw, pitch_deg=0, roll_deg=0, yaw_deg=0):
        """
        Calibrate raw magnetometer data using hard iron bias, soft iron matrix,
        and a dynamically generated boresight rotation matrix from Euler angles.
        
        The boresight matrix uses the ZYX (Yaw-Pitch-Roll) intrinsic sequence:
        R = R_x(roll) @ R_y(pitch) @ R_z(yaw)
        
        Args: 
            mag_raw: Raw magnetometer readings [x, y, z] in microTesla
            pitch_deg: Physical tilt around Y-axis in degrees
            roll_deg: Physical tilt around X-axis in degrees
            yaw_deg: Physical tilt around Z-axis in degrees
            
        Returns:
            Calibrated and structurally aligned readings [x, y, z] in microTesla
        """
        # Convert to numpy array for vectorized operations
        mag_raw = np.array(mag_raw)
        
        # 1. HARD IRON BIAS (Offset correction)
        magnet_bias = np.array([0, 0, 0])
        hard_iron_bias = np.array([-131.191364, 13729.563502, 2404.219664])
        soft_iron_matrix = np.array([
            [1.024690, 0.047071, -0.005928],
            [0.047071, 0.919793, 0.065566],
            [-0.005928, 0.065566, 1.006852]
        ])
        
        # 3. GENERATE BORESIGHT CORRECTION MATRIX (R_mc) FROM EULER ANGLES
        # Convert angles from degrees to radians
        alpha = np.radians(yaw_deg)    # Rotation around Z
        beta  = np.radians(pitch_deg)  # Rotation around Y
        gamma = np.radians(roll_deg)   # Rotation around X
        
        # Individual axis rotation matrices
        R_z = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha),  np.cos(alpha), 0],
            [0,              0,             1]
        ])
        
        R_y = np.array([
            [ np.cos(beta), 0, np.sin(beta)],
            [ 0,            1, 0           ],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        
        R_x = np.array([
            [1, 0,              0             ],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma),  np.cos(gamma)]
        ])
        
        # Combined ZYX transformation matrix
        R_mc = R_x @ R_y @ R_z
        
        # --- Execute Calibration Pipeline ---
        
        # Step A: Apply ellipsoid correction (Raw -> Corrected Sensor Frame)
        mag_ellipsoid_corrected = (mag_raw - hard_iron_bias - magnet_bias) @ soft_iron_matrix.T
        
        # Step B: Apply Boresight Alignment (Corrected Sensor Frame -> Physical Board Frame)
        mag_final = R_mc @ mag_ellipsoid_corrected
        
        
        return mag_final.tolist()
    
    def calibrate_accelerometer(self, accel_raw, tune_roll_deg=0, tune_pitch_deg=0):
        """
        Calibrate raw accelerometer data using bias parameters, a static calibration matrix,
        and a dynamically calculated boresight correction matrix based on custom tuning angles.
        """
        # Bias parameters (offset correction)
        bias_x = -0.011205
        bias_y = -0.012890
        bias_z = 0.024886
        
        # Calibration matrix (scale and misalignment correction)
        cal_matrix_xx = 0.989532; cal_matrix_xy = -0.001051; cal_matrix_xz = 0.001429
        cal_matrix_yx = -0.001051; cal_matrix_yy = 0.989770; cal_matrix_yz = -0.001372
        cal_matrix_zx = 0.001429; cal_matrix_zy = -0.001372; cal_matrix_zz = 1.003220

        # Dynamic Boresight Matrix Calculation
        # Convert tuning parameters to radians and invert signs to rotate backwards
        alpha = -np.radians(tune_roll_deg)
        beta = -np.radians(tune_pitch_deg)
        
        # Rotation about X-axis (Roll)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])
        
        # Rotation about Y-axis (Pitch)
        R_y = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        
        # Combine matrices using the Roll-then-Pitch order (R_y * R_x)
        R_correct = np.dot(R_y, R_x)
        
        # Map the generated elements directly to your variables
        cor_matrix_xx, cor_matrix_xy, cor_matrix_xz = R_correct[0]
        cor_matrix_yx, cor_matrix_yy, cor_matrix_yz = R_correct[1]
        cor_matrix_zx, cor_matrix_zy, cor_matrix_zz = R_correct[2]

        # Step 1: Apply bias correction (remove offset)
        accel_biased = [
            accel_raw[0] - bias_x,
            accel_raw[1] - bias_y,
            accel_raw[2] - bias_z
        ]
        
        # Step 2: Apply calibration matrix (correct scale and alignment)
        accel_calibrated = [
            cal_matrix_xx * accel_biased[0] + cal_matrix_xy * accel_biased[1] + cal_matrix_xz * accel_biased[2],
            cal_matrix_yx * accel_biased[0] + cal_matrix_yy * accel_biased[1] + cal_matrix_yz * accel_biased[2],
            cal_matrix_zx * accel_biased[0] + cal_matrix_zy * accel_biased[1] + cal_matrix_zz * accel_biased[2]
        ]
        # print(f"Calibrated Accel: [X: {accel_calibrated[0]:.4f}g,\tY: {accel_calibrated[1]:.4f}g,\tZ: {accel_calibrated[2]:.4f}g]")

        # Step 3: Apply correction matrix (final adjustment)
        accel_corrected = [
            cor_matrix_xx * accel_calibrated[0] + cor_matrix_xy * accel_calibrated[1] + cor_matrix_xz * accel_calibrated[2],
            cor_matrix_yx * accel_calibrated[0] + cor_matrix_yy * accel_calibrated[1] + cor_matrix_yz * accel_calibrated[2],
            cor_matrix_zx * accel_calibrated[0] + cor_matrix_zy * accel_calibrated[1] + cor_matrix_zz * accel_calibrated[2]
        ]
        # print(f"Corrected Accel:  [X: {accel_corrected[0]:.4f}g,\tY: {accel_corrected[1]:.4f}g,\tZ: {accel_corrected[2]:.4f}g]")

        return accel_corrected
    
    def parse_data_packet(self, data_bytes, base_time):
        """Parse a single data packet of 4+12+7=23 bytes."""
        if len(data_bytes) < 23:
            return None
        
        timestamp_offset = struct.unpack('>I', data_bytes[0:4])[0]  # Big-endian unsigned int
        timestamp = base_time + timestamp_offset
        timestamp = timestamp / 1000.0
        
        gyro_raw = struct.unpack('<hhh', data_bytes[4:10])
        accel_raw = struct.unpack('<hhh', data_bytes[10:16])
        
        mag_status = data_bytes[16]
        mag_data = struct.unpack('>hhh', data_bytes[17:23])
        
        accel_g = [x * 2.0 / 32768 for x in accel_raw]
        gyro_dps = [x * 245.0 / 32768 for x in gyro_raw]
        mag_uT = [x * 1.1 for x in mag_data]

        # Apply coordinate transformations

        tmp_ax, tmp_ay, tmp_az = accel_g[0], accel_g[1], accel_g[2]

        accel_g[0] = -tmp_ay
        accel_g[1] = tmp_az
        accel_g[2] = tmp_ax

        tmp_gx, tmp_gy, tmp_gz = gyro_dps[0], gyro_dps[1], gyro_dps[2]

        gyro_dps[0] = tmp_gy
        gyro_dps[2] = -tmp_gx
        gyro_dps[1] = -tmp_gz
        


        temp = mag_uT[0]
        mag_uT[0] = -mag_uT[1]
        mag_uT[1] = temp

        mag_uT[2] = -mag_uT[2]




        
        return timestamp, accel_g, gyro_dps, mag_uT
    
    def rotate_quaternion_z_neg90_manual(self, q):
        """Rotate quaternion by -90 degrees about Z-axis."""
        angle_rad = np.radians(-90)
        w_rot = np.cos(angle_rad / 2)
        z_rot = np.sin(angle_rad / 2)
        rotation_quat = np.array([w_rot, 0, 0, z_rot])
        
        w1, x1, y1, z1 = rotation_quat
        w2, x2, y2, z2 = q
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def quaternion_a_to_b(self, q_a):
        """Convert quaternion from system A to system B."""
        w, x, y, z = q_a
        q_b = np.array([w, -y, -x, -z])
        return q_b / np.linalg.norm(q_b)
    
    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles with modified coordinate system."""
        w, x, y, z = q
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.degrees(np.arctan2(t3, t4))
        
        # roll_x = -roll_x
        # pitch_y = -pitch_y
        
        return roll_x, pitch_y, yaw_z
    
    def process_file(self, filepath):
        """Process a single data file."""
        try:
            file_num = self.extract_file_number(os.path.basename(filepath))
            if file_num <= self.last_file_num:
                return
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            hex_string = ''.join(content.split())
            data = bytes.fromhex(hex_string)

            data = data[6:131] + data[131+8:244]
            packet_size = 23
            num_packets = len(data) // packet_size
            if num_packets == 0:
                return

            parsed_packets = []
            for i in range(num_packets):
                packet_offset = i * packet_size
                if packet_offset + packet_size > len(data):
                    break

                packet_data = data[packet_offset:packet_offset + packet_size]
                result = self.parse_data_packet(packet_data, 0)
                if not result:
                    continue

                timestamp, accel_g, gyro_dps, mag_uT = result
                if timestamp <= self.last_packet_timestamp:
                    continue

                parsed_packets.append((timestamp, accel_g, gyro_dps, mag_uT))

            if not parsed_packets:
                self.last_file_num = file_num
                self.update_file_extraction_rate()
                handler = next((h for h in observers if isinstance(h, DataFileHandler)), None)
                if handler:
                    handler.save_last_processed(file_num)
                return

            parsed_packets.sort(key=lambda item: item[0])
            for timestamp, accel_g, gyro_dps, mag_uT in parsed_packets:
                self.last_packet_timestamp = timestamp
                self.update_sensor_input_rate(timestamp)

                mag_calibrated = self.calibrate_magnetometer(mag_uT)
                accel_calibrated = self.calibrate_accelerometer(accel_g)


                sample_to_log = mag_calibrated if logging_source == "magnetometer" else accel_g


                ## For "l" one shot logging
                if one_shot_logging_trigger.is_set():
                    log_sensor_sample(sample_to_log, one_shot=True)
                    one_shot_logging_trigger.clear()
                    
                ## For "m" continuous logging
                log_sensor_sample(sample_to_log)




                data_queue.put((timestamp, accel_calibrated, gyro_dps, mag_calibrated, 0))
                self.data_buffer.append((timestamp, accel_calibrated, gyro_dps, mag_calibrated))
            
            self.update_file_extraction_rate()
            self.last_file_num = file_num
            handler = next((h for h in observers if isinstance(h, DataFileHandler)), None)
            if handler:
                handler.save_last_processed(file_num)
                
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    def process_buffered_data(self):
        """Process all data in the buffer (for initial load)."""
        self.data_buffer.sort(key=lambda x: x[0])
        for timestamp, accel_g, gyro_dps, mag_uT in self.data_buffer:
            data_queue.put((timestamp, accel_g, gyro_dps, mag_uT, 0))

def processing_thread(processor):
    """Background thread for data processing."""
    global q_prev, forced_invert_signs, gyro_predicted_field
    filter = processor.filter
    q = processor.q
    lockin = processor.lockin
    lowpass_mag = None
    ac_lowpass_mag = None
    mod_q_lowpass = None
    alpha = 0.08
    alpha_amp = 0.4
    alpha_q = 0.01
    
    # Update rate counters
    packet_count = 0
    start_time = time.time()
    update_rate = 0.0
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        while not shutdown_event.is_set():
            try:
                current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get(timeout=1.0)
                # print(f"Sign Corrected Raw a[0]: {accel_g[0]:.3f} g, a[1]: {accel_g[1]:.3f} g, a[2]: {accel_g[2]:.3f} g, m[0]: {mag_uT[0]:.3f} uT, m[1]: {mag_uT[1]:.3f} uT, m[2]: {mag_uT[2]:.3f} uT, gyro[0]: {gyro_dps[0]:.3f} dps, gyro[1]: {gyro_dps[1]:.3f} dps, gyro[2]: {gyro_dps[2]:.3f} dps")

                # Track calculation update frequency
                packet_count += 1
                if packet_count >= 50:
                    end_time = time.time()
                    update_rate = packet_count / (end_time - start_time)
                    packet_count = 0
                    start_time = time.time()

                # Explicitly handle manual reset trigger requested via keyboard input
                if forced_invert_signs:
                    lockin.x_in = [0.0, 0.0, 0.0]
                    lockin.x_quad = [0.0, 0.0, 0.0]
                    lockin.hpf_dc_offsets = [0.0, 0.0, 0.0]
                    lockin.hpf_mag_uT = [0.0, 0.0, 0.0]
                    q = filter.init()
                    q_prev = None
                    gyro_predicted_field = None
                    forced_invert_signs = False
                    print("Tracking states reset manually.")

                # Calculate expected AC field orientation relative to the sensor body frame
                if q_prev is not None:
                    ac_field_global = np.array([-1.0, 0.0, 0.0])  # North is world -X in SEU
                    R = quaternion_to_rotation_matrix(q_prev)
                    gyro_predicted_field = R.T @ ac_field_global
                else:
                    gyro_predicted_field = None

                # Update lock-in amplifier utilizing the same gyro-predicted sign resolution logic as bt_angleEst.py
                mag_amplitudes, mag_phases, highpass_mag, _ = lockin.update(mag_uT, current_time)
                
                # Update Madgwick filter with calculated signed amplitudes
                q_filter, error, omega_corrected = filter.update(
                    np.radians(gyro_dps),
                    np.array(accel_g),
                    np.array(mag_amplitudes),
                    current_time=current_time
                )
                q = quaternion_ned_to_seu(q_filter)
                q_prev = q.copy()  # Cache target frame for next iteration loop prediction
                
                # Apply low-pass filters
                mag_array = np.array(mag_uT)
                if lowpass_mag is None:
                    lowpass_mag = mag_array
                else:
                    lowpass_mag = alpha * mag_array + (1 - alpha) * lowpass_mag
                
                # # Rotate quaternion

                mod_q = q
                
                # Apply quaternion low-pass
                if mod_q_lowpass is None:
                    mod_q_lowpass = mod_q
                mod_q_lowpass = alpha_q * mod_q + (1 - alpha_q) * mod_q_lowpass
                mod_q = mod_q_lowpass
                
                # Get Euler angles
                pitch, roll, yaw = processor.quaternion_to_euler(mod_q)
                
                # Apply low-pass to amplitudes
                mag_amplitudes = np.array(mag_amplitudes)
                if ac_lowpass_mag is None:
                    ac_lowpass_mag = mag_amplitudes
                ac_lowpass_mag = alpha_amp * mag_amplitudes + (1 - alpha_amp) * ac_lowpass_mag
                
                # Print results (Now includes current data update rate)
                mag_magnitude = np.sqrt(mag_array[0]**2 + mag_array[1]**2 + mag_array[2]**2)
                ac_magnitude = np.sqrt(mag_amplitudes[0]**2 + mag_amplitudes[1]**2 + mag_amplitudes[2]**2)

                print(
                    # f"Yaw: {yaw:.3f}°\tPitch: {pitch:.3f}°\tRoll: {roll:.3f}°\t"
                    # f"Mag: {mag_magnitude:.1f} µT\tAC: {ac_magnitude:.1f} µT\t"
                    # f" Raw  mX: {mag_uT[0]:.3f} uT, mY: {mag_uT[1]:.3f} uT, mZ: {mag_uT[2]:.3f} uT"
                    f"qw: {mod_q[0]:.3f}\tqx: {mod_q[1]:.3f}\tqy: {mod_q[2]:.3f}\tqz: {mod_q[3]:.3f}\tYaw: {yaw:.3f}°\t Pitch: {pitch:.3f}°\tRoll: {roll:.3f}°"
                    # f"Sensor: {processor.input_rate_hz:.2f} Hz\t"
                    # f"File: {processor.file_extraction_rate_hz:.2f} files/s\t"
                    # f"files #: {processor.files_completed}\t"
                    # f"Processing: {update_rate:.2f} Hz\t"
                    # f"D.P. Timestamp: {current_time:.3f}s"
                    # f"Mag: {mag_uT[0]:.3f} uT, {mag_uT[1]:.3f} uT, {mag_uT[2]:.3f} uT\tAccel: {accel_g[0]:.3f} g, {accel_g[1]:.3f} g, {accel_g[2]:.3f} g\tGyro: {gyro_dps[0]:.3f} dps, {gyro_dps[1]:.3f} dps, {gyro_dps[2]:.3f} dps"
                    )
                
                # Update plots
                plot_time.append(current_time)
                plot_mag_x.append(mag_uT[0])
                plot_mag_y.append(mag_uT[1])
                plot_mag_z.append(mag_uT[2])
                plot_amp_x.append(mag_amplitudes[0])
                plot_amp_y.append(mag_amplitudes[1])
                plot_amp_z.append(mag_amplitudes[2])
                plot_lp_x.append(lowpass_mag[0])
                plot_lp_y.append(lowpass_mag[1])
                plot_lp_z.append(lowpass_mag[2])
                plot_hp_x.append(highpass_mag[0])
                plot_hp_y.append(highpass_mag[1])
                plot_hp_z.append(highpass_mag[2])
                
                # Send to TCP clients
                send_to_tcp_clients(mod_q, lowpass_mag, ac_lowpass_mag)
                data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                if not shutdown_event.is_set():
                    print(f"Processing error: {e}")               
                time.sleep(0.1)

def send_to_tcp_clients(q, mag_uT, ac_mag_uT):
    """Send quaternion and magnetometer data to connected TCP clients."""
    global tcp_clients
    if not tcp_clients or not tcp_server_running:
        return
    
    data = {
        'qw': float(q[0]),
        'qx': float(q[1]),
        'qy': float(q[2]),
        'qz': float(q[3]),
        'magX': float(mag_uT[0]),
        'magY': float(mag_uT[1]),
        'magZ': float(mag_uT[2]),
        'lockInX': float(ac_mag_uT[0]),
        'lockInY': float(ac_mag_uT[1]),
        'lockInZ': float(ac_mag_uT[2]),
        'timestamp': time.time()
    }
    
    data_str = json.dumps(data)
    data_bytes = data_str.encode('utf-8')
    length = len(data_bytes)
    
    clients_to_remove = []
    for client_socket in tcp_clients:
        try:
            client_socket.sendall(struct.pack('>I', length))
            client_socket.sendall(data_bytes)
        except (socket.error, BrokenPipeError):
            clients_to_remove.append(client_socket)
    
    for client_socket in clients_to_remove:
        tcp_clients.remove(client_socket)
        client_socket.close()

def tcp_server_thread():
    """Run TCP server to send data to aprilTagLocalization."""
    global tcp_server_running, tcp_server_socket, tcp_clients
    
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_server_socket.bind((TCP_SERVER_HOST, TCP_SERVER_PORT))
    tcp_server_socket.listen(5)
    tcp_server_socket.settimeout(0.5)
    tcp_server_running = True
    
    print(f"TCP server listening on {TCP_SERVER_HOST}:{TCP_SERVER_PORT}")
    
    while tcp_server_running and not shutdown_event.is_set():
        try:
            client_socket, addr = tcp_server_socket.accept()
            client_socket.settimeout(0.5)
            tcp_clients.append(client_socket)
            print(f"Client connected from {addr}")
        except socket.timeout:
            continue
        except Exception as e:
            if tcp_server_running and not shutdown_event.is_set():
                print(f"TCP server error: {e}")
    
    for client_socket in tcp_clients:
        try:
            client_socket.close()
        except:
            pass
    tcp_clients.clear()
    
    if tcp_server_socket:
        tcp_server_socket.close()

def keyboard_listener():
    """Listen for keyboard input to correct phase signs or reset states."""
    global forced_invert_signs
    print("Press 'c' to correct phase / invert signs using dynamic tracking context")
    print("Press 'r' to reset DC offsets")
    print("Press 's' to switch logging source (magnetometer or IMU accelerometer)")
    print("Press 'l' to log one sample from the selected source")
    print("Press 'm' to start/stop continuous logging for the selected source")
    print("Press 'q' to quit")
    
    while not shutdown_event.is_set():
        try:
            if keyboard.is_pressed('c'):
                print("'c' pressed: Phase correction requested.")
                forced_invert_signs = True
                time.sleep(0.5)
            
            if keyboard.is_pressed('r'):
                lockin = next((th for th in threading.enumerate() if hasattr(th, 'lockin')), None)
                if lockin:
                    lockin.reset_dc_offsets()
                time.sleep(0.5)

            # if keyboard.is_pressed('s'):
            #     toggle_logging_source()
            #     time.sleep(0.5)

            if keyboard.is_pressed('l'):
                one_shot_logging_trigger.set()
                print(f"One {logging_source} sample will be logged to {get_log_filename()}")
                time.sleep(0.5)

            if keyboard.is_pressed('m'):
                if logging_enabled:
                    stop_logging()
                else:
                    start_logging()
                time.sleep(0.5)
                
            if keyboard.is_pressed('q'):
                print("Quitting...")
                shutdown_event.set()
                break
                
            time.sleep(0.1)
        except:
            break

def live_plot():
    """Create real-time plots of sensor data."""
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    ax1, ax2, ax3, ax4 = axs
    
    ax1.set_title('Raw Magnetometer (uT)')
    ax2.set_title('Lock-in AC Amplitude (uT)')
    ax3.set_title('Low-pass Mag (uT)')
    ax4.set_title('High-pass Mag (uT)')
    ax4.set_xlabel('Time (s)')
    
    lines1 = [ax1.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines2 = [ax2.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines3 = [ax3.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines4 = [ax4.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    
    ax1.set_ylim(-15000, 15000)
    ax2.set_ylim(-300, 300)
    ax3.set_ylim(-5000, 5000)
    ax4.set_ylim(-400, 400)
    
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    def init():
        for l in lines1 + lines2 + lines3 + lines4:
            l.set_data([], [])
        return lines1 + lines2 + lines3 + lines4

    def update(frame):
        t = np.array(plot_time)
        if t.size == 0:
            return lines1 + lines2 + lines3 + lines4
        t = t - t[0]
        
        data1 = [np.array(plot_mag_x), np.array(plot_mag_y), np.array(plot_mag_z)]
        data2 = [np.array(plot_amp_x), np.array(plot_amp_y), np.array(plot_amp_z)]
        data3 = [np.array(plot_lp_x), np.array(plot_lp_y), np.array(plot_lp_z)]
        data4 = [np.array(plot_hp_x), np.array(plot_hp_y), np.array(plot_hp_z)]
        
        for i in range(3):
            lines1[i].set_data(t, data1[i])
            lines2[i].set_data(t, data2[i])
            lines3[i].set_data(t, data3[i])
            lines4[i].set_data(t, data4[i])
        
        for ax in axs:
            ax.relim()
            ax.autoscale_view(scaley=False)
        
        return lines1 + lines2 + lines3 + lines4

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

def process_existing_files(directory, processor):
    """Process all existing files in the directory in order."""
    files = []
    for filename in os.listdir(directory):
        if filename.startswith('imgInfo_') and filename.endswith('.txt'):
            file_num = processor.extract_file_number(filename)
            if file_num > 0:
                files.append((file_num, filename))
    
    files.sort(key=lambda x: x[0])
    print(f"Found {len(files)} existing files to process")
    
    for file_num, filename in files:
        filepath = os.path.join(directory, filename)
        processor.process_file(filepath)

def cleanup():
    """Clean up resources before shutdown."""
    global tcp_server_running, observers
    print("\nCleaning up...")
    shutdown_event.set()
    
    for observer in observers:
        try:
            observer.stop()
            observer.join(timeout=1.0)
        except:
            pass
    
    tcp_server_running = False
    stop_logging()
    
    try:
        while not data_queue.empty():
            data_queue.get_nowait()
            data_queue.task_done()
    except:
        pass
    
    print("Cleanup complete")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nInterrupt received, shutting down...")
    cleanup()
    sys.exit(0)

def parse_args():
    """Parse command-line arguments."""
    show_plot = '--plot' in sys.argv
    return show_plot


def main():
    """Main function to run the application."""
    global observers
    
    show_plot = parse_args()

    system_signal.signal(system_signal.SIGINT, signal_handler)
    system_signal.signal(system_signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    processor = DataProcessor()
    
    print("Processing existing files...")
    process_existing_files(DATA_DIRECTORY, processor)
    processor.process_buffered_data()
    
    processing_thread_instance = threading.Thread(target=processing_thread, args=(processor,))
    processing_thread_instance.daemon = True
    processing_thread_instance.start()
    
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    tcp_thread = threading.Thread(target=tcp_server_thread)
    tcp_thread.daemon = True
    tcp_thread.start()
    
    event_handler = DataFileHandler(processor)
    observer = Observer()
    observers.append(observer)
    observer.schedule(event_handler, DATA_DIRECTORY, recursive=False)
    observer.start()
    
    print(f"Monitoring directory: {os.path.abspath(DATA_DIRECTORY)}")
    print("Waiting for new files...")
    print("Press Ctrl+C to stop")
    if show_plot:
        print("Plotting enabled via --plot")
    
    try:
        if show_plot:
            live_plot()
        else:
            while not shutdown_event.is_set():
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Plot error: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    observers = []
    main()