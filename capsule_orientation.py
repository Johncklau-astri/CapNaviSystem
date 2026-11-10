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
import argparse
import serial
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

shutdown_event = threading.Event()  # Add this line
observers = []  # Move this to global scope

# Constants
DATA_DIRECTORY = r"R:\Release_Internal_V1A\output"  # Directory to monitor for txt files
TIMESTAMP_FILE = "last_processed.txt"  # File to store last processed file number
LOG_FILENAME = "calibration_log.txt"
DELETE_PROCESSED_FILES = False
DEBUG_INGESTION = False
STREAM_RESTART_IDLE_SECONDS = 2.0
CALIBRATION_SERIAL_PORT = 'COM11'
CALIBRATION_SERIAL_BAUDRATE = 800000
CALIBRATION_SERIAL_TIMEOUT = 0.1
CALIBRATION_OUTPUT_FILE = 'capsule_clock_calibration.json'
CALIBRATION_FILE_WAIT_TIMEOUT = 10.0
CALIBRATION_SIDECAR_GRACE_TIMEOUT = 0.5
CAPSULE_CLOCK_OFFSET_SECONDS = 0.0

# Replace the identity matrix with the BORESIGHT_R_MC block printed by
# Sensor_calibration/magnetometer_calibration-main/mag_boresight_calib.py.
# It transforms ellipsoid-corrected magnetometer vectors into the mounting frame.
BORESIGHT_R_MC = np.eye(3)

# Replace this identity matrix with the ACCEL_BORESIGHT_R block printed by
# Sensor_calibration/accelerometer-calibration-main/accel_boresight_calib.py.
# It maps bias/scale-corrected accelerometer vectors into the capsule frame.
ACCEL_BORESIGHT_R = np.eye(3)

# The calibrated magnetometer frame has Y/Z opposite to the NWU robot frame.
# This is a proper 180-degree rotation about +X, not a single-axis reflection.
MAGNETOMETER_TO_NWU_R = np.diag([1.0, -1.0, -1.0])

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
calibration_requested = False
logging_enabled = False
log_file = None
logging_source = "magnetometer"
one_shot_logging_trigger = threading.Event()
logging_lock = threading.Lock()
q_prev = None  # Global reference tracking for gyro-predicted field direction

MAG_DISTURBANCE_RATIO_MIN = 0.01
MAG_DISTURBANCE_RATIO_MAX = 0.03


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


def compute_magnetometer_weight(raw_mag_uT, ac_mag_uT):
    """Estimate how much trust to place in the lock-in magnetic direction."""
    raw_magnitude = float(np.linalg.norm(np.asarray(raw_mag_uT, dtype=float)))
    ac_magnitude = float(np.linalg.norm(np.asarray(ac_mag_uT, dtype=float)))
    if raw_magnitude <= 1e-9 or ac_magnitude <= 1e-9:
        return 0.0

    ratio = ac_magnitude / raw_magnitude
    if ratio <= MAG_DISTURBANCE_RATIO_MIN:
        return 0.0
    if ratio >= MAG_DISTURBANCE_RATIO_MAX:
        return 1.0
    return (ratio - MAG_DISTURBANCE_RATIO_MIN) / (
        MAG_DISTURBANCE_RATIO_MAX - MAG_DISTURBANCE_RATIO_MIN
    )


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
        
        # World frame is NWU: +X north, +Y west, +Z up.
        self.gravity_ref = np.array([0.0, 0.0, -1.0])  # Gravity points down
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
        self.filtered_q = self.q.copy()
        self.last_time = None
        return self.q

    def update(self, gyro, accel, mag, current_time=None, mag_weight=1.0):
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

        accel_normalized = self.normalize_vector(np.asarray(accel, dtype=float))
        mag_array = None if mag is None else np.asarray(mag, dtype=float)
        use_mag = mag_array is not None and np.linalg.norm(mag_array) > 0 and mag_weight > 0.0
        mag_normalized = self.normalize_vector(mag_array) if use_mag else None
        
        # Extract quaternion elements
        q0, q1, q2, q3 = self.q
        
        accel_f = np.array([
            2.0 * (q0*q2 - q1*q3) - accel_normalized[0],
            -2.0 * (q0*q1 + q2*q3) - accel_normalized[1],
            -2.0 * (0.5 - q1*q1 - q2*q2) - accel_normalized[2],
        ])
        accel_J = np.array([
            [2.0*q2, -2.0*q3, 2.0*q0, -2.0*q1],
            [-2.0*q1, -2.0*q0, -2.0*q3, -2.0*q2],
            [0.0, 4.0*q1, 4.0*q2, 0.0],
        ])

        gradient = accel_J.T @ accel_f
        if use_mag:
            h = self.quaternion_multiply(
                self.quaternion_multiply(self.q, np.array([0.0, *mag_normalized])),
                self.quaternion_conjugate(self.q)
            )[1:]
            b = np.array([0.0, np.linalg.norm(h[:2]), 0.0, h[2]])
            mag_f = np.array([
                2.0 * b[1] * (0.5 - q2*q2 - q3*q3) + 2.0 * b[3] * (q1*q3 - q0*q2) - mag_normalized[0],
                2.0 * b[1] * (q1*q2 - q0*q3) + 2.0 * b[3] * (q0*q1 + q2*q3) - mag_normalized[1],
                2.0 * b[1] * (q0*q2 + q1*q3) + 2.0 * b[3] * (0.5 - q1*q1 - q2*q2) - mag_normalized[2]
            ])
            mag_J = np.array([
                [-2.0*b[3]*q2, 2.0*b[3]*q3, -4.0*b[1]*q2 - 2.0*b[3]*q0, -4.0*b[1]*q3 + 2.0*b[3]*q1],
                [-2.0*b[1]*q3 + 2.0*b[3]*q1, 2.0*b[1]*q2 + 2.0*b[3]*q0, 2.0*b[1]*q1 + 2.0*b[3]*q3, -2.0*b[1]*q0 + 2.0*b[3]*q2],
                [2.0*b[1]*q2, 2.0*b[1]*q3 - 4.0*b[3]*q1, 2.0*b[1]*q0 - 4.0*b[3]*q2, 2.0*b[1]*q1]
            ])
            gradient = gradient + float(mag_weight) * (mag_J.T @ mag_f)

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

        # Quaternion q and -q represent the same pose. Align their signs
        # before filtering so a rotation through 180 degrees is continuous.
        q_for_filter = self.q
        if np.dot(q_for_filter, self.filtered_q) < 0:
            q_for_filter = -q_for_filter
        self.filtered_q = self.normalize_quaternion(
            (1 - self.alpha) * self.filtered_q + self.alpha * q_for_filter
        )
        
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
        self.polarity = None  # Resolves the remaining +/- ambiguity of an AC field.
        self.last_relative_vector = None
        self.last_output_vector = None
        self.last_sign_resolution_time = None
        self.min_sign_amplitude_uT = 15.0
        self.max_unexplained_turn_deg = 120.0
        self.base_turn_allowance_deg = 10.0
        self.gyro_turn_allowance = 1.5
        self.max_sign_resolution_dt = 0.25

    def high_pass_filter(self, mag_uT):
            """Apply high-pass filter to remove DC offset from input signals."""
            _hpf_filtered = [0.0, 0.0, 0.0]
            
            for i in range(3):
                # 1. Update the running DC offset estimate
                self.hpf_dc_offsets[i] = (1 - self.hp_alpha) * self.hpf_dc_offsets[i] + self.hp_alpha * mag_uT[i]
                
                # 2. Subtract it from the raw value to isolate the AC wave
                _hpf_filtered[i] = mag_uT[i] - self.hpf_dc_offsets[i]
                
            return _hpf_filtered

    def calibrate_polarity(self, reference_field):
        """Latch the AC field's global sign at a pose with known field direction."""
        if self.last_relative_vector is None:
            print("Calibration skipped: no lock-in field estimate is available yet.")
            return False

        reference = np.asarray(reference_field, dtype=float)
        estimate = np.array(self.last_relative_vector, dtype=float, copy=True)
        reference_norm = np.linalg.norm(reference)
        estimate_norm = np.linalg.norm(estimate)
        if estimate_norm == 0 or reference_norm == 0:
            print("Calibration skipped: lock-in field estimate has zero magnitude.")
            return False

        reference /= reference_norm
        estimate /= estimate_norm

        alignment = float(np.dot(estimate, reference))
        self.polarity = 1.0 if alignment >= 0 else -1.0
        self.last_output_vector = estimate * self.polarity
        self.last_sign_resolution_time = None
        print(
            "AC polarity calibrated to {:+.0f} (reference alignment: {:+.3f})".format(
                self.polarity, alignment
            )
        )
        return True

    def resolve_polarity(self, signed_amplitudes, current_time, predicted_field=None, gyro_dps=None):
        """Resolve the equivalent global lock-in polarity branches."""
        signed_amplitudes = np.asarray(signed_amplitudes, dtype=float)
        magnitude = np.linalg.norm(signed_amplitudes)
        previous = self.last_output_vector
        previous_norm = 0.0 if previous is None else np.linalg.norm(previous)

        if magnitude > 0:
            candidates = (signed_amplitudes, -signed_amplitudes)
            sign_reference = predicted_field
            if sign_reference is None or np.linalg.norm(sign_reference) == 0:
                sign_reference = previous

            if sign_reference is not None and np.linalg.norm(sign_reference) > 0:
                signed_amplitudes = max(
                    candidates,
                    key=lambda candidate: float(np.dot(candidate, sign_reference)),
                )

            if (
                previous_norm > 0
                and magnitude >= self.min_sign_amplitude_uT
                and self.last_sign_resolution_time is not None
            ):
                cosine = float(np.dot(signed_amplitudes, previous) / (magnitude * previous_norm))
                turn_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                elapsed = min(
                    max(current_time - self.last_sign_resolution_time, 0.0),
                    self.max_sign_resolution_dt,
                )
                gyro_magnitude_dps = 0.0 if gyro_dps is None else np.linalg.norm(gyro_dps)
                allowed_turn_deg = self.base_turn_allowance_deg + (
                    self.gyro_turn_allowance * gyro_magnitude_dps * elapsed
                )
                if turn_deg > self.max_unexplained_turn_deg and turn_deg > allowed_turn_deg:
                    signed_amplitudes = max(
                        candidates,
                        key=lambda candidate: float(np.dot(candidate, previous)),
                    )

        self.last_output_vector = signed_amplitudes.copy()
        self.last_sign_resolution_time = current_time
        return signed_amplitudes

    def update(self, mag_uT, current_time, predicted_field=None, gyro_dps=None):
    # """Paper-based lock-in amplifier that uses phase differences only."""
        global gyro_predicted_field

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
                # Wrap the phase difference to [-pi, pi] before comparing it.
                phase_diff = abs(np.angle(np.exp(1j * (raw_phases[i] - raw_phases[ref_axis]))))
                # If phase difference > 90°, they're opposite
                if phase_diff > np.pi/2:
                    signs[i] = -1.0
                else:
                    signs[i] = 1.0
        
        # Create signed amplitude vector (still ambiguous: ±vector)
        _signed_amplitudes = np.array([amplitudes[i] * signs[i] for i in range(3)])
        self.last_relative_vector = _signed_amplitudes.copy()
        # print(f"Raw phases (degrees): {[np.degrees(p) for p in raw_phases]}, Signs: {signs}, Amplitudes: {amplitudes}")
        
        # Calibration establishes the initial sign, but the reference axis can
        # change from one sample to another. Resolve the remaining global +/-
        # ambiguity continuously against the predicted or prior body field.
        if self.polarity is not None:
            _signed_amplitudes *= self.polarity

        # The phase method produces two equivalent global sign branches. Choose
        # the branch closest to the expected body-frame +X field, then prevent
        # a discontinuous branch swap unless gyro motion can explain it.
        _signed_amplitudes = self.resolve_polarity(
            _signed_amplitudes,
            current_time,
            predicted_field=predicted_field,
            gyro_dps=gyro_dps,
        )
        
        return _signed_amplitudes, [np.degrees(p) for p in raw_phases], _filtered_mag, current_time

    def reset_dc_offsets(self):
        """Reset DC offset estimates (useful when conditions change dramatically)."""
        self.hpf_dc_offsets = [0.0, 0.0, 0.0]
        self.hpf_mag_uT = [0.0, 0.0, 0.0]
        self.last_output_vector = None
        self.last_sign_resolution_time = None
        print("DC offsets reset")


class DataFileHandler(FileSystemEventHandler):
    """Handler for new file creation events."""
    def __init__(self, processor):
        self.processor = processor
        self.processed_files = set()
        self.known_revisions = {}
        self.startup_files = set()
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
            self.startup_files.discard(event.src_path)
            if DEBUG_INGESTION:
                print(f"[INGEST] watcher created: {os.path.basename(event.src_path)}")
            self.handle_file_with_retry(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            if DEBUG_INGESTION:
                print(f"[INGEST] watcher modified: {os.path.basename(event.src_path)}")
            self.handle_file_with_retry(event.src_path)

    def prime_directory(self, directory):
        """Record files already handled during startup processing."""
        try:
            self.known_revisions = {
                os.path.join(directory, filename): self.file_revision(os.path.join(directory, filename))
                for filename in os.listdir(directory)
                if filename.startswith('imgInfo_') and filename.endswith('.txt')
            }
            self.startup_files = set(self.known_revisions)
            if DEBUG_INGESTION:
                print(f"[INGEST] primed {len(self.known_revisions)} existing telemetry files")
        except OSError as e:
            print(f"Error priming data directory: {e}")

    @staticmethod
    def file_revision(filepath):
        stat = os.stat(filepath)
        return stat.st_mtime_ns, stat.st_size

    def scan_directory(self, directory):
        """Process image-info files missed by filesystem events."""
        try:
            files = []
            for filename in os.listdir(directory):
                if filename.startswith('imgInfo_') and filename.endswith('.txt'):
                    filepath = os.path.join(directory, filename)
                    revision = self.file_revision(filepath)
                    if self.known_revisions.get(filepath) != revision:
                        file_num = self.processor.extract_file_number(filename)
                        files.append((file_num, filepath, revision))

            for _, filepath, revision in sorted(files):
                if DEBUG_INGESTION:
                    print(
                        f"[INGEST] reconciliation found: {os.path.basename(filepath)} "
                        f"(mtime_ns={revision[0]}, size={revision[1]})"
                    )
                self.handle_file_with_retry(filepath)
                self.known_revisions[filepath] = revision
        except OSError as e:
            print(f"Error scanning data directory: {e}")

    def handle_file_with_retry(self, filepath):
        """Handles race conditions when reading files that are being written."""
        max_retries = 10
        retry_delay = 0.02
        for attempt in range(max_retries):
            try:
                with open(filepath, 'r') as f:
                    pass
                if DEBUG_INGESTION:
                    print(f"[INGEST] reading: {os.path.basename(filepath)}")
                self.processor.process_file(filepath)
                self.known_revisions[filepath] = self.file_revision(filepath)
                return
            except PermissionError:
                if DEBUG_INGESTION:
                    print(f"[INGEST] file locked; retry {attempt + 1}/{max_retries}: {os.path.basename(filepath)}")
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
        self.last_file_mtime_ns = -1
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
    
    def calibrate_magnetometer(self, mag_raw):

        # Convert to numpy array for vectorized operations
        mag_raw = np.array(mag_raw)
        
        # 1. HARD IRON BIAS (Offset correction)
        hard_iron_bias = np.array([1100.025574, 13921.673267, 2861.224321])
        soft_iron_matrix = np.array([
            [1.003489, 0.001235, 0.010487],
            [0.001235, 0.981175, 0.018241],
            [0.010487, 0.018241, 0.966444]
        ])
        # --- Execute Calibration Pipeline ---
        
        # Step A: Apply ellipsoid correction (Raw -> Corrected Sensor Frame)
        mag_ellipsoid_corrected = (mag_raw - hard_iron_bias ) @ soft_iron_matrix.T
        
        # Step B: Apply Boresight Alignment (Corrected Sensor Frame -> Physical Board Frame)
        BORESIGHT_R_MC = np.eye(3)  # Identity matrix for demonstration; replace with actual boresight rotation if available
        # BORESIGHT_R_MC = np.array([
        #     [-0.9980132617, -0.0307965777, -0.0549645357],
        #     [-0.0331427909, 0.9985551949, 0.0422974965],
        #     [0.0535825045, 0.0440351405, -0.9975920116],
        # ])
        mag_final = MAGNETOMETER_TO_NWU_R @ BORESIGHT_R_MC @ mag_ellipsoid_corrected
        
        
        return mag_final.tolist()

    def calibrate_magnetometer_ellipsoidOnly(self, mag_raw):
        """
        Calibrate raw magnetometer data using hard iron bias and soft iron matrix only.
        
        Args: 
            mag_raw: Raw magnetometer readings [x, y, z] in microTesla

        Returns:
            Calibrated readings [x, y, z] in microTesla
        """
        mag_raw = np.array(mag_raw)
        hard_iron_bias = np.array([1100.025574, 13921.673267, 2861.224321])
        soft_iron_matrix = np.array([
            [1.003489, 0.001235, 0.010487],
            [0.001235, 0.981175, 0.018241],
            [0.010487, 0.018241, 0.966444]
        ])
        mag_ellipsoid_corrected = (mag_raw - hard_iron_bias) @ soft_iron_matrix.T
        return mag_ellipsoid_corrected.tolist()
    
    def calibrate_magnetometer_NWU(self, mag_raw):
        mag_raw = np.array(mag_raw)
        hard_iron_bias = np.array([1100.025574, 13921.673267, 2861.224321])
        soft_iron_matrix = np.array([
            [1.003489, 0.001235, 0.010487],
            [0.001235, 0.981175, 0.018241],
            [0.010487, 0.018241, 0.966444]
        ])
        mag_ellipsoid_corrected = (mag_raw - hard_iron_bias) @ soft_iron_matrix.T

        # BORESIGHT_R_MC = np.array([
        #     [0.9347739378, -0.1333265924, 0.3292745130],
        #     [0.1146922726, 0.9905296224, 0.0754768163],
        #     [-0.3362192258, -0.0327885186, 0.9412128055],
        # ])

        BORESIGHT_R_MC = np.array([
            [0.9203227096, 0.0233920774, -0.3904598838],
            [-0.0276143643, -0.9918343230, -0.1245075202],
            [-0.3901840040, 0.1253693999, -0.9121616943],
        ])
        ## the nwu matrix flips signs of y and z values
        ## the flipping made yaw to the opposite direction , and using original rotation is enough 3,nov2026
        mag_final = MAGNETOMETER_TO_NWU_R @ BORESIGHT_R_MC @ mag_ellipsoid_corrected
        # mag_final =  BORESIGHT_R_MC @ mag_ellipsoid_corrected


        return mag_final.tolist()
    
    def calibrate_accelerometer(self, accel_raw):
        """Apply ellipsoid calibration followed by the fixed boresight rotation."""
        # Bias parameters (offset correction)

        bias_x = -0.000987171
        bias_y = 0.008617489
        bias_z = -0.015150045

        cal_matrix_xx = 0.978955736; cal_matrix_xy = -0.016683116; cal_matrix_xz = -0.012363124
        cal_matrix_yx = -0.016683116; cal_matrix_yy = 1.012684068; cal_matrix_yz = 0.012009726
        cal_matrix_zx = -0.012363124; cal_matrix_zy = 0.012009726; cal_matrix_zz = 0.990183901
        ACCEL_BORESIGHT_R = np.array([
            [ 0.918319961, -0.007293519, -0.395771720],
            [ 0.064792239,  0.989115057,  0.132111200],
            [ 0.390500211, -0.146963288,  0.908796664],
        ])


        # ACCEL_BORESIGHT_R =np.eye(3)  # Identity matrix for demonstration; replace with actual boresight rotation if available

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

        return ACCEL_BORESIGHT_R @ np.asarray(accel_calibrated)
    
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

        # print(f"Raw mag: {mag_uT[0]:.3f} uT, {mag_uT[1]:.3f} uT, {mag_uT[2]:.3f} uT")
        # print(f"Raw accel: {accel_g[0]:.3f} g, {accel_g[1]:.3f} g, {accel_g[2]:.3f} g")




        
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
            file_mtime_ns = os.stat(filepath).st_mtime_ns
            if DEBUG_INGESTION:
                print(
                    f"[INGEST] processing: {os.path.basename(filepath)} "
                    f"(file_num={file_num}, size={os.path.getsize(filepath)}, "
                    f"last_file_num={self.last_file_num}, last_timestamp={self.last_packet_timestamp:.3f})"
                )
            idle_seconds = (file_mtime_ns - self.last_file_mtime_ns) / 1_000_000_000
            if (
                (file_num < self.last_file_num or idle_seconds >= STREAM_RESTART_IDLE_SECONDS)
                and file_mtime_ns > self.last_file_mtime_ns
            ):
                print(
                    "[INGEST] detected restarted telemetry stream; resetting packet timestamp "
                    f"tracking (idle={idle_seconds:.3f}s)"
                )
                self.last_packet_timestamp = -1.0
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            hex_string = ''.join(content.split())
            data = bytes.fromhex(hex_string)

            data = data[6:131] + data[131+8:244]
            packet_size = 23
            num_packets = len(data) // packet_size
            if num_packets == 0:
                if DEBUG_INGESTION:
                    print(f"[INGEST] waiting for complete telemetry data: {os.path.basename(filepath)}")
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
                if DEBUG_INGESTION:
                    print(
                        f"[INGEST] skipped {os.path.basename(filepath)}: all {num_packets} packets "
                        f"were at or before timestamp {self.last_packet_timestamp:.3f}"
                    )
                self.last_file_num = file_num
                self.last_file_mtime_ns = file_mtime_ns
                self.update_file_extraction_rate()
                handler = next((h for h in observers if isinstance(h, DataFileHandler)), None)
                if handler:
                    handler.save_last_processed(file_num)
                return

            parsed_packets.sort(key=lambda item: item[0])
            for timestamp, accel_g, gyro_dps, mag_uT in parsed_packets:
                self.last_packet_timestamp = timestamp
                self.update_sensor_input_rate(timestamp)

                ## mag_calibrated makes magwick gets the z rotation opposite
                # mag_calibrated = self.calibrate_magnetometer(mag_uT)
                accel_calibrated = self.calibrate_accelerometer(accel_g)
                #use ellipsoid only for boresight calib
                mag_calibrated_ellipsoid = self.calibrate_magnetometer_ellipsoidOnly(mag_uT)
                # print(f"Mag calibrated (ellipsoid only): {mag_calibrated_ellipsoid}")
                ## mag_calibrated_noNWU makes correct direction in 3 axis in magwick
                mag_calibrated_NWU = self.calibrate_magnetometer_NWU(mag_uT)
                # print(f"Mag calibrated (NWU): {mag_calibrated_NWU}, Mag calibrated (ellipsoid only): {mag_calibrated_ellipsoid}")


                sample_to_log = mag_calibrated_ellipsoid if logging_source == "magnetometer" else accel_g


                ## For "l" one shot logging
                if one_shot_logging_trigger.is_set():
                    log_sensor_sample(sample_to_log, one_shot=True)
                    one_shot_logging_trigger.clear()
                    
                ## For "m" continuous logging
                log_sensor_sample(sample_to_log)

                # print(f"Raw mag: {mag_uT}, Calibrated mag: {mag_calibrated}")



                data_queue.put((timestamp, accel_calibrated, gyro_dps, mag_calibrated_NWU, 0))

                self.data_buffer.append((timestamp, accel_calibrated, gyro_dps, mag_calibrated_NWU))
            
            self.update_file_extraction_rate()
            self.last_file_num = file_num
            self.last_file_mtime_ns = file_mtime_ns
            if DEBUG_INGESTION:
                print(
                    f"[INGEST] queued {len(parsed_packets)}/{num_packets} packets from "
                    f"{os.path.basename(filepath)}; latest timestamp={self.last_packet_timestamp:.3f}; "
                    f"queue size={data_queue.qsize()}"
                )
            handler = next((h for h in observers if isinstance(h, DataFileHandler)), None)
            if handler:
                handler.save_last_processed(file_num)

            if DELETE_PROCESSED_FILES:
                try:
                    os.remove(filepath)
                    if DEBUG_INGESTION:
                        print(f"[INGEST] deleted processed file: {os.path.basename(filepath)}")
                except FileNotFoundError:
                    pass
                except OSError as e:
                    print(f"Error deleting processed file {filepath}: {e}")
                
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    def process_buffered_data(self):
        """Process all data in the buffer (for initial load)."""
        self.data_buffer.sort(key=lambda x: x[0])
        for timestamp, accel_g, gyro_dps, mag_uT in self.data_buffer:
            data_queue.put((timestamp, accel_g, gyro_dps, mag_uT, 0))

def processing_thread(processor):
    """Background thread for data processing."""
    global q_prev, calibration_requested, gyro_predicted_field
    filter = processor.filter
    q = processor.q
    lockin = processor.lockin
    lowpass_mag = None
    ac_lowpass_mag = None
    mod_q_lowpass = None
    alpha_amp = 0.4
    alpha_q = 0.4
    
    # Update rate counters
    packet_count = 0
    start_time = time.time()
    update_rate = 0.0
    dc_window = deque()
    dc_window_duration = 1.0 / lockin.target_freq
    
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

                # Calculate expected AC field orientation relative to the sensor body frame
                if q_prev is not None:
                    ac_field_global = np.array([1.0, 0.0, 0.0])  # North is world +X in NWU
                    R = quaternion_to_rotation_matrix(q_prev)
                    gyro_predicted_field = R.T @ ac_field_global
                else:
                    gyro_predicted_field = None

                # Update lock-in amplifier utilizing the same gyro-predicted sign resolution logic as bt_angleEst.py
                mag_amplitudes, mag_phases, highpass_mag, _ = lockin.update(
                    mag_uT,
                    current_time,
                    predicted_field=gyro_predicted_field,
                    gyro_dps=gyro_dps,
                )

                if calibration_requested:
                    # 'C' is pressed with the robot at the negative-yaw reference pose.
                    if lockin.calibrate_polarity(np.array([1.0, 1.0, 0.0])):
                        mag_amplitudes = lockin.last_relative_vector * lockin.polarity
                        q = filter.init()
                        q_prev = q.copy()
                        mod_q_lowpass = q.copy()
                    calibration_requested = False
                
                # Update Madgwick filter with calculated signed amplitudes
                mag_weight = compute_magnetometer_weight(mag_uT, mag_amplitudes)

                q_filter, error, omega_corrected = filter.update(
                    np.radians(gyro_dps),
                    np.array(accel_g),
                    np.array(mag_amplitudes),
                    current_time=current_time,
                    mag_weight=mag_weight,
                )
                q = q_filter
                if mod_q_lowpass is not None and np.dot(q, mod_q_lowpass) < 0:
                    q = -q
                q_prev = q.copy()  # Cache target frame for next iteration loop prediction
                
                # Apply low-pass filters
                mag_array = np.array(mag_uT)
                dc_window.append((current_time, mag_array.copy()))
                window_start = current_time - dc_window_duration
                while len(dc_window) > 1 and dc_window[0][0] < window_start:
                    dc_window.popleft()
                lowpass_mag = np.mean(
                    np.stack([sample for _, sample in dc_window]),
                    axis=0,
                )
                
                # # Rotate quaternion

                mod_q = q
                
                # Apply quaternion low-pass
                if mod_q_lowpass is None:
                    mod_q_lowpass = mod_q.copy()
                mod_q_lowpass = filter.normalize_quaternion(
                    alpha_q * mod_q + (1 - alpha_q) * mod_q_lowpass
                )
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
                ## Yaw Z, Pitch Y, Roll X
                # tmp_lp_mag_x = lowpass_mag[0]
                # tmp_lp_mag_y = -lowpass_mag[1]
                # tmp_lp_magz = -lowpass_mag[2]

                # tmp_lp = np.array([tmp_lp_mag_x, tmp_lp_mag_y, tmp_lp_magz])
                print(
                    f"Yaw: {yaw:.3f}°\tPitch: {pitch:.3f}°\tRoll: {roll:.3f}°\tq: {mod_q[0]:.3f}, {mod_q[1]:.3f}, {mod_q[2]:.3f}, {mod_q[3]:.3f}\t  LockIn: {mag_amplitudes[0]:.3f} uT, {mag_amplitudes[1]:.3f} uT, {mag_amplitudes[2]:.3f} uT\tlow pass mag: {lowpass_mag[0]:.3f} uT, {lowpass_mag[1]:.3f} uT, {lowpass_mag[2]:.3f} uT"
                # #     # f"Raw mag: {mag_array[0]:.3f} uT, {mag_array[1]:.3f} uT, {mag_array[2]:.3f} uT lowpass_mag: {lowpass_mag[0]:.3f} uT, {lowpass_mag[1]:.3f} uT, {lowpass_mag[2]:.3f} uT"
                # #     # f"Mag: {mag_magnitude:.1f} µT\tAC: {ac_magnitude:.1f} µT\t"
                # #     # f"Calib Mag  mX: {mag_uT[0]:.3f} uT, mY: {mag_uT[1]:.3f} uT, mZ: {mag_uT[2]:.3f} uT, lowpass_mag: {lowpass_mag[0]:.3f} uT, {tmp_lp_mag_y:.3f} uT, {tmp_lp_magz:.3f} uT"
                # #     # f"qw: {mod_q[0]:.3f}\tqx: {mod_q[1]:.3f}\tqy: {mod_q[2]:.3f}\tqz: {mod_q[3]:.3f}\tYaw: {yaw:.3f}°\t Pitch: {pitch:.3f}°\tRoll: {roll:.3f}°"
                # #     # f"Sensor: {processor.input_rate_hz:.2f} Hz\t"
                # #     # f"File: {processor.file_extraction_rate_hz:.2f} files/s\t"
                # #     # f"files #: {processor.files_completed}\t"
                # #     # f"Processing: {update_rate:.2f} Hz\t"
                # #     # f"D.P. Timestamp: {current_time:.3f}s"
                # #     # f"q: {mod_q[0]:.3f}, {mod_q[1]:.3f}, {mod_q[2]:.3f}, {mod_q[3]:.3f}\t"
                # #     # f"Magx: {mag_uT[0]:.3f} uT, Magy: {mag_uT[1]:.3f} uT, Magz: {mag_uT[2]:.3f} uT"
                # #     # f"Gyrox: {gyro_dps[0]:.3f} dps\tGyroy: {gyro_dps[1]:.3f} dps\tGyroz: {gyro_dps[2]:.3f} dps"
                # f"accel: {accel_g[0]:.3f} g, {accel_g[1]:.3f} g, {accel_g[2]:.3f} g\t lp_mag: {lowpass_mag[0]:.3f} uT, {lowpass_mag[1]:.3f} uT, {lowpass_mag[2]:.3f} uT"

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


                # Apply the output-frame sign convention without changing the
                # filter state used by the next sample.
                mag_to_send = lowpass_mag.copy()
                # temp = mag_to_send[0]
                # mag_to_send[0] = mag_to_send[2]
                # mag_to_send[2] = temp
                # mag_to_send[1] = -mag_to_send[1]

                # Send to TCP clients
                send_to_tcp_clients(
                    mod_q,
                    mag_to_send,
                    ac_lowpass_mag,
                    timestamp_capsule=current_time,
                )
                data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                if not shutdown_event.is_set():
                    print(f"Processing error: {e}")               
                time.sleep(0.1)

def send_to_tcp_clients(q, mag_uT, ac_mag_uT, timestamp_capsule=None):
    """Send quaternion and magnetometer data to connected TCP clients."""
    global tcp_clients
    if not tcp_clients or not tcp_server_running:
        return
    
    if timestamp_capsule is None:
        timestamp_capsule = time.time()
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
        'timestamp_capsule': float(timestamp_capsule),
        'timestamp_computer': float(timestamp_capsule - CAPSULE_CLOCK_OFFSET_SECONDS),
        'timestamp_sent': time.time(),
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
    global calibration_requested
    print("Press 'c' at the identity reference pose to calibrate AC field polarity")
    print("Press 'r' to reset DC offsets")
    print("Press 's' to switch logging source (magnetometer or IMU accelerometer)")
    print("Press 'l' to log one sample from the selected source")
    print("Press 'm' to start/stop continuous logging for the selected source")
    
    while not shutdown_event.is_set():
        try:
            if keyboard.is_pressed('c'):
                print("'c' pressed: AC polarity calibration requested.")
                calibration_requested = True
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
                
            # if keyboard.is_pressed('q'):
            #     print("Quitting...")
            #     shutdown_event.set()
            #     break
                
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

def reconcile_directory(event_handler, directory, interval=0.5):
    """Periodically recover files when a filesystem event is missed."""
    while not shutdown_event.wait(interval):
        event_handler.scan_directory(directory)


def _read_telemetry_samples(filepath, processor):
    """Read packet timestamps from one telemetry file using the normal parser."""
    with open(filepath, 'r') as input_file:
        hex_string = ''.join(input_file.read().split())
    data = bytes.fromhex(hex_string)
    data = data[6:131] + data[131 + 8:244]

    samples = []
    for offset in range(0, len(data) - 22, 23):
        result = processor.parse_data_packet(data[offset:offset + 23], 0)
        if result is not None:
            samples.append(result[0])
    return samples


def _telemetry_file_snapshot(processor):
    """Return existing telemetry names and the greatest packet timestamp."""
    files = [
        path for path in Path(DATA_DIRECTORY).glob('imgInfo_*.txt')
        if path.is_file()
    ]
    file_names = {path.name for path in files}
    latest_timestamp = -1.0
    for path in files:
        try:
            samples = _read_telemetry_samples(path, processor)
        except (OSError, ValueError):
            continue
        if samples:
            latest_timestamp = max(latest_timestamp, max(samples))
    return file_names, latest_timestamp


def _find_newest_sensor_timestamp(processor, previous_names, previous_timestamp, timeout,
                                  expected_new_files=2):
    """Find the final fresh packet in the newest file created during a trial."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new_files = [
            path for path in Path(DATA_DIRECTORY).glob('imgInfo_*.txt')
            if path.is_file() and path.name not in previous_names
        ]
        new_files.sort(key=lambda path: processor.extract_file_number(path.name))
        if len(new_files) >= expected_new_files:
            newest_file = new_files[-1]
            try:
                samples = [
                    timestamp for timestamp in _read_telemetry_samples(newest_file, processor)
                    if timestamp > previous_timestamp
                ]
            except (OSError, ValueError):
                samples = []
            if samples:
                return newest_file, samples[-1]
        time.sleep(0.02)
    return None, None


def _read_serial_line(connection):
    """Read one UART line while preserving extra lines received in one read."""
    if not hasattr(connection, '_calibration_line_buffer'):
        connection._calibration_line_buffer = b''

    while True:
        line_endings = [
            index for index in (
                connection._calibration_line_buffer.find(b'\r'),
                connection._calibration_line_buffer.find(b'\n'),
            ) if index >= 0
        ]
        if line_endings:
            line_end = min(line_endings)
            raw_line = connection._calibration_line_buffer[:line_end]
            remainder = connection._calibration_line_buffer[line_end + 1:]
            if remainder.startswith(b'\n') and connection._calibration_line_buffer[line_end:line_end + 2] == b'\r\n':
                remainder = remainder[1:]
            connection._calibration_line_buffer = remainder
            return time.time(), raw_line.decode('utf-8', errors='replace').strip()

        chunk = connection.read(4096)
        if not chunk:
            return None, None
        connection._calibration_line_buffer += chunk


def _wait_for_serial_text(connection, text, timeout):
    """Wait for text in serial output and return its computer receive time."""
    print(f"[calibration] waiting for serial text {text!r} (timeout={timeout:.1f}s)", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        received_time, line = _read_serial_line(connection)
        if line is not None:
            print(f"[calibration] serial received: {line!r}", flush=True)
        if line is not None and text in line:
            print(f"[calibration] found serial text {text!r}", flush=True)
            return received_time
    raise TimeoutError(f"Timed out waiting for serial output: {text}")


def run_time_calibration(trials=20, output_file=CALIBRATION_OUTPUT_FILE,
                         serial_port=CALIBRATION_SERIAL_PORT,
                         baudrate=CALIBRATION_SERIAL_BAUDRATE,
                         timeout=CALIBRATION_FILE_WAIT_TIMEOUT,
                         warmup_trials=1, max_missing_sidecar_retries=8):
    """Estimate capsule-to-computer clock offset using the second image.

    A capture without its telemetry sidecar is discarded and retried.
    """
    processor = DataProcessor()
    print(
        f"[calibration] opening serial port {serial_port} at {baudrate} baud...",
        flush=True,
    )
    connection = serial.Serial(
        serial_port,
        baudrate=baudrate,
        timeout=CALIBRATION_SERIAL_TIMEOUT,
        write_timeout=CALIBRATION_SERIAL_TIMEOUT,
    )
    print(
        f"[calibration] opened serial port {serial_port} at {baudrate} baud",
        flush=True,
    )
    offsets = []
    trial_records = []
    discarded_capture_count = 0

    try:
        connection.reset_input_buffer()
        print("[calibration] sending: at+photo=p,1,2\\r", flush=True)
        connection.write(b'at+photo=p,1,2\r')
        _wait_for_serial_text(connection, 'Total image number', timeout)

        total_sequences = warmup_trials + trials
        for sequence_number in range(1, total_sequences + 1):
            missing_sidecar_retries = 0
            while True:
                previous_names, previous_timestamp = _telemetry_file_snapshot(processor)
                command_time = time.time()
                print(
                    f"[calibration] trial {sequence_number}: sending: at+photo+\\r",
                    flush=True,
                )
                connection.write(b'at+photo+\r')

                image_start_count = 0
                second_image_start_time = None
                image_start_deadline = time.monotonic() + timeout
                print(
                    f"[calibration] trial {sequence_number}: waiting for "
                    f"the second +++Image-Start marker (timeout={timeout:.1f}s)",
                    flush=True,
                )
                while second_image_start_time is None:
                    if time.monotonic() >= image_start_deadline:
                        raise TimeoutError(
                            f"Timed out waiting for the second image-start marker "
                            f"during calibration trial {sequence_number}"
                        )
                    received_time, line = _read_serial_line(connection)
                    if line is None:
                        continue
                    print(f"[calibration] serial received: {line!r}", flush=True)
                    if '+++Image-Start' in line:
                        image_start_count += 1
                        if image_start_count == 2:
                            second_image_start_time = received_time

                second_image_end_time = _wait_for_serial_text(connection, 'ImgCt2', timeout)
                telemetry_file, capsule_timestamp = _find_newest_sensor_timestamp(
                    processor, previous_names, previous_timestamp,
                    min(timeout, CALIBRATION_SIDECAR_GRACE_TIMEOUT),
                    expected_new_files=2,
                )
                if telemetry_file is not None:
                    break

                missing_sidecar_retries += 1
                discarded_capture_count += 1
                if missing_sidecar_retries > max_missing_sidecar_retries:
                    raise TimeoutError(
                        f"No fresh telemetry packet found after {missing_sidecar_retries} "
                        f"recaptures for calibration trial {sequence_number}"
                    )
                print(
                    f"Calibration capture missing telemetry sidecar; recapturing "
                    f"trial {sequence_number} ({missing_sidecar_retries}/"
                    f"{max_missing_sidecar_retries})"
                )

            offset = capsule_timestamp - second_image_start_time
            record = {
                'trial': sequence_number,
                'discarded_capture_retries': missing_sidecar_retries,
                'command_sent_computer_time': command_time,
                'second_image_start_computer_time': second_image_start_time,
                'second_image_end_computer_time': second_image_end_time,
                'telemetry_file': str(telemetry_file),
                'previous_sensor_timestamp': previous_timestamp,
                'capsule_timestamp': capsule_timestamp,
                'offset_seconds': offset,
                'warmup': sequence_number <= warmup_trials,
            }
            trial_records.append(record)
            if sequence_number <= warmup_trials:
                print(
                    f"Calibration warm-up {sequence_number}/{warmup_trials}: "
                    f"offset={offset:.6f}s, file={telemetry_file.name} (excluded)"
                )
            else:
                offsets.append(offset)
                print(
                    f"Calibration trial {sequence_number - warmup_trials}/{trials}: "
                    f"offset={offset:.6f}s, file={telemetry_file.name}"
                )

        result = {
            'clock_model': 'constant_offset',
            'offset_seconds': float(np.median(offsets)),
            'offset_std_seconds': float(np.std(offsets)),
            'trial_count': len(offsets),
            'warmup_trials': warmup_trials,
            'serial_port': serial_port,
            'baudrate': baudrate,
            'ignored_first_image': True,
            'discarded_capture_count': discarded_capture_count,
            'trials': trial_records,
        }
        with open(output_file, 'w', encoding='utf-8') as calibration_file:
            json.dump(result, calibration_file, indent=2)
        print(f"Saved capsule clock calibration to {output_file}")
        print(
            f"Median offset: {result['offset_seconds']:.6f}s "
            f"(std={result['offset_std_seconds']:.6f}s)"
        )
        return result
    finally:
        connection.close()


def load_capsule_clock_offset(calibration_file=CALIBRATION_OUTPUT_FILE):
    """Load the calibrated capsule-minus-computer clock offset."""
    try:
        with open(calibration_file, 'r', encoding='utf-8') as input_file:
            calibration = json.load(input_file)
        offset = float(calibration['offset_seconds'])
        print(f"Loaded capsule clock offset: {offset:.6f}s from {calibration_file}")
        return offset
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Capsule clock calibration unavailable ({calibration_file}): {exc}")
        print("Using capsule timestamps without clock conversion.")
        return 0.0

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
    parser = argparse.ArgumentParser(description='Capsule orientation processor')
    parser.add_argument('--plot', action='store_true', help='Show live telemetry plots')
    parser.add_argument('--calibration-trials', type=int, default=20)
    parser.add_argument('--calibration-warmup-trials', type=int, default=1)
    parser.add_argument('--calibration-output', default=CALIBRATION_OUTPUT_FILE)
    parser.add_argument(
        '--skip-time-calibration',
        action='store_true',
        help='Start telemetry extraction without calibrating the capsule clock',
    )
    parser.add_argument(
        '--debug-ingestion',
        action='store_true',
        help='Print telemetry file discovery, parsing, and queue diagnostics',
    )
    return parser.parse_args()


def main():
    """Main function to run the application."""
    global observers, CAPSULE_CLOCK_OFFSET_SECONDS, DEBUG_INGESTION
    
    args = parse_args()
    DEBUG_INGESTION = args.debug_ingestion
    show_plot = args.plot

    system_signal.signal(system_signal.SIGINT, signal_handler)
    system_signal.signal(system_signal.SIGTERM, signal_handler)
    atexit.register(cleanup)

    if args.skip_time_calibration:
        print("Skipping capsule clock calibration; using a 0.000000s clock offset.")
    else:
        print("Capsule clock calibration is required before normal telemetry mode.")
        calibration_result = run_time_calibration(
            trials=args.calibration_trials,
            warmup_trials=args.calibration_warmup_trials,
            output_file=args.calibration_output,
        )
        CAPSULE_CLOCK_OFFSET_SECONDS = float(calibration_result['offset_seconds'])
        print(
            f"Time calibration complete for this power session; "
            f"starting normal telemetry extraction with offset "
            f"{CAPSULE_CLOCK_OFFSET_SECONDS:.6f}s."
        )
    
    processor = DataProcessor()

    # Snapshot existing telemetry so only files arriving after startup are processed.
    event_handler = DataFileHandler(processor)
    event_handler.prime_directory(DATA_DIRECTORY)
    
    processing_thread_instance = threading.Thread(target=processing_thread, args=(processor,))
    processing_thread_instance.daemon = True
    processing_thread_instance.start()
    
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    tcp_thread = threading.Thread(target=tcp_server_thread)
    tcp_thread.daemon = True
    tcp_thread.start()
    
    observer = Observer()
    observers.append(observer)
    observer.schedule(event_handler, DATA_DIRECTORY, recursive=False)
    observer.start()

    reconciliation_thread = threading.Thread(
        target=reconcile_directory,
        args=(event_handler, DATA_DIRECTORY),
        daemon=True,
    )
    reconciliation_thread.start()
    
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