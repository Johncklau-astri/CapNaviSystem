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
DATA_DIRECTORY = r"C:\Users\johncklau\Desktop\FX3_PCApp\FX3DisplayImage\Release_Internal_V1A\output"  # Directory to monitor for txt files
TIMESTAMP_FILE = "last_processed.txt"  # File to store last processed file number
LOG_FILENAME = "orientation_log.txt"

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
calibration_trigger = threading.Event()
logging_enabled = False
log_file = None

class MadgwickFilter:
    def __init__(self, beta=0.5):
        """Initialize Madgwick filter parameters."""
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        self.gravity_ref = np.array([0.0, 0.0, 1.0])
        self.magnetic_ref = np.array([0.0, -1.0, 0.0])

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
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        return q / norm if norm > 0 else q

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def init(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        return self.q

    def update(self, gyro, accel, mag, current_time=None):
        """Update Madgwick filter with gyro, accelerometer and magnetometer data."""
        if current_time is None:
            current_time = time.time()
            
        if self.last_time is None:
            dt = 0.01
        else:
            dt = current_time - self.last_time
            
        self.last_time = current_time
        
        accel_normalized = self.normalize_vector(accel)
        mag_normalized = self.normalize_vector(mag)
        
        q0, q1, q2, q3 = self.q
        
        h = self.quaternion_multiply(
            self.quaternion_multiply(self.q, np.array([0.0, *mag_normalized])),
            self.quaternion_conjugate(self.q)
        )[1:]
        
        b = np.array([0.0, np.linalg.norm([self.magnetic_ref[0], self.magnetic_ref[1]]), 
                        0.0, self.magnetic_ref[2]])
        
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
        
        gradient = J.T @ f
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / np.linalg.norm(gradient)
        
        q_dot_gyro = 0.5 * self.quaternion_multiply(self.q, np.array([0.0, *gyro]))
        q_dot_gradient = self.beta * gradient
        
        corrected_gyro = gyro - 2.0 * self.beta * gradient[1:]
        
        q_dot = q_dot_gyro - q_dot_gradient
        self.q = self.q + q_dot * dt
        self.q = self.normalize_quaternion(self.q)
        
        error = gradient[1:] * self.beta
        return self.q, error, corrected_gyro

class LockInAmplifier:
    def __init__(self, target_freq=10.0, alpha=0.03, hp_alpha=0.01):
        self.target_freq = target_freq
        self.alpha = alpha
        self.hp_alpha = hp_alpha
        self.x_in = [0.0, 0.0, 0.0]
        self.x_quad = [0.0, 0.0, 0.0]
        self.dc_offsets = [0.0, 0.0, 0.0]
        self.filtered_signals = [0.0, 0.0, 0.0]
        self.last_time = None
        self.phase_offsets = [0.0, 0.0, 0.0]
        self.is_calibrated = False
        self.first_sample_time = None

    def high_pass_filter(self, mag_uT):
        """Apply high-pass filter to remove DC offset."""
        filtered = []
        for i in range(3):
            self.dc_offsets[i] = (1 - self.hp_alpha) * self.dc_offsets[i] + self.hp_alpha * mag_uT[i]
            filtered_signal = mag_uT[i] - self.dc_offsets[i]
            self.filtered_signals[i] = filtered_signal
            filtered.append(filtered_signal)
        return filtered

    def calibrate(self, mag_uT, current_time):
        """Calibrate phase angles."""
        if self.last_time is None:
            self.last_time = current_time
            self.first_sample_time = current_time
            return
            
        filtered_mag = self.high_pass_filter(mag_uT)
        dt = current_time - self.last_time
        self.last_time = current_time
        t = current_time - self.first_sample_time
        
        ref_in = np.sin(2 * np.pi * self.target_freq * t)
        ref_quad = np.cos(2 * np.pi * self.target_freq * t)

        for i in range(3):
            prod_in = filtered_mag[i] * ref_in
            prod_quad = filtered_mag[i] * ref_quad
            self.x_in[i] = (1 - self.alpha) * self.x_in[i] + self.alpha * prod_in
            self.x_quad[i] = (1 - self.alpha) * self.x_quad[i] + self.alpha * prod_quad
            current_phase = np.arctan2(self.x_quad[i], self.x_in[i])

            if current_phase >= 0:
                self.phase_offsets[i] = 1

        self.is_calibrated = True
        print(f"Calibration complete. Phase offsets: {[np.degrees(off) for off in self.phase_offsets]}")
        print(f"DC offsets (uT): {self.dc_offsets}")
        

    def update(self, mag_uT, current_time):
        """Update lock-in amplifier with new magnetometer data."""

        if self.last_time is None:
            self.last_time = current_time
            self.first_sample_time = current_time
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], mag_uT

        filtered_mag = self.high_pass_filter(mag_uT)
        
        dt = current_time - self.last_time
        self.last_time = current_time
        t = current_time 
        
        ref_in = np.sin(2 * np.pi * self.target_freq * t)
        ref_quad = np.cos(2 * np.pi * self.target_freq * t)

        amplitudes = []
        phase_angles = []
        
        for i in range(3):
            prod_in = filtered_mag[i] * ref_in
            prod_quad = filtered_mag[i] * ref_quad
            self.x_in[i] = (1 - self.alpha) * self.x_in[i] + self.alpha * prod_in
            self.x_quad[i] = (1 - self.alpha) * self.x_quad[i] + self.alpha * prod_quad
            amplitude = np.sqrt(self.x_in[i]**2 + self.x_quad[i]**2)
            phase = np.arctan2(self.x_quad[i], self.x_in[i])
            
            if self.is_calibrated and self.phase_offsets[i] == 1:
                phase = -phase
            
            signed_amplitude = amplitude if phase > 0 else -amplitude

            amplitudes.append(signed_amplitude)
            phase_angles.append(np.degrees(phase))
            
        return amplitudes, phase_angles, filtered_mag

    def reset_dc_offsets(self):
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
            self.processor.process_file(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            self.processor.process_file(event.src_path)

class DataProcessor:
    """Process data from image info files."""
    
    # Magnetometer calibration parameters
    HARD_IRON_BIAS = [-24.425272, -9.840492, 16.169081]
    
    SOFT_IRON_MATRIX = np.array([
        [19.096303, -0.080634, -0.013220],
        [-0.080634, 19.686204, 0.048664],
        [-0.013220, 0.048664, 19.386885]
    ])
    
    # Accelerometer calibration parameters
    ACCEL_BIAS = [-0.016082, -0.018219, 0.152182]
    ACCEL_MATRIX = np.array([
        [1.009967, 0.007280, -0.029018],
        [0.007280, 1.020470, 0.057813],
        [-0.029018, 0.057813, 1.183160]
    ])
    
    def __init__(self):
        self.filter = MadgwickFilter()
        self.q = self.filter.init()
        self.lockin = LockInAmplifier(target_freq=10.0, alpha=0.03, hp_alpha=0.01)
        self.lowpass_mag = None
        self.ac_lowpass_mag = None
        self.mod_q_lowpass = None
        self.alpha = 0.08
        self.alpha_amp = 0.002
        self.alpha_q = 0.01
        self.last_file_num = -1
        self.data_buffer = []
        
    def extract_file_number(self, filename):
        """Extract the sequence number from filename (imgInfo_XXXXX_...)."""
        match = re.search(r'imgInfo_(\d+)_', filename)
        return int(match.group(1)) if match else -1
    
    def calibrate_magnetometer(self, mag_raw):
        """Apply magnetometer calibration."""
        mag_corrected = [
            mag_raw[0] - self.HARD_IRON_BIAS[0],
            mag_raw[1] - self.HARD_IRON_BIAS[1],
            mag_raw[2] - self.HARD_IRON_BIAS[2]
        ]
        return mag_corrected
    
    def calibrate_accelerometer(self, accel_raw):
        """Apply accelerometer calibration."""
        accel_biased = [
            accel_raw[0] - self.ACCEL_BIAS[0],
            accel_raw[1] - self.ACCEL_BIAS[1],
            accel_raw[2] - self.ACCEL_BIAS[2]
        ]
        
        accel_calibrated = [
            self.ACCEL_MATRIX[0][0] * accel_biased[0] + 
            self.ACCEL_MATRIX[0][1] * accel_biased[1] + 
            self.ACCEL_MATRIX[0][2] * accel_biased[2],
            self.ACCEL_MATRIX[1][0] * accel_biased[0] + 
            self.ACCEL_MATRIX[1][1] * accel_biased[1] + 
            self.ACCEL_MATRIX[1][2] * accel_biased[2],
            self.ACCEL_MATRIX[2][0] * accel_biased[0] + 
            self.ACCEL_MATRIX[2][1] * accel_biased[1] + 
            self.ACCEL_MATRIX[2][2] * accel_biased[2]
        ]
        
        return accel_calibrated
    
    def parse_data_packet(self, data_bytes, base_time):
        """Parse a single data packet of 4+12+7=23 bytes."""
        if len(data_bytes) < 23:
            return None
        
        # Parse timestamp (4 bytes) - based on your data, it appears to be big-endian
        # The first packet shows: 00 08 D5 C9
        timestamp_offset = struct.unpack('>I', data_bytes[0:4])[0]  # Big-endian unsigned int
        timestamp = base_time + timestamp_offset
        # convert timestamp to seconds
        timestamp = timestamp / 1000.0
        
        # Parse IMU data (12 bytes: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z)
        # From your data, these appear to be little-endian based on the values
        gyro_raw = struct.unpack('<hhh', data_bytes[4:10])
        accel_raw = struct.unpack('<hhh', data_bytes[10:16])
        
        # Parse magnetometer data (7 bytes)
        mag_status = data_bytes[16]
        # Mag data appears to be big-endian from your pattern (values like FB4A, 08BB)
        mag_data = struct.unpack('>hhh', data_bytes[17:23])
        
        # print(f"Timestamp raw: 0x{data_bytes[0]:02X}{data_bytes[1]:02X}{data_bytes[2]:02X}{data_bytes[3]:02X} = {timestamp_offset}")
        # print(f"Mag status: 0x{mag_status:02X}")
        # print(f"Mag data raw: {mag_data}")
        
        # Convert raw values to physical units
        accel_g = [x * 2.0 / 32768 for x in accel_raw]
        gyro_dps = [x * 245.0 / 32768 for x in gyro_raw]
        mag_uT = [x * 1.1 for x in mag_data]


        
        # print(f"Converted - Timestamp: {timestamp:.3f}, Accel (g): {accel_g}, Gyro (dps): {gyro_dps}, Mag (uT): {mag_uT}")
        
        # Apply coordinate transformations
        # Swap X and Y for accel
        temp = accel_g[0]
        accel_g[0] = -accel_g[1]
        accel_g[1] = -temp
        
        # Transform gyro
        temp = -gyro_dps[0]
        gyro_dps[0] = gyro_dps[1]
        gyro_dps[1] = -temp
        gyro_dps[2] = -gyro_dps[2]
        
        # Transform magnetometer
        temp = mag_uT[0]
        mag_uT[0] = -mag_uT[2]
        mag_uT[2] = temp
        mag_uT[1] = -mag_uT[1]

        # print(f"a_x: {accel_g[0]:.3f} g, a_y: {accel_g[1]:.3f} g, a_z: {accel_g[2]:.3f} g, "
        #       f"g_x: {gyro_dps[0]:.1f} dps, g_y: {gyro_dps[1]:.1f} dps, g_z: {gyro_dps[2]:.1f} dps, "
        #       f"m_x: {mag_uT[0]:.1f} uT, m_y: {mag_uT[1]:.1f} uT, m_z: {mag_uT[2]:.1f} uT")
        
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
        
        roll_x = -roll_x
        pitch_y = -pitch_y
        
        return roll_x, pitch_y, yaw_z
    
    def process_file(self, filepath):
        """Process a single data file."""
        try:
            file_num = self.extract_file_number(os.path.basename(filepath))
            
            # Skip if we've already processed this file or if it's older
            if file_num <= self.last_file_num:
                return
            
            # print(f"\nProcessing file: {os.path.basename(filepath)} (num: {file_num})")
            
            with open(filepath, 'r') as f:  # Open as text, not binary
                content = f.read()
            
            # Remove all whitespace (spaces, newlines, tabs)
            hex_string = ''.join(content.split())
            

            #Remove all the overhead data 
            # Convert hex string to bytes
            data = bytes.fromhex(hex_string)

            new_data_count = data[5]
            data = data[6:131] + data[131+8:244]
            # #Print raw bytes 23 bytes a line
            # for i in range(0, len(data), 23):
            #     print(f"Packet {i//23 + 1}: {data[i:i+23].hex().upper()}")
            
            # print(f"Raw bytes: {data[:].hex().upper()}")
            
            if len(data) < 6:
                return
            
            # Parse header
            # First 4 bytes: D4 00 00 00 (ignore)
            # Byte 5: new data counter
            
            # print(f"New data packets in this file: {new_data_count}")

            # h = data[9]
            # print(f"Decimal: {h}")
            # print(f"Hexadecimal: 0x{h:02X}")
            # print(f"Binary: {h:08b}")
            # print(data[9])
            
            if new_data_count == 0:
                return
            
            data_start = 0
            packet_size = 23  # 4(timestamp) + 12(imu) + 7(mag)
            
            # Base timestamp from filename or system time
            base_time = time.time()
            
            for i in range(min(new_data_count, 10)):  # Max 10 packets per file
                packet_offset = data_start + i * packet_size
                if packet_offset + packet_size > len(data):
                    break
                
                packet_data = data[packet_offset:packet_offset + packet_size]
                result = self.parse_data_packet(packet_data, 0)
                
                if result:
                    timestamp, accel_g, gyro_dps, mag_uT = result
                    
                    # Apply calibrations
                    mag_calibrated = self.calibrate_magnetometer(mag_uT)
                    accel_calibrated = self.calibrate_accelerometer(accel_g)
                    
                    # Queue for processing
                    data_queue.put((timestamp, accel_calibrated, gyro_dps, mag_calibrated, 0))
                    self.data_buffer.append((timestamp, accel_calibrated, gyro_dps, mag_calibrated))
            
            self.last_file_num = file_num
            
            # Save last processed file number
            handler = next((h for h in observers if isinstance(h, DataFileHandler)), None)
            if handler:
                handler.save_last_processed(file_num)
                
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    def process_buffered_data(self):
        """Process all data in the buffer (for initial load)."""
        # Sort by timestamp
        self.data_buffer.sort(key=lambda x: x[0])
        
        for timestamp, accel_g, gyro_dps, mag_uT in self.data_buffer:
            data_queue.put((timestamp, accel_g, gyro_dps, mag_uT, 0))

def processing_thread():
    """Background thread for data processing."""
    filter = MadgwickFilter()
    q = filter.init()
    lockin = LockInAmplifier(target_freq=10.0, alpha=0.03, hp_alpha=0.01)
    lowpass_mag = None
    ac_lowpass_mag = None
    mod_q_lowpass = None
    alpha = 0.08
    alpha_amp = 0.002
    alpha_q = 0.01
    
    processor = DataProcessor()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        while not shutdown_event.is_set():  # Changed condition
            try:
                current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get(timeout=1.0)

                # print(f"Accel (g): {accel_g}, Gyro (dps): {gyro_dps}, Mag (uT): {mag_uT}")
                # print(f"Time=: {current_time}\t Mag_x: {mag_uT[0]:.1f} uT\t, Mag_y: {mag_uT[1]:.1f} uT\t, Mag_z: {mag_uT[2]:.1f} uT")
                
                # Process with lock-in amplifier
                if calibration_trigger.is_set():
                    lockin.calibrate(mag_uT, current_time)
                    calibration_trigger.clear()
                
                mag_amplitudes, mag_phases, highpass_mag = lockin.update(mag_uT, current_time)
                
                # Update Madgwick filter
                q, error, omega_corrected = filter.update(
                    np.radians(gyro_dps),
                    np.array(accel_g),
                    np.array(mag_amplitudes),
                    current_time=current_time
                )
                
                # Apply low-pass filters
                mag_array = np.array(mag_uT)
                if lowpass_mag is None:
                    lowpass_mag = mag_array
                else:
                    lowpass_mag = alpha * mag_array + (1 - alpha) * lowpass_mag
                
                # Rotate quaternion
                mod_q = processor.rotate_quaternion_z_neg90_manual(q)
                mod_q = processor.quaternion_a_to_b(mod_q)
                
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
                
                # Print results
                mag_magnitude = np.sqrt(mag_array[0]**2 + mag_array[1]**2 + mag_array[2]**2)
                ac_magnitude = np.sqrt(mag_amplitudes[0]**2 + mag_amplitudes[1]**2 + mag_amplitudes[2]**2)
                
                print(f"Yaw: {yaw:.3f}°\tPitch: {pitch:.3f}°\tRoll: {roll:.3f}°\t"
                      f"Mag: {mag_magnitude:.1f} µT\tAC Mag: {ac_magnitude:.1f} µT")
                
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
                # print(f"Sent data to TCP clients: q={mod_q}, lowpass_mag={lowpass_mag}, ac_lowpass_mag={ac_lowpass_mag}")
                
                data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                if not shutdown_event.is_set():  # Only print if not shutting down
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
    tcp_server_socket.settimeout(0.5)  # Shorter timeout
    tcp_server_running = True
    
    print(f"TCP server listening on {TCP_SERVER_HOST}:{TCP_SERVER_PORT}")
    
    while tcp_server_running and not shutdown_event.is_set():  # Check both
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
    
    # Clean up
    for client_socket in tcp_clients:
        try:
            client_socket.close()
        except:
            pass
    tcp_clients.clear()
    
    if tcp_server_socket:
        tcp_server_socket.close()

def keyboard_listener():
    """Listen for keyboard input to trigger calibration."""
    print("Press 'c' to calibrate phase angles (when in known orientation)")
    print("Press 'r' to reset DC offsets")
    print("Press 'q' to quit")
    
    while not shutdown_event.is_set():  # Check shutdown
        try:
            if keyboard.is_pressed('c'):
                print("Calibration triggered...")
                calibration_trigger.set()
                time.sleep(0.5)
            
            if keyboard.is_pressed('r'):
                lockin = next((th for th in threading.enumerate() if hasattr(th, 'lockin')), None)
                if lockin:
                    lockin.reset_dc_offsets()
                time.sleep(0.5)
                
            if keyboard.is_pressed('q'):
                print("Quitting...")
                shutdown_event.set()  # Signal shutdown
                break
                
            time.sleep(0.1)
        except:
            break  # Exit on any error during shutdown

# def live_plot():
#     """Headless mode - no plot displayed."""
#     print("Running in headless mode (no plot)")
#     print("Press Ctrl+C to stop")
    
#     # Just keep the main thread alive
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("\nShutting down...")

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

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100)
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
    
    # Sort by file number (older to newer)
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
    
    # Stop observers
    for observer in observers:
        try:
            observer.stop()
            observer.join(timeout=1.0)
        except:
            pass
    
    # Stop TCP server
    tcp_server_running = False
    
    # Clear queues
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

def main():
    """Main function to run the application."""
    global observers
    
    # Register signal handlers
# Register signal handlers
    system_signal.signal(system_signal.SIGINT, signal_handler)
    system_signal.signal(system_signal.SIGTERM, signal_handler)
    
    # Register cleanup on exit
    atexit.register(cleanup)
    
    # Initialize processor
    processor = DataProcessor()
    
    # Process existing files first
    print("Processing existing files...")
    process_existing_files(DATA_DIRECTORY, processor)
    
    # Process buffered data
    processor.process_buffered_data()
    
    # Start background threads (set daemon=False for better cleanup)
    processing_thread_instance = threading.Thread(target=processing_thread)
    processing_thread_instance.daemon = True  # Still daemon but with proper shutdown
    processing_thread_instance.start()
    
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    tcp_thread = threading.Thread(target=tcp_server_thread)
    tcp_thread.daemon = True
    tcp_thread.start()
    
    # Set up file monitoring
    event_handler = DataFileHandler(processor)
    observer = Observer()
    observers.append(observer)  # Add to global list
    observer.schedule(event_handler, DATA_DIRECTORY, recursive=False)
    observer.start()
    
    print(f"Monitoring directory: {os.path.abspath(DATA_DIRECTORY)}")
    print("Waiting for new files...")
    print("Press Ctrl+C to stop")
    
    try:
        # Run the plot (this will block until closed)
        live_plot()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Plot error: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    observers = []
    main()