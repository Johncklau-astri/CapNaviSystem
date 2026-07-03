import asyncio
# from anyio import current_time
from bleak import BleakScanner, BleakClient
import struct
import time
import numpy as np
from collections import deque
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from concurrent.futures import ThreadPoolExecutor
import keyboard  # Added for keyboard input detection
import socket
import json
import struct
from scipy.spatial.transform import Rotation

LOG_FILENAME = "magnetometer_log.txt"
log_file = None
logging_enabled = False

accel_logging_trigger = threading.Event()

def start_logging():
    """Start logging magnetometer data to file."""
    global log_file, logging_enabled
    try:
        log_file = open(LOG_FILENAME, 'w')
        logging_enabled = True
        print(f"Started logging magnetometer data to {LOG_FILENAME}")
    except Exception as e:
        print(f"Error opening log file: {e}")

def stop_logging():
    """Stop logging and close the file."""
    global log_file, logging_enabled
    if log_file:
        log_file.close()
        log_file = None
    logging_enabled = False
    print("Stopped logging magnetometer data")

def log_magnetometer_data(mag_uT):
    """Log magnetometer data in the format: X,Y,Z"""
    global log_file, logging_enabled
    if logging_enabled and log_file:
        try:
            # Format: X,Y,Z\n (with 1 decimal place like your sample)
            log_file.write(f"{mag_uT[0]:.1f},{mag_uT[1]:.1f},{mag_uT[2]:.1f}\n")
            log_file.flush()  # Ensure data is written immediately
        except Exception as e:
            print(f"Error writing to log file: {e}")


mag_uT_buffer = deque(maxlen=250)

# For real-time plotting (buffer length: 1000 samples)
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

# TCP Server for sending data to aprilTagLocalization
TCP_SERVER_HOST = 'localhost'
TCP_SERVER_PORT=  12347  # New port for bt_angleEst data
tcp_clients = []
tcp_server_running = False
tcp_server_socket = None

# Define Nordic UART Service (NUS) UUIDs
NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # TX from device perspective
NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # RX from device perspective

# Variables for frequency calculation
last_time = time.time()
frequency_hz = 0.0
timestamp_deque = deque(maxlen=10)

# Connection management
connection_active = True
max_reconnection_attempts = 1000
reconnection_delay = 5  # seconds

q_prev = None  # To store previous quaternion for gyro prediction
gyro_predicted_field = None  # To be set based on gyro predictions
forced_invert_signs = False  # Set to True to force sign inversion for testing
counter = 0

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
        self.magnetic_ref = np.array([0.0, 1.0, 0.0])  # Magnetic field points North

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



def quaternion_a_to_b(q_a):
    """
    Convert quaternion from system A to system B.
    
    System A: X→North, Y→East, Z→Down
    System B: X→West, Y→South, Z→Up
    
    Transformations:
    - X rotation in A = -Y rotation in B
    - Y rotation in A = -X rotation in B  
    - Z rotation in A = -Z rotation in B
    
    Parameters:
    q_a: quaternion in system A as [w, x, y, z] or (w, x, y, z)
    
    Returns:
    q_b: quaternion in system B as [w, x, y, z]
    """
    
    # Ensure quaternion is in numpy array format
    q_a = np.array(q_a)
    
    # Extract components
    w, x, y, z = q_a
    
    # Create transformation quaternion that maps:
    # A_X → -B_Y, A_Y → -B_X, A_Z → -B_Z
    # This is equivalent to a 90-degree rotation around the Z-axis
    # followed by appropriate sign changes
    
    # The transformation can be represented as swapping axes with sign changes:
    # x_b = -y_a, y_b = -x_a, z_b = -z_a
    
    # Construct the transformed quaternion
    q_b = np.array([w, -y, -x, -z])
    
    # Normalize the result to ensure it's a valid unit quaternion
    q_b = q_b / np.linalg.norm(q_b)
    
    return q_b
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

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles with modified coordinate system.
    Positive pitch = -X rotation, Positive roll = -Y rotation, Yaw = +Z rotation.
    """
    w, x, y, z = q
    
    # Calculate the standard conversions
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.degrees(np.arctan2(t3, t4))
    
    # Apply coordinate system transformation
    # Positive pitch = -X rotation → negate pitch
    # Positive roll = -Y rotation → negate roll
    # Yaw remains unchanged
    roll_x = -roll_x    # Positive roll = -Y rotation
    pitch_y = -pitch_y  # Positive pitch = -X rotation
    
    return roll_x, pitch_y, yaw_z

def live_plot():
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    (ax1, ax2, ax3, ax4) = axs
    ax1.set_title('Raw Magnetometer (uT)')
    ax2.set_title('Lock-in AC Amplitude (uT)')
    ax3.set_title('Low pass mag (uT)')
    ax4.set_title('EULER angles (degrees)')
    ax4.set_xlabel('Time (s)')
    lines1 = [ax1.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines2 = [ax2.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines3 = [ax3.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines4 = [ax4.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    ax1.set_ylim(-10000, 10000)
    ax2.set_ylim(-300, 300)
    ax3.set_ylim(-5000, 5000)
    ax4.set_ylim(-180, 180)  # Adjust based on expected omega range
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


def rotate_quaternion_z_neg90_manual(q):
    """
    Rotate quaternion by -90 degrees about Z-axis using manual quaternion multiplication.
    
    Args:
        q: Quaternion in [w, x, y, z] format
        
    Returns:
        Rotated quaternion in [w, x, y, z] format
    """
    # Quaternion for -90 degree rotation about Z-axis
    angle_rad = np.radians(-90)
    w_rot = np.cos(angle_rad / 2)
    z_rot = np.sin(angle_rad / 2)
    rotation_quat = np.array([w_rot, 0, 0, z_rot])  # [w, x, y, z]
    
    # Quaternion multiplication: q_result = rotation_quat * q
    w1, x1, y1, z1 = rotation_quat
    w2, x2, y2, z2 = q
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


# Initialize filters and data queue
filter = MadgwickFilter(alpha=1.0, beta=0.3)  # Using Madgwick instead of Mahony
q = filter.init()

lockin = LockInAmplifier(target_freq=10.0, alpha=0.06 , hp_alpha=0.06)

data_queue = queue.Queue()
calibration_trigger = threading.Event()

def rotate_quaternion_yaw_180(q):
    """
    Rotate quaternion by 180 degrees around Z-axis.
    
    Args:
        q: Quaternion in [w, x, y, z] format
        
    Returns:
        Rotated quaternion in [w, x, y, z] format
    """
    # Quaternion for 180-degree rotation about Z-axis
    # angle = 180° = π radians
    # half-angle = 90° = π/2
    # cos(90°) = 0, sin(90°) = 1
    # So rotation quaternion = [cos(π/2), 0, 0, sin(π/2)] = [0, 0, 0, 1]
    rotation_quat = np.array([0.0, 0.0, 0.0, 1.0])  # [w, x, y, z]
    
    # Quaternion multiplication: q_result = rotation_quat * q
    w1, x1, y1, z1 = rotation_quat
    w2, x2, y2, z2 = q
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def process_data(current_time, accel_g, gyro_dps, mag_uT, running_freq_hz):
    global q_prev, gyro_predicted_field,forced_invert_signs
    if calibration_trigger.is_set():
        lockin.calibrate(mag_uT, current_time)
        calibration_trigger.clear()
    # Check if we need to flip sign permanently
    if forced_invert_signs:
        # CRITICAL: Reset lock-in internal state when flipping sign
        lockin.x_in = [0.0, 0.0, 0.0]
        lockin.x_quad = [0.0, 0.0, 0.0]
        lockin.dc_offsets = [0.0, 0.0, 0.0]
        lockin.filtered_signals = [0.0, 0.0, 0.0]
        
        # Also reset the filter to prevent orientation jump
        filter.init()
        q_prev = None  # Force gyro prediction to rebuild
        
        print("Lock-in state reset due to sign inversion")
        forced_invert_signs = False


    # Calculate expected magnetic field based on previous orientation
    if q_prev is not None:

        # print(f"Previous quaternion: {q_prev}")
        # AC field is known to be along global X axis
        ac_field_global = np.array([1.0, 0.0, 0.0])  # Helmholtz coil axis
        
        # Rotate to sensor frame using previous quaternion
        R = quaternion_to_rotation_matrix(q_prev)
        expected_field = R.T @ ac_field_global  # Global to body frame
        
        # Pass to lock-in for sign resolution
        gyro_predicted_field = expected_field
    
    # Update lock-in
    _lock_in_amp, mag_phases, highpass_mag_uT, t = lockin.update(mag_uT, current_time)

    # if(forced_invert_signs):
    #     mag_amplitudes = [-a for a in mag_amplitudes] 
    #     forced_invert_signs = False  # Reset after forcing sign inversion for testing

    # Use raw magnetometer data for Madgwick filter, not lock-in amplitudes
    q, error, omega_corrected = filter.update(
        np.radians(gyro_dps),
        np.array(accel_g),
        np.array(_lock_in_amp),
        current_time=current_time
    )
    q_prev = q.copy()

    return q, error, omega_corrected, _lock_in_amp, mag_phases , highpass_mag_uT ,t
def calculate_dc_offset(lockin, mag_raw, t_current, alpha_dc=0.05):
    """
    Calculate DC offset by removing lock-in detected AC component.
    Includes safety checks for None values.
    """
    # SAFETY CHECK: Ensure lock-in components are valid
    if lockin.x_in is None or lockin.x_quad is None:
        print("Warning: Lock-in components are None, returning raw data")
        return mag_raw, np.zeros_like(mag_raw)
    
    # Also check if they're lists with valid values
    if not isinstance(lockin.x_in, (list, np.ndarray)) or not isinstance(lockin.x_quad, (list, np.ndarray)):
        print(f"Warning: Lock-in components have invalid type: {type(lockin.x_in)}")
        return mag_raw, np.zeros_like(mag_raw)
    
    # Generate reference signals at current time
    ref_in = np.sin(2 * np.pi * lockin.target_freq * t_current)
    ref_quad = np.cos(2 * np.pi * lockin.target_freq * t_current)
    
    try:
        # Reconstruct AC component with safety
        ac_component = np.array([
            lockin.x_in[0] * ref_in + lockin.x_quad[0] * ref_quad,
            lockin.x_in[1] * ref_in + lockin.x_quad[1] * ref_quad,
            lockin.x_in[2] * ref_in + lockin.x_quad[2] * ref_quad
        ])
    except (TypeError, IndexError) as e:
        print(f"Error reconstructing AC component: {e}")
        return mag_raw, np.zeros_like(mag_raw)
    
    # Calculate instantaneous DC (raw - AC)
    dc_instantaneous = mag_raw - ac_component
    
    return dc_instantaneous, ac_component

class DCEstimator:
    """Robust DC offset estimator with adaptive filtering"""
    
    def __init__(self, alpha=0.05, outlier_threshold=5.0):
        self.alpha = alpha
        self.outlier_threshold = outlier_threshold
        self.dc_filtered = None
        self.dc_history = deque(maxlen=100)
        self.dc_std = None
        self.initialized = False
        
    def update(self, dc_instantaneous):
        """
        Update DC estimate with outlier rejection and adaptive filtering.
        Includes safety checks for None values.
        """
        # Safety check
        if dc_instantaneous is None:
            if self.dc_filtered is not None:
                return self.dc_filtered
            return np.array([0.0, 0.0, 0.0])
        
        # Ensure we're working with numpy array
        if not isinstance(dc_instantaneous, np.ndarray):
            dc_instantaneous = np.array(dc_instantaneous)
        
        # Initialize on first call
        if self.dc_filtered is None:
            self.dc_filtered = dc_instantaneous.copy()
            self.dc_history.append(dc_instantaneous.copy())
            self.initialized = True
            return self.dc_filtered
        
        # Update history for statistics
        self.dc_history.append(dc_instantaneous.copy())
        
        # Calculate running statistics if we have enough history
        effective_alpha = self.alpha
        if len(self.dc_history) > 10:
            dc_array = np.array(self.dc_history)
            self.dc_std = np.std(dc_array, axis=0)
            
            # Reject outliers
            deviation = np.abs(dc_instantaneous - self.dc_filtered)
            if np.any(deviation > self.outlier_threshold * self.dc_std):
                effective_alpha = self.alpha * 0.1
        
        # Apply filtered update
        self.dc_filtered = (1 - effective_alpha) * self.dc_filtered + effective_alpha * dc_instantaneous
        
        return self.dc_filtered
    
    def reset(self):
        """Reset estimator state"""
        self.dc_filtered = None
        self.dc_history.clear()
        self.dc_std = None
        self.initialized = False

# Initialize globally
dc_estimator = DCEstimator(alpha=0.03, outlier_threshold=3.0)


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

def sensor_to_mounting_frame(sensor_reading):
    """
    Transform a single sensor reading from calibration frame to mounting frame
    
    Args:
        sensor_reading: [bx, by, bz] in calibration frame (after ellipsoid fitting)
    
    Returns:
        mounting_frame: [bx_corr, by_corr, bz_corr] in orthogonal mounting frame
    """
    
    # Rotation matrix from VPO alignment (calibration frame -> mounting frame)
    # Based on misalignment angle of 3.333 degrees
    R_mc = np.array([
        [-0.99679348, 0.06167051, 0.05098528],
        [-0.05431766, -0.9893842, 0.13479058],
        [0.05875663, 0.13158897, 0.98956152]
    ])
    
    sensor_reading = np.array(sensor_reading)
    mounting_frame = R_mc @ sensor_reading
    return mounting_frame


def processing_thread():
    loop_timestamps = deque(maxlen=10)
    dc_estimator = DCEstimator(alpha=0.03)
    
    # Warm-up tracking
    warmup_samples = 0
    WARMUP_REQUIRED = 50  
    
    # Low-pass filter parameters for magnetometer data
    alpha = lockin.hp_alpha  # Use same alpha as lock-in high-pass for consistency
    lowpass_mag_uT = None
    final_dc = np.array([0.0, 0.0, 0.0])

    while True:
        try:
            loop_start_time = time.time()
            
            # 1. Grab freshest packet from queue, skipping backlog to prevent lag
            current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get(timeout=1.0)
            while not data_queue.empty():
                current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get_nowait()
            
            # --- START BENCHMARK ---
            math_start = time.perf_counter()

            # 2. Execute core signal processing directly (removes thread pool blocking overhead)
            q, error, omega_corrected, mag_amplitudes, mag_phases, highpass_mag_uT, t_lockin = \
                process_data(current_time, accel_g, gyro_dps, mag_uT, running_freq_hz)

            mag_uT_array = np.array(mag_uT)
            warmup_samples += 1
            
            # 3. Maintain base low-pass filter (Updated exactly ONCE per loop)
            if lowpass_mag_uT is None:
                lowpass_mag_uT = mag_uT_array
            else:
                lowpass_mag_uT = alpha * mag_uT_array + (1 - alpha) * lowpass_mag_uT

            # 4. DC Offset Estimation Logic
            if warmup_samples > WARMUP_REQUIRED:
                dc_instantaneous, ac_component = calculate_dc_offset(lockin, mag_uT_array, t_lockin)
                
                if dc_instantaneous is not None and ac_component is not None:
                    final_dc = dc_estimator.update(dc_instantaneous)
                else:
                    # Fallback to local low-pass filter if lock-in returns None
                    final_dc = lowpass_mag_uT
            else:
                # During warmup, use the running low-pass data
                final_dc = lowpass_mag_uT

            # 5. Coordinate transformations & Euler conversion
            mod_q = rotate_quaternion_z_neg90_manual(q)
            mod_q = quaternion_a_to_b(mod_q)
            pitch, roll, yaw = quaternion_to_euler(mod_q)

            # --- END BENCHMARK ---
            math_duration = (time.perf_counter() - math_start) * 1000.0  # Convert to milliseconds

            # 6. Loop Frequency Diagnostics
            loop_timestamps.append(loop_start_time)
            loop_freq_hz = 0.0
            if len(loop_timestamps) == 10:
                elapsed_time = loop_start_time - loop_timestamps[0]
                if elapsed_time > 0:
                    loop_freq_hz = 9 / elapsed_time

            # 7. Outputs and Client Transmissions
            print(
                f"Yaw: {yaw:.1f}°\tPitch: {pitch:.1f}°\t Roll: {roll:.1f}°,\t "
                f"AC Amplitudes (µT): [X: {mag_amplitudes[0]:.1f}, Y: {mag_amplitudes[1]:.1f}, Z: {mag_amplitudes[2]:.1f}]\t "
                f"BLE f: {running_freq_hz:.1f} Hz, Loop f: {loop_freq_hz:.1f} Hz, "
                f"Calculation Time: {math_duration:.2f} ms"
            )

            send_to_tcp_clients(mod_q, final_dc)

            # 8. Append to graphing histories
            plot_time.append(current_time)
            plot_mag_x.append(mag_uT[0])
            plot_mag_y.append(mag_uT[1])
            plot_mag_z.append(mag_uT[2])
            plot_amp_x.append(mag_amplitudes[0])
            plot_amp_y.append(mag_amplitudes[1])
            plot_amp_z.append(mag_amplitudes[2])
            plot_lp_x.append(lowpass_mag_uT[0])
            plot_lp_y.append(lowpass_mag_uT[1])
            plot_lp_z.append(lowpass_mag_uT[2])
            plot_hp_x.append(yaw)
            plot_hp_y.append(pitch)
            plot_hp_z.append(roll)
            
            data_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing thread error: {e}")


# def processing_thread():
#     loop_timestamps = deque(maxlen=10)
#     dc_estimator = DCEstimator(alpha=0.03)
    
#     # Add a counter for lock-in warm-up
#     warmup_samples = 0
#     WARMUP_REQUIRED = 50  # Wait for 50 samples before using lock-in for DC
    
#     # Low-pass filter parameters for magnetometer data
#     alpha = 0.1
#     lowpass_mag_uT = None
#     final_dc = 0
    
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         while True:
#             try:
#                 loop_start_time = time.time()
#                 current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get(timeout=1.0)
#                 while not data_queue.empty():
#                     current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get_nowait()
                
#                 future = executor.submit(process_data, current_time, accel_g, gyro_dps, mag_uT, running_freq_hz)
#                 q, error, omega_corrected, mag_amplitudes, mag_phases, highpass_mag_uT , t_lockin= future.result()

#                 # Convert mag_uT to numpy array
#                 mag_uT_array = np.array(mag_uT)
                
#                 # Increment warmup counter
#                 warmup_samples += 1
                
#                 # METHOD 1: Calculate DC using lock-in reconstruction (only after warmup)
#                 if warmup_samples > WARMUP_REQUIRED:
#                     dc_instantaneous, ac_component = calculate_dc_offset(lockin, mag_uT_array, t_lockin)
                    
#                     # Safety check for None
#                     if dc_instantaneous is not None and ac_component is not None:
#                         dc_estimate = dc_estimator.update(dc_instantaneous)
#                         final_dc = dc_estimate
                        
#                         # Optional: Verify AC removal
#                         ac_removal_quality = np.linalg.norm(mag_uT_array - (final_dc + ac_component))
#                         # print(f"DC Estimate: [{final_dc[0]:.1f}, {final_dc[1]:.1f}, {final_dc[2]:.1f}] uT | "
#                         #       f"AC Removal Error: {ac_removal_quality:.3f} uT")
#                     else:
#                         # Fallback to low-pass filter
#                         if lowpass_mag_uT is None:
#                             lowpass_mag_uT = mag_uT_array
#                         else:
#                             alpha_lp = 0.35
#                             lowpass_mag_uT = alpha_lp * mag_uT_array + (1 - alpha_lp) * lowpass_mag_uT
#                         final_dc = lowpass_mag_uT
#                 else:
#                     # During warmup, use simple low-pass filter
#                     if lowpass_mag_uT is None:
#                         lowpass_mag_uT = mag_uT_array
#                     else:
#                         alpha_lp = 0.05
#                         lowpass_mag_uT = alpha_lp * mag_uT_array + (1 - alpha_lp) * lowpass_mag_uT
#                     final_dc = lowpass_mag_uT
                

#                 # Compare the two methods
#                 dc_difference = np.linalg.norm(dc_estimate - lowpass_mag_uT)
                
#                 # Use dc_estimate as your final DC value (it's better than lowpass_mag_uT)
#                 final_dc = dc_estimate
                
#                 # Optional: Verify AC removal
#                 ac_removal_quality = np.linalg.norm(mag_uT_array - (final_dc + ac_component))
                
#                 # print(
#                 #     f"DC Estimate: [{final_dc[0]:.1f}, {final_dc[1]:.1f}, {final_dc[2]:.1f}] uT | "
#                 #     # f"AC Removal Error: {ac_removal_quality:.3f} uT | "
#                 #     f"vs LP Diff: {dc_difference:.3f} uT"
#                 # )
                
#                 # Apply low-pass filter to magnetometer data
#                 if lowpass_mag_uT is None:
#                     lowpass_mag_uT = mag_uT_array  # Initialize with first reading
#                 else:
#                     lowpass_mag_uT = alpha * mag_uT_array + (1 - alpha) * lowpass_mag_uT

#                 mod_q = rotate_quaternion_z_neg90_manual(q)
#                 mod_q = quaternion_a_to_b(mod_q)

#                 pitch, roll, yaw = quaternion_to_euler(mod_q)

#                 loop_timestamps.append(loop_start_time)
#                 loop_freq_hz = 0.0
#                 if len(loop_timestamps) == 10:
#                     elapsed_time = loop_start_time - loop_timestamps[0]
#                     if elapsed_time > 0:
#                         loop_freq_hz = 9 / elapsed_time

#                 accel_amp = np.sqrt(accel_g[0]**2 + accel_g[1]**2 + accel_g[2]**2)
#                 lowpass_mag_uT_amp = np.sqrt(lowpass_mag_uT[0]**2 + lowpass_mag_uT[1]**2 + lowpass_mag_uT[2]**2)


#                 ## calculate Mag DC theoretical
#                 mag_dc_theoretical = np.array([0.0, 0.0, 0.0])  # Replace with actual theoretical values if available

#                 np_mag_amplitudes = np.array(mag_amplitudes)
                
#                 mag_dc_theoretical = mag_uT_array - np_mag_amplitudes*np.sin(2*np.pi*10*t_lockin+ np.radians(mag_phases))



#                 # # Earth mag field removal
#                 # #Earth field measured in global frame, NOT capsule sensor axis, NEED to convert sensor reading to global
#                 # earth_mag_global = np.array([-11,-23,-22])

#                 # earth_mag_sensor = quaternion_to_rotation_matrix(mod_q).T@earth_mag_global
#                 # earth_mag_sensor = np.array([-earth_mag_sensor[1],-earth_mag_sensor[0],-earth_mag_sensor[2]])


#                 # adjusted_mag_uT = lowpass_mag_uT - earth_mag_sensor
#                 # adjusted_mag_uT_mag = np.sqrt(adjusted_mag_uT[0]**2 + adjusted_mag_uT[1]**2 + adjusted_mag_uT[2]**2)                

#                 # mountFrame = sensor_to_mounting_frame(lowpass_mag_uT)
#                 mag_uT_amp = np.sqrt(mag_uT_array[0]**2 + mag_uT_array[1]**2 + mag_uT_array[2]**2)

#                 print(
#                     f"Yaw: {yaw:.1f}°\tPitch: {pitch:.1f}°\t Roll: {roll:.1f}°,\t "
#                     # f"mag_dc_theoretical (uT): [X: {mag_dc_theoretical[0]:.1f}, Y: {mag_dc_theoretical[1]:.1f}, Z: {mag_dc_theoretical[2]:.1f}]\t "
#                     # f"accel: x:{accel_g[0]:.3f}g, y:{accel_g[1]:.3f}g, z:{accel_g[2]:.3f}g\t "
#                     f"mod_q: {mod_q[0]:.3f} , {mod_q[1]:.3f} , {mod_q[2]:.3f} , {mod_q[3]:.3f}\t"
#                     f"AC Amplitudes (µT): [X: {mag_amplitudes[0]:.1f}, Y: {mag_amplitudes[1]:.1f}, Z: {mag_amplitudes[2]:.1f}]\t "
#                     # f"DC Estimate (uT): [X: {final_dc[0]:.1f}\t, Y: {final_dc[1]:.1f}\t, Z: {final_dc[2]:.1f}]\t "
#                     # f"Calibrated Mag (uT): [X: {mag_uT[0]:.1f}, Y: {mag_uT[1]:.1f}, Z: {mag_uT[2]:.1f}, \tMagnitude: {mag_uT_amp:.1f}]\t "
#                     # f"mountFrame Mag (uT): [X: {mountFrame[0]:.1f}, Y: {mountFrame[1]:.1f}, Z: {mountFrame[2]:.1f}]\t "
#                     # f"LP Mag (uT): [X: {lowpass_mag_uT[0]:.1f},\tY: {lowpass_mag_uT[1]:.1f},\tZ: {lowpass_mag_uT[2]:.1f}\t "
#                     # f"earth_mag_sensor (uT): [X: {earth_mag_sensor[0]:.1f}, Y: {earth_mag_sensor[1]:.1f}, Z: {earth_mag_sensor[2]:.1f}]\t"
#                     # f"Adj Mag (uT): [X: {adjusted_mag_uT[0]:.1f}, Y: {adjusted_mag_uT[1]:.1f},  \tZ: {adjusted_mag_uT[2]:.1f} , mag:{adjusted_mag_uT_mag:.1f}]\t "
#                     f"BLE f: {running_freq_hz:.1f} Hz, Loop f: {loop_freq_hz:.1f} Hz"
#                 )



#                 send_to_tcp_clients(mod_q, final_dc)

#                 plot_time.append(current_time)
#                 plot_mag_x.append(mag_uT[0])
#                 plot_mag_y.append(mag_uT[1])
#                 plot_mag_z.append(mag_uT[2])
#                 plot_amp_x.append(mag_amplitudes[0])
#                 plot_amp_y.append(mag_amplitudes[1])
#                 plot_amp_z.append(mag_amplitudes[2])
#                 plot_lp_x.append(lowpass_mag_uT[0])
#                 plot_lp_y.append(lowpass_mag_uT[1])
#                 plot_lp_z.append(lowpass_mag_uT[2])
#                 plot_hp_x.append(yaw)
#                 plot_hp_y.append(pitch)
#                 plot_hp_z.append(roll)
#                 data_queue.task_done()

#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 print(f"Processing thread error: {e}")



def rotate_quaternion_scipy(q, angle_degrees, axis='z'):
    """
    Rotate quaternion using scipy's Rotation class
    """
    # Create rotation from axis-angle
    axis_rotation = Rotation.from_rotvec(np.radians(angle_degrees) * np.eye(3)['xyz'.index(axis)])
    
    # Convert input quaternion to scipy Rotation
    current_rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # [x, y, z, w] format
    
    # Apply rotation
    result_rot = axis_rotation * current_rot
    
    # Convert back to [w, x, y, z] format
    result_quat = result_rot.as_quat()  # [x, y, z, w]
    return np.array([result_quat[3], result_quat[0], result_quat[1], result_quat[2]])
# --- Global variables for tracking true dataset updates ---
last_time = time.time()
frequency_hz = 0.0
timestamp_deque = deque(maxlen=10)

# Store the last known RAW sensor values to detect duplicates/stale data
last_raw_accel = None
last_raw_gyro = None
last_raw_mag = None


def notification_handler(sender, data):
    global last_time, frequency_hz, last_raw_accel, last_raw_gyro, last_raw_mag
    
    if len(data) >= 20 and data[0] == 0x12 and data[1] == 0x34:
        # 1. Extract raw byte segments
        gyro_data = data[2:8]
        accel_data = data[8:14]
        mag_data = data[15:21]
        
        # 2. Unpack raw integer tuples
        accel_raw = struct.unpack("<hhh", accel_data)
        gyro_raw = struct.unpack("<hhh", gyro_data)
        mag_raw = struct.unpack(">hhh", mag_data)

        # 3. Value Change Detection (Check if the dataset is actually new)
        is_new_dataset = (
            accel_raw != last_raw_accel or 
            gyro_raw != last_raw_gyro or 
            mag_raw != last_raw_mag
        )

        # Update historical cache
        last_raw_accel = accel_raw
        last_raw_gyro = gyro_raw
        last_raw_mag = mag_raw
 
        # 4. Only update the timestamp rolling average if the data changed
        running_freq_hz = 0.0
        if is_new_dataset:
            current_time = time.time()
            timestamp_deque.append(current_time)
            if len(timestamp_deque) == 10:
                elapsed_time = current_time - timestamp_deque[0]
                if elapsed_time > 0:
                    running_freq_hz = 9 / elapsed_time
        else:
            # If the deque already has readings, maintain the last valid calculated frequency
            if len(timestamp_deque) == 10:
                elapsed_time = timestamp_deque[-1] - timestamp_deque[0]
                if elapsed_time > 0:
                    running_freq_hz = 9 / elapsed_time

        # 5. Process scaling & transformation logic as usual
        accel_g = [x * 2.0 / 32768 for x in accel_raw]
        gyro_dps = [x * 245.0 / 32768 for x in gyro_raw]
        mag_uT = [x * 1.1 for x in mag_raw]

        temp = accel_g[0]
        accel_g[0] = accel_g[1]
        accel_g[1] = temp

        temp = -gyro_dps[0]
        gyro_dps[0] = -gyro_dps[1]
        gyro_dps[1] = temp
        gyro_dps[2] = -gyro_dps[2]

        temp = mag_uT[0]
        mag_uT[0] = -mag_uT[2]
        mag_uT[2] = temp
        mag_uT[1] = -mag_uT[1]

        # Calibrate sensors
        mag_uT_calibrated = calibrate_magnetometer(mag_uT)
        accel_g = calibrate_accelerometer(accel_g)

        if accel_logging_trigger.is_set():
            log_single_accelerometer_point(mag_uT_calibrated)
            accel_logging_trigger.clear()

        if logging_enabled:
            log_magnetometer_data(mag_uT)

        # 6. Feed the queue with the true dataset update rate
        data_queue.put((time.time(), accel_g, gyro_dps, mag_uT_calibrated, running_freq_hz))


# def notification_handler(sender, data):
#     global last_time, frequency_hz
#     current_time = time.time()
#     timestamp_deque.append(current_time)
#     running_freq_hz = 0.0
#     if len(timestamp_deque) == 10:
#         elapsed_time = current_time - timestamp_deque[0]
#         if elapsed_time > 0:
#             running_freq_hz = 9 / elapsed_time
#     if len(data) >= 20 and data[0] == 0x12 and data[1] == 0x34:
#         gyro_data = data[2:8]
#         accel_data = data[8:14]
#         mag_data = data[15:21]
#         accel = struct.unpack("<hhh", accel_data)
#         gyro = struct.unpack("<hhh", gyro_data)
#         mag = struct.unpack(">hhh", mag_data)

#         accel_g = [x * 2.0 / 32768 for x in accel]
#         gyro_dps = [x * 245.0 / 32768 for x in gyro]
#         mag_uT = [x * 1.1 for x in mag]

#         temp = accel_g[0]
#         accel_g[0] = accel_g[1]
#         accel_g[1] = temp

#         temp = -gyro_dps[0]
#         gyro_dps[0] = -gyro_dps[1]
#         gyro_dps[1] = temp
#         gyro_dps[2] = -gyro_dps[2]

#         temp = mag_uT[0]
#         mag_uT[0] = -mag_uT[2]
#         mag_uT[2] = temp
#         mag_uT[1] = -mag_uT[1]

#         # print(f"Raw Accel: [X: {accel_g[0]:.2f}g, Y: {accel_g[1]:.2f}g, Z: {accel_g[2]:.2f}g]")

#         # Calibrate magnetometer data
#         mag_uT_calibrated = calibrate_magnetometer(mag_uT)
#         # print(f"Calib Mag: [X: {mag_uT_calibrated[0]:.1f} uT, Y: {mag_uT_calibrated[1]:.1f} uT, Z: {mag_uT_calibrated[2]:.1f} uT]")
#         # print(f"Calib mag: [X: {mag_uT_calibrated[0]:.1f} uT, Y: {mag_uT_calibrated[1]:.1f} uT, Z: {mag_uT_calibrated[2]:.1f} uT]")

#         # print(f"Raw Mag (uT): {mag_uT}, Calibrated Mag (uT): {mag_uT_calibrated}")
#         #Calibrate accelerometer data
#         accel_g = calibrate_accelerometer(accel_g)
#         # print(f"Corrected Accel: [X: {accel_g[0]:.2f}g,\tY: {accel_g[1]:.2f}g,\tZ: {accel_g[2]:.2f}g]")

#         # Check if we need to log a single accelerometer point
#         if accel_logging_trigger.is_set():
#             log_single_accelerometer_point(mag_uT_calibrated)

#             accel_logging_trigger.clear()

#         if logging_enabled:
#             log_magnetometer_data(mag_uT)
#         # Log magnetometer data if logging is enabled
#         # if logging_enabled:
#         #     log_magnetometer_data(mag_uT_calibrated)

#         data_queue.put((current_time, accel_g, gyro_dps, mag_uT_calibrated, running_freq_hz))




def calibrate_accelerometer(accel_raw, tune_roll_deg=0, tune_pitch_deg=0):
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

import numpy as np

def calibrate_magnetometer(mag_raw, pitch_deg=0, roll_deg=0, yaw_deg=0):
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
    hard_iron_bias = np.array([19.744134, -7.018865, 14.425593])
    soft_iron_matrix = np.array([
        [1.000023, -0.037484, -0.004686],
        [-0.037484, 1.005176, 0.007214],
        [-0.004686, 0.007214, 0.991864]
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



def send_to_tcp_clients(q, mag_uT):
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
    
    # Remove disconnected clients
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
    tcp_server_socket.settimeout(1.0)
    tcp_server_running = True
    
    print(f"BT AngleEst TCP server listening on {TCP_SERVER_HOST}:{TCP_SERVER_PORT}")
    
    while tcp_server_running:
        try:
            client_socket, addr = tcp_server_socket.accept()
            client_socket.settimeout(1.0)
            tcp_clients.append(client_socket)
            print(f"BT AngleEst: Client connected from {addr}")
        except socket.timeout:
            continue
        except Exception as e:
            if tcp_server_running:
                print(f"BT AngleEst TCP server error: {e}")
    
    # Cleanup
    for client_socket in tcp_clients:
        client_socket.close()
    tcp_server_socket.close()


def log_single_accelerometer_point(accel_g):
    """Log a single accelerometer data point to file."""
    try:
        with open("accelerometer_single_log.txt", 'a') as f:  # 'a' for append mode
            f.write(f"{accel_g[0]:.3f},{accel_g[1]:.3f},{accel_g[2]:.3f}\n")
        print(f"Logged accelerometer point: X={accel_g[0]:.3f}, Y={accel_g[1]:.3f}, Z={accel_g[2]:.3f}")
    except Exception as e:
        print(f"Error writing accelerometer data: {e}")


def keyboard_listener():
    """Listen for keyboard input to trigger calibration and logging."""
    print("Press 'c' to calibrate the sensor's phase angles when in known orientation.")
    print("Press 'm' to start/stop logging magnetometer data.")
    print("Press 'l' to log a single accelerometer data point.")
    global forced_invert_signs
    while True:
        if keyboard.is_pressed('c'):
            # print("Calibration triggered. Ensure sensor is in known orientation.")
            # calibration_trigger.set()
            # time.sleep(0.5)  # Debounce to prevent multiple triggers
            forced_invert_signs = True
        
        if keyboard.is_pressed('m'):
            global logging_enabled
            if not logging_enabled:
                start_logging()
            else:
                stop_logging()
            time.sleep(0.5)  # Debounce to prevent multiple triggers
        
        if keyboard.is_pressed('l'):
            print("Logging single accelerometer data point...")
            accel_logging_trigger.set()
            time.sleep(0.5)  # Debounce to prevent multiple triggers
        
        time.sleep(0.1)


async def connect_and_receive(device_name):
    global connection_active
    
    print(f"Scanning for device with name: {device_name}...")
    
    reconnection_attempts = 0
    
    while connection_active and reconnection_attempts < max_reconnection_attempts:
        try:
            devices = await BleakScanner.discover()
            target_device = None
            for device in devices:
                if device.name == device_name:
                    target_device = device
                    break
            
            if not target_device:
                print(f"Device {device_name} not found! Attempt {reconnection_attempts + 1}/{max_reconnection_attempts}")
                reconnection_attempts += 1
                if reconnection_attempts < max_reconnection_attempts:
                    print(f"Retrying in {reconnection_delay} seconds...")
                    await asyncio.sleep(reconnection_delay)
                continue
            
            print(f"Found device: {target_device.name} ({target_device.address})")
            print(f"Attempting connection {reconnection_attempts + 1}/{max_reconnection_attempts}...")
            
            # Reset reconnection attempts on successful discovery
            reconnection_attempts = 0
            
            async with BleakClient(
                target_device.address, 
                timeout=30.0,
                winrt=dict(use_cached_services=False)  # Windows-specific fix
            ) as client:
                print(f"Connected to {target_device.name}")
                await client.start_notify(NUS_TX_UUID, notification_handler)
                print("Notifications enabled. Receiving data...")
                
                # Monitor connection status
                while client.is_connected and connection_active:
                    await asyncio.sleep(1)
                
                if not client.is_connected:
                    print("Device disconnected unexpectedly")
                    # Try to stop notifications gracefully
                    try:
                        await client.stop_notify(NUS_TX_UUID)
                    except:
                        pass
                    
                if connection_active:
                    print(f"Attempting reconnection in {reconnection_delay} seconds...")
                    await asyncio.sleep(reconnection_delay)
                    
        except asyncio.TimeoutError:
            print(f"Connection timeout on attempt {reconnection_attempts + 1}/{max_reconnection_attempts}")
            reconnection_attempts += 1
            if reconnection_attempts < max_reconnection_attempts:
                print(f"Retrying in {reconnection_delay} seconds...")
                await asyncio.sleep(reconnection_delay)
        except Exception as e:
            print(f"Connection error on attempt {reconnection_attempts + 1}/{max_reconnection_attempts}: {e}")
            reconnection_attempts += 1
            if reconnection_attempts < max_reconnection_attempts:
                print(f"Retrying in {reconnection_delay} seconds...")
                await asyncio.sleep(reconnection_delay)
    
    if reconnection_attempts >= max_reconnection_attempts:
        print(f"Failed to connect after {max_reconnection_attempts} attempts")
    print("Connection manager stopped")

async def main():
    device_name = "Nordic_UART_Service"

    processing_thread_instance = threading.Thread(target=processing_thread, daemon=True)
    processing_thread_instance.start()

    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()

    # Start TCP server thread
    tcp_thread = threading.Thread(target=tcp_server_thread, daemon=True)
    tcp_thread.start()
    
    try:
        await connect_and_receive(device_name)
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        global connection_active
        connection_active = False

def start_asyncio():
    asyncio.run(main())

def stop_connection():
    """Function to stop the connection manager from another thread"""
    global connection_active
    connection_active = False

if __name__ == "__main__":
    asyncio_thread = threading.Thread(target=start_asyncio, daemon=True)
    asyncio_thread.start()
    
    try:
        live_plot()
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_connection()
    except Exception as e:
        print(f"Plot error: {e}")
        stop_connection()