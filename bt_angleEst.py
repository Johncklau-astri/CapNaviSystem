import asyncio
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
plot_roll = deque(maxlen=PLOT_BUFFER_LEN)
plot_pitch = deque(maxlen=PLOT_BUFFER_LEN)
plot_yaw = deque(maxlen=PLOT_BUFFER_LEN)
plot_omega_x = deque(maxlen=PLOT_BUFFER_LEN)
plot_omega_y = deque(maxlen=PLOT_BUFFER_LEN)
plot_omega_z = deque(maxlen=PLOT_BUFFER_LEN)

# TCP Server for sending data to aprilTagLocalization
TCP_SERVER_HOST = 'localhost'
TCP_SERVER_PORT = 12347  # New port for bt_angleEst data
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
max_reconnection_attempts = 10
reconnection_delay = 5  # seconds

class MadgwickFilter:
    def __init__(self, beta=0.5):
        """
        Initialize Madgwick filter parameters.
        
        Args:
            beta (float): Algorithm gain beta
            dt (float): Sample period [s]
        """
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None  # Track last update time
        
        # Reference vectors in Earth frame
        self.gravity_ref = np.array([0.0, 0.0, 1.0])  # Gravity points down
        self.magnetic_ref = np.array([0.0, -1.0, 0.0])  # Magnetic field points North

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
        
        # Return error vector for debugging (approximated from gradient)
        error = gradient[1:] * self.beta
        
        return self.q, error, corrected_gyro

class LockInAmplifier:
    def __init__(self, target_freq=10.0, alpha=0.01):
        self.target_freq = target_freq
        self.alpha = alpha
        self.x_in = [0.0, 0.0, 0.0]
        self.x_quad = [0.0, 0.0, 0.0]
        self.last_time = time.time()
        self.phase_offsets = [0.0, 0.0, 0.0]  # Store phase offsets for X, Y, Z
        self.is_calibrated = False

    def calibrate(self, mag_uT, current_time):
        """Calibrate phase angles to be negative in the current orientation."""
        dt = current_time - self.last_time
        self.last_time = current_time
        t = current_time
        ref_in = np.sin(2 * np.pi * self.target_freq * t)
        ref_quad = np.cos(2 * np.pi * self.target_freq * t)

        for i in range(3):
            prod_in = mag_uT[i] * ref_in
            prod_quad = mag_uT[i] * ref_quad
            self.x_in[i] = (1 - self.alpha) * self.x_in[i] + self.alpha * prod_in
            self.x_quad[i] = (1 - self.alpha) * self.x_quad[i] + self.alpha * prod_quad
            current_phase = np.arctan2(self.x_quad[i], self.x_in[i])

            # Adjust phase to be negative (shift to range [-180, 0])
            if(current_phase >= 0):
                self.phase_offsets[i] = 1

        self.is_calibrated = True
        print("Calibration complete. Phase offsets (degrees):", [np.degrees(off) for off in self.phase_offsets])

    def update(self, mag_uT, current_time):
        dt = current_time - self.last_time
        self.last_time = current_time
        t = current_time
        ref_in = np.sin(2 * np.pi * self.target_freq * t)
        ref_quad = np.cos(2 * np.pi * self.target_freq * t)

        amplitudes = []
        phase_angles = []
        for i in range(3):
            prod_in = mag_uT[i] * ref_in
            prod_quad = mag_uT[i] * ref_quad
            self.x_in[i] = (1 - self.alpha) * self.x_in[i] + self.alpha * prod_in
            self.x_quad[i] = (1 - self.alpha) * self.x_quad[i] + self.alpha * prod_quad
            amplitude = np.sqrt(self.x_in[i]**2 + self.x_quad[i]**2)
            phase = np.arctan2(self.x_quad[i], self.x_in[i])
            # Apply phase offset
            if self.is_calibrated:
                if self.phase_offsets[i] == 1:
                    phase = -phase  # Shift by 180 degrees
                else:
                    phase = phase
            if phase > 0:
                amplitude = amplitude
            else:
                amplitude = -amplitude        

            amplitudes.append(amplitude)
            phase_angles.append(np.degrees(phase))
        return amplitudes, phase_angles


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
    ax3.set_title('Euler Angles (degrees)')
    ax4.set_title('Corrected Omega (deg/s)')
    ax4.set_xlabel('Time (s)')
    lines1 = [ax1.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines2 = [ax2.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines3 = [ax3.plot([], [], label=lbl)[0] for lbl in ['Roll', 'Pitch', 'Yaw']]
    lines4 = [ax4.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    ax1.set_ylim(-10000, 10000)
    ax2.set_ylim(-1000, 1000)
    ax3.set_ylim(-180, 180)
    ax4.set_ylim(-200, 200)  # Adjust based on expected omega range
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
        data3 = [np.array(plot_roll), np.array(plot_pitch), np.array(plot_yaw)]
        data4 = [np.array(plot_omega_x), np.array(plot_omega_y), np.array(plot_omega_z)]
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
filter = MadgwickFilter()# Using Madgwick instead of Mahony
q = filter.init()

lockin = LockInAmplifier(target_freq=10.0, alpha=0.03)

data_queue = queue.Queue()
calibration_trigger = threading.Event()

def process_data(current_time, accel_g, gyro_dps, mag_uT, running_freq_hz):
    if calibration_trigger.is_set():
        lockin.calibrate(mag_uT, current_time)
        calibration_trigger.clear()
    
    mag_amplitudes, mag_phases = lockin.update(mag_uT, current_time)
    
    # Use raw magnetometer data for Madgwick filter, not lock-in amplitudes
    q, error, omega_corrected = filter.update(
        np.radians(gyro_dps),
        np.array(accel_g),
        np.array(mag_amplitudes),
        current_time=current_time
    )

    return q, error, omega_corrected, mag_amplitudes, mag_phases

def processing_thread():
    loop_timestamps = deque(maxlen=10)
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            try:
                loop_start_time = time.time()
                current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get(timeout=1.0)
                while not data_queue.empty():
                    current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get_nowait()
                
                future = executor.submit(process_data, current_time, accel_g, gyro_dps, mag_uT, running_freq_hz)
                q, error, omega_corrected, mag_amplitudes, mag_phases = future.result()

                # Store mag_uT in buffer and calculate average
                mag_uT_buffer.append(mag_uT)
                avg_mag_uT = np.mean(list(mag_uT_buffer), axis=0) if mag_uT_buffer else mag_uT

                mod_q = rotate_quaternion_z_neg90_manual(q)
                mod_q = quaternion_a_to_b(mod_q)

               

                pitch , roll,  yaw = quaternion_to_euler(mod_q)

                loop_timestamps.append(loop_start_time)
                loop_freq_hz = 0.0
                if len(loop_timestamps) == 10:
                    elapsed_time = loop_start_time - loop_timestamps[0]
                    if elapsed_time > 0:
                        loop_freq_hz = 9 / elapsed_time

                accel_mag = np.sqrt(accel_g[0]**2 + accel_g[1]**2 + accel_g[2]**2)

                print(
                    f"Yaw: {yaw:.1f}°\t "
                    f"Madgwick Pitch: {pitch:.1f}°\t Roll: {roll:.1f}°,\t\t "
                    f"q: {q[0]:.2f} , {q[1]:.2f} , {q[2]:.2f} , {q[3]:.2f}\t"
                    f"mod_q: {mod_q[0]:.2f} , {mod_q[1]:.2f} , {mod_q[2]:.2f} , {mod_q[3]:.2f}\t"
                    f"Avg Mag (uT): [X: {avg_mag_uT[0]:.1f}, Y: {avg_mag_uT[1]:.1f}, Z: {avg_mag_uT[2]:.1f}]\t "
                    f"Phases (deg): [X: {mag_phases[0]:.1f}, Y: {mag_phases[1]:.1f}, Z: {mag_phases[2]:.1f}]\t "
                    # f"Cur. Time: {current_time:.2f}\t "
                    # f"accel: x:{accel_g[0]:.2f}g, y:{accel_g[1]:.2f}g, z:{accel_g[2]:.2f}g\t "
                    # f"gyro: x:{gyro_dps[0]:.1f}°/s, y:{gyro_dps[1]:.1f}°/s, z:{gyro_dps[2]:.1f}°/s\t "
                    # # f"Error: [{error[0]:.3f}, {error[1]:.3f}, {error[2]:.3f}],\t\t "
                    # # f"omega_corrected (deg/s): [{np.degrees(omega_corrected[0]):.1f}, {np.degrees(omega_corrected[1]):.1f}, {np.degrees(omega_corrected[2]):.1f}],\t "
                    f"AC Amplitudes (µT): [X: {mag_amplitudes[0]:.1f}, Y: {mag_amplitudes[1]:.1f}, Z: {mag_amplitudes[2]:.1f}]\t "
                    f"BLE f: {running_freq_hz:.1f} Hz, Loop f: {loop_freq_hz:.1f} Hz"
                )
                send_to_tcp_clients(mod_q, avg_mag_uT)

                plot_time.append(current_time)
                plot_mag_x.append(mag_uT[0])
                plot_mag_y.append(mag_uT[1])
                plot_mag_z.append(mag_uT[2])
                plot_amp_x.append(mag_amplitudes[0])
                plot_amp_y.append(mag_amplitudes[1])
                plot_amp_z.append(mag_amplitudes[2])
                plot_roll.append(roll)
                plot_pitch.append(pitch)
                plot_yaw.append(yaw)
                plot_omega_x.append(np.degrees(omega_corrected[0]))
                plot_omega_y.append(np.degrees(omega_corrected[1]))
                plot_omega_z.append(np.degrees(omega_corrected[2]))
                data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing thread error: {e}")

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

def notification_handler(sender, data):
    global last_time, frequency_hz
    current_time = time.time()
    timestamp_deque.append(current_time)
    running_freq_hz = 0.0
    if len(timestamp_deque) == 10:
        elapsed_time = current_time - timestamp_deque[0]
        if elapsed_time > 0:
            running_freq_hz = 9 / elapsed_time
    if len(data) >= 20 and data[0] == 0x12 and data[1] == 0x34:
        gyro_data = data[2:8]
        accel_data = data[8:14]
        mag_data = data[15:21]
        accel = struct.unpack("<hhh", accel_data)
        gyro = struct.unpack("<hhh", gyro_data)
        mag = struct.unpack(">hhh", mag_data)

        accel_g = [x * 2.0 / 32768 for x in accel]
        gyro_dps = [x * 245.0 / 32768 for x in gyro]
        mag_uT = [x * 1.1 for x in mag]

        temp = accel_g[0]
        accel_g[0] = accel_g[1]
        accel_g[1] = temp


        temp = -gyro_dps[0]
        gyro_dps[0] = -gyro_dps[1]
        gyro_dps[1] = temp
        gyro_dps[2] = -gyro_dps[2]
        

        temp = mag_uT[0]
        mag_uT[0] = mag_uT[2]
        mag_uT[2] = temp





        data_queue.put((current_time, accel_g, gyro_dps, mag_uT, running_freq_hz))

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

def keyboard_listener():
    """Listen for keyboard input to trigger calibration."""
    print("Press 'c' to calibrate the sensor's phase angles when in known orientation.")
    while True:
        if keyboard.is_pressed('c'):
            print("Calibration triggered. Ensure sensor is in known orientation.")
            calibration_trigger.set()
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