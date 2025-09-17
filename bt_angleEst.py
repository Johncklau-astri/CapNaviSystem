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

# For real-time plotting (buffer length: 1000 samples)
PLOT_BUFFER_LEN = 1000
plot_time = deque(maxlen=PLOT_BUFFER_LEN)
plot_mag_x = deque(maxlen=PLOT_BUFFER_LEN)
plot_mag_y = deque(maxlen=PLOT_BUFFER_LEN)
plot_mag_z = deque(maxlen=PLOT_BUFFER_LEN)
plot_amp_x = deque(maxlen=PLOT_BUFFER_LEN)
plot_amp_y = deque(maxlen=PLOT_BUFFER_LEN)
plot_amp_z = deque(maxlen=PLOT_BUFFER_LEN)

# Define Nordic UART Service (NUS) UUIDs
NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # TX from device perspective
NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # RX from device perspective

# Variables for frequency calculation
last_time = time.time()
frequency_hz = 0.0
timestamp_deque = deque(maxlen=10)

class KalmanFilter:
    def __init__(self, process_noise=0.005, measurement_noise=0.1):
        self.state = np.zeros(4)  # [pitch, roll, bias_x, bias_y]
        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.last_time = time.time()
        self.roll_unwrapped = 0.0

    def update(self, accel_g, gyro_dps, current_time):
        ax, ay, az = accel_g
        gx, gy, gz = gyro_dps
        dt = current_time - self.last_time
        self.last_time = current_time

        # Predict
        F = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[dt, 0], [0, dt], [0, 0], [0, 0]])
        u = np.array([gx, gy])
        self.state = F @ self.state + B @ u
        self.P = F @ self.P @ F.T + self.Q

        # Adjust for accelerometer data signage
        if az > 0 and ay < 0:
            extended_pitch = (180 + np.arctan2(ay, np.sqrt(ax**2 + az**2)) * 180 / np.pi)
        elif az > 0 and ay > 0:
            extended_pitch = (-180 + np.arctan2(ay, np.sqrt(ax**2 + az**2)) * 180 / np.pi)
        else:
            extended_pitch = -np.arctan2(ay, np.sqrt(ax**2 + az**2)) * 180 / np.pi

        # if az > 0 and ax < 0:
        #     extended_roll = (-180 - np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi)
        # elif az > 0 and ax > 0:
        #     extended_roll = (180 - np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi)
        # else:
        #     extended_roll = np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi

        

        # extended_pitch = -np.arctan2(ay, np.sqrt(ax**2 + az**2)) * 180 / np.pi

        extended_roll = np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi

        # Measurement
        z = np.array([extended_pitch, extended_roll])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.state[0], self.state[1]

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

def find_yaw(pitch_deg, roll_deg, transformed_vector, mag_phase):
    def rotation_x(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    
    def rotation_y(angle):
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    def sin_theta_cross_product(vector_a, vector_b, signed=False, reference_axis=None):
        # Convert inputs to numpy arrays
        a = np.array(vector_a)
        b = np.array(vector_b)
        
        # Check if vectors are 3D
        if a.shape != (3,) or b.shape != (3,):
            raise ValueError("Vectors must be 3-dimensional")
        
        # Compute the cross product
        cross_product = np.cross(a, b)
        
        # Compute magnitudes
        mag_cross = np.linalg.norm(cross_product)
        mag_a = np.linalg.norm(a)
        mag_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if mag_a == 0 or mag_b == 0:
            raise ValueError("Vector magnitude cannot be zero")
        
        # Compute sin(theta)
        sin_theta = mag_cross / (mag_a * mag_b)
        
        # Ensure sin_theta is within [-1, 1] for numerical precision
        sin_theta = np.clip(sin_theta, -1, 1)
        
        # If signed is requested, determine the sign based on the cross product's direction
        if signed:
            if reference_axis is None:
                # Default to z-axis (e.g., [0, 0, 1]) for reference
                reference_axis = np.array([0, 0, 1])
            else:
                reference_axis = np.array(reference_axis)
                if reference_axis.shape != (3,):
                    raise ValueError("Reference axis must be 3-dimensional")
            
            # Compute the sign based on the dot product of cross product and reference axis
            sign = np.sign(np.dot(cross_product, reference_axis))
            sin_theta *= sign
        
        return sin_theta
    
    # Define unit vectors
    unit_z = np.array([0, 0, 1])
    CaplongAxis = np.array([1, 0, 0])
    
    # Convert pitch and roll to radians
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    
    # Compute the normal vector to the plane (capf_vertical)
    capf_vertical = rotation_y(-pitch) @ rotation_x(-roll) @ unit_z
    capf_vertical = capf_vertical / np.linalg.norm(capf_vertical)
    
    # Normalize input vector
    transformed_vector = transformed_vector / np.linalg.norm(transformed_vector)
    
    # Project vectors onto the plane perpendicular to capf_vertical
    acProj = transformed_vector - np.dot(transformed_vector, capf_vertical) * capf_vertical
    CaplongAxisProj = CaplongAxis - np.dot(CaplongAxis, capf_vertical) * capf_vertical

    # print(f"acProj (before norm): {acProj},transformed_vector: {transformed_vector}")
    
    # Normalize projections
    acProj = acProj / np.linalg.norm(acProj) if np.linalg.norm(acProj) != 0 else acProj
    CaplongAxisProj = CaplongAxisProj / np.linalg.norm(CaplongAxisProj) if np.linalg.norm(CaplongAxisProj) != 0 else CaplongAxisProj
    
    # Compute the smallest angle using dot product
    cos_theta = np.clip(np.dot(CaplongAxisProj, acProj), -1.0, 1.0)  # Clip to avoid numerical errors
    theta = np.arccos(cos_theta)  # Angle in radians
    theta_deg = np.degrees(theta)

    sine = sin_theta_cross_product(CaplongAxisProj, acProj, signed=True, reference_axis= -1*capf_vertical )
    # print(f"sin_theta_cross_product_value: {sine}")
    
    # # Define a 2D coordinate system in the plane
    # # Basis vector 1: CaplongAxisProj (normalized)
    # v = CaplongAxisProj
    # # Basis vector 2: Perpendicular to u and capf_vertical
    # u = np.cross(capf_vertical, v)
    # u = u / np.linalg.norm(u) if np.linalg.norm(u) != 0 else u

    # # Project vectors onto the 2D plane (u, v)
    # acProj_2d = np.array([np.dot(acProj, u), np.dot(acProj, v)])
    # CaplongAxisProj_2d = np.array([np.dot(CaplongAxisProj, u), np.dot(CaplongAxisProj, v)])
    
    # # Compute 2D cross product to determine direction
    # cross_2d = np.cross(CaplongAxisProj_2d, acProj_2d)


    # print(f"Yaw angle (degrees): {theta_deg}")
    
    # Determine counterclockwise angle
    if sine > 0:
        yaw = theta_deg  # acProj is counterclockwise from CaplongAxisProj
    elif sine < 0:
        yaw = 360 - theta_deg  # acProj is clockwise, so use 360 - theta
    else:
        yaw = 0 if theta_deg == 0 else 180  # Parallel or anti-parallel
    
    return yaw,sine,acProj,CaplongAxisProj,capf_vertical

def live_plot():
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    (ax1, ax2) = axs
    ax1.set_title('Raw Magnetometer (uT)')
    ax2.set_title('Lock-in AC Amplitude (uT)')
    ax2.set_xlabel('Time (s)')
    lines1 = [ax1.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines2 = [ax2.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    ax1.set_ylim(-10000, 10000)
    ax2.set_ylim(-1000, 1000)
    ax1.legend()
    ax2.legend()

    def init():
        for l in lines1 + lines2:
            l.set_data([], [])
        return lines1 + lines2

    def update(frame):
        t = np.array(plot_time)
        if t.size == 0:
            return lines1 + lines2
        t = t - t[0]
        data1 = [np.array(plot_mag_x), np.array(plot_mag_y), np.array(plot_mag_z)]
        data2 = [np.array(plot_amp_x), np.array(plot_amp_y), np.array(plot_amp_z)]
        for i in range(3):
            lines1[i].set_data(t, data1[i])
            lines2[i].set_data(t, data2[i])
        for ax in axs:
            ax.relim()
            ax.autoscale_view()
        return lines1 + lines2

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100)
    plt.tight_layout()
    plt.show()

# Initialize filters and data queue
kalman = KalmanFilter(process_noise=0.005, measurement_noise=0.1)
lockin = LockInAmplifier(target_freq=10.0, alpha=0.03)
data_queue = queue.Queue()
calibration_trigger = threading.Event()

def process_data(current_time, accel_g, gyro_dps, mag_uT, running_freq_hz):
    pitch_kalman, roll_kalman = kalman.update(accel_g, gyro_dps, current_time)
    if calibration_trigger.is_set():
        lockin.calibrate(mag_uT, current_time)
        calibration_trigger.clear()
    mag_amplitudes, mag_phases = lockin.update(mag_uT, current_time)
    return pitch_kalman, roll_kalman, mag_amplitudes, mag_phases

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
                pitch_kalman, roll_kalman, mag_amplitudes, mag_phases = future.result()

                loop_timestamps.append(loop_start_time)
                loop_freq_hz = 0.0
                if len(loop_timestamps) == 10:
                    elapsed_time = loop_start_time - loop_timestamps[0]
                    if elapsed_time > 0:
                        loop_freq_hz = 9 / elapsed_time

                yaw,sine,acProj,CaplongAxisProj,capf_vertical = find_yaw(pitch_kalman, roll_kalman, mag_amplitudes, mag_phases)
                print(
                    f"Yaw: {yaw:.2f}°\tSine: {sine:.2f}°\t "if yaw is not None else "Yaw: N/A\t    ",
                    f"acx: {acProj[0]:.2f}\tacy: {acProj[1]:.2f}\tacz: {acProj[2]:.2f}\t"
                    f"CaplongAxisProj x: {CaplongAxisProj[0]:.2f}\t  y: {CaplongAxisProj[1]:.2f}\t  z: {CaplongAxisProj[2]:.2f}\t"
                    f"Capf_vertical x: {capf_vertical[0]:.2f}\t y: {capf_vertical[1]:.2f}\t z: {capf_vertical[2]:.2f}\t",
                    f"Kalman Pitch: {pitch_kalman:.2f}°\t Roll: {roll_kalman:.2f}°,\t "
                    # f"AC Amplitudes (µT): [X: {mag_amplitudes[0]:.2f}, Y: {mag_amplitudes[1]:.2f}, Z: {mag_amplitudes[2]:.2f}]\t  "
                    f"AC Phases (°): [X: {mag_phases[0]:.2f}, Y: {mag_phases[1]:.2f}, Z: {mag_phases[2]:.2f}]\t"
                    # f"BLE Freq: {running_freq_hz:.2f} Hz, Loop Freq: {loop_freq_hz:.2f} Hz, Queue Size: {data_queue.qsize()}"
                )

                plot_time.append(current_time)
                plot_mag_x.append(mag_uT[0])
                plot_mag_y.append(mag_uT[1])
                plot_mag_z.append(mag_uT[2])
                plot_amp_x.append(mag_amplitudes[0])
                plot_amp_y.append(mag_amplitudes[1])
                plot_amp_z.append(mag_amplitudes[2])
                data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing thread error: {e}")

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
        accel_g[2] = -accel_g[2]
        gyro_dps[0] = -gyro_dps[0]
        gyro_dps[1] = -gyro_dps[1]
        temp = mag_uT[0]
        mag_uT[0] = mag_uT[2]
        mag_uT[2] = temp
        data_queue.put((current_time, accel_g, gyro_dps, mag_uT, running_freq_hz))

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
    print(f"Scanning for device with name: {device_name}...")
    devices = await BleakScanner.discover()
    target_device = None
    for device in devices:
        if device.name == device_name:
            target_device = device
            break
    if not target_device:
        print(f"Device {device_name} not found!")
        return
    print(f"Found device: {target_device.name} ({target_device.address})")
    async with BleakClient(target_device.address) as client:
        print(f"Connected to {target_device.name}")
        await client.start_notify(NUS_TX_UUID, notification_handler)
        print("Notifications enabled. Waiting for data...")
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await client.stop_notify(NUS_TX_UUID)
            print("Disconnected")

async def main():
    device_name = "Nordic_UART_Service"
    processing_thread_instance = threading.Thread(target=processing_thread, daemon=True)
    processing_thread_instance.start()
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    try:
        await connect_and_receive(device_name)
    except Exception as e:
        print(f"Error: {e}")

def start_asyncio():
    asyncio.run(main())

if __name__ == "__main__":
    asyncio_thread = threading.Thread(target=start_asyncio, daemon=True)
    asyncio_thread.start()
    live_plot()