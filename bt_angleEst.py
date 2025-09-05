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

def live_plot():
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    (ax1, ax2) = axs
    ax1.set_title('Raw Magnetometer (uT)')
    ax2.set_title('Lock-in AC Amplitude (uT)')
    ax2.set_xlabel('Time (s)')

    # Lines for each axis
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
        t = t - t[0]  # zero start time

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
        self.roll_unwrapped = 0.0  # Track unwrapped roll


    def update(self, accel_g, gyro_dps, current_time):
        ax, ay, az = accel_g
        gx, gy, gz = gyro_dps

        # print(f"Gyroscope (dps): \tgx: {gx:.2f}, \tgy: {gy:.2f}, \tgz: {gz:.2f} Accel (g): ax: {ax}, ay: {ay}, az: {az}")
        dt = current_time - self.last_time
        self.last_time = current_time

        # Predict
        F = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[dt, 0], [0, dt], [0, 0], [0, 0]])
        
        u = np.array([gx, gy])
        self.state = F @ self.state + B @ u
        self.P = F @ self.P @ F.T + self.Q

        #Adjust to take account into signage of accel data since taking square root of them would remove this info
        #this adjustment would allow the pitch and roll calculated have range of +/-180 degrees
        #Pitch
        if( az > 0 and ay < 0 ):
            extended_pitch = (180+np.arctan2(ay, np.sqrt(ax**2 + az**2)) * 180 / np.pi)
        elif( az > 0 and ay > 0 ):
            extended_pitch = (-180-np.arctan2(ay, np.sqrt(ax**2 + az**2)) * 180 / np.pi)
        else:
            extended_pitch = -np.arctan2(ay, np.sqrt(ax**2 + az**2)) * 180 / np.pi
        #Roll
        if( az > 0 and ax < 0 ):
            extended_roll = (-180-np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi)
        elif( az > 0 and ax > 0 ):
            extended_roll = (180-np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi)
        else:
            extended_roll = np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi

        # Measurement
        z = np.array([
            extended_pitch,
            extended_roll
        ])
        # Print accelerometer angles
        # print(f"Accelerometer angles (°): Pitch: {z[0]:.2f}, Roll: {z[1]:.2f}  Accelerometer (g): ax: {ax}, ay: {ay}, az: {az} | ")


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
        self.target_freq = target_freq  # 10 Hz
        self.alpha = alpha  # Low-pass filter coefficient
        self.x_in = [0.0, 0.0, 0.0]  # In-phase components (x, y, z)
        self.x_quad = [0.0, 0.0, 0.0]  # Quadrature components (x, y, z)
        self.last_time = time.time()

    def update(self, mag_uT, current_time):
        dt = current_time - self.last_time
        self.last_time = current_time

        # Generate reference signals
        t = current_time
        ref_in = np.sin(2 * np.pi * self.target_freq * t)  # In-phase (sine)
        ref_quad = np.cos(2 * np.pi * self.target_freq * t)  # Quadrature (cosine)

        # Process each axis
        amplitudes = []
        phase_angles = []  # Store phase angles for each axis
        for i in range(3):  # x, y, z
            # Multiply signal with reference
            prod_in = mag_uT[i] * ref_in
            prod_quad = mag_uT[i] * ref_quad

            # Low-pass filter (exponential moving average)
            self.x_in[i] = (1 - self.alpha) * self.x_in[i] + self.alpha * prod_in
            self.x_quad[i] = (1 - self.alpha) * self.x_quad[i] + self.alpha * prod_quad

            # Compute amplitude
            amplitude = np.sqrt(self.x_in[i]**2 + self.x_quad[i]**2)
            amplitudes.append(amplitude)

            # Compute phase angle in degrees relative to a reference 10Hz sine wave of 0 phase angle
            phase = np.degrees(np.arctan2(self.x_quad[i], self.x_in[i]))
            phase_angles.append(phase)

        return amplitudes, phase_angles  # Return [amp_x, amp_y, amp_z], [phase_x, phase_y, phase_z]
    
    
def find_yaw(pitch_deg, roll_deg, transformed_vector, mag_phase):
    """
    Compute the yaw angle given pitch (x-axis rotation), roll (y-axis rotation) in degrees,
    and the transformed unit vector, assuming ZYX intrinsic rotation.

    Parameters:
    - pitch_deg (float): Pitch angle in degrees (rotation about x-axis).
    - roll_deg (float): Roll angle in degrees (rotation about y-axis).
    - transformed_vector (np.ndarray): 3D vector after ZYX intrinsic rotation, [mx, my, mz].

    Returns:
    - float: Yaw angle in degrees, or None if the solution is invalid.
    """
    # Convert angles to radians
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    
    # Define rotation matrices
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
    
    # Ensure transformed vector is normalized
    transformed_vector = np.array(transformed_vector)
    norm = np.linalg.norm(transformed_vector)
    # if not np.isclose(norm, 1.0, rtol=1e-5):
    #     print(f"Warning: Transformed vector magnitude is {norm:.2f}. Normalizing.")
    transformed_vector = transformed_vector / norm
    
    # Compute combined pitch and roll rotation matrix (ZYX order: roll after pitch)
    R_pr = rotation_y(roll) @ rotation_x(pitch)
    
    # Compute inverse of R_pr (since R_pr is orthogonal, inverse = transpose)
    R_pr_inv = R_pr.T
    
    # Compute u = R_pr^-1 @ transformed_vector
    u = R_pr_inv @ transformed_vector
    # print(f"Intermediate vector u: {u}")
    
    # # u should be of the form [cos(yaw), sin(yaw), 0]
    # if not np.isclose(u[2], 0, atol=1e-5):
    #     print(f"Warning: Inconsistent input. z-component of u is {u[2]:.6f} (not zero).")
    #     return None
    
    # Extract cos(yaw) and sin(yaw)
    cos_yaw, sin_yaw = u[0], u[1]
    
    # # Check if u is a valid unit vector in xy-plane
    # if not np.isclose(cos_yaw**2 + sin_yaw**2, 1.0, rtol=1e-5):
    #     print(f"Warning: Invalid solution. Magnitude of u's xy components is {cos_yaw**2 + sin_yaw**2:.6f}.")
    #     return None
    
    # Compute yaw using atan2 and convert to degrees
    yaw = np.degrees(np.atan2(sin_yaw, cos_yaw))

    return yaw if mag_phase[1] > 0 else -yaw


# Initialize filters and data queue
kalman = KalmanFilter(process_noise=0.005, measurement_noise=0.1)
lockin = LockInAmplifier(target_freq=10.0, alpha=0.03)
data_queue = queue.Queue()



def process_data(current_time, accel_g, gyro_dps, mag_uT, running_freq_hz):
    """Process Kalman filter and lock-in amplifier calculations in a separate thread."""
    pitch_kalman, roll_kalman = kalman.update(accel_g, gyro_dps, current_time)
    mag_amplitudes, mag_phases = lockin.update(mag_uT, current_time)  # Unpack amplitudes and phases
    return pitch_kalman, roll_kalman, mag_amplitudes, mag_phases

def processing_thread():
    """Process IMU data from the queue, prioritizing the latest packet."""
    loop_timestamps = deque(maxlen=10)
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            try:
                loop_start_time = time.time()
                # Get the latest packet, discarding older ones
                current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get(timeout=1.0)
                while not data_queue.empty():
                    current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get_nowait()
                
                # Offload calculations to a separate thread
                future = executor.submit(process_data, current_time, accel_g, gyro_dps, mag_uT, running_freq_hz)
                pitch_kalman, roll_kalman, mag_amplitudes, mag_phases = future.result()

                # Calculate loop frequency
                loop_timestamps.append(loop_start_time)
                loop_freq_hz = 0.0
                if len(loop_timestamps) == 10:
                    elapsed_time = loop_start_time - loop_timestamps[0]
                    if elapsed_time > 0:
                        loop_freq_hz = 9 / elapsed_time

                # Calculate yaw angle
                yaw = find_yaw(pitch_kalman, roll_kalman, mag_amplitudes,mag_phases)

                # # # Print results with queue size
                print(f"Yaw: {yaw:.2f}°" if yaw is not None else "Yaw: No solution",
                      f"Kalman Pitch: {pitch_kalman:.2f}°, Roll: {roll_kalman:.2f}°, "
                      f"AC Amplitudes (µT): [X: {mag_amplitudes[0]:.2f}, Y: {mag_amplitudes[1]:.2f}, Z: {mag_amplitudes[2]:.2f}]"
                        f"  AC Phases (°): [X: {mag_phases[0]:.2f}, Y: {mag_phases[1]:.2f}, Z: {mag_phases[2]:.2f}] "
                      f"BLE Freq: {running_freq_hz:.2f} Hz, Loop Freq: {loop_freq_hz:.2f} Hz, Queue Size: {data_queue.qsize()}")                  
                
                # Update plot buffers
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
    """Handle incoming BLE notifications from NUS TX characteristic."""
    global last_time, frequency_hz

    # Record current timestamp
    current_time = time.time()
    timestamp_deque.append(current_time)

    # Calculate running average frequency for last 10 packets
    running_freq_hz = 0.0
    if len(timestamp_deque) == 10:
        elapsed_time = current_time - timestamp_deque[0]
        if elapsed_time > 0:
            running_freq_hz = 9 / elapsed_time

    # Process packet
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

        ## adjust for imu and HALL orientation on PCB

        accel_g[2] = -accel_g[2]

        gyro_dps[0] = -gyro_dps[0]
        gyro_dps[1] = -gyro_dps[1]
        gyro_dps[2] = gyro_dps[2]

        temp = mag_uT[0]
        temp1 = mag_uT[2]
        mag_uT[0] = mag_uT[1]
        mag_uT[1] = temp1
        mag_uT[2] = temp



        # print(f"Accel (g): ax: {accel_g[0]:.2f}, ay: {accel_g[1]:.2f}, az: {accel_g[2]:.2f} |"
        #        f" Gyro (°/s): gx: {gyro_dps[0]:.2f}, gy: {gyro_dps[1]:.2f}, gz: {gyro_dps[2]:.2f} |"
        #        f" Mag (µT): mx: {mag_uT[0]:.2f}, my: {mag_uT[1]:.2f}, mz: {mag_uT[2]:.2f}")

        # Queue data for processing
        data_queue.put((current_time, accel_g, gyro_dps, mag_uT, running_freq_hz))

async def connect_and_receive(device_name):
    """Scan for the nRF52820, connect, and receive IMU data."""
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
    # Start processing thread
    processing_thread_instance = threading.Thread(target=processing_thread, daemon=True)
    processing_thread_instance.start()
    try:
        await connect_and_receive(device_name)
    except Exception as e:
        print(f"Error: {e}")

def start_asyncio():
    asyncio.run(main())

if __name__ == "__main__":
    # Start asyncio BLE task in its own thread
    asyncio_thread = threading.Thread(target=start_asyncio, daemon=True)
    asyncio_thread.start()

    # Run live plotting in the main thread
    live_plot()