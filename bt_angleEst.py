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

# Add a deque for yaw angle running average
YAW_AVG_WINDOW = 10  # Number of samples for running average
yaw_deque = deque(maxlen=YAW_AVG_WINDOW)

# Kalman Filter class for pitch and roll
class KalmanFilter:
    def __init__(self, process_noise=0.005, measurement_noise=0.1):
        self.state = np.zeros(4)  # [pitch, roll, bias_x, bias_y]
        self.P = np.eye(4) * 0.1  # Error covariance
        self.Q = np.eye(4) * process_noise  # Process noise covariance
        self.R = np.eye(2) * measurement_noise  # Measurement noise covariance
        self.last_time = time.time()

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

        # Measurement
        z = np.array([
            np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi,
            np.arctan2(-ay, az) * 180 / np.pi
        ])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        return self.state[0], self.state[1]

class LockInAmplifier:
    def __init__(self, target_freq=10.0, alpha=0.01, amplitude_alpha=0.01):
        self.target_freq = target_freq  # 10 Hz
        self.alpha = alpha  # Low-pass filter coefficient for in-phase/quadrature
        self.amplitude_alpha = amplitude_alpha  # Low-pass filter coefficient for amplitudes
        self.x_in = [0.0, 0.0, 0.0]  # In-phase components (x, y, z)
        self.x_quad = [0.0, 0.0, 0.0]  # Quadrature components (x, y, z)
        self.filtered_amplitudes = [0.0, 0.0, 0.0]  # Filtered amplitudes (x, y, z)
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
        for i in range(3):  # x, y, z
            # Multiply signal with reference
            prod_in = mag_uT[i] * ref_in
            prod_quad = mag_uT[i] * ref_quad

            # Low-pass filter (exponential moving average) for in-phase and quadrature
            self.x_in[i] = (1 - self.alpha) * self.x_in[i] + self.alpha * prod_in
            self.x_quad[i] = (1 - self.alpha) * self.x_quad[i] + self.alpha * prod_quad

            # Compute raw amplitude
            amplitude = np.sqrt(self.x_in[i]**2 + self.x_quad[i]**2)
            amplitudes.append(amplitude)

            # Apply low-pass filter to amplitude
            self.filtered_amplitudes[i] = (1 - self.amplitude_alpha) * self.filtered_amplitudes[i] + self.amplitude_alpha * amplitude

        return amplitudes, self.filtered_amplitudes  # Return both raw and filtered amplitudes

# Initialize filters and data queue
kalman = KalmanFilter(process_noise=0.005, measurement_noise=0.1)
lockin = LockInAmplifier(target_freq=10.0, alpha=0.005, amplitude_alpha=0.03)
#Alpha: The low-pass filter coefficient (alpha=0.01) corresponds to a cutoff frequency suitable for a 100 Hz data rate. If amplitudes respond too slowly, increase alpha (e.g., to 0.05); if too noisy, decrease it (e.g., to 0.005).


data_queue = queue.Queue()

# Add deques for filtered amplitudes
plot_amp_x_filtered = deque(maxlen=PLOT_BUFFER_LEN)
plot_amp_y_filtered = deque(maxlen=PLOT_BUFFER_LEN)
plot_amp_z_filtered = deque(maxlen=PLOT_BUFFER_LEN)

def processing_thread():
    while True:
        try:
            # Get data from queue
            current_time, accel_g, gyro_dps, mag_uT, running_freq_hz = data_queue.get(timeout=1.0)
            
            # Compute Kalman angles
            pitch_kalman, roll_kalman = kalman.update(accel_g, gyro_dps, current_time)
            
            # Compute Lock-in amplitudes (raw and filtered)
            mag_amplitudes, mag_amplitudes_filtered = lockin.update(mag_uT, current_time)
            
            # Compute yaw angle using filtered amplitudes for better stability
            yaw_angle = np.arctan2(mag_amplitudes_filtered[1], mag_amplitudes_filtered[2] + 1e-6) * 180 / np.pi
            
            # Apply running average filter to yaw angle
            yaw_deque.append(yaw_angle)
            yaw_filtered = np.mean(yaw_deque) if yaw_deque else yaw_angle

            # Print results with raw and filtered amplitudes
            print(f"Yaw: {yaw_angle:.2f}°, Filtered Yaw: {yaw_filtered:.2f}°\t"
                  f"Kalman Pitch: {pitch_kalman:.2f}°, Roll: {roll_kalman:.2f}°\t"
                  f"Raw AC Amplitudes (µT): [X: {mag_amplitudes[0]:.2f}, Y: {mag_amplitudes[1]:.2f}, Z: {mag_amplitudes[2]:.2f}]\t"
                  f"Filtered AC Amplitudes (µT): [X: {mag_amplitudes_filtered[0]:.2f}, Y: {mag_amplitudes_filtered[1]:.2f}, Z: {mag_amplitudes_filtered[2]:.2f}]")

            # Save to file with raw and filtered amplitudes
            with open("imu_data.csv", "a") as f:
                f.write(f"{current_time},{accel_g[0]},{accel_g[1]},{accel_g[2]},"
                        f"{gyro_dps[0]},{gyro_dps[1]},{gyro_dps[2]},"
                        f"{mag_uT[0]},{mag_uT[1]},{mag_uT[2]},"
                        f"{running_freq_hz},{pitch_kalman},{roll_kalman},"
                        f"{mag_amplitudes[0]},{mag_amplitudes[1]},{mag_amplitudes[2]},"
                        f"{mag_amplitudes_filtered[0]},{mag_amplitudes_filtered[1]},{mag_amplitudes_filtered[2]},"
                        f"{yaw_angle},{yaw_filtered}\n")

            # Update plot buffers
            plot_time.append(current_time)
            plot_mag_x.append(mag_uT[0])
            plot_mag_y.append(mag_uT[1])
            plot_mag_z.append(mag_uT[2])
            plot_amp_x.append(mag_amplitudes[0])
            plot_amp_y.append(mag_amplitudes[1])
            plot_amp_z.append(mag_amplitudes[2])
            plot_amp_x_filtered.append(mag_amplitudes_filtered[0])
            plot_amp_y_filtered.append(mag_amplitudes_filtered[1])
            plot_amp_z_filtered.append(mag_amplitudes_filtered[2])

            data_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing thread error: {e}")

def live_plot():
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    (ax1, ax2) = axs
    ax1.set_title('Raw Magnetometer (uT)')
    ax2.set_title('Lock-in AC Amplitude (uT)')
    ax2.set_xlabel('Time (s)')

    # Lines for each axis (raw and filtered)
    lines1 = [ax1.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    lines2 = [ax2.plot([], [], label=f'{lbl} Raw', color=col)[0] for lbl, col in zip(['X', 'Y', 'Z'], ['b', 'g', 'r'])]
    lines2_filtered = [ax2.plot([], [], label=f'{lbl} Filtered', color=col, linestyle='--')[0] for lbl, col in zip(['X', 'Y', 'Z'], ['b', 'g', 'r'])]

    raw_limit = 20000
    ac_limit = 500
    ax1.set_ylim(-raw_limit, raw_limit)
    ax2.set_ylim(-ac_limit, ac_limit)

    ax1.legend()
    ax2.legend()

    def init():
        for l in lines1 + lines2 + lines2_filtered:
            l.set_data([], [])
        return lines1 + lines2 + lines2_filtered

    def update(frame):
        t = np.array(plot_time)
        if t.size == 0:
            return lines1 + lines2 + lines2_filtered
        t = t - t[0]  # Zero start time

        data1 = [np.array(plot_mag_x), np.array(plot_mag_y), np.array(plot_mag_z)]
        data2 = [np.array(plot_amp_x), np.array(plot_amp_y), np.array(plot_amp_z)]
        data2_filtered = [np.array(plot_amp_x_filtered), np.array(plot_amp_y_filtered), np.array(plot_amp_z_filtered)]

        for i in range(3):
            lines1[i].set_data(t, data1[i])
            lines2[i].set_data(t, data2[i])
            lines2_filtered[i].set_data(t, data2_filtered[i])
        
        for ax in axs:
            ax.relim()
            ax.autoscale_view(scaley=False)  # Preserve y-limits

        return lines1 + lines2 + lines2_filtered

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100)
    plt.tight_layout()
    plt.show()

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
    # Initialize CSV file with header as before...

    # Start data processing thread
    processing_thread_instance = threading.Thread(target=processing_thread, daemon=True)
    processing_thread_instance.start()

    # Start asyncio BLE task in its own thread
    asyncio_thread = threading.Thread(target=start_asyncio, daemon=True)
    asyncio_thread.start()

    # Now, in the main thread, do the plotting
    live_plot()