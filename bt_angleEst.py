import asyncio
from bleak import BleakScanner, BleakClient
import struct
import time
import numpy as np

# Define Nordic UART Service (NUS) UUIDs
NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # TX from device perspective (PC writes to this)
NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # RX from device perspective (PC receives from this)

# Callback to handle incoming data
def notification_handler(sender, data):
    """Handle incoming BLE notifications from NUS TX characteristic."""
    # print(f"Time: {time.time():.2f} Received {len(data)} bytes")

    # Check for expected packet format (0x12, 0x34, 12 bytes accel/gyro, 6 bytes mag)
    if len(data) >= 20 and data[0] == 0x12 and data[1] == 0x34:
        # Extract accelerometer (6 bytes), gyroscope (6 bytes), magnetometer (6 bytes)
        gyro_data = data[2:8]  # x, y, z (2 bytes each, 16-bit signed)
        accel_data = data[8:14]  # x, y, z (2 bytes each, 16-bit signed)
        mag_data = data[15:21]  # x, y, z (2 bytes each, 16-bit signed)

        # Convert to integers (assuming little-endian, signed 16-bit)
        accel = struct.unpack("<hhh", accel_data)  # 3 signed shorts
        gyro = struct.unpack("<hhh", gyro_data)
        mag = struct.unpack(">hhh", mag_data)
        
        # Convert to physical units (example scales, adjust based on LSM6DSO/AK09973D settings)
        accel_g = [x * 2.0 / 32768 for x in accel]  # ±2g range, 16-bit
        gyro_dps = [x * 245.0 / 32768 for x in gyro]  # ±245 dps range
        mag_uT = [x * 1.1 for x in mag]  # ±36mT range (AK09973D High sensitivity mode)

        #        Format each value to 2 decimal places
        print(f"Time: {time.time():.2f}\tReceived {len(data)} bytes\tAccelerometer (g):[{accel_g[0]:.2f}, {accel_g[1]:.2f}, {accel_g[2]:.2f}]\t"
              f"Gyroscope (dps):[{gyro_dps[0]:.2f}, {gyro_dps[1]:.2f}, {gyro_dps[2]:.2f}]\t"
              f"Magnetometer (µT):[{mag_uT[0]:.2f}, {mag_uT[1]:.2f}, {mag_uT[2]:.2f}]")


        # Optionally save to file or process for MCE orientation estimation
        with open("imu_data.csv", "a") as f:
            timestamp = time.time()
            f.write(f"{timestamp},{accel_g[0]},{accel_g[1]},{accel_g[2]},"
                    f"{gyro_dps[0]},{gyro_dps[1]},{gyro_dps[2]},"
                    f"{mag_uT[0]},{mag_uT[1]},{mag_uT[2]}\n")

async def connect_and_receive(device_name):
    """Scan for the nRF52820, connect, and receive IMU data."""
    print(f"Scanning for device with name: {device_name}...")

    # Scan for devices
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

        # Enable notifications on NUS TX characteristic
        await client.start_notify(NUS_TX_UUID, notification_handler)
        print("Notifications enabled. Waiting for data...")

        # Keep the connection alive
        try:
            while True:
                await asyncio.sleep(1)  # Keep connection alive
        except asyncio.CancelledError:
            await client.stop_notify(NUS_TX_UUID)
            print("Disconnected")

async def main():
    # Replace with your device's advertised name (from CONFIG_BT_DEVICE_NAME)
    device_name = "Nordic_UART_Service"  # Update this based on your firmware setting

    try:
        await connect_and_receive(device_name)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())