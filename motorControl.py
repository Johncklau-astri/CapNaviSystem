import threading
import time
import select
import sys
import msvcrt
import numpy as np
from scipy.spatial.transform import Rotation as R
import socket
import json
import struct
from collections import defaultdict
from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_frame import GsUsbFrame
from gs_usb.constants import (
    CAN_EFF_FLAG,
    CAN_ERR_FLAG,
    CAN_RTR_FLAG,
)
class Motor:
    def __init__(self, name, node_id):
        self.name = name
        self.node_id = node_id
        self.current_position = 0
        self.last_position = 0
        self.current_position_real_unit = 0.0
        self.last_position_real_unit = 0.0
        self.statusword = 0
        self.status = 'Uninitialized'
        self.last_update_time = None
        self.operational = False
        self.moveCommand = False
        self.is_stopped = False
        self.offset = 0.0

    def update_moveCommand(self, hasMoveCommand):
        self.moveCommand = hasMoveCommand


    def update_position_and_status(self, position, statusword):
        """Updates internal state from CAN TPDO data"""
        self.current_position = position
        self.statusword = statusword
        self.last_update_time = time.time()
        
        if self.name == 'rotary':
            raw_unit = self.current_position / (12.9704 * 10 * 10)
            # print(f"raw unit type: {type(raw_unit)}")
            self.current_position_real_unit = round(raw_unit, 2)
            # print(f"Rotary current_position_real_unit: {self.current_position_real_unit} last pos: {self.last_position_real_unit}")
            
        if(self.name == 'vert1' or self.name == 'vert2'):
            self.current_position_real_unit = self.current_position / 4000
        
        if(self.name == 'hori1' or self.name == 'hori2'):
            self.current_position_real_unit = self.current_position / 8000

        if(self.name == 'pitch'):
            self.current_position_real_unit = self.current_position / (2.04 * (4000/360))

        self.is_stopped = self.current_position_real_unit == self.last_position_real_unit

        self.last_position_real_unit = self.current_position_real_unit
    
    def __repr__(self):
        return (f"Motor({self.name}, ID: {self.node_id}, "
                f"Current Pos: {self.current_position}, "
                f"Statusword: 0x{self.statusword:04X}, "
                f"Status: {self.status})")
    

def magnet_offset(Motor):
    for motor in Motor:
        motor.offset = motor.current_position_real_unit
        print(f"Motor {motor.name} offset set to {motor.offset:.2f}")

import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_magnet_pose_and_sendMagAbsPos(Motor):
    max_r = 0.33
    r = max_r - (Motor[3].current_position_real_unit/100 - Motor[3].offset)
    
    # 1. Calculate your angles in degrees
    roll = 0 
    pitch = -1 * (Motor[5].current_position_real_unit - Motor[5].offset)
    yaw = -1 * (Motor[0].current_position_real_unit - Motor[0].offset)
    
    Z = (Motor[1].current_position_real_unit - Motor[1].offset)/100
    X = -1* r * np.sin(np.radians(yaw))
    Y = 1 * r * np.cos(np.radians(yaw))-max_r


    # 2. Convert to Quaternion using SciPy
    # 'zyx' specifies the axes: Yaw(Z), Roll(Y), Pitch(X)
    # degrees=True allows us to pass your variables directly
    euler_angles = [yaw, pitch, roll] 
    rotation = R.from_euler('ZXY', euler_angles, degrees=True)
    
    # SciPy outputs quaternions in [x, y, z, w] format
    quat = rotation.as_quat() 
    # print(f"Roll: {roll:.3f}, Pitch: {pitch:.3f}, Yaw: {yaw:.3f}, X: {X:.3f}, Y: {Y:.3f}, Z: {Z:.3f}, r: {r:.3f}. q: [{quat[3]:.4f}, {quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}]")

    encoder_timestamps = [
        motor.last_update_time for motor in Motor
        if motor.last_update_time is not None
    ]
    pose_timestamp = max(encoder_timestamps) if encoder_timestamps else time.time()
    
    # print(f"Magnet Pos- X: {X:.3f}, Y: {Y:.3f}, Z: {Z:.3f} q: [{quat[3]:.4f}, {quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}] r: {r:.3f}")

    # Prepare data to send
    data_to_send = {
        "x": float(f"{X:.3f}"),  # Convert to float for JSON serialization
        "y": float(f"{Y:.3f}"),
        "z": float(f"{Z:.3f}"),
        "qw": float(f"{quat[3]:.4f}"),
        "qx": float(f"{quat[0]:.4f}"),
        "qy": float(f"{quat[1]:.4f}"),
        "qz": float(f"{quat[2]:.4f}"),
        "timestamp": float(pose_timestamp),
        "record_data":False
    }
    # print(f"Sending Magnet Pos- X: {X:.3f}, Y: {Y:.3f}, Z: {Z:.3f} q: [{quat[3]:.4f}, {quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}]")

    # Send data to arduino_magnetic_reader.py
    send_data_to_arduino(data_dict=data_to_send)
    
    return X, Y, Z, quat

        
# gs_usb general also can import from gs_usb_structures.py
GS_USB_ECHO_ID = 0
GS_USB_NONE_ECHO_ID = 0xFFFFFFFF
coveredCanConverterId = "002000325641570920373632"
no_coverCanConverterId = "003B00425343571020363330"
canBusBitRate = 500000

waitTime = 0.01

rotary = 1
vert1 = 2
vert2 = 3
hori1 = 4
hori2 = 5
pitch = 6


# Inital motor positions

rotaryDisplacement = 0 # orignal value: 25
vertDisplacement = 0
horiDisplacement = 0

topsheetHalfWidth = 0
topsheetToCircleCenter = 0
magnetMinHeight = 0
maxHoriToCircleCenter = 0

pitchAngle = 0 # orignal value: -45
yawAngle = rotaryDisplacement
rollAngle = 0



motors = [
    Motor('rotary', 1),
    Motor('vert1', 2),
    Motor('vert2', 3),
    Motor('hori1', 4),
    Motor('hori2', 5),
    Motor('pitch', 6)
]

threads = []

pitch_up = [0,0,0,30]
pitch_down = [0,0,0,-30]
half_pitch_up = [0,0,0,15]
half_pitch_down = [0,0,0,-15]

vert_start_offset = 12
hori_start_offset = 10
layer_z_sep = -2

rev_vert_start_offset = 15
rev_hori_start_offset = 10
rev_movement_sequence = [
[       0,         rev_vert_start_offset,     rev_hori_start_offset,   0                ],   # 0
##First Revolution
[360,0,0,0],
##Height lowering
[0,-3,0,0],
## Second Revolution
[-360,0,0,0],
## Height lowering after second revolution
[0,-3,0,0],
## Third Revolution
[360,0,0,0],
## Height lowering after third revolution
[0,-3,0,0],
## Fourth Revolution
[-360,0,0,0]

]

p2p_movement_sequence= [
## Rotary motor movement        vert motor movement     hori motor movement         pitch motor movement
[       0,         vert_start_offset,     hori_start_offset,   0                ],   # 0
## Move to 1st layer starting position
[       -140,         0,     0,   0                ],   # 0
[       0,         0,     0,   0                ],   # 0

half_pitch_down,
pitch_up,
pitch_down,
[       20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0

pitch_up,
pitch_down,
[       20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
## 1st layer finished, move to 2nd layer starting position
[       0,         layer_z_sep,     0,   0                ],   # 0
pitch_up,
pitch_down,

## start 2nd layer scanning
[       -20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
## 2nd layer finished, move to 3rd layer starting position
[       0,         layer_z_sep,     0,   0                ],   # 0
pitch_up,
pitch_down,
##starting 3rd layer scan
[       20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,

## 3rd layer finished, move to 4th layer starting position
[       0,         layer_z_sep,     0,   0                ],   # 0
pitch_up,
pitch_down,
##starting 4th layer scan
[       -20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       -20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,

## 4th layer finished, move to 5th layer starting position
[       0,         layer_z_sep,     0,   0                ],   # 0
pitch_up,
pitch_down,
##starting 5th layer scan
[       20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       30,         0,     0,   0                ],   # 0
pitch_up,
pitch_down,
[       20,         0,     0,   0                ],   # 0
pitch_up,
pitch_down
]


can_lock = threading.Lock()

# below macro is for gs_usb 0.3.0 and above
GS_CAN_MODE_NORMAL = 0
GS_CAN_MODE_LISTEN_ONLY = (1 << 0)
GS_CAN_MODE_LOOP_BACK = (1 << 1)
# GS_CAN_MODE_TRIPLE_SAMPLE = (1 << 2)
GS_CAN_MODE_ONE_SHOT = (1 << 3)
GS_CAN_MODE_HW_TIMESTAMP = (1 << 4)






def vert_sync_monitoring(dev, motors_list, tolerance=2000, interval=0.05, stop_event=None):
    """
    Background monitor thread for vert1 and vert2 position sync.
    Monitors position updates from TPDO messages and checks sync.
    If the two vertical motors drift apart by more than tolerance, send emergency stop.
    """
    v1 = next((m for m in motors_list if m.name == 'vert1'), None)
    v2 = next((m for m in motors_list if m.name == 'vert2'), None)

    if not v1 or not v2:
        print("[Vert Sync] Error: Vertical motors not found.")
        return

    print(f"[Vert Sync] Background monitoring started (Tolerance: {tolerance})")

    while not stop_event.is_set():
        try:
            # Check sync if both motors have recent updates
            if v1.last_update_time and v2.last_update_time:
                time_diff = abs(time.time() - v1.last_update_time)
                time_diff2 = abs(time.time() - v2.last_update_time)
                if time_diff < 0.5 and time_diff2 < 0.5:  # Recent updates
                    diff = abs(v1.current_position - v2.current_position)
                    if diff > tolerance and v1.current_position != 0 and v2.current_position != 0:  # Avoid false trigger at zeroing
                        print(f"\n[!!!] VERTICAL SYNC ERROR: V1={v1.current_position}, V2={v2.current_position}, Diff={diff}")
                        emergency_stop_all(dev, motors_list)
                        break

            time.sleep(interval)  # Check at specified interval

        except Exception as e:
            print(f"[Vert Sync Monitor] Error: {e}")
            time.sleep(0.01)

    print("[Vert Sync] Monitoring stopped.")


def scan_for_device(target_serial):
    """
    Scans for a GsUsb device with the specified serial number.
    Returns True and the device if found, False and None if not.
    """
    devs = GsUsb.scan()
    if len(devs) == 0:
        print("No gs_usb devices found")
        return False, None
    else:
        print(f"{len(devs)} GsUsb devices found")
    
    tmpScanGsUsb = GsUsb(1)
    for dev in devs:
        # Check if the device's serial number matches the target
        if hasattr(dev, 'serial_number') and dev.serial_number == coveredCanConverterId:
            tmpScanGsUsb = dev
            print(f"Covered Converter Found")
        elif hasattr(dev, 'serial_number') and dev.serial_number == no_coverCanConverterId:
            print(f"No-cover Found")
        
    if hasattr(tmpScanGsUsb, 'serial_number') and tmpScanGsUsb.serial_number == coveredCanConverterId:
        print(f"""
###################################
Covered Converter Found and Assigned
###################################
        """)
        return True, tmpScanGsUsb
    else:
        print(f"Can't Find Target Converter")
        return False, None
    
    return False, None

def send_data_to_arduino(host='localhost', port=12345, data_dict=None):
    """
    Send data to arduino_magnetic_reader.py via TCP socket.
    
    Args:
        host (str): Server hostname (default: localhost)
        port (int): Server port (default: 12345)
        data_dict (dict): Data to send (e.g., {"x": 123.45, "y": 67.89, ...})
    
    Returns:
        bool: True if sent successfully, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.01)  # 1-second timeout
            sock.connect((host, port))
            # Convert dict to JSON string and encode
            data_str = json.dumps(data_dict)
            data_bytes = data_str.encode('utf-8')
            # Prepend 4-byte length header
            length = len(data_bytes)
            length_bytes = struct.pack('>I', length)  # Big-endian unsigned int
            # Send length header + data
            sock.sendall(length_bytes + data_bytes)
            return True
    except (socket.timeout, socket.error) as e:
        # print(f"Failed to send data to Arduino reader: {e}")
        return False

def vertPreMove(dev, motors):
    setUpVertMotorMove(dev, vert2, 0.05, 50, 100, 100)
    launchMoveCommand(dev, motors)
    time.sleep(0.2)  # Wait for the move to complete
    setUpVertMotorMove(dev, vert2, -0.05, 50, 100, 100)
    launchMoveCommand(dev, motors)
    time.sleep(0.2)  # Wait for the move to complete
    setUpVertMotorMove(dev, vert1, 0.05, 50, 100, 100)
    launchMoveCommand(dev, motors)
    time.sleep(0.2)  # Wait for the move to complete
    setUpVertMotorMove(dev, vert1, -0.05, 50, 100, 100)
    launchMoveCommand(dev, motors)

def automated_movement(dev,motors): 
    global worldX, worldY, worldZ # Assume actual_pos tracks motor positions
    global rollAngle, pitchAngle, yawAngle
    interval = 1    ## in cm
    x_number = 7   ## in quantity
    y_number = 8    ## in quantity
    z_number = 20   ## in quantity
    
    pitch_enable = True
    pitch_angle = 20
    current_pitch_low = False
    current_pitch_0 = True
    current_pitch_high = False

    data_to_send = {
        "record_data": True
    }

    speed = 65
    pitch_speed = 30
    pitch_accel = 40
    acc= 250
    dec = 250
    print("Magnet should be at middle height of test range, the also the closest point to capsule")

    hori_timeout = 0.6 ##1.3
    vert_timeout = 0.6   ##0.5
    send_timeout = 0
    pitch_timeout = 0.7

    for x in range(0,x_number):
        # # Returning pitch to zero
        # if (x != 0):
        #     setUpPitchMotorMove(dev, pitch, 0, speed, acc, dec)
        #
        if( pitch_enable):
            if(current_pitch_0):
                setUpPitchMotorMove(dev, pitch, pitch_angle, pitch_speed, pitch_accel, pitch_accel)
                launchMoveCommand(dev, motors)
                current_pitch_0 = False
                current_pitch_high = True
                current_pitch_low = False     
        if( x == 0):    
            ## Start at lowest point
            setUpHoriMotorMove(dev, hori1, -(y_number-1)*interval, 30, acc, dec)
            setUpHoriMotorMove(dev, hori2, -(y_number-1)*interval, 30, acc, dec)            
            setUpVertMotorMove(dev, vert1, -z_number*interval/2, 30, acc, dec)
            setUpVertMotorMove(dev, vert2, -z_number*interval/2, 30, acc, dec)
            print("x=0")
            print("Press S to start the automated movement at this X-plane")
        elif ( x != 0 and y_number % 2 == 0):
            setUpHoriMotorMove(dev, hori1, -(y_number-1)*interval, speed, acc, dec)
            setUpHoriMotorMove(dev, hori2, -(y_number-1)*interval, speed, acc, dec)
            print("x!=0 and y%2==0")
            print("Press S to start the automated movement at this X-plane")
        elif ( x != 0 and y_number % 2 != 0):
            setUpHoriMotorMove(dev, hori1, -(y_number-1)*interval, speed, acc, dec)
            setUpHoriMotorMove(dev, hori2, -(y_number-1)*interval, speed, acc, dec)
            setUpVertMotorMove(dev, vert1, -z_number*interval, 30, acc, dec)
            setUpVertMotorMove(dev, vert2, -z_number*interval, 30, acc, dec)
            print("x!=0 and y%2!=0")
            print("Press S to start the automated movement at this X-plane")
        launchMoveCommand(dev, motors)

        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 's':
                    print(f"Starting movement for x iteration ={x}")
                    break
                elif key == 'q':
                    print("\nEmergency stop triggered.")
                    for motor in motors:
                        quick_stop_frame = GsUsbFrame(
                            can_id=0x600 + motor.node_id,
                            data=bytes.fromhex("2B4060000B000000")  # Quick Stop
                        )
                        try:
                            dev.send(quick_stop_frame)
                            print(f"Emergency stop: Quick Stop sent to Motor {motor.node_id} ({motor.name})")
                        except Exception as e:
                            print(f"Failed to send Quick Stop to Motor {motor.node_id}: {e}")
                        reset_frame = GsUsbFrame(
                            can_id=0x600 + motor.node_id,
                            data=bytes.fromhex("2B40600006000000")  # Switch On Disabled
                        )
                        try:
                            dev.send(reset_frame)
                            print(f"Reset: Motor {motor.node_id} ({motor.name}) set to Switch On Disabled")
                        except Exception as e:
                            print(f"Failed to reset Motor {motor.node_id}: {e}")
                    print("Exiting...")
                    return
            time.sleep(0.1)  # Prevent CPU overuse



        for y in range(0, y_number):
            if(y != 0):
                setUpHoriMotorMove(dev, hori1, interval, speed, acc, dec)
                setUpHoriMotorMove(dev, hori2, interval, speed, acc, dec)
                launchMoveCommand(dev, motors)
                time.sleep(hori_timeout)  # Wait for the move to complete
            for z in range(0, z_number+1):
                if(z != 0):
                    if( y % 2 == 0):
                        setUpVertMotorMove(dev, vert1, interval, speed, acc, dec)
                        setUpVertMotorMove(dev, vert2, interval, speed, acc, dec)
                        launchMoveCommand(dev, motors)
                        time.sleep(vert_timeout)  # Wait f  or the move to complete        

                    else:
                        setUpVertMotorMove(dev, vert1, -interval, speed, acc, dec)
                        setUpVertMotorMove(dev, vert2, -interval, speed, acc, dec)
                        launchMoveCommand(dev, motors)
                        time.sleep(vert_timeout)  # Wait for the move to complete        

                if( pitch_enable):

                    send_data_to_arduino(data_dict=data_to_send)
                    time.sleep(send_timeout)  # Brief pause

                    if(current_pitch_low):
                        for i in range(0,2):
                            setUpPitchMotorMove(dev, pitch, pitch_angle, pitch_speed, pitch_accel, pitch_accel)
                            launchMoveCommand(dev, motors)
                            time.sleep(pitch_timeout)
                            send_data_to_arduino(data_dict=data_to_send)
                            time.sleep(send_timeout)  # Brief pause
                            print("Sent signal")
     
                        current_pitch_low = False
                        current_pitch_0 = False
                        current_pitch_high = True                       

                    elif(current_pitch_high):
                        for i in range(0,2):
                            setUpPitchMotorMove(dev, pitch, -pitch_angle, pitch_speed, pitch_accel, pitch_accel)
                            launchMoveCommand(dev, motors)
                            time.sleep(pitch_timeout)
                            send_data_to_arduino(data_dict=data_to_send)
                            time.sleep(send_timeout)  # Brief pause
                            print("Sent signal")
     
                        current_pitch_high = False
                        current_pitch_0 = False
                        current_pitch_low = True

                # q = euler_to_quaternion(0, 0, 0) ## dummy value as they are not used in optimization because its the aprilTag orientation that matters
                # worldX =0
                # worldY = 0
                # worldZ = 0
                # # Send TCP signal with record_data=True
                # # Prepare data to send

                # send_data_to_arduino(data_dict=data_to_send)
                # time.sleep(send_timeout)  # Brief pause
                # print(f"x: {x}, y: {y}, z: {z}")
                
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 'q':
                        print("\nEmergency stop triggered.")
                        # Stop the listener thread before quitting
                        for motor in motors:
                            quick_stop_frame = GsUsbFrame(
                                can_id=0x600 + motor.node_id,
                                data=bytes.fromhex("2B4060000B000000")  # Quick Stop
                            )
                            try:
                                dev.send(quick_stop_frame)
                                print(f"Emergency stop: Quick Stop sent to Motor {motor.node_id} ({motor.name})")
                            except Exception as e:
                                print(f"Failed to send Quick Stop to Motor {motor.node_id}: {e}")
                        
                        # Reset to Switch On Disabled state
                        for motor in motors:
                            reset_frame = GsUsbFrame(
                                can_id=0x600 + motor.node_id,
                                data=bytes.fromhex("2B40600006000000")  # Controlword 0x0006: Switch On Disabled
                            )
                            try:
                                dev.send(reset_frame)
                                print(f"Reset: Motor {motor.node_id} ({motor.name}) set to Switch On Disabled")
                            except Exception as e:
                                print(f"Failed to reset Motor {motor.node_id}: {e}")
                                
                        print("Exiting...")
            else:
                continue
            break
    print("Automated movement completed.")



def rotary_zeroing(dev):
    motor_id = rotary
    
    #activating special function
    special_function_frame = GsUsbFrame(
        can_id=0x600 + motor_id,
        data=bytes.fromhex("2340320101000000")
    )
    dev.send(special_function_frame)
    time.sleep(waitTime)

    motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600006000000"))
    dev.send(motor_init_frame)
    print(f"Motor {motor_id} shutdown ")
    time.sleep(waitTime)

    release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600007000000"))
    dev.send(release_brake_frame)
    print(f"Motor {motor_id} ready switch on")
    time.sleep(waitTime)


    zeroing_mode_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2F60600006000000"))
    dev.send(zeroing_mode_frame)
    print(f"Motor {motor_id} set to zeroing mode")
    time.sleep(waitTime)

    cw_mode_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2F98600011000000"))
    dev.send(cw_mode_frame)
    print(f"Motor {motor_id} set to CW mode")
    time.sleep(waitTime)

    zeroing_acc_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("239A6000E8030000"))
    dev.send(zeroing_acc_frame)
    print(f"Motor {motor_id} set zeroing acceleration")
    time.sleep(waitTime)

    zeroing_speed_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2399600135000000"))
    dev.send(zeroing_speed_frame)
    print(f"Motor {motor_id} set zeroing speed")
    time.sleep(waitTime)

    start_zeroing_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060001F000000"))
    dev.send(start_zeroing_frame)
    print(f"Motor {motor_id} zeroing started")


def disable_rotary_tpdo(dev):

        motor_id = rotary
    # Disable TPDO mapping temporarily
     # Disable TPDO1
        disable_tpdo1_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("2F00180200000000")  # Set number of mapped objects to 0
        )
        dev.send(disable_tpdo1_frame)
        time.sleep(waitTime)
        # Disable TPDO1
        disable_tpdo1_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("2F01180200000000")  # Set number of mapped objects to 0
        )
        dev.send(disable_tpdo1_frame)
        time.sleep(waitTime)

           # Disable TPDO1
        disable_tpdo1_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("2F02180200000000")  # Set number of mapped objects to 0
        )
        dev.send(disable_tpdo1_frame)
        time.sleep(waitTime)
        
           # Disable TPDO1
        disable_tpdo1_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("2F03180200000000")  # Set number of mapped objects to 0
        )
        dev.send(disable_tpdo1_frame)
        time.sleep(waitTime)

def disable_stepper_28XTpdo(dev):
    for motor_id in [vert1, vert2, hori1, hori2, pitch]:
        # Disable TPDO1
        disable_tpdo1_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("2301180183020080")  # Set number of mapped objects to 0
        )
        dev.send(disable_tpdo1_frame)
        time.sleep(waitTime)

        


def tpdo_setup(motor_id, dev):
    """
    Configures TPDO1 to transmit Actual Position (0x6064) and Statusword (0x6041).
    
    Args:
        motor_id (int): CANopen Node ID of the motor
        dev: USB-CAN device object
    """
    try:
        deactivate_tpdo_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
             data=bytes.fromhex("2300180181010080")  # Set number of mapped objects to 0
        )
        dev.send(deactivate_tpdo_frame)
        time.sleep(waitTime)

        deactivate_tpdoNumber_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
             data=bytes.fromhex("2F001A0000000000")  # Set number of mapped objects to 0
        )
        dev.send(deactivate_tpdoNumber_frame)
        time.sleep(waitTime)


        # Set synchronous transmission type (0x01 = synchronous)
        sync_tpdo_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("2F00180201000000")
        )
        dev.send(sync_tpdo_frame)
        time.sleep(waitTime)



        # Map Actual Position (0x6064, 32-bit)
        map_pos_frame1 = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("23001A0220006460")
        )
        dev.send(map_pos_frame1)
        time.sleep(waitTime)

        # Map Statusword (0x6041, 16-bit)
        map_status_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("23001A0110004160")  # 16-bit statusword
        )
        dev.send(map_status_frame)
        time.sleep(waitTime)

        # Enable TPDO mapping with 2 objects
        enable_mapping_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("2F001A0002000000")  # 2 mapped objects
        )
        dev.send(enable_mapping_frame)
        time.sleep(waitTime)

                # Enable TPDO mapping with 2 objects
        enable_tpdo_frame = GsUsbFrame(
            can_id=0x600 + motor_id,
            data=bytes.fromhex("2300180181010000")  # 2 mapped objects
        )
        dev.send(enable_tpdo_frame)
        time.sleep(waitTime)

        print(f"TPDO setup completed for Motor {motor_id}: Actual Position (0x6064) and Statusword (0x6041)")
    except Exception as e:
        print(f"Error setting up TPDO for Motor {motor_id}: {e}")



# def tpdo_listener(dev, stop_event, motors_list):
#     """
#     Listens for TPDO messages and updates motor positions and statuswords.
    
#     Args:
#         dev: USB-CAN device object
#         stop_event: Threading event to stop the listener
#         motors_list (list): List of Motor objects to update
#     """
#     iframe = GsUsbFrame()
#     while not stop_event.is_set():
#         try:
#             if dev.read(iframe, 100):  # 100ms timeout
#                 if 0x180 <= iframe.can_id <= 0x1FF:  # TPDO1 range
#                     node_id = iframe.can_id - 0x180
#                     for motor in motors_list:
#                         if motor.node_id == node_id and len(iframe.data) >= 6:
#                             current_pos = int.from_bytes(iframe.data[0:4], 'little', signed=True)
#                             statusword = int.from_bytes(iframe.data[4:6], 'little', signed=False)
#                             motor.update_position_and_status(current_pos, statusword)
#                             # Optional: Print for debugging
#                             # print(f"Motor {motor.name}: Pos={current_pos}, Status=0x{statusword:04X}")
#         except Exception as e:
#             print(f"TPDO listener error: {e}")
#             time.sleep(0.1)

def zeroing_motor(motor_id, dev, motors_list, timeout=30.0):
    """
    Zeroes a motor and monitors its statusword until complete, with estop capability.
    
    Args:
        motor_id (int): CANopen Node ID of the motor
        dev: USB-CAN device object
        motors_list (list): List of Motor objects
        timeout (float): Max time (seconds) to wait for zeroing (default: 30s)

    """
    if(motor_id == rotary):
        rotary_zeroing(dev)
        return
    motor = next((m for m in motors_list if m.node_id == motor_id), None)
    if not motor:
        print(f"Error: Motor with ID {motor_id} not found in motors list.")
        return

    print(f"\nStarting zeroing process for Motor {motor_id} ({motor.name})...")

    # Initializing and enabling motor
    try:
        motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000100"))
        dev.send(motor_init_frame)
        print(f"Motor {motor_id} initialized")
        time.sleep(waitTime)

        release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000300"))
        dev.send(release_brake_frame)
        print(f"Motor {motor_id} brake released")
        time.sleep(waitTime)

        enable_motor_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000F00"))
        dev.send(enable_motor_frame)
        print(f"Motor {motor_id} enabled")
        time.sleep(waitTime)

        zeroing_mode_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2F60600006000000"))
        dev.send(zeroing_mode_frame)
        print(f"Motor {motor_id} set to zeroing mode")
        time.sleep(waitTime)

        ccw_mode_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2F98600002000000"))
        dev.send(ccw_mode_frame)
        print(f"Motor {motor_id} set to CCW mode")
        time.sleep(waitTime)



        if(motor_id == pitch):
            zeroing_speed_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2399600101000000"))
            zeroing_acc_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B9A600001000000"))
        else:
            zeroing_speed_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("239960010A000000"))

            zeroing_acc_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B9A6000E8030000"))
        
        dev.send(zeroing_acc_frame)
        time.sleep(waitTime)
        dev.send(zeroing_speed_frame)
        print(f"Motor {motor_id} set zeroing acceleration")
        time.sleep(waitTime)


    except Exception as e:
        print(f"Error during zeroing setup for Motor {motor_id}: {e}")
        return


def setUpVertMotorMove(dev, motor_id, target_pos, speed, acc, dec):
    
    # Move a motor to the target position.

    # 4000 STEPS/REV
    # 1 cm/Rev


    #target_pos is absolute position in cm
    target_pos = (int)(float(target_pos)*4000)

    target_pos = decimal_to_lsb_left_hex(target_pos)

    speed = decimal_to_lsb_left_hex(speed)
    acc = decimal_to_lsb_left_hex(acc)
    dec = decimal_to_lsb_left_hex(dec)

    # motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600001000000"))
    # dev.send(motor_init_frame)
    # print(f"Motor {motor_id} initialized")
    # time.sleep(waitTime)

    # release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600003000000"))
    # dev.send(release_brake_frame)
    # print(f"Motor {motor_id} brake released")
    # time.sleep(waitTime)

    enable_motor_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000F000000"))
    dev.send(enable_motor_frame)
    # print(f"Motor {motor_id} enabled")
    time.sleep(waitTime)

    #Set profile Position

    position_mode_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2F60600001000000"))
    dev.send(position_mode_frame)
    # print(f"Motor {motor_id} set to position mode")
    time.sleep(waitTime)

    #set speed
    speed_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("23816000"+speed))
    dev.send(speed_frame)
    # print(f"Motor {motor_id} set speed")
    time.sleep(waitTime)

    #set acceleration
    acc_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B836000"+acc))
    dev.send(acc_frame)
    # print(f"Motor {motor_id} set acceleration")
    time.sleep(waitTime)

    #set deceleration
    dec_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B846000"+dec))
    dev.send(dec_frame)
    # print(f"Motor {motor_id} set deceleration")
    time.sleep(waitTime)

    #set target position
    target_pos_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("237A6000"+target_pos))
    dev.send(target_pos_frame)
    # print(f"Motor {motor_id} set target position")
    time.sleep(waitTime)

    motor = next((m for m in motors if m.node_id == motor_id), None)
    if motor is not None:
        motor.update_moveCommand(True)
    else:
        print(f"Warning: No motor found with node_id {motor_id}")

def motors_init_brakeRelease(dev, motors_list):
    for motor in motors_list:
        if motor.node_id == rotary:
            rotaryInit(dev, motor.node_id)
        else:
            motor_id = motor.node_id
            motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600001000000"))
            dev.send(motor_init_frame)
            print(f"Motor {motor_id} initialized")
            time.sleep(waitTime)

            release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600003000000"))
            dev.send(release_brake_frame)
            print(f"Motor {motor_id} brake released")
            time.sleep(waitTime)
def all_motors_brakeRelease(dev,motors_list):
    for motor in motors_list:
        if(motor.node_id == rotary):
            rotaryInit(dev,motor.node_id)
        else:
            motor_id = motor.node_id
            motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000F000000"))
            dev.send(motor_init_frame)
            print(f"Motor {motor_id} initialized")
            time.sleep(waitTime)

            release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600003000000"))
            dev.send(release_brake_frame)
            print(f"Motor {motor_id} brake released")
            time.sleep(waitTime)


def rotaryInit(dev,motor_id):

    motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600006000000"))
    dev.send(motor_init_frame)
    print(f"Motor {motor_id} initialized")
    time.sleep(waitTime)

    pos_setting_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2BF2600001000000"))
    dev.send(pos_setting_frame)
    print(f"Motor {motor_id} set position setting")
    time.sleep(waitTime)


    release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600007000000"))
    dev.send(release_brake_frame)
    print(f"Motor {motor_id} brake released")
    time.sleep(waitTime)


def setUpRotaryMotorMove(dev, motor_id, output_pos, speed, acc, dec):

    # speed is rpm
    # target_pos is a tenth of a degree
    # acce and dec are in rpm/s
    # 3600 steps / rev
    # Gear ratio to final output is 12.9704:1
    # 1 cm = 0.5 rev                            
    motor_pos = int(float(output_pos) * 12.9704 * 10 * 10)

    motor_pos = decimal_to_lsb_left_hex(motor_pos)

    speed = decimal_to_lsb_left_hex(speed)
    acc = decimal_to_lsb_left_hex(acc)
    dec = decimal_to_lsb_left_hex(dec) 


    #Set profile Position

    position_mode_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2F60600001000000"))
    dev.send(position_mode_frame)
    # print(f"Motor {motor_id} set to position mode")
    time.sleep(waitTime)

    #set speed
    speed_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("23816000"+speed))
    dev.send(speed_frame)
    # print(f"Motor {motor_id} set speed: "+ speed)
    time.sleep(waitTime)

    #set acceleration
    acc_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("23836000"+acc))
    dev.send(acc_frame)
    # print(f"Motor {motor_id} set acceleration: "+acc)
    time.sleep(waitTime)

    #set deceleration
    dec_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("23846000"+dec))
    dev.send(dec_frame)
    # print(f"Motor {motor_id} set deceleration")
    time.sleep(waitTime)

    release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060006F000000"))
    dev.send(release_brake_frame)
    # print(f"Motor {motor_id} brake released")
    time.sleep(waitTime)

    #set target position
    target_pos_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("237A6000"+motor_pos))
    dev.send(target_pos_frame)
    # print(f"Motor {motor_id} set target position")
    time.sleep(waitTime)

    # motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600006000000"))
    # dev.send(motor_init_frame)
    # print(f"Motor {motor_id} initialized")
    # time.sleep(0.5)

    # pos_setting_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2BF2600001000000"))
    # dev.send(pos_setting_frame)
    # print(f"Motor {motor_id} set position setting")


    # release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600007000000"))
    # dev.send(release_brake_frame)
    # print(f"Motor {motor_id} brake released")
    # time.sleep(0.5)

    # enable_motor_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060001F000000"))
    # dev.send(enable_motor_frame)
    # print(f"Motor {motor_id} enabled")
    # time.sleep(waitTime)

    motor = next((m for m in motors if m.node_id == motor_id), None)

    motor.update_moveCommand(True)

def setUpHoriMotorMove(dev, motor_id, target_pos, speed, acc, dec):

    # Move a motor to the target position.

    # 4000 STEPS/REV
    # 0.5 cm/Rev
    
    #target_pos is absolute position in cm
    target_pos = (int)(float(target_pos)*4000*2)

    target_pos = decimal_to_lsb_left_hex(target_pos)

    speed = decimal_to_lsb_left_hex(speed)
    acc = decimal_to_lsb_left_hex(acc)
    dec = decimal_to_lsb_left_hex(dec)

    # motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000100"))
    # dev.send(motor_init_frame)
    # print(f"Motor {motor_id} initialized")
    # time.sleep(waitTime)

    # release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000300"))
    # dev.send(release_brake_frame)
    # print(f"Motor {motor_id} brake released")
    # time.sleep(waitTime)
    

    enable_motor_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000F00"))
    dev.send(enable_motor_frame)
    # print(f"Motor {motor_id} enabled")
    time.sleep(waitTime)

    #Set profile Position

    position_mode_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2F60600001000000"))
    dev.send(position_mode_frame)
    # print(f"Motor {motor_id} set to position mode")
    time.sleep(waitTime)

    #set speed
    speed_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("23816000"+speed))
    dev.send(speed_frame)
    # print(f"Motor {motor_id} set speed")
    time.sleep(waitTime)

    #set acceleration
    acc_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B836000"+acc))
    dev.send(acc_frame)
    # print(f"Motor {motor_id} set acceleration")
    time.sleep(waitTime)

    #set deceleration
    dec_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B846000"+dec))
    dev.send(dec_frame)
    # print(f"Motor {motor_id} set deceleration")
    time.sleep(waitTime)

    #set target position
    target_pos_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("237A6000"+target_pos))
    dev.send(target_pos_frame)
    # print(f"Motor {motor_id} set target position")
    time.sleep(waitTime)

    motor = next((m for m in motors if m.node_id == motor_id), None)

    motor.update_moveCommand(True)

def setUpPitchMotorMove(dev, motor_id, output_pos, speed, acc, dec):

    # Move a motor to the target position.

    # 4000 STEPS/REV
    
    #target_pos is absolute position in degree

    # print(f"Pitch motor set move to relative position: {output_pos} degrees")

    pulley_ratio = 2.04

    target_pos = int(float(output_pos)*2.04*(4000/360))

    target_pos = decimal_to_lsb_left_hex(target_pos)

    speed = decimal_to_lsb_left_hex(speed)
    acc = decimal_to_lsb_left_hex(acc)
    dec = decimal_to_lsb_left_hex(dec)

    # motor_init_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000100"))
    # dev.send(motor_init_frame)
    # print(f"Motor {motor_id} initialized")
    # time.sleep(waitTime)

    # release_brake_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000300"))
    # dev.send(release_brake_frame)
    # print(f"Motor {motor_id} brake released")
    # time.sleep(waitTime)

    enable_motor_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B4060000F00"))
    dev.send(enable_motor_frame)
    # print(f"Motor {motor_id} enabled")
    time.sleep(waitTime)

    #Set profile Position

    position_mode_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2F60600001000000"))
    dev.send(position_mode_frame)
    # print(f"Motor {motor_id} set to position mode")
    time.sleep(waitTime)

    #set speed
    speed_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("23816000"+speed))
    dev.send(speed_frame)
    # print(f"Motor {motor_id} set speed")
    time.sleep(waitTime)

    #set acceleration
    acc_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B836000"+acc))
    dev.send(acc_frame)
    # print(f"Motor {motor_id} set acceleration")
    time.sleep(waitTime)

    #set deceleration
    dec_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B846000"+dec))
    dev.send(dec_frame)
    # print(f"Motor {motor_id} set deceleration")
    time.sleep(waitTime)

    #set target position
    target_pos_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("237A6000"+target_pos))
    dev.send(target_pos_frame)
    # print(f"Motor {motor_id} set target position")
    time.sleep(waitTime)

    motor = next((m for m in motors if m.node_id == motor_id), None)

    motor.update_moveCommand(True)



import time
import msvcrt

def launchMoveCommand(dev, motors_list):
    """
    Launches move commands for all motors with moveCommand=True, with estop and completion monitoring.
    
    Args:
        dev: USB-CAN device object
        motors_list (list): List of Motor objects
    """
    # Identify motors with move commands
    motors_to_move = [motor for motor in motors_list if motor.moveCommand]
    if not motors_to_move:
        print("No motors with active move commands.")
        return

    print(f"\nLaunching move commands for {len(motors_to_move)} motor(s)...")

    # Launch move commands for all relevant motors
    try:
        with can_lock:
            for motor in motors_to_move:
                motor_launch_frame = GsUsbFrame(
                    can_id=0x600 + motor.node_id, 
                    data=bytes.fromhex("2B4060005F000000")  # Controlword 0x001F: Start movement
                )
                dev.send(motor_launch_frame)
                print(f"Motor {motor.node_id} ({motor.name}) move command launched")
                motor.update_moveCommand(False)  # Reset moveCommand flag
    except Exception as e:
        print(f"Error launching move commands: {e}")
        return


    print("Move command sequence completed.")

    

def decimal_to_lsb_left_hex(decimal_num):
    """
    Converts a decimal number to an 8-digit hex string (4 bytes) with LSB on the left.
    
    Args:
        decimal_num (int): Decimal number to convert (supports negative numbers)
    
    Returns:
        str: 8-digit hex string with LSB on the left (e.g., 10 -> "0A000000")
    """
    # Convert to 4-byte (32-bit) binary, handling signed integers
    # Use 'little' endian to get LSB first
    byte_data = decimal_num.to_bytes(4, byteorder='little', signed=True)
    
    # Convert bytes to hex string, remove '0x' prefix, and ensure 8 digits
    hex_str = byte_data.hex().upper()
    hex_str = hex_str.zfill(8)  # Pad with zeros to ensure 8 digits
    
    return hex_str



def can_reader_thread(dev, stop_event, motors_list):
    """Background thread to read CAN packets, detect SDO write error frames, and handle TPDO messages."""
    debug_mode = False  # Set to True for verbose logging of all frames

    while not stop_event.is_set():
        try:
            iframe = GsUsbFrame()
            if dev.read(iframe, 10):  # Read with 10ms timeout
                # Check type of iframe.data and convert if necessary
                if isinstance(iframe.data, bytes):
                    data_bytes = iframe.data
                elif isinstance(iframe.data, list):
                    if debug_mode:
                        print(f"Debug: iframe.data is a list: CAN ID=0x{iframe.can_id:03X}, Data={iframe.data}")
                    data_bytes = bytes(iframe.data)  # Convert list of ints to bytes
                else:
                    print(f"Debug: Unexpected iframe.data type: {type(iframe.data)}, CAN ID=0x{iframe.can_id:03X}")
                    continue  # Skip this frame

                # Handle TPDO messages (0x180 to 0x1FF for TPDO1)
                if 0x180 <= iframe.can_id <= 0x1FF :  # TPDO1 range
                    node_id = iframe.can_id - 0x180
                    for motor in motors_list:
                        if motor.node_id == node_id and len(data_bytes) >= 6:
                            current_pos = int.from_bytes(data_bytes[2:6], 'little', signed=True)
                            statusword = int.from_bytes(data_bytes[0:2], 'little', signed=False)
                            motor.update_position_and_status(current_pos, statusword)
                            motor.last_update_time = time.time()
                            # print(f"Motor {motor.name}: Pos={current_pos}")

                            # if debug_mode:
                            #     print(f"Motor {motor.name}: Pos={current_pos}, Status=0x{statusword:04X}")
                
                # elif 0x281 == iframe.can_id:  # Example: Rotary motor Position TPDO
                #     current_pos = int.from_bytes(data_bytes[0:4], 'little', signed=True)
                #     print(f"frame data: {data_bytes.hex().upper()}")
                #     motors_list[0].update_position_and_status(current_pos,0)
                #     motors_list[0].last_update_time = time.time()
                #     # print(f"Rotary Motor {motors_list[0].name}: Pos={current_pos}")
                   
 
                # Check for SDO server response range (0x580 to 0x5FF)
                elif 0x580 <= iframe.can_id <= 0x5FF and len(data_bytes) >= 1:
                    if data_bytes[0] == 0x80:  # First byte is 0x80 (Abort SDO Transfer)
                        node_id = iframe.can_id - 0x580
                        data_str = data_bytes.hex().upper()
                        if len(data_bytes) >= 8:
                            index = int.from_bytes(data_bytes[1:3], 'little')
                            subindex = data_bytes[3]
                            error_code = int.from_bytes(data_bytes[4:8], 'little')
                            print(f"SDO Write Error from Node {node_id}: "
                                  f"CAN ID=0x{iframe.can_id:03X}, Data={data_str}, "
                                  f"Object=0x{index:04X}, Subindex=0x{subindex:02X}, "
                                  f"Error Code=0x{error_code:08X}")
                        else:
                            print(f"SDO Write Error from Node {node_id}: "
                                  f"CAN ID=0x{iframe.can_id:03X}, Data={data_str} (short frame)")
                
                # Optional verbose logging of all frames
                elif debug_mode:
                    data_str = data_bytes.hex().upper()
                    print(f"Received: CAN ID=0x{iframe.can_id:03X}, Data={data_str}")

        except Exception as e:
            print(f"Error reading CAN frame: {e}")
            time.sleep(0.1)  # Avoid tight loop on error
    print("CAN reader thread stopped.")


def sync_master(dev, stop_event, interval=0.1):
    """
    Sends SYNC messages at a specified interval with error handling.

    Args:
        dev: CAN device
        stop_event: Threading event to stop the sync thread
        interval (float): Time between SYNC messages in seconds (default: 0.1s)
    """
    sync_frame = GsUsbFrame(can_id=0x80, data=b'')  # Predefine SYNC frame
    attempt_count = 0
    max_attempts = 5  # Retry limit before giving up

    while not stop_event.is_set():
        try:
            if dev is None:
                print("Error: CAN device not initialized.")
                break
            
            # Send SYNC message
            dev.send(sync_frame)
            attempt_count = 0  # Reset retry counter on success
            # print(f"SYNC message sent (ID: 0x80)")  # Debug output
            time.sleep(interval)  # Wait for the specified interval

        except OSError as e:
            if e.errno == 10060:  # Timeout error
                attempt_count += 1
                print(f"SYNC master timeout [Errno 10060]: Attempt {attempt_count}/{max_attempts}")
                if attempt_count >= max_attempts:
                    print("Max SYNC attempts reached. Stopping SYNC master.")
                    break
                time.sleep(1.0)  # Longer delay before retrying
            else:
                print(f"SYNC master error: {e}")
                time.sleep(1.0)  # Wait before retrying on other errors

        except Exception as e:
            print(f"Unexpected SYNC master error: {e}")
            time.sleep(1.0)  # Wait before retrying

    print("SYNC master stopped.")


def position_printer_thread(motors_list, stop_event, interval):
    """Print the current position of all motors in one line repeatedly."""
    while not stop_event.is_set():
        try:
            pos_line = " | ".join(f"{motor.name}:{motor.current_position_real_unit:.6f}" for motor in motors_list)
            # print(f"[Motor Positions] {pos_line}")



            calculate_magnet_pose_and_sendMagAbsPos(motors_list)

            # is_stopped = " | ".join(f"{motor.name}:{'Stopped' if motor.is_stopped else 'Moving'}" for motor in motors_list)
            # print(f"[Motor Status] {is_stopped}")
            time.sleep(interval)
        except Exception as e:
            print(f"Position printer error: {e}")
            time.sleep(0.1)


def slowShutDown(dev,motors_list):
    for motor in motors_list:
        motor_id = motor.node_id
        motor_shutdown_frame = GsUsbFrame(can_id=0x600 + motor_id, data=bytes.fromhex("2B40600006000000"))
        dev.send(motor_shutdown_frame)
        time.sleep(0.5)


def euler_to_quaternion(roll, pitch, yaw, degrees=True):
    """
    Convert Euler angles to quaternion (ZYX convention) using NumPy.
    
    Args:
        roll (float): Rotation around X-axis (degrees or radians).
        pitch (float): Rotation around Y-axis (degrees or radians).
        yaw (float): Rotation around Z-axis (degrees or radians).
        degrees (bool): True if angles are in degrees, False if radians.
    
    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    
    w = cr * cp * cy + sr * sp * sy
    x = cr * sp * cy - sr * cp * sy
    y = sr * cp * cy + cr * sp * sy
    z = sr * sp * cy - cr * cp * sy
    
    return np.array([w, x, y, z])

def rotateto1000(dev):

    
    rotary_travel = 27.8   # degrees (positive/negative)
    pitch_travel = 65.20
    vert_travel = 15
    hori_travel   = 6

    motors_init_brakeRelease(dev, motors)
    setUpPitchMotorMove(dev, pitch, pitch_travel, 1, 1, 1) 
    launchMoveCommand(dev, motors)
    time.sleep(3)  # Wait for pitch motor to start moving

    setUpVertMotorMove(dev, vert1, vert_travel, 10, 100, 100)
    setUpVertMotorMove(dev, vert2, vert_travel, 10, 100, 100)
    launchMoveCommand(dev, motors)
    time.sleep(0.5)  # Wait for vertical motors to start moving

    # setUpHoriMotorMove(dev, hori1, hori_travel, 7, 100, 100)
    # setUpHoriMotorMove(dev, hori2, hori_travel, 7, 100, 100)
    # launchMoveCommand(dev, motors)
    # time.sleep(0.5)  # Wait for horizontal motors to start moving

    setUpRotaryMotorMove(dev, rotary, rotary_travel, 50, 280, 280) 
    launchMoveCommand(dev, motors)

def autoZeroing(dev,motors):

    ## Zeroing hori motors
    zeroing_motor(hori1, dev, motors)
    zeroing_motor(hori2, dev, motors)
    start_zeroing_frame = GsUsbFrame(can_id=0x600 + hori1, data=bytes.fromhex("2B4060001F000000"))
    start_zeroing_frame1 = GsUsbFrame(can_id=0x600 + hori2, data=bytes.fromhex("2B4060001F000000"))
    dev.send(start_zeroing_frame)
    dev.send(start_zeroing_frame1)
    print(f"hori Motor  zeroing started")
    time.sleep(10)  # Wait for pitch motor to start zeroing




    ## Zeroing vert motors
    zeroing_motor(vert1, dev, motors)
    zeroing_motor(vert2, dev, motors)
    start_zeroing_frame = GsUsbFrame(can_id=0x600 + vert1, data=bytes.fromhex("2B4060001F000000"))
    start_zeroing_frame1 = GsUsbFrame(can_id=0x600 + vert2, data=bytes.fromhex("2B4060001F000000"))
    dev.send(start_zeroing_frame)
    dev.send(start_zeroing_frame1)
    print(f"Vert Motor  zeroing started")

    ## Zeroing pitch motor
    zeroing_motor(pitch, dev, motors)
    start_zeroing_frame = GsUsbFrame(can_id=0x600 + pitch, data=bytes.fromhex("2B4060001F000000"))
    dev.send(start_zeroing_frame)
    print(f"Pitch Motor  zeroing started")
    time.sleep(1)  # Wait for pitch motor to start zeroing


    ## Zeroing rotary motor
    zeroing_motor(rotary, dev, motors)



def main():
    # Existing code...
    dev = None  # Initialize dev outside the conditional block
    stop_event = threading.Event()  # Event to stop the thread
    reader_thread = None  # To track the reader thread
 

    try:
        while True:
            user_input = input("""
    Choose an option:
    1. Scan for Device
    2. Set NMT state for all nodes
    3. Zeroing Motors
    4. Start Movement Sequence
    5. Launch Move Command
    6. Manuel
    7. Release Brakes
    8. Slow Shut Down
    9. Auto Mag Field Caliration
    10. Auto Zeroing
    11. Rotate to 1000
    12. Start pos listener for Vert1,2 
    13. Offset                         
    q. Quit
    Enter your choice:\n\n""").strip()
        

            if user_input == "1":
                found, dev = scan_for_device(coveredCanConverterId)
                if found:
                    print(f"Device found: {dev}")
                    if not dev.set_bitrate(canBusBitRate):
                        print("Cannot set bitrate for gs_usb")
                        continue
                    print("Set bitrate successfully")
                    dev.start(GS_CAN_MODE_NORMAL)
                    print("Device started in GS_CAN_MODE_NORMAL")
                    print(f"""
    ###################################
        Can Converter Start success
        #########################
        Reader Thread Started
    ###################################
                    """)
                    reader_thread = threading.Thread(target=can_reader_thread, args=(dev, stop_event, motors))
                    reader_thread.daemon = True
                    reader_thread.start()
                    print("Reader thread started.")
                    time.sleep(0.5)  # Give the reader thread a moment to start and read any initial frames

                    disable_stepper_28XTpdo(dev)
                    
                    # # Set up TPDO for all motors
                    #No need to use these, default tpdo0 is enough
                    # tpdo_setup(vert1, dev)
                    # tpdo_setup(vert2, dev) 
                    tpdo_setup(rotary, dev)
                    
                    # Start SYNC master thread
                    sync_stop_event = threading.Event()
                    sync_thread = threading.Thread(target=sync_master, args=(dev, sync_stop_event, 0.01))
                    sync_thread.daemon = True
                    sync_thread.start()
                    print("SYNC master started.")
                    
                    # Start the background vertical sync monitor thread
                    vert_sync_stop_event = threading.Event()
                    t_poll = threading.Thread(
                        target=vert_sync_monitoring,
                        args=(dev, motors, 1000, 0.01, vert_sync_stop_event)
                    )
                    t_poll.daemon = True
                    t_poll.start()
                    print("Vertical sync monitoring is now RUNNING in the background.")

                    time.sleep(0.5)  # Allow some time for threads to start and stabilize

                    if any(name == 'position_printer' for name, _ in threads):
                        print("Motor position printer is already running.")
                        continue

                    pos_printer_stop_event = threading.Event()
                    pos_printer_thread_obj = threading.Thread(
                        target=position_printer_thread,
                        args=(motors, pos_printer_stop_event, 0.01)
                    )
                    pos_printer_thread_obj.daemon = True
                    pos_printer_thread_obj.start()
                    threads.append(('position_printer', pos_printer_stop_event))
                    print("Motor position printer started. It will print current positions once per second.")
                    
                else:
                    print("Device not found.")


            elif user_input == "2":

                nmt_input = input("Enter NMT state for all nodes: ").strip()
                #01 oP
                #02 sTOP
                #80 Pre-op
                #81 reset node
                #82 reset communication

                # Set up their NMT state
                data = nmt_input+"00"
                data = bytes.fromhex(data)  # Convert to bytes: b"\x01\x00"
                nmtAll2Op_frame = GsUsbFrame(can_id=0x000, data=data)
                if(nmt_input == "01"):
                    ret = input("Did you restart the motor power after e-stop? (y/n): ").strip().lower()
                    if ret != "y":
                        print("Please restart the motor power before sending the stop command.")
                        continue
                dev.send(nmtAll2Op_frame)
                vertPreMove(dev, motors)

            elif user_input == "3":
                # Zeroing motors
                print("\nMotors Zeroing Options:")
                zeroingOptions = input("""
    press 1 to Zeroing Rotary motor 
    press 2 to Zeroing Vert1 motor
    press 3 to Zeroing Vert2 motor
    press 4 to Zeroing Hori1 motor
    press 5 to zero both vert motors
    press 6 to zero both hori motors
    press 7 to zero pitch motor
    press q to quit            
                """).strip()

                if(zeroingOptions == "5"):
                    zeroing_motor(vert1, dev, motors)
                    zeroing_motor(vert2, dev, motors)

                    start_zeroing_frame = GsUsbFrame(can_id=0x600 + vert1, data=bytes.fromhex("2B4060001F000000"))
                    start_zeroing_frame1 = GsUsbFrame(can_id=0x600 + vert2, data=bytes.fromhex("2B4060001F000000"))
                    dev.send(start_zeroing_frame)
                    dev.send(start_zeroing_frame1)
                    print(f"Vert Motor  zeroing started")
                    continue

                if(zeroingOptions == "6"):
                    zeroing_motor(hori1, dev, motors)
                    zeroing_motor(hori2, dev, motors)

                    start_zeroing_frame = GsUsbFrame(can_id=0x600 + hori1, data=bytes.fromhex("2B4060001F000000"))
                    start_zeroing_frame1 = GsUsbFrame(can_id=0x600 + hori2, data=bytes.fromhex("2B4060001F000000"))
                    dev.send(start_zeroing_frame)
                    dev.send(start_zeroing_frame1)
                    print(f"hori Motor  zeroing started")
                    continue

                if(zeroingOptions == "7"):
                    zeroing_motor(pitch, dev, motors)

                    start_zeroing_frame = GsUsbFrame(can_id=0x600 + pitch, data=bytes.fromhex("2B4060001F000000"))
                    dev.send(start_zeroing_frame)
                    print(f"Pitch Motor  zeroing started")
                    continue



                zeroing_motor(int(zeroingOptions), dev, motors)

        
            elif user_input == "4":
                if dev is None:
                    print("Please scan for a device first (Option 1).")
                    continue

                # Initialize motors and release brakes
                motors_init_brakeRelease(dev, motors)
                relRotarySpeed = 65
                relRotaryAcc = 180
                relVertSpeed = 10
                relVertAcc = 100
                relHoriSpeed = 20
                relHoriAcc = 100
                relPitchSpeed = 20
                relPitchAcc = 30

                print("\nStarting movement sequence. Press Enter to start each movement, or 'q' for emergency stop.")

                for i, movement in enumerate(rev_movement_sequence):
                    print(f"\nPreparing movement {i+1}/{len(rev_movement_sequence)}...")

                    relRotaryDeg = movement[0]

                    relVertDisp = movement[1]

                    relHoriDisp = movement[2]

                    relPitchDeg = movement[3]

                    # Set up motor movements
                    setUpRotaryMotorMove(dev, rotary, relRotaryDeg, relRotarySpeed, relRotaryAcc, relRotaryAcc)
                    time.sleep(0.05)  # Small delay to ensure commands are sent in order
                    setUpVertMotorMove(dev, vert1, relVertDisp, relVertSpeed, relVertAcc, relVertAcc)
                    time.sleep(0.05)  # Small delay to ensure commands are sent in order
                    setUpVertMotorMove(dev, vert2, relVertDisp, relVertSpeed, relVertAcc, relVertAcc)
                    time.sleep(0.05)  # Small delay to ensure commands are sent in order
                    setUpHoriMotorMove(dev, hori1, relHoriDisp, relHoriSpeed, relHoriAcc, relHoriAcc)
                    time.sleep(0.05)  # Small delay to ensure commands are sent in order
                    setUpHoriMotorMove(dev, hori2, relHoriDisp, relHoriSpeed, relHoriAcc, relHoriAcc)
                    time.sleep(0.1)  # Small delay to ensure commands are sent in order
                    setUpPitchMotorMove(dev, pitch, relPitchDeg, relPitchSpeed, relPitchAcc, relPitchAcc)
                    print(f"Movement {i+1} configured: Rotary={relRotaryDeg}°, Vert={relVertDisp}cm, Hori={relHoriDisp}cm, Pitch={relPitchDeg}°")
                    print("Pitch Motor Pos: ", motors[5].current_position_real_unit)

                    print("Motors configured. Press Enter to launch this movement, or 'q' to emergency stop.")

                    # Wait for Enter or 'q'
                    while True:

                        all_stop = True

                        for motor in motors:
                            if not motor.is_stopped:
                                all_stop = False
                                # print(f"Motor {motor.node_id} ({motor.name}) is still moving.")
                                
                        # print(f"all stop: {all_stop}")
                        if all_stop:
                            data_to_send = {
                                "record_data": True
                            }
                            send_data_to_arduino(data_dict = data_to_send)
                            time.sleep(0.5)  # Wait a moment to ensure data is sent before launching next command
                            launchMoveCommand(dev, motors)
                            print(f"Movement {i+1} launched.")
                            time.sleep(1)  # Small delay to ensure command is sent before checking for next input
                            break  # Proceed to next movement                                

                        if msvcrt.kbhit():  # Check if a key has been pressed
                            key = msvcrt.getch().decode('utf-8').lower()
                            if key == 'q':
                                # Send Quick Stop command to all motors
                                for motor in motors:
                                    quick_stop_frame = GsUsbFrame(
                                        can_id=0x600 + motor.node_id,
                                        data=bytes.fromhex("2B4060000B000000")  # Quick Stop
                                    )
                                    try:
                                        dev.send(quick_stop_frame)
                                        print(f"Emergency stop: Quick Stop sent to Motor {motor.node_id} ({motor.name})")
                                    except Exception as e:
                                        print(f"Failed to send Quick Stop to Motor {motor.node_id}: {e}")
                                
                                # Reset to Switch On Disabled state
                                for motor in motors:
                                    reset_frame = GsUsbFrame(
                                        can_id=0x600 + motor.node_id,
                                        data=bytes.fromhex("2B40600006000000")  # Controlword 0x0006: Switch On Disabled
                                    )
                                    try:
                                        dev.send(reset_frame)
                                        print(f"Reset: Motor {motor.node_id} ({motor.name}) set to Switch On Disabled")
                                    except Exception as e:
                                        print(f"Failed to reset Motor {motor.node_id}: {e}")
                                
                                print("Movement sequence aborted due to emergency stop.")
                                return  # Exit Option 10
                            elif key == '\r' :  # Enter key
                                launchMoveCommand(dev, motors)
                                print(f"Movement {i+1} launched.")
                                break  # Proceed to next movement
                        time.sleep(0.01)  # Small delay to prevent CPU hogging

                print("Movement sequence completed.")

            elif user_input == "5":
                launchMoveCommand(dev, motors)
                print("Move Command Launched")

            elif user_input == "6":
                if dev is None:
                    print("Please scan for a device first (Option 1).")
                    continue

                global rotaryDisplacement, vertDisplacement, horiDisplacement, pitchAngle, rollAngle , yawAngle
                # Predefined travel amounts
                rotary_travel = 30.0    # degrees (positive/negative)
                vert_travel = 5      # cm (positive/negative, for vert1 and vert2)
                hori_travel = 3      # cm (positive/negative, for hori1 and hori2)
                pitch_travel = 10.0     # degrees (positive/negative)

                vert_travel_fine = 0.05  # cm for fine control
                pitch_travel_fine = 0.2  # degrees for fine control
                hori_travel_fine = 0.5  # cm for fine control
                rotary_travel_fine = 0.2    # degrees for fine control

                vert_travel_super_fine = 0.1  # cm for super fine control

                # Fixed speed, acceleration, deceleration
                speed = 50  # RPM or steps/s (adjust as needed)
                acc = 100   # RPM/s or steps/s²
                dec = 100   # RPM/s or steps/s²

                motors_init_brakeRelease(dev, motors)

                print("\nEntering Manual Mode. Select a motor or pais1r to control with keypresses.")
                print("Available options:")
                print("  1: Rotary (h: CW, j: CCW)")
                print("  2: Vert1+Vert2 (h: UP, j: DOWN)")
                print("  3: Hori1+Hori2 (h: UP, j: DOWN)")
                print("  4: Pitch (h: +5°, j: -5°)")
                print("  5: Vert1 (h: UP, j: DOWN)")
                print("  6: Vert2 (h: UP, j: DOWN)")
                print("  7: Hori1 (h: UP, j: DOWN)")
                print("  8: Hori2 (h: UP, j: DOWN)")
                print("Press 'q' to exit manual mode.")

                selected_option = None

                while True:
                    # Prompt for motor selection if none is selected
                    if selected_option is None:
                        motor_input = input("Select motor option (1-8) or 'q' to quit: ").strip().lower()
                        if motor_input == 'q':
                            print("Exiting manual mode.")
                            break
                        try:
                            selected_option = int(motor_input)
                            if selected_option not in range(1, 9):  # Updated to 1-8
                                print("Invalid option. Choose 1-8.")
                                selected_option = None
                                continue
                            motor_names = {
                                1: "Rotary",
                                2: "Vert1+Vert2",
                                3: "Hori1+Hori2",
                                4: "Pitch",
                                5: "Vert1",
                                6: "Vert2",
                                7: "Hori1",
                                8: "Hori2"
                            }
                            print(f"Selected {motor_names[selected_option]}. Use 'h' to move positive, 'j' to move negative, 's' to select another motor.")
                        except ValueError:
                            print("Invalid input. Enter 1-8 or 'q'.")
                            continue

                    # Check for keypress to move the selected motor
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()

                    

                        if key == 'q':
                            print("Exiting manual mode.")
                            break
                        elif key == 's':  # 's' to reselect motor
                            selected_option = None
                            print("Available options:")
                            print("  1: Rotary (h: CW, j: CCW)")
                            print("  2: Vert1+Vert2 (h: UP, j: DOWN)")
                            print("  3: Hori1+Hori2 (h: UP, j: DOWN)")
                            print("  4: Pitch (h: +5°, j: -5°)")
                            print("  5: Vert1 (h: UP, j: DOWN)")
                            print("  6: Vert2 (h: UP, j: DOWN)")
                            print("  7: Hori1 (h: UP, j: DOWN)")
                            print("  8: Hori2 (h: UP, j: DOWN)")
                            print("Press 'q' to exit manual mode.")                        
                            print("Returning to motor selection.")
                            continue

                        try:
                            # Rotary (Option 1)
                            if selected_option == 1 and key in ('h', 'j'):
                                travel = -rotary_travel if key == 'h' else rotary_travel
                                setUpRotaryMotorMove(dev, rotary, travel, 200, 280, 280) 
                                launchMoveCommand(dev, motors)

                                rotaryDisplacement = rotaryDisplacement + rotary_travel if key == 'h' else rotaryDisplacement - rotary_travel
                                yawAngle = yawAngle + rotary_travel if key == 'h' else yawAngle - rotary_travel 

                                print(f"Current Rotary Displacement: {rotaryDisplacement} degrees")
                            elif selected_option == 1 and key in ('n', 'm'):
                                travel = -rotary_travel_fine if key == 'n' else rotary_travel_fine
                                setUpRotaryMotorMove(dev, rotary, travel, 350, 280, 280) 
                                launchMoveCommand(dev, motors)

                                rotaryDisplacement = rotaryDisplacement + rotary_travel_fine if key == 'n' else rotaryDisplacement - rotary_travel_fine
                                yawAngle = yawAngle + rotary_travel_fine if key == 'n' else yawAngle - rotary_travel_fine 

                                print(f"Current Rotary Displacement: {rotaryDisplacement} degrees (fine control)")

                            # Vert1 and Vert2 in unison (Option 2)
                            elif selected_option == 2 and key in ('h', 'j'):
                                travel = vert_travel if key == 'h' else -vert_travel
                                setUpVertMotorMove(dev, vert1, travel, speed, acc, dec)
                                setUpVertMotorMove(dev, vert2, travel, speed, acc, dec)
                                launchMoveCommand(dev, motors)

                                vertDisplacement = vertDisplacement + vert_travel if key == 'h' else vertDisplacement - vert_travel
                                print(f"Vert1 and Vert2 moved by {'+' if travel > 0 else ''}{travel} cm")
                                print(f"Current Vert Displacement: {vertDisplacement} cm")

                            # Verti1 and Vert2 in unison fine control with "n,m" keys
                            elif selected_option == 2 and key in ('n', 'm'):
                                travel = vert_travel_fine if key == 'n' else -vert_travel_fine
                                setUpVertMotorMove(dev, vert1, travel, speed, acc, dec)
                                setUpVertMotorMove(dev, vert2, travel, speed, acc, dec)
                                launchMoveCommand(dev, motors)

                                vertDisplacement = vertDisplacement + vert_travel_fine if key == 'n' else vertDisplacement - vert_travel_fine
                                print(f"Vert1 and Vert2 moved by {'+' if travel > 0 else ''}{travel} cm (fine control)")
                                print(f"Current Vert Displacement: {vertDisplacement} cm")


                            # Hori1 and Hori2 in unison (Option 3)
                            elif selected_option == 3 and key in ('h', 'j'):
                                travel = hori_travel if key == 'h' else -hori_travel
                                setUpHoriMotorMove(dev, hori1, travel, speed, acc, dec)
                                setUpHoriMotorMove(dev, hori2, travel, speed, acc, dec)
                                launchMoveCommand(dev, motors)
                                horiDisplacement = horiDisplacement + hori_travel if key == 'h' else horiDisplacement - hori_travel
                                print(f"Current Hori Displacement: {horiDisplacement} cm")
                                print(f"Hori1 and Hori2 moved by {'+' if travel > 0 else ''}{travel} cm")
                            # Hori1 and Hori2 in unison fine control with "n,m" keys
                            elif selected_option == 3 and key in ('n', 'm'):
                                travel = hori_travel_fine if key == 'n' else -hori_travel_fine
                                setUpHoriMotorMove(dev, hori1, travel, speed, acc, dec)
                                setUpHoriMotorMove(dev, hori2, travel, speed, acc, dec)
                                launchMoveCommand(dev, motors)
                                horiDisplacement = horiDisplacement + hori_travel_fine if key == 'n' else horiDisplacement - hori_travel_fine
                                print(f"Current Hori Displacement: {horiDisplacement} cm (fine control)")
                                print(f"Hori1 and Hori2 moved by {'+' if travel > 0 else ''}{travel} cm (fine control)")

                            # Pitch (Option 4)
                            elif selected_option == 4 and key in ('h', 'j'):
                                travel = pitch_travel if key == 'h' else -pitch_travel
                                setUpPitchMotorMove(dev, pitch, travel, 30, 40, 40)
                                launchMoveCommand(dev, motors)
                                pitchAngle = pitchAngle + pitch_travel if key == 'h' else pitchAngle - pitch_travel

                                print(f"Pitch moved by {'+' if travel > 0 else ''}{travel} degrees")
                                print(f"Current Pitch Angle: {pitchAngle} degrees")
                            # Pitch fine control with "n,m" keys
                            elif selected_option == 4 and key in ('n', 'm'):
                                travel = pitch_travel_fine if key == 'n' else -pitch_travel_fine
                                setUpPitchMotorMove(dev, pitch, travel, 30, acc, dec)
                                launchMoveCommand(dev, motors)
                                pitchAngle = pitchAngle + pitch_travel_fine if key == 'n' else pitchAngle - pitch_travel_fine

                                print(f"Pitch moved by {'+' if travel > 0 else ''}{travel} degrees (fine control)")
                                print(f"Current Pitch Angle: {pitchAngle} degrees")

                            # Vert1 individually (Option 5)
                            elif selected_option == 5 and key in ('h', 'j'):
                                travel = vert_travel_super_fine if key == 'h' else -vert_travel_super_fine
                                setUpVertMotorMove(dev, vert1, travel, speed, acc, dec)
                                launchMoveCommand(dev, motors)
                                print(f"Vert1 moved by {'+' if travel > 0 else ''}{travel} cm")

                            # Vert2 individually (Option 6)
                            elif selected_option == 6 and key in ('h', 'j'):
                                travel = vert_travel_super_fine if key == 'h' else -vert_travel_super_fine
                                setUpVertMotorMove(dev, vert2, travel, speed, acc, dec)
                                launchMoveCommand(dev, motors)
                                print(f"Vert2 moved by {'+' if travel > 0 else ''}{travel} cm")

                            # Hori1 individually (Option 7)
                            elif selected_option == 7 and key in ('h', 'j'):
                                travel = vert_travel_super_fine if key == 'h' else -vert_travel_super_fine
                                setUpHoriMotorMove(dev, hori1, travel, speed, acc, dec)
                                launchMoveCommand(dev, motors)
                                print(f"Hori1 moved by {'+' if travel > 0 else ''}{travel} cm")

                            # Hori2 individually (Option 8)
                            elif selected_option == 8 and key in ('h', 'j'):
                                travel = vert_travel_super_fine if key == 'h' else -vert_travel_super_fine
                                setUpHoriMotorMove(dev, hori2, travel, speed, acc, dec)
                                launchMoveCommand(dev, motors)
                                print(f"Hori2 moved by {'+' if travel > 0 else ''}{travel} cm")


                            
                            
                            print("Rotary Displacement:", rotaryDisplacement)
                            print("Vert Displacement:", vertDisplacement)
                            print("Hori Displacement:", horiDisplacement)
                            print("Pitch Angle:", pitchAngle)
                            print("Yaw Angle:", yawAngle)

                            worldX = topsheetToCircleCenter + (maxHoriToCircleCenter - horiDisplacement)*np.cos(np.deg2rad(rotaryDisplacement))
                            worldY = topsheetHalfWidth + (maxHoriToCircleCenter - horiDisplacement)*np.sin(np.deg2rad(rotaryDisplacement))
                            worldZ = vertDisplacement + magnetMinHeight

                            print("\n\n")
                            

                            print(f"World X Position: {worldX} cm")    
                            print(f"World Y Position: {worldY} cm")
                            print(f"World Z Position: {worldZ} cm")

                            print("Roll angle:", rollAngle)
                            print("Pitch angle:", pitchAngle)
                            print("Yaw angle:", yawAngle)

                            quaternion = euler_to_quaternion(rollAngle, pitchAngle, yawAngle, degrees=True)
                            print(f"Quaternion [w, x, y, z]: {quaternion}")
                            print("\n\n")
                            
                            # Prepare data to send
                            data_to_send = {
                                "x": float(worldX),  # Convert to float for JSON serialization
                                "y": float(worldY),
                                "z": float(worldZ),
                                "qw": float(quaternion[0]),
                                "qx": float(quaternion[1]),
                                "qy": float(quaternion[2]),
                                "qz": float(quaternion[3]),
                                "record_data":False
                            }

                            # Send data to arduino_magnetic_reader.py
                            if send_data_to_arduino(data_dict=data_to_send):
                                print("Sent data to Arduino reader:", data_to_send)
                            else:
                                print("Failed to send data to Arduino reader.")



                        except Exception as e:
                            print(f"Error during manual move: {e}")

                    time.sleep(0.01)  # Small delay to prevent CPU hogging


            elif user_input == "7":
                if dev is None:
                    print("Please scan for a device first (Option 1).")
                    continue
                all_motors_brakeRelease(dev, motors)
                print("Brakes released for all motors.")        
                


            elif user_input == "8":
                slowShutDown(dev, motors)
                print("Slow Shutdown Command Launched")

            elif user_input == "9":
                if dev is None:
                    print("Please scan for a device first (Option 1).")
                    continue
                automated_movement(dev,motors)

            elif user_input == "10":
                if dev is None:
                    print("Please scan for a device first (Option 1).")
                    continue
                autoZeroing(dev,motors)
            elif user_input == "11":
                if dev is None:
                    print("Please scan for a device first (Option 1).")
                    continue
                rotateto1000(dev)

            elif user_input == "12":
                if dev is None:
                    print("Please scan for device first.")
                    continue
                disable_stepper_28XTpdo(dev)
                
                # # Set up TPDO for all motors
                #No need to use these, default tpdo0 is enough
                # tpdo_setup(vert1, dev)
                # tpdo_setup(vert2, dev) 
                
                # Start SYNC master thread
                sync_stop_event = threading.Event()
                sync_thread = threading.Thread(target=sync_master, args=(dev, sync_stop_event, 0.01))
                sync_thread.daemon = True
                sync_thread.start()
                print("SYNC master started.")
                
                # Start the background vertical sync monitor thread
                vert_sync_stop_event = threading.Event()
                t_poll = threading.Thread(
                    target=vert_sync_monitoring,
                    args=(dev, motors, 10000, 0.001, vert_sync_stop_event)
                )
                t_poll.daemon = True
                t_poll.start()
                print("Vertical sync monitoring is now RUNNING in the background.")
                
                # Store the stop events for cleanup
                threads.append(('sync', sync_stop_event))
                threads.append(('vert_sync', vert_sync_stop_event))

            elif user_input == "13":
                magnet_offset(motors)



            elif user_input.lower() == "q":
                # Stop all threads before quitting
                stop_event.set()
                for thread_name, thread_stop_event in threads:
                    thread_stop_event.set()
                    print(f"Stopping {thread_name} thread...")
                
                for motor in motors:
                    quick_stop_frame = GsUsbFrame(
                        can_id=0x600 + motor.node_id,
                        data=bytes.fromhex("2B4060000B000000")  # Quick Stop
                    )
                    try:
                        dev.send(quick_stop_frame)
                        print(f"Emergency stop: Quick Stop sent to Motor {motor.node_id} ({motor.name})")
                    except Exception as e:
                        print(f"Failed to send Quick Stop to Motor {motor.node_id}: {e}")
                
                # Reset to Switch On Disabled state
                for motor in motors:
                    reset_frame = GsUsbFrame(
                        can_id=0x600 + motor.node_id,
                        data=bytes.fromhex("2B40600006000000")  # Controlword 0x0006: Switch On Disabled
                    )
                    try:
                        dev.send(reset_frame)
                        print(f"Reset: Motor {motor.node_id} ({motor.name}) set to Switch On Disabled")
                    except Exception as e:
                        print(f"Failed to reset Motor {motor.node_id}: {e}")
                        
                print("Exiting...")
                break
            
            else:
                print("Invalid input. Please try again.")
    except KeyboardInterrupt:
        emergency_stop_all(dev, motors)
    finally:
        if stop_event:
            stop_event.set()  # Ensure the reader thread is stopped
        for thread_name, thread_stop_event in threads:
            thread_stop_event.set()

            
def emergency_stop_all(dev, motors_list):
    """Sends Quick Stop and Switch On Disabled frames to all motors."""
    if dev is None:
        return
    
    data = "8200"
    data = bytes.fromhex(data)  # Convert to bytes: b"\x01\x00"
    nmtAll2Reset_frame = GsUsbFrame(can_id=0x000, data=data)
    dev.send(nmtAll2Reset_frame)
        
    print("\n[!] Keyboard Interrupt: Sending Emergency Stop to all motors...")
    for motor in motors_list:
        # 1. Send Quick Stop (Controlword 0x000B)
        quick_stop_frame = GsUsbFrame(
            can_id=0x600 + motor.node_id,
            data=bytes.fromhex("2B4060000B000000")
        )
        # 2. Send Switch On Disabled (Controlword 0x0006)
        reset_frame = GsUsbFrame(
            can_id=0x600 + motor.node_id,
            data=bytes.fromhex("2B40600000000000")
        )
                # 2. Send Switch On Disabled (Controlword 0x0006)
        clear_frame = GsUsbFrame(
            can_id=0x600 + motor.node_id,
            data=bytes.fromhex("2B4060006F020000")
        )
        try:
            dev.send(quick_stop_frame)
            time.sleep(0.001) # Small delay to ensure bus doesn't saturate
            dev.send(reset_frame)
            dev.send(clear_frame)
            print(f"Emergency stop sent to Motor {motor.node_id} ({motor.name})")
        except Exception as e:
            print(f"Failed to send e-stop to Motor {motor.node_id}: {e}")
    data = "8200"
    data = bytes.fromhex(data)  # Convert to bytes: b"\x01\x00"
    nmtAll2Reset_frame = GsUsbFrame(can_id=0x000, data=data)
    dev.send(nmtAll2Reset_frame)
    time.sleep(3)

    ## print warning to cut off power to motor to clear pos memory
    while True:
        print("Waiting for user to restart motor power...")
        time.sleep(0.3)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass