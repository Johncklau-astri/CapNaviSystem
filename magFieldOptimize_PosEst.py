import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution, minimize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import json
import os
import socket
import struct
import threading
import argparse
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
POSITION_SERVER_HOST = 'localhost'
POSITION_SERVER_PORT = 12346
CSV_FILE = 'Allpoints.csv'
STOP_EVENT = threading.Event()
ARROW_SCALE = 5
ARROW_WIDTH = 0.002

# Constants
mu_0 = 4 * np.pi * 1e-7
muT_to_T = 1e-6
factor = mu_0 / (4 * np.pi) * 1e6

# --- TCP Server for Position Estimation ---
def run_position_server(host=POSITION_SERVER_HOST, port=POSITION_SERVER_PORT, dipole_params=None, pos_bounds=None):
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        server_socket.settimeout(0.1)
        while not STOP_EVENT.is_set():
            try:
                client_socket, addr = server_socket.accept()
                client_socket.settimeout(2.0)
                handle_client(client_socket, addr, dipole_params, pos_bounds)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Server error: {e}")
        server_socket.close()
    except Exception as e:
        print(f"Failed to start server: {e}")
    finally:
        if 'server_socket' in locals():
            server_socket.close()

def handle_client(client_socket, addr, dipole_params, pos_bounds):
    try:
        while not STOP_EVENT.is_set():
            # Receive data length
            length_bytes = b''
            while len(length_bytes) < 4:
                chunk = client_socket.recv(4 - len(length_bytes))
                if not chunk:
                    raise ConnectionError("Client disconnected")
                length_bytes += chunk
            length = struct.unpack('>I', length_bytes)[0]
            
            # Receive data
            data_bytes = b''
            while len(data_bytes) < length:
                chunk = client_socket.recv(length - len(data_bytes))
                if not chunk:
                    raise ConnectionError("Client disconnected")
                data_bytes += chunk
            
            # Parse JSON data
            data = json.loads(data_bytes.decode('utf-8'))
            B_custom = [data['magX'], data['magY'], data['magZ']]
            quat_sensor = [data['qw'], data['qx'], data['qy'], data['qz']]
            
            # Validate inputs
            if any(abs(m) > 8000 for m in B_custom):
                print(f"Invalid magnetic field values: {B_custom} µT exceed ±8000 µT")
                response = {'estX': 0.0, 'estY': 0.0, 'estZ': 0.0, 'error': 'Invalid magnetic field values'}
            else:
                quat_norm = np.sqrt(sum(x**2 for x in quat_sensor))
                if not 0.99 <= quat_norm <= 1.01:
                    print(f"Invalid quaternion norm: {quat_norm:.3f}")
                    response = {'estX': 0.0, 'estY': 0.0, 'estZ': 0.0, 'error': 'Invalid quaternion norm'}
                else:
                    quat_sensor = [x / quat_norm for x in quat_sensor]  # Normalize quaternion
                    # Estimate position
                    start_time = time.time()
                    r_est, r_init = estimate_position_custom(
                        B_custom, quat_sensor, dipole_params['m'], pos_bounds,
                        dipole_params['a_x'], dipole_params['b_x'],
                        dipole_params['a_y'], dipole_params['b_y'],
                        dipole_params['a_z'], dipole_params['b_z']
                    )
                    estimation_time = time.time() - start_time
                    print(f"\nArduino Input: B = {B_custom} µT, Quaternion = {quat_sensor}")
                    print(f"Initial Guess: x={r_init[0]:.6f}, y={r_init[1]:.6f}, z={r_init[2]:.6f} m")
                    print(f"Estimated Position: x={r_est[0]:.6f}, y={r_est[1]:.6f}, z={r_est[2]:.6f} m")
                    response = {'estX': float(r_est[0]), 'estY': float(r_est[1]), 'estZ': float(r_est[2])}
            
            # Send response back to client
            response_str = json.dumps(response)
            response_bytes = response_str.encode('utf-8')
            client_socket.sendall(struct.pack('>I', len(response_bytes)))
            client_socket.sendall(response_bytes)
    except (ConnectionError, socket.error, json.JSONDecodeError) as e:
        print(f"Client {addr} error: {e}")
    except Exception as e:
        print(f"Unexpected error handling client {addr}: {e}")
    finally:
        client_socket.close()

# --- Helper Functions ---
def load_dipole_params():
    param_file = 'dipole_model_params.json'
    if not os.path.exists(param_file):
        return None
    try:
        with open(param_file, 'r') as f:
            params = json.load(f)
        required_keys = ['m_x', 'm_y', 'm_z', 'a_x', 'b_x', 'a_y', 'b_y', 'a_z', 'b_z']
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            print(f"Missing keys in {param_file}: {missing_keys}")
            return None
        m = np.array([params['m_x'], params['m_y'], params['m_z']])
        if np.linalg.norm(m) > 1000:
            print(f"Dipole moment norm ({np.linalg.norm(m):.3f} A·m²) is too large.")
            return None
        for key in ['a_x', 'a_y', 'a_z']:
            if not 0.1 <= params[key] <= 10:
                print(f"{key} ({params[key]:.3f}) out of valid range [0.1, 10].")
                return None
        for key in ['b_x', 'b_y', 'b_z']:
            if not -1 <= params[key] <= 1:
                print(f"{key} ({params[key]:.3f}) out of valid range [-1, 1].")
                return None
        return {
            'm': m,
            'a_x': params['a_x'],
            'b_x': params['b_x'],
            'a_y': params['a_y'],
            'b_y': params['b_y'],
            'a_z': params['a_z'],
            'b_z': params['b_z']
        }
    except json.JSONDecodeError as e:
        print(f"Failed to decode {param_file}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {param_file}: {e}")
        return None

def quaternion_to_rotation_matrix(q):
    q_w, q_x, q_y, q_z = q
    n = q_w**2 + q_x**2 + q_y**2 + q_z**2
    if n < 1e-6:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * q_w * q_x, s * q_w * q_y, s * q_w * q_z
    xx, yy, zz = s * q_x * q_x, s * q_y * q_y, s * q_z * q_z
    xy, yz, zx = s * q_x * q_y, s * q_y * q_z, s * q_z * q_x
    R = np.array([
        [1.0 - (yy + zz), xy - wz, zx + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [zx - wy, yz + wx, 1.0 - (xx + yy)]
    ])
    return R

def dipole_field(r, m, R_sensor=None, a_x=1.0, b_x=0.0, a_y=1.0, b_y=0.0, a_z=1.0, b_z=0.0):
    x, y, z = r
    p_x = a_x * x + b_x
    p_y = a_y * y + b_y
    p_z = a_z * z + b_z
    p = [p_x, p_y, p_z]
    m_x, m_y, m_z = m
    r2 = p_x**2 + p_y**2 + p_z**2
    if r2 < 1e-6:
        return np.zeros(3)
    r5 = r2**2.5
    r_dot_m = p_x * m_x + p_y * m_y + p_z * m_z
    B_x = factor * (3 * p_x * r_dot_m / r5 - m_x / r2)
    B_y = factor * (3 * p_y * r_dot_m / r5 - m_y / r2)
    B_z = factor * (3 * p_z * r_dot_m / r5 - m_z / r2)
    B_global = np.array([B_x, B_y, B_z]) * muT_to_T
    if R_sensor is None:
        R_sensor = np.eye(3)
    B_local = R_sensor.T @ B_global
    return B_local

def residuals_m(m, positions, B_measured, quaternions):
    residuals = []
    for r, B_meas, quat in zip(positions, B_measured, quaternions):
        R_sensor = quaternion_to_rotation_matrix(quat)
        B_pred = dipole_field(r, m, R_sensor)
        residuals.append(B_pred - B_meas)
    return np.ravel(residuals)

def objective_m(m, positions, B_measured, quaternions):
    res = residuals_m(m, positions, B_measured, quaternions)
    return np.sum(res**2)

def residuals_ab(params, m_fixed, positions, B_measured, quaternions):
    a_x, b_x, a_y, b_y, a_z, b_z = params
    residuals = []
    for r, B_meas, quat in zip(positions, B_measured, quaternions):
        R_sensor = quaternion_to_rotation_matrix(quat)
        B_pred = dipole_field(r, m_fixed, R_sensor, a_x, b_x, a_y, b_y, a_z, b_z)
        residuals.append(B_pred - B_meas)
    return np.ravel(residuals)

def objective_ab(params, m_fixed, positions, B_measured, quaternions, lambda_reg=1e-3):
    a_x, b_x, a_y, b_y, a_z, b_z = params
    res = residuals_ab(params, m_fixed, positions, B_measured, quaternions)
    reg = lambda_reg * ((a_x - 1.0)**2 + (a_y - 1.0)**2 + (a_z - 1.0)**2)
    return np.sum(res**2) + reg

def residuals_position(r, B_measured, m, R_sensor, a_x, b_x, a_y, b_y, a_z, b_z):
    B_pred = dipole_field(r, m, R_sensor, a_x, b_x, a_y, b_y, a_z, b_z)
    return B_pred - B_measured

def initial_position_guess(B_measured, m, R_sensor, pos_bounds):
    B_global = R_sensor @ B_measured
    B_mag = np.linalg.norm(B_global)
    m_mag = np.linalg.norm(m)
    if B_mag < 1e-10 or m_mag < 1e-10:
        return np.mean(pos_bounds, axis=0)
    r_est = ((mu_0 / (4 * np.pi)) * (2 * m_mag) / B_mag) ** (1/3)
    B_unit = B_global / B_mag if B_mag > 0 else np.array([1, 0, 0])
    r_guess = r_est * B_unit
    r_guess = np.clip(r_guess, pos_bounds[0], pos_bounds[1])
    return r_guess

def compute_metrics(m, positions, B_measured, quaternions, a_x=1.0, b_x=0.0, a_y=1.0, b_y=0.0, a_z=1.0, b_z=0.0):
    B_pred = np.array([
        dipole_field(r, m, quaternion_to_rotation_matrix(quat), a_x, b_x, a_y, b_y, a_z, b_z)
        for r, quat in zip(positions, quaternions)
    ])
    errors = B_pred - B_measured
    rmse = np.sqrt(np.mean(errors**2)) * 1e6
    angular_errors = []
    for b_p, b_m in zip(B_pred, B_measured):
        if np.linalg.norm(b_m) > 1e-10:
            cos_theta = np.dot(b_p, b_m) / (np.linalg.norm(b_p) * np.linalg.norm(b_m))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta) * 180 / np.pi
            angular_errors.append(theta)
        else:
            angular_errors.append(np.nan)
    mean_angular_error = np.nanmean(angular_errors) if angular_errors else np.nan
    return rmse, mean_angular_error, angular_errors, B_pred * 1e6

def estimate_position_custom(B_custom, quat_sensor, m, pos_bounds, a_x, b_x, a_y, b_y, a_z, b_z):
    B_custom_T = np.array(B_custom) * muT_to_T
    R_sensor = quaternion_to_rotation_matrix(quat_sensor)
    r_init = initial_position_guess(B_custom_T, m, R_sensor, pos_bounds)
    try:
        result = least_squares(residuals_position, r_init,
                              args=(B_custom_T, m, R_sensor, a_x, b_x, a_y, b_y, a_z, b_z),
                              bounds=pos_bounds, method='trf', ftol=1e-8, xtol=1e-8)
        return result.x, r_init
    except Exception as e:
        print(f"Error estimating custom position: {str(e)}")
        return r_init, r_init

def interactive_position_estimation(m, pos_bounds, a_x, b_x, a_y, b_y, a_z, b_z, true_positions=None):
    print("\nInteractive Position Estimation")
    print("Enter magnetic field in µT (e.g., 10.5,-5.2,3.1) and quaternion (e.g., 1,0,0,0)")
    print("Enter 'quit' to exit")
    while True:
        B_input = input("Magnetic field [B_x, B_y, B_z] in µT (comma-separated): ").strip()
        if B_input.lower() == 'quit':
            break
        try:
            B_custom = [float(x) for x in B_input.split(',')]
            if len(B_custom) != 3:
                print("Error: Please enter exactly 3 values for B_x, B_y, B_z")
                continue
            if any(abs(m) > 8000 for m in B_custom):
                print(f"Invalid magnetic field values: {B_custom} exceed ±8000 µT")
                continue
        except ValueError:
            print("Error: Invalid input. Use format like '10.5,-5.2,3.1'")
            continue
        quat_input = input("Sensor quaternion [qw, qx, qy, qz] (comma-separated): ").strip()
        if quat_input.lower() == 'quit':
            break
        try:
            quat_sensor = [float(x) for x in quat_input.split(',')]
            if len(quat_sensor) != 4:
                print("Error: Please enter exactly 4 values for qw, qx, qy, qz")
                continue
            norm = np.sqrt(sum(x**2 for x in quat_sensor))
            if norm < 1e-6:
                print("Error: Invalid quaternion (zero norm)")
                continue
            if not 0.99 <= norm <= 1.01:
                print(f"Invalid quaternion norm: {norm:.3f}")
                continue
            quat_sensor = [x / norm for x in quat_sensor]
        except ValueError:
            print("Error: Invalid input. Use format like '1,0,0,0'")
            continue
        start_time = time.time()
        r_est, r_init = estimate_position_custom(B_custom, quat_sensor, m, pos_bounds, a_x, b_x, a_y, b_y, a_z, b_z)
        estimation_time = time.time() - start_time
        print(f"\nInput: B = {B_custom} µT, Quaternion = {quat_sensor}")
        print(f"Initial Guess: x={r_init[0]:.6f}, y={r_init[1]:.6f}, z={r_init[2]:.6f} m")
        print(f"Estimated Position: x={r_est[0]:.6f}, y={r_est[1]:.6f}, z={r_est[2]:.6f} m")
        if true_positions is not None:
            idx = input("Enter data point index for true position comparison (or press Enter to skip): ").strip()
            try:
                idx = int(idx)
                if 0 <= idx < len(true_positions):
                    true_pos = true_positions[idx]
                    error = np.linalg.norm(r_est - true_pos)
                    print(f"True Position (index {idx}): x={true_pos[0]:.6f}, y={true_pos[1]:.6f}, z={true_pos[2]:.6f} m")
                    print(f"Positional Error: {error:.6f} m")
                else:
                    print("Error: Invalid index")
            except ValueError:
                pass

def load_and_filter_data():
    encodings = ['utf-8', 'latin1', 'windows-1252', 'utf-8-sig']
    data = None
    for encoding in encodings:
        try:
            data = pd.read_csv(CSV_FILE, encoding=encoding)
            break
        except UnicodeDecodeError:
            pass
        except FileNotFoundError:
            print(f"'{CSV_FILE}' not found.")
            return None, None, None
    
    if data is None:
        print("Could not read CSV with any encoding.")
        return None, None, None

    data.columns = data.columns.str.replace('µ', 'u', regex=False)
    data.columns = data.columns.str.replace('μ', 'u', regex=False)

    required_columns = ['Rel_X_m', 'Rel_Y_m', 'Rel_Z_m', 'Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT',
                        'Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing columns in CSV: {missing_columns}")
        print(f"Available columns: {list(data.columns)}")
        return None, None, None

    positions = data[['Rel_X_m', 'Rel_Y_m', 'Rel_Z_m']].values
    B_measured = data[['Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT']].values * muT_to_T
    quaternions = data[['Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz']].values

    for i, q in enumerate(quaternions):
        norm = np.sqrt(np.sum(q**2))
        if norm < 1e-6:
            print(f"Invalid quaternion at index {i} (norm={norm:.6f}). Setting to identity.")
            quaternions[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            quaternions[i] = q / norm

    min_distance = 0.05
    max_B = 10000 * muT_to_T
    mask = (np.linalg.norm(positions, axis=1) > min_distance) & (np.linalg.norm(B_measured, axis=1) < max_B)
    positions = positions[mask]
    B_measured = B_measured[mask]
    quaternions = quaternions[mask]

    return positions, B_measured, quaternions

def run_csv_interactive(dipole_params, pos_bounds, positions):
    positions_data, B_measured, quaternions = load_and_filter_data()
    if positions_data is None:
        print("Failed to load data for CSV interactive estimation.")
        return

    if len(positions_data) < 5:
        print(f"Too few measurements ({len(positions_data)}). At least 5 are recommended.")
        return

    r_estimated = np.zeros_like(positions_data)
    r_initial = np.zeros_like(positions_data)
    pos_errors = np.zeros(len(positions_data))

    for i, (B, quat) in enumerate(zip(B_measured, quaternions)):
        try:
            R_sensor = quaternion_to_rotation_matrix(quat)
            r_init = initial_position_guess(B, dipole_params['m'], R_sensor, pos_bounds)
            r_initial[i] = r_init
            result = least_squares(residuals_position, r_init,
                                  args=(B, dipole_params['m'], R_sensor,
                                        dipole_params['a_x'], dipole_params['b_x'],
                                        dipole_params['a_y'], dipole_params['b_y'],
                                        dipole_params['a_z'], dipole_params['b_z']),
                                  bounds=pos_bounds, method='trf', ftol=1e-8, xtol=1e-8)
            r_estimated[i] = result.x
            pos_errors[i] = np.linalg.norm(r_estimated[i] - positions_data[i])
        except Exception as e:
            print(f"Error estimating position for point {i}: {str(e)}")
            r_estimated[i] = positions_data[i]
            r_initial[i] = positions_data[i]
            pos_errors[i] = 0.0

    mean_pos_error = np.mean(pos_errors)
    std_pos_error = np.std(pos_errors)
    print(f"Position Estimation Results:")
    print(f"  Mean Positional Error = {mean_pos_error:.6f} m")
    print(f"  Std Positional Error = {std_pos_error:.6f} m")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions_data[:, 0], positions_data[:, 1], positions_data[:, 2], c='b', label='Actual Positions', alpha=0.5)
    ax.scatter(r_estimated[:, 0], r_estimated[:, 1], r_estimated[:, 2], c='r', label='Estimated Positions', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Actual vs Estimated Positions')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pos_errors, bins=20)
    ax.set_xlabel('Positional Error (m)')
    ax.set_title('Positional Errors')
    plt.close()

def run_optimization():
    print("Starting optimization...")
    positions, B_measured, quaternions = load_and_filter_data()
    if positions is None:
        print("Failed to load data for optimization.")
        return None, None

    N = len(positions)
    if N < 5:
        print(f"Too few measurements ({N}). At least 5 are recommended.")
        return None, None

    print(f"Loaded {N} data points for optimization.")

    # Initial dipole moment guess
    m0 = np.array([-0.1, 0, 0])
    for i, r in enumerate(positions):
        x, y, z = r
        if abs(y) < 0.01 and abs(z) < 0.01:
            B_local = B_measured[i]
            R_sensor = quaternion_to_rotation_matrix(quaternions[i])
            B_global = R_sensor @ B_local
            B_x = B_global[0] * 1e6
            m0_x = (B_x * abs(x)**3) / (factor * 2)
            if -1000 <= m0_x <= 0:
                m0[0] = m0_x
            break
    print(f"Initial dipole moment guess: m = {m0}")

    methods = ['TRF', 'Dogbox', 'LM', 'Differential Evolution', 'Nelder-Mead', 'Neural Network']
    results = []
    bounds_m = ([-1000, -1000, -1000], [0, 1000, 1000])

    # Stage 1: Optimize dipole moment m
    for method in methods:
        print(f"\nRunning {method} optimization...")
        start_time = time.time()
        try:
            if method == 'TRF':
                result = least_squares(residuals_m, m0, args=(positions, B_measured, quaternions),
                                      bounds=bounds_m, method='trf', ftol=1e-8, xtol=1e-8)
                m_best = result.x
            elif method == 'Dogbox':
                result = least_squares(residuals_m, m0, args=(positions, B_measured, quaternions),
                                      bounds=bounds_m, method='dogbox', ftol=1e-8, xtol=1e-8)
                m_best = result.x
            elif method == 'LM':
                result = least_squares(residuals_m, m0, args=(positions, B_measured, quaternions),
                                      method='lm', ftol=1e-8, xtol=1e-8)
                m_best = np.clip(result.x, bounds_m[0], bounds_m[1])
            elif method == 'Differential Evolution':
                bounds_de = [(-1000, 0), (-1000, 1000), (-1000, 1000)]
                result = differential_evolution(objective_m, bounds_de, args=(positions, B_measured, quaternions),
                                               strategy='best1bin', popsize=15, tol=1e-6)
                m_best = result.x
            elif method == 'Nelder-Mead':
                result = minimize(objective_m, m0, args=(positions, B_measured, quaternions),
                                 method='Nelder-Mead',
                                 options={'xatol': 1e-6, 'fatol': 1e-6})
                m_best = np.clip(result.x, bounds_m[0], bounds_m[1])
            elif method == 'Neural Network':
                # Neural Network optimization
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(10,)),  # 3 pos + 4 quat + 3 B_meas
                    Dense(32, activation='relu'),
                    Dense(3)  # Predict m_x, m_y, m_z
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                
                # Prepare input data: [x, y, z, qw, qx, qy, qz, Bx, By, Bz]
                X = np.hstack((positions, quaternions, B_measured * 1e6))
                y = np.tile(m0, (N, 1))  # Initial target is m0 repeated
                model.fit(X, y, epochs=50, batch_size=32, verbose=0,
                          callbacks=[tqdm_callback(N, 'Neural Network Training')])
                m_best = model.predict(X[-1:], verbose=0)[0]
                m_best = np.clip(m_best, bounds_m[0], bounds_m[1])
            
            time_taken = time.time() - start_time
            rmse, ang_err, ang_errs, B_pred = compute_metrics(m_best, positions, B_measured, quaternions)
            results.append({
                'Method': method,
                'm': m_best,
                'RMSE': rmse,
                'Angular Error': ang_err,
                'Time': time_taken,
                'Angular Errors': ang_errs,
                'B_pred': B_pred
            })
            print(f"{method} completed: RMSE = {rmse:.6f} µT, Angular Error = {ang_err:.6f}°, Time = {time_taken:.2f}s")
        except Exception as e:
            print(f"Error in {method}: {str(e)}")
            results.append({
                'Method': method,
                'm': None,
                'RMSE': np.inf,
                'Angular Error': np.inf,
                'Time': np.inf,
                'Angular Errors': None,
                'B_pred': None
            })

    valid_results = [r for r in results if r['m'] is not None]
    if not valid_results:
        print("No optimization methods succeeded in first stage.")
        return None, None
    best_result = min(valid_results, key=lambda x: (x['RMSE'], x['Angular Error']))
    m_best = best_result['m']
    rmse_best = best_result['RMSE']
    ang_err_best = best_result['Angular Error']
    ang_errs_best = best_result['Angular Errors']
    B_pred_best = best_result['B_pred']
    best_method = best_result['Method']
    print(f"\nBest Stage 1 Method: {best_method}, RMSE = {rmse_best:.6f} µT, Angular Error = {ang_err_best:.6f}°")

    # Stage 2: Optimize a, b parameters
    print("\nRunning Second Stage (Nelder-Mead for a, b parameters)...")
    start_time = time.time()
    ab0 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    try:
        result_ab = minimize(objective_ab, ab0, args=(m_best, positions, B_measured, quaternions, 1e-3),
                            method='Nelder-Mead',
                            bounds=[(0.1, 10), (-1, 1), (0.1, 10), (-1, 1), (0.1, 10), (-1, 1)],
                            options={'xatol': 1e-6, 'fatol': 1e-6})
        a_x_best, b_x_best, a_y_best, b_y_best, a_z_best, b_z_best = result_ab.x
        time_taken = time.time() - start_time
        rmse_final, ang_err_final, ang_errs_final, B_pred_final = compute_metrics(
            m_best, positions, B_measured, quaternions, a_x_best, b_x_best, a_y_best, b_y_best, a_z_best, b_z_best)
        print(f"Second Stage completed: RMSE = {rmse_final:.6f} µT, Angular Error = {ang_err_final:.6f}°, Time = {time_taken:.2f}s")
    except Exception as e:
        print(f"Error in Second Stage (Nelder-Mead): {str(e)}")
        a_x_best, b_x_best, a_y_best, b_y_best, a_z_best, b_z_best = 1.0, 0.0, 1.0, 0.0, 1.0, 0.0
        rmse_final, ang_err_final, ang_errs_final, B_pred_final = rmse_best, ang_err_best, ang_errs_best, B_pred_best

    # Stage 3: Refine m with fixed a, b
    print("\nRunning Third Stage (TRF for m refinement)...")
    start_time = time.time()
    try:
        def residuals_m_third(m, positions, B_measured, quaternions, a_x, b_x, a_y, b_y, a_z, b_z):
            residuals = []
            for r, B_meas, quat in zip(positions, B_measured, quaternions):
                R_sensor = quaternion_to_rotation_matrix(quat)
                B_pred = dipole_field(r, m, R_sensor, a_x, b_x, a_y, b_y, a_z, b_z)
                residuals.append(B_pred - B_meas)
            return np.ravel(residuals)
        
        result_m_third = least_squares(residuals_m_third, m_best,
                                      args=(positions, B_measured, quaternions, a_x_best, b_x_best, a_y_best, b_y_best, a_z_best, b_z_best),
                                      bounds=bounds_m, method='trf', ftol=1e-8, xtol=1e-8)
        m_third = result_m_third.x
        time_taken = time.time() - start_time
        rmse_third, ang_err_third, ang_errs_third, B_pred_third = compute_metrics(
            m_third, positions, B_measured, quaternions, a_x_best, b_x_best, a_y_best, b_y_best, a_z_best, b_z_best)
        print(f"Third Stage completed: RMSE = {rmse_third:.6f} µT, Angular Error = {ang_err_third:.6f}°, Time = {time_taken:.2f}s")
    except Exception as e:
        print(f"Error in Third Stage (TRF): {str(e)}")
        m_third = m_best
        rmse_third, ang_err_final, ang_errs_third, B_pred_third = rmse_final, ang_err_final, ang_errs_final, B_pred_final

    # Select best results
    if rmse_third < rmse_final:
        m_final = m_third
        rmse_final = rmse_third
        ang_err_final = ang_err_third
        ang_errs_final = ang_errs_third
        B_pred_final = B_pred_third
        best_stage = "Third Stage (TRF)"
    else:
        m_final = m_best
        best_stage = f"Second Stage ({best_method})"
    print(f"\nFinal Best Stage: {best_stage}, RMSE = {rmse_final:.6f} µT, Angular Error = {ang_err_final:.6f}°")

    dipole_params = {
        'm': m_final,
        'a_x': a_x_best,
        'b_x': b_x_best,
        'a_y': a_y_best,
        'b_y': b_y_best,
        'a_z': a_z_best,
        'b_z': b_z_best
    }

    params = {
        'm_x': float(m_final[0]),
        'm_y': float(m_final[1]),
        'm_z': float(m_final[2]),
        'a_x': float(a_x_best),
        'b_x': float(b_x_best),
        'a_y': float(a_y_best),
        'b_y': float(b_y_best),
        'a_z': float(a_z_best),
        'b_z': float(b_z_best),
        'rmse': float(rmse_final),
        'angular_error': float(ang_err_final)
    }
    with open('dipole_model_params.json', 'w') as f:
        json.dump(params, f, indent=4)
    print("Dipole parameters saved to 'dipole_model_params.json'.")

    pos_bounds = (
        [positions[:,0].min(), positions[:,1].min(), positions[:,2].min()],
        [positions[:,0].max(), positions[:,1].max(), positions[:,2].max()]
    )

    # Generate and show plots only after optimization
    print("\nGenerating visualization plots...")
    generate_vector_plots(positions, B_measured, quaternions, dipole_params, best_stage, ang_errs_final, B_pred_final)

    return dipole_params, pos_bounds, positions, B_measured, quaternions, ang_errs_final, B_pred_final

def generate_vector_plots(positions, B_measured, quaternions, dipole_params, best_stage, ang_errs_final, B_pred_final):
    print("Creating vector field visualizations...")
    N = len(positions)
    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    z_vals = positions[:, 2]
    U_meas, V_meas, W_meas = np.zeros(N), np.zeros(N), np.zeros(N)
    U_theo, V_theo, W_theo = np.zeros(N), np.zeros(N), np.zeros(N)
    U_opt, V_opt, W_opt = np.zeros(N), np.zeros(N), np.zeros(N)

    for i in tqdm(range(N), desc="Computing vector fields"):
        r, B_m, quat = positions[i], B_measured[i], quaternions[i]
        R_sensor = quaternion_to_rotation_matrix(quat)
        B_theo = dipole_field(r, np.array([-0.1, 0, 0]), R_sensor) * 1e6
        B_opt = dipole_field(r, dipole_params['m'], R_sensor,
                             dipole_params['a_x'], dipole_params['b_x'],
                             dipole_params['a_y'], dipole_params['b_y'],
                             dipole_params['a_z'], dipole_params['b_z']) * 1e6
        U_meas[i], V_meas[i], W_meas[i] = B_m[0] * 1e6, B_m[1] * 1e6, B_m[2] * 1e6
        U_theo[i], V_theo[i], W_theo[i] = B_theo[0], B_theo[1], B_theo[2]
        U_opt[i], V_opt[i], W_opt[i] = B_opt[0], B_opt[1], B_opt[2]

    max_magnitude = max(
        np.sqrt(U_meas**2 + V_meas**2 + W_meas**2).max(),
        np.sqrt(U_theo**2 + V_theo**2 + W_theo**2).max(),
        np.sqrt(U_opt**2 + W_opt**2 + W_opt**2).max()
    )
    if max_magnitude > 0:
        U_meas, V_meas, W_meas = U_meas / max_magnitude, V_meas / max_magnitude, W_meas / max_magnitude
        U_theo, V_theo, W_theo = U_theo / max_magnitude, V_theo / max_magnitude, W_theo / max_magnitude
        U_opt, V_opt, W_opt = U_opt / max_magnitude, V_opt / max_magnitude, W_opt / max_magnitude
    else:
        print("Zero field magnitude encountered. Vectors not scaled.")

    # 2D Vector Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(x_vals, z_vals, c=B_measured[:, 0] * 1e6, cmap='viridis', edgecolors='k', label='Measured B_x (µT)')
    fig.colorbar(sc, ax=ax, label='B_x (µT)')
    try:
        q1 = ax.quiver(x_vals, z_vals, U_meas, W_meas, color='green', label='Measured Vectors',
                       scale=ARROW_SCALE, width=ARROW_WIDTH)
        q2 = ax.quiver(x_vals, z_vals, U_theo, W_theo, color='red', label='Theoretical Vectors',
                       scale=ARROW_SCALE, width=ARROW_WIDTH)
        q3 = ax.quiver(x_vals, z_vals, U_opt, W_opt, color='blue', label='Optimized Vectors',
                       scale=ARROW_SCALE, width=ARROW_WIDTH)
    except AttributeError:
        ax.clear()
        sc = ax.scatter(x_vals, z_vals, c=B_measured[:, 0] * 1e6, cmap='viridis', edgecolors='k', label='Measured B_x (µT)')
        fig.colorbar(sc, ax=ax, label='B_x (µT)')
        q1 = ax.quiver(x_vals, z_vals, U_meas, W_meas, color='green', label='Measured Vectors',
                       scale=ARROW_SCALE, width=ARROW_WIDTH)
        q2 = ax.quiver(x_vals, z_vals, U_theo, W_theo, color='red', label='Theoretical Vectors',
                       scale=ARROW_SCALE, width=ARROW_WIDTH)
        q3 = ax.quiver(x_vals, z_vals, U_opt, W_opt, color='blue', label='Optimized Vectors',
                       scale=ARROW_SCALE, width=ARROW_WIDTH)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'Magnetic Field Vectors (2D, Best Method: {best_stage}, m = {dipole_params["m"].round(3)} A·m²)')
    ax.legend()
    ax.grid(True)
    ref_length = 0.5
    ref_magnitude = ref_length * max_magnitude
    ax.quiverkey(q1, X=0.85, Y=0.95, U=ref_length, label=f'{ref_magnitude:.0f} µT', labelpos='E', coordinates='figure')
    plt.show()

    # 3D Vector Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_vals, y_vals, z_vals, c=B_measured[:, 0] * 1e6, cmap='viridis', edgecolors='k', label='Measured B_x (µT)')
    fig.colorbar(sc, ax=ax, label='B_x (µT)')
    try:
        q1 = ax.quiver(x_vals, y_vals, z_vals, U_meas, V_meas, W_meas, color='green', label='Measured Vectors',
                       length=0.05, arrow_length_ratio=0.3, linewidths=1)
        q2 = ax.quiver(x_vals, y_vals, z_vals, U_theo, V_theo, W_theo, color='red', label='Theoretical Vectors',
                       length=0.05, arrow_length_ratio=0.3, linewidths=1)
        q3 = ax.quiver(x_vals, y_vals, z_vals, U_opt, V_opt, W_opt, color='blue', label='Optimized Vectors',
                       length=0.05, arrow_length_ratio=0.3, linewidths=1)
    except AttributeError:
        ax.clear()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x_vals, y_vals, z_vals, c=B_measured[:, 0] * 1e6, cmap='viridis', edgecolors='k', label='Measured B_x (µT)')
        fig.colorbar(sc, ax=ax, label='B_x (µT)')
        q1 = ax.quiver(x_vals, y_vals, z_vals, U_meas, V_meas, W_meas, color='green', label='Measured Vectors',
                       length=0.05, arrow_length_ratio=0.3, linewidths=1)
        q2 = ax.quiver(x_vals, y_vals, z_vals, U_theo, V_theo, W_theo, color='red', label='Theoretical Vectors',
                       length=0.05, arrow_length_ratio=0.3, linewidths=1)
        q3 = ax.quiver(x_vals, y_vals, z_vals, U_opt, V_opt, W_opt, color='blue', label='Optimized Vectors',
                       length=0.05, arrow_length_ratio=0.3, linewidths=1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Magnetic Field Vectors (3D, Best Method: {best_stage}, m = {dipole_params["m"].round(3)} A·m²)')
    ax.legend()
    ax.grid(True)
    ax.quiver(0, 0, 0, ref_length, 0, 0, color='black', label=f'Reference: {ref_magnitude:.0f} µT',
              arrow_length_ratio=0.3, linewidths=1)
    plt.show()

    # Removed separate residual and angular error plots to avoid saving

# Custom tqdm callback for Keras
class TqdmCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_samples, desc):
        super().__init__()
        self.total_samples = total_samples
        self.desc = desc
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=50, desc=self.desc)  # Assuming 50 epochs

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        self.pbar.set_postfix(loss=logs.get('loss'))

    def on_train_end(self, logs=None):
        self.pbar.close()

def tqdm_callback(total_samples, desc):
    return TqdmCallback(total_samples, desc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Position estimation server or interactive modes")
    parser.add_argument('--optimize', action='store_true', help="Run optimization to generate dipole parameters")
    parser.add_argument('--interactive', action='store_true', help="Run CSV-based position estimation with plotting")
    parser.add_argument('--console', action='store_true', help="Run console-based interactive position estimation")
    args = parser.parse_args()

    dipole_params = load_dipole_params()
    positions, B_measured, quaternions = load_and_filter_data()
    pos_bounds = None
    best_stage = "Existing Parameters"
    ang_errs_final = None
    B_pred_final = None

    if positions is not None:
        pos_bounds = (
            [positions[:,0].min(), positions[:,1].min(), positions[:,2].min()-0.1],
            [positions[:,0].max(), positions[:,1].max(), positions[:,2].max()+0.1]
        )
        print(f"Position bounds set to: {pos_bounds}")

    if args.optimize:
        result = run_optimization()
        if result is None:
            print("Optimization failed.")
            exit(1)
        dipole_params, pos_bounds, positions, B_measured, quaternions, ang_errs_final, B_pred_final = result

    if dipole_params is None:
        print("No valid dipole parameters. Run with --optimize to generate them or ensure 'dipole_model_params.json' exists.")
        exit(1)

    if args.console:
        server_thread = threading.Thread(target=run_position_server, args=(POSITION_SERVER_HOST, POSITION_SERVER_PORT, dipole_params, pos_bounds))
        server_thread.daemon = True
        server_thread.start()
        interactive_position_estimation(
            dipole_params['m'], pos_bounds,
            dipole_params['a_x'], dipole_params['b_x'],
            dipole_params['a_y'], dipole_params['b_y'],
            dipole_params['a_z'], dipole_params['b_z'],
            positions
        )
    elif args.interactive:
        server_thread = threading.Thread(target=run_position_server, args=(POSITION_SERVER_HOST, POSITION_SERVER_PORT, dipole_params, pos_bounds))
        server_thread.daemon = True
        server_thread.start()
        run_csv_interactive(dipole_params, pos_bounds, positions)
    else:
        run_position_server(POSITION_SERVER_HOST, POSITION_SERVER_PORT, dipole_params, pos_bounds)