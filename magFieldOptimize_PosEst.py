import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution, minimize
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
import joblib
from collections import deque
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# --- Configuration ---
POSITION_SERVER_HOST = 'localhost'
POSITION_SERVER_PORT = 12346
CSV_FILE = 'new_1cm.csv'
STOP_EVENT = threading.Event()
ARROW_SCALE = 5
ARROW_WIDTH = 0.002
max_Bfield = 20000  # Maximum magnetic field in µT
MODEL_FILE = 'correction_models.pkl'
COMPARISON_LOG = 'ml_comparison_log.csv'

POSITION_TIMING_PERIODS = deque(maxlen=100)
POSITION_TIMING_LOCK = threading.Lock()
POSITION_TIMING_LAST_PRINT = 0.0

# Constants
mu_0 = 4 * np.pi * 1e-7
muT_to_T = 1e-6
factor = mu_0 / (4 * np.pi) * 1e6

# --- Model Save/Load Functions ---
def save_correction_models(xgb_x, xgb_y, xgb_z):
    """Save ML correction models to file"""
    if xgb_x is not None and xgb_y is not None and xgb_z is not None:
        try:
            joblib.dump({'x': xgb_x, 'y': xgb_y, 'z': xgb_z}, MODEL_FILE)
            print(f"✅ ML models saved to {MODEL_FILE}")
            return True
        except Exception as e:
            print(f"❌ Failed to save ML models: {e}")
            return False
    return False

def load_correction_models():
    """Load ML correction models from file"""
    if os.path.exists(MODEL_FILE):
        try:
            models = joblib.load(MODEL_FILE)
            print(f"✅ ML models loaded from {MODEL_FILE}")
            return models['x'], models['y'], models['z']
        except Exception as e:
            print(f"❌ Failed to load ML models: {e}")
    return None, None, None

# --- Comparison Logging ---
def init_comparison_log():
    """Initialize or load comparison log"""
    if not os.path.exists(COMPARISON_LOG):
        df = pd.DataFrame(columns=[
            'timestamp', 'Bx_uT', 'By_uT', 'Bz_uT',
            'qw', 'qx', 'qy', 'qz',
            'physics_x', 'physics_y', 'physics_z',
            'ml_x', 'ml_y', 'ml_z',
            'physics_error', 'ml_error',
            'true_x', 'true_y', 'true_z'
        ])
        df.to_csv(COMPARISON_LOG, index=False)
        print(f"✅ Created comparison log: {COMPARISON_LOG}")

def format_position_timing_summary(step_durations_ms, avg_hz):
    """Format a compact timing summary for position estimation requests."""
    parts = [f"{name}={value:.2f}ms" for name, value in step_durations_ms.items()]
    return "Position timing | " + " | ".join(parts) + f" | avg_hz={avg_hz:.2f}"


def log_comparison_result(B_custom, quat_sensor, r_physics, r_ml, true_pos=None):
    """Log comparison between physics-only and ML-corrected estimates"""
    try:
        # Load existing log
        if os.path.exists(COMPARISON_LOG):
            df = pd.read_csv(COMPARISON_LOG)
        else:
            df = pd.DataFrame(columns=[
                'timestamp', 'Bx_uT', 'By_uT', 'Bz_uT',
                'qw', 'qx', 'qy', 'qz',
                'physics_x', 'physics_y', 'physics_z',
                'ml_x', 'ml_y', 'ml_z',
                'physics_error', 'ml_error',
                'true_x', 'true_y', 'true_z'
            ])
        
        # Calculate errors if true position available
        physics_error = np.linalg.norm(r_physics - true_pos) if true_pos is not None else np.nan
        ml_error = np.linalg.norm(r_ml - true_pos) if true_pos is not None else np.nan
        
        # Create new row
        new_row = {
            'timestamp': datetime.now().isoformat(),
            'Bx_uT': B_custom[0], 'By_uT': B_custom[1], 'Bz_uT': B_custom[2],
            'qw': quat_sensor[0], 'qx': quat_sensor[1], 'qy': quat_sensor[2], 'qz': quat_sensor[3],
            'physics_x': r_physics[0], 'physics_y': r_physics[1], 'physics_z': r_physics[2],
            'ml_x': r_ml[0], 'ml_y': r_ml[1], 'ml_z': r_ml[2],
            'physics_error': physics_error, 'ml_error': ml_error,
            'true_x': true_pos[0] if true_pos is not None else np.nan,
            'true_y': true_pos[1] if true_pos is not None else np.nan,
            'true_z': true_pos[2] if true_pos is not None else np.nan
        }
        
        # Append and save
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(COMPARISON_LOG, index=False)
        
        # Print comparison if true position available
        if true_pos is not None:
            improvement = ((physics_error - ml_error) / physics_error * 100) if physics_error > 0 else 0
            print(f"\n📊 ML Comparison:")
            print(f"   Physics error: {physics_error*1000:.2f} mm")
            print(f"   ML error: {ml_error*1000:.2f} mm")
            print(f"   Improvement: {improvement:.1f}%")
            
    except Exception as e:
        print(f"❌ Failed to log comparison: {e}")
# Add this function after load_and_filter_data() and before run_optimization_enhanced()

def split_data_for_staged_optimization(positions, B_measured, quaternions, data_df, fraction=0.33, random_state=42):
    """
    Split data into two sets:
    - optimization_set: fraction of data for multipole optimization
    - ml_training_set: full dataset for ML training (including optimization set)
    """
    n_samples = len(positions)
    n_opt = int(n_samples * fraction)
    
    # Randomly select indices for optimization
    np.random.seed(random_state)
    opt_indices = np.random.choice(n_samples, n_opt, replace=False)
    
    # Create optimization subset
    opt_positions = positions[opt_indices]
    opt_B_measured = B_measured[opt_indices]
    opt_quaternions = quaternions[opt_indices]
    opt_data = data_df.iloc[opt_indices].copy() if data_df is not None else None
    
    print(f"\n📊 Data Split:")
    print(f"   Total samples: {n_samples}")
    print(f"   Optimization set: {n_opt} ({fraction*100:.0f}%)")
    print(f"   ML training set: {n_samples} (100% of data)")
    
    return opt_positions, opt_B_measured, opt_quaternions, opt_data, opt_indices


def run_optimization_staged(fraction=0.33):
    """
    Run staged optimization: 
    - Use fraction of data for dipole parameter optimization (faster)
    - Use ALL data for ML training (more accurate)
    """
    print("\n" + "="*60)
    print("STAGED OPTIMIZATION MODE")
    print("="*60)
    print(f"Using {fraction*100:.0f}% of data for dipole optimization")
    print(f"Using 100% of data for ML training\n")
    
    # Load all data
    positions, B_measured, quaternions, data_df = load_and_filter_data()
    if positions is None:
        print("Failed to load data.")
        return None, None, None, None, None, None, None, None
    
    N = len(positions)
    print(f"Total data points available: {N}")
    
    # Split data for optimization (use only fraction)
    opt_positions, opt_B_measured, opt_quaternions, opt_data, opt_indices = split_data_for_staged_optimization(
        positions, B_measured, quaternions, data_df, fraction=fraction
    )
    
    N_opt = len(opt_positions)
    print(f"\n🔧 Running dipole optimization on {N_opt} points ({fraction*100:.0f}% of data)...")
    
    # First stage: optimize dipole moment m using subset
    m0 = np.array([-0.1, 0, 0])
    
    # Try to get a better initial guess
    for i, r in enumerate(opt_positions):
        x, y, z = r
        if abs(y) < 0.01 and abs(z) < 0.01:
            B_local = opt_B_measured[i]
            R_sensor = quaternion_to_rotation_matrix(opt_quaternions[i])
            B_global = R_sensor @ B_local
            B_x = B_global[0] * 1e6
            m0_x = (B_x * abs(x)**3) / (factor * 2)
            if -1000 <= m0_x <= 0:
                m0[0] = m0_x
                print(f"  Initial guess from on-axis point: m_x = {m0_x:.6f}")
                break
    
    bounds_m = ([-1000, -1000, -1000], [0, 1000, 1000])
    
    print("\n📈 Stage 1: Optimizing dipole moment on subset...")
    start_time = time.time()
    
    try:
        # Use TRF method for robust optimization
        result = least_squares(
            residuals_m, m0, 
            args=(opt_positions, opt_B_measured, opt_quaternions),
            bounds=bounds_m, 
            method='trf', 
            ftol=1e-8, 
            xtol=1e-8
        )
        m_best = result.x
        optimization_time = time.time() - start_time
        
        print(f"  ✅ Dipole optimization completed in {optimization_time:.2f}s")
        print(f"  Optimized dipole: m = [{m_best[0]:.6f}, {m_best[1]:.6f}, {m_best[2]:.6f}] A·m²")
        
    except Exception as e:
        print(f"  ❌ Dipole optimization failed: {e}")
        return None, None, None, None, None, None, None, None
    
    # Stage 2: Optimize quadratic coefficients with fixed dipole moment (still on subset)
    print("\n📈 Stage 2: Optimizing quadratic coefficients on subset...")
    start_time = time.time()
    
    abc0 = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]  # [a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z]
    
    abc_bounds = [
        (-10, 10), (-10, 10), (-1, 1),  # x coefficients
        (-10, 10), (-10, 10), (-1, 1),  # y coefficients  
        (-10, 10), (-10, 10), (-1, 1)   # z coefficients
    ]
    
    try:
        result_abc = minimize(
            objective_abc, abc0, 
            args=(m_best, opt_positions, opt_B_measured, opt_quaternions, 1e-3),
            method='Nelder-Mead',
            bounds=abc_bounds,
            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000}
        )
        
        a_x_best, b_x_best, c_x_best, a_y_best, b_y_best, c_y_best, a_z_best, b_z_best, c_z_best = result_abc.x
        optimization_time = time.time() - start_time
        
        print(f"  ✅ Quadratic optimization completed in {optimization_time:.2f}s")
        print(f"  X transform: {a_x_best:.6f}·x² + {b_x_best:.6f}·x + {c_x_best:.6f}")
        print(f"  Y transform: {a_y_best:.6f}·y² + {b_y_best:.6f}·y + {c_y_best:.6f}")
        print(f"  Z transform: {a_z_best:.6f}·z² + {b_z_best:.6f}·z + {c_z_best:.6f}")
        
    except Exception as e:
        print(f"  ❌ Quadratic optimization failed: {e}")
        # Fall back to linear model
        a_x_best, b_x_best, c_x_best = 0.0, 1.0, 0.0
        a_y_best, b_y_best, c_y_best = 0.0, 1.0, 0.0
        a_z_best, b_z_best, c_z_best = 0.0, 1.0, 0.0
    
    # Stage 3: Re-optimize dipole with quadratic coefficients (still on subset)
    print("\n📈 Stage 3: Final dipole refinement with quadratic coefficients...")
    start_time = time.time()
    
    try:
        def residuals_m_with_quad(m, positions, B_measured, quaternions, 
                                   a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z):
            residuals = []
            for r, B_meas, quat in zip(positions, B_measured, quaternions):
                R_sensor = quaternion_to_rotation_matrix(quat)
                B_pred = dipole_field_enhanced(r, m, R_sensor, 
                                               a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z)
                residuals.append(B_pred - B_meas)
            return np.ravel(residuals)
        
        result_m_final = least_squares(
            residuals_m_with_quad, m_best,
            args=(opt_positions, opt_B_measured, opt_quaternions,
                  a_x_best, b_x_best, c_x_best,
                  a_y_best, b_y_best, c_y_best,
                  a_z_best, b_z_best, c_z_best),
            bounds=bounds_m, 
            method='trf', 
            ftol=1e-8, 
            xtol=1e-8
        )
        m_final = result_m_final.x
        optimization_time = time.time() - start_time
        
        print(f"  ✅ Final refinement completed in {optimization_time:.2f}s")
        print(f"  Final dipole: m = [{m_final[0]:.6f}, {m_final[1]:.6f}, {m_final[2]:.6f}] A·m²")
        
    except Exception as e:
        print(f"  ❌ Final refinement failed: {e}")
        m_final = m_best
    
    # Compute metrics on subset (to verify optimization)
    rmse_subset, ang_err_subset, ang_errs_subset, B_pred_subset = compute_metrics_enhanced(
        m_final, opt_positions, opt_B_measured, opt_quaternions,
        a_x_best, b_x_best, c_x_best, a_y_best, b_y_best, c_y_best, a_z_best, b_z_best, c_z_best
    )
    
    print(f"\n📊 Subset Performance (on {N_opt} points):")
    print(f"  RMSE: {rmse_subset:.2f} µT")
    print(f"  Angular Error: {ang_err_subset:.2f}°")
    
    # Compute metrics on FULL dataset (to see generalization)
    print(f"\n📊 Validating on FULL dataset ({N} points)...")
    
    rmse_full, ang_err_full, ang_errs_full, B_pred_full = compute_metrics_enhanced(
        m_final, positions, B_measured, quaternions,
        a_x_best, b_x_best, c_x_best, a_y_best, b_y_best, c_y_best, a_z_best, b_z_best, c_z_best
    )
    
    print(f"  RMSE: {rmse_full:.2f} µT")
    print(f"  Angular Error: {ang_err_full:.2f}°")
    
    if rmse_full < rmse_subset:
        print(f"\n✨ Model generalizes well! Full dataset RMSE improved.")
    else:
        print(f"\n⚠️  Model generalization gap: {rmse_full - rmse_subset:.2f} µT")
        print(f"   Consider increasing fraction (currently {fraction*100:.0f}%)")
    
    # Create dipole parameters dictionary
    dipole_params = {
        'm': m_final,
        'a_x': a_x_best, 'b_x': b_x_best, 'c_x': c_x_best,
        'a_y': a_y_best, 'b_y': b_y_best, 'c_y': c_y_best,
        'a_z': a_z_best, 'b_z': b_z_best, 'c_z': c_z_best
    }
    
    # Save parameters
    params = {
        'm_x': float(m_final[0]),
        'm_y': float(m_final[1]),
        'm_z': float(m_final[2]),
        'a_x': float(a_x_best), 'b_x': float(b_x_best), 'c_x': float(c_x_best),
        'a_y': float(a_y_best), 'b_y': float(b_y_best), 'c_y': float(c_y_best),
        'a_z': float(a_z_best), 'b_z': float(b_z_best), 'c_z': float(c_z_best),
        'rmse_subset': float(rmse_subset),
        'rmse_full': float(rmse_full),
        'angular_error': float(ang_err_full),
        'optimization_fraction': fraction,
        'optimization_points': N_opt,
        'total_points': N
    }
    
    with open('dipole_model_params.json', 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"\n💾 Parameters saved to dipole_model_params.json")
    
    pos_bounds = (
        [positions[:,0].min(), positions[:,1].min(), positions[:,2].min()],
        [positions[:,0].max(), positions[:,1].max(), positions[:,2].max()]
    )
    
    # NOW train ML models on FULL dataset (all data)
    print("\n" + "="*60)
    print("ML TRAINING ON FULL DATASET")
    print("="*60)
    print(f"Training ML correction models using ALL {N} data points...")
    
    # First, update the CSV with estimates using the optimized model on ALL data
    print("\n📝 Generating position estimates for all data points...")
    
    r_estimated = np.zeros_like(positions)
    r_initial = np.zeros_like(positions)
    
    for i, (r_true, B, quat) in enumerate(zip(positions, B_measured, quaternions)):
        B_uT = B / muT_to_T
        r_est, r_init = estimate_position_custom(
            B_uT, quat, dipole_params['m'], pos_bounds,
            dipole_params['a_x'], dipole_params['b_x'], dipole_params['c_x'],
            dipole_params['a_y'], dipole_params['b_y'], dipole_params['c_y'],
            dipole_params['a_z'], dipole_params['b_z'], dipole_params['c_z']
        )
        r_estimated[i] = r_est
        r_initial[i] = r_init
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{N} points...")
    
    # Update DataFrame with estimates
    if data_df is not None:
        data_df['Est_Xm'] = r_estimated[:, 0]
        data_df['Est_Ym'] = r_estimated[:, 1]
        data_df['Est_Zm'] = r_estimated[:, 2]
        data_df['Error_Xm'] = r_estimated[:, 0] - positions[:, 0]
        data_df['Error_Ym'] = r_estimated[:, 1] - positions[:, 1]
        data_df['Error_Zm'] = r_estimated[:, 2] - positions[:, 2]
        
        # Save updated CSV
        try:
            data_df.to_csv(CSV_FILE, index=False, encoding='utf-8')
            print(f"\n✅ Updated {CSV_FILE} with position estimates")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    # Now train ML models on ALL data
    print("\n🎯 Training ML correction models on ALL data...")
    xgb_models = train_correction_model(data_df, save=True)
    
    # Apply correction model to update CSV
    if xgb_models[0] is not None:
        print("\n📝 Applying ML correction to all data points...")
        data_df = apply_correction_model(data_df, xgb_models[0], xgb_models[1], xgb_models[2])
        
        try:
            data_df.to_csv(CSV_FILE, index=False, encoding='utf-8')
            print(f"✅ Updated {CSV_FILE} with ML-corrected positions")
        except Exception as e:
            print(f"Error saving corrected CSV: {e}")
        
        # Compute final statistics
        original_errors = np.linalg.norm(r_estimated - positions, axis=1)
        corrected_errors = np.linalg.norm(data_df[['Corrected_Xm', 'Corrected_Ym', 'Corrected_Zm']].values - positions, axis=1)
        
        print("\n📊 Final Position Estimation Statistics:")
        print(f"  Original mean error: {np.mean(original_errors)*1000:.2f} mm")
        print(f"  Original std error: {np.std(original_errors)*1000:.2f} mm")
        print(f"  ML-corrected mean error: {np.mean(corrected_errors)*1000:.2f} mm")
        print(f"  ML-corrected std error: {np.std(corrected_errors)*1000:.2f} mm")
        improvement = (np.mean(original_errors) - np.mean(corrected_errors)) / np.mean(original_errors) * 100
        print(f"  Improvement: {improvement:.1f}%")
    
    return dipole_params, pos_bounds, positions, B_measured, quaternions, ang_errs_full, B_pred_full, "Staged Optimization"



# --- TCP Server for Position Estimation ---
def run_position_server(host=POSITION_SERVER_HOST, port=POSITION_SERVER_PORT, 
                       dipole_params=None, pos_bounds=None, xgb_models=None,
                       enable_ml=True):
    """Enhanced server with optional ML correction"""
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        server_socket.settimeout(0.1)
        
        print(f"\n🚀 Position Server Started on {host}:{port}")
        print(f"   ML Correction: {'ENABLED' if enable_ml and xgb_models[0] is not None else 'DISABLED'}")
        print(f"   Comparison logging: {COMPARISON_LOG}\n")
        
        while not STOP_EVENT.is_set():
            try:
                client_socket, addr = server_socket.accept()
                client_socket.settimeout(2.0)
                # Pass enable_ml flag to client handler
                client_thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, addr, dipole_params, pos_bounds, xgb_models, enable_ml)
                )
                client_thread.daemon = True
                client_thread.start()
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

# def handle_client(client_socket, addr, dipole_params, pos_bounds, xgb_models, enable_ml=True):
#     """Enhanced client handler with ML comparison"""
#     try:
#         while not STOP_EVENT.is_set():
#             # Receive data length
#             length_bytes = b''
#             while len(length_bytes) < 4:
#                 chunk = client_socket.recv(4 - len(length_bytes))
#                 if not chunk:
#                     raise ConnectionError("Client disconnected")
#                 length_bytes += chunk
#             length = struct.unpack('>I', length_bytes)[0]
            
#             # Receive data
#             data_bytes = b''
#             while len(data_bytes) < length:
#                 chunk = client_socket.recv(length - len(data_bytes))
#                 if not chunk:
#                     raise ConnectionError("Client disconnected")
#                 data_bytes += chunk
            
#             # Parse JSON data
#             data = json.loads(data_bytes.decode('utf-8'))
#             B_custom = [data['magX'], data['magY'], data['magZ']]
#             quat_sensor = [data['qw'], data['qx'], data['qy'], data['qz']]
            
#             # Validate inputs
#             if any(abs(m) > max_Bfield for m in B_custom):
#                 print(f"⚠️ Invalid magnetic field values: {B_custom} µT")
#                 response = {'estX': 0.0, 'estY': 0.0, 'estZ': 0.0, 
#                            'mlX': 0.0, 'mlY': 0.0, 'mlZ': 0.0,
#                            'error': 'Invalid magnetic field values'}
#             else:
#                 quat_norm = np.sqrt(sum(x**2 for x in quat_sensor))
#                 if not 0.99 <= quat_norm <= 1.01:
#                     print(f"⚠️ Invalid quaternion norm: {quat_norm:.3f}")
#                     response = {'estX': 0.0, 'estY': 0.0, 'estZ': 0.0,
#                                'mlX': 0.0, 'mlY': 0.0, 'mlZ': 0.0,
#                                'error': 'Invalid quaternion norm'}
#                 else:
#                     quat_sensor = [x / quat_norm for x in quat_sensor]
                    
#                     # Estimate position with and without ML
#                     start_time = time.time()
                    
#                     # Physics-only estimate
#                     r_physics, r_init = estimate_position_custom(
#                         B_custom, quat_sensor, dipole_params['m'], pos_bounds,
#                         dipole_params['a_x'], dipole_params['b_x'], dipole_params['c_x'],
#                         dipole_params['a_y'], dipole_params['b_y'], dipole_params['c_y'],
#                         dipole_params['a_z'], dipole_params['b_z'], dipole_params['c_z'],
#                         xgb_models=None  # Force physics-only
#                     )
                    
#                     # ML-corrected estimate (if enabled and models available)
#                     r_ml = r_physics.copy()
#                     if enable_ml and xgb_models[0] is not None:
#                         r_ml, _ = estimate_position_custom(
#                             B_custom, quat_sensor, dipole_params['m'], pos_bounds,
#                             dipole_params['a_x'], dipole_params['b_x'], dipole_params['c_x'],
#                             dipole_params['a_y'], dipole_params['b_y'], dipole_params['c_y'],
#                             dipole_params['a_z'], dipole_params['b_z'], dipole_params['c_z'],
#                             xgb_models
#                         )
                    
#                     estimation_time = time.time() - start_time
                    
#                     # Log comparison
#                     log_comparison_result(B_custom, quat_sensor, r_physics, r_ml)
                    
#                     print(f"\n📡 Client {addr}:")
#                     print(f"   Input: B = {B_custom} µT")
#                     print(f"   Physics: x={r_physics[0]:.6f}, y={r_physics[1]:.6f}, z={r_physics[2]:.6f} m")
#                     if enable_ml and xgb_models[0] is not None:
#                         print(f"   ML: x={r_ml[0]:.6f}, y={r_ml[1]:.6f}, z={r_ml[2]:.6f} m")
#                         diff = r_ml - r_physics
#                         print(f"   Δ: {diff[0]*1000:.2f}, {diff[1]*1000:.2f}, {diff[2]*1000:.2f} mm")
                    
#                     # Send both estimates back
#                     response = {
#                         'estX': float(r_physics[0]),  # Physics-only for backward compatibility
#                         'estY': float(r_physics[1]),
#                         'estZ': float(r_physics[2]),
#                         # 'mlX': float(r_ml[0]),        # ML-corrected
#                         # 'mlY': float(r_ml[1]),
#                         # 'mlZ': float(r_ml[2]),
#                         # 'ml_enabled': bool(enable_ml and xgb_models[0] is not None)
#                     }
            
#             # Send response
#             response_str = json.dumps(response)
#             response_bytes = response_str.encode('utf-8')
#             client_socket.sendall(struct.pack('>I', len(response_bytes)))
#             client_socket.sendall(response_bytes)
            
#     except (ConnectionError, socket.error, json.JSONDecodeError) as e:
#         print(f"Client {addr} error: {e}")
#     except Exception as e:
#         print(f"Unexpected error handling client {addr}: {e}")
#     finally:
#         client_socket.close()

def handle_client(client_socket, addr, dipole_params, pos_bounds, xgb_models, enable_ml=True):
    """Enhanced client handler with ML comparison and both quadratic/cubic comparison"""
    global POSITION_TIMING_LAST_PRINT
    try:
        # Try to load quadratic model for comparison
        quad_params = None
        try:
            with open('dipole_model_params.json', 'r') as f:
                quad_data = json.load(f)
                if all(k in quad_data for k in ['m_x', 'm_y', 'm_z', 'a_x', 'b_x', 'c_x', 'a_y', 'b_y', 'c_y', 'a_z', 'b_z', 'c_z']):
                    quad_params = {
                        'm': np.array([quad_data['m_x'], quad_data['m_y'], quad_data['m_z']]),
                        'a_x': quad_data['a_x'], 'b_x': quad_data['b_x'], 'c_x': quad_data['c_x'],
                        'a_y': quad_data['a_y'], 'b_y': quad_data['b_y'], 'c_y': quad_data['c_y'],
                        'a_z': quad_data['a_z'], 'b_z': quad_data['b_z'], 'c_z': quad_data['c_z']
                    }
        except:
            pass
        
        while not STOP_EVENT.is_set():



            # # --- FLUSH ACCUMULATED OLD DATA ---
            # client_socket.setblocking(False)
            # try:
            #     while True:
            #         # Keep reading chunks until the OS buffer is completely empty
            #         dump = client_socket.recv(4096)
            #         if not dump:
            #             break
            # except BlockingIOError:
            #     pass  # Buffer is successfully cleared
            # finally:
            #     client_socket.setblocking(True)


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
            if any(abs(m) > max_Bfield for m in B_custom):
                print(f"⚠️ Invalid magnetic field values: {B_custom} µT")
                response = {'estX': 0.0, 'estY': 0.0, 'estZ': 0.0, 
                           'mlX': 0.0, 'mlY': 0.0, 'mlZ': 0.0,
                           'error': 'Invalid magnetic field values'}
            else:
                quat_norm = np.sqrt(sum(x**2 for x in quat_sensor))
                if not 0.99 <= quat_norm <= 1.01:
                    print(f"⚠️ Invalid quaternion norm: {quat_norm:.3f}")
                    response = {'estX': 0.0, 'estY': 0.0, 'estZ': 0.0,
                               'mlX': 0.0, 'mlY': 0.0, 'mlZ': 0.0,
                               'error': 'Invalid quaternion norm'}
                else:
                    quat_sensor = [x / quat_norm for x in quat_sensor]
                    
                    start_time = time.perf_counter()
                    step_names = ['validation', 'cubic_estimate', 'quadratic_estimate', 'ml_correct', 'response']
                    step_durations_ms = {name: 0.0 for name in step_names}

                    step_start = time.perf_counter()
                    # Cubic estimate (current model)
                    r_cubic, r_init = estimate_position_unified(
                        B_custom, quat_sensor, dipole_params, pos_bounds,
                        xgb_models=None
                    )
                    step_durations_ms['cubic_estimate'] = (time.perf_counter() - step_start) * 1000.0
                    
                    step_start = time.perf_counter()
                    # Quadratic estimate (if available)
                    r_quad = None
                    if quad_params is not None:
                        r_quad, _ = estimate_position_unified(
                            B_custom, quat_sensor, quad_params, pos_bounds,
                            xgb_models=None
                        )
                    step_durations_ms['quadratic_estimate'] = (time.perf_counter() - step_start) * 1000.0
                    
                    # step_start = time.perf_counter()
                    # # ML-corrected estimate (if enabled)
                    # r_ml = r_cubic.copy()
                    # if enable_ml and xgb_models[0] is not None:
                    #     r_ml, _ = estimate_position_unified(
                    #         B_custom, quat_sensor, dipole_params, pos_bounds,
                    #         xgb_models
                    #     )
                    step_durations_ms['ml_correct'] = (time.perf_counter() - step_start) * 1000.0
                    
                    step_start = time.perf_counter()
                    # Log comparison
                    # log_comparison_result(B_custom, quat_sensor, r_cubic, r_ml)
                    step_durations_ms['response'] = (time.perf_counter() - step_start) * 1000.0
                    
                    request_period = time.perf_counter() - start_time
                    with POSITION_TIMING_LOCK:
                        POSITION_TIMING_PERIODS.append(request_period)
                        avg_period = sum(POSITION_TIMING_PERIODS) / len(POSITION_TIMING_PERIODS)
                        avg_hz = 1.0 / avg_period if avg_period > 0.0 else 0.0
                        global POSITION_TIMING_LAST_PRINT
                        if time.perf_counter() - POSITION_TIMING_LAST_PRINT >= 0.5:
                            print(format_position_timing_summary(step_durations_ms, avg_hz))
                            POSITION_TIMING_LAST_PRINT = time.perf_counter()
                    
                    # Print results
                    # print(f"\n📡 Client {addr}:")
                    # print(f"   Model: CUBIC (primary)")
                    # print(f"   Input: B = [{B_custom[0]:.2f}, {B_custom[1]:.2f}, {B_custom[2]:.2f}] µT")
                    # print(f"   Input q = [{quat_sensor[0]:.2f}, {quat_sensor[1]:.2f}, {quat_sensor[2]:.2f}, {quat_sensor[3]:.2f}]")
                    # print(f"   cubic Physics: x={r_cubic[0]:.6f}, y={r_cubic[1]:.6f}, z={r_cubic[2]:.6f} m")
                    
                    if r_quad is not None:
                        print(f"  In.B: [{B_custom[0]:.2f}, {B_custom[1]:.2f}, {B_custom[2]:.2f}] µT] phy. only: x={r_quad[0]:.3f}, y={r_quad[1]:.3f}, z={r_quad[2]:.3f} m Cur_Time: {time.time():3f}")
                    
                    # if enable_ml and xgb_models[0] is not None:
                    #     print(f"   ML: x={r_ml[0]:.6f}, y={r_ml[1]:.6f}, z={r_ml[2]:.6f} m")
                    #     diff = r_ml - r_cubic
                    #     print(f"   Δ: {diff[0]*1000:.2f}, {diff[1]*1000:.2f}, {diff[2]*1000:.2f} mm")
                    
                    response = {
                        'estX': float(r_cubic[0]),
                        'estY': float(r_cubic[1]),
                        'estZ': float(r_cubic[2]),
                    }
            
            # Send response
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
def load_dipole_params():
    """Load dipole parameters, supporting both quadratic and cubic models"""
    # Try cubic parameters first
    cubic_param_file = 'dipole_model_params_cubic.json'
    quad_param_file = 'dipole_model_params.json'
    
    # Check for cubic parameters
    if os.path.exists(cubic_param_file):
        try:
            with open(cubic_param_file, 'r') as f:
                params = json.load(f)
            
            # Check if it has cubic parameters
            cubic_keys = ['a2_x', 'b2_x', 'c2_x', 'a2_y', 'b2_y', 'c2_y', 'a2_z', 'b2_z', 'c2_z',
                         'a3_x', 'b3_x', 'c3_x', 'd3_x', 'a3_y', 'b3_y', 'c3_y', 'd3_y', 
                         'a3_z', 'b3_z', 'c3_z', 'd3_z']
            
            if all(key in params for key in cubic_keys):
                m = np.array(params['m']) if isinstance(params['m'], list) else np.array([params['m_x'], params['m_y'], params['m_z']])
                print(f"✅ Loaded CUBIC model from {cubic_param_file}")
                return {
                    'model_type': 'cubic',
                    'm': m,
                    'a2_x': params['a2_x'], 'b2_x': params['b2_x'], 'c2_x': params['c2_x'],
                    'a2_y': params['a2_y'], 'b2_y': params['b2_y'], 'c2_y': params['c2_y'],
                    'a2_z': params['a2_z'], 'b2_z': params['b2_z'], 'c2_z': params['c2_z'],
                    'a3_x': params['a3_x'], 'b3_x': params['b3_x'], 'c3_x': params['c3_x'], 'd3_x': params['d3_x'],
                    'a3_y': params['a3_y'], 'b3_y': params['b3_y'], 'c3_y': params['c3_y'], 'd3_y': params['d3_y'],
                    'a3_z': params['a3_z'], 'b3_z': params['b3_z'], 'c3_z': params['c3_z'], 'd3_z': params['d3_z']
                }
        except Exception as e:
            print(f"Error loading cubic params: {e}")
    
    # Fall back to quadratic parameters
    if not os.path.exists(quad_param_file):
        return None
    
    try:
        with open(quad_param_file, 'r') as f:
            params = json.load(f)
        
        # Check for enhanced parameters first, fall back to linear
        enhanced_keys = ['m_x', 'm_y', 'm_z', 'a_x', 'b_x', 'c_x', 'a_y', 'b_y', 'c_y', 'a_z', 'b_z', 'c_z']
        linear_keys = ['m_x', 'm_y', 'm_z', 'a_x', 'b_x', 'a_y', 'b_y', 'a_z', 'b_z']
        
        if all(key in params for key in enhanced_keys):
            required_keys = enhanced_keys
            is_enhanced = True
        elif all(key in params for key in linear_keys):
            required_keys = linear_keys
            is_enhanced = False
        else:
            missing_enhanced = [key for key in enhanced_keys if key not in params]
            missing_linear = [key for key in linear_keys if key not in params]
            print(f"Missing keys in {quad_param_file}: Enhanced: {missing_enhanced}, Linear: {missing_linear}")
            return None
        
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            print(f"Missing keys in {quad_param_file}: {missing_keys}")
            return None
        
        m = np.array([params['m_x'], params['m_y'], params['m_z']])
        if np.linalg.norm(m) > 1000:
            print(f"Dipole moment norm ({np.linalg.norm(m):.3f} A·m²) is too large.")
            return None
        
        if is_enhanced:
            print(f"✅ Loaded QUADRATIC model from {quad_param_file}")
            return {
                'model_type': 'quadratic',
                'm': m,
                'a_x': params['a_x'], 'b_x': params['b_x'], 'c_x': params['c_x'],
                'a_y': params['a_y'], 'b_y': params['b_y'], 'c_y': params['c_y'],
                'a_z': params['a_z'], 'b_z': params['b_z'], 'c_z': params['c_z']
            }
        else:
            print(f"✅ Loaded LINEAR model from {quad_param_file}")
            return {
                'model_type': 'linear',
                'm': m,
                'a_x': 0.0, 'b_x': params['a_x'], 'c_x': params['b_x'],
                'a_y': 0.0, 'b_y': params['a_y'], 'c_y': params['b_y'],
                'a_z': 0.0, 'b_z': params['a_z'], 'c_z': params['b_z']
            }
            
    except json.JSONDecodeError as e:
        print(f"Failed to decode {quad_param_file}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {quad_param_file}: {e}")
        return None
# # --- Helper Functions ---
# def load_dipole_params():
#     param_file = 'dipole_model_params.json'
#     if not os.path.exists(param_file):
#         return None
#     try:
#         with open(param_file, 'r') as f:
#             params = json.load(f)
#         # Check for enhanced parameters first, fall back to linear
#         enhanced_keys = ['m_x', 'm_y', 'm_z', 'a_x', 'b_x', 'c_x', 'a_y', 'b_y', 'c_y', 'a_z', 'b_z', 'c_z']
#         linear_keys = ['m_x', 'm_y', 'm_z', 'a_x', 'b_x', 'a_y', 'b_y', 'a_z', 'b_z']
        
#         if all(key in params for key in enhanced_keys):
#             # Enhanced parameters with quadratic terms
#             required_keys = enhanced_keys
#             is_enhanced = True
#         elif all(key in params for key in linear_keys):
#             # Linear parameters, set quadratic terms to zero
#             required_keys = linear_keys
#             is_enhanced = False
#         else:
#             missing_enhanced = [key for key in enhanced_keys if key not in params]
#             missing_linear = [key for key in linear_keys if key not in params]
#             print(f"Missing keys in {param_file}: Enhanced: {missing_enhanced}, Linear: {missing_linear}")
#             return None
        
#         missing_keys = [key for key in required_keys if key not in params]
#         if missing_keys:
#             print(f"Missing keys in {param_file}: {missing_keys}")
#             return None
        
#         m = np.array([params['m_x'], params['m_y'], params['m_z']])
#         if np.linalg.norm(m) > 1000:
#             print(f"Dipole moment norm ({np.linalg.norm(m):.3f} A·m²) is too large.")
#             return None
        
#         if is_enhanced:
#             return {
#                 'm': m,
#                 'a_x': params['a_x'], 'b_x': params['b_x'], 'c_x': params['c_x'],
#                 'a_y': params['a_y'], 'b_y': params['b_y'], 'c_y': params['c_y'],
#                 'a_z': params['a_z'], 'b_z': params['b_z'], 'c_z': params['c_z']
#             }
#         else:
#             # Linear parameters - set quadratic terms to zero
#             return {
#                 'm': m,
#                 'a_x': 0.0, 'b_x': params['a_x'], 'c_x': params['b_x'],  # Map a_x->b_x, b_x->c_x
#                 'a_y': 0.0, 'b_y': params['a_y'], 'c_y': params['b_y'],
#                 'a_z': 0.0, 'b_z': params['a_z'], 'c_z': params['b_z']
#             }
            
#     except json.JSONDecodeError as e:
#         print(f"Failed to decode {param_file}: {e}")
#         return None
#     except Exception as e:
#         print(f"Unexpected error loading {param_file}: {e}")
#         return None
def estimate_position_unified(B_custom, quat_sensor, dipole_params, pos_bounds, xgb_models=None):
    """
    Unified position estimation that works with both quadratic and cubic models
    """
    if any(np.isnan(B_custom)) or any(np.isnan(quat_sensor)) or any(abs(b) > max_Bfield for b in B_custom):
        print(f"Invalid input: B_custom={B_custom}, quat_sensor={quat_sensor}")
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    
    B_custom_T = np.array(B_custom) * muT_to_T
    R_sensor = quaternion_to_rotation_matrix(quat_sensor)
    r_init = initial_position_guess(B_custom_T, dipole_params['m'], R_sensor, pos_bounds)
    
    try:
        # Choose estimation method based on model type
        if dipole_params.get('model_type') == 'cubic':
            # Use cubic model
            result = least_squares(
                residuals_position_cubic_wrapper, r_init,
                args=(B_custom_T, dipole_params['m'], R_sensor, dipole_params),
                bounds=pos_bounds, method='trf', ftol=1e-6, xtol=1e-6
            )
        else:
            # Use quadratic model
            result = least_squares(
                residuals_position_enhanced, r_init,
                args=(B_custom_T, dipole_params['m'], R_sensor, 
                      dipole_params.get('a_x', 0.0), dipole_params.get('b_x', 1.0), dipole_params.get('c_x', 0.0),
                      dipole_params.get('a_y', 0.0), dipole_params.get('b_y', 1.0), dipole_params.get('c_y', 0.0),
                      dipole_params.get('a_z', 0.0), dipole_params.get('b_z', 1.0), dipole_params.get('c_z', 0.0)),
                bounds=pos_bounds, method='trf', ftol=1e-6, xtol=1e-6
            )
        
        if not result.success or np.any(np.isnan(result.x)):
            print(f"Optimization failed for B_custom={B_custom}")
            return r_init, r_init
        
        r_est = result.x
        
        # Apply ML correction if models are provided
        if xgb_models is not None and all(model is not None for model in xgb_models):
            xgb_x, xgb_y, xgb_z = xgb_models
            
            # Prepare features for correction model
            features = np.array([[B_custom[0], B_custom[1], B_custom[2],
                                 quat_sensor[0], quat_sensor[1], quat_sensor[2], quat_sensor[3],
                                 r_est[0], r_est[1], r_est[2]]])
            
            if np.any(np.isnan(features)):
                print(f"NaN in features: {features}")
                return r_est, r_init
            
            # Predict errors
            error_x_pred = xgb_x.predict(features)[0]
            error_y_pred = xgb_y.predict(features)[0]
            error_z_pred = xgb_z.predict(features)[0]
            
            # Correct position
            r_ml = np.array([r_est[0] - error_x_pred, 
                            r_est[1] - error_y_pred, 
                            r_est[2] - error_z_pred])
            
            return r_ml, r_init
        
        return r_est, r_init
        
    except Exception as e:
        print(f"Error estimating position: {str(e)}")
        return r_init, r_init

def residuals_position_cubic_wrapper(r, B_measured, m, R_sensor, dipole_params):
    """Wrapper for cubic position residuals"""
    B_pred = dipole_field_enhanced_cubic(
        r, m, R_sensor,
        dipole_params['a2_x'], dipole_params['b2_x'], dipole_params['c2_x'],
        dipole_params['a2_y'], dipole_params['b2_y'], dipole_params['c2_y'],
        dipole_params['a2_z'], dipole_params['b2_z'], dipole_params['c2_z'],
        dipole_params['a3_x'], dipole_params['b3_x'], dipole_params['c3_x'], dipole_params['d3_x'],
        dipole_params['a3_y'], dipole_params['b3_y'], dipole_params['c3_y'], dipole_params['d3_y'],
        dipole_params['a3_z'], dipole_params['b3_z'], dipole_params['c3_z'], dipole_params['d3_z']
    )
    return B_pred - B_measured

def diagnose_pitch_misalignment(data_df):
    """
    Diagnostic focusing on pitch variation (since you have good pitch data)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr
    from scipy.optimize import least_squares
    
    print("\n" + "="*60)
    print("PITCH-BASED MAGNET ALIGNMENT DIAGNOSTIC")
    print("="*60)
    
    # Extract data
    positions = data_df[['Rel_X_m', 'Rel_Y_m', 'Rel_Z_m']].values
    pitch = data_df['Cube2_Pitch_deg'].values
    roll = data_df['Cube2_Roll_deg'].values
    yaw = data_df['Cube2_Yaw_deg'].values
    B_meas = data_df[['Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT']].values
    quats = data_df[['Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz']].values
    
    print(f"\nData Summary:")
    print(f"  Pitch range: {pitch.min():.1f}° to {pitch.max():.1f}° (range: {pitch.max()-pitch.min():.1f}°)")
    print(f"  Roll range: {roll.min():.1f}° to {roll.max():.1f}° (limited)")
    print(f"  Yaw range: {yaw.min():.1f}° to {yaw.max():.1f}° (limited)")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Field magnitude vs pitch
    ax = axes[0, 0]
    B_mag = np.sqrt(B_meas[:,0]**2 + B_meas[:,1]**2 + B_meas[:,2]**2)
    scatter = ax.scatter(pitch, B_mag, c=positions[:,0], cmap='viridis', alpha=0.5)
    ax.set_xlabel('Pitch Angle (deg)')
    ax.set_ylabel('Field Magnitude (µT)')
    ax.set_title('Field Magnitude vs Pitch\n(colored by X position)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='X Position (m)')
    
    # Calculate correlation
    corr_mag, p_mag = pearsonr(pitch, B_mag)
    ax.text(0.05, 0.95, f'corr={corr_mag:.3f}\np={p_mag:.3f}', 
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Field components vs pitch
    colors = ['red', 'green', 'blue']
    labels = ['Bx', 'By', 'Bz']
    
    ax = axes[0, 1]
    for i in range(3):
        ax.scatter(pitch, B_meas[:, i], c=colors[i], alpha=0.3, s=10, label=labels[i])
    ax.set_xlabel('Pitch Angle (deg)')
    ax.set_ylabel('Field Component (µT)')
    ax.set_title('Field Components vs Pitch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Fit simple dipole model and analyze pitch-dependent residuals
    ax = axes[0, 2]
    
    # Fit dipole model (assuming perfect alignment)
    def simple_residuals(m, positions, B_meas, quats):
        m_vec = m
        residuals = []
        for r, B, q in zip(positions, B_meas/1e6, quats):
            R_sensor = quaternion_to_rotation_matrix(q)
            B_pred_global = dipole_field_enhanced(r, m_vec, None)
            B_pred_local = R_sensor.T @ B_pred_global
            residuals.append(B_pred_local - B)
        return np.ravel(residuals)
    
    m0 = np.array([-0.1, 0, 0])
    result = least_squares(simple_residuals, m0, 
                          args=(positions, B_meas/1e6, quats),
                          bounds=([-1, -1, -1], [0, 1, 1]))
    m_opt = result.x
    
    # Calculate residuals
    residuals_list = []
    for r, B, q in zip(positions, B_meas/1e6, quats):
        R_sensor = quaternion_to_rotation_matrix(q)
        B_pred_global = dipole_field_enhanced(r, m_opt, None)
        B_pred_local = R_sensor.T @ B_pred_global
        res = np.linalg.norm(B_pred_local - B) * 1e6
        residuals_list.append(res)
    
    # Convert to numpy array for indexing
    residuals = np.array(residuals_list)
    
    scatter = ax.scatter(pitch, residuals, c=positions[:,1], cmap='plasma', alpha=0.5)
    ax.set_xlabel('Pitch Angle (deg)')
    ax.set_ylabel('Residual Magnitude (µT)')
    ax.set_title('Model Residuals vs Pitch\n(colored by Y position)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Y Position (m)')
    
    corr_resid, p_resid = pearsonr(pitch, residuals)
    ax.text(0.05, 0.95, f'corr={corr_resid:.3f}\np={p_resid:.3f}', 
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Check if pitch effect depends on X position
    ax = axes[1, 0]
    
    # Split data by X position
    x_median = np.median(positions[:,0])
    mask_pos_x = positions[:,0] > x_median
    mask_neg_x = positions[:,0] < x_median
    
    ax.scatter(pitch[mask_pos_x], residuals[mask_pos_x], 
              c='red', alpha=0.5, s=10, label='X > median')
    ax.scatter(pitch[mask_neg_x], residuals[mask_neg_x], 
              c='blue', alpha=0.5, s=10, label='X < median')
    ax.set_xlabel('Pitch Angle (deg)')
    ax.set_ylabel('Residual Magnitude (µT)')
    ax.set_title('Residuals vs Pitch\n(colored by X position)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Fit a model with pitch-dependent correction
    ax = axes[1, 1]
    
    # Try to estimate if magnet is tilted in pitch direction
    # This would show up as Bx varying with pitch in a specific way
    
    # Normalize pitch to [-1, 1] range
    pitch_norm = 2 * (pitch - pitch.min()) / (pitch.max() - pitch.min()) - 1
    
    # Look for quadratic relationship between Bx and pitch
    from numpy.polynomial import Polynomial
    
    # Fit Bx vs pitch
    p_x = Polynomial.fit(pitch_norm, B_meas[:,0], 2)
    pitch_fit = np.linspace(-1, 1, 100)
    Bx_fit = p_x(pitch_fit)
    
    ax.scatter(pitch_norm, B_meas[:,0], alpha=0.3, s=10, label='Data')
    ax.plot(pitch_fit, Bx_fit, 'r-', linewidth=2, label='Quadratic fit')
    ax.set_xlabel('Normalized Pitch')
    ax.set_ylabel('Bx (µT)')
    ax.set_title('Bx vs Pitch - looking for asymmetry')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # The quadratic coefficient tells us about misalignment
    quad_coef = p_x.convert().coef[2] if len(p_x.convert().coef) > 2 else 0
    ax.text(0.05, 0.95, f'Quadratic coef: {quad_coef:.3f}', 
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Simple misalignment estimate
    ax = axes[1, 2]
    
    # If magnet is perfectly aligned along X, Bx should be symmetric with pitch
    # Asymmetry suggests tilt
    
    # Split by pitch sign
    pos_pitch = pitch > 0
    neg_pitch = pitch < 0
    
    if np.sum(pos_pitch) > 10 and np.sum(neg_pitch) > 10:
        Bx_pos = np.mean(B_meas[pos_pitch, 0])
        Bx_neg = np.mean(B_meas[neg_pitch, 0])
        
        ax.bar(['Pitch < 0', 'Pitch > 0'], [Bx_neg, Bx_pos])
        ax.set_ylabel('Mean Bx (µT)')
        ax.set_title(f'Bx asymmetry: {Bx_pos/Bx_neg:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Estimate tilt angle
        # This is a rough estimate - the actual relationship is more complex
        if abs(Bx_pos - Bx_neg) > 1.0:
            estimated_tilt = np.arctan(abs(Bx_pos - Bx_neg) / np.mean([abs(Bx_pos), abs(Bx_neg)])) * 180 / np.pi
            ax.text(0.5, 0.8, f'Est. tilt: {estimated_tilt:.1f}°', 
                   ha='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('pitch_diagnostic.png', dpi=150)
    plt.show()
    
    # Print summary
    print("\n" + "-"*40)
    print("DIAGNOSIS SUMMARY")
    print("-"*40)
    
    print(f"\nPitch - Field Magnitude correlation: {corr_mag:.3f} (p={p_mag:.3f})")
    print(f"Pitch - Residual correlation: {corr_resid:.3f} (p={p_resid:.3f})")
    
    if abs(corr_resid) > 0.2 and p_resid < 0.05:
        print("\n⚠️  Significant pitch-dependent residuals detected!")
        print("   This suggests the magnet may be tilted in the pitch direction.")
        print("   Recommended: Add pitch tilt to your calibration model.")
    elif abs(corr_resid) > 0.1:
        print("\n⚠️  Weak pitch dependence detected - may need more data or wider range")
    else:
        print("\n✅ No significant pitch dependence - magnet appears well-aligned in pitch")
    
    if abs(quad_coef) > 0.1:
        print(f"\n⚠️  Significant quadratic term ({quad_coef:.3f}) in Bx vs pitch")
        print("   This also suggests misalignment in the pitch axis")
    
    return {
        'pitch_range': [pitch.min(), pitch.max()],
        'mag_correlation': corr_mag,
        'residual_correlation': corr_resid,
        'quadratic_coef': quad_coef
    }

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

def dipole_field_enhanced(r, m, R_sensor=None, a_x=0.0, b_x=1.0, c_x=0.0, a_y=0.0, b_y=1.0, c_y=0.0, a_z=0.0, b_z=1.0, c_z=0.0):
    """
    Enhanced magnetic field model with quadratic position corrections
    r: position vector [x, y, z] in meters
    m: dipole moment vector [mx, my, mz] in A·m²
    a_x, b_x, c_x: quadratic coefficients for x-coordinate transformation: p_x = a_x * x^2 + b_x * x + c_x
    """
    x, y, z = r
    
    # Apply quadratic corrections to position
    p_x = a_x * x**2 + b_x * x + c_x
    p_y = a_y * y**2 + b_y * y + c_y
    p_z = a_z * z**2 + b_z * z + c_z
    p = [p_x, p_y, p_z]
    
    m_x, m_y, m_z = m
    r2 = p_x**2 + p_y**2 + p_z**2
    
    if r2 < 1e-12:
        return np.zeros(3)
    
    r_mag = np.sqrt(r2)
    r5 = r2**2.5
    
    # Dipole field
    r_dot_m = p_x * m_x + p_y * m_y + p_z * m_z
    
    B_x = factor * (3 * p_x * r_dot_m / r5 - m_x / (r2 * r_mag))
    B_y = factor * (3 * p_y * r_dot_m / r5 - m_y / (r2 * r_mag))
    B_z = factor * (3 * p_z * r_dot_m / r5 - m_z / (r2 * r_mag))
    
    B_global = np.array([B_x, B_y, B_z])
    B_global *= muT_to_T
    
    if R_sensor is None:
        R_sensor = np.eye(3)
    
    B_local = R_sensor.T @ B_global
    return B_local
def dipole_field_enhanced_cubic(r, m, R_sensor=None, 
                                 # Quadratic coefficients (power 2)
                                 a2_x=0.0, b2_x=1.0, c2_x=0.0,
                                 a2_y=0.0, b2_y=1.0, c2_y=0.0,
                                 a2_z=0.0, b2_z=1.0, c2_z=0.0,
                                 # Cubic coefficients (power 3)
                                 a3_x=0.0, b3_x=0.0, c3_x=0.0, d3_x=0.0,
                                 a3_y=0.0, b3_y=0.0, c3_y=0.0, d3_y=0.0,
                                 a3_z=0.0, b3_z=0.0, c3_z=0.0, d3_z=0.0):
    """
    Enhanced magnetic field model with quadratic AND cubic position corrections
    r: position vector [x, y, z] in meters
    m: dipole moment vector [mx, my, mz] in A·m²
    
    Quadratic: p = a2 * x^2 + b2 * x + c2
    Cubic: p = a3 * x^3 + b3 * x^2 + c3 * x + d3
    """
    x, y, z = r
    
    # Apply cubic corrections to position (can use quadratic, cubic, or both)
    p_x = (a3_x * x**3 + b3_x * x**2 + c3_x * x + d3_x) + (a2_x * x**2 + b2_x * x + c2_x)
    p_y = (a3_y * y**3 + b3_y * y**2 + c3_y * y + d3_y) + (a2_y * y**2 + b2_y * y + c2_y)
    p_z = (a3_z * z**3 + b3_z * z**2 + c3_z * z + d3_z) + (a2_z * z**2 + b2_z * z + c2_z)
    
    m_x, m_y, m_z = m
    r2 = p_x**2 + p_y**2 + p_z**2
    
    if r2 < 1e-12:
        return np.zeros(3)
    
    r_mag = np.sqrt(r2)
    r5 = r2**2.5
    
    # Dipole field
    r_dot_m = p_x * m_x + p_y * m_y + p_z * m_z
    
    B_x = factor * (3 * p_x * r_dot_m / r5 - m_x / (r2 * r_mag))
    B_y = factor * (3 * p_y * r_dot_m / r5 - m_y / (r2 * r_mag))
    B_z = factor * (3 * p_z * r_dot_m / r5 - m_z / (r2 * r_mag))
    
    B_global = np.array([B_x, B_y, B_z])
    B_global *= muT_to_T
    
    if R_sensor is None:
        R_sensor = np.eye(3)
    
    B_local = R_sensor.T @ B_global
    return B_local
def residuals_m_with_cubic(params, positions, B_measured, quaternions,
                          a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
                          a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z):
    """Residuals for dipole moment optimization with cubic terms"""
    m = params
    residuals = []
    for r, B_meas, quat in zip(positions, B_measured, quaternions):
        R_sensor = quaternion_to_rotation_matrix(quat)
        B_pred = dipole_field_enhanced_cubic(
            r, m, R_sensor,
            a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
            a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z
        )
        residuals.append(B_pred - B_meas)
    return np.ravel(residuals)

def objective_cubic(params, m_fixed, positions, B_measured, quaternions, lambda_reg=1e-3):
    """Objective function for cubic coefficients optimization with regularization"""
    # params order: 12 quadratic + 12 cubic = 24 parameters
    # Quadratic: a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z
    # Cubic: a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z
    a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z, \
    a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z = params
    
    res = []
    for r, B_meas, quat in zip(positions, B_measured, quaternions):
        R_sensor = quaternion_to_rotation_matrix(quat)
        B_pred = dipole_field_enhanced_cubic(
            r, m_fixed, R_sensor,
            a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
            a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z
        )
        res.append(B_pred - B_meas)
    
    res_array = np.ravel(res)
    
    # Regularization to prefer simpler models (closer to identity)
    reg = lambda_reg * (
        # Quadratic regularization (prefer linear)
        (a2_x - 0.0)**2 + (b2_x - 1.0)**2 + (c2_x - 0.0)**2 +
        (a2_y - 0.0)**2 + (b2_y - 1.0)**2 + (c2_y - 0.0)**2 +
        (a2_z - 0.0)**2 + (b2_z - 1.0)**2 + (c2_z - 0.0)**2 +
        # Cubic regularization (prefer zero)
        (a3_x - 0.0)**2 + (b3_x - 0.0)**2 + (c3_x - 1.0)**2 + (d3_x - 0.0)**2 +
        (a3_y - 0.0)**2 + (b3_y - 0.0)**2 + (c3_y - 1.0)**2 + (d3_y - 0.0)**2 +
        (a3_z - 0.0)**2 + (b3_z - 0.0)**2 + (c3_z - 1.0)**2 + (d3_z - 0.0)**2
    )
    
    return np.sum(res_array**2) + reg
def run_optimization_with_cubic():
    """Run optimization with quadratic AND cubic correction terms"""
    positions, B_measured, quaternions, data_df = load_and_filter_data()
    if positions is None:
        print("Failed to load data for optimization.")
        return None, None, None, None, None, None, None, None

    N = len(positions)
    if N < 10:
        print(f"Too few measurements ({N}). At least 10 are recommended for cubic model.")
        return None, None, None, None, None, None, None, None

    print(f"\n{'='*60}")
    print("OPTIMIZATION WITH CUBIC CORRECTIONS")
    print(f"{'='*60}")
    print(f"Data points: {N}")

    # Stage 1: Optimize dipole moment with simple linear model
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
                print(f"  Initial guess from on-axis point: m_x = {m0_x:.6f}")
                break

    bounds_m = ([-1000, -1000, -1000], [0, 1000, 1000])
    
    print("\n📈 Stage 1: Optimizing dipole moment (linear model)...")
    start_time = time.time()
    
    try:
        result = least_squares(
            residuals_m, m0, 
            args=(positions, B_measured, quaternions),
            bounds=bounds_m, 
            method='trf', 
            ftol=1e-8, 
            xtol=1e-8
        )
        m_best = result.x
        print(f"  ✅ Completed in {time.time() - start_time:.2f}s")
        print(f"  Dipole: m = [{m_best[0]:.6f}, {m_best[1]:.6f}, {m_best[2]:.6f}] A·m²")
        
    except Exception as e:
        print(f"  ❌ Dipole optimization failed: {e}")
        return None, None, None, None, None, None, None, None

    # Stage 2: Optimize quadratic + cubic coefficients
    print("\n📈 Stage 2: Optimizing quadratic + cubic coefficients...")
    start_time = time.time()
    
    # Initial parameters: quadratic (identity) + cubic (identity)
    # Order: 9 quadratic + 12 cubic = 21 parameters
    params0 = np.array([
        # Quadratic (a2, b2, c2) - 9 parameters
        0.0, 1.0, 0.0,   # X
        0.0, 1.0, 0.0,   # Y
        0.0, 1.0, 0.0,   # Z
        # Cubic (a3, b3, c3, d3) - 12 parameters
        0.0, 0.0, 1.0, 0.0,  # X: a3*x^3 + b3*x^2 + c3*x + d3
        0.0, 0.0, 1.0, 0.0,  # Y
        0.0, 0.0, 1.0, 0.0   # Z
    ])
    
    # Define bounds for coefficients - MUST match length of params0 (21 parameters)
    bounds_cubic = []
    # Quadratic bounds (a2, b2, c2) - 3 per axis * 3 axes = 9 bounds
    for _ in range(3):  # X, Y, Z axes
        bounds_cubic.append((-10, 10))   # a2
        bounds_cubic.append((-10, 10))   # b2
        bounds_cubic.append((-1, 1))     # c2
    # Cubic bounds (a3, b3, c3, d3) - 4 per axis * 3 axes = 12 bounds
    for _ in range(3):  # X, Y, Z axes
        bounds_cubic.append((-5, 5))     # a3
        bounds_cubic.append((-5, 5))     # b3
        bounds_cubic.append((-2, 2))     # c3
        bounds_cubic.append((-0.5, 0.5)) # d3
    
    # Verify bounds length
    print(f"  Parameters: {len(params0)}")
    print(f"  Bounds: {len(bounds_cubic)}")
    
    try:
        result_cubic = minimize(
            objective_cubic, params0,
            args=(m_best, positions, B_measured, quaternions, 1e-4),
            method='L-BFGS-B',  # Changed from Nelder-Mead to L-BFGS-B which handles bounds better
            bounds=bounds_cubic,
            options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 1000}
        )
        
        if not result_cubic.success:
            print(f"  ⚠️ Optimization warning: {result_cubic.message}")
        
        # Extract parameters
        params_opt = result_cubic.x
        
        # Quadratic parameters
        a2_x, b2_x, c2_x = params_opt[0], params_opt[1], params_opt[2]
        a2_y, b2_y, c2_y = params_opt[3], params_opt[4], params_opt[5]
        a2_z, b2_z, c2_z = params_opt[6], params_opt[7], params_opt[8]
        
        # Cubic parameters
        a3_x, b3_x, c3_x, d3_x = params_opt[9], params_opt[10], params_opt[11], params_opt[12]
        a3_y, b3_y, c3_y, d3_y = params_opt[13], params_opt[14], params_opt[15], params_opt[16]
        a3_z, b3_z, c3_z, d3_z = params_opt[17], params_opt[18], params_opt[19], params_opt[20]
        
        print(f"  ✅ Completed in {time.time() - start_time:.2f}s")
        print(f"\n  Quadratic Corrections:")
        print(f"    X: {a2_x:.6f}·x² + {b2_x:.6f}·x + {c2_x:.6f}")
        print(f"    Y: {a2_y:.6f}·y² + {b2_y:.6f}·y + {c2_y:.6f}")
        print(f"    Z: {a2_z:.6f}·z² + {b2_z:.6f}·z + {c2_z:.6f}")
        print(f"\n  Cubic Corrections:")
        print(f"    X: {a3_x:.6f}·x³ + {b3_x:.6f}·x² + {c3_x:.6f}·x + {d3_x:.6f}")
        print(f"    Y: {a3_y:.6f}·y³ + {b3_y:.6f}·y² + {c3_y:.6f}·y + {d3_y:.6f}")
        print(f"    Z: {a3_z:.6f}·z³ + {b3_z:.6f}·z² + {c3_z:.6f}·z + {d3_z:.6f}")
        
    except Exception as e:
        print(f"  ❌ Cubic optimization failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to quadratic-only model
        a2_x, b2_x, c2_x = 0.0, 1.0, 0.0
        a2_y, b2_y, c2_y = 0.0, 1.0, 0.0
        a2_z, b2_z, c2_z = 0.0, 1.0, 0.0
        a3_x, b3_x, c3_x, d3_x = 0.0, 0.0, 1.0, 0.0
        a3_y, b3_y, c3_y, d3_y = 0.0, 0.0, 1.0, 0.0
        a3_z, b3_z, c3_z, d3_z = 0.0, 0.0, 1.0, 0.0

    # Stage 3: Re-optimize dipole with cubic coefficients
    print("\n📈 Stage 3: Final dipole refinement with cubic coefficients...")
    start_time = time.time()
    
    try:
        def residuals_m_final(m, positions, B_measured, quaternions):
            residuals = []
            for r, B_meas, quat in zip(positions, B_measured, quaternions):
                R_sensor = quaternion_to_rotation_matrix(quat)
                B_pred = dipole_field_enhanced_cubic(
                    r, m, R_sensor,
                    a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
                    a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z
                )
                residuals.append(B_pred - B_meas)
            return np.ravel(residuals)
        
        result_m_final = least_squares(
            residuals_m_final, m_best,
            args=(positions, B_measured, quaternions),
            bounds=bounds_m,
            method='trf',
            ftol=1e-8,
            xtol=1e-8
        )
        m_final = result_m_final.x
        print(f"  ✅ Completed in {time.time() - start_time:.2f}s")
        print(f"  Final dipole: m = [{m_final[0]:.6f}, {m_final[1]:.6f}, {m_final[2]:.6f}] A·m²")
        
    except Exception as e:
        print(f"  ❌ Final refinement failed: {e}")
        m_final = m_best

    # Compute metrics for comparison
    print("\n📊 Computing performance metrics...")
    
    # Quadratic-only model metrics (using only quadratic terms, zero cubic)
    rmse_quad, ang_err_quad, _, _ = compute_metrics_enhanced(
        m_final, positions, B_measured, quaternions,
        a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z
    )
    
    # Cubic model metrics
    rmse_cubic, ang_err_cubic, ang_errs_cubic, B_pred_cubic = compute_metrics_cubic(
        m_final, positions, B_measured, quaternions,
        a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
        a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z
    )
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"Quadratic Model:  RMSE = {rmse_quad:.2f} µT, Angular Error = {ang_err_quad:.2f}°")
    print(f"Cubic Model:      RMSE = {rmse_cubic:.2f} µT, Angular Error = {ang_err_cubic:.2f}°")
    
    improvement_rmse = (rmse_quad - rmse_cubic) / rmse_quad * 100
    improvement_ang = (ang_err_quad - ang_err_cubic) / ang_err_quad * 100 if ang_err_quad > 0 else 0
    
    print(f"\nImprovement: RMSE: {improvement_rmse:+.1f}%, Angular: {improvement_ang:+.1f}%")
    
    if rmse_cubic < rmse_quad:
        print(f"\n✅ Cubic model performs better!")
    else:
        print(f"\n⚠️  Quadratic model performs better. Cubic terms may be overfitting.")
    
    # Save parameters - FIXED: convert numpy arrays to lists for JSON serialization
    dipole_params = {
        'm': m_final.tolist() if hasattr(m_final, 'tolist') else list(m_final),  # Convert to list
        'a2_x': float(a2_x), 'b2_x': float(b2_x), 'c2_x': float(c2_x),
        'a2_y': float(a2_y), 'b2_y': float(b2_y), 'c2_y': float(c2_y),
        'a2_z': float(a2_z), 'b2_z': float(b2_z), 'c2_z': float(c2_z),
        'a3_x': float(a3_x), 'b3_x': float(b3_x), 'c3_x': float(c3_x), 'd3_x': float(d3_x),
        'a3_y': float(a3_y), 'b3_y': float(b3_y), 'c3_y': float(c3_y), 'd3_y': float(d3_y),
        'a3_z': float(a3_z), 'b3_z': float(b3_z), 'c3_z': float(c3_z), 'd3_z': float(d3_z),
        'rmse_quadratic': float(rmse_quad),
        'rmse_cubic': float(rmse_cubic),
        'angular_error_quadratic': float(ang_err_quad),
        'angular_error_cubic': float(ang_err_cubic),
        'improvement_rmse_percent': float(improvement_rmse),
        'improvement_angular_percent': float(improvement_ang)
    }
    
    with open('dipole_model_params_cubic.json', 'w') as f:
        json.dump(dipole_params, f, indent=4)
    
    print(f"\n💾 Parameters saved to dipole_model_params_cubic.json")
    
    pos_bounds = (
        [positions[:,0].min(), positions[:,1].min(), positions[:,2].min()],
        [positions[:,0].max(), positions[:,1].max(), positions[:,2].max()]
    )
    
    # Return also the cubic coefficients for potential use
    return dipole_params, pos_bounds, positions, B_measured, quaternions, ang_errs_cubic, B_pred_cubic, "Cubic Model"

def compute_metrics_cubic(m, positions, B_measured, quaternions,
                         a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
                         a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z):
    """Compute metrics using cubic dipole field model"""
    B_pred = np.array([
        dipole_field_enhanced_cubic(
            r, m, quaternion_to_rotation_matrix(quat),
            a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
            a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z
        )
        for r, quat in zip(positions, quaternions)
    ])
    
    errors = B_pred - B_measured
    rmse = np.sqrt(np.mean(errors**2)) * 1e6
    
    angular_errors = []
    for b_p, b_m in zip(B_pred, B_measured):
        if np.linalg.norm(b_m) > 1e-10 and np.linalg.norm(b_p) > 1e-10:
            cos_theta = np.dot(b_p, b_m) / (np.linalg.norm(b_p) * np.linalg.norm(b_m))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta) * 180 / np.pi
            angular_errors.append(theta)
        else:
            angular_errors.append(np.nan)
    
    mean_angular_error = np.nanmean(angular_errors) if angular_errors else np.nan
    return rmse, mean_angular_error, angular_errors, B_pred * 1e6
def estimate_position_with_cubic(B_custom, quat_sensor, m, pos_bounds,
                                 a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
                                 a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z):
    """Position estimation using cubic model"""
    if any(np.isnan(B_custom)) or any(np.isnan(quat_sensor)):
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    
    B_custom_T = np.array(B_custom) * muT_to_T
    R_sensor = quaternion_to_rotation_matrix(quat_sensor)
    r_init = initial_position_guess(B_custom_T, m, R_sensor, pos_bounds)
    
    def residuals_position_cubic(r, B_measured, m, R_sensor,
                                  a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
                                  a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z):
        B_pred = dipole_field_enhanced_cubic(
            r, m, R_sensor,
            a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
            a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z
        )
        return B_pred - B_measured
    
    try:
        result = least_squares(
            residuals_position_cubic, r_init,
            args=(B_custom_T, m, R_sensor,
                  a2_x, b2_x, c2_x, a2_y, b2_y, c2_y, a2_z, b2_z, c2_z,
                  a3_x, b3_x, c3_x, d3_x, a3_y, b3_y, c3_y, d3_y, a3_z, b3_z, c3_z, d3_z),
            bounds=pos_bounds, method='trf', ftol=1e-6, xtol=1e-6
        )
        
        if not result.success or np.any(np.isnan(result.x)):
            return r_init, r_init
        
        return result.x, r_init
        
    except Exception as e:
        print(f"Error in cubic position estimation: {str(e)}")
        return r_init, r_init
    
def compare_quadratic_vs_cubic():
    """Compare quadratic vs cubic model performance"""
    print("\n" + "="*60)
    print("COMPARING QUADRATIC VS CUBIC MODELS")
    print("="*60)
    
    # Run cubic optimization
    dipole_params_cubic, pos_bounds, positions, B_measured, quaternions, ang_errs, B_pred, model_name = run_optimization_with_cubic()
    
    if dipole_params_cubic is None:
        print("Cubic optimization failed.")
        return
    
    # Also run quadratic optimization for comparison
    print("\n" + "="*60)
    print("RUNNING QUADRATIC OPTIMIZATION FOR COMPARISON")
    print("="*60)
    
    dipole_params_quad, _, _, _, _, _, _, _ = run_optimization_enhanced()
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Load results
    with open('dipole_model_params_cubic.json', 'r') as f:
        cubic_results = json.load(f)
    
    with open('dipole_model_params.json', 'r') as f:
        quad_results = json.load(f)
    
    # Bar plot comparison
    models = ['Quadratic', 'Cubic']
    rmse_values = [quad_results.get('rmse', 0), cubic_results.get('rmse_cubic', 0)]
    ang_values = [quad_results.get('angular_error', 0), cubic_results.get('angular_error_cubic', 0)]
    
    axes[0,0].bar(models, rmse_values, color=['blue', 'green'])
    axes[0,0].set_ylabel('RMSE (µT)')
    axes[0,0].set_title('Field Prediction RMSE')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].bar(models, ang_values, color=['blue', 'green'])
    axes[0,1].set_ylabel('Angular Error (degrees)')
    axes[0,1].set_title('Field Direction Error')
    axes[0,1].grid(True, alpha=0.3)
    
    # Improvement text
    improvement_rmse = cubic_results.get('improvement_rmse_percent', 0)
    improvement_ang = cubic_results.get('improvement_angular_percent', 0)
    
    axes[1,0].text(0.5, 0.5, f'RMSE Improvement: {improvement_rmse:+.1f}%\nAngular Improvement: {improvement_ang:+.1f}%',
                  ha='center', va='center', fontsize=14, transform=axes[1,0].transAxes,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1,0].set_title('Performance Summary')
    axes[1,0].axis('off')
    
    # Residual plot comparison
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('quadratic_vs_cubic_comparison.png', dpi=150)
    plt.show()
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    if improvement_rmse > 0:
        print(f"✅ Cubic model improves RMSE by {improvement_rmse:.1f}%")
    else:
        print(f"⚠️  Cubic model degrades RMSE by {abs(improvement_rmse):.1f}%")
    
    if improvement_ang > 0:
        print(f"✅ Cubic model improves angular error by {improvement_ang:.1f}%")
    else:
        print(f"⚠️  Cubic model degrades angular error by {abs(improvement_ang):.1f}%")

# For backward compatibility
def dipole_field(r, m, R_sensor=None, a_x=1.0, b_x=0.0, a_y=1.0, b_y=0.0, a_z=1.0, b_z=0.0):
    """Wrapper for linear model - maps to quadratic with a_x=0"""
    return dipole_field_enhanced(r, m, R_sensor, 0.0, a_x, b_x, 0.0, a_y, b_y, 0.0, a_z, b_z)

def residuals_m(params, positions, B_measured, quaternions):
    """Residuals for dipole moment optimization using enhanced model"""
    m = params
    residuals = []
    for r, B_meas, quat in zip(positions, B_measured, quaternions):
        R_sensor = quaternion_to_rotation_matrix(quat)
        # Use enhanced model with default quadratic coefficients (linear transformation)
        B_pred = dipole_field_enhanced(r, m, R_sensor, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
        residuals.append(B_pred - B_meas)
    return np.ravel(residuals)

def objective_m(m, positions, B_measured, quaternions):
    res = residuals_m(m, positions, B_measured, quaternions)
    return np.sum(res**2)

def residuals_abc(params, m_fixed, positions, B_measured, quaternions):
    """Residuals for quadratic coefficients optimization"""
    a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z = params
    residuals = []
    for r, B_meas, quat in zip(positions, B_measured, quaternions):
        R_sensor = quaternion_to_rotation_matrix(quat)
        B_pred = dipole_field_enhanced(r, m_fixed, R_sensor, a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z)
        residuals.append(B_pred - B_meas)
    return np.ravel(residuals)

def objective_abc(params, m_fixed, positions, B_measured, quaternions, lambda_reg=1e-3):
    a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z = params
    res = residuals_abc(params, m_fixed, positions, B_measured, quaternions)
    # Regularization to prefer simpler models (closer to identity transformation)
    reg = lambda_reg * (
        (a_x - 0.0)**2 + (b_x - 1.0)**2 + (c_x - 0.0)**2 +
        (a_y - 0.0)**2 + (b_y - 1.0)**2 + (c_y - 0.0)**2 +
        (a_z - 0.0)**2 + (b_z - 1.0)**2 + (c_z - 0.0)**2
    )
    return np.sum(res**2) + reg

def residuals_position_enhanced(r, B_measured, m, R_sensor, a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z):
    """Residuals for position estimation using enhanced model"""
    B_pred = dipole_field_enhanced(r, m, R_sensor, a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z)
    return B_pred - B_measured

def initial_position_guess(B_measured, m, R_sensor, pos_bounds):
    B_global = R_sensor @ B_measured
    B_mag = np.linalg.norm(B_global)
    m_mag = np.linalg.norm(m)
    
    if B_mag < 1e-10 or m_mag < 1e-10:
        return np.mean(pos_bounds, axis=0)
    
    # Use multiple initial guesses including bounds and center
    guesses = []
    
    # Traditional dipole estimate
    r_est = ((mu_0 / (4 * np.pi)) * (2 * m_mag) / B_mag) ** (1/3)
    B_unit = B_global / B_mag
    guesses.append(r_est * B_unit)
    
    # Add center point
    guesses.append(np.mean(pos_bounds, axis=0))
    
    # Add points near bounds (but not exactly at bounds)
    bounds_center = np.mean(pos_bounds, axis=0)
    for i in range(3):
        point = bounds_center.copy()
        point[i] = pos_bounds[0][i] + 0.1 * (pos_bounds[1][i] - pos_bounds[0][i])
        guesses.append(point)
        
        point = bounds_center.copy()
        point[i] = pos_bounds[1][i] - 0.1 * (pos_bounds[1][i] - pos_bounds[0][i])
        guesses.append(point)
    
    # Evaluate which guess has smallest residual
    best_guess = None
    best_residual = np.inf
    
    for guess in guesses:
        guess_clipped = np.clip(guess, pos_bounds[0], pos_bounds[1])
        # Use enhanced model with default coefficients for initial guess
        residual = np.linalg.norm(residuals_position_enhanced(guess_clipped, B_measured, m, R_sensor, 
                                                             0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0))
        if residual < best_residual:
            best_residual = residual
            best_guess = guess_clipped
            
    return best_guess

def compute_metrics_enhanced(m, positions, B_measured, quaternions, 
                           a_x=0.0, b_x=1.0, c_x=0.0, 
                           a_y=0.0, b_y=1.0, c_y=0.0, 
                           a_z=0.0, b_z=1.0, c_z=0.0):
    """Compute metrics using enhanced dipole field model"""
    B_pred = np.array([
        dipole_field_enhanced(r, m, quaternion_to_rotation_matrix(quat), 
                            a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z)
        for r, quat in zip(positions, quaternions)
    ])
    errors = B_pred - B_measured
    rmse = np.sqrt(np.mean(errors**2)) * 1e6
    angular_errors = []
    for b_p, b_m in zip(B_pred, B_measured):
        if np.linalg.norm(b_m) > 1e-10 and np.linalg.norm(b_p) > 1e-10:
            cos_theta = np.dot(b_p, b_m) / (np.linalg.norm(b_p) * np.linalg.norm(b_m))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta) * 180 / np.pi
            angular_errors.append(theta)
        else:
            angular_errors.append(np.nan)
    mean_angular_error = np.nanmean(angular_errors) if angular_errors else np.nan
    return rmse, mean_angular_error, angular_errors, B_pred * 1e6

# For backward compatibility
def compute_metrics(m, positions, B_measured, quaternions, a_x=1.0, b_x=0.0, a_y=1.0, b_y=0.0, a_z=1.0, b_z=0.0):
    return compute_metrics_enhanced(m, positions, B_measured, quaternions, 
                                  0.0, a_x, b_x, 0.0, a_y, b_y, 0.0, a_z, b_z)

def estimate_position_custom(B_custom, quat_sensor, m, pos_bounds, 
                           a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z, 
                           xgb_models=None):
    """Enhanced position estimation with optional ML correction"""
    if any(np.isnan(B_custom)) or any(np.isnan(quat_sensor)) or any(abs(b) > max_Bfield for b in B_custom):
        print(f"Invalid input: B_custom={B_custom}, quat_sensor={quat_sensor}")
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    
    B_custom_T = np.array(B_custom) * muT_to_T
    R_sensor = quaternion_to_rotation_matrix(quat_sensor)
    r_init = initial_position_guess(B_custom_T, m, R_sensor, pos_bounds)
    
    try:
        # Physics-based optimization
        result = least_squares(residuals_position_enhanced, r_init,
                              args=(B_custom_T, m, R_sensor, a_x, b_x, c_x, 
                                   a_y, b_y, c_y, a_z, b_z, c_z),
                              bounds=pos_bounds, method='trf', ftol=1e-6, xtol=1e-6)
        
        if not result.success or np.any(np.isnan(result.x)):
            print(f"Optimization failed for B_custom={B_custom}")
            return r_init, r_init
        
        r_est = result.x
        
        # Apply ML correction if models are provided
        if xgb_models is not None and all(model is not None for model in xgb_models):
            xgb_x, xgb_y, xgb_z = xgb_models
            
            # Prepare features for correction model
            features = np.array([[B_custom[0], B_custom[1], B_custom[2],
                                 quat_sensor[0], quat_sensor[1], quat_sensor[2], quat_sensor[3],
                                 r_est[0], r_est[1], r_est[2]]])
            
            if np.any(np.isnan(features)):
                print(f"NaN in features: {features}")
                return r_est, r_init
            
            # Predict errors
            error_x_pred = xgb_x.predict(features)[0]
            error_y_pred = xgb_y.predict(features)[0]
            error_z_pred = xgb_z.predict(features)[0]
            
            # Correct position
            r_ml = np.array([r_est[0] - error_x_pred, 
                            r_est[1] - error_y_pred, 
                            r_est[2] - error_z_pred])
            
            return r_ml, r_init
        
        return r_est, r_init
        
    except Exception as e:
        print(f"Error estimating position: {str(e)}")
        return r_init, r_init

def train_correction_model(data_df, save=True):
    """
    Train XGBoost models and optionally save them
    """
    features = ['Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT', 
                'Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz',
                'Est_Xm', 'Est_Ym', 'Est_Zm']
    targets = ['Error_Xm', 'Error_Ym', 'Error_Zm']
    
    missing_cols = [col for col in features + targets if col not in data_df.columns]
    if missing_cols:
        print(f"Cannot train correction model. Missing columns: {missing_cols}")
        return None, None, None
    
    valid_data = data_df[features + targets].dropna()
    print(f"Training data shape after dropping NaN: {valid_data.shape}")
    
    if len(valid_data) < 10:  # Increased minimum for better training
        print(f"⚠️ Too few valid data points ({len(valid_data)}) for training. Need at least 10.")
        return None, None, None
    
    X = valid_data[features].values
    y_x = valid_data['Error_Xm'].values
    y_y = valid_data['Error_Ym'].values
    y_z = valid_data['Error_Zm'].values
    
    print(f"Training data shapes - X: {X.shape}")
    print(f"Error ranges - X: [{y_x.min():.6f}, {y_x.max():.6f}] m")
    print(f"               Y: [{y_y.min():.6f}, {y_y.max():.6f}] m")
    print(f"               Z: [{y_z.min():.6f}, {y_z.max():.6f}] m")
    
    # Split data
    X_train, X_val, y_x_train, y_x_val = train_test_split(X, y_x, test_size=0.2, random_state=42)
    _, _, y_y_train, y_y_val = train_test_split(X, y_y, test_size=0.2, random_state=42)
    _, _, y_z_train, y_z_val = train_test_split(X, y_z, test_size=0.2, random_state=42)
    
    # Train models with better parameters
    xgb_x = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_y = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_z = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    # Train
    xgb_x.fit(X_train, y_x_train)
    xgb_y.fit(X_train, y_y_train)
    xgb_z.fit(X_train, y_z_train)
    
    # Evaluate
    mse_x = mean_squared_error(y_x_val, xgb_x.predict(X_val))
    mse_y = mean_squared_error(y_y_val, xgb_y.predict(X_val))
    mse_z = mean_squared_error(y_z_val, xgb_z.predict(X_val))
    
    rmse_x = np.sqrt(mse_x) * 1000  # Convert to mm
    rmse_y = np.sqrt(mse_y) * 1000
    rmse_z = np.sqrt(mse_z) * 1000
    
    print(f"\n📊 Correction Model Performance:")
    print(f"   X: RMSE = {rmse_x:.2f} mm")
    print(f"   Y: RMSE = {rmse_y:.2f} mm")
    print(f"   Z: RMSE = {rmse_z:.2f} mm")
    
    # Feature importance
    importance_x = xgb_x.feature_importances_
    print(f"\n📈 Feature Importances:")
    for name, imp in zip(features, importance_x):
        print(f"   {name}: {imp:.3f}")
    
    # Save models if requested
    if save:
        save_correction_models(xgb_x, xgb_y, xgb_z)
    
    return xgb_x, xgb_y, xgb_z

def apply_correction_model(data_df, xgb_x, xgb_y, xgb_z):
    """
    Apply the correction model to adjust estimated positions and compute new errors.
    Returns updated DataFrame with corrected positions and errors.
    """
    if xgb_x is None or xgb_y is None or xgb_z is None:
        print("No correction models available. Using original estimates.")
        return data_df
    
    features = ['Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT', 
                'Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz',
                'Est_Xm', 'Est_Ym', 'Est_Zm']
    
    valid_data = data_df[features].dropna()
    X = valid_data[features].values
    
    # Predict errors
    error_x_pred = np.zeros(len(data_df))
    error_y_pred = np.zeros(len(data_df))
    error_z_pred = np.zeros(len(data_df))
    
    valid_indices = valid_data.index
    error_x_pred[valid_indices] = xgb_x.predict(X)
    error_y_pred[valid_indices] = xgb_y.predict(X)
    error_z_pred[valid_indices] = xgb_z.predict(X)
    
    # Correct estimated positions
    data_df['Corrected_Xm'] = data_df['Est_Xm'] - error_x_pred
    data_df['Corrected_Ym'] = data_df['Est_Ym'] - error_y_pred
    data_df['Corrected_Zm'] = data_df['Est_Zm'] - error_z_pred
    
    # Compute new errors
    data_df['Corrected_Error_Xm'] = data_df['Corrected_Xm'] - data_df['Rel_X_m']
    data_df['Corrected_Error_Ym'] = data_df['Corrected_Ym'] - data_df['Rel_Y_m']
    data_df['Corrected_Error_Zm'] = data_df['Corrected_Zm'] - data_df['Rel_Z_m']
    
    return data_df

def interactive_position_estimation(m, pos_bounds, a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z, true_positions=None):
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
            if any(abs(m) > max_Bfield for m in B_custom):
                print(f"Invalid magnetic field values: {B_custom} exceed ±{max_Bfield} µT")
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
        r_est, r_init = estimate_position_custom(B_custom, quat_sensor, m, pos_bounds, a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z)
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
            return None, None, None, None
    
    if data is None:
        print("Could not read CSV with any encoding.")
        return None, None, None, None

    data.columns = data.columns.str.replace('µ', 'u', regex=False)
    data.columns = data.columns.str.replace('μ', 'u', regex=False)

    required_columns = ['Order', 'Rel_X_m', 'Rel_Y_m', 'Rel_Z_m', 
                       'Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz',
                       'Cube2_Pitch_deg', 'Cube2_Roll_deg', 'Cube2_Yaw_deg',
                       'Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing columns in CSV: {missing_columns}")
        print(f"Available columns: {list(data.columns)}")
        return None, None, None, None

    # Initialize estimation and error columns if missing
    for col in ['Est_Xm', 'Est_Ym', 'Est_Zm', 'Error_Xm', 'Error_Ym', 'Error_Zm',
                'Corrected_Xm', 'Corrected_Ym', 'Corrected_Zm', 
                'Corrected_Error_Xm', 'Corrected_Error_Ym', 'Corrected_Error_Zm']:
        if col not in data.columns:
            data[col] = np.nan

    # Filter rows with NaN in required columns
    mask = (data['Order'].notna() & data['Rel_X_m'].notna() & data['Rel_Y_m'].notna() & 
            data['Rel_Z_m'].notna() & data['Mag_X_uT'].notna() & data['Mag_Y_uT'].notna() & 
            data['Mag_Z_uT'].notna() & data['Cube2_Qw'].notna() & data['Cube2_Qx'].notna() &
            data['Cube2_Qy'].notna() & data['Cube2_Qz'].notna() &
            data['Cube2_Pitch_deg'].notna() & data['Cube2_Roll_deg'].notna() & 
            data['Cube2_Yaw_deg'].notna())
    data = data[mask].reset_index(drop=True)

    positions = data[['Rel_X_m', 'Rel_Y_m', 'Rel_Z_m']].values
    B_measured = data[['Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT']].values * muT_to_T
    quaternions = data[['Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz']].values

    # Normalize quaternions
    for i, q in enumerate(quaternions):
        norm = np.sqrt(np.sum(q**2))
        if norm < 1e-6:
            print(f"Invalid quaternion at index {i} (norm={norm:.6f}). Setting to identity.")
            quaternions[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            quaternions[i] = q / norm

    # Filter extreme values
    min_distance = 0
    max_B = 20000 * muT_to_T
    mask = (np.linalg.norm(positions, axis=1) > min_distance) & (np.linalg.norm(B_measured, axis=1) < max_B)
    positions = positions[mask]
    B_measured = B_measured[mask]
    quaternions = quaternions[mask]
    data = data[mask].reset_index(drop=True)

    # Reorder columns to match desired order
    desired_columns = ['Order', 'Rel_X_m', 'Rel_Y_m', 'Rel_Z_m', 
                       'Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz',
                       'Cube2_Pitch_deg', 'Cube2_Roll_deg', 'Cube2_Yaw_deg',
                       'Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT',
                       'Est_Xm', 'Est_Ym', 'Est_Zm', 'Error_Xm', 'Error_Ym', 'Error_Zm',
                       'Corrected_Xm', 'Corrected_Ym', 'Corrected_Zm', 
                       'Corrected_Error_Xm', 'Corrected_Error_Ym', 'Corrected_Error_Zm']
    other_cols = [col for col in data.columns if col not in desired_columns]
    data = data[desired_columns + other_cols]

    return positions, B_measured, quaternions, data

def run_csv_interactive(dipole_params, pos_bounds, positions):
    positions_data, B_measured, quaternions, data_df = load_and_filter_data()
    if positions_data is None:
        print("Failed to load data for CSV interactive estimation.")
        return

    if len(positions_data) < 5:
        print(f"Too few measurements ({len(positions_data)}). At least 5 are recommended.")
        return

    # Clear existing estimation and correction columns
    for col in ['Est_Xm', 'Est_Ym', 'Est_Zm', 'Error_Xm', 'Error_Ym', 'Error_Zm',
                'Corrected_Xm', 'Corrected_Ym', 'Corrected_Zm', 
                'Corrected_Error_Xm', 'Corrected_Error_Ym', 'Corrected_Error_Zm']:
        data_df[col] = np.nan

    r_estimated = np.zeros_like(positions_data)
    r_initial = np.zeros_like(positions_data)
    pos_errors = np.zeros_like(positions_data)

    print("Computing position estimates for all points...")
    for i, (B, quat) in enumerate(zip(B_measured, quaternions)):
        try:
            if any(np.isnan(B)) or any(np.isnan(quat)):
                print(f"Skipping point {i} due to NaN: B={B}, quat={quat}")
                r_estimated[i] = positions_data[i]
                r_initial[i] = positions_data[i]
                pos_errors[i] = [0.0, 0.0, 0.0]
                continue
            if any(abs(b / muT_to_T) > max_Bfield for b in B):
                print(f"Skipping point {i}: Magnetic field {B / muT_to_T} exceeds ±{max_Bfield} µT")
                r_estimated[i] = positions_data[i]
                r_initial[i] = positions_data[i]
                pos_errors[i] = [0.0, 0.0, 0.0]
                continue
            R_sensor = quaternion_to_rotation_matrix(quat)
            r_init = initial_position_guess(B, dipole_params['m'], R_sensor, pos_bounds)
            r_initial[i] = r_init
            result = least_squares(residuals_position_enhanced, r_init,
                                  args=(B, dipole_params['m'], R_sensor,
                                        dipole_params['a_x'], dipole_params['b_x'], dipole_params['c_x'],
                                        dipole_params['a_y'], dipole_params['b_y'], dipole_params['c_y'],
                                        dipole_params['a_z'], dipole_params['b_z'], dipole_params['c_z']),
                                  bounds=pos_bounds, method='dogbox', ftol=1e-6, xtol=1e-6)
            if not result.success or np.any(np.isnan(result.x)):
                print(f"Optimization failed for point {i}: B={B / muT_to_T}")
                r_estimated[i] = r_init
                pos_errors[i] = r_init - positions_data[i]
            else:
                r_estimated[i] = result.x
                pos_errors[i] = result.x - positions_data[i]
            print(f"Point {i}: Est_Xm={r_estimated[i,0]:.6f}, Est_Ym={r_estimated[i,1]:.6f}, Est_Zm={r_estimated[i,2]:.6f}\t\t Initial Guess: x={r_init[0]:.6f}, y={r_init[1]:.6f}, z={r_init[2]:.6f}\t\t Est-Init: x={r_estimated[i,0]-r_init[0]:.6f}, y={r_estimated[i,1]-r_init[1]:.6f}, z={r_estimated[i,2]-r_init[2]:.6f}")
        except Exception as e:
            print(f"Error estimating position for point {i}: {str(e)}")
            r_estimated[i] = positions_data[i]
            r_initial[i] = positions_data[i]
            pos_errors[i] = [0.0, 0.0, 0.0]

    # Update DataFrame with estimated positions and errors
    data_df['Est_Xm'] = r_estimated[:, 0]
    data_df['Est_Ym'] = r_estimated[:, 1]
    data_df['Est_Zm'] = r_estimated[:, 2]
    data_df['Error_Xm'] = pos_errors[:, 0]
    data_df['Error_Ym'] = pos_errors[:, 1]
    data_df['Error_Zm'] = pos_errors[:, 2]

    # Reorder columns to match desired order
    desired_columns = ['Order', 'Rel_X_m', 'Rel_Y_m', 'Rel_Z_m', 
                       'Cube2_Qw', 'Cube2_Qx', 'Cube2_Qy', 'Cube2_Qz',
                       'Cube2_Pitch_deg', 'Cube2_Roll_deg', 'Cube2_Yaw_deg',
                       'Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT',
                       'Est_Xm', 'Est_Ym', 'Est_Zm', 'Error_Xm', 'Error_Ym', 'Error_Zm',
                       'Corrected_Xm', 'Corrected_Ym', 'Corrected_Zm', 
                       'Corrected_Error_Xm', 'Corrected_Error_Ym', 'Corrected_Error_Zm']
    other_cols = [col for col in data_df.columns if col not in desired_columns]
    data_df = data_df[desired_columns + other_cols]

    # Save updated CSV before training correction model
    try:
        data_df.to_csv(CSV_FILE, index=False, encoding='utf-8')
        print(f"Updated {CSV_FILE} with estimated positions and errors")
    except Exception as e:
        print(f"Error saving CSV file: {str(e)}")
        return

    # Train correction model
    xgb_x, xgb_y, xgb_z = train_correction_model(data_df)
    
    # Apply correction model
    if xgb_x is not None:
        data_df = apply_correction_model(data_df, xgb_x, xgb_y, xgb_z)
        # Reorder columns again to maintain desired order
        data_df = data_df[desired_columns + other_cols]
        try:
            data_df.to_csv(CSV_FILE, index=False, encoding='utf-8')
            print(f"Updated {CSV_FILE} with corrected positions")
        except Exception as e:
            print(f"Error saving CSV file with corrected positions: {str(e)}")

    # Compute statistics for original and corrected errors
    original_errors = np.linalg.norm(pos_errors, axis=1)
    corrected_errors = np.linalg.norm(data_df[['Corrected_Error_Xm', 'Corrected_Error_Ym', 'Corrected_Error_Zm']].values, axis=1)
    
    mean_pos_error = np.mean(original_errors)
    std_pos_error = np.std(original_errors)
    mean_corrected_error = np.nanmean(corrected_errors)
    std_corrected_error = np.nanstd(corrected_errors)
    
    print(f"Position Estimation Results (Original):")
    print(f"  Mean Positional Error = {mean_pos_error:.6f} m")
    print(f"  Std Positional Error = {std_pos_error:.6f} m")
    print(f"Position Estimation Results (Corrected):")
    print(f"  Mean Corrected Positional Error = {mean_corrected_error:.6f} m")
    print(f"  Std Corrected Error = {std_corrected_error:.6f} m")

    # Plotting original vs estimated vs corrected positions
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions_data[:, 0], positions_data[:, 1], positions_data[:, 2], c='b', label='Actual Positions', alpha=0.5)
    ax.scatter(r_estimated[:, 0], r_estimated[:, 1], r_estimated[:, 2], c='r', label='Estimated Positions', alpha=0.5)
    ax.scatter(data_df['Corrected_Xm'], data_df['Corrected_Ym'], data_df['Corrected_Zm'], c='g', label='Corrected Positions', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Actual vs Estimated vs Corrected Positions')
    ax.legend()
    plt.savefig('position_scatter_3d.png')
    plt.show()

    # Plot histograms of errors
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(original_errors, bins=20, alpha=0.5, label='Original Errors', color='red')
    ax.hist(corrected_errors, bins=20, alpha=0.5, label='Corrected Errors', color='green')
    ax.set_xlabel('Positional Error (m)')
    ax.set_title('Original vs Corrected Positional Errors')
    ax.legend()
    plt.savefig('positional_errors_histogram_csv.png')
    plt.close()

def run_optimization_enhanced():
    positions, B_measured, quaternions, data_df = load_and_filter_data()
    if positions is None:
        print("Failed to load data for optimization.")
        return None, None, None, None, None, None, None, None

    N = len(positions)
    if N < 5:
        print(f"Too few measurements ({N}). At least 5 are recommended.")
        return None, None, None, None, None, None, None, None

    # First stage: optimize dipole moment m with simple model
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

    methods = ['TRF', 'Dogbox', 'LM', 'Differential Evolution', 'Nelder-Mead']
    results = []
    bounds_m = ([-1000, -1000, -1000], [0, 1000, 1000])

    for method in methods:
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
        return None, None, None, None, None, None, None, None
    best_result = min(valid_results, key=lambda x: (x['RMSE'], x['Angular Error']))
    m_best = best_result['m']
    rmse_best = best_result['RMSE']
    ang_err_best = best_result['Angular Error']
    ang_errs_best = best_result['Angular Errors']
    B_pred_best = best_result['B_pred']
    best_method = best_result['Method']

    # Second stage: optimize quadratic coefficients with fixed dipole moment
    start_time = time.time()
    abc0 = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]  # [a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z]
    
    # Set bounds for quadratic coefficients
    abc_bounds = [
        (-10, 10), (-10, 10), (-1, 1),  # x coefficients
        (-10, 10), (-10, 10), (-1, 1),  # y coefficients  
        (-10, 10), (-10, 10), (-1, 1)   # z coefficients
    ]
    
    try:
        result_abc = minimize(objective_abc, abc0, args=(m_best, positions, B_measured, quaternions, 1e-3),
                            method='Nelder-Mead',
                            bounds=abc_bounds,
                            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
        
        a_x_best, b_x_best, c_x_best, a_y_best, b_y_best, c_y_best, a_z_best, b_z_best, c_z_best = result_abc.x
        time_taken = time.time() - start_time
        
        rmse_final, ang_err_final, ang_errs_final, B_pred_final = compute_metrics_enhanced(
            m_best, positions, B_measured, quaternions, 
            a_x_best, b_x_best, c_x_best, a_y_best, b_y_best, c_y_best, a_z_best, b_z_best, c_z_best
        )
        
        print(f"Quadratic optimization completed in {time_taken:.2f}s")
        print(f"X transformation: {a_x_best:.6f}*x² + {b_x_best:.6f}*x + {c_x_best:.6f}")
        print(f"Y transformation: {a_y_best:.6f}*y² + {b_y_best:.6f}*y + {c_y_best:.6f}")  
        print(f"Z transformation: {a_z_best:.6f}*z² + {b_z_best:.6f}*z + {c_z_best:.6f}")
        
    except Exception as e:
        print(f"Error in Quadratic Stage (Nelder-Mead): {str(e)}")
        # Fall back to linear model
        a_x_best, b_x_best, c_x_best, a_y_best, b_y_best, c_y_best, a_z_best, b_z_best, c_z_best = 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
        rmse_final, ang_err_final, ang_errs_final, B_pred_final = rmse_best, ang_err_best, ang_errs_best, B_pred_best

    # Third stage: re-optimize dipole moment with fixed quadratic coefficients
    start_time = time.time()
    try:
        def residuals_m_third_enhanced(m, positions, B_measured, quaternions, a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z):
            residuals = []
            for r, B_meas, quat in zip(positions, B_measured, quaternions):
                R_sensor = quaternion_to_rotation_matrix(quat)
                B_pred = dipole_field_enhanced(r, m, R_sensor, a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z)
                residuals.append(B_pred - B_meas)
            return np.ravel(residuals)
        
        result_m_third = least_squares(residuals_m_third_enhanced, m_best,
                                      args=(positions, B_measured, quaternions, 
                                            a_x_best, b_x_best, c_x_best, 
                                            a_y_best, b_y_best, c_y_best, 
                                            a_z_best, b_z_best, c_z_best),
                                      bounds=bounds_m, method='trf', ftol=1e-8, xtol=1e-8)
        m_third = result_m_third.x
        time_taken = time.time() - start_time
        
        rmse_third, ang_err_third, ang_errs_third, B_pred_third = compute_metrics_enhanced(
            m_third, positions, B_measured, quaternions,
            a_x_best, b_x_best, c_x_best, a_y_best, b_y_best, c_y_best, a_z_best, b_z_best, c_z_best
        )
        
    except Exception as e:
        print(f"Error in Third Stage (TRF): {str(e)}")
        m_third = m_best
        rmse_third, ang_err_third, ang_errs_third, B_pred_third = rmse_final, ang_err_final, ang_errs_final, B_pred_final

    # Choose the best result
    if rmse_third < rmse_final:
        m_final = m_third
        rmse_final = rmse_third
        ang_err_final = ang_err_third
        ang_errs_final = ang_errs_third
        B_pred_final = B_pred_third
        best_stage = "Third Stage (TRF with Quadratic)"
    else:
        m_final = m_best
        best_stage = f"Quadratic Stage ({best_method})"

    dipole_params = {
        'm': m_final,
        'a_x': a_x_best, 'b_x': b_x_best, 'c_x': c_x_best,
        'a_y': a_y_best, 'b_y': b_y_best, 'c_y': c_y_best,
        'a_z': a_z_best, 'b_z': b_z_best, 'c_z': c_z_best
    }

    # Save parameters
    params = {
        'm_x': float(m_final[0]),
        'm_y': float(m_final[1]),
        'm_z': float(m_final[2]),
        'a_x': float(a_x_best), 'b_x': float(b_x_best), 'c_x': float(c_x_best),
        'a_y': float(a_y_best), 'b_y': float(b_y_best), 'c_y': float(c_y_best),
        'a_z': float(a_z_best), 'b_z': float(b_z_best), 'c_z': float(c_z_best),
        'rmse': float(rmse_final),
        'angular_error': float(ang_err_final)
    }
    
    with open('dipole_model_params.json', 'w') as f:
        json.dump(params, f, indent=4)

    pos_bounds = (
        [positions[:,0].min(), positions[:,1].min(), positions[:,2].min()],
        [positions[:,0].max(), positions[:,1].max(), positions[:,2].max()]
    )
    
    return dipole_params, pos_bounds, positions, B_measured, quaternions, ang_errs_final, B_pred_final, best_stage

def generate_vector_plots(positions, B_measured, quaternions, dipole_params, best_stage, ang_errs_final, B_pred_final):
    N = len(positions)
    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    z_vals = positions[:, 2]
    U_meas, V_meas, W_meas = np.zeros(N), np.zeros(N), np.zeros(N)
    U_theo, V_theo, W_theo = np.zeros(N), np.zeros(N), np.zeros(N)
    U_opt, V_opt, W_opt = np.zeros(N), np.zeros(N), np.zeros(N)

    for i, (r, B_m, quat) in enumerate(zip(positions, B_measured, quaternions)):
        R_sensor = quaternion_to_rotation_matrix(quat)
        # Use enhanced model for all calculations
        B_theo = dipole_field_enhanced(r, np.array([-0.1, 0, 0]), R_sensor) * 1e6
        B_opt = dipole_field_enhanced(r, dipole_params['m'], R_sensor,
                                     dipole_params['a_x'], dipole_params['b_x'], dipole_params['c_x'],
                                     dipole_params['a_y'], dipole_params['b_y'], dipole_params['c_y'],
                                     dipole_params['a_z'], dipole_params['b_z'], dipole_params['c_z']) * 1e6
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
    plt.savefig('vector_plot_2d.png')
    plt.show()

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
    plt.savefig('vector_plot_3d.png')
    plt.show()

    residuals_final = B_pred_final - B_measured * 1e6
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, component in enumerate(['X', 'Y', 'Z']):
        axes[i].scatter(B_measured[:, i] * 1e6, B_pred_final[:, i], alpha=0.5)
        axes[i].plot([B_measured[:, i].min() * 1e6, B_measured[:, i].max() * 1e6],
                     [B_measured[:, i].min() * 1e6, B_measured[:, i].max() * 1e6], 'r--')
        axes[i].set_xlabel(f'Measured B_{component} (µT)')
        axes[i].set_ylabel(f'Predicted B_{component} (µT)')
        axes[i].set_title(f'B_{component} Fit')
    plt.tight_layout()
    plt.savefig('scatter_plots.png')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, component in enumerate(['X', 'Y', 'Z']):
        axes[i].hist(residuals_final[:, i], bins=20)
        axes[i].set_xlabel(f'Residual B_{component} (µT)')
        axes[i].set_title(f'B_{component} Residuals')
    plt.tight_layout()
    plt.savefig('residuals_histogram.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([theta for theta in ang_errs_final if not np.isnan(theta)], bins=20)
    ax.set_xlabel('Angular Error (degrees)')
    ax.set_title('Angular Errors')
    plt.savefig('angular_errors_histogram.png')
    plt.close()

# --- New function to analyze comparison results ---
def analyze_ml_performance():
    """Analyze and plot ML vs Physics performance"""
    if not os.path.exists(COMPARISON_LOG):
        print("No comparison log found.")
        return
    
    df = pd.read_csv(COMPARISON_LOG)
    
    if len(df) == 0:
        print("No data in comparison log.")
        return
    
    # Filter rows with valid errors
    df_valid = df.dropna(subset=['physics_error', 'ml_error'])
    
    if len(df_valid) == 0:
        print("No valid comparison data with ground truth.")
        return
    
    # Calculate statistics
    physics_errors = df_valid['physics_error'].values * 1000  # Convert to mm
    ml_errors = df_valid['ml_error'].values * 1000
    
    physics_mean = np.mean(physics_errors)
    physics_std = np.std(physics_errors)
    ml_mean = np.mean(ml_errors)
    ml_std = np.std(ml_errors)
    
    improvements = ((physics_errors - ml_errors) / physics_errors * 100)
    avg_improvement = np.mean(improvements)
    
    print("\n" + "="*60)
    print("📊 ML PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Total samples with ground truth: {len(df_valid)}")
    print(f"\nPhysics-only:")
    print(f"  Mean error: {physics_mean:.2f} mm")
    print(f"  Std dev: {physics_std:.2f} mm")
    print(f"\nML-corrected:")
    print(f"  Mean error: {ml_mean:.2f} mm")
    print(f"  Std dev: {ml_std:.2f} mm")
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram comparison
    axes[0,0].hist(physics_errors, bins=20, alpha=0.5, label='Physics', color='red')
    axes[0,0].hist(ml_errors, bins=20, alpha=0.5, label='ML', color='green')
    axes[0,0].set_xlabel('Error (mm)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Error Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[0,1].scatter(physics_errors, ml_errors, alpha=0.5)
    max_err = max(physics_errors.max(), ml_errors.max())
    axes[0,1].plot([0, max_err], [0, max_err], 'r--', label='Equal performance')
    axes[0,1].plot([0, max_err], [0, max_err*0.5], 'g--', label='50% improvement')
    axes[0,1].set_xlabel('Physics Error (mm)')
    axes[0,1].set_ylabel('ML Error (mm)')
    axes[0,1].set_title('Physics vs ML Error')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Improvement histogram
    axes[1,0].hist(improvements, bins=20, color='blue', alpha=0.7)
    axes[1,0].axvline(x=0, color='red', linestyle='--', label='No improvement')
    axes[1,0].set_xlabel('Improvement (%)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Improvement Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Time series if timestamp available
    if 'timestamp' in df_valid.columns:
        times = pd.to_datetime(df_valid['timestamp'])
        axes[1,1].plot(times, physics_errors, 'r-', label='Physics', alpha=0.7)
        axes[1,1].plot(times, ml_errors, 'g-', label='ML', alpha=0.7)
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Error (mm)')
        axes[1,1].set_title('Error over Time')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('ml_performance_analysis.png', dpi=150)
    plt.show()
    
    return {
        'physics_mean': physics_mean,
        'physics_std': physics_std,
        'ml_mean': ml_mean,
        'ml_std': ml_std,
        'avg_improvement': avg_improvement
    }



#### BORESIGHT CALIBRATION
def estimate_position_single(b_target, dipole_params, pos_bounds, R_sensor):
    """
    Helper function to estimate a single position based on a target B-field.
    """
    # Load polynomial distortion parameters
    a_x = dipole_params.get('a_x', 0.0)
    b_x = dipole_params.get('b_x', 1.0)
    c_x = dipole_params.get('c_x', 0.0)
    a_y = dipole_params.get('a_y', 0.0)
    b_y = dipole_params.get('b_y', 1.0)
    c_y = dipole_params.get('c_y', 0.0)
    a_z = dipole_params.get('a_z', 0.0)
    b_z = dipole_params.get('b_z', 1.0)
    c_z = dipole_params.get('c_z', 0.0)
    m_fixed = dipole_params['m']

    def pos_objective(pos):
        # Calculate theoretical B-field at given position 'pos'
        b_theoretical = dipole_field_enhanced(
            pos, m_fixed, R_sensor, 
            a_x, b_x, c_x, 
            a_y, b_y, c_y, 
            a_z, b_z, c_z
        )
        b_theoretical *= 1e6  # Convert to microtesla
        # print(f"Testing position: {pos}, Theoretical B: {b_theoretical}, Target B: {b_target}")
        # return the raw residual vector. least_squares squares it automatically.
        return b_theoretical - b_target
    
    # Correctly separate lower and upper limits
    lower_bounds = np.array(pos_bounds[0])  # [xmin, ymin, zmin]
    upper_bounds = np.array(pos_bounds[1])  # [xmax, ymax, zmax]
    
    # Calculate the midpoint across each axis for the initial guess
    initial_guess = (lower_bounds + upper_bounds) / 2.0
    
    # Run trust-region reflective least_squares optimization to find position
    res = least_squares(
        pos_objective, 
        x0=initial_guess, 
        bounds=(lower_bounds, upper_bounds),
        method='trf'
    )
    print(f"  Estimated position: {res.x}, Residual norm: {res.cost:.6f}")
    return res.x


def boresight_objective(boresight_angles, df, dipole_params, pos_bounds, return_stats=False):
    r_boresight = R.from_euler('xyz', boresight_angles, degrees=True)
    
    errors = []
    
    for _, row in df.iterrows():
        b_raw = np.array([row['Mag_X_uT'], row['Mag_Y_uT'], row['Mag_Z_uT']])
        r_sensor_base = R.from_quat([row['Cube2_Qx'], row['Cube2_Qy'], row['Cube2_Qz'], row['Cube2_Qw']])
        
        r_sensor_corrected = r_sensor_base * r_boresight
        R_sensor_input = r_sensor_corrected.as_matrix() 
        
        b_target = b_raw 
        pos_gt = np.array([row['Rel_X_m'], row['Rel_Y_m'], row['Rel_Z_m']])
        
        pos_est = estimate_position_single(b_target, dipole_params, pos_bounds, R_sensor_input)
        error = np.linalg.norm(pos_est - pos_gt)
        errors.append(error)
        
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    rmse_error = np.sqrt(np.mean(errors**2))
    
    if return_stats:
        return mean_error, std_error, rmse_error
    
    return rmse_error
def run_dynamic_calibration(csv_file, dipole_params, pos_bounds, subsample_rate):
    """
    Main function to execute the boresight calibration.
    """
    print(f"Loading data from {csv_file} for dynamic calibration...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}.")
        return

    # Subsample data to speed up optimization (O(N) operation inside the optimizer)
    if subsample_rate > 1:
        df = df.iloc[::subsample_rate, :]
        print(f"Subsampled data by a factor of {subsample_rate}. Using {len(df)} points for calibration.")

    # Initial guess for boresight angles (0, 0, 0 degrees)
    initial_boresight = np.array([1.0, 2.0, 1.0])
    
    # Boresight bounds (e.g., +/- 45 degrees maximum misalignment)
    bounds = [(-45.0, 45.0), (-45.0, 45.0), (-45.0, 45.0)]

    print("Calculating initial error metrics before boresight calibration...")
   # --- Inside your calibration setup function ---

    print("Calculating initial error metrics before boresight calibration...")
    initial_angles = [0.0, 0.0, 0.0]

    # Unpack all 3 metrics for baseline tracking
    initial_mean, initial_std, initial_rmse = boresight_objective(
        initial_angles, df, dipole_params, pos_bounds, return_stats=True
    )
    print(f"Initial Metrics (No Boresight):")
    print(f"  Mean Error: {initial_mean:.4f} m")
    print(f"  Std Dev:    {initial_std:.4f} m")
    print(f"  RMSE:       {initial_rmse:.4f} m\n")

    print("Starting optimization to minimize RMSE...")
    start_time = time.time()

    # Pass False so the optimizer receives only the scalar RMSE value
    result = minimize(
        boresight_objective, 
        initial_angles, 
        method='Nelder-Mead', 
        args=(df, dipole_params, pos_bounds, False)
    )

    end_time = time.time()

    if result.success:
        # Evaluate the optimized angles to get the final statistics
        final_mean, final_std, final_rmse = boresight_objective(
            result.x, df, dipole_params, pos_bounds, return_stats=True
        )
        
        print("\n--- Calibration Successful ---")
        print(f"Optimization took {end_time - start_time:.2f} seconds.\n")
        
        print(f"Initial RMSE:                 {initial_rmse:.4f} m")
        print(f"Final RMSE:                   {final_rmse:.4f} m")
        print(f"RMSE Improvement:             {(initial_rmse - final_rmse)*1000:.1f} mm\n")
        
        print(f"Initial Mean Position Error:  {initial_mean:.4f} m (± {initial_std:.4f} m)")
        print(f"Final Mean Position Error:    {final_mean:.4f} m (± {final_std:.4f} m)")
        print(f"Mean Error Improvement:       {(initial_mean - final_mean)*1000:.1f} mm\n")
        
        print(f"Best Boresight Offsets (Degrees):")
        print(f"  Roll:  {result.x[0]:.8f}°")
        print(f"  Pitch: {result.x[1]:.8f}°")
        print(f"  Yaw:   {result.x[2]:.8f}°")




# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Position estimation server with ML comparison")
    parser.add_argument('--optimize', action='store_true', help="Run optimization to generate dipole parameters")
    parser.add_argument('--staged', action='store_true', help="Use staged optimization (fast dipole optimization on subset, ML on all data)")
    parser.add_argument('--opt-fraction', type=float, default=0.33, help="Fraction of data for dipole optimization (default: 0.33)")
    parser.add_argument('--interactive', action='store_true', help="Run CSV-based position estimation with plotting")
    parser.add_argument('--console', action='store_true', help="Run console-based interactive position estimation")
    parser.add_argument('--diagnose', action='store_true', help="Run magnet alignment diagnostic on CSV data")
    parser.add_argument('--analyze', action='store_true', help="Analyze ML performance from comparison log")
    parser.add_argument('--no-ml', action='store_true', help="Disable ML correction in server mode")
    parser.add_argument('--force-train', action='store_true', help="Force retraining of ML models")
    parser.add_argument('--cubic', action='store_true', help="Use cubic model instead of quadratic")
    parser.add_argument('--compare-models', action='store_true', help="Compare quadratic vs cubic performance")
    parser.add_argument('--calibrate-boresight', action='store_true', help="Run dynamic boresight calibration using Mag_Field_log.csv to minimize position error")
    args = parser.parse_args()

    # Initialize comparison log
    init_comparison_log()
    
    # Handle compare models mode
    if args.compare_models:
        compare_quadratic_vs_cubic()
        exit(0)

    # Initialize variables
    dipole_params = None
    pos_bounds = None
    positions = None
    B_measured = None
    quaternions = None
    ang_errs_final = None
    B_pred_final = None
    best_stage = None

    # Handle cubic optimization mode (runs optimization and saves file)
    if args.cubic:
        print("\n🔧 Running cubic optimization...")
        result = run_optimization_with_cubic()
        if result is None or len(result) < 8:
            print("Cubic optimization failed.")
            exit(1)
        dipole_params, pos_bounds, positions, B_measured, quaternions, ang_errs_final, B_pred_final, best_stage = result
        
        # After cubic optimization, we have the params, so skip loading
        # Set a flag to indicate we already have params
        params_already_loaded = True
    else:
        params_already_loaded = False

    # Handle analysis mode
    if args.analyze:
        analyze_ml_performance()
        exit(0)

    # Load dipole parameters if not already loaded from cubic optimization
    if not params_already_loaded:
        dipole_params = load_dipole_params()
        
    if dipole_params is None and not args.optimize:
        print("No valid dipole parameters. Run with --optimize to generate them.")
        exit(1)

    # Print which model is being used
    if dipole_params:
        model_type = dipole_params.get('model_type', 'quadratic')
        print(f"\n📌 Using {model_type.upper()} model for position estimation")
        if model_type == 'cubic':
            print(f"   Dipole moment: {dipole_params['m']}")
            print(f"   Quadratic corrections applied")
            print(f"   Cubic corrections applied")

    # Load data if we don't already have it from cubic optimization
    if positions is None:
        positions, B_measured, quaternions, data_df = load_and_filter_data()
    
    if pos_bounds is None and positions is not None:
        pos_bounds = (
            [positions[:,0].min(), positions[:,1].min(), positions[:,2].min()],
            [positions[:,0].max(), positions[:,1].max(), positions[:,2].max()]
        )
        print(f"Position bounds: {pos_bounds}")

    # Load data_df if needed
    if 'data_df' not in locals() or data_df is None:
        _, _, _, data_df = load_and_filter_data()

    # Handle ML models
    xgb_models = (None, None, None)
    
    if args.force_train or (args.optimize and args.interactive):
        # Force retraining
        if data_df is not None:
            print("\n🔄 Force training ML models...")
            xgb_models = train_correction_model(data_df, save=True)
    elif not args.no_ml:
        # Try to load existing models
        xgb_models = load_correction_models()
        
        # If no models found and we have data, train new ones
        if xgb_models[0] is None and data_df is not None:
            print("\n🔄 No existing ML models found. Training new ones...")
            xgb_models = train_correction_model(data_df, save=True)

    if args.diagnose:
        if data_df is not None:
            print("\nRunning magnet alignment diagnostic...")
            diagnostic_results = diagnose_pitch_misalignment(data_df)
            print("\nDiagnostic complete. Check 'pitch_diagnostic.png'")
            exit(0)
        else:
            print("Error: Could not load data for diagnostic")
            exit(1)

    if args.optimize and not args.cubic:  # Don't re-optimize if we already did cubic
        print("\n🔧 Running optimization...")
        
        if args.staged:
            # Use staged optimization (fast dipole on subset, ML on all data)
            result = run_optimization_staged(fraction=args.opt_fraction)
            if result is None or len(result) < 8:
                print("Staged optimization failed.")
                exit(1)
            dipole_params, pos_bounds, positions, B_measured, quaternions, ang_errs_final, B_pred_final, best_stage = result
        else:
            # Use original optimization
            result = run_optimization_enhanced()
            if result is None or len(result) < 8:
                print("Optimization failed.")
                exit(1)
            dipole_params, pos_bounds, positions, B_measured, quaternions, ang_errs_final, B_pred_final, best_stage = result
        
        if positions is not None:
            generate_vector_plots(positions, B_measured, quaternions, dipole_params, best_stage, ang_errs_final, B_pred_final)

    if dipole_params is None:
        print("No valid dipole parameters. Run with --optimize to generate them.")
        exit(1)

    # Start server with appropriate mode
    if args.console:
        server_thread = threading.Thread(
            target=run_position_server,
            args=(POSITION_SERVER_HOST, POSITION_SERVER_PORT, dipole_params, pos_bounds, xgb_models, not args.no_ml)
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Check if quadratic params exist for interactive mode
        if 'a_x' in dipole_params:
            interactive_position_estimation(
                dipole_params['m'], pos_bounds,
                dipole_params.get('a_x', 0.0), dipole_params.get('b_x', 1.0), dipole_params.get('c_x', 0.0),
                dipole_params.get('a_y', 0.0), dipole_params.get('b_y', 1.0), dipole_params.get('c_y', 0.0),
                dipole_params.get('a_z', 0.0), dipole_params.get('b_z', 1.0), dipole_params.get('c_z', 0.0),
                positions
            )
        else:
            print("Interactive mode only supported for quadratic model currently.")
            
    elif args.interactive:
        server_thread = threading.Thread(
            target=run_position_server,
            args=(POSITION_SERVER_HOST, POSITION_SERVER_PORT, dipole_params, pos_bounds, xgb_models, not args.no_ml)
        )
        server_thread.daemon = True
        server_thread.start()
        
        run_csv_interactive(dipole_params, pos_bounds, positions)
    if args.calibrate_boresight:
    # Assumes pos_bounds and dipole_params are already loaded/defined in main scope
        best_boresight = run_dynamic_calibration(
            'Mag_Field_log.csv', 
            dipole_params, 
            np.array(pos_bounds), 
            subsample_rate=1  # Change this to 1 to use ALL data (will take much longer)
        )
        if best_boresight is not None:
            print("\nTo use these offsets, apply them before your position estimation loop:")
            print(f"boresight_matrix = R.from_euler('xyz', {list(np.round(best_boresight, 3))}, degrees=True)")
            # You can also add logic here to save these to a JSON config file.
    else:
        # Normal server mode with ML enabled by default
        run_position_server(
            POSITION_SERVER_HOST, POSITION_SERVER_PORT, 
            dipole_params, pos_bounds, xgb_models, 
            enable_ml=not args.no_ml
        )