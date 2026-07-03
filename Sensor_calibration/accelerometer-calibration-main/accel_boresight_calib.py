import numpy as np

def calculate_pitch_roll_misalignment(file_path, stable_window_size=50, noise_threshold=0.005):
    """
    Parses comma-separated accelerometer logs, isolates a stable resting period,
    calculates pitch/roll misalignment angles, and generates a 3D correction matrix.
    Outputs the matrix parameters formatted as individual flat variables.
    Sequence order: Roll (X-axis) then Pitch (Y-axis).
    """
    data = []
    
    # 1. Parse the comma-separated text file safely
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                if len(parts) >= 3:
                    data.append([float(parts[0].strip()), 
                                 float(parts[1].strip()), 
                                 float(parts[2].strip())])
            except ValueError:
                print(f"Skipping malformed text/header on line {line_num}: {line}")

    if not data:
        raise ValueError("No valid data rows found in the log file.")

    accel_data = np.array(data)
    total_samples = len(accel_data)
    
    # 2. Find a stable resting window
    best_start_idx = 0
    min_variance = float('inf')
    
    # Safeguard window size if data log is shorter than default settings
    effective_window = min(stable_window_size, total_samples)
    
    for i in range(total_samples - effective_window + 1):
        window = accel_data[i : i + effective_window]
        window_variance = np.sum(np.var(window, axis=0))
        
        if window_variance < min_variance:
            min_variance = window_variance
            best_start_idx = i

    stable_data = accel_data[best_start_idx : best_start_idx + effective_window]
    
    if np.mean(np.std(stable_data, axis=0)) > noise_threshold:
        print("⚠️ Warning: Could not find a perfectly stationary window. Results might be noisy.")

    # 3. Compute the mean static gravity vector components
    ax_mean, ay_mean, az_mean = np.mean(stable_data, axis=0)
    
    # 4. Standard Industrial Aerospace Trigonometry (ISO/Euler conventions)
    roll_rad = np.arctan2(ay_mean, az_mean)
    pitch_rad = np.arctan2(-ax_mean, np.sqrt(ay_mean**2 + az_mean**2))
    
    roll_deg = np.degrees(roll_rad)
    pitch_deg = np.degrees(pitch_rad)
    
    # 5. Generate the 3D Correction Matrix
    # We invert the angles (negative sign) to rotate back to the ideal alignment
    alpha = -roll_rad
    beta = -pitch_rad
    
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
    
    # CHANGED: Flipped to R_y * R_x. Because matrix operators act from right-to-left,
    # this applies Roll (R_x) first, and then applies Pitch (R_y) second.
    R_correct = np.dot(R_y, R_x)
    
    # 6. Output Calibration Report with individual flat elements
    print("=" * 60)
    print("         IMU BORESIGHT CALIBRATION & CORRECTION REPORT        ")
    print("=" * 60)
    print(f"Analyzed {total_samples} CSV data points.")
    print(f"Isolated stable baseline at rows {best_start_idx} to {best_start_idx + effective_window}.")
    print(f"Averaged Gravity Vectors -> X: {ax_mean:.4f}g, Y: {ay_mean:.4f}g, Z: {az_mean:.4f}g")
    print("-" * 60)
    print(f"📐 ROLL MISALIGNMENT  (X-Axis): {roll_deg:+.4f}°")
    print(f"📐 PITCH MISALIGNMENT (Y-Axis): {pitch_deg:+.4f}°")
    print("-" * 60)
    print("🔄 INDIVIDUAL CORRECTION MATRIX ELEMENTS (Roll -> Pitch Order):")
    print(f"cor_matrix_xx = {R_correct[0, 0]: .6f}")
    print(f"cor_matrix_xy = {R_correct[0, 1]: .6f}")
    print(f"cor_matrix_xz = {R_correct[0, 2]: .6f}")
    print(f"cor_matrix_yx = {R_correct[1, 0]: .6f}")
    print(f"cor_matrix_yy = {R_correct[1, 1]: .6f}")
    print(f"cor_matrix_yz = {R_correct[1, 2]: .6f}")
    print(f"cor_matrix_zx = {R_correct[2, 0]: .6f}")
    print(f"cor_matrix_zy = {R_correct[2, 1]: .6f}")
    print(f"cor_matrix_zz = {R_correct[2, 2]: .6f}")
    print("=" * 60)
    
    return pitch_deg, roll_deg, R_correct

def apply_correction(raw_data, R_correct):
    """
    Applies the 3x3 correction matrix to an array of raw accelerometer data.
    Assumes raw_data shape is (N, 3).
    """
    return np.dot(raw_data, R_correct.T)

# --- Execution ---
if __name__ == "__main__":
    log_file = "accelerometer_single_log.txt" 
    try:
        pitch, roll, R_correct = calculate_pitch_roll_misalignment(log_file)
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{log_file}'. Please place it in the same directory.")