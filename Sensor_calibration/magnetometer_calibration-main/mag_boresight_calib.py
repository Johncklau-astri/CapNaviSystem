import numpy as np
import pandas as pd
import itertools

def vpo_alignment_deterministic(measurements_x, measurements_y):
    """
    Deterministic VPO alignment using ALL possible triplets.
    No randomness - same result every run.
    Inputs are expected to be in microteslas (uT).
    """
    
    def estimate_axis_from_all_triplets(measurements, axis_name):
        """Use ALL possible triplets to estimate rotation axis."""
        n_samples = len(measurements)
        
        if n_samples < 3:
            raise ValueError(f"Need at least 3 measurements for {axis_name} rotation")
        
        print(f"\n  Processing {axis_name} rotation: {n_samples} samples")
        
        # Generate ALL possible triplets (deterministic)
        all_indices = list(itertools.combinations(range(n_samples), 3))
        n_triplets = len(all_indices)
        print(f"  Using all {n_triplets} possible triplets")
        
        axes_list = []
        weights = []
        
        for idx in all_indices:
            B1, B2, B3 = measurements[idx[0]], measurements[idx[1]], measurements[idx[2]]
            
            # Eq. (12): differences
            B12 = B2 - B1
            B13 = B3 - B1
            
            # Eq. (13): cross product
            v = np.cross(B12, B13)
            norm_v = np.linalg.norm(v)
            
            if norm_v > 1e-6:
                axis = v / norm_v
                # Weight by cross product magnitude (larger = more reliable triplet)
                weight = norm_v
                axes_list.append(axis)
                weights.append(weight)
        
        axes_array = np.array(axes_list)
        weights_array = np.array(weights)
        
        print(f"  Valid triplets: {len(axes_array)}/{n_triplets}")
        
        # Weighted average using outer product
        sum_outer = np.zeros((3, 3))
        for axis, w in zip(axes_array, weights_array):
            sum_outer += w * np.outer(axis, axis)
        
        eigenvalues, eigenvectors = np.linalg.eigh(sum_outer)
        mean_axis = eigenvectors[:, -1]  # Largest eigenvalue
        
        # Ensure consistent sign
        signs = np.sign(np.dot(axes_array, mean_axis))
        if np.sum(signs) < 0:
            mean_axis = -mean_axis
        
        # Calculate weighted statistics
        angles = np.arccos(np.clip(np.abs(np.dot(axes_array, mean_axis)), -1, 1))
        weighted_std = np.sqrt(np.average((angles - np.average(angles, weights=weights_array))**2, 
                                          weights=weights_array)) * 180 / np.pi
        
        # Calculate confidence (higher = more consistent)
        consistency = np.average(np.abs(np.dot(axes_array, mean_axis)), weights=weights_array)
        
        return mean_axis, weighted_std, consistency, len(axes_array)
    
    print("\n" + "="*70)
    print("DETERMINISTIC VPO ALIGNMENT (All possible triplets)")
    print("="*70)
    
    # Analyze data quality first
    print("\n" + "="*60)
    print("Data Quality Analysis")
    print("="*60)
    
    for name, data in [("X Rotation", measurements_x), ("Y Rotation", measurements_y)]:
        mag = np.linalg.norm(data, axis=1)
        mag_var = np.std(mag) / np.mean(mag) * 100
        print(f"\n{name}:")
        print(f"  Samples: {len(data)}")
        print(f"  Mean |B|: {np.mean(mag):.2f} uT")
        print(f"  |B| variation: {mag_var:.2f}%")
        if mag_var > 5:
            print(f"  ⚠ High variation - VPO results may be unreliable")
    
    # Estimate axes using ALL triplets
    print("\n" + "="*60)
    print("Estimating Rotation Axes")
    print("="*60)
    
    print("\nEstimating i axis (X mounting axis)...")
    i_vec, i_std, i_cons, n_i = estimate_axis_from_all_triplets(measurements_x, "X")
    print(f"  i axis = [{i_vec[0]:.8f}, {i_vec[1]:.8f}, {i_vec[2]:.8f}]")
    print(f"  Weighted angular uncertainty: {i_std:.4f}°")
    print(f"  Consistency (1=perfect): {i_cons:.6f}")
    
    print("\nEstimating j axis (Y mounting axis)...")
    j_vec, j_std, j_cons, n_j = estimate_axis_from_all_triplets(measurements_y, "Y")
    print(f"  j axis = [{j_vec[0]:.8f}, {j_vec[1]:.8f}, {j_vec[2]:.8f}]")
    print(f"  Weighted angular uncertainty: {j_std:.4f}°")
    print(f"  Consistency (1=perfect): {j_cons:.6f}")
    
    # Check orthogonality
    dot_ij = np.dot(i_vec, j_vec)
    angle_ij = np.arccos(np.clip(np.abs(dot_ij), -1, 1)) * 180 / np.pi
    print(f"\nAngle between i and j: {angle_ij:.4f}° (should be 90°)")
    
    if angle_ij < 85 or angle_ij > 95:
        print(f"  ⚠ WARNING: Axes are not perpendicular!")
        print(f"  This suggests your X and Y rotations were not about orthogonal axes")
    
    # Construct orthogonal mounting frame using Gram-Schmidt
    i_vec = i_vec / np.linalg.norm(i_vec)
    j_vec = j_vec - np.dot(j_vec, i_vec) * i_vec
    j_vec = j_vec / np.linalg.norm(j_vec)
    k_vec = np.cross(i_vec, j_vec)
    
    # Ensure right-handed system (determinant = +1)
    R_mc = np.vstack([i_vec, j_vec, k_vec])
    if np.linalg.det(R_mc) < 0:
        k_vec = -k_vec
        R_mc = np.vstack([i_vec, j_vec, k_vec])
    
    return R_mc, i_vec, j_vec, k_vec, i_std, j_std, i_cons, j_cons


def compute_misalignment_angle_stable(R_mc):
    """
    Compute net misalignment angle and separate individual axis angles from rotation matrix.
    Returns the smallest net angle in degrees.
    """
    # 1. Total Scalar Misalignment
    trace = np.trace(R_mc)
    cos_phi = 0.5 * (trace - 1)
    cos_phi = np.clip(cos_phi, -1, 1)
    phi_rad = np.arccos(cos_phi)
    phi_deg = np.degrees(phi_rad)
    
    qw = 0.5 * np.sqrt(max(0, 1 + trace))
    if qw > 0:
        phi_quat = 2 * np.arccos(np.clip(qw, -1, 1))
        phi_quat_deg = np.degrees(phi_quat)
    else:
        phi_quat_deg = phi_deg
    
    phi_final = min(phi_deg, phi_quat_deg, 180 - phi_deg, 180 - phi_quat_deg)
    
    # 2. Extract Euler Angles (Roll, Pitch, Yaw) assuming Z-Y-X sequence
    # Dealing with standard matrix index: R_mc[row, col]
    if abs(R_mc[2, 0]) < 0.99999:  # Check for gimbal lock
        pitch = -np.arcsin(R_mc[2, 0])
        roll = np.arctan2(R_mc[2, 1], R_mc[2, 2])
        yaw = np.arctan2(R_mc[1, 0], R_mc[0, 0])
    else:
        # Gimbal lock fallback
        yaw = 0.0
        if R_mc[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R_mc[0, 1], R_mc[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R_mc[0, 1], -R_mc[0, 2])

    # 3. Projection Deviation Angles (Direct structural misalignment of individual axes)
    # Ideal axes are Identity matrix vectors: [1,0,0], [0,1,0], [0,0,1]
    # We find the angle between calculated frame vectors and their nearest ideal base counterparts
    dev_x = np.degrees(np.arccos(np.clip(abs(R_mc[0, 0]), -1, 1)))
    dev_y = np.degrees(np.arccos(np.clip(abs(R_mc[1, 1]), -1, 1)))
    dev_z = np.degrees(np.arccos(np.clip(abs(R_mc[2, 2]), -1, 1)))

    print("\n" + "="*60)
    print("DETAILED AXIS MISALIGNMENT BREAKDOWN")
    print("="*60)
    print(f"Euler Angles (Z-Y-X Sequence):")
    print(f"  Rotation around X-axis (Roll) : {np.degrees(roll):.4f}°")
    print(f"  Rotation around Y-axis (Pitch): {np.degrees(pitch):.4f}°")
    print(f"  Rotation around Z-axis (Yaw)  : {np.degrees(yaw):.4f}°")
    print(f"\nAbsolute Component Axis Deviations from Ideal Baseline:")
    print(f"  X-Axis (i) Tilt Error: {dev_x:.4f}°")
    print(f"  Y-Axis (j) Tilt Error: {dev_y:.4f}°")
    print(f"  Z-Axis (k) Tilt Error: {dev_z:.4f}°")
    
    return phi_final


def verify_constant_projection(measurements, axis_vec, axis_name):
    """Verify that projection onto rotation axis is constant."""
    projection = measurements @ axis_vec
    mean_proj = np.mean(projection)
    std_proj = np.std(projection)
    
    print(f"\n{axis_name} axis projection:")
    print(f"  Mean: {mean_proj:.4f} uT")
    print(f"  Std dev: {std_proj:.4f} uT")
    
    if abs(mean_proj) > 1e-6:
        print(f"  Relative std: {std_proj/abs(mean_proj)*100:.2f}%")
        quality = max(0, 100 * (1 - std_proj/abs(mean_proj)))
    else:
        print(f"  Relative std: N/A")
        quality = max(0, 100 * (1 - std_proj))
        
    print(f"  Quality: {quality:.1f}%")
    return projection, quality


def main():
    # Load data
    print("Loading magnetometer data...")
    df_x = pd.read_csv('bore_x_log.txt', header=None, names=['Bx', 'By', 'Bz'])
    df_y = pd.read_csv('bore_y_log.txt', header=None, names=['Bx', 'By', 'Bz'])
    
    measurements_x = df_x[['Bx', 'By', 'Bz']].values
    measurements_y = df_y[['Bx', 'By', 'Bz']].values
    
    print(f"\nData loaded:")
    print(f"  X rotation: {len(measurements_x)} samples")
    print(f"  Y rotation: {len(measurements_y)} samples")
    
    # Perform deterministic VPO alignment
    R_mc, i_vec, j_vec, k_vec, i_std, j_std, i_cons, j_cons = vpo_alignment_deterministic(
        measurements_x, measurements_y
    )
    
    # Verify constant projection
    print("\n" + "="*60)
    print("Verification: Constant Projection onto Rotation Axes")
    print("="*60)
    
    proj_x, qual_x = verify_constant_projection(measurements_x, i_vec, "X rotation onto i")
    proj_y, qual_y = verify_constant_projection(measurements_y, j_vec, "Y rotation onto j")
    
    # Compute misalignment angle
    print("\n" + "="*60)
    print("MISALIGNMENT RESULT")
    print("="*60)
    
    misalignment = compute_misalignment_angle_stable(R_mc)
    
    print(f"\n★ NET TOTAL MISALIGNMENT ANGLE = {misalignment:.6f} degrees ★")
    print(f"\nEstimated uncertainties:")
    print(f"  i axis uncertainty: ±{i_std:.4f}°")
    print(f"  j axis uncertainty: ±{j_std:.4f}°")
    print(f"  Combined: ±{(i_std + j_std)/2:.4f}°")
    
    print(f"\nConfidence metrics:")
    print(f"  i axis consistency: {i_cons*100:.2f}%")
    print(f"  j axis consistency: {j_cons*100:.2f}%")
    print(f"  X projection quality: {qual_x:.1f}%")
    print(f"  Y projection quality: {qual_y:.1f}%")
    
    # Overall confidence score
    overall_confidence = (i_cons + j_cons + qual_x/100 + qual_y/100) / 4 * 100
    print(f"\nOverall confidence: {overall_confidence:.1f}%")
    
    if overall_confidence < 70:
        print("\n⚠ WARNING: Low confidence in result")
    
    print("\n" + "="*60)
    print("ROTATION MATRIX R^{mc}")
    print("="*60)
    print(R_mc)
    
    print("\n" + "="*60)
    print("AXIS UNIT VECTORS (in calibration frame coordinates)")
    print("="*60)
    print(f"  i (X mounting) = [{i_vec[0]:.10f}, {i_vec[1]:.10f}, {i_vec[2]:.10f}]")
    print(f"  j (Y mounting) = [{j_vec[0]:.10f}, {j_vec[1]:.10f}, {j_vec[2]:.10f}]")
    print(f"  k (Z mounting) = [{k_vec[0]:.10f}, {k_vec[1]:.10f}, {k_vec[2]:.10f}]")
    
    return R_mc, misalignment

if __name__ == "__main__":
    R_mc, misalignment = main()