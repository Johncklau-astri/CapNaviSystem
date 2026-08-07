#!/usr/bin/env python3
"""Fit an accelerometer ellipsoid and correct measurements to a gravity sphere.

The calibration equation is ``a_calibrated = A @ (a_raw - b)``. Capture samples
with the sensor stationary in many orientations before running this module.
"""

import argparse
import json
from pathlib import Path

import numpy as np


class AccelerometerEllipsoidCalibrator:
    """Estimate accelerometer bias and scale/misalignment from static samples."""

    def __init__(self, gravity=1.0):
        if gravity <= 0:
            raise ValueError("Gravity magnitude must be positive.")
        self.gravity = float(gravity)
        self.bias = np.zeros(3)
        self.matrix = np.eye(3)

    @staticmethod
    def load_data(file_path):
        """Load three-axis samples from comma- or whitespace-delimited text."""
        samples = []
        with open(file_path, "r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.replace(",", " ").split()
                if len(fields) != 3:
                    raise ValueError(
                        f"Expected three values on line {line_number}, found {len(fields)}."
                    )
                try:
                    samples.append([float(field) for field in fields])
                except ValueError as error:
                    raise ValueError(f"Invalid numeric data on line {line_number}: {line}") from error

        data = np.asarray(samples, dtype=float)
        if data.shape[0] < 10:
            raise ValueError("At least 10 stationary samples spanning many orientations are required.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Input contains non-finite values.")
        return data

    @staticmethod
    def _fit_ellipsoid(samples):
        """Return the positive-definite quadratic form M, vector n, and constant d."""
        x_values, y_values, z_values = samples.T
        design = np.array([
            x_values**2, y_values**2, z_values**2,
            2 * y_values * z_values, 2 * x_values * z_values, 2 * x_values * y_values,
            2 * x_values, 2 * y_values, 2 * z_values, np.ones_like(x_values),
        ])
        scatter = design @ design.T
        scatter_11 = scatter[:6, :6]
        scatter_12 = scatter[:6, 6:]
        scatter_21 = scatter[6:, :6]
        scatter_22 = scatter[6:, 6:]
        constraint = np.array([
            [-1, 1, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 0],
            [1, 1, -1, 0, 0, 0],
            [0, 0, 0, -4, 0, 0],
            [0, 0, 0, 0, -4, 0],
            [0, 0, 0, 0, 0, -4],
        ], dtype=float)

        try:
            reduced_scatter = scatter_11 - scatter_12 @ np.linalg.solve(scatter_22, scatter_21)
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(constraint, reduced_scatter))
        except np.linalg.LinAlgError as error:
            raise ValueError("Input does not span enough independent orientations for an ellipsoid fit.") from error

        candidates = []
        for index, eigenvalue in enumerate(eigenvalues):
            eigenvector = np.real_if_close(eigenvectors[:, index], tol=1_000)
            if not np.isrealobj(eigenvector):
                continue
            first_part = np.asarray(eigenvector, dtype=float)
            second_part = -np.linalg.solve(scatter_22, scatter_21 @ first_part)
            matrix = np.array([
                [first_part[0], first_part[5], first_part[4]],
                [first_part[5], first_part[1], first_part[3]],
                [first_part[4], first_part[3], first_part[2]],
            ])
            vector = second_part[:3]
            constant = float(second_part[3])
            for sign in (1.0, -1.0):
                signed_matrix = sign * matrix
                if np.all(np.linalg.eigvalsh(signed_matrix) > 0):
                    candidates.append((abs(eigenvalue), signed_matrix, sign * vector, sign * constant))

        if not candidates:
            raise ValueError("No physical ellipsoid could be fitted; capture more diverse static orientations.")
        _, matrix, vector, constant = max(candidates, key=lambda candidate: candidate[0])
        return matrix, vector, constant

    def calibrate(self, samples):
        """Fit the ellipsoid and populate ``bias`` and ``matrix``."""
        quadratic, linear, constant = self._fit_ellipsoid(samples)
        try:
            self.bias = -np.linalg.solve(quadratic, linear)
        except np.linalg.LinAlgError as error:
            raise ValueError("The fitted ellipsoid has a singular center.") from error

        scale = self.bias @ quadratic @ self.bias - constant
        if scale <= 0:
            raise ValueError("The fitted ellipsoid is not physically valid.")
        eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
        if np.any(eigenvalues <= 0):
            raise ValueError("The fitted ellipsoid is not positive definite.")
        self.matrix = self.gravity * (eigenvectors * np.sqrt(eigenvalues / scale)) @ eigenvectors.T
        return self.bias, self.matrix

    def apply(self, samples):
        """Apply the fitted calibration to an ``(N, 3)`` array."""
        return (self.matrix @ (samples - self.bias).T).T

    def save(self, file_path):
        """Save calibration parameters and their application equation as JSON."""
        payload = {
            "gravity_magnitude": self.gravity,
            "bias": self.bias.tolist(),
            "correction_matrix": self.matrix.tolist(),
            "equation": "a_calibrated = correction_matrix @ (a_raw - bias)",
        }
        with open(file_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)


def print_report(calibrator, raw_samples, calibrated_samples):
    """Print reusable parameters and before/after norm statistics."""
    raw_norms = np.linalg.norm(raw_samples, axis=1)
    calibrated_norms = np.linalg.norm(calibrated_samples, axis=1)
    print("Accelerometer ellipsoid calibration")
    print(f"Samples: {len(raw_samples)}; target gravity: {calibrator.gravity:.6f}")
    print(f"Raw norm:        mean={raw_norms.mean():.6f}, std={raw_norms.std():.6f}")
    print(f"Calibrated norm: mean={calibrated_norms.mean():.6f}, std={calibrated_norms.std():.6f}")
    print("\nbias = np.array([")
    for value in calibrator.bias:
        print(f"    {value:.9f},")
    print("])\n")
    print("A = np.array([")
    for row in calibrator.matrix:
        print("    [" + ", ".join(f"{value:.9f}" for value in row) + "],")
    print("])")
    print("\nPaste this into DataProcessor.calibrate_accelerometer(), replacing its bias and calibration matrix blocks:")
    print(f"        bias_x = {calibrator.bias[0]:.9f}")
    print(f"        bias_y = {calibrator.bias[1]:.9f}")
    print(f"        bias_z = {calibrator.bias[2]:.9f}\n")
    print(
        "        cal_matrix_xx = {0:.9f}; cal_matrix_xy = {1:.9f}; cal_matrix_xz = {2:.9f}".format(
            *calibrator.matrix[0]
        )
    )
    print(
        "        cal_matrix_yx = {0:.9f}; cal_matrix_yy = {1:.9f}; cal_matrix_yz = {2:.9f}".format(
            *calibrator.matrix[1]
        )
    )
    print(
        "        cal_matrix_zx = {0:.9f}; cal_matrix_zy = {1:.9f}; cal_matrix_zz = {2:.9f}".format(
            *calibrator.matrix[2]
        )
    )


def plot_samples(raw_samples, calibrated_samples, gravity):
    """Show 3D and planar raw-versus-calibrated scatter plots."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 9))
    axes = [figure.add_subplot(221, projection="3d"), figure.add_subplot(222),
            figure.add_subplot(223), figure.add_subplot(224)]
    axes[0].scatter(*raw_samples.T, color="tab:red", s=10, label="Raw")
    axes[0].scatter(*calibrated_samples.T, color="tab:blue", s=10, label="Calibrated")
    axes[0].set(xlabel="X", ylabel="Y", zlabel="Z", title="3D accelerometer samples")
    axes[0].legend()
    for axis, first, second, title in zip(axes[1:], (0, 0, 1), (1, 2, 2), ("XY", "XZ", "YZ")):
        axis.scatter(raw_samples[:, first], raw_samples[:, second], color="tab:red", s=10, label="Raw")
        axis.scatter(calibrated_samples[:, first], calibrated_samples[:, second], color="tab:blue", s=10, label="Calibrated")
        axis.set(xlabel="XYZ"[first], ylabel="XYZ"[second], title=f"{title} projection")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.2 * gravity, 1.2 * gravity)
        axis.set_ylim(-1.2 * gravity, 1.2 * gravity)
        axis.grid()
        axis.legend()
    figure.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, default=Path("calibration_log.txt"))
    parser.add_argument("--gravity", type=float, default=1.0, help="Expected gravity magnitude; default: 1.0")
    parser.add_argument("--json", type=Path, help="Write calibration parameters to this JSON file")
    parser.add_argument("--calibrated-output", type=Path, help="Write calibrated samples as comma-separated text")
    parser.add_argument("--plot", action="store_true", help="Display raw and calibrated plots")
    args = parser.parse_args()

    calibrator = AccelerometerEllipsoidCalibrator(args.gravity)
    raw_samples = calibrator.load_data(args.file)
    calibrator.calibrate(raw_samples)
    calibrated_samples = calibrator.apply(raw_samples)
    print_report(calibrator, raw_samples, calibrated_samples)
    if args.json:
        calibrator.save(args.json)
        print(f"Saved calibration to {args.json}")
    if args.calibrated_output:
        np.savetxt(args.calibrated_output, calibrated_samples, delimiter=",", fmt="%.9f")
        print(f"Saved calibrated samples to {args.calibrated_output}")
    if args.plot:
        plot_samples(raw_samples, calibrated_samples, calibrator.gravity)


if __name__ == "__main__":
    main()
