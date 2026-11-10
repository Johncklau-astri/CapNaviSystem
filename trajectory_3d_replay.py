"""Replay exported capsule poses and their corresponding photographs in 3D.

Each pose is read from a ``pos.json`` file.  ``position_world`` is interpreted
in metres and ``camera_world`` is interpreted as a quaternion in ``[w, x, y,
z]`` order.  The image named by ``image_file`` is displayed alongside the 3D
replay at the same frame.

Examples:
    python trajectory_3d_replay.py
    python trajectory_3d_replay.py TESTSEQUENCE/00897
    python trajectory_3d_replay.py --fps 10 --no-loop
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseFrame:
    position: np.ndarray
    timestamp: float | None
    motor_position: np.ndarray | None
    camera_quaternion_wxyz: np.ndarray
    image_path: Path
    capture_id: str


def _as_vector(values, length, field, path):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{path}: {field} must contain {length} finite numbers")
    return vector


def discover_manifests(data_path: Path) -> list[Path]:
    """Find manifests in one sequence folder or recursively below a directory."""
    if data_path.is_file() and data_path.name == "pos.json":
        return [data_path]
    if data_path.is_dir() and (data_path / "pos.json").is_file():
        return [data_path / "pos.json"]
    if data_path.is_dir():
        return sorted(data_path.rglob("pos.json"))
    raise FileNotFoundError(f"Pose data path does not exist: {data_path}")


def load_frames(data_path: Path) -> list[PoseFrame]:
    frames = []
    for manifest_path in discover_manifests(data_path):
        try:
            with manifest_path.open("r", encoding="utf-8") as input_file:
                payload = json.load(input_file)
            position = _as_vector(payload["position_world"], 3, "position_world", manifest_path)
            timestamp = payload.get("timestamp", payload.get("timestamp_computer"))
            if timestamp is not None:
                timestamp = float(timestamp)
                if not np.isfinite(timestamp):
                    raise ValueError("timestamp must be finite")
            motor_position = payload.get("motor_position_world")
            if motor_position is not None:
                motor_position = _as_vector(
                    motor_position, 3, "motor_position_world", manifest_path
                )
            quaternion = _as_vector(payload["camera_world"], 4, "camera_world", manifest_path)
            if np.linalg.norm(quaternion) < 1e-12:
                raise ValueError("camera_world quaternion has zero length")
            image_path = manifest_path.parent / payload["image_file"]
            if not image_path.is_file():
                raise FileNotFoundError(f"image_file does not exist: {image_path}")
            frames.append(PoseFrame(
                position=position,
                timestamp=timestamp,
                motor_position=motor_position,
                camera_quaternion_wxyz=quaternion,
                image_path=image_path,
                capture_id=str(payload.get("capture_id", manifest_path.parent.name)),
            ))
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            print(f"Skipping {manifest_path}: {error}", file=sys.stderr)

    if not frames:
        raise ValueError(f"No usable pose manifests found below {data_path}")
    return frames


def _interpolate_over_speed(values, times, max_speed_mps):
    """Replace samples in over-speed segments with time-based interpolation."""
    elapsed = np.diff(times)
    valid_segments = elapsed > 0
    segment_distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
    segment_speeds = np.zeros(len(elapsed), dtype=float)
    segment_speeds[valid_segments] = (
        segment_distances[valid_segments] / elapsed[valid_segments]
    )
    invalid = np.zeros(len(values), dtype=bool)
    over_speed = segment_speeds > max_speed_mps
    invalid[:-1] |= over_speed
    invalid[1:] |= over_speed
    cleaned = values.copy()

    # Do not interpolate across duplicate or reset timestamps.
    run_starts = np.r_[0, np.flatnonzero(~valid_segments) + 1]
    run_ends = np.r_[np.flatnonzero(~valid_segments), len(values) - 1]
    for run_start, run_end in zip(run_starts, run_ends):
        run_indices = np.arange(run_start, run_end + 1)
        valid_indices = run_indices[~invalid[run_indices]]
        if len(valid_indices) >= 2:
            for axis in range(values.shape[1]):
                cleaned[run_indices, axis] = np.interp(
                    times[run_indices],
                    times[valid_indices],
                    values[valid_indices, axis],
                )
        elif len(valid_indices) == 1:
            cleaned[run_indices] = values[valid_indices[0]]
    return cleaned


def smooth_positions(
    positions: np.ndarray,
    timestamps: np.ndarray | None = None,
    window_length: int = 5,
    outlier_sigma: float = 4.0,
    max_speed_mps: float | None = 0.02,
) -> np.ndarray:
    """Reject over-speed samples, interpolate them, and smooth position jitter.

    Samples belonging to a segment faster than ``max_speed_mps`` are rejected.
    Rejected runs are linearly interpolated between the nearest valid samples.
    A point is treated as an outlier when its distance from the coordinate-wise
    median of its three-sample neighborhood is unusually large compared with
    the robust MAD of all such distances.  Outliers are reconstructed by local
    interpolation, then a short quadratic Savitzky-Golay filter removes
    ordinary noise.
    """
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("positions must have shape (number_of_frames, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("positions must contain only finite numbers")
    if outlier_sigma <= 0:
        raise ValueError("outlier_sigma must be greater than zero")
    if max_speed_mps is not None and max_speed_mps < 0:
        raise ValueError("max_speed_mps must be non-negative or None")
    if len(values) < 3:
        return values.copy()

    cleaned = values.copy()
    if timestamps is not None and max_speed_mps:
        times = np.asarray(timestamps, dtype=float)
        if times.shape != (len(values),) or not np.all(np.isfinite(times)):
            raise ValueError("timestamps must match positions and contain finite values")

        cleaned = _interpolate_over_speed(cleaned, times, max_speed_mps)

    neighborhood_median = np.median(
        np.stack((cleaned[:-2], cleaned[1:-1], cleaned[2:])),
        axis=0,
    )
    prediction_error = np.linalg.norm(
        cleaned[1:-1] - neighborhood_median,
        axis=1,
    )
    median_error = float(np.median(prediction_error))
    mad = float(np.median(np.abs(prediction_error - median_error)))
    robust_scale = 1.4826 * mad
    step_scale = float(np.median(np.linalg.norm(np.diff(cleaned, axis=0), axis=1)))
    threshold = median_error + outlier_sigma * max(robust_scale, step_scale * 0.5, 1e-12)
    outlier_indices = np.flatnonzero(prediction_error > threshold) + 1
    cleaned[outlier_indices] = (cleaned[outlier_indices - 1] + cleaned[outlier_indices + 1]) * 0.5

    if window_length < 3:
        if timestamps is not None and max_speed_mps:
            cleaned = _interpolate_over_speed(cleaned, times, max_speed_mps)
        return cleaned
    window_length = min(window_length, len(cleaned))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        if timestamps is not None and max_speed_mps:
            cleaned = _interpolate_over_speed(cleaned, times, max_speed_mps)
        return cleaned
    cleaned = savgol_filter(cleaned, window_length, polyorder=2, axis=0, mode="interp")
    if timestamps is not None and max_speed_mps:
        cleaned = _interpolate_over_speed(cleaned, times, max_speed_mps)
    return cleaned


def smooth_frames(
    frames: list[PoseFrame],
    window_length: int = 5,
    outlier_sigma: float = 4.0,
    max_speed_mps: float | None = 0.03,
    fallback_fps: float | None = None,
) -> list[PoseFrame]:
    if len(frames) < 3:
        return frames.copy()
    frame_timestamps = [frame.timestamp for frame in frames]
    if all(timestamp is not None for timestamp in frame_timestamps):
        timestamps = np.asarray(frame_timestamps, dtype=float)
        if fallback_fps is not None and fallback_fps > 0:
            fallback_interval = 1.0 / fallback_fps
            timestamps = timestamps.copy()
            for index in range(1, len(timestamps)):
                if timestamps[index] <= timestamps[index - 1]:
                    timestamps[index] = timestamps[index - 1] + fallback_interval
    elif fallback_fps is not None and fallback_fps > 0:
        timestamps = np.arange(len(frames), dtype=float) / fallback_fps
    else:
        timestamps = None
    positions = smooth_positions(
        np.asarray([frame.position for frame in frames]),
        timestamps=timestamps,
        window_length=window_length,
        outlier_sigma=outlier_sigma,
        max_speed_mps=max_speed_mps,
    )
    return [
        PoseFrame(
            position=position,
            timestamp=frame.timestamp,
            motor_position=frame.motor_position,
            camera_quaternion_wxyz=frame.camera_quaternion_wxyz,
            image_path=frame.image_path,
            capture_id=frame.capture_id,
        )
        for frame, position in zip(frames, positions)
    ]


def make_axes(position, quaternion_wxyz, axis_length):
    """Return line vertices for the camera's oriented local XYZ axes."""
    rotation = Rotation.from_quat([
        quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]
    ])
    axes = rotation.as_matrix() @ (np.eye(3) * axis_length)
    return np.asarray([
        position, position + axes[:, 0],
        position, position + axes[:, 1],
        position, position + axes[:, 2],
    ], dtype=float)


def make_lens_direction(position, quaternion_wxyz, axis_length):
    """Return line vertices for the capsule's local -Y lens direction."""
    rotation = Rotation.from_quat([
        quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]
    ])
    direction = -rotation.as_matrix()[:, 1] * axis_length
    return np.asarray([position, position + direction], dtype=float)


def make_capsule_body(position, quaternion_wxyz, body_length):
    """Return endpoints for a simple line-shaped capsule body."""
    rotation = Rotation.from_quat([
        quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]
    ])
    local_axis = rotation.as_matrix()[:, 0] * (body_length * 0.5)
    return np.asarray([position - local_axis, position + local_axis], dtype=float)


class TrajectoryReplay(QtWidgets.QMainWindow):
    def __init__(self, frames: list[PoseFrame], fps: float, loop: bool):
        super().__init__()
        self.frames = frames
        self.frame_index = 0
        self.loop = loop
        self.paused = False
        self.replay_fps = fps
        self.setWindowTitle("Capsule World-Trajectory Replay")
        self.resize(1400, 800)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        self.setCentralWidget(root)

        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = self._camera_distance()
        self.view.opts["elevation"] = 25
        self.view.opts["azimuth"] = 45
        layout.addWidget(self.view, stretch=3)

        photo_panel = QtWidgets.QWidget()
        photo_layout = QtWidgets.QVBoxLayout(photo_panel)
        self.photo_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.photo_label.setMinimumSize(360, 270)
        self.photo_label.setStyleSheet("background: #101318; color: #d7dce5;")
        self.photo_label.setScaledContents(False)
        self.info_label = QtWidgets.QLabel()
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        photo_layout.addWidget(self.photo_label, stretch=1)
        photo_layout.addWidget(self.info_label)
        layout.addWidget(photo_panel, stretch=2)

        self._add_scene_items()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.timer.start(max(1, round(1000.0 / fps)))
        self.show_frame()

    def _camera_distance(self):
        positions = np.asarray([frame.position for frame in self.frames])
        span = np.ptp(positions, axis=0).max()
        return max(0.25, float(span * 2.5))

    def _add_scene_items(self):
        positions = np.asarray([frame.position for frame in self.frames])
        axis_size = max(0.05, float(np.ptp(positions, axis=0).max()))
        axes = gl.GLAxisItem()
        axes.setSize(x=axis_size, y=axis_size, z=axis_size)
        self.view.addItem(axes)
        label_font = QtGui.QFont()
        label_font.setPointSize(16)
        label_font.setBold(True)
        for axis_name, axis_index, color in (
            ("X axis", 0, (1.0, 0.25, 0.25, 1.0)),
            ("Y axis", 1, (0.25, 1.0, 0.25, 1.0)),
            ("Z axis", 2, (0.35, 0.55, 1.0, 1.0)),
        ):
            label_position = np.zeros(3)
            label_position[axis_index] = axis_size * 1.2
            self.view.addItem(gl.GLTextItem(
                pos=label_position,
                text=axis_name,
                color=color,
                font=label_font,
            ))
        self.lens_direction = gl.GLLinePlotItem(
            color=(1.0, 0.85, 0.1, 1.0), width=7, antialias=True, mode="lines"
        )
        self.view.addItem(self.lens_direction)
        camera_label_font = QtGui.QFont(label_font)
        camera_label_font.setPointSize(14)
        camera_label_font.setBold(True)
        self.lens_direction_label = gl.GLTextItem(
            text="Lens direction (-Y)",
            color=(1.0, 0.85, 0.1, 1.0),
            font=camera_label_font,
        )
        self.view.addItem(self.lens_direction_label)
        self.view.addItem(gl.GLGridItem())
        self.trail = gl.GLLinePlotItem(pos=positions, color=(0.1, 0.8, 1.0, 0.9), width=2, antialias=True)
        self.view.addItem(self.trail)
        self.path_points = gl.GLScatterPlotItem(pos=positions, size=5, color=(0.9, 0.7, 0.2, 0.8))
        self.view.addItem(self.path_points)
        self.marker = gl.GLScatterPlotItem(pos=positions[:1], size=14, color=(1.0, 0.2, 0.15, 1.0))
        self.view.addItem(self.marker)
        self.body = gl.GLLinePlotItem(color=(1.0, 0.35, 0.1, 1.0), width=8, antialias=True, mode="lines")
        self.view.addItem(self.body)
        self.axes = gl.GLLinePlotItem(color=(1.0, 1.0, 1.0, 1.0), width=3, antialias=True, mode="lines")
        self.view.addItem(self.axes)

    def advance(self):
        if self.paused:
            return
        if self.frame_index == len(self.frames) - 1:
            if not self.loop:
                self.timer.stop()
                return
            self.frame_index = 0
        else:
            self.frame_index += 1
        self.show_frame()

    def show_frame(self):
        frame = self.frames[self.frame_index]
        linear_speed = None
        angular_speed = None
        speed_is_estimated = False
        previous_frame = None
        elapsed = None
        if frame.timestamp is not None:
            for candidate in reversed(self.frames[:self.frame_index]):
                if (
                    candidate.timestamp is not None
                    and candidate.timestamp < frame.timestamp
                ):
                    previous_frame = candidate
                    break
        if previous_frame is None and self.frame_index > 0:
            previous_frame = self.frames[self.frame_index - 1]
            elapsed = 1.0 / self.replay_fps
            speed_is_estimated = True
        elif previous_frame is not None:
            elapsed = frame.timestamp - previous_frame.timestamp

        if previous_frame is not None and elapsed is not None and elapsed > 0:
            linear_speed = np.linalg.norm(frame.position - previous_frame.position) / elapsed
            previous_rotation = Rotation.from_quat([
                previous_frame.camera_quaternion_wxyz[1],
                previous_frame.camera_quaternion_wxyz[2],
                previous_frame.camera_quaternion_wxyz[3],
                previous_frame.camera_quaternion_wxyz[0],
            ])
            current_rotation = Rotation.from_quat([
                frame.camera_quaternion_wxyz[1],
                frame.camera_quaternion_wxyz[2],
                frame.camera_quaternion_wxyz[3],
                frame.camera_quaternion_wxyz[0],
            ])
            angular_speed = np.degrees(
                (previous_rotation.inv() * current_rotation).magnitude()
            ) / elapsed
        self.marker.setData(pos=np.asarray([frame.position]))
        axis_length = self._axis_length()
        self.body.setData(pos=make_capsule_body(frame.position, frame.camera_quaternion_wxyz, axis_length * 1.8))
        self.axes.setData(pos=make_axes(frame.position, frame.camera_quaternion_wxyz, axis_length))
        lens_direction = make_lens_direction(
            frame.position, frame.camera_quaternion_wxyz, axis_length * 1.35
        )
        self.lens_direction.setData(pos=lens_direction)
        self.lens_direction_label.setData(pos=lens_direction[1])
        pixmap = QtGui.QPixmap(str(frame.image_path))
        if pixmap.isNull():
            self.photo_label.setText(f"Could not load\n{frame.image_path.name}")
        else:
            self.photo_label.setPixmap(pixmap.scaled(
                self.photo_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            ))
        position = frame.position
        quaternion = frame.camera_quaternion_wxyz / np.linalg.norm(frame.camera_quaternion_wxyz)
        motor_position_text = "unavailable"
        if frame.motor_position is not None:
            motor_position_text = (
                f"[{frame.motor_position[0]:.4f}, {frame.motor_position[1]:.4f}, "
                f"{frame.motor_position[2]:.4f}]"
            )
        speed_suffix = " (replay estimate)" if speed_is_estimated else ""
        linear_speed_text = (
            "unavailable"
            if linear_speed is None
            else f"{linear_speed:.4f} m/s{speed_suffix}"
        )
        angular_speed_text = (
            "unavailable"
            if angular_speed is None
            else f"{angular_speed:.2f} deg/s{speed_suffix}"
        )
        self.info_label.setText(
            f"Frame {self.frame_index + 1}/{len(self.frames)}\n"
            f"{frame.capture_id}\n"
            f"motor_position_world (m): {motor_position_text}\n"
            f"position_world (m): [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]\n"
            f"linear speed: {linear_speed_text}\n"
            f"angular speed: {angular_speed_text}\n"
            f"camera_world [w, x, y, z]: [{', '.join(f'{value:.4f}' for value in quaternion)}]"
        )

    def _axis_length(self):
        positions = np.asarray([frame.position for frame in self.frames])
        span = np.ptp(positions, axis=0).max()
        return max(0.01, float(span * 0.12))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.paused = not self.paused
        elif event.key() == QtCore.Qt.Key_Right:
            self.frame_index = min(self.frame_index + 1, len(self.frames) - 1)
            self.show_frame()
        elif event.key() == QtCore.Qt.Key_Left:
            self.frame_index = max(self.frame_index - 1, 0)
            self.show_frame()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "frames"):
            self.show_frame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=Path(r"R:\Release_Internal_V1A\export"),
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Odd Savitzky-Golay window for position smoothing (0 disables it)",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=4.0,
        help="Robust-MAD threshold for isolated position jumps",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=0.03,
        help="Reject trajectory samples exceeding this speed in metres/second (0 disables)",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.smooth_window != 0 and args.smooth_window < 3:
        parser.error("--smooth-window must be zero or at least three")
    if args.outlier_sigma <= 0:
        parser.error("--outlier-sigma must be greater than zero")
    if args.max_speed < 0:
        parser.error("--max-speed must be non-negative")
    frames = load_frames(args.data_path)
    if args.smooth_window:
        frames = smooth_frames(
            frames,
            args.smooth_window,
            args.outlier_sigma,
            max_speed_mps=args.max_speed or None,
            fallback_fps=args.fps,
        )
    app = QtWidgets.QApplication(sys.argv)
    window = TrajectoryReplay(frames, args.fps, loop=not args.no_loop)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()