"""Interactive synthetic data source for capsule_calibration.py boresight testing."""

import json
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


DATA_DIRECTORY = r"R:\Release_Internal_V1A\output"
INTRINSIC_FILE = "capsule_intrinsics.npz"
GROUND_TRUTH_FILE = "ground_truth_offset.npz"
TCP_HOST = "localhost"
TCP_PORT = 12347

IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 720
FOCAL_LENGTH_PX = 500.0
BOARD_DISTANCE_M = 0.15
BOARD_COLUMNS = 7
BOARD_ROWS = 9
SQUARE_SIZE_M = 0.026
MARKER_SIZE_M = 0.01976
RENDER_SCALE = 1_000

# Camera-to-PCB rotation that capsule_calibration.py is expected to recover.
GROUND_TRUTH_EULER_DEG = (2, -3, 4)

# Physical capsule-shell orientations: roll, pitch, yaw, in degrees.
TRAJECTORY_EULER_DEG = [
    (0.0, 0.0, 0.0),
    (12.0, -8.0, 6.0),
    (-14.0, 7.0, -10.0),
    (18.0, 10.0, 14.0),
    (-20.0, -11.0, 9.0),
    (8.0, 17.0, -15.0),
    (-9.0, -16.0, 18.0),
    (22.0, -5.0, -17.0),
    (-17.0, 14.0, 4.0),
    (5.0, -20.0, 20.0),
    (15.0, 18.0, -7.0),
    (-22.0, -14.0, -13.0),
]


class ImuTelemetryServer:
    """Streams the latest capsule orientation to one or more TCP clients."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server = None
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()

    def set_rotation(self, rotation_matrix):
        quaternion_xyzw = R.from_matrix(rotation_matrix).as_quat()
        with self._lock:
            self._quaternion = np.array(
                [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]]
            )

    def stop(self):
        self._stop_event.set()
        if self._server is not None:
            self._server.close()
        self._thread.join(timeout=1.0)

    def _payload(self):
        with self._lock:
            qw, qx, qy, qz = self._quaternion
        message = json.dumps({"qw": float(qw), "qx": float(qx), "qy": float(qy), "qz": float(qz)})
        encoded = message.encode("utf-8")
        return struct.pack(">I", len(encoded)) + encoded

    def _serve_client(self, client):
        client.settimeout(1.0)
        try:
            while not self._stop_event.is_set():
                client.sendall(self._payload())
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass
        finally:
            client.close()

    def _serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server = server
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen()
        server.settimeout(0.5)
        print(f"IMU telemetry server listening on {self.host}:{self.port}")

        try:
            while not self._stop_event.is_set():
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                print(f"IMU client connected from {address[0]}:{address[1]}")
                threading.Thread(target=self._serve_client, args=(client,), daemon=True).start()
        except OSError:
            if not self._stop_event.is_set():
                raise
        finally:
            server.close()


def camera_matrix():
    return np.array(
        [[FOCAL_LENGTH_PX, 0.0, 540.0], [0.0, FOCAL_LENGTH_PX, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_board_image():
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLUMNS, BOARD_ROWS), SQUARE_SIZE_M, MARKER_SIZE_M, aruco_dictionary
    )
    image_size = (BOARD_COLUMNS * RENDER_SCALE, BOARD_ROWS * RENDER_SCALE)
    return board.generateImage(image_size, marginSize=0, borderBits=1)


def render_charuco_view(board_image, board_to_camera, intrinsic_matrix, distortion_coefficients):
    """Render the board from a fixed camera position as its orientation changes."""
    board_center = np.array(
        [BOARD_COLUMNS * SQUARE_SIZE_M / 2.0, BOARD_ROWS * SQUARE_SIZE_M / 2.0, 0.0]
    )
    camera_position_in_board = board_center + np.array([0.0, 0.0, -BOARD_DISTANCE_M])
    translation = -board_to_camera @ camera_position_in_board
    rvec = R.from_matrix(board_to_camera).as_rotvec().reshape(3, 1)
    object_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [BOARD_COLUMNS * SQUARE_SIZE_M, 0.0, 0.0],
            [BOARD_COLUMNS * SQUARE_SIZE_M, BOARD_ROWS * SQUARE_SIZE_M, 0.0],
            [0.0, BOARD_ROWS * SQUARE_SIZE_M, 0.0],
        ],
        dtype=np.float32,
    )
    projected_corners, _ = cv2.projectPoints(
        object_corners, rvec, translation.reshape(3, 1), intrinsic_matrix, distortion_coefficients
    )
    source_corners = np.array(
        [[0, 0], [board_image.shape[1] - 1, 0], [board_image.shape[1] - 1, board_image.shape[0] - 1], [0, board_image.shape[0] - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_corners, projected_corners.reshape(-1, 2).astype(np.float32))
    return cv2.warpPerspective(
        board_image,
        homography,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def save_frame(frame, pose_index):
    # capsule_calibration.load_capsule_frame rotates production frames by 180 deg.
    # Emit synthetic frames in that raw orientation so the calibration path sees
    # the same rendered camera-frame convention used to define ground truth.
    raw_frame = cv2.rotate(frame, cv2.ROTATE_180)
    numbered_path = os.path.join(DATA_DIRECTORY, f"imgInfo_{pose_index:05d}.jpg")
    live_path = os.path.join(DATA_DIRECTORY, "live_feed.jpg")
    for destination in (numbered_path, live_path):
        temporary_path = f"{destination}.tmp.jpg"
        if not cv2.imwrite(temporary_path, raw_frame):
            raise OSError(f"Unable to write simulated image to {DATA_DIRECTORY}")
        os.replace(temporary_path, destination)
    return numbered_path


def main():
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    intrinsic_matrix = camera_matrix()
    distortion_coefficients = np.zeros((5, 1), dtype=np.float64)
    np.savez(INTRINSIC_FILE, camera_matrix=intrinsic_matrix, dist_coeffs=distortion_coefficients)

    ground_truth = R.from_euler("xyz", GROUND_TRUTH_EULER_DEG, degrees=True).as_matrix()
    np.savez(GROUND_TRUTH_FILE, R_cam_to_pcb_GT=ground_truth)
    board_image = build_board_image()

    server = ImuTelemetryServer(TCP_HOST, TCP_PORT)
    server.start()
    print(f"Wrote camera intrinsics to {INTRINSIC_FILE}")
    print(f"Ground-truth boresight (roll, pitch, yaw): {GROUND_TRUTH_EULER_DEG}")
    print("Start capsule_calibration.py in 'boresight' mode, then press ENTER for each pose.")

    try:
        for pose_index, capsule_euler in enumerate(TRAJECTORY_EULER_DEG, start=1):
            input(f"\nPose {pose_index}/{len(TRAJECTORY_EULER_DEG)} {capsule_euler}. Press ENTER to publish: ")
            capsule_rotation = R.from_euler("xyz", capsule_euler, degrees=True).as_matrix()
            # capsule_calibration stores R_cam_to_board = R_board_to_cam.T.
            # This makes its stored camera pose capsule_rotation @ ground_truth,
            # which satisfies A @ ground_truth = ground_truth @ B.
            board_to_camera = ground_truth.T @ capsule_rotation.T
            frame = render_charuco_view(
                board_image, board_to_camera, intrinsic_matrix, distortion_coefficients
            )
            image_path = save_frame(frame, pose_index)
            server.set_rotation(capsule_rotation)
            print(f"Published {os.path.basename(image_path)} and matching IMU telemetry.")
        input("\nAll poses are published. Capture the final pose, then press ENTER to stop the simulator: ")
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()