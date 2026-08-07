"""Batch-undistort JPEG images using capsule_intrinsics.npz."""

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INTRINSICS = Path(__file__).with_name("capsule_intrinsics.npz")
JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Undistort all JPEG images in a directory using a saved capsule calibration."
    )
    parser.add_argument("input_directory", type=Path, help="Directory containing source JPEG images.")
    parser.add_argument("output_directory", type=Path, help="Directory for undistorted JPEG images.")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=DEFAULT_INTRINSICS,
        help=f"Calibration .npz file (default: {DEFAULT_INTRINSICS.name}).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process JPEG images in input subdirectories.",
    )
    return parser.parse_args()


def find_jpeg_images(input_directory, recursive):
    paths = input_directory.rglob("*") if recursive else input_directory.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS)


def main():
    args = parse_arguments()
    input_directory = args.input_directory.resolve()
    output_directory = args.output_directory.resolve()
    intrinsics_path = args.intrinsics.resolve()

    if not input_directory.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_directory}")
    if not intrinsics_path.is_file():
        raise SystemExit(f"Intrinsic calibration file does not exist: {intrinsics_path}")
    if input_directory == output_directory:
        raise SystemExit("Output directory must be different from the input directory.")

    try:
        calibration = np.load(intrinsics_path)
        camera_matrix = calibration["camera_matrix"]
        dist_coeffs = calibration["dist_coeffs"]
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Could not load calibration parameters: {error}") from error

    image_paths = find_jpeg_images(input_directory, args.recursive)
    if not image_paths:
        print(f"No JPEG images found in: {input_directory}")
        return

    processed_count = 0
    skipped_count = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            skipped_count += 1
            continue

        relative_path = image_path.relative_to(input_directory)
        output_path = output_directory / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

        if not cv2.imwrite(str(output_path), undistorted_image):
            print(f"Skipped unwritable output: {output_path}")
            skipped_count += 1
            continue

        processed_count += 1
        print(f"Undistorted: {relative_path}")

    print(f"Completed: {processed_count} image(s) written to {output_directory}")
    if skipped_count:
        print(f"Skipped: {skipped_count} image(s)")


if __name__ == "__main__":
    main()