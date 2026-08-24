import importlib.util
import json
import sys
import types
import struct
import numpy as np


ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'aprilTagLocalization.py'


def load_module():
    if 'pupil_apriltags' not in sys.modules:
        sys.modules['pupil_apriltags'] = types.SimpleNamespace(Detector=object)
    if 'cv2' not in sys.modules:
        sys.modules['cv2'] = types.ModuleType('cv2')
    if 'serial' not in sys.modules:
        serial_module = types.ModuleType('serial')
        serial_module.Serial = object
        serial_module.SerialException = Exception
        sys.modules['serial'] = serial_module

    spec = importlib.util.spec_from_file_location('capture_export_module', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_capture_bundle_writes_ready_pose_and_image(tmp_path):
    module = load_module()
    source_image = tmp_path / 'image_00003_0x20240719.jpg'
    source_image.write_bytes(b'jpeg-data')
    module.CAPTURE_EXPORT_DIRECTORY = str(tmp_path / 'export')

    request = {
        'capture_id': '00003_0x20240719',
        'image_path': str(source_image),
        'timestamp': 123.456,
        'orientation_world': [1.0, 0.0, 0.0, 0.0],
        'motor_data': {
            'x': 1.0, 'y': 2.0, 'z': 3.0,
            'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,
        },
    }
    response = {'estX': 0.1, 'estY': -0.2, 'estZ': 0.3}

    exported_capture = module.export_capture_bundle(request, response)

    output_dir = tmp_path / 'export' / '00003'
    assert exported_capture['image_path'] == output_dir / 'image_00003_0x20240719.jpg'
    assert exported_capture['position_world'].tolist() == [1.1, 1.8, 3.3]
    assert exported_capture['orientation_world'] == [1.0, 0.0, 0.0, 0.0]
    assert (output_dir / 'image_00003_0x20240719.jpg').read_bytes() == b'jpeg-data'
    assert (output_dir / 'capture.ready').is_file()
    payload = __import__('json').loads((output_dir / 'pos.json').read_text())
    assert payload == {
        'capture_id': '00003_0x20240719',
        'image_file': 'image_00003_0x20240719.jpg',
        'timestamp': 123.456,
        'timestamp_computer': None,
        'motor_time_error': None,
        'motor_orientation': [1.0, 0.0, 0.0, 0.0],
        'position_world': [1.1, 1.8, 3.3],
        'orientation_world': [1.0, 0.0, 0.0, 0.0],
    }


def test_motor_pose_history_interpolates_position_and_quaternion():
    module = load_module()
    module.motor_pose_history.clear()
    module.motor_pose_history.extend([
        (10.0, {
            'timestamp': 10.0,
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,
        }),
        (11.0, {
            'timestamp': 11.0,
            'x': 1.0, 'y': 2.0, 'z': 3.0,
            'qw': 0.0, 'qx': 1.0, 'qy': 0.0, 'qz': 0.0,
        }),
    ])

    pose, time_error = module.get_motor_pose_at_time(10.5)

    assert time_error == 0.0
    assert pose['x'] == 0.5
    assert pose['y'] == 1.0
    assert pose['z'] == 1.5
    assert abs(np.linalg.norm([
        pose['qw'], pose['qx'], pose['qy'], pose['qz'],
    ]) - 1.0) < 1e-9


def test_load_latest_exported_capture_reads_ready_bundle(tmp_path):
    module = load_module()
    export_directory = tmp_path / 'export'
    complete_bundle = export_directory / '00003'
    complete_bundle.mkdir(parents=True)
    (complete_bundle / 'capture.jpg').write_bytes(b'jpeg-data')
    (complete_bundle / 'pos.json').write_text(__import__('json').dumps({
        'image_file': 'capture.jpg',
        'position_world': [1.0, 2.0, 3.0],
        'orientation_world': [1.0, 0.0, 0.0, 0.0],
        'motor_orientation': [0.0, 1.0, 0.0, 0.0],
    }))
    (complete_bundle / 'capture.ready').touch()

    incomplete_bundle = export_directory / '00004'
    incomplete_bundle.mkdir()
    (incomplete_bundle / 'pos.json').write_text('{}')

    module.CAPTURE_EXPORT_DIRECTORY = str(export_directory)

    exported_capture = module.load_latest_exported_capture()

    assert exported_capture['image_path'] == complete_bundle / 'capture.jpg'
    assert exported_capture['position_world'] == [1.0, 2.0, 3.0]
    assert exported_capture['orientation_world'] == [1.0, 0.0, 0.0, 0.0]
    assert exported_capture['motor_orientation'] == [0.0, 1.0, 0.0, 0.0]


def test_queue_latest_image_capture_uses_final_sensor_timestamp(tmp_path):
    module = load_module()
    source_directory = tmp_path / 'output'
    source_directory.mkdir()
    image_path = source_directory / 'image_00025_0x20240719.jpg'
    image_path.write_bytes(b'jpeg-data')

    extracted_data = bytearray(244)
    packet_timestamps = [1000 * packet_number for packet_number in range(1, 9)] + [1500]
    for packet_number, timestamp in enumerate(packet_timestamps):
        packet = struct.pack('>I', timestamp) + bytes(19)
        packet_offset = 6 + packet_number * 23 if packet_number < 5 else 139 + (packet_number - 5) * 23
        extracted_data[packet_offset:packet_offset + 23] = packet
    sensor_file = source_directory / 'imgInfo_00025_0x20240719.txt'
    sensor_file.write_text(extracted_data.hex())

    calibration_file = tmp_path / 'capsule_clock_calibration.json'
    calibration_file.write_text(json.dumps({'offset_seconds': 12.5}))
    module.CAPTURE_SOURCE_DIRECTORY = str(source_directory)
    module.CAPSULE_CLOCK_CALIBRATION_FILE = str(calibration_file)
    queued_capture = {}

    def record_capture(capture, motor_data):
        queued_capture.update(capture)
        return True

    module.request_capture_position_estimation = record_capture

    module.queue_latest_image_capture(
        {'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,
         'magX': 1.0, 'magY': 2.0, 'magZ': 3.0},
        {'x': 0.0, 'y': 0.0, 'z': 0.0,
         'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0},
    )

    assert queued_capture['timestamp'] == 8.0
    assert queued_capture['timestamp_computer'] == -4.5