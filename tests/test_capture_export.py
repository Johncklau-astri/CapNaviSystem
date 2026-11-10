import importlib.util
import json
import sys
import types
import struct
import numpy as np
import pytest


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
    assert np.allclose(
        exported_capture['camera_world'],
        module.rotation_matrix_to_quaternion(module.R_CAM_TO_PCB),
    )
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
        'camera_world': module.rotation_matrix_to_quaternion(module.R_CAM_TO_PCB),
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


def test_pose_filter_uses_last_export_and_either_threshold():
    module = load_module()
    pose_filter = module.CapturePoseFilter(8, 5)
    identity = [1, 0, 0, 0]
    selected, pose = pose_filter.select([0, 0, 0], identity, 'first')
    assert selected
    pose_filter.record_export(pose)

    selected, _ = pose_filter.select([0.007, 0, 0], [-1, 0, 0, 0], 'nearby')
    assert not selected
    selected, pose = pose_filter.select([0.008, 0, 0], identity, 'distance')
    assert selected
    pose_filter.record_export(pose)
    assert not pose_filter.select([0.009, 0, 0], identity, 'new-reference')[0]

    half_angle = np.deg2rad(5) / 2
    rotation = [2 * np.cos(half_angle), 0, 0, 2 * np.sin(half_angle)]
    selected, pose = pose_filter.select([0.008, 0, 0], rotation, 'rotation')
    assert selected
    pose_filter.record_export(pose)
    assert not pose_filter.select([0.008, 0, 0], [-x for x in rotation], 'same-rotation')[0]


@pytest.mark.parametrize('distance, angle', [
    (0, 5), (-1, 5), (float('nan'), 5), (float('inf'), 5),
    (8, 0), (8, -1), (8, 181), (8, float('nan')), (8, float('inf')),
])
def test_pose_filter_rejects_invalid_thresholds(distance, angle):
    module = load_module()
    with pytest.raises(ValueError):
        module.CapturePoseFilter(distance, angle)


def test_pose_filter_invalid_pose_and_disabled_filter(capsys):
    module = load_module()
    pose_filter = module.CapturePoseFilter(8, 5)
    selected, pose = pose_filter.select([0, 0, 0], [1, 0, 0, 0], 'first')
    pose_filter.record_export(pose)
    selected, invalid_pose = pose_filter.select([float('nan'), 0, 0], [0, 0, 0, 0], 'invalid')
    assert selected and invalid_pose is None
    pose_filter.record_export(invalid_pose)
    assert not pose_filter.select([0, 0, 0], [1, 0, 0, 0], 'same')[0]
    assert 'invalid pose; exporting without filtering' in capsys.readouterr().out

    disabled_filter = module.CapturePoseFilter(8, 5, enabled=False)
    assert disabled_filter.select([0, 0, 0], [1, 0, 0, 0], 'first')[0]
    assert disabled_filter.select([0, 0, 0], [1, 0, 0, 0], 'same')[0]


def test_export_pose_filter_only_advances_after_successful_bundle(tmp_path, monkeypatch):
    module = load_module()
    module.CAPTURE_EXPORT_DIRECTORY = str(tmp_path / 'export')
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'jpeg-data')
    request = {
        'capture_id': '00001_test',
        'image_path': str(image),
        'timestamp': 1.0,
        'orientation_world': [1, 0, 0, 0],
        'motor_data': {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,
        },
    }

    def export(capture_id, x):
        request['capture_id'] = capture_id
        return module.export_capture_bundle(request, {'estX': x, 'estY': 0, 'estZ': 0})

    assert export('00001_test', 0) is not None
    assert export('00002_test', 0.007) is None
    assert not (tmp_path / 'export' / '00002').exists()

    original_copy = module.shutil.copy2
    with monkeypatch.context() as patch:
        def fail_copy(*args):
            raise OSError('copy failed')
        patch.setattr(module.shutil, 'copy2', fail_copy)
        with pytest.raises(OSError, match='copy failed'):
            export('00003_test', 0.008)
    assert module.shutil.copy2 is original_copy
    assert export('00004_test', 0.008) is not None
    assert (tmp_path / 'export' / '00004' / 'capture.ready').is_file()
    assert image.read_bytes() == b'jpeg-data'


def test_export_pose_filter_uses_camera_rotation_and_configured_thresholds(tmp_path):
    module = load_module()
    module.CAPTURE_EXPORT_DIRECTORY = str(tmp_path / 'export')
    module.capture_pose_filter = module.CapturePoseFilter(12, 10)
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'jpeg-data')
    request = {
        'capture_id': '00001_test', 'image_path': str(image), 'timestamp': 1.0,
        'orientation_world': [1, 0, 0, 0],
        'motor_data': {'x': 0.0, 'y': 0.0, 'z': 0.0,
                       'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0},
    }

    def export(capture_id, angle_deg, x=0):
        half_angle = np.deg2rad(angle_deg) / 2
        request['capture_id'] = capture_id
        request['orientation_world'] = [np.cos(half_angle), 0, 0, np.sin(half_angle)]
        return module.export_capture_bundle(request, {'estX': x, 'estY': 0, 'estZ': 0})

    assert export('00001_test', 0) is not None
    assert export('00002_test', 9.9, 0.011) is None
    assert export('00003_test', 10) is not None
    assert export('00004_test', 10, 0.012) is not None


def test_shutdown_uses_same_pose_filter(tmp_path, capsys):
    module = load_module()
    module.CAPTURE_EXPORT_DIRECTORY = str(tmp_path / 'export')
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'jpeg-data')
    motor = {'x': 0.0, 'y': 0.0, 'z': 0.0,
             'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0}
    for capture_id, x in [('00001_test', 0), ('00002_test', 0.007), ('00003_test', 0.008)]:
        request = {'capture_id': capture_id, 'image_path': str(image),
                   'timestamp': 1.0, 'orientation_world': [1, 0, 0, 0],
                   'motor_data': motor, 'capture_queued_monotonic': 1,
                   'position_sent_monotonic': 2, 'position_response_monotonic': 3}
        module.capture_position_response_queue.put(
            (request, {'estX': x, 'estY': 0, 'estZ': 0})
        )

    module.shutdown_capture_export_executor()

    assert (tmp_path / 'export' / '00001' / 'capture.ready').is_file()
    assert not (tmp_path / 'export' / '00002').exists()
    assert (tmp_path / 'export' / '00003' / 'capture.ready').is_file()
    output = capsys.readouterr().out
    assert output.count('image.jpg') == 2
    assert 'filtered' not in output
    assert 'CAPTURE TIMING' not in output


def test_normal_export_dispatch_uses_pose_filter(tmp_path, capsys):
    module = load_module()
    module.CAPTURE_EXPORT_DIRECTORY = str(tmp_path / 'export')
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'jpeg-data')
    motor = {'x': 0.0, 'y': 0.0, 'z': 0.0,
             'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0}
    for capture_id, x in [('00001_test', 0), ('00002_test', 0.007)]:
        request = {'capture_id': capture_id, 'image_path': str(image),
                   'timestamp': 1.0, 'orientation_world': [1, 0, 0, 0],
                   'motor_data': motor, 'capture_queued_monotonic': 1,
                   'position_sent_monotonic': 2, 'position_response_monotonic': 3}
        module.capture_position_response_queue.put(
            (request, {'estX': x, 'estY': 0, 'estZ': 0})
        )

    module.export_completed_capture_bundles()
    module.capture_export_executor.shutdown(wait=True)

    assert (tmp_path / 'export' / '00001' / 'capture.ready').is_file()
    assert not (tmp_path / 'export' / '00002').exists()
    assert module.export_completed_capture_bundles()['image_path'] == tmp_path / 'export' / '00001' / 'image.jpg'
    output = capsys.readouterr().out
    assert 'image.jpg' in output
    assert 'filtered' not in output
    assert 'CAPTURE TIMING' not in output


def test_export_output_aligns_positions_and_preserves_every_filename(tmp_path, capsys):
    module = load_module()
    capsys.readouterr()
    rows = [
        ('image_00001_0x20240719.jpg', [0.12345, -0.5, 0.0001]),
        ('a-much-longer-image-name-that-does-not-change-numeric-columns.jpg', [-0.5, 1.0, -0.0001]),
    ]
    for name, position in rows:
        future = module.Future()
        future.set_result({'image_path': tmp_path / name, 'position_world': position})
        module.capture_export_pending += 1
        module.finish_capture_export(future)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"{'X (m)':>12}  {'Y (m)':>12}  {'Z (m)':>12}  Photo"
    assert len(lines) == 3
    assert lines[1] == f"{0.12345:12.4f}  {-0.5:12.4f}  {0.0001:12.4f}  {rows[0][0]}"
    assert lines[2] == f"{-0.5:12.4f}  {1.0:12.4f}  {-0.0001:12.4f}  {rows[1][0]}"
    assert module.capture_export_pending == 0


def test_export_output_keeps_errors_visible_and_skips_empty_results(capsys):
    module = load_module()
    capsys.readouterr()
    empty_future = module.Future()
    empty_future.set_result(None)
    failed_future = module.Future()
    failed_future.set_exception(OSError('disk unavailable'))
    module.capture_export_pending = 2
    module.finish_capture_export(empty_future)
    module.finish_capture_export(failed_future)
    assert capsys.readouterr().out == '[CAPTURE] background export failed: disk unavailable\n'
    assert module.capture_export_pending == 0


def test_queue_latest_image_does_not_log_routine_candidates(tmp_path, monkeypatch, capsys):
    module = load_module()
    capsys.readouterr()
    image = tmp_path / 'image_00001_test.jpg'
    monkeypatch.setattr(module, 'get_image_sensor_timestamp', lambda capture_id: (1.0, 1.0))
    monkeypatch.setattr(module, 'request_capture_position_estimation', lambda capture, motor: True)

    result = module.queue_latest_image_capture({}, {}, candidate_path=image)

    assert result == image
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('export_enabled', [True, False])
def test_bt_loop_reports_timing_only_when_export_is_disabled(monkeypatch, capsys, export_enabled):
    module = load_module()
    capsys.readouterr()

    class StopAfterOneLoop:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

    class DummyThread:
        def __init__(self, *args, **kwargs):
            self.daemon = False

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    class DummyObserver(DummyThread):
        def schedule(self, *args, **kwargs):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(module, 'stop_event', StopAfterOneLoop())
    monkeypatch.setattr(module.threading, 'Thread', DummyThread)
    monkeypatch.setattr(module, 'Observer', DummyObserver)
    monkeypatch.setattr(module, 'serial_port', None)
    monkeypatch.setattr(module, 'prime_latest_capture_event', lambda: None)
    monkeypatch.setattr(module, 'shutdown_capture_export_executor', lambda enabled: None)
    monkeypatch.setattr(module, 'get_latest_bt_data', lambda: None)
    monkeypatch.setattr('builtins.input', lambda prompt: 'y' if export_enabled else 'n')

    module.run_bt_only_loop(types.SimpleNamespace(display_export=False, export_fps=None))

    output = capsys.readouterr().out
    assert ('Export rate=' in output) is not export_enabled