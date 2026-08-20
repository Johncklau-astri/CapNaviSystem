import importlib.util
import sys
import types


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
        'position_world': [1.1, 1.8, 3.3],
        'orientation_world': [1.0, 0.0, 0.0, 0.0],
    }