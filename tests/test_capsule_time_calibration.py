import importlib.util
import sys
import types


ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'capsule_orientation.py'


def load_module():
    if 'keyboard' not in sys.modules:
        sys.modules['keyboard'] = types.ModuleType('keyboard')
    if 'serial' not in sys.modules:
        serial_module = types.ModuleType('serial')
        serial_module.Serial = object
        serial_module.SerialException = Exception
        sys.modules['serial'] = serial_module
    if 'watchdog.observers' not in sys.modules:
        observers = types.ModuleType('watchdog.observers')
        observers.Observer = object
        sys.modules['watchdog.observers'] = observers
    if 'watchdog.events' not in sys.modules:
        events = types.ModuleType('watchdog.events')
        events.FileSystemEventHandler = object
        sys.modules['watchdog.events'] = events

    spec = importlib.util.spec_from_file_location('capsule_orientation_calibration', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_newest_sensor_timestamp_ignores_stale_packets(tmp_path, monkeypatch):
    module = load_module()
    module.DATA_DIRECTORY = str(tmp_path)
    old_file = tmp_path / 'imgInfo_00001_0x20240719.txt'
    first_new_file = tmp_path / 'imgInfo_00002_0x20240719.txt'
    new_file = tmp_path / 'imgInfo_00003_0x20240719.txt'
    old_file.write_text('old')
    first_new_file.write_text('first')
    new_file.write_text('new')

    monkeypatch.setattr(
        module,
        '_read_telemetry_samples',
        lambda path, processor: (
            [10.0, 10.1] if path == new_file else
            [9.5] if path == first_new_file else
            [9.0]
        ),
    )

    selected_file, timestamp = module._find_newest_sensor_timestamp(
        module.DataProcessor(),
        {'imgInfo_00001_0x20240719.txt'},
        10.05,
        timeout=0.05,
        expected_new_files=2,
    )

    assert selected_file == new_file
    assert timestamp == 10.1


def test_read_serial_line_uses_computer_receive_time():
    module = load_module()

    class FakeSerial:
        def __init__(self):
            self.reads = [
                b'15:33:19:250 -> +++Image-Start\r\n'
                b'15:33:19:289 -> ImgCt2\n'
            ]

        def read(self, size):
            return self.reads.pop(0) if self.reads else b''

    original_time = module.time.time
    module.time.time = lambda: 1234.5
    try:
        connection = FakeSerial()
        received_time, line = module._read_serial_line(connection)
        _, completion_line = module._read_serial_line(connection)
    finally:
        module.time.time = original_time

    assert received_time == 1234.5
    assert line.endswith('+++Image-Start')
    assert completion_line.endswith('ImgCt2')


def test_time_calibration_recaptures_when_telemetry_sidecar_is_missing(tmp_path, monkeypatch):
    module = load_module()

    class FakeSerial:
        def __init__(self, *args, **kwargs):
            self.writes = []

        def reset_input_buffer(self):
            pass

        def write(self, command):
            self.writes.append(command)

        def close(self):
            pass

    connection = FakeSerial()
    image_starts = iter([
        (100.0, '+++Image-Start'), (101.0, '+++Image-Start'),
        (200.0, '+++Image-Start'), (201.0, '+++Image-Start'),
    ])
    telemetry_responses = iter([(None, None), (tmp_path / 'imgInfo_00004.txt', 205.0)])
    telemetry_timeouts = []

    def find_telemetry(*args, **kwargs):
        telemetry_timeouts.append(args[3])
        return next(telemetry_responses)

    monkeypatch.setattr(module.serial, 'Serial', lambda *args, **kwargs: connection)
    monkeypatch.setattr(module, '_wait_for_serial_text', lambda *args: 0.0)
    monkeypatch.setattr(module, '_read_serial_line', lambda *args: next(image_starts))
    monkeypatch.setattr(module, '_telemetry_file_snapshot', lambda *args: (set(), -1.0))
    monkeypatch.setattr(module, '_find_newest_sensor_timestamp', find_telemetry)

    result = module.run_time_calibration(
        trials=1,
        warmup_trials=0,
        output_file=tmp_path / 'clock.json',
    )

    assert connection.writes == [b'at+photo=p,1,2\r', b'at+photo+\r', b'at+photo+\r']
    assert result['discarded_capture_count'] == 1
    assert result['trials'][0]['discarded_capture_retries'] == 1
    assert result['offset_seconds'] == 4.0
    assert telemetry_timeouts == [module.CALIBRATION_SIDECAR_GRACE_TIMEOUT] * 2


def test_main_starts_normal_extraction_after_time_calibration(monkeypatch):
    module = load_module()
    started = []

    class FakeHandler:
        def __init__(self, processor):
            self.processor = processor

        def prime_directory(self, directory):
            started.append(('primed', directory))

    class FakeThread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started.append(('thread', self.target))

    class FakeObserver:
        def schedule(self, handler, directory, recursive):
            started.append(('scheduled', directory, recursive))

        def start(self):
            started.append(('observer',))

        def stop(self):
            pass

        def join(self, timeout):
            pass

    monkeypatch.setattr(
        module,
        'parse_args',
        lambda: types.SimpleNamespace(
            plot=False,
            calibration_trials=2,
            calibration_warmup_trials=1,
            calibration_output='clock.json',
        ),
    )
    monkeypatch.setattr(
        module,
        'run_time_calibration',
        lambda **kwargs: (
            started.append(('calibrated', kwargs)),
            {'offset_seconds': -12.5},
        )[1],
    )
    monkeypatch.setattr(module, 'DataFileHandler', FakeHandler)
    monkeypatch.setattr(module.threading, 'Thread', FakeThread)
    monkeypatch.setattr(module, 'Observer', FakeObserver)
    monkeypatch.setattr(module.system_signal, 'signal', lambda *args: None)
    monkeypatch.setattr(module.atexit, 'register', lambda callback: None)
    monkeypatch.setattr(module, 'cleanup', lambda: started.append(('cleaned',)))
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()))

    module.main()

    assert started[0][0] == 'calibrated'
    assert ('primed', module.DATA_DIRECTORY) in started
    assert ('observer',) in started