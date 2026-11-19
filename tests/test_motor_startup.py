from unittest.mock import Mock

import pytest

import motorControl


@pytest.fixture
def startup(monkeypatch):
    device = Mock()
    device.set_bitrate.return_value = True
    scan = Mock(return_value=(True, device))
    premove = Mock()
    monkeypatch.setattr(motorControl, "scan_for_device", scan)
    monkeypatch.setattr(motorControl, "vertPreMove", premove)
    monkeypatch.setattr(motorControl, "disable_stepper_28XTpdo", Mock())
    monkeypatch.setattr(motorControl, "tpdo_setup", Mock())
    monkeypatch.setattr(motorControl.threading, "Thread", Mock())
    monkeypatch.setattr(motorControl.time, "sleep", Mock())
    monkeypatch.setattr(motorControl, "threads", [])
    monkeypatch.setattr(motorControl.sys, "argv", ["motorControl.py"])
    return device, scan, premove


def run_until_menu_exhausted(monkeypatch, choices=()):
    choices = iter(choices)
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        try:
            return next(choices)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    with pytest.raises(EOFError):
        motorControl.main()
    return prompts


def assert_operational_frame(device):
    frame = device.send.call_args_list[0].args[0]
    assert frame.can_id == 0
    assert frame.can_dlc == 2
    assert bytes(frame.data[:2]) == b"\x01\x00"


def test_default_startup_scans_and_sends_operational_nmt(startup, monkeypatch):
    device, scan, premove = startup
    prompts = run_until_menu_exhausted(monkeypatch)

    scan.assert_called_once_with(motorControl.coveredCanConverterId)
    device.set_bitrate.assert_called_once_with(motorControl.canBusBitRate)
    device.start.assert_called_once_with(motorControl.GS_CAN_MODE_NORMAL)
    device.send.assert_called_once()
    assert_operational_frame(device)
    premove.assert_called_once_with(device, motorControl.motors)
    assert len(prompts) == 1 and "Choose an option" in prompts[0]


@pytest.mark.parametrize("failure", ["not_found", "bitrate"])
def test_failed_connection_waits_for_manual_retry(startup, monkeypatch, failure):
    device, scan, premove = startup
    if failure == "not_found":
        scan.side_effect = [(False, None), (True, device)]
    else:
        device.set_bitrate.side_effect = [False, True]

    prompts = run_until_menu_exhausted(monkeypatch, ["1"])

    assert scan.call_count == 2
    assert len(prompts) == 2 and all("Choose an option" in p for p in prompts)
    device.start.assert_called_once()
    device.send.assert_called_once()
    assert_operational_frame(device)
    premove.assert_called_once()


def test_manual_nmt_still_requires_power_confirmation(startup, monkeypatch):
    device, _, premove = startup
    prompts = run_until_menu_exhausted(monkeypatch, ["2", "01", "n"])

    assert any("restart the motor power" in prompt for prompt in prompts)
    device.send.assert_called_once()
    premove.assert_called_once()


def test_auto_nmt_flag_still_prompts_for_zeroing(startup, monkeypatch):
    device, _, _ = startup
    monkeypatch.setattr(motorControl.sys, "argv", ["motorControl.py", "--auto-scan", "--auto-nmt"])
    prompts = run_until_menu_exhausted(monkeypatch, ["n"])

    device.send.assert_called_once()
    assert "automatic zeroing" in prompts[0]
    assert "Choose an option" in prompts[1]
