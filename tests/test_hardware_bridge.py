import builtins
import pytest

from pipeline import hardware_bridge
from pipeline.hardware_bridge import HardwareBuzzerBridge


class _FakeSerial:
    def __init__(self, port: str, baudrate: int, timeout: float = 0.01):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.writes = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


def test_command_mapping_priority():
    assert HardwareBuzzerBridge.compute_command("SOBER") == b"0"
    assert HardwareBuzzerBridge.compute_command("INSUFFICIENT_DATA") == b"0"
    assert HardwareBuzzerBridge.compute_command("MILD_IMPAIRMENT") == b"1"
    assert HardwareBuzzerBridge.compute_command("HIGH_RISK_INTOXICATED") == b"1"
    assert HardwareBuzzerBridge.compute_command("SOBER", voice_slurred_detected=True) == b"1"
    assert HardwareBuzzerBridge.compute_command("SOBER", microsleep_detected=True) == b"2"
    assert HardwareBuzzerBridge.compute_command("MILD_IMPAIRMENT", lockout_confirmed=True) == b"2"


def test_mode_off_never_opens_serial():
    called = []

    def factory(*args, **kwargs):
        called.append(args)
        return _FakeSerial(*args, **kwargs)

    bridge = HardwareBuzzerBridge(mode="off", port="/dev/ttyUSB0", serial_factory=factory)
    assert bridge.connect() is False
    assert bridge.connected is False
    assert called == []
    assert bridge.send_state("HIGH_RISK_INTOXICATED") is None


def test_mode_pi_only_skips_on_non_pi(monkeypatch):
    monkeypatch.setattr(hardware_bridge, "is_raspberry_pi", lambda: False)
    bridge = HardwareBuzzerBridge(
        mode="pi-only",
        port="/dev/ttyUSB0",
        serial_factory=_FakeSerial,
    )
    assert bridge.connect() is False
    assert bridge.connected is False


def test_auto_connect_deduplication_and_keepalive(monkeypatch):
    fake_holder = {}

    def factory(port, baudrate, timeout=0.01):
        conn = _FakeSerial(port, baudrate, timeout=timeout)
        fake_holder["conn"] = conn
        return conn

    monkeypatch.setattr(hardware_bridge, "discover_esp32_ports", lambda allow_generic_linux_usb=False: ["/dev/ttyUSB0"])
    bridge = HardwareBuzzerBridge(mode="auto", keepalive_sec=2.0, serial_factory=factory)
    assert bridge.connect() is True
    assert bridge.connected is True
    assert bridge.active_port == "/dev/ttyUSB0"

    assert bridge.send_state("SOBER", now=10.0) == b"0"
    assert bridge.send_state("SOBER", now=10.5) is None  # deduplicated
    assert bridge.send_state("MILD_IMPAIRMENT", now=10.6) == b"1"  # immediate transition
    assert bridge.send_state("MILD_IMPAIRMENT", now=12.7) == b"1"  # keepalive resend
    assert bridge.send_state("SOBER", lockout_confirmed=True, now=12.8) == b"2"

    assert fake_holder["conn"].writes == [b"0", b"1", b"1", b"2"]


def test_close_silences_buzzer_with_zero_byte():
    conn = _FakeSerial("COM3", 115200)
    bridge = HardwareBuzzerBridge(mode="auto", port="COM3", serial_factory=lambda *a, **kw: conn)
    assert bridge.connect() is True
    bridge.send_state("HIGH_RISK_INTOXICATED", now=1.0)
    bridge.close()

    assert conn.writes[-1] == b"0"
    assert conn.closed is True
    assert bridge.connected is False


def test_missing_pyserial_graceful_noop(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "serial":
            raise ImportError("No module named 'serial'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    bridge = HardwareBuzzerBridge(mode="auto", port="/dev/ttyUSB0")
    assert bridge.connect() is False
    assert bridge.connected is False
