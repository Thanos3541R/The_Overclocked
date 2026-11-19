"""Auto-detecting Raspberry Pi 5 + ESP32 Hardware-in-the-Loop (HIL) Buzzer Bridge.

Sends single-byte driver state commands over USB UART (115200 baud) to an
ESP32 DevKit V1 driving a 3-pin active buzzer:
  - b'0': SOBER (silent)
  - b'1': Sustained MILD_IMPAIRMENT / HIGH_RISK_INTOXICATED / corroborated slurred voice (warning beep)
  - b'2': lockout_confirmed / microsleep_detected (instant continuous alarm)

Safe by default: if `pyserial` is not installed or no ESP32 USB-UART device is
connected (e.g., laptop dev or GitHub Actions CI), all operations are silent no-ops.
"""
from __future__ import annotations

import glob
import platform
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

# Known USB-to-UART bridge Vendor IDs used on ESP32 DevKit boards:
#   0x10C4 = Silicon Labs CP210x
#   0x1A86 = WCH CH340 / CH9102
#   0x0403 = FTDI FT232R
#   0x303A = Espressif native USB (ESP32-S2/S3/C3)
ESP32_USB_VIDS = frozenset({0x10C4, 0x1A86, 0x0403, 0x303A})
ESP32_DESC_KEYWORDS = ("cp210", "ch340", "ch9102", "esp32", "usb-serial", "usb serial", "uart")


def is_raspberry_pi() -> bool:
    """Return True if running on Raspberry Pi hardware."""
    if platform.system() != "Linux":
        return False
    for model_path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            text = Path(model_path).read_text(encoding="utf-8", errors="ignore").strip("\x00\n ")
            if "raspberry pi" in text.lower():
                return True
        except OSError:
            continue
    return False


def discover_esp32_ports(allow_generic_linux_usb: bool = False) -> List[str]:
    """Discover candidate USB serial ports belonging to an ESP32 board."""
    candidates: List[str] = []
    try:
        from serial.tools import list_ports

        for port_info in list_ports.comports():
            vid = getattr(port_info, "vid", None)
            desc = (getattr(port_info, "description", "") or "").lower()
            hwid = (getattr(port_info, "hwid", "") or "").lower()
            if (vid in ESP32_USB_VIDS) or any(
                kw in desc or kw in hwid for kw in ESP32_DESC_KEYWORDS
            ):
                if port_info.device not in candidates:
                    candidates.append(port_info.device)
    except Exception:
        pass

    if allow_generic_linux_usb and platform.system() == "Linux":
        for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
            for dev in sorted(glob.glob(pattern)):
                if dev not in candidates:
                    candidates.append(dev)

    return candidates


class HardwareBuzzerBridge:
    """Non-blocking, auto-detecting serial bridge for the ESP32 active buzzer."""

    def __init__(
        self,
        mode: str = "auto",
        port: Optional[str] = None,
        baudrate: int = 115200,
        keepalive_sec: float = 2.0,
        warning_debounce_sec: float = 1.5,
        high_risk_debounce_sec: float = 0.5,
        gate_hold_sec: float = 1.0,
        serial_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if mode not in ("auto", "pi-only", "off"):
            raise ValueError(f"Invalid hil mode: {mode!r}. Expected 'auto', 'pi-only', or 'off'.")
        self.mode = mode
        self.requested_port = port
        self.baudrate = baudrate
        self.keepalive_sec = keepalive_sec
        self.warning_debounce_sec = max(0.0, float(warning_debounce_sec))
        self.high_risk_debounce_sec = max(0.0, float(high_risk_debounce_sec))
        self.gate_hold_sec = max(0.0, float(gate_hold_sec))
        self._serial_factory = serial_factory

        self.is_pi: bool = is_raspberry_pi()
        self.active_port: Optional[str] = None
        self._serial: Any = None
        self._last_cmd: Optional[bytes] = None
        self._last_sent_time: float = 0.0
        self._pending_warn_start: Optional[float] = None
        self._warn_active: bool = False
        self._gate_start: Optional[float] = None

    @property
    def connected(self) -> bool:
        return self._serial is not None

    def reset_debounce(self) -> None:
        """Clear warning persistence timers."""
        self._pending_warn_start = None
        self._warn_active = False
        self._gate_start = None

    def connect(self) -> bool:
        """Probe and connect to the ESP32 serial port if permitted by mode."""
        if self.mode == "off":
            return False
        if self.mode == "pi-only" and not self.is_pi:
            return False

        factory = self._serial_factory
        if factory is None:
            try:
                import serial  # type: ignore

                factory = serial.Serial
            except Exception:
                return False

        ports = (
            [self.requested_port]
            if self.requested_port
            else discover_esp32_ports(allow_generic_linux_usb=self.is_pi)
        )

        for candidate in ports:
            try:
                conn = factory(candidate, self.baudrate, timeout=0.01)
                self._serial = conn
                self.active_port = candidate
                self._last_cmd = None
                self._last_sent_time = 0.0
                self.reset_debounce()
                host_tag = "Raspberry Pi detected — " if self.is_pi else ""
                print(
                    f"[DMS-HIL] {host_tag}ESP32 Buzzer Bridge active on "
                    f"{candidate} @ {self.baudrate} baud"
                )
                return True
            except Exception:
                continue

        return False

    @staticmethod
    def compute_command(
        classification: str,
        lockout_confirmed: bool = False,
        microsleep_detected: bool = False,
        voice_slurred_detected: bool = False,
        risk_score: Optional[float] = None,
    ) -> bytes:
        """Map driver assessment state to 1-byte ESP32 buzzer command.

        Voice slurring acts as secondary corroboration when `risk_score` is provided
        (requiring risk_score >= 25.0) so ambient mic hum during SOBER frames never
        overrides visual sobriety.
        """
        if lockout_confirmed or microsleep_detected:
            return b"2"
        if classification in ("MILD_IMPAIRMENT", "HIGH_RISK_INTOXICATED"):
            return b"1"
        if voice_slurred_detected and (risk_score is None or risk_score >= 25.0):
            return b"1"
        return b"0"

    def resolve_command(
        self,
        classification: str,
        lockout_confirmed: bool = False,
        microsleep_detected: bool = False,
        voice_slurred_detected: bool = False,
        risk_score: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bytes:
        """Apply tiered persistence debouncing and brief IMU-gate hold before emitting."""
        ts = time.monotonic() if now is None else now

        # Critical alarms (lockout / microsleep) bypass debounce immediately
        if lockout_confirmed or microsleep_detected:
            self.reset_debounce()
            return b"2"

        # Brief IMU shock / pose gate holds active/pending state up to gate_hold_sec
        if classification == "INSUFFICIENT_DATA":
            if self._gate_start is None:
                self._gate_start = ts
            if (ts - self._gate_start) <= self.gate_hold_sec:
                return b"1" if self._warn_active else b"0"
            self.reset_debounce()
            return b"0"

        self._gate_start = None
        raw_cmd = self.compute_command(
            classification=classification,
            lockout_confirmed=False,
            microsleep_detected=False,
            voice_slurred_detected=voice_slurred_detected,
            risk_score=risk_score,
        )

        if raw_cmd == b"1":
            if self._pending_warn_start is None:
                self._pending_warn_start = ts
            req_sec = (
                self.high_risk_debounce_sec
                if classification == "HIGH_RISK_INTOXICATED"
                else self.warning_debounce_sec
            )
            if self._warn_active or (ts - self._pending_warn_start) >= req_sec:
                self._warn_active = True
                return b"1"
            return b"0"

        # Genuine SOBER frame immediately clears warning state
        self.reset_debounce()
        return b"0"

    def send_state(
        self,
        classification: str,
        lockout_confirmed: bool = False,
        microsleep_detected: bool = False,
        voice_slurred_detected: bool = False,
        risk_score: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Optional[bytes]:
        """Transmit debounced state byte on transition or keepalive. Safe no-op if disconnected."""
        if self._serial is None:
            return None

        ts = time.monotonic() if now is None else now
        cmd = self.resolve_command(
            classification=classification,
            lockout_confirmed=lockout_confirmed,
            microsleep_detected=microsleep_detected,
            voice_slurred_detected=voice_slurred_detected,
            risk_score=risk_score,
            now=ts,
        )
        needs_send = (cmd != self._last_cmd) or ((ts - self._last_sent_time) >= self.keepalive_sec)
        if not needs_send:
            return None

        try:
            self._serial.write(cmd)
            self._last_cmd = cmd
            self._last_sent_time = ts
            return cmd
        except Exception:
            return None

    def close(self) -> None:
        """Silence buzzer and close the serial port cleanly."""
        self.reset_debounce()
        if self._serial is None:
            return
        try:
            self._serial.write(b"0")
        except Exception:
            pass
        try:
            self._serial.close()
        except Exception:
            pass
        self._serial = None
        self.active_port = None
        self._last_cmd = None
