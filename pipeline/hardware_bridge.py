"""Auto-detecting Raspberry Pi 5 + ESP32 Hardware-in-the-Loop (HIL) Buzzer Bridge.

Sends single-byte driver state commands over USB UART (115200 baud) to an
ESP32 DevKit V1 driving a 3-pin active buzzer:
  - b'0': SOBER (silent)
  - b'1': MILD_IMPAIRMENT / HIGH_RISK_INTOXICATED / voice_slurred_detected (warning beep)
  - b'2': lockout_confirmed / microsleep_detected (continuous alarm)

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
        serial_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if mode not in ("auto", "pi-only", "off"):
            raise ValueError(f"Invalid hil mode: {mode!r}. Expected 'auto', 'pi-only', or 'off'.")
        self.mode = mode
        self.requested_port = port
        self.baudrate = baudrate
        self.keepalive_sec = keepalive_sec
        self._serial_factory = serial_factory

        self.is_pi: bool = is_raspberry_pi()
        self.active_port: Optional[str] = None
        self._serial: Any = None
        self._last_cmd: Optional[bytes] = None
        self._last_sent_time: float = 0.0

    @property
    def connected(self) -> bool:
        return self._serial is not None

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
    ) -> bytes:
        """Map driver assessment state to 1-byte ESP32 buzzer command."""
        if lockout_confirmed or microsleep_detected:
            return b"2"
        if classification in ("MILD_IMPAIRMENT", "HIGH_RISK_INTOXICATED") or voice_slurred_detected:
            return b"1"
        return b"0"

    def send_state(
        self,
        classification: str,
        lockout_confirmed: bool = False,
        microsleep_detected: bool = False,
        voice_slurred_detected: bool = False,
        now: Optional[float] = None,
    ) -> Optional[bytes]:
        """Transmit state byte on transition or keepalive interval. Safe no-op if disconnected."""
        if self._serial is None:
            return None

        cmd = self.compute_command(
            classification=classification,
            lockout_confirmed=lockout_confirmed,
            microsleep_detected=microsleep_detected,
            voice_slurred_detected=voice_slurred_detected,
        )
        ts = time.monotonic() if now is None else now
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
