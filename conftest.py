"""Root pytest configuration.

Ensures that the project root directory is always on sys.path regardless of how
pytest is invoked (direct CLI `pytest`, `python -m pytest`, IDE test runner, CI).
"""
import os
import sys
import types
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import sounddevice as _sd  # noqa: F401
except Exception:
    # Headless CI runners with libportaudio2 but no PulseAudio daemon raise
    # sounddevice.PortAudioError on import, which crashes mediapipe's audio_record import.
    _dummy_sd = types.ModuleType("sounddevice")
    _dummy_sd.InputStream = None
    _dummy_sd.PortAudioError = Exception
    sys.modules["sounddevice"] = _dummy_sd
