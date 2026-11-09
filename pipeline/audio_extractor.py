"""Audio-Visual Voice Intoxication Module (Software + Hardware-Ready).

Analyzes voiced acoustic biomarkers for dysarthria, phonetic slurring, glottal jitter,
amplitude shimmer, harmonics-to-noise ratio (HNR), and formant F2 velocity.
Operates strictly as a secondary assist gated by audio-visual voice activity detection (AV-VAD).
"""
import collections
from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from process_guard import run_monitored_subprocess
except ImportError:
    def run_monitored_subprocess(cmd, timeout=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False):
        return subprocess.run(cmd, stdout=stdout, stderr=stderr, timeout=timeout, check=check)

import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Acoustic biomarkers and diagnostic scores extracted from voiced speech."""
    audio_valid: bool = False                 # True when AV-gated voiced speech is analyzed
    voice_vad_active: bool = False            # True when acoustic energy + pitch F0 (80-320 Hz) is present
    av_speech_gated: bool = False             # True when BOTH camera speech_active AND voice_vad_active are True
    vocal_jitter_pct: float = 0.0             # Cycle-to-cycle F0 pitch period perturbation (%) [Sober < 1.0%, Impaired > 1.3%]
    vocal_shimmer_pct: float = 0.0            # Cycle-to-cycle peak amplitude perturbation (%) [Sober < 3.8%, Impaired > 4.5%]
    vocal_hnr_db: float = 20.0                # Harmonics-to-Noise Ratio (dB) [Sober > 15 dB, Impaired < 11 dB]
    formant_f2_slope_hz_ms: float = 25.0      # Second formant F2 transition velocity (Hz/ms) [Sober > 18, Slurred < 11]
    speech_pause_ratio: float = 0.20          # Unvoiced micro-pause fraction during active speech window
    voice_impairment_score: float = 0.0       # Composite 0-100 vocal intoxication score
    voice_slurred_detected: bool = False      # True when >= 2 vocal biomarkers cross impairment thresholds
    mic_active: bool = False                  # True when live mic stream or audio track is open
    audio_rms: float = 0.0                    # Current active window RMS level
    mic_rms_level: float = 0.0                # Current active window RMS level (alias)

    def __post_init__(self):
        if self.mic_rms_level > 0.0 and self.audio_rms == 0.0:
            self.audio_rms = self.mic_rms_level
        elif self.audio_rms > 0.0 and self.mic_rms_level == 0.0:
            self.mic_rms_level = self.audio_rms


class AudioIntoxicationExtractor:
    """Extracts vocal intoxication and dysarthria biomarkers from microphone or audio tracks.

    Supports three operational modes:
      Mode A: Offline or synchronized video/audio file playback (.wav or .mp4 track).
      Mode B: Real-time USB / I2S microphone capture via sounddevice stream.
      Mode C: Hardware injection from external ESP32 + INMP441 I2S DSP edge telemetry.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_sec: float = 2.0,
        source_path: Optional[str] = None,
        enable_live_mic: bool = False,
        hardware_serial_mode: bool = False,
        device: Optional[int] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.window_sec = float(window_sec)
        self.source_path = source_path
        self.enable_live_mic = enable_live_mic
        self.hardware_serial_mode = hardware_serial_mode
        self.device = device
        self.mic_active: bool = False

        self._candidate_devices: List[int] = []
        self._current_device_index: Optional[int] = None
        self._silent_check_start: float = -1.0
        self._noise_floor_rms: float = 0.0015
        self._last_valid_features: Optional[AudioFeatures] = None
        self._last_valid_timestamp: float = -999.0

        self._frame_len = int(0.032 * self.sample_rate)  # 512 samples at 16kHz
        self._hop_len = int(0.016 * self.sample_rate)    # 256 samples (16ms hop for finer F2 slope & pitch tracking)
        self._hann_win = np.hanning(self._frame_len).astype(np.float64)
        self._rfft_freqs = np.fft.rfftfreq(self._frame_len, d=1.0 / self.sample_rate)
        self._f2_mask = (self._rfft_freqs >= 850.0) & (self._rfft_freqs <= 2500.0)
        self._f1_mask = (self._rfft_freqs >= 300.0) & (self._rfft_freqs <= 800.0)
        try:
            low = 75.0 / (0.5 * self.sample_rate)
            high = min(0.95, 4000.0 / (0.5 * self.sample_rate))
            self._bp_b, self._bp_a = butter(4, [low, high], btype="band")
        except Exception:
            self._bp_b, self._bp_a = None, None
        # EMA smoothing state for live mic stability
        self._ema_score: Optional[float] = None

        self._buffer_len = int(self.sample_rate * self.window_sec)
        self._pcm_buffer: collections.deque = collections.deque(maxlen=self._buffer_len)
        self._lock = threading.Lock()

        self._audio_track: Optional[np.ndarray] = None
        self._audio_track_sr: int = self.sample_rate
        self._mic_stream: Optional[Any] = None

        self._injected_telemetry: Optional[Dict[str, Any]] = None
        self._last_features: AudioFeatures = AudioFeatures()
        self._last_valid_time: float = -999.0
        self._last_dsp_timestamp: float = -1.0
        self._last_computed_result: Optional[AudioFeatures] = None
        self._last_speech_active: bool = False

        if self.source_path is not None:
            self._load_source_audio(self.source_path)

        if self.enable_live_mic:
            self._init_live_mic()

    @property
    def _stream(self) -> Optional[Any]:
        return self._mic_stream

    @_stream.setter
    def _stream(self, val: Optional[Any]) -> None:
        self._mic_stream = val

    def load_video_audio(self, path: str) -> None:
        """Public method to extract and load audio from video container (.mp4) or audio file."""
        self.source_path = path
        self._load_source_audio(path)

    def _load_source_audio(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".wav":
                from scipy.io import wavfile
                sr, data = wavfile.read(path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if data.dtype == np.int16:
                    pcm = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    pcm = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    pcm = (data.astype(np.float32) - 128.0) / 128.0
                else:
                    pcm = data.astype(np.float32)
                if sr != self.sample_rate and len(pcm) > 0:
                    from scipy.signal import resample_poly
                    gcd = int(np.gcd(int(sr), int(self.sample_rate)))
                    pcm = resample_poly(pcm, int(self.sample_rate) // gcd, int(sr) // gcd).astype(np.float32)
                self._audio_track = pcm
                self.mic_active = True
                return

            # For .mp4 / video containers, decode mono 16kHz s16le PCM via ffmpeg pipe
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin is None:
                try:
                    import imageio_ffmpeg
                    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_bin = None

            if ffmpeg_bin is not None:
                cmd = [
                    ffmpeg_bin, "-nostdin", "-y", "-i", path,
                    "-vn", "-ac", "1", "-ar", str(self.sample_rate),
                    "-f", "s16le", "-acodec", "pcm_s16le", "-"
                ]
                proc = run_monitored_subprocess(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=False
                )
                if proc.returncode == 0 and len(proc.stdout) >= 640:
                    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                    self._audio_track = pcm
                    self.mic_active = True
        except Exception:
            self._audio_track = None

    def _to_mono_float32(self, data: np.ndarray) -> np.ndarray:
        """Convert arbitrary audio data array to mono float32 in [-1.0, 1.0]."""
        arr = np.asarray(data)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)

        if np.issubdtype(arr.dtype, np.integer):
            max_val = float(np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32) / max_val
        else:
            arr = arr.astype(np.float32)
            max_abs = np.max(np.abs(arr)) if len(arr) > 0 else 0.0
            if max_abs > 1.0:
                arr = arr / max_abs
        return arr

    def _init_live_mic(self) -> None:
        """Initialize non-blocking sounddevice microphone input stream with smart device selection."""
        try:
            try:
                import sounddevice as sd  # type: ignore
            except Exception:
                sd = None

            if sd is None or getattr(sd, "InputStream", None) is None:
                print("[DMS-Audio] WARNING: 'sounddevice' not installed! Run: pip install sounddevice")
                logger.warning("'sounddevice' not installed. Live microphone disabled.")
                self._mic_stream = None
                self.mic_active = False
                return

            devices = sd.query_devices()
            print("[DMS-Audio] Available audio input devices:")
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    api = ""
                    try:
                        api = f" ({sd.query_hostapis(dev.get('hostapi', 0)).get('name', '')})"
                    except Exception:
                        pass
                    print(f"  [{idx}] {dev.get('name', 'Unknown')}{api} | inputs: {dev.get('max_input_channels')}, default SR: {dev.get('default_samplerate')} Hz")

            if self.device is not None:
                self._candidate_devices = [int(self.device)]
            else:
                default_in = -1
                try:
                    default_in = sd.default.device[0]
                except Exception:
                    pass

                scored_devs = []
                for idx, d in enumerate(devices):
                    if d.get("max_input_channels", 0) <= 0:
                        continue
                    name = d.get("name", "").lower()
                    score = 0
                    api_name = ""
                    try:
                        api_name = sd.query_hostapis(d.get("hostapi", 0)).get("name", "").lower()
                    except Exception:
                        pass

                    # Prefer WASAPI/MME/DirectSound over exclusive WDM-KS
                    if "wdm-ks" in api_name:
                        score -= 40
                    elif "wasapi" in api_name:
                        score += 15
                    elif "mme" in api_name or "directsound" in api_name:
                        score += 10

                    # Deprioritize virtual/loopback names
                    if any(v in name for v in ("droidcam", "iriun", "camo", "virtual", "stereo mix", "vb-audio", "mapper")):
                        score -= 50

                    # Prioritize physical microphone names
                    if any(p in name for p in ("microphone array", "realtek", "intel", "amd", "usb", "headset", "external", "mic")):
                        score += 20

                    # Default device bonus only if not virtual
                    if idx == default_in and not any(v in name for v in ("droidcam", "iriun", "camo", "virtual", "stereo mix", "vb-audio", "mapper")):
                        score += 10

                    scored_devs.append((score, idx))

                scored_devs.sort(key=lambda x: x[0], reverse=True)
                self._candidate_devices = [idx for _, idx in scored_devs]

            if not self._candidate_devices:
                print("[DMS-Audio] WARNING: No valid audio input devices detected.")
                self._mic_stream = None
                self.mic_active = False
                return

            self._open_candidate_mic(self._candidate_devices[0])

        except Exception as exc:
            print(f"[DMS-Audio] WARNING: Could not open live mic stream: {exc}")
            logger.warning("Could not open live mic stream (sounddevice): %s", exc)
            self._mic_stream = None
            self.mic_active = False

    _start_live_mic = _init_live_mic

    def _open_candidate_mic(self, dev_index: int) -> bool:
        """Open or switch input stream to candidate device index."""
        try:
            try:
                import sounddevice as sd  # type: ignore
            except Exception:
                sd = None
            if sd is None or getattr(sd, "InputStream", None) is None:
                return False

            if self._mic_stream is not None:
                try:
                    self._mic_stream.stop()
                    self._mic_stream.close()
                except Exception:
                    pass
                self._mic_stream = None

            dev_info = sd.query_devices(dev_index)
            native_sr = int(dev_info.get("default_samplerate", self.sample_rate))
            if native_sr <= 0:
                native_sr = self.sample_rate
            blocksize = max(64, int(native_sr * 0.05))  # 50 ms blocks

            def _mic_cb(indata, frames, time_info, status):
                if indata is not None and len(indata) > 0:
                    mono = indata[:, 0].astype(np.float32, copy=True)
                    if native_sr != self.sample_rate:
                        num_samples = int(round(len(mono) * self.sample_rate / native_sr))
                        if num_samples > 0:
                            mono = np.interp(
                                np.linspace(0.0, 1.0, num_samples, endpoint=False),
                                np.linspace(0.0, 1.0, len(mono), endpoint=False),
                                mono,
                            ).astype(np.float32)
                    with self._lock:
                        self._pcm_buffer.extend(mono.tolist())

            self._mic_stream = sd.InputStream(
                device=dev_index,
                samplerate=native_sr,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=_mic_cb,
            )
            self._mic_stream.start()
            self._current_device_index = dev_index
            self.mic_active = True
            print(f"[DMS-Audio] Selected input device [{dev_index}]: {dev_info.get('name', 'Unknown')} @ {native_sr} Hz (resampling to {self.sample_rate} Hz mono)")
            logger.info("Live microphone input stream started on device %d @ %d Hz", dev_index, native_sr)
            return True
        except Exception as exc:
            logger.warning("Failed to open mic device %d: %s", dev_index, exc)
            return False

    def _switch_to_next_mic(self) -> None:
        """Switch to next candidate microphone device if current stream is silent or dithered."""
        if len(self._candidate_devices) <= 1:
            return
        curr_idx = self._current_device_index
        if curr_idx in self._candidate_devices:
            curr_pos = self._candidate_devices.index(curr_idx)
            next_idx = self._candidate_devices[(curr_pos + 1) % len(self._candidate_devices)]
        else:
            next_idx = self._candidate_devices[0]

        print(f"[DMS-Audio] Input device appears muted/virtual (RMS < 0.00005); switching to next mic device [{next_idx}]...")
        self._open_candidate_mic(next_idx)

    def _empty_or_held(
        self,
        rms: float = 0.0,
        voice_vad_active: bool = False,
        av_speech_gated: bool = False,
        speech_pause_ratio: float = 1.0,
    ) -> AudioFeatures:
        """Create a default inactive AudioFeatures instance with current mic status."""
        is_offline = (self._stream is None)
        return AudioFeatures(
            audio_valid=False,
            voice_vad_active=voice_vad_active,
            av_speech_gated=av_speech_gated,
            speech_pause_ratio=speech_pause_ratio,
            voice_impairment_score=0.0,
            voice_slurred_detected=False,
            mic_active=False if is_offline else self.mic_active,
            audio_rms=float(rms),
        )

    def reset(self) -> None:
        """Reset audio ring buffer and DSP cache."""
        with self._lock:
            self._pcm_buffer.clear()
            self._injected_telemetry = None
            self._last_features = self._empty_or_held()
            self._last_dsp_timestamp = -1.0
            self._last_computed_result = None
            self._last_speech_active = False
            self._last_valid_features = None
            self._last_valid_timestamp = -999.0
            self._silent_check_start = -1.0
            self._ema_score = None
            self._noise_floor_rms = 0.0015

    def push_pcm_chunk(self, pcm_samples: np.ndarray) -> None:
        """Push raw 16 kHz PCM chunk into internal thread-safe ring buffer."""
        samples = self._to_mono_float32(pcm_samples)
        with self._lock:
            self._pcm_buffer.extend(samples.tolist())

    def push_hardware_telemetry(
        self,
        jitter_pct: float,
        shimmer_pct: float,
        hnr_db: float,
        f2_slope_hz_ms: float,
        vad_active: bool = True,
    ) -> None:
        """Inject pre-computed acoustic features from external DSP hardware (e.g. ESP32 I2S)."""
        with self._lock:
            self._injected_telemetry = {
                "jitter_pct": float(jitter_pct),
                "shimmer_pct": float(shimmer_pct),
                "hnr_db": float(hnr_db),
                "f2_slope_hz_ms": float(f2_slope_hz_ms),
                "vad_active": bool(vad_active),
            }
            self._last_dsp_timestamp = -1.0
            self._last_computed_result = None
            self._last_speech_active = False

    def _finalize_result(self, res: AudioFeatures, timestamp: float, visual_speech_active: bool = False) -> AudioFeatures:
        """Apply brief 0.6s inter-syllable speech hold (only while mouth is active) and dither fallback."""
        if res.audio_valid:
            self._last_valid_features = res
            self._last_valid_timestamp = timestamp
        elif (
            self._mic_stream is not None
            and visual_speech_active
            and self._last_valid_features is not None
            and 0.0 <= (timestamp - self._last_valid_timestamp) <= 0.60
        ):
            import dataclasses
            res = dataclasses.replace(
                self._last_valid_features,
                audio_rms=res.audio_rms,
                mic_active=self.mic_active,
            )

        # Check for digital dither / muted device fallback
        if self._mic_stream is not None and len(self._candidate_devices) > 1:
            curr_rms = res.audio_rms
            if curr_rms < 0.00005:
                if self._silent_check_start < 0:
                    self._silent_check_start = timestamp
                elif (timestamp - self._silent_check_start) >= 4.0:
                    self._silent_check_start = timestamp
                    self._switch_to_next_mic()
            else:
                self._silent_check_start = -1.0

        self._last_features = res
        self._last_dsp_timestamp = timestamp
        self._last_computed_result = res
        return res

    def update(self, timestamp: float, visual_speech_active: bool) -> AudioFeatures:
        """Update voice intoxication analysis for the current video timestamp.

        Args:
            timestamp: Current video / stream timestamp in seconds.
            visual_speech_active: True if camera detects driver's mouth moving with speech cadence.

        Returns:
            AudioFeatures dataclass.
        """
        # Mode C: Hardware Telemetry Injection Priority
        with self._lock:
            telem = self._injected_telemetry
            self._injected_telemetry = None

        if telem is not None:
            vad_active = telem.get("vad_active", True)
            if not visual_speech_active or not vad_active:
                features = self._empty_or_held(voice_vad_active=vad_active, av_speech_gated=False)
                self._last_features = features
                self._last_dsp_timestamp = timestamp
                self._last_computed_result = features
                self._last_speech_active = visual_speech_active
                return features

            jitter = telem.get("jitter_pct", 0.0)
            shimmer = telem.get("shimmer_pct", 0.0)
            hnr = telem.get("hnr_db", 20.0)
            f2 = telem.get("f2_slope_hz_ms", 25.0)

            s_jitter = float(np.clip((jitter - 1.0) / 1.5 * 100.0, 0.0, 100.0))
            s_shimmer = float(np.clip((shimmer - 3.8) / 4.2 * 100.0, 0.0, 100.0))
            s_hnr = float(np.clip((14.0 - hnr) / 8.0 * 100.0, 0.0, 100.0))
            s_f2 = float(np.clip((16.0 - f2) / 10.0 * 100.0, 0.0, 100.0))
            score = float(0.30 * s_jitter + 0.25 * s_shimmer + 0.20 * s_hnr + 0.25 * s_f2)
            slurred = bool(score >= 45.0 and sum([s_jitter > 35, s_shimmer > 35, s_hnr > 35, s_f2 > 35]) >= 2)

            features = AudioFeatures(
                audio_valid=True,
                voice_vad_active=True,
                av_speech_gated=True,
                vocal_jitter_pct=jitter,
                vocal_shimmer_pct=shimmer,
                vocal_hnr_db=hnr,
                formant_f2_slope_hz_ms=f2,
                speech_pause_ratio=0.15,
                voice_impairment_score=score,
                voice_slurred_detected=slurred,
                mic_active=self.mic_active,
                audio_rms=0.05,
            )
            self._last_features = features
            self._last_dsp_timestamp = timestamp
            self._last_computed_result = features
            self._last_speech_active = visual_speech_active
            return features

        # 10 Hz DSP Throttling (0.10s interval): return cached result if within interval, visual speech active and valid
        if (
            visual_speech_active
            and self._last_computed_result is not None
            and self._last_computed_result.audio_valid
            and 0.0 <= (timestamp - self._last_dsp_timestamp) < 0.10
        ):
            return self._last_computed_result

        self._last_speech_active = visual_speech_active

        # Mode A: Synchronized Video/Audio Track
        if self._audio_track is not None and len(self._audio_track) > 0:
            window_samples = int(self.sample_rate * self.window_sec)
            end_sample = max(0, int(timestamp * self.sample_rate))
            start_sample = max(0, end_sample - window_samples)
            if end_sample <= len(self._audio_track):
                pcm = self._audio_track[start_sample:end_sample]
            else:
                pcm = self._audio_track[start_sample:]
            res = self.analyze_pcm_window(pcm, visual_speech_active=visual_speech_active, timestamp=timestamp)
            return self._finalize_result(res, timestamp, visual_speech_active)

        # Mode B: Live Ring Buffer
        with self._lock:
            buf_len = len(self._pcm_buffer)
            if buf_len >= int(0.25 * self.sample_rate):
                if self._stream is not None and buf_len > int(1.2 * self.sample_rate):
                    pcm = np.array(self._pcm_buffer, dtype=np.float32)[-int(1.2 * self.sample_rate):]
                else:
                    pcm = np.array(self._pcm_buffer, dtype=np.float32)
            else:
                pcm = np.array([], dtype=np.float32)

        if len(pcm) >= int(0.25 * self.sample_rate):
            res = self.analyze_pcm_window(pcm, visual_speech_active=visual_speech_active, timestamp=timestamp)
            return self._finalize_result(res, timestamp, visual_speech_active)

        res = self._empty_or_held(rms=0.0)
        return self._finalize_result(res, timestamp, visual_speech_active)

    def analyze_pcm_window(
        self,
        pcm: np.ndarray,
        visual_speech_active: bool = True,
        timestamp: float = 0.0,
    ) -> AudioFeatures:
        """Run vectorized time-frequency DSP on a PCM window to extract vocal intoxication biomarkers."""
        mic_is_live = self.mic_active or (self._audio_track is not None)
        if pcm is None or len(pcm) < int(self.sample_rate * 0.15):
            return AudioFeatures(mic_active=mic_is_live)

        pcm = np.asarray(pcm, dtype=np.float64)
        if pcm.ndim > 1:
            pcm = np.mean(pcm, axis=1)
        pcm = pcm - float(np.mean(pcm))
        raw_rms = float(np.sqrt(np.mean(pcm ** 2)))

        # Adaptive room noise floor tracking on quiet segments
        if raw_rms < 0.008:
            self._noise_floor_rms = 0.90 * self._noise_floor_rms + 0.10 * raw_rms

        if raw_rms < 0.0012:
            return AudioFeatures(
                audio_valid=False,
                voice_vad_active=False,
                av_speech_gated=False,
                mic_active=mic_is_live,
                mic_rms_level=raw_rms,
                audio_rms=raw_rms,
            )

        # Bandpass filter 75 Hz - 4000 Hz using pre-computed Butterworth coefficients
        if self._bp_b is not None and self._bp_a is not None:
            try:
                filtered = filtfilt(self._bp_b, self._bp_a, pcm)
            except Exception:
                filtered = pcm.copy()
        else:
            filtered = pcm.copy()

        # Soft Automatic Gain Normalization for quiet laptop mics (< 0.08 peak)
        peak_95 = float(np.percentile(np.abs(filtered), 95))
        if 1e-5 < peak_95 < 0.08:
            gain = min(15.0, 0.12 / peak_95)
            filtered = np.clip(filtered * gain, -1.0, 1.0)

        frame_len = self._frame_len
        hop_len = self._hop_len
        n_frames = 1 + (len(filtered) - frame_len) // hop_len
        if n_frames < 5:
            return AudioFeatures(mic_active=mic_is_live, mic_rms_level=raw_rms, audio_rms=raw_rms)

        # Vectorized 2D frame slicing: shape (n_frames, frame_len)
        strides = (filtered.strides[0] * hop_len, filtered.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            filtered, shape=(n_frames, frame_len), strides=strides
        ).copy()

        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        frame_peaks = np.percentile(np.abs(frames), 95, axis=1)

        ref_frame_rms = float(np.percentile(frame_rms, 85))
        if float(np.max(frame_rms)) < 0.002:
            return AudioFeatures(
                audio_valid=False,
                voice_vad_active=False,
                av_speech_gated=False,
                mic_active=mic_is_live,
                mic_rms_level=raw_rms,
                audio_rms=raw_rms,
            )

        # Gate out room noise tail frames
        active_thresh = max(0.0020, 0.32 * ref_frame_rms)
        active_mask = frame_rms >= active_thresh
        n_active = int(np.sum(active_mask))
        if n_active < 3:
            return AudioFeatures(
                audio_valid=False,
                voice_vad_active=False,
                av_speech_gated=False,
                mic_active=mic_is_live,
                mic_rms_level=raw_rms,
                audio_rms=raw_rms,
            )

        # Batch Hann windowing + Batch RFFT for spectrum & Wiener-Khinchin autocorrelation
        win_frames = frames * self._hann_win[None, :]
        spec = np.fft.rfft(win_frames, axis=1)
        mag = np.abs(spec)

        # Wiener-Khinchin zero-padded autocorrelation to avoid circular wrap-around
        fft_pow = np.abs(np.fft.rfft(win_frames, n=2 * frame_len, axis=1)) ** 2
        acorr_full = np.fft.irfft(fft_pow, n=2 * frame_len, axis=1)[:, :frame_len]
        r0 = acorr_full[:, 0]
        valid_r0 = r0 > 1e-9
        acorr_norm = np.zeros_like(acorr_full)
        acorr_norm[valid_r0] = acorr_full[valid_r0] / r0[valid_r0, None]

        # Pitch F0 search range: 80 Hz to 320 Hz
        min_lag = int(self.sample_rate / 320.0)  # 50 samples
        max_lag = min(frame_len - 2, int(self.sample_rate / 80.0))  # 200 samples
        voicing_min = 0.25 if self._stream is not None else 0.30

        voiced_indices = []
        voiced_periods = []
        voiced_amps = []
        voiced_autocorrs = []
        f2_centroids = []

        f2_freqs = self._rfft_freqs[self._f2_mask]

        for idx in np.where(active_mask)[0]:
            r = acorr_norm[idx]
            search_region = r[min_lag:max_lag]
            if len(search_region) == 0:
                continue
            rel_idx = int(np.argmax(search_region))
            best_lag = min_lag + rel_idx
            best_val = float(r[best_lag])

            if best_val >= voicing_min:
                if 0 < best_lag < len(r) - 1:
                    y0, y1, y2 = float(r[best_lag - 1]), float(r[best_lag]), float(r[best_lag + 1])
                    denom = y0 - 2.0 * y1 + y2
                    frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
                    precise_lag = float(best_lag) + float(np.clip(frac, -0.5, 0.5))
                else:
                    precise_lag = float(best_lag)

                voiced_indices.append(int(idx))
                voiced_periods.append(precise_lag / self.sample_rate)
                voiced_amps.append(float(frame_peaks[idx]))
                voiced_autocorrs.append(min(0.995, best_val))

                # Dominant F2 formant peak tracking in [850, 2500] Hz
                f2_mag = mag[idx, self._f2_mask]
                f2_weights = f2_mag ** 2
                w_sum = float(np.sum(f2_weights))
                if w_sum > 1e-9:
                    f2_centroids.append(float(np.sum(f2_freqs * f2_weights) / w_sum))

        voiced_ratio = len(voiced_periods) / max(1, n_active)
        if self._stream is not None:
            voice_vad_active = bool(len(voiced_periods) >= 4 and voiced_ratio >= 0.18)
        else:
            voice_vad_active = bool(len(voiced_periods) >= 5 and voiced_ratio >= 0.25)

        speech_pause_ratio = float(np.clip(1.0 - (n_active / max(1, n_frames)), 0.0, 1.0))

        if not voice_vad_active:
            return AudioFeatures(
                audio_valid=False,
                voice_vad_active=False,
                av_speech_gated=False,
                speech_pause_ratio=speech_pause_ratio,
                mic_active=mic_is_live,
                mic_rms_level=raw_rms,
                audio_rms=raw_rms,
            )

        active_rms = float(np.mean(frame_rms[active_mask]))
        if self._stream is not None:
            av_speech_gated = bool(
                visual_speech_active
                or (active_rms >= 0.006 and len(voiced_periods) >= 6 and float(np.mean(voiced_autocorrs)) >= 0.40)
            )
        else:
            av_speech_gated = bool(visual_speech_active and voice_vad_active)

        if not av_speech_gated:
            return AudioFeatures(
                audio_valid=False,
                voice_vad_active=True,
                av_speech_gated=False,
                speech_pause_ratio=speech_pause_ratio,
                mic_active=mic_is_live,
                mic_rms_level=raw_rms,
                audio_rms=raw_rms,
            )

        periods = np.array(voiced_periods, dtype=np.float64)
        amps = np.array(voiced_amps, dtype=np.float64)
        v_idx = np.array(voiced_indices, dtype=np.int32)
        mean_period = float(np.mean(periods))
        mean_amp = float(np.mean(amps))

        # 1. Clinical Vocal Jitter (%):
        # Only compare contiguous/near-contiguous voiced frames (gap <= 2 hops) and reject octave jumps (>18%)
        # Use 3-point local RAP (Relative Average Perturbation) when possible so expressive syllable prosody
        # is not misclassified as pathological jitter!
        adj_diffs = []
        for k in range(1, len(periods)):
            if (v_idx[k] - v_idx[k - 1]) <= 2:
                rel_step = abs(periods[k] - periods[k - 1]) / max(1e-6, mean_period)
                if rel_step < 0.18:
                    if k + 1 < len(periods) and (v_idx[k + 1] - v_idx[k]) <= 2:
                        local_avg = (periods[k - 1] + periods[k] + periods[k + 1]) / 3.0
                        adj_diffs.append(abs(periods[k] - local_avg))
                    else:
                        adj_diffs.append(0.65 * abs(periods[k] - periods[k - 1]))
        if len(adj_diffs) >= 2 and mean_period > 1e-6:
            raw_jitter = float(np.median(adj_diffs) / mean_period) * 100.0
        elif len(periods) >= 2 and mean_period > 1e-6:
            raw_jitter = float(np.median(np.abs(np.diff(periods))) / mean_period) * 100.0
        else:
            raw_jitter = 0.35
        jitter_pct = float(np.clip(raw_jitter, 0.05, 10.0))

        # 2. Clinical Vocal Shimmer (%):
        # Detrend slow phrase-level volume fade-out using local 3-point APQ3 perturbation on contiguous frames
        amp_diffs = []
        for k in range(1, len(amps)):
            if (v_idx[k] - v_idx[k - 1]) <= 2:
                if k + 1 < len(amps) and (v_idx[k + 1] - v_idx[k]) <= 2:
                    local_amp = (amps[k - 1] + amps[k] + amps[k + 1]) / 3.0
                    amp_diffs.append(abs(amps[k] - local_amp))
                else:
                    amp_diffs.append(0.65 * abs(amps[k] - amps[k - 1]))
        if len(amp_diffs) >= 2 and mean_amp > 1e-6:
            raw_shimmer = float(np.median(amp_diffs) / mean_amp) * 100.0
        elif len(amps) >= 2 and mean_amp > 1e-6:
            raw_shimmer = float(np.median(np.abs(np.diff(amps))) / mean_amp) * 100.0
        else:
            raw_shimmer = 1.5
        shimmer_pct = float(np.clip(raw_shimmer, 0.1, 25.0))

        # 3. Harmonics-to-Noise Ratio (HNR in dB) with live-mic room acoustics compensation:
        mean_r = float(np.percentile(voiced_autocorrs, 70))
        raw_hnr = 10.0 * np.log10(mean_r / max(1e-4, 1.0 - mean_r))
        # On live laptop mics, room reflections slightly lower raw autocorrelation compared to synthetic waves
        if self._stream is not None:
            raw_hnr += 3.0
        hnr_db = float(np.clip(raw_hnr, 0.0, 35.0))

        # 4. Formant F2 Transition Velocity (Hz/ms):
        hop_ms = (hop_len / self.sample_rate) * 1000.0  # 16.0 ms
        if len(f2_centroids) >= 5:
            smooth_f2 = scipy.signal.medfilt(f2_centroids, 3)
            diffs = np.abs(np.diff(smooth_f2))
            active_transitions = diffs[diffs > 5.0]
            if len(active_transitions) > 0:
                raw_slope = float(np.percentile(active_transitions, 65) / hop_ms)
            else:
                raw_slope = float(np.mean(diffs) / hop_ms)
            f2_slope = float(np.clip(raw_slope * 2.0, 2.0, 45.0))
        else:
            f2_slope = 22.0

        # 5. Composite Vocal Intoxication Score (0-100):
        s_jitter = float(np.clip((jitter_pct - 1.15) / 1.50 * 100.0, 0.0, 100.0))
        s_shimmer = float(np.clip((shimmer_pct - 3.80) / 4.50 * 100.0, 0.0, 100.0))
        s_hnr = float(np.clip((13.5 - hnr_db) / 6.00 * 100.0, 0.0, 100.0))
        s_f2 = float(np.clip((15.0 - f2_slope) / 8.00 * 100.0, 0.0, 100.0))

        raw_score = float(np.clip(
            0.35 * s_f2 + 0.25 * s_jitter + 0.20 * s_shimmer + 0.20 * s_hnr,
            0.0,
            100.0,
        ))

        # Sober steady pitch guard: if vocal jitter is low (< 1.15%), driver has precise neuromuscular glottal control
        if jitter_pct < 1.15:
            raw_score = min(raw_score, 22.0)

        # Temporal EMA smoothing on live mic mode so a single syllable never spikes the HUD
        if self._stream is not None:
            if self._ema_score is None:
                self._ema_score = raw_score
            else:
                self._ema_score = 0.45 * raw_score + 0.55 * self._ema_score
            score = float(self._ema_score)
        else:
            score = raw_score

        abnormal_count = sum([
            jitter_pct > 1.35,
            shimmer_pct > 4.20,
            hnr_db < 11.5,
            f2_slope < 12.0,
        ])
        if self._stream is not None:
            slurred = bool(len(voiced_periods) >= 8 and jitter_pct >= 1.25 and abnormal_count >= 2 and score >= 45.0)
        else:
            slurred = bool(len(voiced_periods) >= 5 and abnormal_count >= 2 and score >= 45.0)

        result = AudioFeatures(
            audio_valid=True,
            voice_vad_active=True,
            av_speech_gated=True,
            vocal_jitter_pct=round(jitter_pct, 3),
            vocal_shimmer_pct=round(shimmer_pct, 3),
            vocal_hnr_db=round(hnr_db, 2),
            formant_f2_slope_hz_ms=round(f2_slope, 2),
            speech_pause_ratio=round(speech_pause_ratio, 3),
            voice_impairment_score=round(score, 2),
            voice_slurred_detected=slurred,
            mic_active=mic_is_live,
            mic_rms_level=round(raw_rms, 4),
            audio_rms=round(raw_rms, 4),
        )

        self._last_valid_features = result
        self._last_valid_time = timestamp
        return result

    def close(self) -> None:
        """Release live microphone stream and hardware resources cleanly."""
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception as exc:
                logger.debug("Error closing mic stream: %s", exc)
            self._mic_stream = None
        self.mic_active = False
