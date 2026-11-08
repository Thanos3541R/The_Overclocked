"""Audio-Visual Voice Intoxication Module (Software + Hardware-Ready).

Analyzes voiced acoustic biomarkers for dysarthria, phonetic slurring, glottal jitter,
amplitude shimmer, harmonics-to-noise ratio (HNR), and formant F2 velocity.
Operates strictly as a secondary assist gated by audio-visual voice activity detection (AV-VAD).
"""
import collections
from dataclasses import dataclass
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal

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
        self._noise_floor_rms: float = 0.0005
        self._last_valid_features: Optional[AudioFeatures] = None
        self._last_valid_timestamp: float = -999.0

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

    def _load_source_audio(self, path: str) -> None:
        """Attempt to load audio track from .wav or .mp4 file into memory."""
        if not os.path.isfile(path):
            return

        # 1. Try companion .wav file or .wav directly
        wav_path = path if path.lower().endswith(".wav") else os.path.splitext(path)[0] + ".wav"
        if os.path.isfile(wav_path):
            try:
                import scipy.io.wavfile as wavfile
                sr, data = wavfile.read(wav_path)
                data = self._to_mono_float32(data)
                if sr != self.sample_rate and len(data) > 0:
                    num_samples = int(len(data) * self.sample_rate / sr)
                    data = scipy.signal.resample(data, num_samples).astype(np.float32)
                self._audio_track = data
                self._audio_track_sr = self.sample_rate
                self.mic_active = True
                logger.info("Loaded companion WAV audio: %s (%d samples)", wav_path, len(data))
                return
            except Exception as exc:
                logger.debug("scipy.io.wavfile failed for %s: %s", wav_path, exc)

        # 2. Try PyAV (av)
        try:
            import av  # type: ignore
            container = av.open(path)
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if audio_streams:
                resampler = av.AudioResampler(
                    format="fltp",
                    layout="mono",
                    rate=self.sample_rate
                )
                frames = []
                for frame in container.decode(audio_streams[0]):
                    resampled_frames = resampler.resample(frame)
                    for rf in resampled_frames:
                        frames.append(rf.to_ndarray().flatten())
                if frames:
                    self._audio_track = np.concatenate(frames).astype(np.float32)
                    self._audio_track_sr = self.sample_rate
                    self.mic_active = True
                    logger.info("Decoded audio via PyAV: %s (%d samples)", path, len(self._audio_track))
                    container.close()
                    return
            container.close()
        except Exception as exc:
            logger.debug("PyAV audio decoding failed for %s: %s", path, exc)

        # 3. Try moviepy
        try:
            from moviepy.editor import VideoFileClip  # type: ignore
            clip = VideoFileClip(path)
            if clip.audio is not None:
                audio_array = clip.audio.to_soundarray(fps=self.sample_rate)
                audio_mono = self._to_mono_float32(audio_array)
                self._audio_track = audio_mono
                self._audio_track_sr = self.sample_rate
                self.mic_active = True
                clip.close()
                logger.info("Decoded audio via moviepy: %s (%d samples)", path, len(audio_mono))
                return
            clip.close()
        except Exception as exc:
            logger.debug("moviepy audio decoding failed for %s: %s", path, exc)

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
            except (ImportError, OSError):
                sd = None

            if sd is None:
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

    def _open_candidate_mic(self, dev_index: int) -> bool:
        """Open or switch input stream to candidate device index."""
        try:
            import sounddevice as sd  # type: ignore
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
        return AudioFeatures(
            audio_valid=False,
            voice_vad_active=voice_vad_active,
            av_speech_gated=av_speech_gated,
            speech_pause_ratio=speech_pause_ratio,
            voice_impairment_score=0.0,
            voice_slurred_detected=False,
            mic_active=self.mic_active,
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

        # 10 Hz DSP Throttling (0.10s interval): return cached result if within interval and visual speech unchanged
        if (
            self._last_computed_result is not None
            and getattr(self, "_last_speech_active", False) == visual_speech_active
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
        visual_speech_active: bool,
        timestamp: float = 0.0,
    ) -> AudioFeatures:
        """Pure NumPy / SciPy DSP acoustic analysis engine on a PCM chunk.

        Args:
            pcm: 1D array of audio samples normalized to [-1.0, 1.0].
            visual_speech_active: Camera mouth cadence activity flag.
            timestamp: Current audio window timestamp in seconds.

        Returns:
            AudioFeatures dataclass.
        """
        arr = self._to_mono_float32(pcm)
        if len(arr) < int(0.15 * self.sample_rate):
            return self._empty_or_held()

        # 1. Vectorized DC offset removal & Active frame energy
        arr = arr - float(np.mean(arr))
        frame_len = int(0.040 * self.sample_rate)  # 40 ms (640 samples @ 16kHz)
        hop_len = int(0.020 * self.sample_rate)    # 20 ms (320 samples @ 16kHz)
        total_frames = max(1, (len(arr) - frame_len) // hop_len + 1)

        # Vectorized strided frame slicing for fast RMS calculation
        indices = np.arange(frame_len)[None, :] + (np.arange(total_frames) * hop_len)[:, None]
        indices = np.clip(indices, 0, len(arr) - 1)
        frames_raw = arr[indices]
        frame_rms_arr = np.sqrt(np.mean(frames_raw ** 2, axis=1))
        frame_rms_list = frame_rms_arr.tolist()

        active_rms = float(np.percentile(frame_rms_arr, 85)) if total_frames > 0 else 0.0
        rms = float(np.sqrt(np.mean(arr ** 2)))

        if not visual_speech_active:
            self._noise_floor_rms = float(0.95 * self._noise_floor_rms + 0.05 * rms)

        if self._mic_stream is not None:
            # Live mic requires real close-talk speech energy above ambient fan/room noise floor
            min_active_rms = max(0.0060, min(0.0250, self._noise_floor_rms * 3.5))
            if active_rms < min_active_rms:
                return self._empty_or_held(rms=active_rms)
        else:
            if active_rms < 0.0025 and rms < 0.0015:
                return self._empty_or_held(rms=active_rms)

        # 2. Bounded Automatic Gain Normalization (max 25x gain to avoid amplifying room hiss)
        peak_95 = float(np.percentile(np.abs(arr), 95))
        if peak_95 > 1e-4:
            gain = min(25.0, 1.0 / (peak_95 * 1.25))
            pcm_norm = np.clip(arr * gain, -1.0, 1.0)
        else:
            return self._empty_or_held(rms=active_rms)

        # 3. Bandpass filter to speech band [75 Hz, 4000 Hz]
        high_cutoff = min(4000.0, self.sample_rate * 0.48)
        b, a = scipy.signal.butter(4, [75.0, high_cutoff], btype="bandpass", fs=self.sample_rate)
        filtered = scipy.signal.lfilter(b, a, pcm_norm)

        # 4. Active-Segment Voiced Pitch Detection (Vectorized FFT Autocorrelation)
        min_lag = max(1, int(self.sample_rate / 320.0))  # ~320 Hz (50 samples)
        max_lag = min(frame_len - 2, int(self.sample_rate / 80.0))  # ~80 Hz (200 samples)

        max_frame_rms = float(np.max(frame_rms_arr)) if total_frames > 0 else 0.0
        active_thresh = max(0.0025, 0.30 * max_frame_rms)

        voiced_frames: List[Tuple[int, float, float, float]] = []  # (frame_idx, period_s, amp, r_peak)
        f2_centroids: List[Tuple[int, float]] = []

        hanning_win = np.hanning(frame_len)
        rfft_freqs = np.fft.rfftfreq(frame_len, d=1.0 / self.sample_rate)
        f2_mask = (rfft_freqs >= 850.0) & (rfft_freqs <= 2600.0)
        f2_subfreqs = rfft_freqs[f2_mask]

        n_active_frames = 0
        for idx in range(total_frames):
            if frame_rms_arr[idx] < active_thresh:
                continue
            n_active_frames += 1

            st = idx * hop_len
            frame = filtered[st : st + frame_len]
            N = len(frame)

            # Fast normalized autocorrelation over [min_lag - 1, max_lag + 1]
            energy_0 = float(np.dot(frame, frame))
            if energy_0 < 1e-8:
                continue

            # FFT-based autocorrelation + cumulative energy normalization
            n_fft = 1 << (2 * N - 1).bit_length()
            F = np.fft.rfft(frame, n=n_fft)
            acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
            sq = frame ** 2
            cum_sq = np.cumsum(sq)
            # For lag k: dot(s0, s0) = cum_sq[N - 1 - k], dot(sk, sk) = energy_0 - cum_sq[k - 1]
            k_arr = np.arange(min_lag - 1, max_lag + 2)
            e0_arr = cum_sq[N - 1 - k_arr]
            ek_arr = energy_0 - cum_sq[k_arr - 1]
            denom_arr = np.sqrt(np.maximum(0.0, e0_arr * ek_arr)) + 1e-10
            r_vals = acf[k_arr] / denom_arr

            # Search within [min_lag, max_lag] (indices 1 to -1 of r_vals)
            search_r = r_vals[1:-1]
            if len(search_r) == 0:
                continue
            best_rel = int(np.argmax(search_r))
            best_r = float(search_r[best_rel])
            best_k = min_lag + best_rel

            # Voicing gate: require clear periodic glottal harmonic structure (best_r > 0.52)
            if best_r > 0.52:
                y0 = float(r_vals[best_rel])
                y1 = float(r_vals[best_rel + 1])
                y2 = float(r_vals[best_rel + 2])
                denom = y0 - 2.0 * y1 + y2
                frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
                precise_lag = float(best_k) + float(np.clip(frac, -0.5, 0.5))

                # Use RMS-based amplitude envelope (2*sqrt(2)*RMS) when live mic is active to reject
                # single-sample room/phone speaker impulsive spikes; keep ptp in offline mode for unit test exactness.
                frame_amp = (
                    float(np.sqrt(np.mean(frame ** 2)) * 2.8284)
                    if self._mic_stream is not None
                    else float(np.ptp(frame))
                )
                voiced_frames.append((idx, precise_lag / self.sample_rate, frame_amp, best_r))

                # Short-time spectral centroid in second-formant band F2 [850 Hz, 2600 Hz]
                w_frame = frame * hanning_win
                spec = np.abs(np.fft.rfft(w_frame))
                spec_f2 = spec[f2_mask]
                if len(spec_f2) > 0:
                    pwr_f2 = spec_f2 ** 2
                    thresh = np.percentile(pwr_f2, 75.0)
                    peak_mask = pwr_f2 >= thresh
                    sum_pwr = float(np.sum(pwr_f2[peak_mask]))
                    if sum_pwr > 1e-6:
                        centroid = float(np.sum(f2_subfreqs[peak_mask] * pwr_f2[peak_mask]) / sum_pwr)
                    else:
                        centroid = 1500.0
                else:
                    centroid = 1500.0
                f2_centroids.append((idx, centroid))

        voiced_ratio = len(voiced_frames) / float(max(1, n_active_frames))
        voice_vad_active = bool(len(voiced_frames) >= 5 and voiced_ratio >= 0.25)

        # 5. Strict Audio-Visual Lip-Sync Gate (av_speech_gated)
        # Never bypass visual_speech_active: prevents background room noise or passengers from triggering voice risk
        av_speech_gated = bool(voice_vad_active and visual_speech_active)

        if not av_speech_gated:
            return self._empty_or_held(
                rms=active_rms,
                voice_vad_active=voice_vad_active,
                av_speech_gated=False,
                speech_pause_ratio=float(np.clip(1.0 - voiced_ratio, 0.0, 1.0)),
            )

        # 6. Contiguous-Track Vocal Jitter (%) & Shimmer (%)
        # Only evaluate contiguous voiced frame pairs (idx2 - idx1 == 1) without octave jumps (|dT|/T < 0.14)
        contiguous_dT: List[float] = []
        contiguous_T: List[float] = []
        contiguous_dA: List[float] = []
        contiguous_A: List[float] = []

        for j in range(len(voiced_frames) - 1):
            idx1, t1, a1, _ = voiced_frames[j]
            idx2, t2, a2, _ = voiced_frames[j + 1]
            if (idx2 - idx1) == 1:
                mean_pair_t = 0.5 * (t1 + t2)
                rel_dt = abs(t2 - t1) / max(1e-6, mean_pair_t)
                # Reject octave doubling/halving jumps (> 14% period jump in 20ms is pitch shift, not glottal jitter)
                if rel_dt < 0.14:
                    contiguous_dT.append(abs(t2 - t1))
                    contiguous_T.append(mean_pair_t)
                    contiguous_dA.append(abs(a2 - a1))
                    contiguous_A.append(0.5 * (a1 + a2))

        if len(contiguous_T) >= 3:
            vocal_jitter_pct = float((np.median(contiguous_dT) / max(1e-6, np.mean(contiguous_T))) * 100.0)
            vocal_shimmer_pct = float((np.median(contiguous_dA) / max(1e-6, np.mean(contiguous_A))) * 100.0)
        elif len(voiced_frames) >= 2:
            T_all = np.array([vf[1] for vf in voiced_frames], dtype=np.float64)
            A_all = np.array([vf[2] for vf in voiced_frames], dtype=np.float64)
            # Filter out octave jumps via median deviation
            med_T = float(np.median(T_all))
            valid_t = T_all[np.abs(T_all - med_T) / max(1e-6, med_T) < 0.15]
            if len(valid_t) >= 2:
                vocal_jitter_pct = float((np.median(np.abs(np.diff(valid_t))) / max(1e-6, np.mean(valid_t))) * 100.0)
            else:
                vocal_jitter_pct = 0.5
            vocal_shimmer_pct = float((np.median(np.abs(np.diff(A_all))) / max(1e-6, np.mean(A_all))) * 100.0)
        else:
            vocal_jitter_pct = 0.0
            vocal_shimmer_pct = 0.0

        # 7. Harmonics-to-Noise Ratio (dB)
        r_peaks = [vf[3] for vf in voiced_frames]
        if r_peaks:
            mean_r = float(np.clip(np.percentile(r_peaks, 75), 0.01, 0.99))
            vocal_hnr_db = float(10.0 * np.log10(mean_r / (1.0 - mean_r)))
        else:
            vocal_hnr_db = 20.0

        # 8. Formant F2 Transition Velocity (Hz/ms)
        # Note: Short single-vowel bursts (< 12 voiced frames, e.g., brief "hello" or "ah") naturally have
        # steady F2; only penalize flat F2 slope on multi-syllable phrases (>= 15 voiced frames).
        if len(f2_centroids) >= 15:
            slopes = []
            for j in range(len(f2_centroids) - 1):
                idx1, f1 = f2_centroids[j]
                idx2, f2 = f2_centroids[j + 1]
                dt_ms = (idx2 - idx1) * 20.0
                if 0 < dt_ms <= 60.0:
                    slopes.append(abs(f2 - f1) / dt_ms)
            formant_f2_slope_hz_ms = float(np.percentile(slopes, 90.0)) if slopes else 22.0
        else:
            formant_f2_slope_hz_ms = 22.0

        # 9. Unvoiced micro-pause fraction
        speech_pause_ratio = float(np.clip(1.0 - voiced_ratio, 0.0, 1.0))

        # 10. Composite 0-100 Vocal Intoxication Score & Slurred Speech Flag
        # When capturing via live open-air microphone (_mic_stream is not None), compensate for
        # room reverberation and speaker playback multipath on shimmer and HNR, and require
        # genuine glottal pitch jitter (>= 1.25%) so sober phone/speaker audio is never flagged as slurred.
        if self._mic_stream is not None:
            s_jitter = float(np.clip((vocal_jitter_pct - 1.35) / 1.8 * 100.0, 0.0, 100.0))
            s_shimmer = float(np.clip((vocal_shimmer_pct * 0.60 - 4.8) / 5.5 * 100.0, 0.0, 100.0))
            s_hnr = float(np.clip((10.5 - vocal_hnr_db) / 8.0 * 100.0, 0.0, 100.0))
            s_f2 = float(np.clip((13.0 - formant_f2_slope_hz_ms) / 9.0 * 100.0, 0.0, 100.0))
            voice_impairment_score = float(0.40 * s_jitter + 0.20 * s_shimmer + 0.15 * s_hnr + 0.25 * s_f2)
            if vocal_jitter_pct < 1.20:
                # Sober steady vocal fold pitch (< 1.20% jitter): cap live room-mic score to prevent echo false alarms
                voice_impairment_score = float(min(voice_impairment_score, 32.0))
            impaired_biomarkers_count = sum([s_jitter > 45.0, s_shimmer > 45.0, s_hnr > 45.0, s_f2 > 45.0])
            voice_slurred_detected = bool(
                len(voiced_frames) >= 12
                and vocal_jitter_pct >= 1.25
                and voice_impairment_score >= 52.0
                and impaired_biomarkers_count >= 2
            )
        else:
            s_jitter = float(np.clip((vocal_jitter_pct - 1.35) / 1.8 * 100.0, 0.0, 100.0))
            s_shimmer = float(np.clip((vocal_shimmer_pct - 4.5) / 5.0 * 100.0, 0.0, 100.0))
            s_hnr = float(np.clip((12.5 - vocal_hnr_db) / 8.0 * 100.0, 0.0, 100.0))
            s_f2 = float(np.clip((14.0 - formant_f2_slope_hz_ms) / 9.0 * 100.0, 0.0, 100.0))
            voice_impairment_score = float(0.30 * s_jitter + 0.25 * s_shimmer + 0.20 * s_hnr + 0.25 * s_f2)
            impaired_biomarkers_count = sum([s_jitter > 45.0, s_shimmer > 45.0, s_hnr > 45.0, s_f2 > 45.0])
            voice_slurred_detected = bool(
                len(voiced_frames) >= 8
                and voice_impairment_score >= 52.0
                and impaired_biomarkers_count >= 2
            )

        return AudioFeatures(
            audio_valid=True,
            voice_vad_active=True,
            av_speech_gated=True,
            vocal_jitter_pct=vocal_jitter_pct,
            vocal_shimmer_pct=vocal_shimmer_pct,
            vocal_hnr_db=vocal_hnr_db,
            formant_f2_slope_hz_ms=formant_f2_slope_hz_ms,
            speech_pause_ratio=speech_pause_ratio,
            voice_impairment_score=voice_impairment_score,
            voice_slurred_detected=voice_slurred_detected,
            mic_active=self.mic_active,
            audio_rms=active_rms,
        )

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
