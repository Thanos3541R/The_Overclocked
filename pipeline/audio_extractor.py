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
    ):
        self.sample_rate = int(sample_rate)
        self.window_sec = float(window_sec)
        self.source_path = source_path
        self.enable_live_mic = enable_live_mic
        self.hardware_serial_mode = hardware_serial_mode

        self._buffer_len = int(self.sample_rate * self.window_sec)
        self._pcm_buffer: collections.deque = collections.deque(maxlen=self._buffer_len)
        self._lock = threading.Lock()

        self._audio_track: Optional[np.ndarray] = None
        self._audio_track_sr: int = self.sample_rate
        self._mic_stream: Optional[Any] = None

        self._injected_telemetry: Optional[Dict[str, Any]] = None
        self._last_features: AudioFeatures = AudioFeatures()
        self._last_valid_time: float = -999.0

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
        """Initialize non-blocking sounddevice microphone input stream."""
        try:
            import sounddevice as sd  # type: ignore

            def _mic_cb(indata, frames, time_info, status):
                if indata is not None and len(indata) > 0:
                    self.push_pcm_chunk(indata[:, 0])

            self._mic_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=_mic_cb,
            )
            self._mic_stream.start()
            logger.info("Live microphone input stream started @ %d Hz", self.sample_rate)
        except Exception as exc:
            logger.warning("Could not open live mic stream (sounddevice): %s", exc)
            self._mic_stream = None

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
                features = AudioFeatures(
                    audio_valid=False,
                    voice_vad_active=vad_active,
                    av_speech_gated=False,
                    voice_impairment_score=0.0,
                    voice_slurred_detected=False,
                )
                self._last_features = features
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
            )
            self._last_features = features
            return features

        # Mode A: Synchronized Video/Audio Track
        if self._audio_track is not None and len(self._audio_track) > 0:
            window_samples = int(self.sample_rate * self.window_sec)
            end_sample = max(0, int(timestamp * self.sample_rate))
            start_sample = max(0, end_sample - window_samples)
            if end_sample <= len(self._audio_track):
                pcm = self._audio_track[start_sample:end_sample]
            else:
                pcm = self._audio_track[start_sample:]
            return self.analyze_pcm_window(pcm, visual_speech_active=visual_speech_active)

        # Mode B: Live Ring Buffer
        with self._lock:
            buf_len = len(self._pcm_buffer)
            if buf_len >= int(0.25 * self.sample_rate):
                pcm = np.array(self._pcm_buffer, dtype=np.float32)
            else:
                pcm = np.array([], dtype=np.float32)

        if len(pcm) >= int(0.25 * self.sample_rate):
            return self.analyze_pcm_window(pcm, visual_speech_active=visual_speech_active)

        return AudioFeatures(
            audio_valid=False,
            voice_vad_active=False,
            av_speech_gated=False,
            voice_impairment_score=0.0,
            voice_slurred_detected=False,
        )

    def analyze_pcm_window(self, pcm: np.ndarray, visual_speech_active: bool) -> AudioFeatures:
        """Pure NumPy / SciPy DSP acoustic analysis engine on a PCM chunk.

        Args:
            pcm: 1D array of audio samples normalized to [-1.0, 1.0].
            visual_speech_active: Camera mouth cadence activity flag.

        Returns:
            AudioFeatures dataclass.
        """
        arr = self._to_mono_float32(pcm)
        if len(arr) < int(0.15 * self.sample_rate):
            return AudioFeatures(
                audio_valid=False,
                voice_vad_active=False,
                av_speech_gated=False,
                voice_impairment_score=0.0,
                voice_slurred_detected=False,
            )

        # 1. RMS Energy Check
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if rms <= 0.012:
            return AudioFeatures(
                audio_valid=False,
                voice_vad_active=False,
                av_speech_gated=False,
                voice_impairment_score=0.0,
                voice_slurred_detected=False,
            )

        # 2. Bandpass filter to speech band [75 Hz, 4000 Hz]
        high_cutoff = min(4000.0, self.sample_rate * 0.48)
        b, a = scipy.signal.butter(4, [75.0, high_cutoff], btype="bandpass", fs=self.sample_rate)
        filtered = scipy.signal.lfilter(b, a, arr)

        # 3. Overlapping 40 ms frames with 20 ms hop
        frame_len = int(0.040 * self.sample_rate)
        hop_len = int(0.020 * self.sample_rate)
        min_lag = max(1, int(self.sample_rate / 320.0))  # ~320 Hz
        max_lag = min(frame_len - 1, int(self.sample_rate / 80.0))  # ~80 Hz

        total_frames = max(1, (len(filtered) - frame_len) // hop_len + 1)
        pitch_periods: List[float] = []
        amplitudes: List[float] = []
        r_peaks: List[float] = []
        f2_centroids: List[Tuple[int, float]] = []

        hanning_win = np.hanning(frame_len)
        rfft_freqs = np.fft.rfftfreq(frame_len, d=1.0 / self.sample_rate)
        f2_mask = (rfft_freqs >= 850.0) & (rfft_freqs <= 2600.0)
        f2_subfreqs = rfft_freqs[f2_mask]

        for idx in range(total_frames):
            st = idx * hop_len
            frame = filtered[st : st + frame_len]
            f_rms = np.sqrt(np.mean(frame ** 2))
            if f_rms < 0.010:
                continue

            # Normalized autocorrelation over [min_lag, max_lag]
            best_r = -1.0
            best_k = min_lag
            r_lags: Dict[int, float] = {}

            N = len(frame)
            for k in range(min_lag - 1, max_lag + 2):
                if k < 1 or k >= N:
                    continue
                s0 = frame[: N - k]
                sk = frame[k:]
                denom = np.sqrt(np.dot(s0, s0) * np.dot(sk, sk)) + 1e-10
                r_val = float(np.dot(s0, sk) / denom)
                r_lags[k] = r_val
                if min_lag <= k <= max_lag and r_val > best_r:
                    best_r = r_val
                    best_k = k

            if best_r > 0.35:
                # Sub-sample parabolic interpolation around peak
                k_star = float(best_k)
                if (best_k - 1) in r_lags and (best_k + 1) in r_lags:
                    y0 = r_lags[best_k - 1]
                    y1 = r_lags[best_k]
                    y2 = r_lags[best_k + 1]
                    denom_p = 2.0 * (2.0 * y1 - y0 - y2)
                    if abs(denom_p) > 1e-6:
                        delta = (y2 - y0) / denom_p
                        if abs(delta) < 1.0:
                            k_star += delta

                pitch_periods.append(k_star / self.sample_rate)
                amplitudes.append(float(np.ptp(frame)))
                r_peaks.append(best_r)

                # Short-time spectral centroid in second-formant band F2 [850 Hz, 2600 Hz]
                w_frame = frame * hanning_win
                spec = np.abs(np.fft.rfft(w_frame))
                spec_f2 = spec[f2_mask]
                sum_spec = np.sum(spec_f2)
                if sum_spec > 1e-6:
                    centroid = float(np.sum(f2_subfreqs * spec_f2) / sum_spec)
                else:
                    centroid = 1500.0
                f2_centroids.append((idx, centroid))

        voiced_ratio = len(pitch_periods) / float(max(1, total_frames))
        voice_vad_active = bool(voiced_ratio >= 0.35)

        # Strict Lip-Sync Gate: If mouth is closed or audio is unvoiced, reject audio
        if not visual_speech_active or not voice_vad_active:
            return AudioFeatures(
                audio_valid=False,
                voice_vad_active=voice_vad_active,
                av_speech_gated=False,
                speech_pause_ratio=float(np.clip(1.0 - voiced_ratio, 0.0, 1.0)),
                voice_impairment_score=0.0,
                voice_slurred_detected=False,
            )

        # 4. Vocal Jitter (%)
        N_v = len(pitch_periods)
        if N_v >= 2:
            T = np.array(pitch_periods, dtype=np.float64)
            mean_T = float(np.mean(T))
            diff_T = float(np.mean(np.abs(np.diff(T))))
            vocal_jitter_pct = float((diff_T / max(1e-6, mean_T)) * 100.0)

            A = np.array(amplitudes, dtype=np.float64)
            mean_A = float(np.mean(A))
            diff_A = float(np.mean(np.abs(np.diff(A))))
            vocal_shimmer_pct = float((diff_A / max(1e-6, mean_A)) * 100.0)
        else:
            vocal_jitter_pct = 0.0
            vocal_shimmer_pct = 0.0

        # 5. Harmonics-to-Noise Ratio (dB)
        if r_peaks:
            mean_r = float(np.clip(np.mean(r_peaks), 0.01, 0.99))
            vocal_hnr_db = float(10.0 * np.log10(mean_r / (1.0 - mean_r)))
        else:
            vocal_hnr_db = 20.0

        # 6. Formant F2 Transition Velocity (Hz/ms)
        if len(f2_centroids) >= 2:
            slopes = []
            for j in range(len(f2_centroids) - 1):
                idx1, f1 = f2_centroids[j]
                idx2, f2 = f2_centroids[j + 1]
                dt_ms = (idx2 - idx1) * 20.0
                if dt_ms > 0:
                    slopes.append(abs(f2 - f1) / dt_ms)
            formant_f2_slope_hz_ms = float(np.mean(slopes)) if slopes else 25.0
        else:
            formant_f2_slope_hz_ms = 25.0

        # 7. Unvoiced micro-pause fraction
        speech_pause_ratio = float(np.clip(1.0 - voiced_ratio, 0.0, 1.0))

        # 8. Composite 0-100 Vocal Intoxication Score & Slurred Speech Flag
        s_jitter = float(np.clip((vocal_jitter_pct - 1.0) / 1.5 * 100.0, 0.0, 100.0))
        s_shimmer = float(np.clip((vocal_shimmer_pct - 3.8) / 4.2 * 100.0, 0.0, 100.0))
        s_hnr = float(np.clip((14.0 - vocal_hnr_db) / 8.0 * 100.0, 0.0, 100.0))
        s_f2 = float(np.clip((16.0 - formant_f2_slope_hz_ms) / 10.0 * 100.0, 0.0, 100.0))

        voice_impairment_score = float(0.30 * s_jitter + 0.25 * s_shimmer + 0.20 * s_hnr + 0.25 * s_f2)
        impaired_biomarkers_count = sum([s_jitter > 35.0, s_shimmer > 35.0, s_hnr > 35.0, s_f2 > 35.0])
        voice_slurred_detected = bool(voice_impairment_score >= 45.0 and impaired_biomarkers_count >= 2)

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
