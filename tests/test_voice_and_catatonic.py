"""Unit and integration tests for the Audio-Visual Voice Intoxication Module and Catatonic Stupor Detection.

Verifies:
1. AV-VAD lip-sync gating: Silent drivers never trigger voice impairment even in presence of loud audio.
2. Acoustic DSP analysis: Sober steady speech vs. intoxicated dysarthric/slurred speech.
3. Hardware edge telemetry injection: Direct ESP32 + INMP441 I2S serial hook.
4. Jesse Catatonic Stupor detection: Silent, unblinking fixed stare bypassing open-eye desktop guard.
5. Bayesian secondary voice fusion: Slurred speech accelerates convergence while sober visual stream remains immune.
"""
import numpy as np
import pytest

from pipeline.audio_extractor import AudioFeatures, AudioIntoxicationExtractor
from pipeline.bayesian_filter import BayesianEvidenceAccumulator
from pipeline.classifier import ImpairmentClassifier, ImpairmentAssessment
from pipeline.feature_extractor import FrameFeatures


def test_silent_frames_never_trigger_voice_impairment():
    """Feed noisy audio PCM while visual_speech_active=False.

    Simulates passenger talking or car radio playing while driver's mouth is shut.
    Verify audio_feat.av_speech_gated is False, audio_feat.audio_valid is False,
    and voice_impairment_score == 0.0.
    """
    extractor = AudioIntoxicationExtractor(sample_rate=16000, window_sec=1.0)
    # Loud audio with high energy and noise
    noisy_pcm = np.random.normal(0, 0.4, 16000)

    audio_feat = extractor.analyze_pcm_window(noisy_pcm, visual_speech_active=False)

    assert not audio_feat.av_speech_gated, "AV-VAD must reject audio when visual_speech_active is False"
    assert not audio_feat.audio_valid, "audio_valid must be False when mouth is closed"
    assert audio_feat.voice_impairment_score == 0.0, "Silent driver must receive 0.0 voice impairment score"
    assert not audio_feat.voice_slurred_detected, "No slurred speech can be detected during silence"


def test_sober_vs_slurred_intoxicated_voice_dsp():
    """Synthesize clean sober speech vs. intoxicated dysarthric speech.

    Verify sober waveform yields voice_impairment_score < 25.0 and voice_slurred_detected is False.
    Verify dysarthric waveform yields voice_impairment_score >= 55.0 and voice_slurred_detected is True.
    """
    extractor = AudioIntoxicationExtractor(sample_rate=16000, window_sec=1.0)
    sr = 16000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr

    # 1. Clean Sober Vowel/Formant Waveform (steady F0 = 130 Hz, harmonic series + fast syllable F2 sweeps)
    base = 0.8 * np.sin(2 * np.pi * 130.0 * t) + 0.4 * np.sin(2 * np.pi * 260.0 * t)
    f2_freqs = np.zeros_like(t)
    for i in range(10):  # 10 syllables of 100ms
        st = int(i * 0.100 * sr)
        en = int((i + 1) * 0.100 * sr)
        trans_len = int(0.040 * sr)
        f2_freqs[st : st + trans_len] = np.linspace(1100.0, 2000.0, trans_len)
        f2_freqs[st + trans_len : en] = 2000.0

    f2_phase = 2 * np.pi * np.cumsum(f2_freqs) / sr
    formant = 0.10 * np.sin(f2_phase)
    sober_pcm = base + formant

    sober_feat = extractor.analyze_pcm_window(sober_pcm, visual_speech_active=True)
    assert sober_feat.audio_valid is True
    assert sober_feat.voice_impairment_score < 25.0, f"Expected sober score < 25.0, got {sober_feat.voice_impairment_score:.1f}"
    assert not sober_feat.voice_slurred_detected, "Sober speech must not be flagged as slurred"

    # 2. Intoxicated Dysarthric Waveform (F0 jitter 2.2%, shimmer 6.5%, additive breath noise HNR < 9 dB, flat F2)
    np.random.seed(42)
    t_accum = 0.0
    cycles = []
    amp = 0.8
    while t_accum < duration:
        period = (1.0 / 130.0) * (1.0 + np.random.uniform(-0.15, 0.15))  # cycle-to-cycle jitter
        amp = np.clip(amp + np.random.uniform(-0.35, 0.35), 0.25, 1.0)     # shimmer
        n_samples = max(2, int(period * sr))
        t_c = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        # Vocal tract impulse response with flat F2 resonance around 1400 Hz
        f2_res = 0.45 * np.sin(2 * np.pi * 1400.0 * (t_c / (2 * np.pi * 130.0))) * np.exp(-t_c * 0.8)
        cycles.append(amp * (np.sin(t_c) + f2_res))
        t_accum += len(cycles[-1]) / sr

    pcm_intox = np.concatenate(cycles)[: int(sr * duration)]
    pcm_intox += np.random.normal(0, 0.20, len(pcm_intox))  # additive breath noise
    pcm_intox = pcm_intox / np.max(np.abs(pcm_intox))

    impaired_feat = extractor.analyze_pcm_window(pcm_intox, visual_speech_active=True)
    assert impaired_feat.audio_valid is True
    assert impaired_feat.voice_impairment_score >= 55.0, f"Expected dysarthric score >= 55.0, got {impaired_feat.voice_impairment_score:.1f}"
    assert impaired_feat.voice_slurred_detected is True, "Impaired dysarthric speech must be flagged as slurred"


def test_hardware_i2s_telemetry_injection_hook():
    """Call push_hardware_telemetry simulating external ESP32 + INMP441 I2S DSP packet.

    Verify voice_slurred_detected is True and ImpairmentClassifier includes
    'slurred_voice_dysarthria' in active_markers.
    """
    audio_ext = AudioIntoxicationExtractor()
    audio_ext.push_hardware_telemetry(
        jitter_pct=2.1,
        shimmer_pct=5.8,
        hnr_db=8.5,
        f2_slope_hz_ms=7.5,
        vad_active=True,
    )

    audio_feat = audio_ext.update(0.0, visual_speech_active=True)
    assert audio_feat.voice_slurred_detected is True
    assert audio_feat.voice_impairment_score >= 60.0

    clf = ImpairmentClassifier()
    features = FrameFeatures(
        face_valid=True,
        ear_avg=0.28,
        perclos=0.05,
        speech_active=True,
        audio_valid=audio_feat.audio_valid,
        vocal_jitter_pct=audio_feat.vocal_jitter_pct,
        vocal_shimmer_pct=audio_feat.vocal_shimmer_pct,
        vocal_hnr_db=audio_feat.vocal_hnr_db,
        formant_f2_slope_hz_ms=audio_feat.formant_f2_slope_hz_ms,
        voice_impairment_score=audio_feat.voice_impairment_score,
        voice_slurred_detected=audio_feat.voice_slurred_detected,
    )

    assessment = clf.evaluate_temporal_stream(features)
    assert "slurred_voice_dysarthria" in assessment.active_markers
    assert assessment.marker_scores.get("voice_intoxication") == round(audio_feat.voice_impairment_score, 1)


def test_jesse_catatonic_stupor_silent_stare_detection():
    """Simulate 90 frames (3.0 seconds) of a silent, motionless catatonic stare.

    ear_avg=0.28, perclos=0.0, compensatory_stare_active=True, stare_fixation_duration_s=6.5,
    gaze_yaw_dispersion=0.015, blink_opening_velocity=1.35, speech_active=False, head_postural_sway=0.1.

    Verify sober_attentive_guard does NOT suppress the score and
    assessment.classification in ('MILD_IMPAIRMENT', 'HIGH_RISK_INTOXICATED') with
    assessment.risk_score >= 45.0.
    """
    clf = ImpairmentClassifier()
    assessment = None

    for _ in range(90):
        features = FrameFeatures(
            face_valid=True,
            ear_avg=0.28,
            perclos=0.0,
            compensatory_stare_active=True,
            stare_fixation_duration_s=6.5,
            gaze_yaw_dispersion=0.015,
            blink_opening_velocity=1.35,
            speech_active=False,
            head_postural_sway=0.1,
        )
        assessment = clf.evaluate_temporal_stream(features)

    assert assessment is not None
    assert assessment.classification in ("MILD_IMPAIRMENT", "HIGH_RISK_INTOXICATED")
    assert assessment.risk_score >= 45.0, f"Expected risk_score >= 45.0 during catatonic stupor, got {assessment.risk_score:.1f}"
    assert "catatonic_stare_stupor" in assessment.active_markers


def test_bayesian_secondary_voice_assist_accelerates_convergence():
    """Verify secondary voice assist accelerates Bayesian posterior convergence while sober driver remains immune.

    1. Adding audio_valid=True, voice_slurred_detected=True, voice_impairment_score=75.0
       increases posterior probability faster than visual-only mild impairment.
    2. Voice-alone on a 100% sober visual stream (risk_score < 15.0) never triggers lockout_confirmed.
    """
    # 1. Compare visual-only vs. visual + slurred voice convergence
    filter_visual = BayesianEvidenceAccumulator(lockout_sustained_sec=15.0)
    assess_visual = ImpairmentAssessment(
        risk_score=35.0,
        involuntary_risk_score=35.0,
        classification="MILD_IMPAIRMENT",
    )

    filter_fused = BayesianEvidenceAccumulator(lockout_sustained_sec=15.0)
    assess_fused = ImpairmentAssessment(
        risk_score=35.0,
        involuntary_risk_score=35.0,
        classification="MILD_IMPAIRMENT",
        marker_scores={"voice_intoxication": 75.0},
    )
    setattr(assess_fused, "audio_valid", True)
    setattr(assess_fused, "voice_slurred_detected", True)
    setattr(assess_fused, "voice_impairment_score", 75.0)

    for i in range(30):
        t = i * 0.033
        state_v = filter_visual.update(assess_visual, t)
        state_f = filter_fused.update(assess_fused, t)

    assert state_f.posterior_probability > state_v.posterior_probability, (
        f"Fused posterior ({state_f.posterior_probability:.3f}) must exceed visual-only ({state_v.posterior_probability:.3f})"
    )

    # 2. Voice-alone on a 100% sober visual stream (e.g. risk_score = 10.0)
    filter_sober_voice = BayesianEvidenceAccumulator(lockout_sustained_sec=15.0)
    assess_sober = ImpairmentAssessment(
        risk_score=10.0,
        involuntary_risk_score=10.0,
        classification="SOBER",
        marker_scores={"voice_intoxication": 75.0},
    )
    setattr(assess_sober, "audio_valid", True)
    setattr(assess_sober, "voice_slurred_detected", True)
    setattr(assess_sober, "voice_impairment_score", 75.0)

    for i in range(1800):  # 60 seconds at 30 fps
        t = i * 0.033
        state_sv = filter_sober_voice.update(assess_sober, t)

    assert not state_sv.lockout_confirmed, "Voice alone on a sober visual stream must never trigger lockout"
    assert state_sv.posterior_probability < 0.05, "Sober visual stream must maintain low posterior probability"


def test_quiet_short_phrase_detection():
    """Verify 0.5s spoken vowel burst (amp = 0.015, followed by 1.5s of silence in a 2.0s window).

    With visual_speech_active=True, it must be properly detected (voice_vad_active is True,
    audio_valid is True) and not drowned out by surrounding silence in the 2.0s buffer.
    """
    extractor = AudioIntoxicationExtractor(sample_rate=16000, window_sec=2.0)
    sr = 16000
    duration = 2.0
    pcm = np.zeros(int(sr * duration), dtype=np.float32)

    # 0.5s quiet voiced vowel burst (amplitude 0.015, F0 = 130 Hz)
    burst_len = int(0.50 * sr)
    t_burst = np.arange(burst_len) / sr
    vowel_burst = 0.015 * np.sin(2 * np.pi * 130.0 * t_burst) + 0.007 * np.sin(2 * np.pi * 260.0 * t_burst)
    st = int(0.50 * sr)
    pcm[st : st + burst_len] = vowel_burst

    feat = extractor.analyze_pcm_window(pcm, visual_speech_active=True)
    assert feat.voice_vad_active is True, "0.5s quiet phrase must trigger voice_vad_active"
    assert feat.audio_valid is True, "0.5s phrase with visual speech must be audio_valid"
    assert feat.av_speech_gated is True, "av_speech_gated must be True when visual speech is active"


def test_live_mic_speech_hold_hysteresis():
    """Verify 0.6s speech hold smoothly maintains valid voice features across pauses in live mic mode while mouth is active."""
    extractor = AudioIntoxicationExtractor(sample_rate=16000, window_sec=2.0)
    # Mock live stream object to enable live hold mode
    extractor._mic_stream = object()
    extractor.mic_active = True

    sr = 16000
    t = np.arange(sr) / sr
    burst = 0.05 * np.sin(2 * np.pi * 130.0 * t) + 0.02 * np.sin(2 * np.pi * 260.0 * t)
    extractor.push_pcm_chunk(burst)

    # Frame 0: Spoken word detected
    res0 = extractor.update(timestamp=1.0, visual_speech_active=True)
    assert res0.audio_valid is True, "Spoken word must be audio_valid"

    # Frame 1: Speaker pauses for 0.4s (within 0.6s hold, mouth still active)
    with extractor._lock:
        extractor._pcm_buffer.clear()
        extractor._pcm_buffer.extend(np.zeros(int(0.30 * sr), dtype=np.float32).tolist())
    extractor._last_dsp_timestamp = -1.0
    res1 = extractor.update(timestamp=1.4, visual_speech_active=True)
    assert res1.audio_valid is True, "Hold hysteresis must keep voice valid during 0.4s inter-syllable pause"

    # Frame 2: Speaker silent past 0.6s (timestamp=3.1s, dt=2.1s > 0.6s)
    extractor._last_dsp_timestamp = -1.0
    res2 = extractor.update(timestamp=3.1, visual_speech_active=False)
    assert res2.audio_valid is False, "Hold hysteresis must expire after 0.6s of silence"


def test_ambient_noise_and_sober_hello_never_trigger_slurred():
    """Verify quiet room noise (RMS ~ 0.0015) and a brief sober 'hello' never trigger slurred speech."""
    extractor = AudioIntoxicationExtractor(sample_rate=16000, window_sec=2.0)
    extractor._mic_stream = object()
    extractor.mic_active = True

    sr = 16000
    # 1. Quiet ambient room noise (RMS ~ 0.0015)
    np.random.seed(7)
    quiet_noise = np.random.normal(0, 0.0015, sr * 2).astype(np.float32)
    res_noise = extractor.analyze_pcm_window(quiet_noise, visual_speech_active=True)
    assert res_noise.audio_valid is False
    assert res_noise.voice_impairment_score == 0.0
    assert res_noise.voice_slurred_detected is False

    # 2. Short 0.4s sober "hello" pitch glide (135 Hz -> 145 Hz) in 2.0s buffer
    pcm_hello = np.zeros(sr * 2, dtype=np.float32)
    t_h = np.arange(int(0.40 * sr)) / sr
    f0_glide = np.linspace(135.0, 145.0, len(t_h))
    phase = 2 * np.pi * np.cumsum(f0_glide) / sr
    hello_wave = 0.08 * (np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase))
    pcm_hello[int(0.5 * sr) : int(0.9 * sr)] = hello_wave.astype(np.float32)

    res_hello = extractor.analyze_pcm_window(pcm_hello, visual_speech_active=True)
    assert res_hello.audio_valid is True
    assert res_hello.voice_impairment_score < 25.0, f"Sober hello scored {res_hello.voice_impairment_score:.1f}%"
    assert res_hello.voice_slurred_detected is False


def test_uninitialized_blink_velocity_does_not_trigger_catatonic_stare():
    """Verify a user staring at the desktop screen before their first blink (blink_opening_velocity=0.0)
    with normal gaze_yaw_dispersion (0.047) stays SOBER and does not trigger catatonic_stare_detected.
    """
    clf = ImpairmentClassifier()
    features = FrameFeatures(
        face_valid=True,
        ear_avg=0.35,
        perclos=0.0,
        blink_opening_velocity=0.0,
        gaze_yaw_dispersion=0.047,
        stare_fixation_duration_s=9.2,
        compensatory_stare_active=True,
        deliberate_blink_suppression=True,
        head_postural_sway=1.4,
        speech_active=False,
    )
    assessment = clf.evaluate_temporal_stream(features)
    assert "catatonic_stare_stupor" not in assessment.active_markers
    assert assessment.classification == "SOBER", f"Expected SOBER, got {assessment.classification} (risk={assessment.risk_score})"
    assert assessment.risk_score <= 22.0


def test_voice_remains_strictly_secondary_when_visually_sober():
    """Verify even 100% slurred voice cannot flip a visually sober user (inv_risk < 22%) to intoxicated."""
    clf = ImpairmentClassifier()
    features = FrameFeatures(
        face_valid=True,
        ear_avg=0.33,
        perclos=0.02,
        blink_opening_velocity=2.4,
        gaze_yaw_dispersion=0.05,
        speech_active=True,
        audio_valid=True,
        voice_impairment_score=100.0,
        voice_slurred_detected=True,
    )
    assessment = clf.evaluate_temporal_stream(features)
    assert assessment.classification == "SOBER", f"Voice alone must not override sober visual state, got {assessment.classification}"
    assert assessment.risk_score < 25.0


def test_desktop_screen_stare_with_normal_gaze_dispersion_stays_sober():
    """Verify staring at a desktop monitor (gaze_yaw_dispersion=0.055, ear_avg=0.32, perclos=0.04)
    with a single low-FPS blink (0.70 EAR/s) does NOT false-lock into HIGH_RISK_INTOXICATED.
    """
    clf = ImpairmentClassifier()
    for _ in range(60):
        features = FrameFeatures(
            face_valid=True,
            ear_avg=0.32,
            perclos=0.04,
            compensatory_stare_active=True,
            deliberate_blink_suppression=True,
            stare_fixation_duration_s=9.5,
            gaze_yaw_dispersion=0.055,
            blink_opening_velocity=0.70,
            speech_active=False,
            head_postural_sway=0.50,
            masking_divergence_score=60.0,
            masking_detected=True,
        )
        assessment = clf.evaluate_temporal_stream(features)
    assert assessment.classification == "SOBER", f"Expected SOBER on alert desktop stare, got {assessment.classification} (risk={assessment.risk_score}%)"
    assert assessment.risk_score < 25.0


def test_live_mic_sober_phone_audio_rejected_when_jitter_is_low():
    """Verify live mic mode does not flag slurred speech when vocal jitter is sober (<1.2%),
    even if room echo/phone speaker distorts shimmer and HNR.
    """
    extractor = AudioIntoxicationExtractor(sample_rate=16000, window_sec=1.0)
    extractor._mic_stream = object()  # simulate active live mic stream
    extractor.mic_active = True

    sr = 16000
    t = np.arange(sr) / sr
    # Steady 130 Hz fundamental (sober jitter < 0.8%) but with amplitude modulation & room noise
    env = 0.6 + 0.35 * np.sin(2 * np.pi * 8.0 * t)
    pcm = env * (0.8 * np.sin(2 * np.pi * 130.0 * t) + 0.3 * np.sin(2 * np.pi * 260.0 * t))
    pcm += np.random.normal(0, 0.08, len(pcm))

    feat = extractor.analyze_pcm_window(pcm.astype(np.float32), visual_speech_active=True)
    assert not feat.voice_slurred_detected, "Live mic must not flag sober low-jitter speech as slurred due to room/speaker shimmer"
    assert feat.voice_impairment_score <= 35.0


