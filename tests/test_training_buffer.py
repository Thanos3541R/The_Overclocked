import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.training_buffer import TrainingBuffer, GatedFrameStrategy
from pipeline.feature_extractor import FrameFeatures

def test_feature_vector_assembly():
    tb = TrainingBuffer(subject_id="sub1")
    ff = FrameFeatures(timestamp=1.0, ear_left=0.25, microsleep_detected=True)
    vec = tb._extract_vector(ff)
    
    assert vec.shape == (29,)
    # ear_left is index 0
    assert vec[0] == 0.25
    # microsleep_detected is index 23
    assert vec[23] == 1.0
    
def test_subject_id_tagging():
    tb = TrainingBuffer(subject_id="sub123")
    for i in range(160):
        tb.add_sample(FrameFeatures(timestamp=i*(1/30.0)))
    
    windows = tb.extract_windows()
    assert len(windows) > 0
    assert windows[0].subject_id == "sub123"

def test_window_extraction():
    tb = TrainingBuffer(subject_id="test", window_size_sec=1.0, stride_sec=0.5, fps=10)
    # window_frames = 10, stride_frames = 5
    for i in range(20):
        tb.add_sample(FrameFeatures(timestamp=i*0.1))
        
    windows = tb.extract_windows()
    assert len(windows) >= 2
    assert windows[0].features.shape == (10, 29)
    assert windows[0].timestamps.shape == (10,)
    
def test_gated_frame_drop():
    tb = TrainingBuffer(subject_id="test", gated_strategy=GatedFrameStrategy.DROP)
    tb.add_sample(FrameFeatures(timestamp=0.1, face_valid=True))
    tb.add_sample(FrameFeatures(timestamp=0.2, face_valid=False)) # gated
    tb.add_sample(FrameFeatures(timestamp=0.3, face_valid=True))
    
    assert len(tb) == 2
    assert tb._buffer[0][0] == 0.1
    assert tb._buffer[1][0] == 0.3

def test_gated_frame_hold_last():
    tb = TrainingBuffer(subject_id="test", gated_strategy=GatedFrameStrategy.HOLD_LAST)
    tb.add_sample(FrameFeatures(timestamp=0.1, ear_avg=0.5, face_valid=True))
    tb.add_sample(FrameFeatures(timestamp=0.2, ear_avg=0.1, face_valid=False, imu_gated=True)) # gated
    
    assert len(tb) == 2
    # Second frame should have held the numeric values of the first frame
    assert tb._buffer[1][1][2] == 0.5 # ear_avg
    assert tb._buffer[1][1][-1] == 1.0 # imu_gated/gated flag from current

def test_minimum_fill_threshold():
    tb = TrainingBuffer(subject_id="test", window_size_sec=1.0, fps=10)
    for i in range(6): # 6 frames < 70% of 10
        tb.add_sample(FrameFeatures(timestamp=i*0.1))
    
    windows = tb.extract_windows()
    assert len(windows) == 0

def test_save_load_roundtrip(tmp_path):
    tb = TrainingBuffer(subject_id="test_save", window_size_sec=1.0, stride_sec=1.0, fps=10)
    for i in range(15):
        tb.add_sample(FrameFeatures(timestamp=i*0.1, ear_left=0.5))
        
    file_path = os.path.join(tmp_path, "buffer.npz")
    tb.save_to_disk(file_path)
    
    assert os.path.exists(file_path)
    data = np.load(file_path)
    
    assert "features" in data
    assert "timestamps" in data
    assert "subject_ids" in data
    assert data["features"].shape == (1, 10, 29)
    assert data["subject_ids"][0] == "test_save"

def test_reset():
    tb = TrainingBuffer(subject_id="test")
    tb.add_sample(FrameFeatures(timestamp=0.1))
    assert len(tb) == 1
    tb.reset()
    assert len(tb) == 0
    assert tb.subject_id == "test"

def test_label_persistence(tmp_path):
    tb = TrainingBuffer(subject_id="sub_label", default_label="sober", window_size_sec=1.0, stride_sec=1.0, fps=10)
    for i in range(15):
        tb.add_sample(FrameFeatures(timestamp=i*0.1))
    
    # Check default_label propagation
    windows = tb.extract_windows()
    assert len(windows) > 0
    assert windows[0].label == "sober"

    # Check explicit label override
    windows_override = tb.extract_windows(label="impaired")
    assert windows_override[0].label == "impaired"

    # Check save_to_disk preserves label
    file_path = os.path.join(tmp_path, "labeled_buffer.npz")
    tb.save_to_disk(file_path, label="drunk")
    data = np.load(file_path)
    assert "labels" in data
    assert data["labels"][0] == "drunk"


def test_multimodal_audio_extraction_and_persistence(tmp_path):
    """Verify TrainingBuffer extracts 7-channel audio data, supports push/clear, and roundtrips to disk."""
    tb = TrainingBuffer(subject_id="sub_audio", window_size_sec=1.0, stride_sec=0.5, fps=10)
    assert len(TrainingBuffer.AUDIO_FEATURE_NAMES) == 7

    # Push frames with active speech
    for i in range(15):
        ff = FrameFeatures(
            timestamp=i * 0.1,
            face_valid=True,
            ear_left=0.30,
            audio_valid=True,
            vocal_jitter_pct=0.45,
            vocal_shimmer_pct=2.1,
            vocal_hnr_db=18.5,
            formant_f2_slope_hz_ms=24.0,
            speech_pause_ratio=0.15,
            voice_impairment_score=12.0,
        )
        tb.push(ff)

    assert len(tb) == 15
    windows = tb.extract_windows(label="sober")
    assert len(windows) >= 1
    w0 = windows[0]
    assert w0.features.shape == (10, 29)  # Base feature shape preserved
    assert w0.audio_data is not None
    assert w0.audio_data.shape == (10, 7)
    assert w0.audio_data[0, 0] == 1.0  # audio_valid
    assert pytest.approx(w0.audio_data[0, 1]) == 0.45  # vocal_jitter_pct
    assert pytest.approx(w0.audio_data[0, 3]) == 18.5  # vocal_hnr_db

    # Roundtrip save and load
    npz_path = os.path.join(tmp_path, "multimodal_buf.npz")
    tb.save_to_disk(npz_path, label="sober")
    assert os.path.exists(npz_path)

    loaded_windows = TrainingBuffer.load_from_disk(npz_path)
    assert len(loaded_windows) == len(windows)
    lw0 = loaded_windows[0]
    assert lw0.features.shape == (10, 29)
    assert lw0.audio_data is not None
    assert lw0.audio_data.shape == (10, 7)
    assert lw0.audio_data[0, 0] == 1.0

    # Clear alias
    tb.clear()
    assert len(tb) == 0


def test_compute_window_summary_imputes_neutral_sober_voice():
    """Verify compute_window_summary_features imputes neutral sober defaults on silent windows."""
    from train_baseline import compute_window_summary_features

    feature_names = [
        "ear_avg",
        "audio_valid",
        "vocal_jitter_pct",
        "vocal_shimmer_pct",
        "vocal_hnr_db",
        "formant_f2_slope_hz_ms",
        "speech_pause_ratio",
        "voice_impairment_score",
    ]
    # 2 windows of 10 frames: window 0 is completely silent (audio_valid=0), window 1 is voiced
    win_3d = np.zeros((2, 10, len(feature_names)), dtype=np.float32)
    win_3d[:, :, 0] = 0.30  # ear_avg
    win_3d[1, :, 1] = 1.0   # window 1 audio_valid = 1.0
    win_3d[1, :, 2] = 1.85  # window 1 vocal_jitter_pct = 1.85

    feat_2d, flat_names = compute_window_summary_features(win_3d, feature_names)

    # In window 0 (silent), vocal_jitter_pct_mean should be neutral default 0.35, not NaN or 0
    jitter_mean_idx = flat_names.index("vocal_jitter_pct_mean")
    assert pytest.approx(feat_2d[0, jitter_mean_idx]) == 0.35
    # In window 1 (voiced), vocal_jitter_pct_mean should be 1.85
    assert pytest.approx(feat_2d[1, jitter_mean_idx]) == 1.85

    # Check vocal_shimmer_pct_mean default 1.50
    shimmer_mean_idx = flat_names.index("vocal_shimmer_pct_mean")
    assert pytest.approx(feat_2d[0, shimmer_mean_idx]) == 1.50

    # Check vocal_hnr_db_mean default 20.0
    hnr_mean_idx = flat_names.index("vocal_hnr_db_mean")
    assert pytest.approx(feat_2d[0, hnr_mean_idx]) == 20.0


def test_filter_involuntary_retains_catatonic_and_excludes_audio():
    """Verify filter_involuntary_features retains catatonic stare / tremor while excluding audio."""
    from train_baseline import filter_involuntary_features

    fnames = [
        "stare_fixation_duration_s_mean",
        "stare_fixation_duration_s_max",
        "catatonic_stupor_detected_mean",
        "eyelid_tremor_hz_mean",
        "ear_avg_mean",
        "vocal_jitter_pct_mean",
        "vocal_hnr_db_mean",
        "audio_valid_mean",
        "head_yaw_mean",  # voluntary, excluded
        "head_yaw_std",   # involuntary instability, kept
    ]
    X = np.zeros((2, len(fnames)), dtype=np.float32)
    X_filt, kept = filter_involuntary_features(X, fnames)

    assert "stare_fixation_duration_s_mean" in kept
    assert "stare_fixation_duration_s_max" in kept
    assert "catatonic_stupor_detected_mean" in kept
    assert "eyelid_tremor_hz_mean" in kept
    assert "head_yaw_std" in kept
    assert "ear_avg_mean" in kept

    # Voluntary head yaw and audio features must be excluded
    assert "head_yaw_mean" not in kept
    assert "vocal_jitter_pct_mean" not in kept
    assert "vocal_hnr_db_mean" not in kept
    assert "audio_valid_mean" not in kept


def test_load_dataset_from_npz_concatenates_audio_and_pads_legacy(tmp_path):
    """Verify load_dataset_from_npz concatenates X_audio and pads legacy archives with neutral sober defaults."""
    from train_baseline import load_dataset_from_npz

    # 1. Archive with X_audio (modern)
    modern_path = tmp_path / "modern_clip.npz"
    np.savez(
        str(modern_path),
        features=np.zeros((2, 10, 29), dtype=np.float32),
        feature_names=np.array([f"v_feat_{i}" for i in range(29)]),
        labels=np.array(["sober", "sober"]),
        subject_ids=np.array(["sub_mod", "sub_mod"]),
        timestamps=np.zeros((2, 10)),
        X_audio=np.ones((2, 10, 7), dtype=np.float32) * 5.0,
        audio_feature_names=np.array(TrainingBuffer.AUDIO_FEATURE_NAMES),
    )

    # 2. Archive without X_audio (legacy 29-channel)
    legacy_path = tmp_path / "legacy_clip.npz"
    np.savez(
        str(legacy_path),
        features=np.zeros((2, 10, 29), dtype=np.float32),
        feature_names=np.array([f"v_feat_{i}" for i in range(29)]),
        labels=np.array(["drunk", "drunk"]),
        subject_ids=np.array(["sub_leg", "sub_leg"]),
        timestamps=np.zeros((2, 10)),
    )

    X_3d, y, subs, target_names = load_dataset_from_npz([str(modern_path), str(legacy_path)])
    assert X_3d.shape == (4, 10, 36)  # 29 visual + 7 audio = 36 channels
    assert len(target_names) == 36

    # Modern windows (0, 1) should have 5.0 for audio channels
    jitter_idx = target_names.index("vocal_jitter_pct")
    assert pytest.approx(X_3d[0, 0, jitter_idx]) == 5.0

    # Legacy windows (2, 3) should have padded neutral sober default (0.35)
    assert pytest.approx(X_3d[2, 0, jitter_idx]) == 0.35

