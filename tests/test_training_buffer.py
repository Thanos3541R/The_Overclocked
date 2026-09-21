import pytest
import numpy as np
import os
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
    for i in range(7): # 7 frames < 80% of 10
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
