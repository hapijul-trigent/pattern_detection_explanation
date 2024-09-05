import numpy as np
from src.optical_flow_tracker import OpticalFlowTracker

def test_optical_flow_tracker_init():
    tracker = OpticalFlowTracker(video_path="dummy.mp4", resolution=(640, 480))
    
    assert tracker.video_path == "dummy.mp4", "Video path should be correctly set"
    assert tracker.resolution == (640, 480), "Resolution should be correctly set"
    assert tracker.trajectory_len == 30, "Default trajectory length should be 30"
    assert tracker.detect_interval == 3, "Default detect interval should be 3"

def test_optical_flow_video_processing():
    tracker = OpticalFlowTracker(video_path="dummy.mp4", resolution=(640, 480))

    # Mock video reading function
    def dummy_read():
        return True, np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Mock the process_video method
    tracker.process_video(dummy_read, dummy_read)

    assert True, "Function executed without errors"
