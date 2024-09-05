import torch
import numpy as np
from src.yolovx_trackers import YOLOTracker

def test_yolo_tracker_init():
    model_detect = torch.nn.Identity()
    pose_model = torch.nn.Identity()
    
    tracker = YOLOTracker(model_detect, pose_model, resolution=(640, 480))
    
    assert tracker.modelDetect is not None, "Model detection should be initialized"
    assert tracker.poseModel is not None, "Pose model should be initialized"
    assert tracker.resolution == (640, 480), "Resolution should be set correctly"

def test_yolo_tracker_process_frame():
    model_detect = torch.nn.Identity()
    pose_model = torch.nn.Identity()
    
    tracker = YOLOTracker(model_detect, pose_model, resolution=(640, 480))
    
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    result = tracker.process_frame(dummy_frame)
    
    assert result is not None, "Processing should return valid results"
    assert isinstance(result, tuple), "Result should be a tuple"
