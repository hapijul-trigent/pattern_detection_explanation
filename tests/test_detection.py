import cv2
import numpy as np
from src.detection import optical_flow_motion_detection, visualize_motion_vectors


def test_optical_flow_motion_detection():
    # Dummy frames
    prev_frame = np.ones((480, 640), dtype=np.uint8) * 255
    curr_frame = np.ones((480, 640), dtype=np.uint8) * 200
    
    motion_detected, motion_vectors, avg_magnitude, good_next_pts, good_prev_pts = optical_flow_motion_detection(
        prev_frame, curr_frame
    )
    
    assert isinstance(motion_detected, bool), "motion_detected should be a boolean"
    assert isinstance(motion_vectors, np.ndarray), "motion_vectors should be a numpy array"
    assert isinstance(avg_magnitude, np.ndarray), "avg_magnitude should be a numpy array"
    assert isinstance(good_next_pts, np.ndarray), "good_next_pts should be a numpy array"
    assert isinstance(good_prev_pts, np.ndarray), "good_prev_pts should be a numpy array"

def test_visualize_motion_vectors():
    # Dummy data for visualization
    prev_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    curr_frame = np.ones((480, 640), dtype=np.uint8) * 200
    prev_pts = np.array([[100, 100], [200, 200]], dtype=np.float32)
    next_pts = np.array([[120, 120], [220, 220]], dtype=np.float32)
    magnitudes = np.array([1.5, 2.0])

    visualized_frame = visualize_motion_vectors(prev_frame, curr_frame, prev_pts, next_pts, magnitudes)
    
    assert visualized_frame is not None, "The function should return a valid frame."
    assert visualized_frame.shape == prev_frame.shape, "The visualized frame should have the same shape as the input frame."
