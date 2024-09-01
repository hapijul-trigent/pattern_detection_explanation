import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Tuple
import time
import streamlit as st

### Optical FLow for MOtion Detection
def optical_flow_motion_detection(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float = 1.0,
    max_corners: int = 100,
    quality_level: float = 0.3,
    min_distance: int = 7
) -> Tuple[bool, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Detects motion between two frames using the Lucas-Kanade optical flow method.

    Args:
        prev_frame (np.ndarray): The previous frame as a grayscale image.
        curr_frame (np.ndarray): The current frame as a grayscale image.
        threshold (float): The threshold value for motion detection (default is 1.0).
        max_corners (int): The maximum number of corners to return (default is 100).
        quality_level (float): The quality level for the corners to return (default is 0.3).
        min_distance (int): The minimum distance between corners (default is 7).

    Returns:
        Tuple[bool, np.ndarray, float, np.ndarray, np.ndarray]:
            - bool: True if motion is detected, False otherwise.
            - np.ndarray: The motion vectors (differences between previous and current points).
            - float: The average magnitude of motion vectors.
            - np.ndarray: The good feature points in the current frame.
            - np.ndarray: The good feature points in the previous frame.

    Raises:
        ValueError: If the input frames are not of the same size or not grayscale.
        TypeError: If inputs are not numpy arrays.

    Example:
        prev_frame = cv2.imread('prev_frame.png', cv2.IMREAD_GRAYSCALE)
        curr_frame = cv2.imread('curr_frame.png', cv2.IMREAD_GRAYSCALE)
        motion_detected, motion_vectors, avg_magnitude, good_next_pts, good_prev_pts = optical_flow_motion_detection(prev_frame, curr_frame)
    """
    try:
        if not isinstance(prev_frame, np.ndarray) or not isinstance(curr_frame, np.ndarray):    # Validate input types
            raise TypeError("Input frames must be numpy arrays.")
        
        if prev_frame.shape != curr_frame.shape:                 # Validate frame dimensions and type
            raise ValueError("Input frames must have the same dimensions.")
        if len(prev_frame.shape) != 2 or len(curr_frame.shape) != 2:
            raise ValueError("Input frames must be grayscale images.")

        # Find the corners in the previous frame
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )
    
        prev_pts = cv2.goodFeaturesToTrack(
            prev_frame,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        if prev_pts is None:
            return False, np.array([]), 0.0, np.array([]), np.array([])
        
        # Compute feature points
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=prev_frame, 
            nextImg=curr_frame, 
            prevPts=prev_pts, nextPts=None, **lk_params
        )
        
        if next_pts is not None and status is not None:             # Select good points
            good_next_pts = next_pts[status == 1]
            good_prev_pts = prev_pts[status == 1]
            
            # Compute the magnitude of motion vectors
            motion_vectors = good_next_pts - good_prev_pts
            magnitude = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)         
            
            return any(magnitude > threshold), motion_vectors, magnitude, good_next_pts, good_prev_pts               # Motion is detected if the average magnitude exceeds the threshold
        else:
            return False, np.array([]), 0.0, np.array([]), np.array([])
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False, np.array([]), 0.0, np.array([]), np.array([])

    

def visualize_motion_vectors(prev_frame, curr_frame, prev_pts, next_pts, magnitudes):
    """
    Visualize the motion vectors between two consecutive frames of a video.

    Args:
        prev_frame (numpy.ndarray): The previous frame of the video.
        curr_frame (numpy.ndarray): The current frame of the video.
        prev_pts (list): A list of (x, y) coordinates representing the starting points of the motion vectors.
        next_pts (list): A list of (x, y) coordinates representing the ending points of the motion vectors.

    Returns:
        numpy.ndarray: The current frame with motion vectors visualized as green arrows.
    """
    try:
        color_frame = prev_frame.copy()
        for (x0, y0), (x1, y1), magnitude in zip(prev_pts, next_pts, magnitudes):
            if magnitude > 1.5:
                cv2.arrowedLine(img=color_frame, pt1=(int(x0), int(y0)), pt2=(int(x1), int(y1)), color=(240,0,255), thickness=3, tipLength=0.5, line_type=cv2.LINE_4)
        return color_frame
    except Exception as e:
        logging.error(f"Error visualizing motion vectors: {e}")
        return prev_frame



# Example usage:
# prev_frame = cv2.imread('static/images/1.jpg', cv2.IMREAD_GRAYSCALE)
# prev_frame_c = cv2.imread('static/images/1.jpg')
# curr_frame = cv2.imread('static/images/2.jpg', cv2.IMREAD_GRAYSCALE)
# motion_detected, motion_vectors, avg_magnitude, good_next_pts, good_prev_pts = optical_flow_motion_detection(prev_frame, curr_frame)
# print(motion_detected, good_next_pts.shape, good_prev_pts.shape)
# vframe = visualize_motion_vectors(prev_frame_c, curr_frame, good_prev_pts, good_next_pts)
# print(vframe.shape)
# plt.imsave('vframe.jpg', vframe)



