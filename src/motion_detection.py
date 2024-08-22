import cv2
import numpy as np
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def optical_flow_motion_detection(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: int = 25) -> bool:
    """
    Detects motion between two frames using frame differencing.
    
    Args:
        prev_frame (np.ndarray): The previous frame as a grayscale image.
        curr_frame (np.ndarray): The current frame as a grayscale image.
        threshold (int): The threshold value for motion detection (default is 25).

    Returns:
        bool: True if motion is detected, False otherwise.

    Raises:
        ValueError: If the input frames are not of the same size or not grayscale.
        TypeError: If inputs are not numpy arrays.

    Example:
        prev_frame = cv2.imread('prev_frame.png', cv2.IMREAD_GRAYSCALE)
        curr_frame = cv2.imread('curr_frame.png', cv2.IMREAD_GRAYSCALE)
        motion_detected = detect_motion(prev_frame, curr_frame)
    """
    try:
        # Validate input types
        if not isinstance(prev_frame, np.ndarray) or not isinstance(curr_frame, np.ndarray):
            raise TypeError("Both prev_frame and curr_frame should be numpy arrays.")

        # Validate frame dimensions
        if prev_frame.shape != curr_frame.shape:
            raise ValueError("Input frames must have the same dimensions.")

        # Compute the absolute difference between frames
        diff = cv2.absdiff(prev_frame, curr_frame)

        # Apply a binary threshold to the difference image
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Count the number of non-zero pixels
        non_zero_count = np.count_nonzero(thresh)

        # Log the number of non-zero pixels
        logging.info(f"Number of non-zero pixels in thresholded image: {non_zero_count}")

        # Motion is detected if the count of non-zero pixels exceeds a threshold
        return non_zero_count > 0

    except Exception as e:
        logging.error(f"An error occurred in : {e}")
        return False

# Example usage:
prev_frame = cv2.imread('/workspaces/pattern_detection_explanation/static/images/1.jpg', cv2.IMREAD_GRAYSCALE)
curr_frame = cv2.imread('/workspaces/pattern_detection_explanation/static/images/3.jpg', cv2.IMREAD_GRAYSCALE)
motion_detected = optical_flow_motion_detection(prev_frame, curr_frame)
print(motion_detected)