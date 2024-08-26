import cv2
import numpy as np
import os
import tempfile
from typing import BinaryIO
from streamlit.delta_generator import DeltaGenerator
import logging
import streamlit as st
import time
from .motion_detection import optical_flow_motion_detection


def create_temp_video_file(video_file: BinaryIO, temp_dir: str = 'outputs/temp') -> str:
    """
    Create a temporary file to store a video.

    Args:
        video_file (BinaryIO): A file-like object containing the video data.
        temp_dir (str, optional): The directory where the temporary file should be created. Defaults to 'outputs/temp'.

    Returns:
        str: The path to the created temporary file.
    """
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mp4") as tfile:
            tfile.write(video_file.read())
            return tfile.name
    except Exception as e:
        logger.error(f"Error creating temporary video file: {e}")
        return ""

def process_frame(previous_frame: np.ndarray, curr_frame: np.ndarray):
    previous_gray_frame = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
    curr_gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    motion_detected, motion_vectors, avg_magnitude, good_next_pts, good_prev_pts = optical_flow_motion_detection(prev_frame=previous_gray_frame, curr_frame=curr_gray_frame, threshold=1.5, min_distance=3, quality_level=0.3)
    return motion_detected, motion_vectors, avg_magnitude, good_next_pts, good_prev_pts


def stream_frames(frame, panel: DeltaGenerator):
    _, buffer = cv2.imencode('.png', frame)
    panel.image(
        image=buffer.tobytes(),
        caption="Preview",
        width=None,
        use_column_width=True,
        channels="BGR",
    )



def capture_frames(source: str, resolution):
    cap = cv2.VideoCapture(source)
    ret = True
    ret, previous_frame = cap.read()

    previous_frame = cv2.GaussianBlur(
        src=cv2.resize(previous_frame, resolution, interpolation=cv2.INTER_AREA), 
        ksize=(3,3), sigmaX=0
    )
    while ret:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.GaussianBlur(
            src=cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA), 
            ksize=(3,3), sigmaX=0
        )

        time.sleep(0.04)
        yield ret, previous_frame, frame
        previous_frame = frame

    cap.release()




def get_or_create_session_state_variable(key, default_value=None):
    """
    Retrieves the value of a variable from Streamlit's session state.
    If the variable doesn't exist, it creates it with the provided default value.

    Args:
        key (str): The key of the variable in session state.
        default_value (Any): The default value to assign if the variable doesn't exist.

    Returns:
        Any: The value of the session state variable.
    """
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]



def select_resolution():
    """
    Displays a dropdown menu for selecting video resolution and returns the selected resolution.
    """
    resolution_options = {
        "360p": (640, 360),
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4K": (3840, 2160),
        "8K": (7680, 4320),
    }

    resolution_choice = st.selectbox("Select Resolution", list(resolution_options.keys()), index=0, disabled=st.session_state.start_process_button_clicked)
    resolution = resolution_options[resolution_choice]
    
    return resolution_choice, resolution
