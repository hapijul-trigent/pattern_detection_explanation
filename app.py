import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import os
import logging
import time
import queue
from streamlit.delta_generator import DeltaGenerator
from typing import BinaryIO
from src.motion_detection  import optical_flow_motion_detection, visualize_motion_vectors
from src.movement_tracker import annotate_frame_with_ssd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set main panel
favicon = Image.open("static/images/Trigent_Logo.png")
st.set_page_config(
    page_title="Entities in Clinical Trial | Trigent AXLR8 Labs",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Add logo and title
logo_path = "https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png"
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="{logo_path}" alt="Trigent Logo" style="max-width:100%;">
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Main Page Title and Caption
st.title("Motion Tracker")
st.caption("")

def create_temp_video_file(video_file: BinaryIO, temp_dir: str = 'outputs/temp') -> str:
    """
    Create a temporary file to store a video.

    Args:
        video_file (BinaryIO): A file-like object containing the video data.
        temp_dir (str, optional): The directory where the temporary file should be created. Defaults to 'outputs/temp'.

    Returns:
        str: The path to the created temporary file.
    """
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
    motion_detected, motion_vectors, avg_magnitude, good_next_pts, good_prev_pts = optical_flow_motion_detection(prev_frame=previous_gray_frame, curr_frame=curr_gray_frame, threshold=2, min_distance=9, quality_level=0.5)
    return motion_detected, motion_vectors, avg_magnitude, good_next_pts, good_prev_pts


def stream_frames(frame, panel: DeltaGenerator):
    panel.image(
        image=frame,
        caption="Preview",
        width=500,
        # use_column_width=True,
        channels="BGR",
    )



def capture_frames(source: str):
    cap = cv2.VideoCapture(source)
    ret = True
    ret, previous_frame = cap.read()
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        time.sleep(0.1)
        yield previous_frame, frame
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


def main():
    # States session 
    get_or_create_session_state_variable("video_file", None)
    get_or_create_session_state_variable("previous_frame", None)
    get_or_create_session_state_variable("current_frame", None)
    get_or_create_session_state_variable("motion_detected", False)
    get_or_create_session_state_variable("motion_vectors", None)
    get_or_create_session_state_variable("avg_magnitude", 0.0)
    get_or_create_session_state_variable("good_next_pts", None)
    get_or_create_session_state_variable("good_prev_pts", None)
    get_or_create_session_state_variable("algortihm", default_value='Optical Flow')

    # configPanel, previewPanel
    configPanel, previewPanel = st.columns([1, 3])
    flag = False
    source = None

    # Config Panel
    with configPanel:
        # Upload video
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

        # button to start processing
        if st.button("Start Processing"):
            flag = True

        if video_file is not None:
            source = create_temp_video_file(video_file=video_file)

        if video_file:
            st.header('Configure')
            st.session_state.algortihm = st.selectbox(label='Algorithm', options=['Optical Flow', 'Segmentation', 'Single Shot Detectore(SSD)'])
            

    # PreviewPanel
    with previewPanel:
        # two more columns named objectTrackingPanel, segmentationPanel
        objectTrackingPanel, segmentationPanel = st.columns([1, 1])

        with objectTrackingPanel:
            if flag and source:
                panel = st.empty()

                for previous_frame, curr_frame in capture_frames(source):
                    if st.session_state.algortihm == 'Optical Flow':
                        motion_detected, motion_vectors, avg_magnitude, good_next_pts, good_prev_pts = process_frame(previous_frame, curr_frame)
                        if motion_detected:
                            
                            vframe = visualize_motion_vectors(prev_frame=previous_frame, curr_frame=curr_frame, prev_pts=good_prev_pts, next_pts=good_next_pts)
                            stream_frames(vframe, panel)
                            continue
                        stream_frames(curr_frame, panel)
                        previous_frame = curr_frame
                    elif st.session_state.algortihm == 'Single Shot Detectore(SSD)':
                        vframe = annotate_frame_with_ssd(frame=curr_frame)
                        stream_frames(vframe, panel)
        

        
        
        # Clean up the temporary file after processing
        if source:
            if os.path.exists(source):
                os.remove(source)

if __name__ == "__main__":
    main()
