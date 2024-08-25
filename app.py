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
from src.motion_detection  import visualize_motion_vectors, optical_flow_motion_detection
from src.trackers import run_yolo_tracker, OpticalFlowTracker
from src.loaders import load_yolo_model
from src.utils import (
    get_or_create_session_state_variable, 
    create_temp_video_file, 
    stream_frames, 
    capture_frames,
    process_frame,
    select_resolution
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set main panel
favicon = Image.open("static/images/Trigent_Logo.png")
st.set_page_config(
    page_title="Smart Motion Insights | Trigent AXLR8 Labs",
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
# Main Page Title and Caption
st.title("Smart Motion Insights")
st.caption("Explore advanced motion analysis with our Streamlit app. Detect, track, and predict movement patterns in images and videos, and gain clear insights into predictions. Ideal for educators, researchers, and motion analysis enthusiasts")
st.divider()



# To style the image further, you may need to use additional HTML/CSS in Streamlit:
st.markdown(
    """
    <style>
    data-testid[stImage] {
        border: 5px solid #ff6347;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
    get_or_create_session_state_variable("start_process_button_clicked", default_value=False)
    get_or_create_session_state_variable("processed", default_value=False)

    # Models
    modelDetect, modelSegment = load_yolo_model(model_path="models/")

    # configPanel, previewPanel
    configPanel, previewPanel = st.columns([1, 3], gap="large")
    flag = False
    source = None

    # Config Panel
    with configPanel:
        # Upload video
        video_file = st.file_uploader("Upload a video", type=["mp4",])
            

        if video_file is not None:
            source = create_temp_video_file(video_file=video_file)

        if video_file:

            st.session_state.algortihm = st.selectbox(label='Algorithm', options=['Optical Flow', 'YOLOv8'])
            resolution_choice, resolution = select_resolution()
            
        # button to start processing
        if not st.session_state.processed and video_file:
            if not st.session_state.start_process_button_clicked:
                st.button("Track", on_click=lambda: st.session_state.update(start_process_button_clicked=True))
            else:
                processing, stop = st.columns([1.5, 3])
                with processing:
                    st.button("Tracking...", disabled=True)
                with stop:
                    st.button("Stop", on_click=lambda: st.session_state.update(start_process_button_clicked=False))
        else:
            st.button("Track", on_click=lambda: st.session_state.update(start_process_button_clicked=False), disabled=True)
    # PreviewPanel
    yolov8_thread = None
    with previewPanel:
        st.title(f"Motion Tracking- {st.session_state.algortihm}")
        # two more columns named objectTrackingPanel, segmentationPanel
        objectTrackingPanel, segmentationPanel = st.columns([1, 1], gap='large')

        
        if st.session_state.start_process_button_clicked and source:
            panel = st.empty()
            if st.session_state.algortihm == 'Optical Flow':
                
                optical_flow = OpticalFlowTracker(video_path=source, resolution=resolution)
                optical_flow.process_video(objectTrackingPanel, segmentationPanel)
            elif st.session_state.algortihm == 'YOLOv8':
                with objectTrackingPanel:
                    save_path = run_yolo_tracker(filename=source, modelDetect=modelDetect, modelSegment=modelSegment, file_index=1, resolution=resolution)
                    # if save_path:
                        # panel.video(data=save_path, format='video/mp4', autoplay=True)
            st.session_state.start_process_button_clicked = False
            st.success('Sucessfully Processed!')
    
        
        # Clean up the temporary file after processing
        if source:
            if os.path.exists(source):
                os.remove(source)

if __name__ == "__main__":
    main()
