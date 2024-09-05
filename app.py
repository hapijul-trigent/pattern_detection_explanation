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
from src.yolovx_trackers import YOLOTracker
from src.optical_flow_tracker import OpticalFlowTracker
from src.yolo_world_tracker import YOLOWorldTracker
from src.loaders import load_yolo_model, load_yolo_world_model, load_yolo_pose_model
from src.utils import (
    get_or_create_session_state_variable, 
    create_temp_video_file, 
    stream_frames, 
    capture_frames,
    process_frame,
    select_resolution
)
from streamlit_webrtc import webrtc, webrtc_streamer, WebRtcMode
from torch import cuda
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
from src.video_recognition import ModelInitializer, FramesPreprocessor, ResultFormatter

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
st.title("Pattern Detection (next pattern prediction) and Explanation for Cognitive Learning")
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
    get_or_create_session_state_variable("algortihm", default_value='Track Every Movement')
    get_or_create_session_state_variable("start_process_button_clicked", default_value=False)
    get_or_create_session_state_variable("processed", default_value=False)
    get_or_create_session_state_variable("stop_process_button_clicked", default_value=False)
    get_or_create_session_state_variable("classes_yolo_world", default_value=['person', 'white truck'])
    get_or_create_session_state_variable("classes_yolo_vx", default_value=None)
    get_or_create_session_state_variable("target_classes_yolo_vx", default_value=None)
    get_or_create_session_state_variable("tracker_ids", default_value=None)
    get_or_create_session_state_variable("target_tracker_ids", default_value=None)
    # Models
    device='cuda' if cuda.is_available() else 'cpu'
    model10n = load_yolo_model(model_name='yolov10n.pt')
    st.session_state.update(classes_yolo_vx=model10n.names)
    model10n.to(device)
    model8npose = load_yolo_pose_model(model_name='yolov8m-pose.pt')
    model8npose.to(device)
    yolo_tracker = YOLOTracker(modelDetect=model10n, poseModel=model8npose)
    modelYoloWorld = load_yolo_world_model(model_name='yolov8s-world.pt')

    # Recognition Model
    weights = R3D_18_Weights.DEFAULT
    preprocess = weights.transforms()
    preprocessor = FramesPreprocessor(preprocess)
    model_initializer = ModelInitializer(weights)
    recognition_model = model_initializer.initialize_model()
    
    # Step 3: Preprocess the video
    preprocessor = FramesPreprocessor(preprocess)


    flag = False
    source = None

    # Panels
    upload_panel, config_panel = st.columns([2, 3], gap='large')
    
    # Upload video
    with upload_panel:
        video_file = st.file_uploader("Upload a video", type=["mp4",], disabled=st.session_state.start_process_button_clicked)

        if video_file is not None:
            source = create_temp_video_file(video_file=video_file)
    
    if video_file:
        with config_panel:
            st.session_state.algortihm = st.selectbox(label='Algorithm', options=['Track Objects Movement', 'Track based on Prompt', 'Track Every Movement'], disabled=st.session_state.start_process_button_clicked)
            
            if st.session_state.algortihm == 'Track based on Prompt':
                st.session_state.update(classes_yolo_world=str(st.text_input(label='Enter Objects you want to track: ', placeholder='format as comma sperated: white truck, black car, bycycle....')).split(','))
                print(st.session_state.classes_yolo_world)
            
            if st.session_state.algortihm == 'Track Objects Movement':
                options = list(st.session_state.classes_yolo_vx.values())
                st.session_state.update(target_classes_yolo_vx=st.multiselect(label='Select Objects', options=options, default=options[:5]))
                print(st.session_state.target_classes_yolo_vx)
    
    # button to start processing
    if not st.session_state.processed and video_file:
        if not st.session_state.start_process_button_clicked:
            st.button("Track", on_click=lambda: st.session_state.update(start_process_button_clicked=True))
            pass
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

    if st.session_state.start_process_button_clicked: st.title(f"Insights - {st.session_state.algortihm}")
    # two more columns named objectTrackingPanel, segmentationPanel
    objectTrackingPanel, segmentationPanel = st.columns([1, 1], gap='large')
    
    if st.session_state.start_process_button_clicked and source:
        # panel = st.empty()
        if st.session_state.algortihm == 'Track Every Movement':
            
            optical_flow = OpticalFlowTracker(video_path=source, resolution=(1280, 720))
            optical_flow.process_video(objectTrackingPanel, segmentationPanel)
        
        if st.session_state.algortihm == 'Track Objects Movement':
            with objectTrackingPanel:
                # save_path = run_yolo_tracker(filename=source, modelDetect=modelDetect, modelSegment=modelSegment, file_index=1, resolution=resolution)
                save_path = yolo_tracker.process_video(
                    filename=source, stream_panels=[objectTrackingPanel, segmentationPanel], output_filename='out.mp4',
                    recognition_model=recognition_model, weights=weights, preprocessor=preprocessor
                    )
                # if save_path:
                #     panel.video(data='out.mp4', format='video/mp4', autoplay=True)
        
        if st.session_state.algortihm == 'Track based on Prompt':
                modelYoloWorld.to(device)
                modelYoloWorld.set_classes(st.session_state.classes_yolo_world)
                yolo_word_tracker = YOLOWorldTracker(modelDetect=modelYoloWorld, poseModel=model8npose)
                yolo_word_tracker.process_video(filename=source, stream_panels=[objectTrackingPanel, segmentationPanel], output_filename='out.mp4')
        
        st.session_state.start_process_button_clicked = False
    
        
        # Clean up the temporary file after processing
        # if source:
        #     if os.path.exists(source):
        #         os.remove(source)

if __name__ == "__main__":
    main()
    # Footer with Font Awesome icons
    footer_html = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <div style="text-align: center; margin-right: 10%;">
        <p>
            &copy; 2024, Trigent Software Inc. All rights reserved. |
            <a href="https://www.linkedin.com/company/trigent-software" target="_blank" aria-label="LinkedIn"><i class="fab fa-linkedin"></i></a> |
            <a href="https://www.twitter.com/trigent-software" target="_blank" aria-label="Twitter"><i class="fab fa-twitter"></i></a> |
            <a href="https://www.youtube.com/trigent-software" target="_blank" aria-label="YouTube"><i class="fab fa-youtube"></i></a>
        </p>
    </div>
    """

    # Custom CSS to make the footer sticky
    footer_css = """
    <style>
    .footer {
        position: fixed;
        z-index: 1000;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
    }
    [data-testid="stSidebarNavItems"] {
        max-height: 100%!important;
    }
    </style>
    """

    # Combining the HTML and CSS
    footer = f"{footer_css}<div class='footer'>{footer_html}</div>"

    # Rendering the footer
    st.markdown(footer, unsafe_allow_html=True)