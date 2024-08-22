import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import logging
from PIL import Image



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
st.title("Entities in Clinical Trial Abstracts")  # Placeholder for title
# Placeholder for caption
st.caption("This model extracts to trial design, diseases, drugs, population,Heart_Disease, Hyperlipidemia, Diabetes, Age, Test, Test_Result, Birth_Entity, Drug_BrandName, Date, etc. relevant entities from clinical trial abstracts.")


# Function to process video and detect motion
def process_video(video_file):
    # Create a temporary directory to store processed video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name
        
    output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")

    # Initialize video capture and writer
    video_capture = cv2.VideoCapture(temp_file_path)
    if not video_capture.isOpened():
        st.error("Failed to open video file.")
        return None
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create Background Subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Apply background subtractor
        foreground_mask = bg_subtractor.apply(frame)
        _, threshold = cv2.threshold(foreground_mask, 120, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        video_writer.write(frame)

    video_capture.release()
    video_writer.release()

    # Clean up temporary file
    os.remove(temp_file_path)

    return output_path


uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file, format="video/mp4")

    with st.spinner("Processing video..."):
        output_video_path = process_video(uploaded_file)
        print(output_video_path)

    if output_video_path:
        st.video(output_video_path)
        st.success("Motion detection complete!")
    else:
        st.error("Failed to process video.")
