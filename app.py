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
st.title("Video Processing Frame by Frame")
# Placeholder for caption
st.caption("This model extracts to trial design, diseases, drugs, population,Heart_Disease, Hyperlipidemia, Diabetes, Age, Test, Test_Result, Birth_Entity, Drug_BrandName, Date, etc. relevant entities from clinical trial abstracts.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
import cv2
import tempfile
import os

def process_frame(frame):
    # Apply any processing to the frame (e.g., grayscale conversion)
    # Example: Convert the frame to grayscale
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return processed_frame

def main():


    # Upload video
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Create a temporary file to store the video
        tfile = tempfile.NamedTemporaryFile(delete=False, dir='outputs/temp')
        tfile.write(video_file.read())

        # Open the video file
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()  # Placeholder for displaying the video frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame = process_frame(frame)

            # Convert the processed frame to RGB (needed for Streamlit display)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)

            # Display the frame in Streamlit
            stframe.image(processed_frame_rgb, channels="RGB")

        # Release the video capture object and remove the temporary file
        cap.release()
        os.remove(tfile.name)

if __name__ == "__main__":
    main()
