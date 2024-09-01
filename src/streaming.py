import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Define the color range for object tracking
# Adjust these values to track different colors
lower_color = np.array([35, 140, 60])
upper_color = np.array([85, 255, 255])

def process_frame(frame):
    # Convert the frame to a numpy array
    img = frame.to_ndarray(format="bgr24")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask with the specified color range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)

        # Get the bounding box around the object
        x, y, w, h = cv2.boundingRect(c)

        # Draw the bounding box on the original image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img

def frame_callback(frame):
    # Process the frame using the color-based tracker
    img = process_frame(frame)

    # Return the modified frame
    return img

def main():
    st.title("WebRTC Object Tracker with Frame Callback")
    st.write("This application tracks an object based on color using WebRTC streaming from your camera.")

    webrtc_streamer(key="object-tracker", 
                    mode=WebRtcMode.SENDRECV, 
                    video_frame_callback=frame_callback)

if __name__ == "__main__":
    main()
