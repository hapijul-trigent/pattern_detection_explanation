import cv2
import streamlit as st
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/workspaces/pattern_detection_explanation/models/yolov8n.pt')  # Ensure you have the correct path to your YOLOv8 model

def process_video(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file.name)
    
    # Create a video writer to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('outputs/test_output.mp4', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in the frame
        results = model(frame)
        
        # Draw bounding boxes and labels
        for bbox in results.xyxy[0].numpy():
            x1, y1, x2, y2, conf, cls = bbox
            label = f'Class {int(cls)} {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Write the frame with bounding boxes
        out.write(frame)
    
    # Release the video capture and writer
    cap.release()
    out.release()

    return 'outputs/test_output.mp4'


import streamlit as st

def main():
    st.title('Object Tracking with YOLOv8')

    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Process the video
        output_video = process_video(uploaded_file)
        
        # Display the output video
        st.write('Processed')
        st.video(output_video)

if __name__ == "__main__":
    main()
