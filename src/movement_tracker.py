import cv2
import numpy as np
import streamlit as st
import logging
import threading
from collections import defaultdict



### YOLOv8
def run_yolo_tracker(filename, model, file_index, resolution) -> str:
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    Captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. Runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
        Press 'q' to quit the video display window.
    """
    logger = logging.getLogger(__name__)
    try:
        # Open the video file or stream
        video = cv2.VideoCapture(filename)
        if not video.isOpened():
            raise ValueError(f"Error opening video file or stream: {filename}")

        stream_panel = st.empty()
        track_history = defaultdict(lambda: [])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_path = f'outputs/processed/new.mp4'
        video_writer = cv2.VideoWriter(save_path, fourcc, 24, resolution)

        if not video_writer.isOpened():
            print("Error: Could not open video file for writing.")
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                logger.warning(f"Failed to read frame from {filename}.")
                break

            try:
                # frame = cv2.resize(frame, resolution)
                print(frame.shape)
                results = model.track(frame, persist=True)
                if not results or not results[0].boxes:
                    logger.warning("No tracking results found.")
                    continue

                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # annotated_frame = results[0].plot()
                annotated_frame = frame

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 24:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(57, 255, 20), thickness=16)

                    if points.shape[0] > 2:
                        cv2.arrowedLine(annotated_frame, points[-2, 0], points[-1, 0], color=(0, 92, 255), thickness=24, line_type=cv2.LINE_AA, tipLength=0.3)
                    
                if isinstance(annotated_frame, np.ndarray):
                    # Stream
                    stream_panel.image(annotated_frame, caption=f"Preview_Stream_{file_index}", channels="BGR")
                    video_writer.write(annotated_frame)

                else:
                    logger.error("Error converting frame to numpy array.")
            except Exception as e:
                logger.error(f"Error processing frame: {e}")

        # Release video sources
        video.release()
        video_writer.release()
        logger.info(f"Finished processing video: {filename}")
        return filename
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


