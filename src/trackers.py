import cv2
import numpy as np
import streamlit as st
import logging
import time
from collections import defaultdict



### YOLOv8
def run_yolo_tracker(filename, modelDetect, modelSegment, file_index, resolution) -> str:
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    Captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. Runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.
    """
    logger = logging.getLogger(__name__)
    try:
        # Open the video file or stream
        video = cv2.VideoCapture(filename)
        if not video.isOpened():
            raise ValueError(f"Error opening video file or stream: {filename}")
        
        # if resolution:
        #     width, height = resolution
        #     video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #     video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        frame_time = 1.0 / fps
        stream_panel = st.empty()
        track_history = defaultdict(lambda: [])

        while video.isOpened():
            start = time.time()
            success, frame = video.read()
            if not success:
                logger.warning(f"Failed to read frame from {filename}.")
                break

            try:
                results = modelDetect.track(frame, persist=True, conf=0.15)
                # results_seg = modelDetect.track(modelSegment, persist=True, conf=0.15)
                if not results or not results[0].boxes:
                    logger.warning("No tracking results found.")
                    continue

                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                annotated_frame = results[0].plot()
                # annotated_frame = frame

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    # if len(track) > 24:
                        # track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(57, 255, 20), thickness=8)
                    # cv2.polylines(annotated_frame, [points], isClosed=False, color=(57, 255, 20), thickness=8)

                    if points.shape[0] > 2:
                        cv2.arrowedLine(annotated_frame, points[-2, 0], points[-1, 0], color=(0, 92, 255), thickness=16, line_type=cv2.LINE_AA, tipLength=0.3)
                elapsed_time = time.time() - start
                fps = 1 / elapsed_time
                if isinstance(annotated_frame, np.ndarray):
                    # Stream
                    stream_panel.image(annotated_frame, caption=f"Preview Tracking", channels="BGR")

                else:
                    logger.error("Error converting frame to numpy array.")
                delay = max(0, frame_time - elapsed_time)
                if delay > 0: time.sleep(delay)
                else: time.sleep(0.03); logger.warning('Slow Processing')
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
        # Release video sources
        video.release()
        logger.info(f"Finished processing video: {filename}")
        return filename
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None



# Optical FLow Tracker
class OpticalFlowTracker:

    def __init__(self, video_path, resolution=None):
        self.video_path = video_path
        self.resolution = resolution
        self.trajectory_len = 30
        self.detect_interval = 3
        self.trajectories = []
        self.frame_idx = 0
        self.prev_gray = None

        # Optical Flow Parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.feature_params = dict(
            maxCorners=20,
            qualityLevel=0.3,
            minDistance=10,
            blockSize=7
        )

    def process_video(self, frame_placeholder_col, mask_placeholder_col):
        global logger
        logger = logging.getLogger(__name__)
        logger.info(f"Starting video processing: {self.video_path} with resolution: {self.resolution}")

        video = cv2.VideoCapture(self.video_path)
        
        if not video.isOpened():
            logger.error(f"Error opening video file: {self.video_path}")
            st.error("Error opening video file. Check the file path and try again.")
            return
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        frame_time = 1.0 / fps
        if self.resolution:
            width, height = self.resolution
            video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        with frame_placeholder_col: frame_placeholder = st.empty()
        with mask_placeholder_col:  mask_placeholder = st.empty()

        while True:
            start = time.time()
            success, frame = video.read()
            if not success:
                logger.info("End of video file or failed to read frame.")
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = frame.copy()

            if self.prev_gray is not None and len(self.trajectories) > 0:
                try:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([trajectory[-1] for trajectory in self.trajectories]).reshape(-1, 1, 2)
                    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = d < 1

                    new_trajectories = []

                    for trajectory, (x, y), good_flag in zip(self.trajectories, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        trajectory.append((x, y))
                        if len(trajectory) > self.trajectory_len:
                            del trajectory[0]
                        new_trajectories.append(trajectory)
                        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

                    self.trajectories = new_trajectories
                    for trajectory in self.trajectories:
                        cv2.polylines(img, [np.int32(trajectory)], False, (240, 0, 255), thickness=1)
                        
                        # Circling on both points
                        if len(trajectory) > 0:
                            start_point = tuple(np.int32(trajectory[0]))
                            cv2.circle(img, start_point, 5, (0, 255, 0), -1)
                            end_point = tuple(np.int32(trajectory[-1]))
                            cv2.circle(img, end_point, 5, (0, 0, 255), -1)
                    
                    cv2.putText(img, 'track count: %d' % len(self.trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
                except Exception as e:
                    logger.error(f"Error in optical flow computation: {e}")

            if self.frame_idx % self.detect_interval == 0:
                try:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255

                    for x, y in [np.int32(trajectory[-1]) for trajectory in self.trajectories]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                        


                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.trajectories.append([(x, y)])
                except Exception as e:
                    logger.error(f"Error detecting features: {e}")

            self.frame_idx += 1
            self.prev_gray = frame_gray

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) if 'mask' in locals() else np.zeros_like(img_rgb)

            elapsed_time = time.time() - start
            fps = 1 / elapsed_time
            cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (240,0,255), 2)

            frame_placeholder.image(img_rgb, caption='Optical Flow Tracking', channels='RGB', use_column_width=True)
            mask_placeholder.image(mask_rgb, caption='Feature Pattern', channels='RGB', use_column_width=True)
            delay = max(0, frame_time - elapsed_time)
            print("DELAY:::", delay)
            if delay > 0: time.sleep(delay)
            else: time.sleep(0.03); logger.warning('Slow Processing')

        video.release()
        logger.info("Video processing completed.")
        st.success("Video processing completed.")