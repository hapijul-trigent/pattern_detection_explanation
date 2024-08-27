import cv2
import numpy as np
import streamlit as st
import logging
import time
from collections import defaultdict
import supervision as sv


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
                    if len(track) > 30:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(57, 255, 20), thickness=8)
                    # cv2.polylines(annotated_frame, [points], isClosed=False, color=(57, 255, 20), thickness=8)

                    if points.shape[0] > 2:
                        # cv2.arrowedLine(annotated_frame, points[-2, 0], points[-1, 0], color=(0, 92, 255), thickness=16, line_type=cv2.LINE_AA, tipLength=0.3)
                        start_point = tuple(np.int32(points[0, 0]))
                        cv2.circle(annotated_frame, start_point, 5, (255, 0, 0), -1)
                        end_point = tuple(np.int32(points[-1, 0]))
                        cv2.circle(annotated_frame, end_point, 5, (0, 0, 255), -1)
                elapsed_time = time.time() - start
                fps = 1 / elapsed_time
                frame_time = 1.0 / fps
                if isinstance(annotated_frame, np.ndarray):
                    # Stream
                    stream_panel.image(annotated_frame, caption=f"Preview Tracking", channels="BGR")

                else:
                    logger.error("Error converting frame to numpy array.")
                delay = max(0, frame_time - elapsed_time)
                if delay > 0: time.sleep(delay)
                else: time.sleep(0.04); logger.warning('Slow Processing')
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
        # Release video sources
        video.release()
        logger.info(f"Finished processing video: {filename}")
        return filename
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None
    

# YOLOTracker
import logging
import time
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

class YOLOTracker:
    def __init__(self, modelDetect, modelSegment=None, resolution=None):
        self.modelDetect = modelDetect
        self.modelSegment = modelSegment
        self.resolution = resolution
        self.logger = logging.getLogger(__name__)
        self.track_history = defaultdict(list)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxCornerAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.tracer = sv.TraceAnnotator(thickness=3)
        self.heat_map_annotator = sv.HeatMapAnnotator(radius=30)


    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, List[Tuple[float, float]]]]:
        try:
            
            results = self.modelDetect(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            # print(detections)
            # print(dir(detections))
            detections = self.tracker.update_with_detections(detections)

            if detections.empty():
                return frame, frame

            # Access the attributes based on your `sv.Detections` object structure
            # boxes = detections.boxes.cpu().numpy()
            # scores = detections.scores.cpu().numpy()
            # class_ids = detections.class_ids.cpu().numpy()
            tracker_ids = detections.tracker_id

            # Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2)
            # bbox_data = boxes  # Adjust if necessary to match your attribute names

            # Prepare labels
            labels = [
                f"#{tracker_id} {results.names[class_id]}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
            ]

            labels_for_heatmap = [
                f"#{tracker_id}"
                for tracker_id
                in detections.tracker_id
            ]

            # Annotate the frame
            annotated_frame_heat = self.label_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels_for_heatmap)
            heat_map_frame = self.heat_map_annotator.annotate(scene=annotated_frame_heat, detections=detections)
            annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            annotated_frame = self.tracer.annotate( annotated_frame, detections=detections)

            


            # Prepare trajectory data
            # trajectory_data = {}
            # for detection, tracker_id in zip(detections, tracker_ids):
            #     bbox_center = (detection.boxes[0] + detection.boxes[2] / 2,
            #                    detection.boxes[1] + detection.boxes[3] / 2)
            #     track = self.track_history[tracker_id]
            #     track.append(bbox_center)
            #     if len(track) > 30:
            #         track.pop(0)
            #     trajectory_data[tracker_id] = track

            return annotated_frame, heat_map_frame #, bbox_data, scores, class_ids, tracker_ids, trajectory_data
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            return None #, None, None, None, None, None

    def process_video_with_sink(self, filename: str, stream_panels, output_filename: str = None) -> str:
        try:
            annotation_panel, heat_map_panel = stream_panels
            ### video config
            video_info = sv.VideoInfo.from_video_path(video_path=filename)
            frames_generator = sv.get_video_frames_generator(
                source_path=filename, stride=1
            )
            frame_time = 1.0 / 24.0
            ### Detect, track, annotate, save
            with sv.VideoSink(target_path=filename, video_info=video_info) as sink:
                for frame in frames_generator:
                    start = time.time()
                    annotated_frame = self.process_frame(frame)

                    if isinstance(annotated_frame, np.ndarray):
                        annotation_panel.image(annotated_frame, channels="BGR", use_column_width=True)
                        # heat_map_panel.image(an, channels="BGR", use_column_width=True)
                        sink.write_frame(frame=annotated_frame)
                    else:
                        self.logger.error("Error processing frame.")

                    elapsed_time = time.time() - start
                    delay = max(0, frame_time - elapsed_time)
                    time.sleep(max(0.01, delay))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            self.logger.info(f"Finished processing video: {filename}")
            return output_filename
        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            return None

    def process_video(self, filename: str, stream_panels, output_filename: str = None) -> str:
        self.tracker.reset()
        try:
            # Setup Panels
            with stream_panels[0]: annotation_panel = st.empty()
            with stream_panels[1]:  heat_map_panel = st.empty()

            video = cv2.VideoCapture(filename)
            if not video.isOpened():
                raise ValueError(f"Error opening video file or stream: {filename}")

            fps = video.get(cv2.CAP_PROP_FPS)
            fps = fps if fps > 0 else 30
            frame_time = 1.0 / fps

            if output_filename:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

            while video.isOpened():
                start = time.time()
                success, frame = video.read()
                if not success:
                    self.logger.warning(f"Failed to read frame from {filename}.")
                    break

                annotated_frame, heat_map_frame  = self.process_frame(frame)

                if isinstance(annotated_frame, np.ndarray):
                    if output_filename:
                        out_video.write(annotated_frame)
                    annotation_panel.image(annotated_frame, channels="BGR", use_column_width=True, caption='Annotated Traces')
                    heat_map_panel.image(heat_map_frame, channels="BGR", use_column_width=True, caption='Heat Map Traces')
                else:
                    self.logger.error("Error processing frame.")

                elapsed_time = time.time() - start
                delay = max(0, frame_time - elapsed_time)
                # time.sleep(max(0.01, delay))

                if cv2.waitKey(1) & 0xFF == ord('q') or st.session_state.stop_process_button_clicked:
                    self.logger.info('Stop Button pressed by user.')
                    break

            video.release()
            if output_filename:
                out_video.release()
            self.logger.info(f"Finished processing video: {filename}")
            return output_filename
        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            return None