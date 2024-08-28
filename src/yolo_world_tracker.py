import cv2
import numpy as np
import streamlit as st
import logging
import time
from collections import defaultdict
import supervision as sv
from ultralytics import YOLOWorld
import logging
import time
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import torch

class YOLOWorldTracker:
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
            detections = self.tracker.update_with_detections(detections)

            # if detections.empty():
            #     return frame, frame

            boxes = detections.xyxy
            scores = detections.confidence
            class_ids = detections.class_id
            tracker_ids = detections.tracker_id

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


            return annotated_frame, heat_map_frame
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            return None #, None, None, None, None, None


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

                if cv2.waitKey(1) & 0xFF == ord('q'):
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