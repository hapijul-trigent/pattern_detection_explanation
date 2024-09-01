import cv2 as cv
import numpy as np
import argparse

class BodyParts:
    """Constants for body parts and pose pairs."""
    COCO_BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
    }
    COCO_POSE_PAIRS = [
        ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
    ]

class PoseEstimator:
    """Handles pose estimation using OpenCV's DNN module."""
    def __init__(self, model_path, input_width, input_height, threshold):
        self.net = cv.dnn.readNetFromTensorflow(model_path)
        self.input_width = input_width
        self.input_height = input_height
        self.threshold = threshold

    def estimate_pose(self, frame):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        inp_blob = cv.dnn.blobFromImage(frame, 1.0, (self.input_width, self.input_height),
                                        (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.net.setInput(inp_blob)
        output = self.net.forward()
        output = output[:, :57, :, :]   # MobileNet output [1, 57, -1, -1]
        
        points = []
        for i in range(len(BodyParts.COCO_BODY_PARTS)):
            heatmap = output[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatmap)
            x = (frame_width * point[0]) / output.shape[3]
            y = (frame_height * point[1]) / output.shape[2]
            points.append((int(x), int(y)) if conf > self.threshold else None)
        return points

class PoseRenderer:
    """Renders detected poses on the image frame."""
    
    # Define a color map for each pose pair
    POSE_PAIR_COLORS = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (75, 0, 130),   # Indigo
        (199, 21, 133), # Medium Violet Red
        (255, 20, 147), # Deep Pink
        (147, 112, 219),# Medium Purple
        (0, 128, 0),    # Dark Green
        (128, 0, 0),    # Maroon
        (0, 0, 128),    # Navy
    ]

    @staticmethod
    def draw_pose(frame, points):
        for idx, pair in enumerate(BodyParts.COCO_POSE_PAIRS):
            part_from = pair[0]
            part_to = pair[1]
            id_from = BodyParts.COCO_BODY_PARTS[part_from]
            id_to = BodyParts.COCO_BODY_PARTS[part_to]
            
            if points[id_from] and points[id_to]:
                color = PoseRenderer.POSE_PAIR_COLORS[idx % len(PoseRenderer.POSE_PAIR_COLORS)]
                cv.line(frame, points[id_from], points[id_to], color, 3)
                cv.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, color, cv.FILLED)
                cv.ellipse(frame, points[id_to], (3, 3), 0, 0, 360, color, cv.FILLED)
  
class VideoCaptureHandler:
    """Handles video capture from file or camera."""
    def __init__(self, input_source):
        self.cap = cv.VideoCapture(input_source if input_source else 0)

    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

def main():
    pose_estimator = PoseEstimator(model_path='graph_opt.pb', width=368, height=368, threshold=0.2)
    video_handler = VideoCaptureHandler('/content/human-pose-estimation-opencv/dance.mp4')

    while cv.waitKey(1) < 0:
        has_frame, frame = video_handler.get_frame()
        if not has_frame:
            cv.waitKey()
            break

        points = pose_estimator.estimate_pose(frame)
        PoseRenderer.draw_pose(frame, points)
        cv.imshow('OpenPose using OpenCV', frame)

    video_handler.release()
    cv.destroyAllWindows()
