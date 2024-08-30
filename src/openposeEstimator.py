import cv2 as cv
import numpy as np
import argparse

class ArgumentParser:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='OpenPose human pose estimation using OpenCV.')
        parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
        parser.add_argument('--proto', help='Path to .prototxt')
        parser.add_argument('--model', help='Path to .caffemodel')
        parser.add_argument('--dataset', help='Specify what kind of model was trained. It could be (COCO, MPI, HAND) depends on dataset.')
        parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
        parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
        parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
        parser.add_argument('--scale', default=0.003922, type=float, help='Scale for blob.')
        return parser.parse_args()

class PoseEstimator:
    def __init__(self, proto_path, model_path, dataset, threshold, width, height, scale):
        self.net = cv.dnn.readNet(cv.samples.findFile(proto_path), cv.samples.findFile(model_path))
        self.dataset = dataset
        self.threshold = threshold
        self.width = width
        self.height = height
        self.scale = scale
        self.body_parts, self.pose_pairs = self._get_pose_data()

    def _get_pose_data(self):
        if self.dataset == 'COCO':
            return (
                { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9, "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14, "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 },
                [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
            )
        elif self.dataset == 'MPI':
            return (
                { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9, "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14, "Background": 15 },
                [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
            )
        elif self.dataset == 'HAND':
            return (
                { "Wrist": 0, "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4, "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8, "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12, "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16, "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20 },
                [ ["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"], ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"], ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"], ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"], ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"], ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"], ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"], ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"], ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"], ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"] ]
            )
        else:
            raise ValueError("Invalid dataset specified")

    def estimate_pose(self, frame):
        inp = cv.dnn.blobFromImage(frame, self.scale, (self.width, self.height), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()
        points = []
        for i in range(len(self.body_parts)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frame.shape[1] * point[0]) / out.shape[3]
            y = (frame.shape[0] * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > self.threshold else None)
        return points

class PoseRenderer:
    @staticmethod
    def draw_pose(frame, points, pose_pairs, body_parts):
        for pair in pose_pairs:
            partFrom, partTo = pair
            if partFrom in body_parts and partTo in body_parts:
                idFrom = body_parts[partFrom]
                idTo = body_parts[partTo]
                if points[idFrom] and points[idTo]:
                    cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                    cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = cv.getTickCount()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.imshow('OpenPose using OpenCV', frame)

def main():
    args = ArgumentParser.parse_args()
    pose_estimator = PoseEstimator(args.proto, args.model, args.dataset, args.thr, args.width, args.height, args.scale)
    cap = cv.VideoCapture(args.input if args.input else 0)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        points = pose_estimator.estimate_pose(frame)
        PoseRenderer.draw_pose(frame, points, pose_estimator.pose_pairs, pose_estimator.body_parts)

if __name__ == "__main__":
    main()
