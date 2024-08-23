import cv2
import numpy as np

def annotate_frame_with_ssd(frame: np.ndarray) -> np.ndarray:
    """
    Annotates the input frame with bounding boxes and labels detected by SSD.

    Args:
        frame (np.ndarray): The input frame to be processed, expected to be a BGR image.

    Returns:
        np.ndarray: The annotated frame with bounding boxes and labels.
    """
    # Load the pre-trained SSD model and configuration files
    net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/mobilenet_iter_73000.caffemodel")

    # Prepare the frame for SSD
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))
    net.setInput(blob)
    detections = net.forward()

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:  # Confidence threshold
            # Extract the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label on the frame
            label = f"Object {i}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Detected: {label}")
    return frame
