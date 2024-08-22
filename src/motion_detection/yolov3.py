import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_name='yolov6s', device='cpu'):
    """
    Load the YOLOv6 model.

    Args:
        model_name (str): Name of the YOLOv6 model variant (e.g., 'yolov6s').
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        model: Loaded YOLOv6 model.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)  # Adjust as necessary
        model.to(device).eval()
        logger.info("YOLOv6 model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_image(image):
    """
    Preprocess the input image for YOLOv6.

    Args:
        image (PIL.Image): Input image.

    Returns:
        tensor: Preprocessed image tensor.
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to the model's input size
            transforms.ToTensor(),  # Convert to tensor
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def postprocess_output(output):
    """
    Postprocess the output from YOLOv6.

    Args:
        output (tensor): Raw output from YOLOv6.

    Returns:
        list: Processed detections with bounding boxes and labels.
    """
    try:
        detections = output.xyxy[0].cpu().numpy()  # Get detections from the output
        results = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf > 0.5:  # Confidence threshold
                results.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class': int(cls)
                })
        return results
    except Exception as e:
        logger.error(f"Error postprocessing output: {e}")
        raise

def draw_boxes(image, detections):
    """
    Draw bounding boxes on the image.

    Args:
        image (PIL.Image): Original image.
        detections (list): List of detected objects with bounding boxes.

    Returns:
        PIL.Image: Image with bounding boxes drawn.
    """
    try:
        draw = ImageDraw.Draw(image)
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            label = f'Class {det["class"]} ({det["confidence"]:.2f})'
            draw.text((x1, y1 - 10), label, fill='green')
        return image
    except Exception as e:
        logger.error(f"Error drawing boxes: {e}")
        raise

def main():
    st.title("YOLOv6 Object Detection")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_name='yolov6s', device=device)
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting...")
        
        img_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            output = model(img_tensor)
        detections = postprocess_output(output)
        image_with_boxes = draw_boxes(image.copy(), detections)
        
        st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)

if __name__ == "__main__":
    main()
