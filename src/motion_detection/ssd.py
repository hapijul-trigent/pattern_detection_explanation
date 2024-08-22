import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(device='cpu'):
    """
    Load the pre-trained SSD model.

    Args:
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        model: Loaded SSD model.
    """
    try:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'ssd300', pretrained=True)  # Adjust as necessary
        model.eval().to(device)
        logger.info("SSD model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_image(image):
    """
    Preprocess the input image for SSD.

    Args:
        image (PIL.Image): Input image.

    Returns:
        tensor: Preprocessed image tensor.
    """
    try:
        transform = T.Compose([
            T.ToTensor(),  # Convert to tensor and normalize
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def postprocess_output(output, threshold=0.5):
    """
    Postprocess the output from SSD.

    Args:
        output (tensor): Raw output from SSD.
        threshold (float): Confidence threshold for filtering detections.

    Returns:
        list: Processed detections with bounding boxes and labels.
    """
    try:
        output = output[0]  # Get first image's output
        boxes, labels, scores = output['boxes'], output['labels'], output['scores']
        results = []
        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:  # Confidence threshold
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': score.item(),
                    'class': label.item()
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
    st.title("SSD Object Detection")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device=device)
    
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
