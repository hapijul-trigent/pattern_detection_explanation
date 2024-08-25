import streamlit as st
from ultralytics import YOLO
import logging



@st.cache_resource(show_spinner=False)
def load_yolov8(model_path="models/yolov8n.pt"):
    """
    Load YOLO models and cache them to optimize loading times.

    Returns:
        Model: Model (YOLOv8n).

    Raises:
        FileNotFoundError: If the model files are not found.
        RuntimeError: If there is an error loading the models.
    """
    logger = logging.getLogger(__name__)
    try:
        # Load the YOLO models
        model1 = YOLO(model=model_path, task="detect")
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        raise
    except RuntimeError as e:
        logger.error("Error loading model: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise

    return model1
