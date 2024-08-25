import streamlit as st
from ultralytics import YOLO
import logging
from typing import Tuple


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path="models/yolov8n.pt") -> Tuple[YOLO, YOLO]:
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
        modelDetect = YOLO(model=model_path + "yolov10n.pt", task="detect")
        modelSegment = YOLO(model=model_path + "yolov8n-seg.pt", task="segment")
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        raise
    except RuntimeError as e:
        logger.error("Error loading model: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise

    return modelDetect, modelSegment

