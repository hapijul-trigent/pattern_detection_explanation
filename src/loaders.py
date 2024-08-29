import streamlit as st
from ultralytics import YOLO, YOLOWorld
import logging
from typing import Tuple


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_name="yolov10n.pt") -> YOLO:
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
        model = YOLO(model='models/' + model_name, task="detect")
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        raise
    except RuntimeError as e:
        logger.error("Error loading model: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise

    return model


@st.cache_resource(show_spinner=False)
def load_yolo_world_model(model_name) -> YOLOWorld:
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
        model = YOLOWorld(model='models/' + model_name)
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        raise
    except RuntimeError as e:
        logger.error("Error loading model: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise

    return model


@st.cache_resource(show_spinner=False)
def load_yolo_pose_model(model_name="yolov8n-pose.pt") -> YOLO:
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
        model = YOLO(model='models/' + model_name, task="pose")
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        raise
    except RuntimeError as e:
        logger.error("Error loading model: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise

    return model