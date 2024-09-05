# Pattern Detection Explanation

**Pattern Detection and Motion Explanation for Cognitive Learning**

This repository provides a Streamlit-based application that uses advanced machine learning techniques to detect and predict motion patterns in videos and images. It offers insights into these predictions, making it ideal for researchers, educators, and motion analysis enthusiasts. The application uses state-of-the-art models for object detection, motion tracking, pose estimation, and action recognition, with explainability features for cognitive learning purposes.

---

## Features

- **Object Detection and Tracking**: Leverages YOLO models for object detection and tracking in videos.
- **Optical Flow Tracking**: Uses Lucas-Kanade optical flow method to detect and track motion between video frames.
- **Pose Estimation**: Implements pose detection using OpenCV's DNN module, based on COCO body parts and pose pairs.
- **Action Recognition**: Recognizes actions in video streams using a pre-trained R3D model from PyTorch.
- **LIME-based Explainability**: Provides visual explanations for model predictions using LIME, showing how decisions are made by the model.

---

## Directory Structure

```
pattern_detection_explanation/
│
├── .streamlit/
│   └── config.toml                # Streamlit configuration file
├── static/
│   └── images/                    # Images used in the application (e.g., logos)
│       ├── Trigent_Logo.png
│       └── ... (additional images)
├── src/
│   ├── __init__.py                # Init file for the src directory
│   ├── detection.py               # Optical flow motion detection logic
│   ├── explain_recognition.py     # LIME-based explanation for action recognition
│   ├── loaders.py                 # Model loading utilities
│   ├── openposeEstimator.py       # Pose estimation using OpenCV
│   ├── optical_flow_tracker.py    # Optical flow tracker for motion detection
│   ├── streaming.py               # Video streaming functions
│   ├── utils.py                   # Helper functions for video processing
│   ├── video_recognition.py       # Action recognition with PyTorch models
│   ├── yolo_world_tracker.py      # YOLO-based object tracking with pose estimation
│   └── yolovx_trackers.py         # Advanced object trackers with YOLO models
├── tests/
│   ├── test_drawing.py            # Test for drawing functions
│   ├── test_model.py              # Test for model loading
│   ├── test_postprocessing.py     # Test for post-processing model output
│   └── test_preprocessing.py      # Test for image preprocessing
├── app.py                         # Main Streamlit application script
├── README.md                      # Project overview and instructions
├── requirements.txt               # Python dependencies
└── setup.sh                       # Setup script for deployment
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hapijul-trigent/pattern_detection_explanation.git
   cd pattern_detection_explanation
   ```

2. **Install dependencies**:
   Make sure you have Python 3.7 or above installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Upload a Video**: Upload any `.mp4` video to the application.
2. **Choose a Tracking Algorithm**:
   - Track based on objects (YOLO-based detection)
   - Track all movements (optical flow)
   - Track based on prompt (specific objects)
3. **View Motion Insights**: The application will detect and track objects, estimate poses, or provide motion vectors.
4. **Cognitive Learning Insights**: LIME will provide visual explanations of the detected patterns.

---

## Models Used

- **YOLOv8** for object detection.
- **Optical Flow** using the Lucas-Kanade method.
- **R3D_18** from PyTorch for action recognition.
- **LIME** for model explainability.
- **OpenPose** for pose estimation using OpenCV.

---

## Testing

To run the test suite, use the following command:

```bash
pytest
```
---

## Contact

For further inquiries, feel free to contact the maintainers at [Trigent Software Inc](https://trigent.com).

---
