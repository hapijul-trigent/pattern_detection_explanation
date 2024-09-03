
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch
import streamlit as st


class FramesPreprocessor:
    def __init__(self, preprocess_transform):
        self.preprocess_transform = preprocess_transform

    def preprocess(self, video_tensor):
        # Optionally shorten duration
        video_tensor = video_tensor
        return self.preprocess_transform(video_tensor).unsqueeze(0)

class ModelInitializer:
    def __init__(self, weights):
        self.weights = weights
        self.model = r3d_18(weights=weights)
    
    def initialize_model(self):
        self.model.eval()
        return self.model

class InferenceEngine:
    def __init__(self, model):
        self.model = model

    def infer(self, batch):
        with torch.no_grad():
            prediction = self.model(batch).squeeze(0).softmax(0)
        return prediction

class ResultFormatter:
    def __init__(self, categories):
        self.categories = categories

    def format_result(self, prediction):
        label = prediction.argmax().item()
        score = prediction[label].item()
        category_name = self.categories[label]
        return f"{category_name}: {round(100 * score, 2)}%"

def infer_action(frame_tensor, recognition_model, weights, preprocessor):
    """ Action Inference"""

    batch = preprocessor.preprocess(frame_tensor)

    # Step 4: Perform inference
    engine = InferenceEngine(recognition_model)
    prediction = engine.infer(batch)

    # Step 5: Format and print the result
    formatter = ResultFormatter(weights.meta["categories"])
    return formatter.format_result(prediction)
