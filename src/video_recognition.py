
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch


class FramesPreprocessor:
    def __init__(self, preprocess_transform):
        self.preprocess_transform = preprocess_transform

    def preprocess(self, video_tensor):
        # Optionally shorten duration
        video_tensor = video_tensor[:16]
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

def main(video_path: str):
    video_tensor = frames_tensor

    # Step 2: Initialize model and preprocessing transforms
    weights = R3D_18_Weights.DEFAULT
    preprocess = weights.transforms()
    model_initializer = ModelInitializer(weights)
    model = model_initializer.initialize_model()
    
    # Step 3: Preprocess the video
    preprocessor = FramesPreprocessor(preprocess)
    batch = preprocessor.preprocess(video_tensor)

    # Step 4: Perform inference
    engine = InferenceEngine(model)
    prediction = engine.infer(batch)

    # Step 5: Format and print the result
    formatter = ResultFormatter(weights.meta["categories"])
    result = formatter.format_result(prediction)
    print(result)
