import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
from src.video_recognition import FramesPreprocessor, ModelInitializer, InferenceEngine

def test_frames_preprocessor():
    weights = R3D_18_Weights.DEFAULT
    preprocess = weights.transforms()
    
    video_tensor = torch.randint(0, 255, (3, 128, 128, 3), dtype=torch.uint8)
    preprocessor = FramesPreprocessor(preprocess)
    
    preprocessed_tensor = preprocessor.preprocess(video_tensor)
    assert preprocessed_tensor is not None, "Preprocessed tensor should not be None"
    assert isinstance(preprocessed_tensor, torch.Tensor), "Preprocessed tensor should be a PyTorch tensor"

def test_model_initializer():
    weights = R3D_18_Weights.DEFAULT
    initializer = ModelInitializer(weights)
    
    model = initializer.initialize_model()
    assert isinstance(model, torch.nn.Module), "The initialized model should be a PyTorch module"
    assert model is not None, "The model should be successfully initialized"

def test_inference_engine():
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    engine = InferenceEngine(model)
    
    dummy_batch = torch.randn(1, 3, 16, 112, 112)  # Dummy batch for a 16-frame video clip
    result = engine.infer(dummy_batch)
    
    assert result is not None, "Inference result should not be None"
    assert isinstance(result, torch.Tensor), "The result should be a PyTorch tensor"
