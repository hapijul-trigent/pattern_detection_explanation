import torch
from model_utils import load_model  # Replace 'model_utils' with the actual module name

def test_load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device=device)
    assert model is not None, "Model should be loaded."
    assert next(model.parameters()).device == torch.device(device), f"Model should be on {device}."
