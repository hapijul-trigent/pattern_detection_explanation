from PIL import Image
import torch
from model_utils import preprocess_image  # Replace 'model_utils' with the actual module name

def test_preprocess_image():
    image = Image.new('RGB', (640, 480))  # Create a dummy image
    tensor = preprocess_image(image)
    assert tensor.shape == (1, 3, 640, 640), "Tensor shape should be (1, 3, 640, 640)."
    assert isinstance(tensor, torch.Tensor), "Output should be a tensor."
