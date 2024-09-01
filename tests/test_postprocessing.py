import torch
from model_utils import postprocess_output  # Replace 'model_utils' with the actual module name

def test_postprocess_output():
    output = [{
        'boxes': torch.tensor([[100, 100, 200, 200]]),
        'labels': torch.tensor([1]),
        'scores': torch.tensor([0.9])
    }]
    threshold = 0.5
    detections = postprocess_output(output, threshold=threshold)
    assert len(detections) == 1, "There should be one detection."
    assert detections[0]['class'] == 1, "Class label should be 1."
    assert detections[0]['confidence'] > threshold, "Confidence should be greater than threshold."
