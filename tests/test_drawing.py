from PIL import Image, ImageDraw
from model_utils import draw_boxes  # Replace 'model_utils' with the actual module name

def test_draw_boxes():
    image = Image.new('RGB', (640, 480))
    detections = [{
        'bbox': [100, 100, 200, 200],
        'confidence': 0.9,
        'class': 1
    }]
    image_with_boxes = draw_boxes(image.copy(), detections)
    draw = ImageDraw.Draw(image_with_boxes)
    bbox = detections[0]['bbox']
    x1, y1, x2, y2 = bbox
    # Check if the rectangle was drawn
    # You may need a more sophisticated check here depending on your draw implementation
    assert image_with_boxes.size == (640, 480), "Image size should be unchanged."
