import torch
from PIL import Image
from matplotlib import pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/tumor_detect/weights/best.pt')
results = model('sample_mri.jpg')
results.print()
results.show()  # opens image with bounding box
