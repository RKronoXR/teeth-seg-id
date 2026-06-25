import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.cuda.is_available())

model = maskrcnn_resnet50_fpn(weights=None, num_classes=33)
model.eval()

print("Mask R-CNN baseline OK")
