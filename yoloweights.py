import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.info()

print(model.model)

layers = list(model.model.modules())


conv_layer = None
for layer in layers:
    if isinstance(layer, torch.nn.Conv2d):
        conv_layer = layer
        break

print(f"\n CNN : {conv_layer} \n")
if conv_layer:

    conv_weights = conv_layer.weight.cpu().detach().numpy()

    print("Weights Shape:", conv_weights.shape)
    print(conv_weights)

    plt.imshow(conv_weights[0, 0, :, :], cmap="gray")
    plt.title("First Filter of First Layer")
    plt.show()
else:
    print("No convolutional layer found!")




