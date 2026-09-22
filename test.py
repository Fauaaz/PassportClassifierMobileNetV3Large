import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from pathlib import Path

dataset_path = Path('"C:\\Users\\USER\\Desktop\\pic_dataset\\reject\\00E681E8-F05F-4703-A0B4-2900613F85A7 - Copy.png"')

data_transforms = {
    'test': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    "test": datasets.ImageFolder(
        dataset_path,
        data_transforms["test"]
    )
}


image = Image.open(dataset_path).convert("RGB")

image_tensor = data_transforms["test"](image)
image_tensor = image_tensor.unsqueeze(0)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2,
                                             shuffle=True, num_workers=0)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show(block=True)

# Get a batch of testing data
inputs, classes = next(iter(dataloaders['test']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow()

model = torch.load(
    "fine_tuned_model.pth",
    map_location="cpu",
    weights_only=False
)

print("Loaded successfully!")
print(type(model))

model.eval()

with torch.no_grad():
    outputs = model(inputs)

print(outputs)

prediction = torch.argmax(outputs, dim= 1)
print(prediction)
for i in prediction:
  if i == 0:
    print('Approved')
  else:
    print('Rejected')