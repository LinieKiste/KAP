#!/bin/env python3
import pydicom
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils.loader import DicomDataset
import os

training_data= DicomDataset("data/train.csv")
train_dataloader = DataLoader(training_data, batch_size=10)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
