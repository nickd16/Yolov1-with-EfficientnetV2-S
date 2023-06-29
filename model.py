import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time

class Efficient_YOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.conv1 = nn.Conv2d(1280, 1280, kernel_size=2, stride=2)
        self.convFC = nn.Conv2d(1280, 1280, kernel_size=1)
        self.convOut = nn.Conv2d(in_channels=1280, out_channels=30, kernel_size=1)

    def forward(self, x):
        features = self.model.features
        for layer in features:
            x = layer(x)
        x = self.conv1(x)
        x = self.convFC(x)
        x = self.convOut(x)
        x = x.permute(0, 2, 3, 1)
        return x

def main():
    device = torch.device('cuda')
    model = Efficient_YOLO().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    tensor = torch.randn((4,3,448,448)).to(device)
    s = time.time()
    print(model(tensor).shape)
    e = time.time()
    print(e-s)

if __name__ == '__main__':
    main()