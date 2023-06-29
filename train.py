import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import time
from dataset import pascalVOC, display_bnd
from model import Efficient_YOLO
from loss import YOLOLoss
import cv2 as cv
import tqdm

def train():
    dataset = pascalVOC(train_len=8000)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    device = torch.device('cuda')
    epochs = 1
    lr = 3e-6
    criterion = YOLOLoss()
    model = Efficient_YOLO().to(device)
    model.load_state_dict(torch.load('weights/best_weights.pth'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    for i in range(epochs):
        total_iterations = 2500
        progress_bar  = tqdm.tqdm(total=total_iterations, desc='Epoch', unit='iter')
        total_loss = 0
        for bidx, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            progress_bar.update(1)            
            total_loss+=loss
        if total_loss < best_loss:
            best_loss = total_loss
        torch.save(model.state_dict(), 'weights/current_weights.pth')
        progress_bar.close()
        print(f'Epoch {i+1} | Loss {total_loss} | ')

def test():
    dataset = pascalVOC(train_len=8000, test_len=5000, train=False)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device('cuda')
    model = Efficient_YOLO().eval()
    model.load_state_dict(torch.load('weights/current_weights.pth'))
    for bidx, (x,y) in enumerate(train_loader):
        outputs = model(x)
        outputs = outputs.squeeze(0)
        arr = outputs.tolist()
        x = x.squeeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        x = ((x*std) + mean) * 255
        x = x.permute(1,2,0).to(torch.int).numpy()
        y = y.squeeze(0).tolist()
        x = x.astype(np.uint8)
        display_bnd(x, arr)

def main():
    test()

if __name__ == '__main__':
    main()