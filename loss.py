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

def IOU(tensor1, tensor2):
    x1 = torch.min(tensor1[:,:,:,0]-((1/2)*tensor1[:,:,:,2]), tensor2[:,:,:,0]-((1/2)*tensor2[:,:,:,2]))
    y1 = torch.min(tensor1[:,:,:,1]-((1/2)*tensor1[:,:,:,3]), tensor2[:,:,:,1]-((1/2)*tensor2[:,:,:,3]))
    x2 = torch.max(tensor1[:,:,:,0]+((1/2)*tensor1[:,:,:,2]), tensor2[:,:,:,0]+((1/2)*tensor2[:,:,:,2]))
    y2 = torch.max(tensor1[:,:,:,1]+((1/2)*tensor1[:,:,:,3]), tensor2[:,:,:,1]+((1/2)*tensor2[:,:,:,3]))
    int = ((x2-x1).clamp(0) * (y2-y1).clamp(0))
    a1 = tensor1[:,:,:,2] * tensor1[:,:,:,3]
    a2 = tensor2[:,:,:,2] * tensor2[:,:,:,3]
    return (int/(a1+a2-int+1e-6))

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, x, y, lambd_coord=5, lambd_noobj=0.5):
        bnd1 = x[:,:,:,1:5]
        bnd2 = x[:,:,:,6:10]
        bndy = y[:,:,:,1:5]
        iou1 = IOU(bnd1, bndy)
        iou2 = IOU(bnd2, bndy)
        bnd_score = torch.stack([(iou1 >= iou2) for _ in range(5)], dim=3)
        bnd1s = x[:,:,:,0:5] * bnd_score
        bnd2s = x[:,:,:,5:10] * ~bnd_score
        x[:,:,:,0:5] = bnd1s+bnd2s
        x[6:10] = 0
        obj_score = torch.stack([(y[:,:,:,0] >= 0.5) for _ in range(30)], dim=3)
        Iobj = x * obj_score
        Nobj = x * ~obj_score
        sign1 = torch.sign(Iobj[:,:,:,3:5])
        sign2 = torch.sign(y[:,:,:,3:5])
        
        coord_loss = lambd_coord * self.mse(Iobj[:,:,:,1:3], y[:,:,:,1:3])
        dim_loss = lambd_coord * self.mse(sign1*(torch.sqrt(torch.abs(Iobj[:,:,:,3:5])+ 1e-6)), sign2*(torch.sqrt(torch.abs(y[:,:,:,3:5])+ 1e-6)))
        obj_conf_loss = self.mse(Iobj[:,:,:,0], y[:,:,:,0])
        noobj_conf_loss = lambd_noobj * self.mse(Nobj[:,:,:,0], y[:,:,:,0])
        class_loss = self.mse(Iobj[:,:,:,10:30], y[:,:,:,5:25])

        return coord_loss + dim_loss + obj_conf_loss + noobj_conf_loss + class_loss

def main():
    criterion = YOLOLoss()
    x = torch.randn((64, 7, 7, 30))
    y = torch.randn((64, 7, 7, 25))
    print(criterion(x,y))

if __name__ == "__main__":
    main()