import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2 as cv
import xml.etree.ElementTree as ET
import math
import pickle
import time

def imdisplay(nparr):
    plt.imshow(nparr)
    plt.axis('off')
    plt.show()

def display_bnd(x, label):
    image = x.copy()
    for i in range(7):
        for j in range(7):
            if label[i][j][0] > 0.5:
                x = i+label[i][j][1]
                y = j+label[i][j][2]
                xmid = (x*448)/7
                ymid = (y*448)/7
                x1 = int(xmid-(label[i][j][3]/2)*448)
                y1 = int(ymid-(label[i][j][4]/2)*448)
                x2 = int(xmid+(label[i][j][3]/2)*448)
                y2 = int(ymid+(label[i][j][4]/2)*448)
                cv.rectangle(image, (x1,y1), (x2,y2), (0, 255, 0), 2)
    imdisplay(image)


def parse_xml(tree):
    classes = {'person':0, 'bird':1, 'cat':2, 'cow':3, 'dog':4, 'horse':5, 'sheep':6, 'aeroplane':7, 
               'bicycle':8, 'boat':9, 'bus':10, 'car':11, 'motorbike':12, 'train':13, 'bottle':14, 
               'chair':15, 'diningtable':16, 'pottedplant':17, 'sofa':18, 'tvmonitor':19}
    objects = []
    img = [[[0 for _ in range(25)] for _ in range(7)] for _ in range(7)]
    root = tree.getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    objs = root.findall('object')
    for obj in objs:
        dic = {}
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        x = ((xmax+xmin)/2)/width
        y = ((ymax+ymin)/2)/height
        dic['row'] = int(x * 7)
        dic['col'] = int(y * 7)
        coords = [1, ((x*7)%1), ((y*7)%1), (xmax - xmin)/width, (ymax - ymin)/height]
        one_hot = [0 for i in range(20)]
        one_hot[classes[name]] = 1
        dic['coords'] = coords+one_hot
        objects.append(dic)
    for item in objects:
        i = item['row']
        j = item['col']
        img[i][j] = item['coords']
    return img

class pascalVOC(Dataset):
    def __init__(self, train_len=17125, test_len=0, train=True):
        images = []
        annotations = []
        impath = 'VOCdevkit/VOC2012/JPEGImages'
        if train:
            for file in os.listdir(impath)[:train_len]:
                img = cv.imread(impath + '/' + file)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, (448, 448))
                images.append(img)
            anpath = 'VOCdevkit/VOC2012/Annotations'
            for file in os.listdir(anpath)[:train_len]:
                tree = ET.parse(anpath + '/' + file)
                anno = parse_xml(tree)
                annotations.append(anno)
        else:
            for file in os.listdir(impath)[train_len:train_len+test_len]:
                img = cv.imread(impath + '/' + file)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, (448, 448))
                images.append(img)
            anpath = 'VOCdevkit/VOC2012/Annotations'
            for file in os.listdir(anpath)[train_len:train_len+test_len]:
                tree = ET.parse(anpath + '/' + file)
                anno = parse_xml(tree)
                annotations.append(anno)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.X = images
        self.Y = annotations

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.tensor(self.X[index])
        y = torch.tensor(self.Y[index])
        x = x.permute(2, 0, 1)
        x = x / 255.0
        x = self.normalize(x)
        return x, y
    
def main():
    start = time.time()
    dataset = pascalVOC(length=64)
    end = time.time()

if __name__ == '__main__':
    main()