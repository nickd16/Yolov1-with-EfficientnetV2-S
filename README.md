# Yolov1-with-EfficientnetV2-S
Implementation of Yolov1 with pretrained EfficientnetV2-S model in pytorch

Results from the Test Set (Trained on pascalvoc 2012 dataset)


\![image](https://github.com/nickd16/Yolov1-with-EfficientnetV2-S/assets/108239710/14940662-6235-4f83-acd6-3ce67c1220d6)


![image](https://github.com/nickd16/Yolov1-with-EfficientnetV2-S/assets/108239710/f07fbb82-4a2d-4b28-86f4-9d94a131d065)


The model architecture itself is unique, simple, and computationally efficient. The base of the model is a pretrained EfficientNetV2-S model. The imagenet1k classifier is then replaced with the following:

Convolutional Layer | in=1280 | out=1280 | filter=2x2 | stride=2

Convolutional Layer (FC) | in=1280 | out=1280 | filter=1x1

Convolutional Layer (Outputs) | in=1280 | out=1280 | filter=1x1

Output is a 7x7x30 output map with 20 classes and 2 bounding box predictions per cell (p, x, y, w, h) * 2

All of the data preprocessing is done from scratch using the pascalvoc xml annotations.

he model is then optimized over this loss function from the original YOLO paper:


![image](https://github.com/nickd16/Yolov1-with-EfficientnetV2-S/assets/108239710/85b81dbd-21a3-4e08-bbd0-4b354fb2b657)




