import os
import os.path as osp
import sys
sys.path.append(".")


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import matplotlib.pyplot as plt
import cv2


def get_raw_data():
    def show_box(image):
        RATIO = 80
        colors = ["red", "green", "blue", "yellow"]
        for i in range(4):
            plt.plot([i*RATIO, i*RATIO], [0, image.shape[0]], color=colors[i], linewidth=3)
            plt.plot([(i+5)*RATIO, (i+5)*RATIO], [0, image.shape[0]], color=colors[i], linewidth=3)
    DATA_PATH = "/data/OpenEDS/OpenEDS/openEDS2020-SparseSegmentation/selection"
    for idx, file in enumerate(os.listdir(DATA_PATH)):   
        if file.endswith(".png"):
            image = cv2.imread(osp.join(DATA_PATH, file))
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转灰度图

            plt.figure(figsize=(10,10))
            plt.imshow(frame)
            show_box(frame)
            plt.axis('on')
            plt.savefig(f'./figure/box/{idx}.png')
            plt.close()

        if idx >= 100:
            break




class Kernel(nn.Module):
    def __init__(self, weight=None):
        super(Kernel, self).__init__()
        self.conv = nn.Conv2d(1, 1, 5, stride=1, padding=0, bias=False)
        if weight is not None:
            self.conv.weight.data = weight
        else:
            self.conv.weight.data = torch.ones([1,1,5,5])

    def pre_process(self, image):
        print(image.shape)
        image_small = cv2.resize(image, (8,5))
        torch_frame = torch.from_numpy(image_small).to(torch.float32).unsqueeze(0).unsqueeze(0)
        return torch_frame
    
    def forward(self, image):
        input = self.pre_process(image)
        x = self.conv(input)
        # return x.squeeze()
        out = x.squeeze().argmax()
        return out

def train():
    # get label
    label = []
    with open("./figure/label.txt", "r") as f:
        for line in f.readlines():
            label.append(int(line.strip()))
    label = torch.tensor(label, dtype=torch.int64)
    print(label)

    kernel = Kernel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(kernel.parameters(), lr=0.02)

    DATA_PATH = "/data/OpenEDS/OpenEDS/openEDS2020-SparseSegmentation/selection"
    for _ in range(5):
        for idx, file in enumerate(os.listdir(DATA_PATH)):   
            if file.endswith(".png"):
                image = cv2.imread(osp.join(DATA_PATH, file))
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转灰度图

                optimizer.zero_grad()
                choose = kernel(frame)
                loss = criterion(choose, label[idx])
                loss.backward()
                optimizer.step()

                print(f"label: {label[idx]}, loss: {loss}")

        torch.save(kernel.conv.weight.data, "./segment/logs/kernel_weight_2020.pth")
        weight = kernel.conv.weight.data.numpy()
        print(weight)



def show_result():
    DATA_PATH = "/data/OpenEDS/OpenEDS/openEDS2020-SparseSegmentation/selection"
    RATIO = 80
    kernel = Kernel(torch.load("./segment/logs/kernel_weight_2020.pth"))
    for idx, file in enumerate(os.listdir(DATA_PATH)):   
        if file.endswith(".png"):
            image = cv2.imread(osp.join(DATA_PATH, file))
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转灰度图
            eye_area = kernel(frame) # find the area lightest, assume it's the eye area
            image = image[:, eye_area*RATIO:(eye_area+5)*RATIO]
            frame = frame[:, eye_area*RATIO:(eye_area+5)*RATIO]
            plt.figure(figsize=(10,10))
            plt.imshow(frame)
            plt.axis('on')
            plt.savefig(f'./figure/resized/{idx}.png')
            plt.close()



def main():
    # get_raw_data()
    # train()
    show_result()

if __name__ == "__main__":
    main()