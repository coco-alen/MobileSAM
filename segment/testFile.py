import os
import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import cv2

from mobile_sam import sam_model_registry

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MaskDecoder(nn.Module):
    def __init__(self, input_dim = 256, output_dim = 2, activation = nn.GELU):
        # input_dim should be 256 from tinyVit and output_dim should be 2 for 2 classes
        # should upscale 16x

        super(MaskDecoder, self).__init__()
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim // 4, kernel_size=4, stride=4),
            LayerNorm2d(input_dim // 4),
            activation(),
            nn.ConvTranspose2d(input_dim // 4, input_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(input_dim // 16),
            activation(),
            nn.ConvTranspose2d(input_dim // 16, output_dim, kernel_size=2, stride=2),
            activation(),
        )
    def forward(self, image_features):
        mask = self.output_upscaling(image_features)
        return mask


def preprocess_image(image):
    RATIO = 80
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转灰度图

    # find eye area, filt out background
    frame_small = cv2.resize(frame, (5,8))  # resize to a small size
    torch_frame = torch.from_numpy(frame_small).to(torch.float32).unsqueeze(0).unsqueeze(0)
    eye_area = int(F.conv2d(torch_frame, torch.ones([1,1,5,5]), stride=1, padding=0).squeeze().argmax()) # find the area lightest, assume it's the eye area
    image = image[eye_area*RATIO:(eye_area+5)*RATIO,:,:]
    frame = frame[eye_area*RATIO:(eye_area+5)*RATIO,:]
    return image, eye_area


def main():
    image = cv2.imread('/data/hyou37/MobileSAM/figure/eye1.png')
    image, eye_area = preprocess_image(image)
    image = to_pil_image(image)
    sam_checkpoint = "/data/hyou37/MobileSAM/weights/mobile_sam.pt"
    model_type = "vit_t"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, image_size=400)

    imageEncoder = sam.image_encoder
    # print(imageEncoder)
    imageEncoder.eval()

    
    for name, param in imageEncoder.named_parameters():
        print(f"name is {name}, param grad is {param}")


    decoder = MaskDecoder()

    # image = resize(image, (512, 512))
    image = torch.from_numpy(np.array(image)).to(torch.float32).unsqueeze(0).permute(0,3,1,2)
    print(f"image shape is {image.shape}")

    feature = imageEncoder(image)
    print(f"feature shape is{feature.shape}")

    mask = decoder(feature)
    print(f"mask shape is {mask.shape}")





if __name__ == "__main__":
    main()