import os
import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F 
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import cv2

from mobile_sam import sam_model_registry


def preprocess_image(image):
    RATIO = 80
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转灰度图

    # find eye area, filt out background
    frame_small = cv2.resize(frame, (5,8))  # resize to a small size
    torch_frame = torch.from_numpy(frame_small).to(torch.float32).unsqueeze(0).unsqueeze(0)
    eye_area = int(F.conv2d(torch_frame, torch.ones([1,1,5,5]), stride=1, padding=0).squeeze().argmax()) # find the area lightest, assume it's the eye area
    image = image[eye_area*RATIO:(eye_area+5)*RATIO,:,:]
    return image, eye_area


def main():
    image = cv2.imread('figure/eye5.png')
    image, eye_area = preprocess_image(image)
    image = to_pil_image(image)
    print(image.size)
    sam_checkpoint = "weights/mobile_sam.pt"
    model_type = "vit_t"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, image_size=400)

    imageEncoder = sam.image_encoder
    # print(imageEncoder)
    imageEncoder.eval()

    # image = resize(image, (512, 512))
    image = torch.from_numpy(np.array(image)).to(torch.float32).permute(2,0,1).unsqueeze(0)
    print(image.shape)

    feature = imageEncoder(image)
    print(feature.shape)





if __name__ == "__main__":
    main()