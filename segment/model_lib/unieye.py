import os
import sys

sys.path.append("..")

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from mobile_sam import sam_model_registry

class Unieye(nn.Module):
    def __init__(self, image_encoder:nn.Module, mask_decoder:nn.Module):
        super(Unieye, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        





def build_unieye(sam_checkpoint = "./weights/mobile_sam.pt", image_size = 400):
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, image_size=image_size)
    imageEncoder = sam.image_encoder