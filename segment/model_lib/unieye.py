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
        # self.freeze_encoder()
    
    def freeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        self.image_features = self.image_encoder(image)
        self.mask = self.mask_decoder(self.image_features)
        return self.mask

    def get_important_tokenIdx(self, token_num = 30, downsample_ratio = 16):
        mask = self.mask.argmax(dim=1).unsqueeze(1)
        mask_feature = F.conv2d(mask, 
                                torch.ones(1, 1, downsample_ratio, downsample_ratio, dtype=torch.int64),
                                stride=downsample_ratio)
        
        mask_feature = torch.flatten(mask_feature, start_dim=1)
        _, important_tokenIdx = torch.topk(mask_feature, token_num, dim=1)
        return important_tokenIdx
    
    def get_merged_token(self, token_num = 30, get_notImportant = True):
        
        if get_notImportant:
            token_num-=1

        important_tokenIdx = self.get_important_tokenIdx(token_num)
        image_features = torch.flatten(self.image_features, start_dim=2)
        image_features = image_features.permute(0, 2, 1) # [bs, token, feature]

        B, T, F = image_features.shape
        important_tokens = [torch.index_select(image_features[i], 0, important_tokenIdx[i]) for i in range(B)]
        important_token = torch.stack(important_tokens, dim=0)

        if not get_notImportant: # return topk important token
            return important_token
        else: # return topk-1 important token + others' average
            token_sum = torch.sum(image_features, dim=1)
            Importance_token_sum = torch.sum(important_token, dim=1)
            notImportant_token_mean = (token_sum - Importance_token_sum)/ (T - token_num + 1)
            out = torch.cat([important_token, notImportant_token_mean.unsqueeze(1)], dim=1)
            return out


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

def build_unieye(sam_checkpoint = None, image_size = 400):
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, image_size=image_size)
    imageEncoder = sam.image_encoder
    maskDecoder = MaskDecoder()
    unieye = Unieye(imageEncoder, maskDecoder)
    return unieye