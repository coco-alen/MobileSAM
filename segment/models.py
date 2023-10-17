#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:50:11 2019

@author: manoj
"""


from model_lib.densenet import DenseNet2D
from model_lib.unieye import build_unieye

model_dict = {}

# model_dict['densenet'] = DenseNet2D(dropout=True,prob=0.2)
# model_dict['unieye'] = build_unieye(sam_checkpoint="/data/hyou37/MobileSAM/weights/mobile_sam.pt", image_size=400)

model_dict['densenet'] = DenseNet2D
model_dict['unieye'] = build_unieye
