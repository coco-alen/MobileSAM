import os
from copy import deepcopy


from torch.utils.data import DataLoader 
import torch.nn.functional as F
import torch

from dataset import IrisDataset, IrisDataset2020, transform
from models import model_dict

model = model_dict["unieye"](image_size=224)
model.load_state_dict(torch.load("/data/hyou37/MobileSAM/segment/logs/Unieye_encorder_trainedKernel_res224/models/dense_net240.pt"))
test_set = IrisDataset2020(filepath = '/data/OpenEDS/OpenEDS/openEDS2020-SparseSegmentation',\
                                split = 'test',transform = transform, kernel_weight="/data/hyou37/MobileSAM/segment/logs/kernel_weight_2020.pth", resolution=224)

testloader = DataLoader(test_set, batch_size = 1,
                            shuffle=False, num_workers=2)

data = next(iter(testloader))
mask = model(data[0])
model.get_merged_token()