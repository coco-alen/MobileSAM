import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time

import tvm
from tvm import relay
from tvm import auto_scheduler
from tvm.contrib import graph_executor

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



def throughput(module, repeat=100):

    # batch_size = image.shape[0]
    batch_size = 1
    for i in range(repeat):
        module.run()
    torch.cuda.synchronize()
    print(f"throughput averaged with {repeat} times")
    tic1 = time.time()
    for i in range(repeat):
        module.run()
        torch.cuda.synchronize()
    tic2 = time.time()
    print(
        f"batch_size {batch_size} throughput {repeat * batch_size / (tic2 - tic1)}"
    )
    print(
        f"batch_size {batch_size} latency {(tic2 - tic1) / repeat * 1000} ms"
    )
        
    return

def get_module(model:torch.nn.Module, data, target="cuda", log_file="./weights/tvm/tinyVit_imageEncoder.json"):
    input_shape = data.shape
    dev = tvm.device(str(target), 0)

    model = torch.jit.trace(model, data).eval()

    shape_list = [("input", data.shape)]
    mod, params = relay.frontend.from_pytorch(model, shape_list)

    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    module = graph_executor.GraphModule(lib["default"](dev))

    data_tvm = tvm.nd.array(data, device=dev)
    module.set_input("input", data_tvm)

    return module


def main():

    image = cv2.imread('notebooks/images/picture1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "./weights/mobile_sam.pt"
    model_type = "vit_t"
    target = 'cuda'
    dev = tvm.device(str(target), 0)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # load model
    print(f"loaded model: \n {sam}")

    print(f"target to run model is: {target}")
    sam.eval()

    # predict one label
    predictor = SamPredictor(sam)
    input_image_torch = predictor.set_image(image)
    input_image = sam.preprocess(input_image_torch)
    print(input_image.shape)

    # get tvm module
    module = get_module(sam.image_encoder, input_image, target=target, log_file="./weights/tvm/tinyVit_imageEncoder.json")

    # image encoder throughput
    print("tvm benchmarking...")
    print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
    print("loop testing...")
    throughput(module, repeat=50)

if __name__ == "__main__":
    main()