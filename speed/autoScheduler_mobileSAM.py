import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time
import pickle

import tvm
from tvm import relay
from tvm import meta_schedule as ms
from tvm import auto_scheduler

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def get_task(model:torch.nn.Module, data, target="cuda", cache_dir=None):
    input_shape = data.shape
    model = torch.jit.trace(model, data).eval()

    shape_list = [("input", data.shape)]
    mod, params = relay.frontend.from_pytorch(model, shape_list)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="cuda")

    if cache_dir is not None:
        with open(os.path.join(cache_dir, f"task.pickle"), 'wb') as f:
            pickle.dump(tasks, f)
        with open(os.path.join(cache_dir, f"task_weights.pickle"), 'wb') as f:
            pickle.dump(task_weights, f)

    return tasks, task_weights


def run_tuning(tasks, task_weights, log_file:str = "network.json", result_dir:str = "./weights/tvm"):
    print("Begin tuning...")
    log_file = os.path.join(result_dir, log_file)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=5, min_repeat_ms=25, timeout=10)

    # measure_ctx = auto_scheduler.RPCRunner( key = "3090",
    #                                         host = "127.0.0.1",
    #                                         port = 9190,
    #                                         repeat=1,
    #                                         min_repeat_ms=25,
    #                                         n_parallel = 4,
    #                                         timeout=15)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=500000,
        early_stopping = 500,
        # runner = measure_ctx.runner,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)

def main():

    image = cv2.imread('notebooks/images/picture2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "./weights/mobile_sam.pt"
    model_type = "vit_t"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # load model
    print(f"loaded model: \n {sam}")

    # get input data
    predictor = SamPredictor(sam)
    input_image_torch = predictor.set_image(image)
    input_image = sam.preprocess(input_image_torch)
    print(input_image.shape)

    task,task_weight = get_task(sam.image_encoder, input_image, target="cuda", cache_dir="./weights/tvm")
    run_tuning(task, task_weight, log_file="tinyVit_imageEncoder.json", result_dir="./weights/tvm")

if __name__ == "__main__":
    main()