import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


@torch.no_grad()
def throughput(image, model, repeat=100):

    # batch_size = image.shape[0]
    batch_size = image.shape[0]
    with torch.cuda.amp.autocast(enabled=False):
        for i in range(repeat):
            model(image)
        torch.cuda.synchronize()
        print(f"throughput averaged with {repeat} times")
        tic1 = time.time()
        for i in range(repeat):
            model(image)
            torch.cuda.synchronize()
        tic2 = time.time()
        print(
            f"batch_size {batch_size} throughput {repeat * batch_size / (tic2 - tic1)}"
        )
        print(
            f"batch_size {batch_size} latency {(tic2 - tic1) / repeat * 1000} ms"
        )
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False, with_stack=True) as prof:
    #     model(images)
    # resultList = prof.table().split("\n")
    # prof.export_chrome_trace('./weights/'+ "mobileSAM" +'.json')
    # prof.export_stacks('./weights/'+ "mobileSAM" +'_cpu_stack.json', metric="self_cpu_time_total")
    # prof.export_stacks('./weights/'+ "mobileSAM" +'_gpu_stack.json', metric="self_cuda_time_total")
        
    return

def main():

    image = cv2.imread('notebooks/images/picture1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "./weights/mobile_sam.pt"
    model_type = "vit_t"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # load model
    print(f"loaded model: \n {sam}")

    device = "cuda"
    # device = "cpu"
    print(f"device to run model is: {device}")

    sam.to(device=device)
    sam.eval()

    # predict one label
    predictor = SamPredictor(sam)
    input_image_torch = predictor.set_image(image)
    input_image = sam.preprocess(input_image_torch)
    print(input_image.shape)
    # image encoder throughput
    throughput(input_image, sam.image_encoder, repeat=100)

    # segmentation
    input_point = np.array([[400, 400]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(point_coords=input_point,
                                            point_labels=input_label,
                                            multimask_output=True,
                                            )
    print("mask shape: ", masks.shape)

if __name__ == "__main__":
    main()