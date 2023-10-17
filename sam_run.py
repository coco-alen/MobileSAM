import numpy as np
import torch
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append(".")

import torch.nn.functional as F
from mobile_sam import sam_model_registry, SamPredictor
from gaze_tracking.pupil import Pupil
from gaze_tracking.calibration import Calibration

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   



###清除黑色背景
def ClearBackGround(blur_img):
    height, width = blur_img.shape #获取图片宽高
    
    #去除黑色背景，seedPoint代表初始种子，进行四次，即对四个角都做一次，可去除最外围的黑边
    blur_img = cv2.floodFill(blur_img,mask=None,seedPoint=(0,0),newVal=(0))[1]
    blur_img = cv2.floodFill(blur_img, mask=None, seedPoint=(0,height-1), newVal=(0))[1]
    blur_img = cv2.floodFill(blur_img, mask=None, seedPoint=(width-1, height-1), newVal=(0))[1]
    blur_img = cv2.floodFill(blur_img, mask=None, seedPoint=(width-1, 0), newVal=(0))[1]

    return blur_img


def get_pupil_coord(image, threshold:int = None):
    RATIO = 80
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转灰度图

    # find eye area, filt out background
    frame_small = cv2.resize(frame, (5,8))  # resize to a small size
    torch_frame = torch.from_numpy(frame_small).to(torch.float32).unsqueeze(0).unsqueeze(0)
    eye_area = int(F.conv2d(torch_frame, torch.ones([1,1,5,5]), stride=1, padding=0).squeeze().argmax()) # find the area lightest, assume it's the eye area
    image = image[eye_area*RATIO:(eye_area+5)*RATIO,:]
    frame = frame[eye_area*RATIO:(eye_area+5)*RATIO,:]

    plt.figure(figsize=(10,10))
    plt.imshow(frame)
    plt.axis('on')
    plt.savefig('./figure/input_image.png')

    # find pupil
    if threshold is None:        
        calibration = Calibration()
        threshold = calibration.find_best_threshold(frame)

    _, threshold = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)
    threshold = ClearBackGround(threshold) # floodFill to clear edge black area

    plt.figure(figsize=(10,10))
    plt.imshow(threshold)
    plt.axis('on')
    plt.savefig('./figure/input_image_2.png')

    # find the biggest Contours, assume it's the pupil
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]  
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    moments = cv2.moments(contours[-1])
    pupil_x, pupil_y = int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    coord = np.array([[pupil_x, pupil_y]])
    raw_coord = coord.copy()
    raw_coord[0,1] += eye_area*RATIO

    return coord, raw_coord, image

def get_mask(predictor, coord, image_resized, input_label):
    ### MobileSAM ###
    box = np.array([0, coord[0][1] - 100, image_resized.shape[1], coord[0][1] + 100])

    predictor.set_image(image_resized, image_format="BGR")
    # predictor.set_torch_image(image_resized, original_image_size = image_resized.shape[-2:])
    masks, scores, logits = predictor.predict(
        point_coords=coord,
        point_labels=input_label,
        multimask_output=True,
        box=box,
    )
    return masks, scores, logits

def latency_test(func, input, repeat=500, name="func"):
    import time
    # warm up
    for _ in range(repeat//2):
        func(*input)

    start = time.time()
    for _ in range(repeat):
        func(*input)
    end = time.time()
    print(f"{name} latency: {(end-start)/repeat*1000} ms")


def main():
    image = cv2.imread('figure/eye6.png')
    print(image.shape)
    threshold = 40
    coord, raw_coord, image_resized = get_pupil_coord(image, threshold)

    # print pupil location
    # print(coord)
    input_label = np.array([1])
    plt.figure(figsize=(10,10))
    plt.imshow(image_resized)
    show_points(coord, input_label, plt.gca())
    plt.axis('on')
    plt.savefig('./figure/point_prompt.png')
    print(image_resized.shape)

    sam_checkpoint = "weights/mobile_sam.pt"
    model_type = "vit_t"

    device = "cuda:5" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    # image_torch = predictor.set_image(image_resized, image_format="BGR")

    masks, scores, logits = get_mask(predictor, coord, image_resized, input_label)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image_resized)
        show_mask(mask, plt.gca())
        show_points(coord, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')  
        plt.savefig(f'./figure/mask_{i+1}.png')

    # print(masks.shape) # (number_of_masks) x H x W
    # latency_test(func=get_pupil_coord, input=(image, threshold), name="get_pupil_coord")
    # latency_test(func=get_mask, input=(predictor, coord, image_resized, input_label), name="get_mask")

if __name__ == "__main__":
    main()