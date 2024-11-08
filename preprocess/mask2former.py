from mmseg.apis import init_model, inference_model
import glob, os
import torch
import cv2
import numpy as np
from tqdm import tqdm

def normalize_torch(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())

f1 = lambda x, r: 255*(x/255)**r
f2 = lambda x, s, t: ((t-s)/(s**2 - 255*s)) * x**2 + (1 - (255*(t-s))/(s**2 - 255*s)) * x

def preprocess(img):
    r, s, t = 0.385, 0.567, 0.265
    ratio = 0.65
    img = (1-ratio)*f1(img, r) + ratio * f2(img, s, t)
    img = img / img.max()
    img = (img-0.5) * 1.15 + 0.5
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    return img

def usm_sharpness(img):
    blur_img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.1, blur_img, -0.1, 0)
    return img


THRESHOLD = 0.5
DATA_ROOT = "/root/bindingvolume/CSI_dataset_UNCC"

# Configure model
CONFIG_PATH = "/root/workspace/mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py"
CKPT_PATH = "/root/workspace/mmsegmentation/ckpts/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth"

print("> Initialize Model...")
model = init_model(CONFIG_PATH, CKPT_PATH)

# get data paths
img_data = glob.glob(os.path.join(DATA_ROOT, "Server", "Env*", "*", "*", "*", "*.jpg"))

# Start Preprocess
print("> Proprocessing...")
for img_path in tqdm(img_data):
    img = cv2.imread(img_path)
    img = preprocess(img)
    img = usm_sharpness(img)
    result = inference_model(model, img)
    masks = result.seg_logits.data

    person_mask = normalize_torch(masks[12])

    person_mask[person_mask < THRESHOLD] = 0
    person_mask[person_mask >= THRESHOLD] = 1

    gt = torch.cat(((1 - person_mask).unsqueeze(0), (person_mask + 0.2).unsqueeze(0)), dim=0)
    gt = torch.argmax(gt, dim=0)

    gt = gt.cpu().numpy().astype(np.uint8) * 255
    gt[gt == 1] = 255
    cv2.imwrite(img_path.replace('.jpg', '_mask.png'), gt)