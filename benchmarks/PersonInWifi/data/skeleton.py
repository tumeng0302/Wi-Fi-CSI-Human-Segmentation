# Description: Generate skeleton from image

import argparse
import pyopenpose as op
import glob, os
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", help="Path to OpenPose models", default="/root/openpose/models")
parser.add_argument("--env_folder", help="Path to enviroment", required=True)
args = parser.parse_args()


params = dict()
params["model_folder"] = args.model_folder
params["heatmaps_add_parts"] = True
params["heatmaps_add_bkg"] = True
params["heatmaps_add_PAFs"] = True
params["heatmaps_scale"] = 2

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# parse image
all_images = glob.glob(os.path.join(args.env_folder, 'img', '*', '*', '*', '*.jpg'))

print(all_images[0])
print(len(all_images))

for image_path in tqdm(all_images):
    datum = op.Datum()
    imageToProcess = cv2.imread(image_path)
    imageToProcess = cv2.resize(imageToProcess, (256, 192))
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    keypoints = datum.poseKeypoints
    heatmaps = datum.poseHeatMaps.copy()
    heatmaps = (heatmaps).astype(dtype='uint8')

    np.save(image_path.replace('.jpg', '_skeleton.npy'), heatmaps)
    # cv2.imwrite(image_path.replace('.jpg', '_skeleton.jpg'), datum.cvOutputData)