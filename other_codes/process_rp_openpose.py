import numpy as np

import sys

import glob
import os

#Optional. To visualize data
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_root', type=str)
parser.add_argument('-o', '--out_path', type=str)
args = parser.parse_args()

files = sorted([f for f in os.listdir(args.input_root)])# if 'rp_' in f])

for i, f in enumerate(files):
    print(f)
    input_path = os.path.join(args.input_root, f)

    out_img_path = os.path.join(args.out_path, 'images', f)
    out_json_path = os.path.join(args.out_path, 'json', f)

    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_json_path, exist_ok=True)

    cmd = "cd /home/shunsukesaito/Documents/openpose; ./build/examples/openpose/openpose.bin --image_dir {0} --write_images {1} --write_images_format jpg --write_json {2} --render_pose 2 --face --face_render 2 --hand --hand_render 2".format(input_path,
                                                                                                                                                    out_img_path, out_json_path)
    print(cmd)
    os.system(cmd)
