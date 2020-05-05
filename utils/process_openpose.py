import numpy as np

import sys

import glob
import os

#Optional. To visualize data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_root', type=str)
parser.add_argument('-o', '--out_path', type=str)
args = parser.parse_args()

input_path = args.input_root

out_json_path = args.out_path

os.makedirs(out_json_path, exist_ok=True)
openposeDir="/private/home/hjoo/codes/openpose"
# cmd = "cd /home/shunsukesaito/Documents/openpose; ./build/examples/openpose/openpose.bin --image_dir {0} --write_json {1} --render_pose 2 --face --face_render 2 --hand --hand_render 2".format(input_path, out_json_path)
cmd = "cd /private/home/hjoo/codes/openpose; ./build3/examples/openpose/openpose.bin --image_dir {0} --write_json {1} --face  --hand --render_pose 0 --display 0".format(input_path, out_json_path)
print(cmd)
os.system(cmd)
