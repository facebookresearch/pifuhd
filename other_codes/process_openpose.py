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

cmd = "cd /home/shunsukesaito/Documents/openpose; ./build/examples/openpose/openpose.bin --image_dir {0} --write_json {1} --render_pose 2 --face --face_render 2 --hand --hand_render 2".format(input_path, out_json_path)
print(cmd)
os.system(cmd)
