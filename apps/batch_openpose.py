# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--openpose_dir', type=str, required=True)
parser.add_argument('-i', '--input_root', type=str, required=True)
parser.add_argument('-o', '--out_path', type=str, required=True)
args = parser.parse_args()

op_dir = args.openpose_dir
input_path = args.input_root
out_json_path = args.out_path

os.makedirs(out_json_path, exist_ok=True)

cmd = "cd {0}; ./build/examples/openpose/openpose.bin --image_dir {1} --write_json {2} --render_pose 2 --face --face_render 2 --hand --hand_render 2".format(op_dir, input_path, out_json_path)
print(cmd)
os.system(cmd)
