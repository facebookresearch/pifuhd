# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import argparse
import trimesh


def meshcleaning(file_dir):
    files = sorted([f for f in os.listdir(file_dir) if '.obj' in f])
    for i, file in enumerate(files):
        obj_path = os.path.join(file_dir, file)

        print(f"Processing: {obj_path}")

        mesh = trimesh.load(obj_path)
        cc = mesh.split(only_watertight=False)    

        out_mesh = cc[0]
        bbox = out_mesh.bounds
        height = bbox[1,0] - bbox[0,0]
        for c in cc:
            bbox = c.bounds
            if height < bbox[1,0] - bbox[0,0]:
                height = bbox[1,0] - bbox[0,0]
                out_mesh = c
        
        out_mesh.export(obj_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_dir', type=str, required=True)
    args = parser.parse_args()

    meshcleaning(args.file_dir)