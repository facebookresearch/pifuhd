import os
import argparse
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_dir', type=str, default='./')
parser.add_argument('-n', type=int, default=0, help="Process every n-th index. default is 1. Useful for parallal processing")
args = parser.parse_args()

files = sorted([f for f in os.listdir(args.file_dir) if '.obj' in f])
for i, file in enumerate(files):
    obj_path = os.path.join(args.file_dir, file)
    frameid = int(file[-16:-8])
    if frameid % args.n != 0:
        continue

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