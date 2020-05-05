import os
import argparse
import trimesh


def meshcleaning(file_dir, jobnum =1, jobidx=1):
    print(f"Jobnum: {jobnum} || jobidx: {jobidx}")

    files = sorted([f for f in os.listdir(file_dir) if '.obj' in f])
    for i, file in enumerate(files):
        obj_path = os.path.join(file_dir, file)
        frameid = int(file[-16:-8])

        if jobnum>1:
            if frameid % jobnum != jobidx:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_dir', type=str, default='./')
    parser.add_argument('-jobnum', type=int, default=1, help="number of jobs for parallel processing")
    parser.add_argument('-jobidx', type=int, default=1, help="Process every n-th index. default is 1. Useful for parallal processing")
    args = parser.parse_args()

    meshcleaning(args.file_dir, args.jobnum, args.jobidx)