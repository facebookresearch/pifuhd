import os
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in_dir', type=str, default='./')
parser.add_argument('-o', '--out_dir', type=str, default='./')
parser.add_argument('-s', '--split_size', type=int, default=100)
args = parser.parse_args()

files = sorted([f for f in os.listdir(args.in_dir) if 'sdf.data' in f])

for f in files:
    file_name = f.replace('_sdf.data','')

    sdf_file = open(os.path.join(args.in_dir, f), 'rb')

    pts_num = np.fromfile(sdf_file, dtype=np.int32, count=1)
    sdf = np.fromfile(sdf_file, dtype=np.float32, count=pts_num[0]*4).reshape(pts_num[0], 4)

    idxs = np.arange(0,sdf.shape[0])
    np.random.shuffle(idxs)

    cnt = 0

    interv = sdf.shape[0]//args.split_size + 1
    n_samples = sdf.shape[0]
    
    out_dir = os.path.join(args.out_dir, file_name)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(args.split_size-1):    
        np.save(os.path.join(out_dir, 'sdf%04d.npy' % i), sdf[idxs[i*interv:i*interv+interv]])

    np.save(os.path.join(out_dir, 'sdf%04d.npy' % (args.split_size-1)), sdf[idxs[(args.split_size-1)*interv:]])
