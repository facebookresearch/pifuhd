# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse

from PIFuHD.data import EvalWRectDataset, EvalWPoseDataset
from PIFuHD.options import BaseOptions
from PIFuHD.recontructor import Reconstructor

###############################################################################################
#                   Setting
###############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, default='./sample_images')
parser.add_argument('-o', '--out_path', type=str, default='./results')
parser.add_argument('-c', '--ckpt_path', type=str, default='./checkpoints/PIFuHD.pt')
parser.add_argument('-r', '--resolution', type=int, default=512)
parser.add_argument('--use_rect', action='store_true', help='use rectangle for cropping')
args = parser.parse_args()


###############################################################################################
#                   Upper PIFu
###############################################################################################

def recon(opts: BaseOptions, use_rect=False):
    if use_rect:
        dataset = EvalWRectDataset(opts)
    else:
        dataset = EvalWPoseDataset(opts)

    reconstructor = Reconstructor(opts)
    reconstructor.evaluate(dataset)


def main():
    cmd = ['--dataroot', args.input_path,
           '--results_path', args.out_path,
           '--loadSize', '1024',
           '--resolution', str(args.resolution),
           '--load_netMR_checkpoint_path', args.ckpt_path,
           '--start_id', '-1',
           '--end_id', '-1'
           ]

    options_parser = BaseOptions()
    opt = options_parser.parse(cmd)
    recon(opt, args.use_rect)


if __name__ == '__main__':
    main()
