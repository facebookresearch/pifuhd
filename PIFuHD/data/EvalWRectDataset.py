# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import numpy as np

from PIFuHD.data.EvalDataset import EvalDataset
from .helper_dataset import make_bundles
from .helper_image_crop import crop_image


class EvalWRectDataset(EvalDataset):

    def __init__(self, opt):
        items = make_bundles(opt.dataroot, '_rect.txt')
        super().__init__(opt, items)

    def get_rect(self, index):
        rect_path = self.items[index].meta
        return np.loadtxt(rect_path, dtype=np.int32)

    def get_n_person(self, index):
        rects = self.get_rect(index)
        return rects.shape[0] if len(rects.shape) == 2 else 1

    def get_human_box(self, index):
        return self.get_rect(index)

    def crop_human_box(self, image, rects):
        if len(rects.shape) == 1:
            rects = rects[None]
        pid = min(rects.shape[0] - 1, self.person_id)

        rect = rects[pid].tolist()
        return crop_image(image, rect)
