# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from pifuhd.data.EvalDataset import EvalDataset
from pifuhd.data.helper_image_crop import crop_image


class EvalWMetaDataset(EvalDataset):

    def __init__(self, opt, items):
        super().__init__(opt, items)

    def get_n_person(self, index):
        return len(self.items[index].meta)

    def get_human_box(self, index):
        return self.items[index].meta[0]

    def crop_human_box(self, image, rect):
        return crop_image(image, rect)
