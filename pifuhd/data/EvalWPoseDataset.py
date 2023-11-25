# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import json
import numpy as np
from pifuhd.data.EvalDataset import EvalDataset
from .helper_dataset import make_bundles
from .helper_image_crop import crop_image, face_crop, upperbody_crop, fullbody_crop

crop_callbacks = {'face': face_crop, 'upperbody': upperbody_crop, 'fullbody': fullbody_crop}


class EvalWPoseDataset(EvalDataset):

    def __init__(self, opt):
        items = make_bundles(opt.dataroot, '_keypoints.json')
        super().__init__(opt, items)
        if self.opt.crop_type == 'face':
            self.crop_func = face_crop
        elif self.opt.crop_type == 'upperbody':
            self.crop_func = upperbody_crop
        else:
            self.crop_func = fullbody_crop

    def get_n_person(self, index):
        joint_path = self.items[index].meta
        # Calib
        with open(joint_path) as json_file:
            data = json.load(json_file)
            return len(data['people'])

    def get_human_box(self, index):
        joint_path = self.items[index].meta
        with open(joint_path) as json_file:
            data = json.load(json_file)
            if len(data['people']) == 0:
                raise IOError('non human found!!')

            # if True, the person with the largest height will be chosen.
            # set to False for multi-person processing
            if True:
                selected_data = data['people'][0]
                height = 0
                if len(data['people']) != 1:
                    for i in range(len(data['people'])):
                        tmp = data['people'][i]
                        keypoints = np.array(tmp['pose_keypoints_2d']).reshape(-1, 3)

                        flags = keypoints[:, 2] > 0.5  # openpose
                        # flags = keypoints[:,2] > 0.2  #detectron
                        if sum(flags) == 0:
                            continue
                        bbox = keypoints[flags]
                        bbox_max = bbox.max(0)
                        bbox_min = bbox.min(0)

                        if height < bbox_max[1] - bbox_min[1]:
                            height = bbox_max[1] - bbox_min[1]
                            selected_data = tmp
            else:
                pid = min(len(data['people']) - 1, self.person_id)
                selected_data = data['people'][pid]

            keypoints = np.array(selected_data['pose_keypoints_2d']).reshape(-1, 3)

            flags = keypoints[:, 2] > 0.5  # openpose
            # flags = keypoints[:,2] > 0.2    #detectron

            nflag = flags[0]
            mflag = flags[1]

            check_id = [2, 5, 15, 16, 17, 18]
            cnt = sum(flags[check_id])
            if self.opt.crop_type == 'face' and (not (nflag and cnt > 3)):
                print('Waring: face should not be backfacing.')
            if self.opt.crop_type == 'upperbody' and (not (mflag and nflag and cnt > 3)):
                print('Waring: upperbody should not be backfacing.')
            if self.opt.crop_type == 'fullbody' and sum(flags) < 15:
                print('Waring: not sufficient keypoints.')

        return self.crop_func(keypoints)

    def crop_human_box(self, image, rect):
        return crop_image(image, rect)
