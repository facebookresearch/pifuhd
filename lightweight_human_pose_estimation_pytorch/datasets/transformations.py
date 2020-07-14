import random

import cv2
import numpy as np


class ConvertKeypoints:
    def __call__(self, sample):
        label = sample['label']
        h, w, _ = sample['image'].shape
        keypoints = label['keypoints']
        for keypoint in keypoints:  # keypoint[2] == 0: occluded, == 1: visible, == 2: not in image
            if keypoint[0] == keypoint[1] == 0:
                keypoint[2] = 2
            if (keypoint[0] < 0
                    or keypoint[0] >= w
                    or keypoint[1] < 0
                    or keypoint[1] >= h):
                keypoint[2] = 2
        for other_label in label['processed_other_annotations']:
            keypoints = other_label['keypoints']
            for keypoint in keypoints:
                if keypoint[0] == keypoint[1] == 0:
                    keypoint[2] = 2
                if (keypoint[0] < 0
                        or keypoint[0] >= w
                        or keypoint[1] < 0
                        or keypoint[1] >= h):
                    keypoint[2] = 2
        label['keypoints'] = self._convert(label['keypoints'], w, h)

        for other_label in label['processed_other_annotations']:
            other_label['keypoints'] = self._convert(other_label['keypoints'], w, h)
        return sample

    def _convert(self, keypoints, w, h):
        # Nose, Neck, R hand, L hand, R leg, L leg, Eyes, Ears
        reorder_map = [1, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        converted_keypoints = list(keypoints[i - 1] for i in reorder_map)
        converted_keypoints.insert(1, [(keypoints[5][0] + keypoints[6][0]) / 2,
                                       (keypoints[5][1] + keypoints[6][1]) / 2, 0])  # Add neck as a mean of shoulders
        if keypoints[5][2] == 2 or keypoints[6][2] == 2:
            converted_keypoints[1][2] = 2
        elif keypoints[5][2] == 1 and keypoints[6][2] == 1:
            converted_keypoints[1][2] = 1
        if (converted_keypoints[1][0] < 0
                or converted_keypoints[1][0] >= w
                or converted_keypoints[1][1] < 0
                or converted_keypoints[1][1] >= h):
            converted_keypoints[1][2] = 2
        return converted_keypoints


class Scale:
    def __init__(self, prob=1, min_scale=0.5, max_scale=1.1, target_dist=0.6):
        self._prob = prob
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._target_dist = target_dist

    def __call__(self, sample):
        prob = random.random()
        scale_multiplier = 1
        if prob <= self._prob:
            prob = random.random()
            scale_multiplier = (self._max_scale - self._min_scale) * prob + self._min_scale
        label = sample['label']
        scale_abs = self._target_dist / label['scale_provided']
        scale = scale_abs * scale_multiplier
        sample['image'] = cv2.resize(sample['image'], dsize=(0, 0), fx=scale, fy=scale)
        label['img_height'], label['img_width'], _ = sample['image'].shape
        sample['mask'] = cv2.resize(sample['mask'], dsize=(0, 0), fx=scale, fy=scale)

        label['objpos'][0] *= scale
        label['objpos'][1] *= scale
        for keypoint in sample['label']['keypoints']:
            keypoint[0] *= scale
            keypoint[1] *= scale
        for other_annotation in sample['label']['processed_other_annotations']:
            other_annotation['objpos'][0] *= scale
            other_annotation['objpos'][1] *= scale
            for keypoint in other_annotation['keypoints']:
                keypoint[0] *= scale
                keypoint[1] *= scale
        return sample


class Rotate:
    def __init__(self, pad, max_rotate_degree=40):
        self._pad = pad
        self._max_rotate_degree = max_rotate_degree

    def __call__(self, sample):
        prob = random.random()
        degree = (prob - 0.5) * 2 * self._max_rotate_degree
        h, w, _ = sample['image'].shape
        img_center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(img_center, degree, 1)

        abs_cos = abs(R[0, 0])
        abs_sin = abs(R[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        dsize = (bound_w, bound_h)

        R[0, 2] += dsize[0] / 2 - img_center[0]
        R[1, 2] += dsize[1] / 2 - img_center[1]
        sample['image'] = cv2.warpAffine(sample['image'], R, dsize=dsize,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self._pad)
        sample['label']['img_height'], sample['label']['img_width'], _ = sample['image'].shape
        sample['mask'] = cv2.warpAffine(sample['mask'], R, dsize=dsize,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))  # border is ok
        label = sample['label']
        label['objpos'] = self._rotate(label['objpos'], R)
        for keypoint in label['keypoints']:
            point = [keypoint[0], keypoint[1]]
            point = self._rotate(point, R)
            keypoint[0], keypoint[1] = point[0], point[1]
        for other_annotation in label['processed_other_annotations']:
            for keypoint in other_annotation['keypoints']:
                point = [keypoint[0], keypoint[1]]
                point = self._rotate(point, R)
                keypoint[0], keypoint[1] = point[0], point[1]
        return sample

    def _rotate(self, point, R):
        return [R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2]]


class CropPad:
    def __init__(self, pad, center_perterb_max=40, crop_x=368, crop_y=368):
        self._pad = pad
        self._center_perterb_max = center_perterb_max
        self._crop_x = crop_x
        self._crop_y = crop_y

    def __call__(self, sample):
        prob_x = random.random()
        prob_y = random.random()

        offset_x = int((prob_x - 0.5) * 2 * self._center_perterb_max)
        offset_y = int((prob_y - 0.5) * 2 * self._center_perterb_max)
        label = sample['label']
        shifted_center = (label['objpos'][0] + offset_x, label['objpos'][1] + offset_y)
        offset_left = -int(shifted_center[0] - self._crop_x / 2)
        offset_up = -int(shifted_center[1] - self._crop_y / 2)

        cropped_image = np.empty(shape=(self._crop_y, self._crop_x, 3), dtype=np.uint8)
        for i in range(3):
            cropped_image[:, :, i].fill(self._pad[i])
        cropped_mask = np.empty(shape=(self._crop_y, self._crop_x), dtype=np.uint8)
        cropped_mask.fill(1)

        image_x_start = int(shifted_center[0] - self._crop_x / 2)
        image_y_start = int(shifted_center[1] - self._crop_y / 2)
        image_x_finish = image_x_start + self._crop_x
        image_y_finish = image_y_start + self._crop_y
        crop_x_start = 0
        crop_y_start = 0
        crop_x_finish = self._crop_x
        crop_y_finish = self._crop_y

        w, h = label['img_width'], label['img_height']
        should_crop = True
        if image_x_start < 0:  # Adjust crop area
            crop_x_start -= image_x_start
            image_x_start = 0
        if image_x_start >= w:
            should_crop = False

        if image_y_start < 0:
            crop_y_start -= image_y_start
            image_y_start = 0
        if image_y_start >= w:
            should_crop = False

        if image_x_finish > w:
            diff = image_x_finish - w
            image_x_finish -= diff
            crop_x_finish -= diff
        if image_x_finish < 0:
            should_crop = False

        if image_y_finish > h:
            diff = image_y_finish - h
            image_y_finish -= diff
            crop_y_finish -= diff
        if image_y_finish < 0:
            should_crop = False

        if should_crop:
            cropped_image[crop_y_start:crop_y_finish, crop_x_start:crop_x_finish, :] =\
                sample['image'][image_y_start:image_y_finish, image_x_start:image_x_finish, :]
            cropped_mask[crop_y_start:crop_y_finish, crop_x_start:crop_x_finish] =\
                sample['mask'][image_y_start:image_y_finish, image_x_start:image_x_finish]

        sample['image'] = cropped_image
        sample['mask'] = cropped_mask
        label['img_width'] = self._crop_x
        label['img_height'] = self._crop_y

        label['objpos'][0] += offset_left
        label['objpos'][1] += offset_up
        for keypoint in label['keypoints']:
            keypoint[0] += offset_left
            keypoint[1] += offset_up
        for other_annotation in label['processed_other_annotations']:
            for keypoint in other_annotation['keypoints']:
                keypoint[0] += offset_left
                keypoint[1] += offset_up

        return sample

    def _inside(self, point, width, height):
        if point[0] < 0 or point[1] < 0:
            return False
        if point[0] >= width or point[1] >= height:
            return False
        return True


class Flip:
    def __init__(self, prob=0.5):
        self._prob = prob

    def __call__(self, sample):
        prob = random.random()
        do_flip = prob <= self._prob
        if not do_flip:
            return sample

        sample['image'] = cv2.flip(sample['image'], 1)
        sample['mask'] = cv2.flip(sample['mask'], 1)

        label = sample['label']
        w, h = label['img_width'], label['img_height']
        label['objpos'][0] = w - 1 - label['objpos'][0]
        for keypoint in label['keypoints']:
            keypoint[0] = w - 1 - keypoint[0]
        label['keypoints'] = self._swap_left_right(label['keypoints'])

        for other_annotation in label['processed_other_annotations']:
            other_annotation['objpos'][0] = w - 1 - other_annotation['objpos'][0]
            for keypoint in other_annotation['keypoints']:
                keypoint[0] = w - 1 - keypoint[0]
            other_annotation['keypoints'] = self._swap_left_right(other_annotation['keypoints'])

        return sample

    def _swap_left_right(self, keypoints):
        right = [2, 3, 4, 8, 9, 10, 14, 16]
        left = [5, 6, 7, 11, 12, 13, 15, 17]
        for r, l in zip(right, left):
            keypoints[r], keypoints[l] = keypoints[l], keypoints[r]
        return keypoints
