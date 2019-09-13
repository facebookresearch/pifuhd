import random

import numpy as np
from skimage.filters import gaussian
import torch
from PIL import Image, ImageFilter


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))

# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, 
                 eigval=(0.2175, 0.0188, 0.0045), 
                 eigvec=((-0.5675, 0.7192, 0.4009),
                         (-0.5808, -0.0045, -0.8140),
                         (-0.5836, -0.6948, 0.4203))):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))
