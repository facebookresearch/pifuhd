import os
import sys

import cv2
from pifuhd.data.ImageBundle import ImageBundle


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def pair_ext(f, extension):
    return f.replace('.%s' % (f.split('.')[-1]), extension)


def is_image(file):
    if file.split('.')[-1] in ['png', 'jpeg', 'jpg', 'PNG', 'JPG', 'JPEG']:
        return True
    return False


def scan_image_folder(path, extension):
    try:
        return sorted([os.path.join(path, f) for f in os.listdir(path) if
                       is_image(f) and os.path.exists(os.path.join(path, pair_ext(f, extension)))])
    except FileNotFoundError:
        print(f'error cannot found dataset path:{path}')
        sys.exit(1)


def make_bundles(path, ext):
    path = os.path.abspath(path)
    return [
        ImageBundle(
            img=load_image(file),
            name=os.path.splitext(os.path.basename(file))[0],
            meta=os.path.join(path, pair_ext(file, ext)),
        )
        for file in scan_image_folder(path, ext)
    ]
