# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.render.mesh import load_obj_mesh, compute_normal
from lib.render.camera import Camera
from lib.render.gl.geo_render import GeoRender
from lib.render.gl.color_render import ColorRender
import trimesh

import cv2
import os
import argparse

width = 512
height = 512

def make_rotate(rx, ry, rz):
    
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_dir', type=str, required=True)
parser.add_argument('-ww', '--width', type=int, default=512)
parser.add_argument('-hh', '--height', type=int, default=512)
parser.add_argument('-g', '--geo_render', action='store_true', help='default is normal rendering')

args = parser.parse_args()

if args.geo_render:
    renderer = GeoRender(width=args.width, height=args.height)
else:
    renderer = ColorRender(width=args.width, height=args.height)
cam = Camera(width=1.0, height=args.height/args.width)
cam.ortho_ratio = 1.2
cam.near = -100
cam.far = 10

obj_files = []
for (root,dirs,files) in os.walk(args.file_dir, topdown=True): 
    for file in files:
        if '.obj' in file:
            obj_files.append(os.path.join(root, file))
print(obj_files)

R = make_rotate(math.radians(180),0,0)

for i, obj_path in enumerate(obj_files):

    print(obj_path)
    obj_file = obj_path.split('/')[-1]
    obj_root = obj_path.replace(obj_file,'')
    file_name = obj_file[:-4]

    if not os.path.exists(obj_path):
        continue    
    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices
    faces = mesh.faces

    # vertices = np.matmul(vertices, R.T)
    bbox_max = vertices.max(0)
    bbox_min = vertices.min(0)

    # notice that original scale is discarded to render with the same size
    vertices -= 0.5 * (bbox_max + bbox_min)[None,:]
    vertices /= bbox_max[1] - bbox_min[1]

    normals = compute_normal(vertices, faces)
    
    if args.geo_render:
        renderer.set_mesh(vertices, faces, normals, faces)
    else:
        renderer.set_mesh(vertices, faces, 0.5*normals+0.5, faces) 
        
    cnt = 0
    for j in range(0, 361, 2):
        cam.center = np.array([0, 0, 0])
        cam.eye = np.array([2.0*math.sin(math.radians(j)), 0, 2.0*math.cos(math.radians(j))]) + cam.center

        renderer.set_camera(cam)
        renderer.display()
        
        img = renderer.get_color(0)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        cv2.imwrite(os.path.join(obj_root, 'rot_%04d.png' % cnt), 255*img)
        cnt += 1

    cmd = 'ffmpeg -framerate 30 -i ' + obj_root + '/rot_%04d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(obj_root, file_name + '.mp4')
    os.system(cmd)
    cmd = 'rm %s/rot_*.png' % obj_root
    os.system(cmd)