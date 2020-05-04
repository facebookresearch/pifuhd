import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils.render.mesh import load_obj_mesh, compute_normal
from utils.render.camera import Camera
from utils.render.gl.geo_render import GeoRender
from utils.render.gl.color_render import ColorRender
import pymesh
import tinyobjloader


import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_dir', type=str, default='./', help="directory where obj files exist")
parser.add_argument('-i', '--img_dir', type=str, default='./', help="directory where input image files exist. Not required. If missing, white BG is used")
parser.add_argument('-ww', '--width', type=int, default=1024)
parser.add_argument('-hh', '--height', type=int, default=1024)
parser.add_argument('-g', '--geo_render', action='store_true', help='default is normal rendering')
parser.add_argument('-png', '--png', action='store_true', help='input image type is png. Default if jpg')
parser.add_argument('-v', '--concatVertical', action='store_true', help='concat videos vertically. Default is horizonal')
args = parser.parse_args()

#Choose rendering angles
angles = [-135, -45, 0, 45, 135]
# angles = [0, 45, 135]
# angles = [0]

if args.geo_render:
    renderType = "g"
else:
    renderType = "n"

if args.png:
    imgtype = "png"
else:
    imgtype = "jpg"

if args.geo_render:
    renderer = GeoRender(width=args.width, height=args.height)
else:
    renderer = ColorRender(width=args.width, height=args.height)

cam = Camera(width=1.0, height=args.height/args.width)
cam.ortho_ratio = 2.0
cam.near = -100
cam.far = 10



file_dir = args.file_dir

files = sorted([f for f in os.listdir(file_dir) if '.obj' in f])
os.makedirs(os.path.join(file_dir, 'render'), exist_ok=True)
for i, file in enumerate(files):
    obj_path = os.path.join(file_dir, file)
    file_name = file[:-4]

    if not os.path.exists(obj_path):
        continue      
    # Create reader.
    reader = tinyobjloader.ObjReader()
    # Load .obj(and .mtl) using default configuration
    ret = reader.ParseFromFile(obj_path)

    attrib = reader.GetAttrib()
    shapes = reader.GetShapes()



    if len(attrib.vertices) == 0:
        continue
    if len(shapes[0].mesh.indices) < 9:
        continue
    vertices = attrib.numpy_vertices().reshape(-1,3)
    faces = shapes[0].mesh.numpy_indices().reshape(-1,9)[:,[0,3,6]]

    # vertices, faces = load_obj_mesh(obj_path)

    # print(bbox_max, bbox_min)
    normals = compute_normal(vertices, faces)

    if args.geo_render:
        renderer.set_mesh(vertices, faces, normals, faces)
    else:
        renderer.set_mesh(vertices, faces, 0.5*normals+0.5, faces)
    
    for angle in angles:
        cam.center = np.array([0, 0, 0])
        cam.eye = np.array([2.0*math.sin(math.radians(angle)), 0, 2.0*math.cos(math.radians(angle))]) + cam.center

        renderer.set_camera(cam)
        renderer.display()

        img = renderer.get_color(0)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        outputPath = os.path.join(file_dir, 'render', '%d_%04d.%s' % (angle, i,imgtype))
        cv2.imwrite(outputPath, 255*img)
        print(f"rendering to: {outputPath}")
        # print(file_dir)
    
    file_id = file_name.replace('result_', '')
    file_id = file_id.replace('_512','')
    cmd = 'cp %s/%s.%s %s/input_%04d.%s' % (args.img_dir, file_id, imgtype, file_dir, i, imgtype)
    # cmd = 'cp %s/%s.jpg %s/input_%04d.jpg' % (args.img_dir, file_id, file_dir, i)
    # print(cmd)
    os.system(cmd)

    # if i==10:
    #     break

#Generating Input video
cmd = 'ffmpeg -framerate 30 -i ' + ('%s/input' % (file_dir)) + '_%04d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(file_dir, '%d_input.mp4' % 0)
os.system(cmd)

#Generating Rendered Video
for angle in angles:
    cmd = 'ffmpeg -framerate 30 -i ' + ('%s/render/%d' % (file_dir,angle)) + '_%04d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(file_dir, f'{angle}_{renderType}.mp4')
    os.system(cmd)
    cmd = 'rm %s/render/%d_*.jpg' % (file_dir, angle)
    os.system(cmd)

cmd = 'ffmpeg'
for angle in angles:
    cmd += ' -i %s/%d_%s.mp4' % (file_dir, angle, renderType)


if args.concatVertical:
    cmd += ' -y -filter_complex "[0:v]pad=iw:ih*%d[t0];' % (len(angles))
    for i in range(1,len(angles)-1):
        cmd += ' [t%d][%d:v]overlay=0:h*%d[t%d];' % (i-1,i,i,i)
    cmd += ' [t%d][%d:v]overlay=0:h*%d[out]" -map "[out]"' % (len(angles)-2,len(angles)-1,len(angles)-1)
    cmd += ' %s/cat_%s.mp4' % (file_dir, renderType)
else:   #concat horizontalb
    cmd += ' -y -filter_complex "[0:v]pad=iw*%d:ih[t0];' % (len(angles))
    for i in range(1,len(angles)-1):
        cmd += ' [t%d][%d:v]overlay=w*%d:0[t%d];' % (i-1,i,i,i)
    cmd += ' [t%d][%d:v]overlay=w*%d:0[out]" -map "[out]"' % (len(angles)-2,len(angles)-1,len(angles)-1)
    cmd += ' %s/cat_%s.mp4' % (file_dir, renderType)


os.system(cmd)
