'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import io
import os
import torch
from skimage.io import imread
import numpy as np
import cv2
from tqdm import tqdm_notebook as tqdm
import base64
from IPython.display import HTML

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

from IPython.display import HTML
from base64 import b64encode

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLOrthographicCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    TexturesVertex
)

def set_renderer():
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Initialize an OpenGL perspective camera.
    R, T = look_at_view_transform(2.0, 0, 180) 
    cameras = OpenGLOrthographicCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size = None, 
        max_faces_per_bin = None
    )

    lights = PointLights(device=device, location=((2.0, 2.0, 2.0),))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    return renderer

def get_verts_rgb_colors(obj_path):
  rgb_colors = []

  f = open(obj_path)
  lines = f.readlines()
  for line in lines:
    ls = line.split(' ')
    if len(ls) == 7:
      rgb_colors.append(ls[-3:])

  return np.array(rgb_colors, dtype='float32')[None, :, :]

def generate_video_from_obj(obj_path, image_path, video_path, renderer):
    input_image = cv2.imread(image_path)
    input_image = input_image[:,:input_image.shape[1]//3]
    input_image = cv2.resize(input_image, (512,512))

    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load obj file
    verts_rgb_colors = get_verts_rgb_colors(obj_path)
    verts_rgb_colors = torch.from_numpy(verts_rgb_colors).to(device)
    textures = TexturesVertex(verts_features=verts_rgb_colors)
    # wo_textures = TexturesVertex(verts_features=torch.ones_like(verts_rgb_colors)*0.75)

    # Load obj
    mesh = load_objs_as_meshes([obj_path], device=device)

    # Set mesh
    vers = mesh._verts_list
    faces = mesh._faces_list
    mesh_w_tex = Meshes(vers, faces, textures)
    # mesh_wo_tex = Meshes(vers, faces, wo_textures)

    # create VideoWriter
    fourcc = cv2. VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (1024,512))

    for i in tqdm(range(90)):
        R, T = look_at_view_transform(1.8, 0, i*4, device=device)
        images_w_tex = renderer(mesh_w_tex, R=R, T=T)
        images_w_tex = np.clip(images_w_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
        # images_wo_tex = renderer(mesh_wo_tex, R=R, T=T)
        # images_wo_tex = np.clip(images_wo_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
        image = np.concatenate([input_image, images_w_tex], axis=1)
        out.write(image.astype('uint8'))
    out.release()

def video(path):
    mp4 = open(path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML('<video width=500 controls loop> <source src="%s" type="video/mp4"></video>' % data_url)
