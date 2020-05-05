import numpy as np
from sample.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
from sample.camera import Camera
import os
import cv2
import time
import math
import random
import pyexr

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

def rotateSH(SH, R):
    SHn = SH
    
    # 1st order
    SHn[1] = R[1,1]*SH[1] - R[1,2]*SH[2] + R[1,0]*SH[3]
    SHn[2] = -R[2,1]*SH[1] + R[2,2]*SH[2] - R[2,0]*SH[3]
    SHn[3] = R[0,1]*SH[1] - R[0,2]*SH[2] + R[0,0]*SH[3]

    # 2nd order
    SHn[4:,0] = rotateBand2(SH[4:,0],R)
    SHn[4:,1] = rotateBand2(SH[4:,1],R)
    SHn[4:,2] = rotateBand2(SH[4:,2],R)

    return SHn

def rotateBand2(x, R):
    s_c3 = 0.94617469575
    s_c4 = -0.31539156525
    s_c5 = 0.54627421529

    s_c_scale = 1.0/0.91529123286551084
    s_c_scale_inv = 0.91529123286551084

    s_rc2 = 1.5853309190550713*s_c_scale
    s_c4_div_c3 = s_c4/s_c3
    s_c4_div_c3_x2 = (s_c4/s_c3)*2.0

    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv

    sh0 =  x[3] + x[4] + x[4] - x[1]
    sh1 =  x[0] + s_rc2*x[2] +  x[3] + x[4]
    sh2 =  x[0]
    sh3 = -x[3]
    sh4 = -x[1]
    
    r2x = R[0][0] + R[0][1]
    r2y = R[1][0] + R[1][1]
    r2z = R[2][0] + R[2][1]
    
    r3x = R[0][0] + R[0][2]
    r3y = R[1][0] + R[1][2]
    r3z = R[2][0] + R[2][2]
    
    r4x = R[0][1] + R[0][2]
    r4y = R[1][1] + R[1][2]
    r4z = R[2][1] + R[2][2]
    
    sh0_x = sh0 * R[0][0]
    sh0_y = sh0 * R[1][0]
    d0 = sh0_x * R[1][0]
    d1 = sh0_y * R[2][0]
    d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3)
    d3 = sh0_x * R[2][0]
    d4 = sh0_x * R[0][0] - sh0_y * R[1][0]
    
    sh1_x = sh1 * R[0][2]
    sh1_y = sh1 * R[1][2]
    d0 += sh1_x * R[1][2]
    d1 += sh1_y * R[2][2]
    d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3)
    d3 += sh1_x * R[2][2]
    d4 += sh1_x * R[0][2] - sh1_y * R[1][2]
    
    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y
    
    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y
    
    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y
    
    dst = x
    dst[0] = d0
    dst[1] = -d1
    dst[2] = d2 * s_scale_dst2
    dst[3] = -d3
    dst[4] = d4 * s_scale_dst4

    return dst


def render_prt_ortho_fullbody(out_path, joint_path, folder_name, subject_name, shs, render, render_uv, light_var=1, pitch=[0], size=1024):
    cam = Camera(width=size, height=size)
    cam.ortho_ratio = 0.2 * (1024 // size)
    cam.near = -200
    cam.far = 200
    cam.sanity_check()

    cam_uv = Camera(width=size, height=size)
    cam_uv.ortho_ratio = 0.2 * (1024 // size)
    cam_uv.near = -200
    cam_uv.far = 200

    mesh_file = os.path.join(folder_name, subject_name + '_100k.obj')
    if not os.path.exists(mesh_file):
        mesh_file = os.path.join(folder_name, subject_name + '_100k.OBJ')       
        if not os.path.exists(mesh_file):
            print('ERROR: obj file does not exist!!', mesh_file)
            return 
    prt_file = os.path.join(folder_name, 'bounce', 'bounce0.txt')
    if not os.path.exists(prt_file):
        print('ERROR: prt file does not exist!!!', prt_file)
        return
    faces_prt = os.path.join(folder_name, 'bounce', 'face.npy')
    faces_prt = np.load(faces_prt)
    text_file = os.path.join(folder_name, 'tex', subject_name + '_dif_8k.jpg')
    if not os.path.exists(text_file):
        text_file = os.path.join(folder_name, 'tex', subject_name + '_dif_8k.JPG')
        if not os.path.exists(text_file):
            print('ERROR: dif file does not exist!!', text_file)
            return             
    nmap_file = os.path.join(folder_name, 'tex', subject_name + '_norm_8k.jpg')

    joints_file = os.path.join(joint_path, subject_name + '_100k.npy')
    joint3d = np.load(joints_file)

    texture_image = cv2.imread(text_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    normal_image = cv2.imread(nmap_file)
    normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)

    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(mesh_file, with_normal=True, with_texture=True)
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    
    vmed = joint3d[8]
    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_scale = 180/(vmax[up_axis] - vmin[up_axis])

    render.set_norm_mat(y_scale, vmed)
    render_uv.set_norm_mat(y_scale, vmed)

    tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    prt = np.loadtxt(prt_file)
    render.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, faces_prt, tan, bitan)
    render.set_albedo(texture_image)
    render.set_normal_map(normal_image)

    render_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, faces_prt, tan, bitan)
    render_uv.set_albedo(texture_image)
    render_uv.set_normal_map(normal_image)

    for p in pitch:
        for y in list(range(0,360,2)):
            R = np.matmul(make_rotate(math.radians(p), 0, 0), make_rotate(0, math.radians(y), 0))
            if up_axis == 2:
                R = np.matmul(R, make_rotate(math.radians(90),0,0))

            render.rot_matrix = R
            render_uv.rot_matrix = R
            render.set_camera(cam)
            render_uv.set_camera(cam_uv)

            for j in range(light_var):
                sh_id = random.randint(0,shs.shape[0]-1)
                sh = shs[sh_id]
                sh_angle = 0.2*np.pi*(random.random()-0.5)
                sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

                dic = {'sh': sh, 'ortho_ratio': cam.ortho_ratio, 'scale': y_scale, 'center': vmed, 'R': R}
                
                render.set_sh(sh)        
                render.analytic = False
                render.use_inverse_depth = False
                render.display()

                out_all_f = render.get_color(0)
                out_mask = out_all_f[:,:,3]
                out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGRA)

                os.makedirs(os.path.join(out_path, 'PARAM', subject_name),exist_ok=True)
                np.save(os.path.join(out_path, 'PARAM', subject_name, '%d_%d_%02d.npy'%(y,p,j)),dic)

                os.makedirs(os.path.join(out_path, 'RENDER', subject_name),exist_ok=True)
                cv2.imwrite(os.path.join(out_path, 'RENDER', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_all_f)

                os.makedirs(os.path.join(out_path, 'MASK', subject_name),exist_ok=True)
                cv2.imwrite(os.path.join(out_path, 'MASK', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_mask)

                render_uv.set_sh(sh)
                render_uv.analytic = False
                render_uv.use_inverse_depth = False
                render_uv.display()

                uv_color = render_uv.get_color(0)
                uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)

                os.makedirs(os.path.join(out_path, 'UV_RENDER', subject_name),exist_ok=True)
                cv2.imwrite(os.path.join(out_path, 'UV_RENDER', subject_name, '%d_%d_%02d.jpg'%(y,p,j)),255.0*uv_color)

                if y == 0 and j == 0 and p == pitch[0]:
                    uv_pos = render_uv.get_color(1)
                    uv_mask = uv_pos[:,:,3]
                    os.makedirs(os.path.join(out_path, 'UV_MASK', subject_name),exist_ok=True)
                    cv2.imwrite(os.path.join(out_path, 'UV_MASK', subject_name, '00.png'),255.0*uv_mask)

                    os.makedirs(os.path.join(out_path, 'UV_POS', subject_name),exist_ok=True)
                    data = {'default': uv_pos[:,:,:3]} # default is a reserved name
                    pyexr.write(os.path.join(out_path, 'UV_POS', subject_name, '00.exr'), data) 

                    uv_nml = render_uv.get_color(2)
                    uv_nml = cv2.cvtColor(uv_nml, cv2.COLOR_RGBA2BGR)
                    os.makedirs(os.path.join(out_path, 'UV_NORMAL', subject_name),exist_ok=True)
                    cv2.imwrite(os.path.join(out_path, 'UV_NORMAL', subject_name, '00.png'),255.0*uv_nml)


def render_prt_ortho_f2b(out_path, joint_path, folder_name, subject_name, shs, render, light_var=1, pitch=[0], size=512):
    cam = Camera(width=size, height=size)
    cam.ortho_ratio = 0.2 * (1024 // size)
    cam.near = -200
    cam.far = 200
    cam.sanity_check()

    mesh_file = os.path.join(folder_name, subject_name + '_100k.obj')
    if not os.path.exists(mesh_file):
        mesh_file = os.path.join(folder_name, subject_name + '_100k.OBJ')       
        if not os.path.exists(mesh_file):
            print('ERROR: obj file does not exist!!', mesh_file)
            return 
    prt_file = os.path.join(folder_name, 'bounce', 'bounce0.txt')
    if not os.path.exists(prt_file):
        print('ERROR: prt file does not exist!!!', prt_file)
        return
    faces_prt = os.path.join(folder_name, 'bounce', 'face.npy')
    faces_prt = np.load(faces_prt)
    text_file = os.path.join(folder_name, 'tex', subject_name + '_dif_8k.jpg')
    if not os.path.exists(text_file):
        text_file = os.path.join(folder_name, 'tex', subject_name + '_dif_8k.JPG')
        if not os.path.exists(text_file):
            print('ERROR: dif file does not exist!!', text_file)
            return             
    nmap_file = os.path.join(folder_name, 'tex', subject_name + '_norm_8k.jpg')

    joints_file = os.path.join(joint_path, subject_name + '_100k.npy')
    joint3d = np.load(joints_file)

    texture_image = cv2.imread(text_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    normal_image = cv2.imread(nmap_file)
    normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)

    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(mesh_file, with_normal=True, with_texture=True)
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    
    vmed = np.median(vertices, 0)
    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_scale = 180/(vmax[up_axis] - vmin[up_axis])

    render.set_norm_mat(y_scale, vmed)

    tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    prt = np.loadtxt(prt_file)
    render.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, faces_prt, tan, bitan)
    render.set_albedo(texture_image)
    render.set_normal_map(normal_image)

    for p in pitch:
        for y in list(range(0,360,2)):
            R = np.matmul(make_rotate(math.radians(p), 0, 0), make_rotate(0, math.radians(y), 0))
            if up_axis == 2:
                R = np.matmul(R, make_rotate(math.radians(90),0,0))

            render.rot_matrix = R
            render.set_camera(cam)

            for j in range(light_var):
                sh_id = random.randint(0,shs.shape[0]-1)
                sh = shs[sh_id]
                sh_angle = 0.2*np.pi*(random.random()-0.5)
                sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

                dic = {'sh': sh, 'ortho_ratio': cam.ortho_ratio, 'scale': y_scale, 'center': vmed, 'R': R}
                
                render.set_sh(sh)        
                render.analytic = False
                render.use_inverse_depth = False
                render.display()

                out_all_f = render.get_color(0)
                out_mask = out_all_f[:,:,3]
                out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGRA)

                os.makedirs(os.path.join(out_path, 'PARAM', subject_name),exist_ok=True)
                np.save(os.path.join(out_path, 'PARAM', subject_name, '%d_%d_%02d.npy'%(y,p,j)),dic)

                os.makedirs(os.path.join(out_path, 'RENDER_F', subject_name),exist_ok=True)
                cv2.imwrite(os.path.join(out_path, 'RENDER_F', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_all_f)
                
                out_nml_f = render.get_color(1)
                out_nml_f = cv2.cvtColor(out_nml_f, cv2.COLOR_RGBA2BGR)
                os.makedirs(os.path.join(out_path, 'NORMAL_F', subject_name),exist_ok=True)
                cv2.imwrite(os.path.join(out_path, 'NORMAL_F', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_nml_f)

                render.use_inverse_depth = True
                render.display()

                out_all_b = render.get_color(0)
                out_mask = out_all_b[:,:,3]
                out_all_b = cv2.cvtColor(out_all_b, cv2.COLOR_RGBA2BGRA)

                os.makedirs(os.path.join(out_path, 'RENDER_B', subject_name),exist_ok=True)
                cv2.imwrite(os.path.join(out_path, 'RENDER_B', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_all_b)
                
                out_nml_b = render.get_color(1)
                out_nml_b = cv2.cvtColor(out_nml_b, cv2.COLOR_RGBA2BGR)
                os.makedirs(os.path.join(out_path, 'NORMAL_B', subject_name),exist_ok=True)
                cv2.imwrite(os.path.join(out_path, 'NORMAL_B', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_nml_b)


if __name__ == '__main__':
    shs = np.load('./env_sh_norm.npy')
    root = '/run/media/hjoo/disk/data/pifuhd/data/renderpeople'
    save_dir = '/run/media/hjoo/disk/data/pifuhd/data/output_test'
    joint_dir = '/run/media/hjoo/disk/data/pifuhd/data/renderpeople/POSE'
    generate_f2b = False
    size = 512
    ms_rate = 16 # Warning: mesa doesn't support multi-sampling. if it's run on headless server, either use ms_rate = 1 or egl

    lists = [f for f in os.listdir(root) if 'rp_' in f]
    lists = sorted(lists)

    from sample.gl.prt_render import PRTRender
    render_uv = PRTRender(width=size, height=size, uv_mode=True)
    render = PRTRender(width=size, height=size, ms_rate=ms_rate)

    if generate_f2b:
        for f in lists:
            render_prt_ortho_f2b(save_dir, joint_dir, os.path.join(root,f), f[:-8], shs, render, pitch=[0], size=size)
    else:
        for f in lists:
            render_prt_ortho_fullbody(save_dir, joint_dir, os.path.join(root,f), f[:-8], shs, render, render_uv, pitch=[0], size=size)