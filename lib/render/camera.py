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
import cv2
import numpy as np

from .glm import ortho


class Camera:
    def __init__(self, width=1600, height=1200):
        # Focal Length
        # equivalent 50mm
        focal = np.sqrt(width * width + height * height)
        self.focal_x = focal
        self.focal_y = focal
        # Principal Point Offset
        self.principal_x = width / 2
        self.principal_y = height / 2
        # Axis Skew
        self.skew = 0
        # Image Size
        self.width = width
        self.height = height

        self.near = 1
        self.far = 10

        # Camera Center
        self.eye = np.array([0, 0, -3.6])
        self.center = np.array([0, 0, 0])
        self.direction = np.array([0, 0, -1])
        self.right = np.array([1, 0, 0])
        self.up = np.array([0, 1, 0])

        self.ortho_ratio = None

    def sanity_check(self):
        self.center = self.center.reshape([-1])
        self.direction = self.direction.reshape([-1])
        self.right = self.right.reshape([-1])
        self.up = self.up.reshape([-1])

        assert len(self.center) == 3
        assert len(self.direction) == 3
        assert len(self.right) == 3
        assert len(self.up) == 3

    @staticmethod
    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    def get_real_z_value(self, z):
        z_near = self.near
        z_far = self.far
        z_n = 2.0 * z - 1.0
        z_e = 2.0 * z_near * z_far / (z_far + z_near - z_n * (z_far - z_near))
        return z_e

    def get_rotation_matrix(self):
        rot_mat = np.eye(3)
        d = self.eye - self.center
        d = -self.normalize_vector(d)
        u = self.up
        self.right = -np.cross(u, d)
        u = np.cross(d, self.right)
        rot_mat[0, :] = self.right
        rot_mat[1, :] = u
        rot_mat[2, :] = d

        # s = self.right
        # s = self.normalize_vector(s)
        # rot_mat[0, :] = s
        # u = self.up
        # u = self.normalize_vector(u)
        # rot_mat[1, :] = -u
        # rot_mat[2, :] = self.normalize_vector(self.direction)

        return rot_mat

    def get_translation_vector(self):
        rot_mat = self.get_rotation_matrix()
        trans = -np.dot(rot_mat.T, self.eye)
        return trans

    def get_intrinsic_matrix(self):
        int_mat = np.eye(3)

        int_mat[0, 0] = self.focal_x
        int_mat[1, 1] = self.focal_y
        int_mat[0, 1] = self.skew
        int_mat[0, 2] = self.principal_x
        int_mat[1, 2] = self.principal_y

        return int_mat

    def get_projection_matrix(self):
        ext_mat = self.get_extrinsic_matrix()
        int_mat = self.get_intrinsic_matrix()

        return np.matmul(int_mat, ext_mat)

    def get_extrinsic_matrix(self):
        rot_mat = self.get_rotation_matrix()
        int_mat = self.get_intrinsic_matrix()
        trans = self.get_translation_vector()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_mat
        extrinsic[:3, 3] = trans

        return extrinsic[:3, :]

    def set_rotation_matrix(self, rot_mat):
        self.direction = rot_mat[2, :]
        self.up = -rot_mat[1, :]
        self.right = rot_mat[0, :]

    def set_intrinsic_matrix(self, int_mat):
        self.focal_x = int_mat[0, 0]
        self.focal_y = int_mat[1, 1]
        self.skew = int_mat[0, 1]
        self.principal_x = int_mat[0, 2]
        self.principal_y = int_mat[1, 2]

    def set_projection_matrix(self, proj_mat):
        res = cv2.decomposeProjectionMatrix(proj_mat)
        int_mat, rot_mat, camera_center_homo = res[0], res[1], res[2]
        camera_center = camera_center_homo[0:3] / camera_center_homo[3]
        camera_center = camera_center.reshape(-1)
        int_mat = int_mat / int_mat[2][2]

        self.set_intrinsic_matrix(int_mat)
        self.set_rotation_matrix(rot_mat)
        self.center = camera_center

        self.sanity_check()

    def get_gl_matrix(self):
        z_near = self.near
        z_far = self.far
        rot_mat = self.get_rotation_matrix()
        int_mat = self.get_intrinsic_matrix()
        trans = self.get_translation_vector()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_mat
        extrinsic[:3, 3] = trans
        axis_adj = np.eye(4)
        axis_adj[2, 2] = -1
        axis_adj[1, 1] = -1
        model_view = np.matmul(axis_adj, extrinsic)

        projective = np.zeros([4, 4])
        projective[:2, :2] = int_mat[:2, :2]
        projective[:2, 2:3] = -int_mat[:2, 2:3]
        projective[3, 2] = -1
        projective[2, 2] = (z_near + z_far)
        projective[2, 3] = (z_near * z_far)

        if self.ortho_ratio is None:
            ndc = ortho(0, self.width, 0, self.height, z_near, z_far)
            perspective = np.matmul(ndc, projective)
        else:
            perspective = ortho(-self.width * self.ortho_ratio / 2, self.width * self.ortho_ratio / 2,
                                -self.height * self.ortho_ratio / 2, self.height * self.ortho_ratio / 2,
                                z_near, z_far)

        return perspective, model_view


def KRT_from_P(proj_mat, normalize_K=True):
    res = cv2.decomposeProjectionMatrix(proj_mat)
    K, Rot, camera_center_homog = res[0], res[1], res[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    trans = -Rot.dot(camera_center)
    if normalize_K:
        K = K / K[2][2]
    return K, Rot, trans


def MVP_from_P(proj_mat, width, height, near=0.1, far=10000):
    '''
    Convert OpenCV camera calibration matrix to OpenGL projection and model view matrix
    :param proj_mat: OpenCV camera projeciton matrix
    :param width: Image width
    :param height: Image height
    :param near: Z near value
    :param far: Z far value
    :return: OpenGL projection matrix and model view matrix
    '''
    res = cv2.decomposeProjectionMatrix(proj_mat)
    K, Rot, camera_center_homog = res[0], res[1], res[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    trans = -Rot.dot(camera_center)
    K = K / K[2][2]

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rot
    extrinsic[:3, 3:4] = trans
    axis_adj = np.eye(4)
    axis_adj[2, 2] = -1
    axis_adj[1, 1] = -1
    model_view = np.matmul(axis_adj, extrinsic)

    zFar = far
    zNear = near
    projective = np.zeros([4, 4])
    projective[:2, :2] = K[:2, :2]
    projective[:2, 2:3] = -K[:2, 2:3]
    projective[3, 2] = -1
    projective[2, 2] = (zNear + zFar)
    projective[2, 3] = (zNear * zFar)

    ndc = ortho(0, width, 0, height, zNear, zFar)

    perspective = np.matmul(ndc, projective)

    return perspective, model_view
