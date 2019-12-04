import numpy as np


class CameraPose:
    def __init__(self):
        # Camera's center in world coordinate
        self.center = np.array([0.0, 0.0, 1.0])
        # Camera's Z axis direction in world coordinate
        self.front = np.array([0.0, 0.0, 1.0])
        # Camera's X axis direction in world coordinate
        self.right = np.array([1.0, 0.0, 0.0])
        # Camera's Y axis direction in world coordinate
        self.up = np.array([0.0, 1.0, 0.0])

        self.sanity_check()

    def sanity_check(self):
        self.center = np.array(self.center).reshape([-1])

        self.front = np.array(self.front).reshape([-1])
        self.front = self.normalize_vector(self.front)

        self.up = np.array(self.up).reshape([-1])
        self.right = np.cross(self.up, self.front)
        self.right = self.normalize_vector(self.right)

        self.up = np.cross(self.front, self.right)
        self.up = self.normalize_vector(self.up)

        assert len(self.center) == 3
        assert len(self.front) == 3
        assert len(self.right) == 3
        assert len(self.up) == 3

    @staticmethod
    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    def get_rotation_matrix(self):
        rot_mat = np.eye(3)
        rot_mat[0, :] = self.right
        rot_mat[1, :] = self.up
        rot_mat[2, :] = self.front
        return rot_mat

    def get_translation_vector(self):
        rot_mat = self.get_rotation_matrix()
        trans = -np.dot(rot_mat, self.center)
        return trans

    def get_model_view_mat(self):
        self.sanity_check()
        model_view = np.eye(4)
        model_view[:3, :3] = self.get_rotation_matrix()
        model_view[:3, 3] = self.get_translation_vector()
        return model_view
