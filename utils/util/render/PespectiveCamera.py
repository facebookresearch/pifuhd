from .BaseCamera import BaseCamera
import numpy as np
import math


class PersPectiveCamera(BaseCamera):
    def __init__(self):
        BaseCamera.__init__(self, "PerspectiveCamera")

    def get_projection_mat(self):
        # http://www.songho.ca/opengl/gl_projectionmatrix.html
        projection_mat = np.eye(4)
        projection_mat[0, 0] = 2 / self.magnification_x
        projection_mat[1, 1] = 2 / self.magnification_y
        projection_mat[2, 2] = -(self.far + self.near) / (self.far - self.near)
        projection_mat[2, 3] = -(2 * self.far * self.near) / (self.far - self.near)
        projection_mat[3, 2] = -1
        projection_mat[3, 3] = 0
        return projection_mat

    def set_by_field_of_view(self, fov_x, fov_y=None):
        '''
        Set the intrinsic by given field of view, in angle degrees
        :param fov_x:
        :param fov_y: Optional for y direction; Use the same value as for x direction if None
        '''
        if fov_y is None:
            fov_y = fov_x
        self.set_parameters(
            magnification_x=2 * math.tan(fov_x / 2),
            magnification_y=2 * math.tan(fov_y / 2),
        )

    def set_by_35mm_equivalent_focal_length(self, focal_x, focal_y=None):
        '''
        Set the intrinsic by given 35mm equivalent focal lengths.
        https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
        :param focal_x:
        :param focal_y: Optional for y direction; Use the same value as for x direction if None
        '''
        if focal_y is None:
            focal_y = focal_x
        # 35mm equivalent sensor width and height for this camera
        film_35mm_height = math.sqrt((36 ** 2 + 24 ** 2) / (1 + self.aspect_ratio ** 2))
        film_35mm_width = film_35mm_height * self.aspect_ratio

        self.set_parameters(
            magnification_x=film_35mm_width / focal_x,
            magnification_y=film_35mm_height / focal_y
        )

    def set_by_sensor_and_focal_length(self, sensor_width, sensor_height, focal_x, focal_y=None):
        self.aspect_ratio = sensor_width / sensor_height
        if focal_y is None:
            focal_y = focal_x
        # 35mm equivalent sensor width and height for this camera
        self.set_parameters(
            magnification_x=sensor_width / focal_x,
            magnification_y=sensor_height / focal_y
        )
