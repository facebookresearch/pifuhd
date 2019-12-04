import numpy as np


class BaseCamera:
    def __init__(self, name="BaseCamera"):
        self.name = name
        #
        # The 'magnification' property tells effectively,
        # if there is a ruler at one unit away from the camera,
        # how much length will the ruler appear in the image the camera will capture.
        # This property is useful and transparent for both Perspective and Orthogonal cases.
        #
        # For instance,
        # if the world unit is in meter and magnification_x = 1.8,
        # the camera is set at 0.9m height and looking front,
        # a 1.8m tall man standing 1m away from the camera will be exactly in the camera view.
        #

        # How wide the camera can see from left to right at one unit away
        self.magnification_x = 1
        # How tall the camera can see from bottom to top at one unit away
        self.magnification_y = 1

        # Ratio of camera width to height, e.g. 1:1, 4:3, 16:9.
        self.aspect_ratio = 1

        # Close up clamping distance in world unit
        self.near = 0.1
        # Farthest clamping distance in world unit
        self.far = 10

    def get_name(self):
        return self.name

    def set_parameters(self, magnification_x, magnification_y=None):
        '''

        :param magnification_x:
        :param magnification_y: Optional for y direction; Use the same value as for x direction if None
        '''
        if magnification_y is None:
            magnification_y = magnification_x / self.aspect_ratio

        self.magnification_x = magnification_x
        self.magnification_y = magnification_y

    def get_projection_mat(self):
        # http://www.songho.ca/opengl/gl_projectionmatrix.html
        projection_mat = np.eye(4)
        projection_mat[0, 0] = 2 / self.magnification_x
        projection_mat[1, 1] = 2 / self.magnification_y
        projection_mat[2, 2] = -2 / (self.far - self.near)
        projection_mat[2, 3] = -(self.far + self.near) / (self.far - self.near)
        return projection_mat
