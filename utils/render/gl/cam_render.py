from OpenGL.GLUT import *

from .render import Render


class CamRender(Render):
    def __init__(self, width=1600, height=1200, name='Cam Renderer',
                 program_files=['simple.fs', 'simple.vs'], color_size=1, ms_rate=1):
        Render.__init__(self, width, height, name, program_files, color_size, ms_rate)
        self.camera = None

        glutDisplayFunc(self.display)
        glutKeyboardFunc(self.keyboard)

    def set_camera(self, camera):
        self.camera = camera
        self.projection_matrix, self.model_view_matrix = camera.get_gl_matrix()

    def keyboard(self, key, x, y):
        # up
        eps = 1
        # print(key)
        if key == b'w':
            self.camera.center += eps * self.camera.direction
        elif key == b's':
            self.camera.center -= eps * self.camera.direction
        if key == b'a':
            self.camera.center -= eps * self.camera.right
        elif key == b'd':
            self.camera.center += eps * self.camera.right
        if key == b' ':
            self.camera.center += eps * self.camera.up
        elif key == b'x':
            self.camera.center -= eps * self.camera.up
        elif key == b'i':
            self.camera.near += 0.1 * eps
            self.camera.far += 0.1 * eps
        elif key == b'o':
            self.camera.near -= 0.1 * eps
            self.camera.far -= 0.1 * eps

        self.projection_matrix, self.model_view_matrix = self.camera.get_gl_matrix()

    def show(self):
        glutMainLoop()
