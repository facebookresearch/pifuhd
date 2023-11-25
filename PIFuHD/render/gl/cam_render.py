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

    def set_matrices(self, projection, modelview):
        self.projection_matrix = projection
        self.model_view_matrix = modelview

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
