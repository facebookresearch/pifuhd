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
import numpy as np
from OpenGL.GLUT import *
from .framework import *

_glut_window = None

class Render:
    def __init__(self, width=1600, height=1200, name='GL Renderer',
                 program_files=['simple.fs', 'simple.vs'], color_size=1, ms_rate=1):
        self.width = width
        self.height = height
        self.name = name
        self.display_mode = GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH
        self.use_inverse_depth = False

        global _glut_window
        if _glut_window is None:
            glutInit()
            glutInitDisplayMode(self.display_mode)
            glutInitWindowSize(self.width, self.height)
            glutInitWindowPosition(0, 0)
            _glut_window = glutCreateWindow("My Render.")

            # glEnable(GL_DEPTH_CLAMP)
            glEnable(GL_DEPTH_TEST)

            glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE)
            glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE)
            glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE)

        # init program
        shader_list = []

        for program_file in program_files:
            _, ext = os.path.splitext(program_file)
            if ext == '.vs':
                shader_list.append(loadShader(GL_VERTEX_SHADER, program_file))
            elif ext == '.fs':
                shader_list.append(loadShader(GL_FRAGMENT_SHADER, program_file))
            elif ext == '.gs':
                shader_list.append(loadShader(GL_GEOMETRY_SHADER, program_file))

        self.program = createProgram(shader_list)

        for shader in shader_list:
            glDeleteShader(shader)

        # Init uniform variables
        self.model_mat_unif = glGetUniformLocation(self.program, 'ModelMat')
        self.persp_mat_unif = glGetUniformLocation(self.program, 'PerspMat')

        self.vertex_buffer = glGenBuffers(1)

        # Init screen quad program and buffer
        self.quad_program, self.quad_buffer = self.init_quad_program()

        # Configure frame buffer
        self.frame_buffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)

        self.intermediate_fbo = None
        if ms_rate > 1:
            # Configure texture buffer to render to
            self.color_buffer = []
            for i in range(color_size):
                color_buffer = glGenTextures(1)
                multi_sample_rate = ms_rate
                glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_buffer)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, multi_sample_rate, GL_RGBA32F, self.width, self.height, GL_TRUE)
                glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D_MULTISAMPLE, color_buffer, 0)
                self.color_buffer.append(color_buffer)

            self.render_buffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.render_buffer)
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, multi_sample_rate, GL_DEPTH24_STENCIL8, self.width, self.height)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.render_buffer)

            attachments = []
            for i in range(color_size):
                attachments.append(GL_COLOR_ATTACHMENT0 + i)
            glDrawBuffers(color_size, attachments)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            self.intermediate_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.intermediate_fbo)

            self.screen_texture = []
            for i in range(color_size):
                screen_texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, screen_texture)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, screen_texture, 0)
                self.screen_texture.append(screen_texture)

            glDrawBuffers(color_size, attachments)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        else:
            self.color_buffer = []
            for i in range(color_size):
                color_buffer = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, color_buffer)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, color_buffer, 0)
                self.color_buffer.append(color_buffer)
 
            # Configure depth texture map to render to
            self.depth_buffer = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.depth_buffer)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.width, self.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_buffer, 0)

            attachments = []
            for i in range(color_size):
                attachments.append(GL_COLOR_ATTACHMENT0 + i)
            glDrawBuffers(color_size, attachments)
            self.screen_texture = self.color_buffer

            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        
        # Configure texture buffer if needed
        self.render_texture = None

        # NOTE: original render_texture only support one input
        # this is tentative member of this issue
        self.render_texture_v2 = {}

        # Inner storage for buffer data
        self.vertex_data = None
        self.vertex_dim = None
        self.n_vertices = None

        self.model_view_matrix = None
        self.projection_matrix = None

        glutDisplayFunc(self.display)


    def init_quad_program(self):
        shader_list = []

        shader_list.append(loadShader(GL_VERTEX_SHADER, "quad.vs"))
        shader_list.append(loadShader(GL_FRAGMENT_SHADER, "quad.fs"))

        the_program = createProgram(shader_list)

        for shader in shader_list:
            glDeleteShader(shader)

        # vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        # positions # texCoords
        quad_vertices = np.array(
            [-1.0, 1.0, 0.0, 1.0,
             -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,

             -1.0, 1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 0.0,
             1.0, 1.0, 1.0, 1.0]
        )

        quad_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, quad_buffer)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return the_program, quad_buffer

    def set_mesh(self, vertices, faces):
        self.vertex_data = vertices[faces.reshape([-1])]
        self.vertex_dim = self.vertex_data.shape[1]
        self.n_vertices = self.vertex_data.shape[0]

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_viewpoint(self, projection, model_view):
        self.projection_matrix = projection
        self.model_view_matrix = model_view

    def draw_init(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        glEnable(GL_DEPTH_TEST)

        # glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearColor(1.0, 1.0, 1.0, 0.0)        #Black background

        if self.use_inverse_depth:
            glDepthFunc(GL_GREATER)
            glClearDepth(0.0)
        else:
            glDepthFunc(GL_LESS)
            glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def draw_end(self):
        if self.intermediate_fbo is not None:
            for i in range(len(self.color_buffer)):
                glBindFramebuffer(GL_READ_FRAMEBUFFER, self.frame_buffer)
                glReadBuffer(GL_COLOR_ATTACHMENT0 + i)
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.intermediate_fbo)
                glDrawBuffer(GL_COLOR_ATTACHMENT0 + i)
                glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, GL_COLOR_BUFFER_BIT, GL_NEAREST)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)

    def draw(self):
        self.draw_init()

        glUseProgram(self.program)
        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix.transpose())
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix.transpose())

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, self.vertex_dim, GL_DOUBLE, GL_FALSE, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)

        glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

        self.draw_end()

    def get_color(self, color_id=0):
        glBindFramebuffer(GL_FRAMEBUFFER, self.intermediate_fbo if self.intermediate_fbo is not None else self.frame_buffer)
        glReadBuffer(GL_COLOR_ATTACHMENT0 + color_id)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT, outputType=None)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        rgb = data.reshape(self.height, self.width, -1)
        rgb = np.flip(rgb, 0)
        return rgb

    def get_z_value(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        data = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT, outputType=None)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        z = data.reshape(self.height, self.width)
        z = np.flip(z, 0)
        return z

    def display(self):
        # First we draw a scene.
        # Notice the result is stored in the texture buffer.
        self.draw()

        # Then we return to the default frame buffer since we will display on the screen.
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Do the clean-up.
        # glClearColor(0.0, 0.0, 0.0, 0.0)        #Black background
        glClearColor(1.0, 1.0, 1.0, 0.0)        #Black background
        glClear(GL_COLOR_BUFFER_BIT)

        # We draw a rectangle which covers the whole screen.
        glUseProgram(self.quad_program)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_buffer)

        size_of_double = 8
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, 4 * size_of_double, None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_DOUBLE, GL_FALSE, 4 * size_of_double, c_void_p(2 * size_of_double))

        glDisable(GL_DEPTH_TEST)

        # The stored texture is then mapped to this rectangle.
        # properly assing color buffer texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.screen_texture[0])
        glUniform1i(glGetUniformLocation(self.quad_program, 'screenTexture'), 0)

        glDrawArrays(GL_TRIANGLES, 0, 6)

        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(0)

        glEnable(GL_DEPTH_TEST)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

        glutSwapBuffers()
        glutPostRedisplay()

    def show(self):
        glutMainLoop()
