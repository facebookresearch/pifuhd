import numpy as np
import random

from .framework import *
from .cam_render import CamRender
from OpenGL.GLUT import *

class PRTRender(CamRender):
    def __init__(self, width=1600, height=1200, name='PRT Renderer', uv_mode=False, ms_rate=1):
        program_files = ['prt.vs', 'prt.fs'] if not uv_mode else ['prt_uv.vs', 'prt_uv.fs']
        CamRender.__init__(self, width, height, name, program_files=program_files, color_size=8, ms_rate=ms_rate)

        # WARNING: this differs from vertex_buffer and vertex_data in Render
        self.vert_buffer = {}
        self.vert_data = {}

        self.norm_buffer = {}
        self.norm_data = {}

        self.tan_buffer = {}
        self.tan_data = {}

        self.btan_buffer = {}
        self.btan_data = {}

        self.prt1_buffer = {}
        self.prt1_data = {}
        self.prt2_buffer = {}
        self.prt2_data = {}        
        self.prt3_buffer = {}
        self.prt3_data = {}

        self.uv_buffer = {}
        self.uv_data = {}

        self.render_texture_mat = {}

        self.vertex_dim = {}
        self.n_vertices = {}

        self.norm_mat_unif = glGetUniformLocation(self.program, 'NormMat')
        self.normalize_matrix = np.eye(4)

        self.shcoeff_unif = glGetUniformLocation(self.program, 'SHCoeffs')
        self.shcoeffs = np.zeros((9,3))
        self.shcoeffs[0,:] = 1.0
        #self.shcoeffs[1:,:] = np.random.rand(8,3)

        self.hasAlbedoUnif = glGetUniformLocation(self.program, 'hasAlbedoMap')
        self.hasNormalUnif = glGetUniformLocation(self.program, 'hasNormalMap')

        self.analyticUnif = glGetUniformLocation(self.program, 'analytic')
        self.analytic = False

        self.rot_mat_unif = glGetUniformLocation(self.program, 'RotMat')
        self.rot_matrix = np.eye(3)

    def set_texture(self, mat_name, smplr_name, texture):
        # texture_image: H x W x 3
        width = texture.shape[1]
        height = texture.shape[0]
        texture = np.flip(texture, 0)
        img_data = np.fromstring(texture.tostring(), np.uint8)

        if mat_name not in self.render_texture_mat:
            self.render_texture_mat[mat_name] = {} 
        if smplr_name in self.render_texture_mat[mat_name].keys():
            glDeleteTextures([self.render_texture_mat[mat_name][smplr_name]])
            del self.render_texture_mat[mat_name][smplr_name]
        self.render_texture_mat[mat_name][smplr_name] = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self.render_texture_mat[mat_name][smplr_name])

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 3)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        glGenerateMipmap(GL_TEXTURE_2D)
        
    def set_albedo(self, texture_image, mat_name='all'):
        self.set_texture(mat_name, 'AlbedoMap', texture_image)

    def set_normal_map(self, texture_image, mat_name='all'):
        self.set_texture(mat_name, 'NormalMap', texture_image)

    def set_mesh(self, vertices, faces, norms, faces_nml, uvs, faces_uvs, prt, faces_prt, tans, bitans,mat_name='all'):
        self.vert_data[mat_name] = vertices[faces.reshape([-1])]
        self.n_vertices[mat_name] = self.vert_data[mat_name].shape[0]
        self.vertex_dim[mat_name] = self.vert_data[mat_name].shape[1]

        if mat_name not in self.vert_buffer.keys():
            self.vert_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vert_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.vert_data[mat_name], GL_STATIC_DRAW)

        self.uv_data[mat_name] = uvs[faces_uvs.reshape([-1])]
        if mat_name not in self.uv_buffer.keys():
            self.uv_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.uv_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.uv_data[mat_name], GL_STATIC_DRAW)

        self.norm_data[mat_name] = norms[faces_nml.reshape([-1])]
        if mat_name not in self.norm_buffer.keys():
            self.norm_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.norm_data[mat_name], GL_STATIC_DRAW)

        self.tan_data[mat_name] = tans[faces_nml.reshape([-1])]
        if mat_name not in self.tan_buffer.keys():
            self.tan_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.tan_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.tan_data[mat_name], GL_STATIC_DRAW)

        self.btan_data[mat_name] = bitans[faces_nml.reshape([-1])]
        if mat_name not in self.btan_buffer.keys():
            self.btan_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.btan_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.btan_data[mat_name], GL_STATIC_DRAW)

        self.prt1_data[mat_name] = prt[faces_prt.reshape([-1])][:,:3]
        self.prt2_data[mat_name] = prt[faces_prt.reshape([-1])][:,3:6]
        self.prt3_data[mat_name] = prt[faces_prt.reshape([-1])][:,6:]

        if mat_name not in self.prt1_buffer.keys():
            self.prt1_buffer[mat_name] = glGenBuffers(1)
        if mat_name not in self.prt2_buffer.keys():
            self.prt2_buffer[mat_name] = glGenBuffers(1)
        if mat_name not in self.prt3_buffer.keys():
            self.prt3_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.prt1_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.prt1_data[mat_name], GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.prt2_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.prt2_data[mat_name], GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.prt3_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.prt3_data[mat_name], GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_mesh_mtl(self, vertices, faces, norms, faces_nml, uvs, faces_uvs, prt, faces_prt, tans, bitans):
        for key in faces:
            self.vert_data[key] = vertices[faces[key].reshape([-1])]
            self.n_vertices[key] = self.vert_data[key].shape[0]
            self.vertex_dim[key] = self.vert_data[key].shape[1]

            if key not in self.vert_buffer.keys():
                self.vert_buffer[key] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vert_buffer[key])
            glBufferData(GL_ARRAY_BUFFER, self.vert_data[key], GL_STATIC_DRAW)

            self.uv_data[key] = uvs[faces_uvs[key].reshape([-1])]
            if key not in self.uv_buffer.keys():
                self.uv_buffer[key] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.uv_buffer[key])
            glBufferData(GL_ARRAY_BUFFER, self.uv_data[key], GL_STATIC_DRAW)

            self.norm_data[key] = norms[faces_nml[key].reshape([-1])]
            if key not in self.norm_buffer.keys():
                self.norm_buffer[key] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer[key])
            glBufferData(GL_ARRAY_BUFFER, self.norm_data[key], GL_STATIC_DRAW)

            self.tan_data[key] = tans[faces_nml[key].reshape([-1])]
            if key not in self.tan_buffer.keys():
                self.tan_buffer[key] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.tan_buffer[key])
            glBufferData(GL_ARRAY_BUFFER, self.tan_data[key], GL_STATIC_DRAW)

            self.btan_data[key] = bitans[faces_nml[key].reshape([-1])]
            if key not in self.btan_buffer.keys():
                self.btan_buffer[key] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.btan_buffer[key])
            glBufferData(GL_ARRAY_BUFFER, self.btan_data[key], GL_STATIC_DRAW)

            self.prt1_data[key] = prt[faces_prt[key].reshape([-1])][:,:3]
            self.prt2_data[key] = prt[faces_prt[key].reshape([-1])][:,3:6]
            self.prt3_data[key] = prt[faces_prt[key].reshape([-1])][:,6:]

            if key not in self.prt1_buffer.keys():
                self.prt1_buffer[key] = glGenBuffers(1)
            if key not in self.prt2_buffer.keys():
                self.prt2_buffer[key] = glGenBuffers(1)
            if key not in self.prt3_buffer.keys():
                self.prt3_buffer[key] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.prt1_buffer[key])
            glBufferData(GL_ARRAY_BUFFER, self.prt1_data[key], GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.prt2_buffer[key])
            glBufferData(GL_ARRAY_BUFFER, self.prt2_data[key], GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.prt3_buffer[key])
            glBufferData(GL_ARRAY_BUFFER, self.prt3_data[key], GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def cleanup(self):
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        for key in self.vert_data:
            glDeleteBuffers(1, [self.vert_buffer[key]])
            glDeleteBuffers(1, [self.norm_buffer[key]])
            glDeleteBuffers(1, [self.uv_buffer[key]])

            glDeleteBuffers(1, [self.tan_buffer[key]])
            glDeleteBuffers(1, [self.btan_buffer[key]])
            glDeleteBuffers(1, [self.prt1_buffer[key]])
            glDeleteBuffers(1, [self.prt2_buffer[key]])
            glDeleteBuffers(1, [self.prt3_buffer[key]])

            glDeleteBuffers(1, [])

            for smplr in self.render_texture_mat[key]:
                glDeleteTextures([self.render_texture_mat[key][smplr]])

        self.vert_buffer = {}
        self.vert_data = {}

        self.norm_buffer = {}
        self.norm_data = {}

        self.tan_buffer = {}
        self.tan_data = {}

        self.btan_buffer = {}
        self.btan_data = {}

        self.prt1_buffer = {}
        self.prt1_data = {}

        self.prt2_buffer = {}
        self.prt2_data = {}

        self.prt3_buffer = {}
        self.prt3_data = {}

        self.uv_buffer = {}
        self.uv_data = {}

        self.render_texture_mat = {}

        self.vertex_dim = {}
        self.n_vertices = {}
    
    def randomize_sh(self):
        self.shcoeffs[0,:] = 0.8
        self.shcoeffs[1:,:] = 1.0*np.random.rand(8,3)

    def set_sh(self, sh):
        self.shcoeffs = sh

    def set_norm_mat(self, scale, center):
        N = np.eye(4)
        N[:3, :3] = scale*np.eye(3)
        N[:3, 3] = -scale*center

        self.normalize_matrix = N

    def draw(self):
        glViewport(0, 0, self.width, self.height)

        self.draw_init()

        glDisable(GL_BLEND)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_MULTISAMPLE)

        glUseProgram(self.program)
        glUniformMatrix4fv(self.norm_mat_unif, 1, GL_FALSE, self.normalize_matrix.transpose())
        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix.transpose())
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix.transpose())

        if 'AlbedoMap' in self.render_texture_mat['all']:
            glUniform1ui(self.hasAlbedoUnif, GLuint(1))
        else:
            glUniform1ui(self.hasAlbedoUnif, GLuint(0))

        if 'NormalMap' in self.render_texture_mat['all']:
            glUniform1ui(self.hasNormalUnif, GLuint(1))
        else:
            glUniform1ui(self.hasNormalUnif, GLuint(0))

        glUniform1ui(self.analyticUnif, GLuint(1) if self.analytic else GLuint(0))

        glUniform3fv(self.shcoeff_unif, 9, self.shcoeffs)

        glUniformMatrix3fv(self.rot_mat_unif, 1, GL_FALSE, self.rot_matrix.transpose())

        for mat in self.vert_buffer:
            # Handle vertex buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.vert_buffer[mat])
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, self.vertex_dim[mat], GL_DOUBLE, GL_FALSE, 0, None)

            # Handle normal buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer[mat])
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, None)

            # Handle uv buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.uv_buffer[mat])
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_DOUBLE, GL_FALSE, 0, None)

            # Handle tan buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.tan_buffer[mat])
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 3, GL_DOUBLE, GL_FALSE, 0, None)

            # Handle btan buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.btan_buffer[mat])
            glEnableVertexAttribArray(4)
            glVertexAttribPointer(4, 3, GL_DOUBLE, GL_FALSE, 0, None)

            # Handle PTR buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.prt1_buffer[mat])
            glEnableVertexAttribArray(5)
            glVertexAttribPointer(5, 3, GL_DOUBLE, GL_FALSE, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, self.prt2_buffer[mat])
            glEnableVertexAttribArray(6)
            glVertexAttribPointer(6, 3, GL_DOUBLE, GL_FALSE, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, self.prt3_buffer[mat])
            glEnableVertexAttribArray(7)
            glVertexAttribPointer(7, 3, GL_DOUBLE, GL_FALSE, 0, None)

            for i, smplr in enumerate(self.render_texture_mat[mat]):
                glActiveTexture(GL_TEXTURE0 + i)
                glBindTexture(GL_TEXTURE_2D, self.render_texture_mat[mat][smplr])
                glUniform1i(glGetUniformLocation(self.program, smplr), i)

            glDrawArrays(GL_TRIANGLES, 0, self.n_vertices[mat])

            glDisableVertexAttribArray(7)
            glDisableVertexAttribArray(6)
            glDisableVertexAttribArray(5)
            glDisableVertexAttribArray(4)
            glDisableVertexAttribArray(3)
            glDisableVertexAttribArray(2)
            glDisableVertexAttribArray(1)
            glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)

        self.draw_end()
