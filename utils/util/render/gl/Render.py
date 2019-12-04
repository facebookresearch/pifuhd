import numpy as np
from .Shader import *


class Render(object):
    def __init__(self,
                 width, height,
                 multi_sample_rate=1,
                 num_render_target=1
                 ):
        self.width = width
        self.height = height

        self.vbo_list = []
        self.uniform_dict = {}
        self.texture_dict = {}

        # Configure frame buffer
        # This is an off screen frame buffer holds the rendering results.
        # During display, it is drawn onto the screen by a quad program.
        self.color_fbo = FBO()
        self.color_tex_list = []

        with self.color_fbo:
            for i in range(num_render_target):
                color_tex = Texture()
                self.color_tex_list.append(color_tex)
                with color_tex:
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0,
                                    gl.GL_RGBA, gl.GL_FLOAT, None)
                    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0 + i,
                                              gl.GL_TEXTURE_2D, color_tex, 0)
            gl.glViewport(0, 0, width, height)
            assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE

        # This is the actual frame buffer the shader program is rendered to.
        # For multi-sampling, it is different than the default fbo.
        self._shader_fbo = self.color_fbo
        self._shader_color_tex_list = self.color_tex_list

        # However, if multi-sampling is enabled, we need additional render target textures.
        if multi_sample_rate > 1:
            self._shader_fbo = FBO()
            self._shader_color_tex_list = []
            with self._shader_fbo:
                for i in range(num_render_target):
                    color_tex = MultiSampleTexture()
                    self._shader_color_tex_list.append(color_tex)
                    with color_tex:
                        gl.glTexImage2DMultisample(gl.GL_TEXTURE_2D_MULTISAMPLE, multi_sample_rate, gl.GL_RGBA32F,
                                                   width, height, gl.GL_TRUE)
                    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0 + i,
                                              gl.GL_TEXTURE_2D_MULTISAMPLE, color_tex, 0)

        # Configure depth buffer
        self._shader_depth_tex = RBO()
        with self._shader_fbo:
            with self._shader_depth_tex:
                gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, multi_sample_rate,
                                                    gl.GL_DEPTH24_STENCIL8, width, height)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT,
                                             gl.GL_RENDERBUFFER, self._shader_depth_tex)

        self._init_shader()
        for uniform in self.uniform_dict:
            self.uniform_dict[uniform]["handle"] = gl.glGetUniformLocation(self.shader, uniform)

    def _init_shader(self):
        self.shader = Shader(vs_file='simple.vs', fs_file='simple.fs', gs_file=None)
        # layout (location = 0) in vec3 Position;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))

        # Declare all uniform used in the program
        self.shader.declare_uniform('ModelMat', type_code='f', gl_type=gl.glUniformMatrix4fv)
        self.shader.declare_uniform('PerspMat', type_code='f', gl_type=gl.glUniformMatrix4fv)

    def set_attrib(self, attrib_id, data):
        if not 0 <= attrib_id < len(self.vbo_list):
            print("Error: Attrib index out if bound.")
            return
        vbo = self.vbo_list[attrib_id]
        with vbo:
            data = np.ascontiguousarray(data, vbo.type_code)
            vbo.dim = data.shape[-1]
            vbo.size = data.shape[0]
            gl.glBufferData(gl.GL_ARRAY_BUFFER, data, gl.GL_STATIC_DRAW)

    def set_texture(self, name, texture_image):
        if name not in self.texture_dict:
            print("Error: Unknown texture name.")
            return
        width = texture_image.shape[1]
        height = texture_image.shape[0]
        texture_image = np.flip(texture_image, 0)
        img_data = np.fromstring(texture_image.tostring(), np.uint8)
        tex = self.texture_dict[name]
        with tex:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_data)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 3)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    def draw(self,
             uniform_dict,
             clear_color=[0, 0, 0, 0],
             ):
        with self._shader_fbo:
            # Clean up
            gl.glClearColor(*clear_color)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            with self.shader:
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthFunc(gl.GL_LESS)

                # Setup shader uniforms
                for uniform_name in uniform_dict:
                    self.shader.set_uniform(uniform_name, uniform_dict[uniform_name])

                # Setup up VertexAttrib
                for attrib_id in range(len(self.vbo_list)):
                    vbo = self.vbo_list[attrib_id]
                    with vbo:
                        gl.glEnableVertexAttribArray(attrib_id)
                        gl.glVertexAttribPointer(attrib_id, vbo.dim, vbo.gl_type, gl.GL_FALSE, 0, None)

                # Setup Textures
                for i, texture_name in enumerate(self.texture_dict):
                    gl.glActiveTexture(gl.GL_TEXTURE0 + i)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_dict[texture_name])
                    gl.glUniform1i(gl.glGetUniformLocation(self.shader, texture_name), i)

                # Setup targets
                color_size = len(self.color_tex_list)
                attachments = [gl.GL_COLOR_ATTACHMENT0 + i for i in range(color_size)]
                gl.glDrawBuffers(color_size, attachments)

                gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vbo_list[0].size)

                for attrib_id in range(len(self.vbo_list)):
                    gl.glDisableVertexAttribArray(attrib_id)

                gl.glDisable(gl.GL_DEPTH_TEST)

        # If render_fbo is not color_fbo, we need to copy data
        if self._shader_fbo != self.color_fbo:
            for i in range(len(self.color_tex_list)):
                gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._shader_fbo)
                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.color_fbo)
                gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
                gl.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height,
                                     gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)

    def get_color(self, color_id=0):
        with self.color_fbo:
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + color_id)
            data = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, outputType=None)
            frame = data.reshape(self.height, self.width, -1)
            frame = frame[::-1]  # vertical flip to match GL convention
        return frame
