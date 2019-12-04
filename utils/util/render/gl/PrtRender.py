from .Render import *


class PrtRender(Render):
    def __init__(self,
                 width, height,
                 multi_sample_rate=1
                 ):
        Render.__init__(self, width, height, multi_sample_rate, num_render_target=8)

    def _init_shader(self):
        self.shader = Shader(vs_file='prt.vs', fs_file='prt.fs', gs_file=None)

        # Declare all vertex attributes used in the program
        # layout (location = 0) in vec3 a_Position;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 1) in vec3 a_Normal;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 2) in vec2 a_TextureCoord;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 5) in vec3 a_PRT1;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 6) in vec3 a_PRT2;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 7) in vec3 a_PRT3;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))

        # Declare all uniforms used in the program
        self.shader.declare_uniform('ModelMat', type_code='f', gl_type=gl.glUniformMatrix4fv)
        self.shader.declare_uniform('PerspMat', type_code='f', gl_type=gl.glUniformMatrix4fv)

        self.shader.declare_uniform('SHCoeffs', type_code='f', gl_type=gl.glUniform3fv)

        self.shader.declare_uniform('UVMode', type_code='i', gl_type=gl.glUniform1i)

        # Declare all textures used in the program
        self.texture_dict["AlbedoMap"] = Texture()
