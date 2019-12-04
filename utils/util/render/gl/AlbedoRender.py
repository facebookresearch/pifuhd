from .Render import *


class AlbedoRender(Render):
    def _init_shader(self):
        self.shader = Shader(vs_file='albedo.vs', fs_file='albedo.fs', gs_file=None)
        # layout (location = 0) in vec3 a_Position;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 1) in vec2 a_TextureCoord;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))

        # Declare all uniform used in the program
        self.shader.declare_uniform('ModelMat', type_code='f', gl_type=gl.glUniformMatrix4fv)
        self.shader.declare_uniform('PerspMat', type_code='f', gl_type=gl.glUniformMatrix4fv)

        # Declare all textures used in  the program
        self.texture_dict["TargetTexture"] = Texture()
