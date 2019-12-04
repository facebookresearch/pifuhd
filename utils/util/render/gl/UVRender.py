from .Render import *


class UVRender(Render):
    def _init_shader(self):
        self.shader = Shader(vs_file='uv.vs', fs_file='uv.fs', gs_file=None)
        # layout (location = 0) in vec2 UV;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 1) in vec3 Color;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
