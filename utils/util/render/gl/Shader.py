from .GLObject import *
import numpy as np


class Uniform(object):
    def __init__(self, program, uniform_name, type_code, gl_type):
        self._as_parameter_ = gl.glGetUniformLocation(program, uniform_name)
        self.name = uniform_name
        self.type_code = type_code
        self.gl_type = gl_type


class Shader(GLObject):
    def __init__(self,
                 vs_file='simple.vs',
                 fs_file='simple.fs',
                 gs_file=None):
        # Importing here, when gl context is already present.
        # Otherwise get exception on Python3 because of PyOpenGL bug.
        from OpenGL.GL import shaders
        with open(self._find_shader_file(vs_file), 'r') as f:
            vp_code = f.read()
        with open(self._find_shader_file(fs_file), 'r') as f:
            fp_code = f.read()
        if gs_file:
            with open(self._find_shader_file(gs_file), 'r') as f:
                gp_code = f.read()
            self._as_parameter_ = self._shader = shaders.compileProgram(
                shaders.compileShader(vp_code, gl.GL_VERTEX_SHADER),
                shaders.compileShader(fp_code, gl.GL_FRAGMENT_SHADER),
                shaders.compileShader(gp_code, gl.GL_GEOMETRY_SHADER)
            )
        else:
            self._as_parameter_ = self._shader = shaders.compileProgram(
                shaders.compileShader(vp_code, gl.GL_VERTEX_SHADER),
                shaders.compileShader(fp_code, gl.GL_FRAGMENT_SHADER)
            )
        self._uniforms = {}

    def declare_uniform(self, uniform_name, type_code, gl_type):
        if uniform_name not in self._uniforms:
            self._uniforms[uniform_name] = Uniform(self._shader, uniform_name, type_code, gl_type)
        else:
            self._uniforms[uniform_name].type_code = type_code
            self._uniforms[uniform_name].gl_type = gl_type

    def set_uniform(self, uniform_name, data):
        if uniform_name not in self._uniforms:
            print(
                "Error. Unknown uniform variable. "
                "You need to declare all uniform variables in YourRender::_init_shader() function.")
            return
        uniform = self._uniforms[uniform_name]
        data = np.ascontiguousarray(data, uniform.type_code)

        if uniform.gl_type is gl.glUniformMatrix4fv:
            gl.glUniformMatrix4fv(uniform, 1, gl.GL_TRUE, data)
        elif uniform.gl_type is gl.glUniform3fv:
            gl.glUniform3fv(uniform, data.shape[0], data)
        elif uniform.gl_type is gl.glUniform1i:
            gl.glUniform1i(uniform, data)
        else:
            print(
                "Error. Unknown uniform type. "
                "You need to declare all uniform types in Shader::set_uniform() function.")
            return

    @staticmethod
    def _find_shader_file(name):
        import os
        gl_folder = os.path.dirname(os.path.abspath(__file__))
        glsl_file = os.path.join(gl_folder, 'data', name)
        return glsl_file

    def release(self):
        pass
        # gl.glDeleteProgram(self._as_parameter_)

    def __enter__(self):
        return self._shader.__enter__()

    def __exit__(self, *args):
        return self._shader.__exit__(*args)
