import OpenGL.GL as gl


class GLObject(object):
    def __del__(self):
        self.release()

    def __enter__(self):
        bind_func, const = self._bind
        bind_func(const, self)

    def __exit__(self, *args):
        bind_func, const = self._bind
        bind_func(const, 0)


class FBO(GLObject):
    _bind = gl.glBindFramebuffer, gl.GL_FRAMEBUFFER

    def __init__(self):
        self._as_parameter_ = gl.glGenFramebuffers(1)

    def release(self):
        try:
            if self._as_parameter_ > 0:
                gl.glDeleteFramebuffers(1, [self._as_parameter_])
        except Exception:
            pass


class Texture(GLObject):
    _bind = gl.glBindTexture, gl.GL_TEXTURE_2D

    def __init__(self):
        self._as_parameter_ = gl.glGenTextures(1)

    def release(self):
        try:
            if self._as_parameter_ > 0:
                gl.glDeleteTextures([self._as_parameter_])
        except Exception:
            pass


class MultiSampleTexture(GLObject):
    _bind = gl.glBindTexture, gl.GL_TEXTURE_2D_MULTISAMPLE

    def __init__(self):
        self._as_parameter_ = gl.glGenTextures(1)

    def release(self):
        try:
            if self._as_parameter_ > 0:
                gl.glDeleteTextures([self._as_parameter_])
        except Exception:
            pass


class RBO(GLObject):
    _bind = gl.glBindRenderbuffer, gl.GL_RENDERBUFFER

    def __init__(self):
        self._as_parameter_ = gl.glGenRenderbuffers(1)

    def release(self):
        try:
            if self._as_parameter_ > 0:
                gl.glDeleteRenderbuffers(1, [self._as_parameter_])
        except Exception:
            pass


class VBO(GLObject):
    _bind = gl.glBindBuffer, gl.GL_ARRAY_BUFFER

    def __init__(self, type_code, gl_type, dim=3, size=0):
        self._as_parameter_ = gl.glGenBuffers(1)
        self.type_code = type_code
        self.gl_type = gl_type
        self.dim = dim
        self.size = size

    def release(self):
        try:
            if self._as_parameter_ > 0:
                gl.glDeleteBuffers(1, [self._as_parameter_])
        except Exception:
            pass
