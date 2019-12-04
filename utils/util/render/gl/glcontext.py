"""OpenGL context creation.

Typical usage:

    # Optional PyOpenGL configuration can be done here.
    # import OpenGL
    # OpenGL.ERROR_CHECKING = True

    # 'glcontext' must be imported before any OpenGL.* API.
    from lib.render.gl.glcontext import create_opengl_context

    # Now it's safe to import OpenGL and EGL functions
    import OpenGL.GL as gl

    # create_opengl_context() creates a GL context that is attached to an
    # onscreen window of the specified size. Note that rendering to buffers
    # of other sizes and formats is still possible with OpenGL Framebuffers.
    #
    # Users are expected to directly use the GL API in case more advanced
    # context management is required.
    width, height = 640, 480
    create_opengl_context((width, height))

    # OpenGL context is available here.

"""
from OpenGL.GL import *
from OpenGL.GLUT import *


def create_opengl_context(width, height, name="My Render"):
    '''
    Create on screen OpenGL context and make it current.

      Users are expected to directly use GL API in case more advanced
      context management is required.

    :param width: window width in pixels
    :param height: window height in pixels
    :return:
    '''

    glutInit()
    display_mode = GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH

    glutInitDisplayMode(display_mode)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)

    glut_window = glutCreateWindow(name)

    glEnable(GL_DEPTH_TEST)

    glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE)
    glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE)
    glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE)

    return glut_window
