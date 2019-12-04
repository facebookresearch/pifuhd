import numpy as np


def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


def radians(v):
    return np.radians(v)


def identity():
    return np.identity(4, dtype=np.float32)


def empty():
    return np.zeros([4, 4], dtype=np.float32)


def magnitude(v):
    return np.linalg.norm(v)


def normalize(v):
    m = magnitude(v)
    return v if m == 0 else v / m


def dot(u, v):
    return np.sum(u * v)


def cross(u, v):
    res = vec3(0, 0, 0)
    res[0] = u[1] * v[2] - u[2] * v[1]
    res[1] = u[2] * v[0] - u[0] * v[2]
    res[2] = u[0] * v[1] - u[1] * v[0]
    return res


# below functions can be optimized

def translate(m, v):
    res = np.copy(m)
    res[:, 3] = m[:, 0] * v[0] + m[:, 1] * v[1] + m[:, 2] * v[2] + m[:, 3]
    return res


def rotate(m, angle, v):
    a = angle
    c = np.cos(a)
    s = np.sin(a)

    axis = normalize(v)
    temp = (1 - c) * axis

    rot = empty()
    rot[0][0] = c + temp[0] * axis[0]
    rot[0][1] = temp[0] * axis[1] + s * axis[2]
    rot[0][2] = temp[0] * axis[2] - s * axis[1]

    rot[1][0] = temp[1] * axis[0] - s * axis[2]
    rot[1][1] = c + temp[1] * axis[1]
    rot[1][2] = temp[1] * axis[2] + s * axis[0]

    rot[2][0] = temp[2] * axis[0] + s * axis[1]
    rot[2][1] = temp[2] * axis[1] - s * axis[0]
    rot[2][2] = c + temp[2] * axis[2]

    res = empty()
    res[:, 0] = m[:, 0] * rot[0][0] + m[:, 1] * rot[0][1] + m[:, 2] * rot[0][2]
    res[:, 1] = m[:, 0] * rot[1][0] + m[:, 1] * rot[1][1] + m[:, 2] * rot[1][2]
    res[:, 2] = m[:, 0] * rot[2][0] + m[:, 1] * rot[2][1] + m[:, 2] * rot[2][2]
    res[:, 3] = m[:, 3]
    return res


def perspective(fovy, aspect, zNear, zFar):
    tanHalfFovy = np.tan(fovy / 2)

    res = empty()
    res[0][0] = 1 / (aspect * tanHalfFovy)
    res[1][1] = 1 / (tanHalfFovy)
    res[2][3] = -1
    res[2][2] = - (zFar + zNear) / (zFar - zNear)
    res[3][2] = -(2 * zFar * zNear) / (zFar - zNear)

    return res.T


def ortho(left, right, bottom, top, zNear, zFar):
    # res = np.ones([4, 4], dtype=np.float32)
    res = identity()
    res[0][0] = 2 / (right - left)
    res[1][1] = 2 / (top - bottom)
    res[2][2] = - 2 / (zFar - zNear)
    res[3][0] = - (right + left) / (right - left)
    res[3][1] = - (top + bottom) / (top - bottom)
    res[3][2] = - (zFar + zNear) / (zFar - zNear)
    return res.T


def lookat(eye, center, up):
    f = normalize(center - eye)
    s = normalize(cross(f, up))
    u = cross(s, f)

    res = identity()
    res[0][0] = s[0]
    res[1][0] = s[1]
    res[2][0] = s[2]
    res[0][1] = u[0]
    res[1][1] = u[1]
    res[2][1] = u[2]
    res[0][2] = -f[0]
    res[1][2] = -f[1]
    res[2][2] = -f[2]
    res[3][0] = -dot(s, eye)
    res[3][1] = -dot(u, eye)
    res[3][2] = -dot(f, eye)
    return res.T


def transform(d, m):
    return np.dot(m, d.T).T
