#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
"Routines which operate on quaternions."

import math
import vec


eps = 1e-8
x = 0
y = 1
z = 2
w = 3


def quat(x, y, z, w):
    return (x, y, z, w)


def normalize(q):
    "Returns a normalized quaternion (unit length)"
    mag2 = q[x]*q[x] + q[y]*q[y] + q[z]*q[z] + q[w]*q[w]
    if abs(mag2) > eps and abs(mag2 - 1.0) > eps:
        mag = math.sqrt(mag2)
        return quat(q[x] / mag, q[y] / mag, q[z] / mag, q[w] / mag)
    else:
        return q


def conjugate(q):
    return quat(-q[x], -q[y], -q[z], q[w])


def mulByQuat(q, r):
    "Multiplying q1 with q2 applies the rotation q2 to q1"
    return quat(r[w]*q[x] + r[x]*q[w] - r[y]*q[z] + r[z]*q[y],
                r[w]*q[y] + r[x]*q[z] + r[y]*q[w] - r[z]*q[x],
                r[w]*q[z] - r[x]*q[y] + r[y]*q[x] + r[z]*q[w],
                r[w]*q[w] - r[x]*q[x] - r[y]*q[y] - r[z]*q[z])


def mulByVec(q, v):
    """Multiplying a quaternion q with a vector v applies the q-rotation
    to v"""
    q2 = mulByQuat(mulByQuat(q, quat(v[x], v[y], v[z], 0.0)), conjugate(q))
    return vec.vec(q2[x], q2[y], q2[z])


def quatFromAxisAngle(v, angle):
    "Create a quaternion from an axis vector and an angle."
    vn = vec.normalize(v)
    sinAngle = math.sin(angle / 2.0)
    return (vn[x] * sinAngle,
            vn[y] * sinAngle,
            vn[z] * sinAngle,
            math.cos(angle / 2.0))


def quatToAxisAngle(q):
    "Accepts a quaternion and returns the axis vector and angle."
    angle = 2.0 * math.acos(q[w])
    scale = math.sqrt(1-q[w]*q[w])
    if scale < eps:
        axis = vec.vec(0,1,0)
    else:
        axis = vec.vec(q[x] / scale,
                       q[y] / scale,
                       q[z] / scale)
    return axis, angle
